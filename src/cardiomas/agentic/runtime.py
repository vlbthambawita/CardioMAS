from __future__ import annotations

from collections.abc import Generator
from typing import Any

from cardiomas.agentic.aggregator import aggregate_results
from cardiomas.agentic.executor import execute_plan_events
from cardiomas.agentic.planner import plan_query_events
from cardiomas.agentic.react_agent import make_react_decision, run_react_events
from cardiomas.agentic.responder import compose_answer_events
from cardiomas.autonomy.recovery import AutonomousToolManager
from cardiomas.inference.ollama import build_chat_client, build_embedding_client
from cardiomas.knowledge.corpus import build_agent_corpus, build_corpus, load_agent_corpus, load_corpus
from cardiomas.memory.persistent import PersistentMemory, build_persistent_memory
from cardiomas.memory.session import SessionStore
from cardiomas.schemas.config import RuntimeConfig
from cardiomas.schemas.evidence import EvidenceChunk
from cardiomas.schemas.runtime import AgentEvent, CorpusManifest, QueryResult
from cardiomas.tools.registry import build_registry


class AgenticRuntime:
    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config
        self.sessions = SessionStore()
        self._chat_client = build_chat_client(config.llm)
        self._embedding_client = build_embedding_client(config.embeddings)
        self._autonomy_manager = AutonomousToolManager(config, chat_client=self._chat_client)
        self._persistent_memory: PersistentMemory | None = None
        if config.agent.memory_mode == "persistent":
            self._persistent_memory = build_persistent_memory(
                config.output_dir,
                max_entries=config.agent.persistent_memory_max,
            )
        if config.llm and config.llm.warmup and self._chat_client is not None:
            self._chat_client.warmup()
        # Per-agent corpus cache: agent_name → chunks (populated by build_corpus)
        self._agent_chunks: dict[str, list[EvidenceChunk]] = {}

    def build_corpus(self, force_rebuild: bool = False, agent_name: str | None = None) -> CorpusManifest:
        import json

        # 1. Global corpus
        if force_rebuild or not self.config.corpus_path.exists():
            manifest = build_corpus(self.config, embedding_client=self._embedding_client)
        else:
            manifest = CorpusManifest.model_validate_json(
                self.config.manifest_path.read_text(encoding="utf-8")
            )

        # 2. Per-agent corpora (only when knowledge_scraping is enabled)
        if self.config.knowledge_scraping.enabled:
            targets = (
                [a for a in self.config.named_agents if a.name == agent_name]
                if agent_name
                else self.config.named_agents
            )
            for agent_cfg in targets:
                if not agent_cfg.knowledge.enabled:
                    continue
                build_agent_corpus(
                    agent_cfg, self.config,
                    embedding_client=self._embedding_client,
                    force=force_rebuild,
                )
                self._agent_chunks[agent_cfg.name] = load_agent_corpus(agent_cfg.name, self.config)

        return manifest

    def inspect_tools(self):
        chunks = load_corpus(self.config)
        registry = build_registry(
            self.config,
            chunks,
            embedding_client=self._embedding_client,
            autonomy_manager=self._autonomy_manager,
        )
        return registry.specs()

    def check_ollama(self) -> dict[str, Any]:
        status: dict[str, Any] = {}
        if self.config.llm is not None and self._chat_client is not None:
            status["llm"] = self._chat_client.health_check().model_dump(mode="json")
        else:
            status["llm"] = {"configured": False}
        if self.config.embeddings is not None and self._embedding_client is not None:
            status["embeddings"] = self._embedding_client.health_check().model_dump(mode="json")
        else:
            status["embeddings"] = {"configured": False}
        return status

    def query(self, query: str, force_rebuild: bool = False, agent_name: str | None = None) -> QueryResult:
        result: QueryResult | None = None
        for event in self.query_stream(query, force_rebuild=force_rebuild, agent_name=agent_name):
            if event.type == "final_result":
                result = QueryResult.model_validate(event.data["result"])
        if result is None:
            raise RuntimeError("query_stream completed without producing a final result.")
        return result

    def query_stream(self, query: str, force_rebuild: bool = False, agent_name: str | None = None) -> Generator[AgentEvent, None, QueryResult]:
        yield AgentEvent(type="status", stage="runtime", message="Query started.")
        manifest = self.build_corpus(force_rebuild=force_rebuild)
        yield AgentEvent(type="status", stage="corpus", message="Corpus ready.", data=manifest.model_dump(mode="json"))
        chunks = load_corpus(self.config)

        # Resolve named-agent overrides and per-agent chunks
        named = self.config.active_named_agent(agent_name)
        if named and agent_name and agent_name not in self._agent_chunks and named.knowledge.enabled:
            self._agent_chunks[agent_name] = load_agent_corpus(agent_name, self.config)
        agent_chunks = self._agent_chunks.get(agent_name) if agent_name else None
        agent_retrieval = named.knowledge.retrieval if named else None

        # Apply per-agent config overrides (agent settings, tools; LLM uses global client)
        effective_config = self.config
        if named:
            overrides = {}
            if named.agent:
                overrides["agent"] = named.agent
            if named.tools:
                overrides["tools"] = named.tools
            if overrides:
                effective_config = self.config.model_copy(update=overrides)

        session = self.sessions.start()
        self.sessions.append_query(session.session_id, query)
        self._autonomy_manager.reset_traces()
        self._autonomy_manager.set_session(session.session_id)
        registry = build_registry(
            effective_config,
            chunks,
            embedding_client=self._embedding_client,
            autonomy_manager=self._autonomy_manager,
            agent_chunks=agent_chunks,
            agent_retrieval=agent_retrieval,
        )

        # ── ReAct mode ────────────────────────────────────────────────────
        if effective_config.agent.mode == "react" and self._chat_client is not None:
            (
                answer, citations, evidence, aggregate,
                tool_calls, llm_traces, warnings, react_steps,
            ) = yield from run_react_events(
                query=query,
                config=effective_config,
                registry=registry,
                chat_client=self._chat_client,
                session_store=self.sessions,
                session_id=session.session_id,
                autonomy_manager=self._autonomy_manager,
                persistent_memory=self._persistent_memory,
            )
            decision = make_react_decision(react_steps)
            all_warnings = warnings + manifest.warnings
            result = QueryResult(
                session_id=session.session_id,
                query=query,
                answer=answer,
                decision=decision,
                citations=citations,
                evidence=evidence,
                tool_calls=tool_calls,
                warnings=all_warnings,
                llm_traces=llm_traces,
                repair_traces=self._autonomy_manager.consume_traces(),
                standalone_scripts=aggregate.get("standalone_scripts", []),
                react_steps=react_steps,
            )
            yield AgentEvent(
                type="final_result", stage="runtime",
                message="Query finished (react).",
                data={"result": result.model_dump(mode="json")},
            )
            return result

        # ── Linear mode (legacy, default) ─────────────────────────────────
        decision, planner_traces, planner_warnings = yield from plan_query_events(
            query,
            effective_config,
            registry,
            chat_client=self._chat_client,
        )
        results, calls, execution_warnings = yield from execute_plan_events(
            decision,
            effective_config,
            registry,
            self.sessions,
            session.session_id,
            autonomy_manager=self._autonomy_manager,
        )
        evidence, aggregate = aggregate_results(results)
        answer, citations, responder_traces, responder_warnings = yield from compose_answer_events(
            query,
            effective_config,
            evidence,
            aggregate,
            planner_warnings + execution_warnings + manifest.warnings,
            chat_client=self._chat_client,
        )
        warnings = planner_warnings + execution_warnings + responder_warnings + manifest.warnings

        result = QueryResult(
            session_id=session.session_id,
            query=query,
            answer=answer,
            decision=decision,
            citations=citations,
            evidence=evidence,
            tool_calls=calls,
            warnings=warnings,
            llm_traces=planner_traces + responder_traces,
            repair_traces=self._autonomy_manager.consume_traces(),
            standalone_scripts=aggregate.get("standalone_scripts", []),
        )
        yield AgentEvent(
            type="final_result", stage="runtime",
            message="Query finished.",
            data={"result": result.model_dump(mode="json")},
        )
        return result
