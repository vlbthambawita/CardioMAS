from __future__ import annotations

from collections.abc import Generator
from typing import Any

from cardiomas.agentic.aggregator import aggregate_results
from cardiomas.agentic.executor import execute_plan_events
from cardiomas.agentic.planner import plan_query_events
from cardiomas.agentic.responder import compose_answer_events
from cardiomas.autonomy.recovery import AutonomousToolManager
from cardiomas.inference.ollama import build_chat_client, build_embedding_client
from cardiomas.knowledge.corpus import build_corpus, load_corpus
from cardiomas.memory.session import SessionStore
from cardiomas.schemas.config import RuntimeConfig
from cardiomas.schemas.runtime import AgentEvent, CorpusManifest, QueryResult
from cardiomas.tools.registry import build_registry


class AgenticRuntime:
    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config
        self.sessions = SessionStore()
        self._chat_client = build_chat_client(config.llm)
        self._embedding_client = build_embedding_client(config.embeddings)
        self._autonomy_manager = AutonomousToolManager(config)

    def build_corpus(self, force_rebuild: bool = False) -> CorpusManifest:
        if force_rebuild or not self.config.corpus_path.exists():
            return build_corpus(self.config, embedding_client=self._embedding_client)
        import json

        return CorpusManifest.model_validate_json(self.config.manifest_path.read_text(encoding="utf-8"))

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

    def query(self, query: str, force_rebuild: bool = False) -> QueryResult:
        result: QueryResult | None = None
        for event in self.query_stream(query, force_rebuild=force_rebuild):
            if event.type == "final_result":
                result = QueryResult.model_validate(event.data["result"])
        if result is None:
            raise RuntimeError("query_stream completed without producing a final result.")
        return result

    def query_stream(self, query: str, force_rebuild: bool = False) -> Generator[AgentEvent, None, QueryResult]:
        yield AgentEvent(type="status", stage="runtime", message="Query started.")
        manifest = self.build_corpus(force_rebuild=force_rebuild)
        yield AgentEvent(type="status", stage="corpus", message="Corpus ready.", data=manifest.model_dump(mode="json"))
        chunks = load_corpus(self.config)
        session = self.sessions.start()
        self.sessions.append_query(session.session_id, query)
        self._autonomy_manager.reset_traces()
        self._autonomy_manager.set_session(session.session_id)
        registry = build_registry(
            self.config,
            chunks,
            embedding_client=self._embedding_client,
            autonomy_manager=self._autonomy_manager,
        )

        decision, planner_traces, planner_warnings = yield from plan_query_events(
            query,
            self.config,
            registry,
            chat_client=self._chat_client,
        )
        results, calls, execution_warnings = yield from execute_plan_events(
            decision,
            self.config,
            registry,
            self.sessions,
            session.session_id,
            autonomy_manager=self._autonomy_manager,
        )
        evidence, aggregate = aggregate_results(results)
        answer, citations, responder_traces, responder_warnings = yield from compose_answer_events(
            query,
            self.config,
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
        yield AgentEvent(type="final_result", stage="runtime", message="Query finished.", data={"result": result.model_dump(mode="json")})
        return result
