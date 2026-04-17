from __future__ import annotations

from cardiomas.agentic.aggregator import aggregate_results
from cardiomas.agentic.executor import execute_plan
from cardiomas.agentic.planner import plan_query
from cardiomas.agentic.responder import compose_answer
from cardiomas.knowledge.corpus import build_corpus, load_corpus
from cardiomas.memory.session import SessionStore
from cardiomas.schemas.config import RuntimeConfig
from cardiomas.schemas.runtime import CorpusManifest, QueryResult
from cardiomas.tools.registry import build_registry


class AgenticRuntime:
    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config
        self.sessions = SessionStore()

    def build_corpus(self, force_rebuild: bool = False) -> CorpusManifest:
        if force_rebuild or not self.config.corpus_path.exists():
            return build_corpus(self.config)
        import json

        return CorpusManifest.model_validate_json(self.config.manifest_path.read_text(encoding="utf-8"))

    def inspect_tools(self):
        chunks = load_corpus(self.config)
        registry = build_registry(self.config, chunks)
        return registry.specs()

    def query(self, query: str, force_rebuild: bool = False) -> QueryResult:
        self.build_corpus(force_rebuild=force_rebuild)
        chunks = load_corpus(self.config)
        registry = build_registry(self.config, chunks)
        session = self.sessions.start()
        self.sessions.append_query(session.session_id, query)

        decision = plan_query(query, self.config, registry)
        results, calls, warnings = execute_plan(decision, self.config, registry, self.sessions, session.session_id)
        evidence, aggregate = aggregate_results(results)
        answer, citations = compose_answer(query, self.config, evidence, aggregate, warnings)

        return QueryResult(
            session_id=session.session_id,
            query=query,
            answer=answer,
            decision=decision,
            citations=citations,
            evidence=evidence,
            tool_calls=calls,
            warnings=warnings,
        )
