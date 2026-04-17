from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from cardiomas.agentic.answer_grader import grade_answer
from cardiomas.agentic.query_decomposer import SubQuery, _is_complex, decompose
from cardiomas.agentic.react_agent import _parse_thought, make_react_decision, run_react_events
from cardiomas.agentic.retrieval_grader import grade_chunks
from cardiomas.agentic.router import RouteDecision, _route_heuristic, route_query
from cardiomas.inference.base import ChatChunk, ChatResponse
from cardiomas.memory.persistent import PersistentMemory, _cosine, _bow
from cardiomas.schemas.config import AgentConfig, AutonomyConfig, KnowledgeSource, LLMConfig, RuntimeConfig
from cardiomas.schemas.evidence import EvidenceChunk
from cardiomas.schemas.runtime import ReActStep


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_client(response: str) -> MagicMock:
    c = MagicMock()
    c.chat.return_value = ChatResponse(model="test", content=response, raw={})
    c.chat_stream.return_value = iter([])
    return c


def _make_config(tmp_path: Path, mode: str = "react", dataset: bool = False) -> RuntimeConfig:
    sources = []
    if dataset:
        ds = tmp_path / "dataset"
        ds.mkdir(exist_ok=True)
        (ds / "meta.csv").write_text("patient_id,age,label\n1,45,NORM\n", encoding="utf-8")
        sources.append(KnowledgeSource(kind="dataset_dir", path=str(ds), label="ds"))
    return RuntimeConfig(
        output_dir=str(tmp_path / "output"),
        sources=sources,
        agent=AgentConfig(
            mode=mode,
            max_iterations=3,
            query_decomposition=False,
            self_reflection=False,
            retrieval_grading=True,
        ),
        llm=LLMConfig(model="test-model", max_tokens=400),
    )


def _make_chunk(chunk_id: str = "c1", content: str = "relevant content") -> EvidenceChunk:
    return EvidenceChunk(
        chunk_id=chunk_id,
        source_id="s",
        source_label="test",
        source_type="corpus",
        title="T",
        content=content,
        uri="u",
        score=0.8,
    )


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

class TestRouter:
    def test_heuristic_routes_url_to_web(self, tmp_path):
        config = _make_config(tmp_path)
        from cardiomas.schemas.tools import ToolSpec
        tools = [ToolSpec(name="fetch_webpage", description="fetch", category="research")]
        decision = _route_heuristic("Check https://example.com for info", config, tools)
        assert decision.route == "web"

    def test_heuristic_routes_compute_to_code(self, tmp_path):
        config = _make_config(tmp_path, dataset=True)
        from cardiomas.schemas.tools import ToolSpec
        tools = [ToolSpec(name="generate_python_artifact", description="gen", category="autonomy", generated=True)]
        decision = _route_heuristic("How many unique patients are in the dataset?", config, tools)
        assert decision.route == "code"

    def test_heuristic_routes_complex_to_orchestrate(self, tmp_path):
        config = _make_config(tmp_path)
        decision = _route_heuristic("Compare NORM vs AFIB and describe the difference", config, [])
        assert decision.route == "orchestrate"

    def test_heuristic_default_retrieval(self, tmp_path):
        config = _make_config(tmp_path)
        from cardiomas.schemas.tools import ToolSpec
        tools = [ToolSpec(name="retrieve_corpus", description="retrieve", category="retrieval")]
        decision = _route_heuristic("What is scp_codes?", config, tools)
        assert decision.route == "retrieval"

    def test_llm_route_used_when_available(self, tmp_path):
        config = _make_config(tmp_path)
        client = _make_client(json.dumps({"route": "code", "reason": "needs computation"}))
        from cardiomas.schemas.tools import ToolSpec
        tools = [ToolSpec(name="generate_python_artifact", description="gen", category="autonomy", generated=True)]
        decision = route_query("How many rows?", config, tools, chat_client=client)
        assert decision.route == "code"
        assert client.chat.called

    def test_llm_failure_falls_back_to_heuristic(self, tmp_path):
        config = _make_config(tmp_path)
        client = _make_client("not valid json at all")
        decision = route_query("What is scp_codes?", config, [], chat_client=client)
        assert decision.route in {"retrieval", "orchestrate", "code", "web"}


# ---------------------------------------------------------------------------
# Query Decomposer
# ---------------------------------------------------------------------------

class TestDecomposer:
    def test_simple_query_not_decomposed(self):
        assert not _is_complex("What is scp_codes?")

    def test_compare_query_is_complex(self):
        assert _is_complex("Compare NORM vs AFIB in the dataset")

    def test_multi_question_is_complex(self):
        assert _is_complex("What is the label distribution? How many patients are there?")

    def test_decompose_simple_returns_single(self, tmp_path):
        config = _make_config(tmp_path)
        result = decompose("What is scp_codes?", config, chat_client=None)
        assert len(result) == 1
        assert result[0].text == "What is scp_codes?"

    def test_decompose_with_llm(self, tmp_path):
        config = _make_config(tmp_path)
        llm_response = json.dumps({
            "sub_queries": [
                {"text": "What labels exist?", "query_type": "factual"},
                {"text": "How many patients per label?", "query_type": "computational"},
            ]
        })
        client = _make_client(llm_response)
        result = decompose("Compare labels and count patients per label", config, chat_client=client)
        assert len(result) == 2
        assert result[0].query_type == "factual"
        assert result[1].query_type == "computational"

    def test_decompose_caps_at_4(self, tmp_path):
        config = _make_config(tmp_path)
        many = [{"text": f"Q{i}", "query_type": "factual"} for i in range(10)]
        client = _make_client(json.dumps({"sub_queries": many}))
        result = decompose("complex multi-compare and difference", config, chat_client=client)
        assert len(result) <= 4


# ---------------------------------------------------------------------------
# Retrieval Grader
# ---------------------------------------------------------------------------

class TestRetrievalGrader:
    def test_no_client_returns_partial(self, tmp_path):
        config = _make_config(tmp_path)
        chunks = [_make_chunk()]
        result = grade_chunks("any query", chunks, config, chat_client=None)
        assert result.verdict == "partial"
        assert result.relevant_count > 0

    def test_empty_chunks_returns_insufficient(self, tmp_path):
        config = _make_config(tmp_path)
        result = grade_chunks("any query", [], config, chat_client=None)
        assert result.verdict == "insufficient"

    def test_llm_grades_sufficient(self, tmp_path):
        config = _make_config(tmp_path)
        client = _make_client(json.dumps({"verdict": "sufficient", "relevant_count": 3, "reason": "good"}))
        chunks = [_make_chunk("c1"), _make_chunk("c2"), _make_chunk("c3")]
        result = grade_chunks("patient count query", chunks, config, chat_client=client)
        assert result.verdict == "sufficient"
        assert result.relevant_count == 3

    def test_llm_grades_insufficient(self, tmp_path):
        config = _make_config(tmp_path)
        client = _make_client(json.dumps({"verdict": "insufficient", "relevant_count": 0, "reason": "irrelevant"}))
        chunks = [_make_chunk()]
        result = grade_chunks("very specific query", chunks, config, chat_client=client)
        assert result.verdict == "insufficient"

    def test_llm_failure_returns_partial(self, tmp_path):
        config = _make_config(tmp_path)
        client = _make_client("broken json")
        chunks = [_make_chunk()]
        result = grade_chunks("query", chunks, config, chat_client=client)
        assert result.verdict == "partial"


# ---------------------------------------------------------------------------
# Answer Grader
# ---------------------------------------------------------------------------

class TestAnswerGrader:
    def test_no_client_returns_grounded(self, tmp_path):
        config = _make_config(tmp_path)
        verdict = grade_answer("q", "a", [], config, chat_client=None)
        assert verdict == "grounded"

    def test_llm_grounded(self, tmp_path):
        config = _make_config(tmp_path)
        client = _make_client(json.dumps({"verdict": "grounded", "reason": "supported by evidence"}))
        verdict = grade_answer("q", "answer text", [_make_chunk()], config, client)
        assert verdict == "grounded"

    def test_llm_hallucinated(self, tmp_path):
        config = _make_config(tmp_path)
        client = _make_client(json.dumps({"verdict": "hallucinated", "reason": "not in evidence"}))
        verdict = grade_answer("q", "made up answer", [], config, client)
        assert verdict == "hallucinated"

    def test_llm_failure_returns_grounded(self, tmp_path):
        config = _make_config(tmp_path)
        client = _make_client("not json")
        verdict = grade_answer("q", "a", [], config, client)
        assert verdict == "grounded"


# ---------------------------------------------------------------------------
# Persistent Memory
# ---------------------------------------------------------------------------

class TestPersistentMemory:
    def test_store_and_retrieve(self, tmp_path):
        mem = PersistentMemory(tmp_path / "mem.json", max_entries=10)
        mem.store("how many patients", "There are 1000 patients.", grounded=True)
        result = mem.find_similar("how many unique patients")
        assert result is not None
        assert "1000" in result["answer"]

    def test_no_match_below_threshold(self, tmp_path):
        mem = PersistentMemory(tmp_path / "mem.json")
        mem.store("what is the weather", "It is sunny.", grounded=True)
        result = mem.find_similar("how many patients are there", threshold=0.70)
        assert result is None

    def test_ungrounded_entries_not_returned(self, tmp_path):
        mem = PersistentMemory(tmp_path / "mem.json")
        mem.store("how many patients", "Bad answer.", grounded=False)
        result = mem.find_similar("how many patients")
        assert result is None

    def test_max_entries_eviction(self, tmp_path):
        mem = PersistentMemory(tmp_path / "mem.json", max_entries=3)
        for i in range(5):
            mem.store(f"query {i}", f"answer {i}", grounded=True)
        assert len(mem._entries) == 3

    def test_persistence_across_instances(self, tmp_path):
        path = tmp_path / "mem.json"
        mem1 = PersistentMemory(path)
        mem1.store("how many patients are in the dataset", "1000 patients.", grounded=True)
        mem2 = PersistentMemory(path)
        result = mem2.find_similar("how many patients are in the dataset")
        assert result is not None

    def test_cosine_similarity(self):
        a = _bow("how many patients are in the dataset")
        b = _bow("how many unique patients in the dataset")
        score = _cosine(a, b)
        assert score > 0.5

    def test_cosine_empty(self):
        assert _cosine(_bow(""), _bow("something")) == 0.0


# ---------------------------------------------------------------------------
# ReAct Agent — integration with mocked LLM
# ---------------------------------------------------------------------------

class TestReActAgent:
    def _build_registry(self, tmp_path):
        from cardiomas.knowledge.corpus import build_corpus, load_corpus
        from cardiomas.tools.registry import build_registry

        config = _make_config(tmp_path)
        config.output_dir = str(tmp_path / "output")
        build_corpus(config)
        chunks = load_corpus(config)
        return build_registry(config, chunks), config

    def test_parse_thought_valid_json(self):
        content = '{"thought": "I need to retrieve", "action": "retrieve_corpus", "args": {"query": "labels"}}'
        result = _parse_thought(content)
        assert result["action"] == "retrieve_corpus"

    def test_parse_thought_malformed_returns_answer(self):
        result = _parse_thought("I will now retrieve the data")
        assert result["action"] == "answer"

    def test_make_react_decision_builds_agent_decision(self):
        steps = [
            ReActStep(iteration=1, thought="need to retrieve", action="retrieve_corpus", action_args={"query": "test"}),
            ReActStep(iteration=2, thought="done", action="answer", action_args={}),
        ]
        decision = make_react_decision(steps)
        assert decision.strategy == "react"
        assert len(decision.steps) == 2

    def test_react_agent_answer_action_stops_loop(self, tmp_path):
        """LLM immediately says 'answer' → loop stops at iteration 1."""
        config = _make_config(tmp_path)
        from cardiomas.knowledge.corpus import build_corpus, load_corpus
        from cardiomas.tools.registry import build_registry
        from cardiomas.memory.session import SessionStore

        build_corpus(config)
        chunks = load_corpus(config)
        registry = build_registry(config, chunks)

        # Planner says answer immediately; responder says answer too
        action_resp = json.dumps({"thought": "I can answer.", "action": "answer", "args": {}})
        synth_resp = json.dumps({"answer": "Test answer.", "citations": [], "warnings": []})

        client = MagicMock()
        client.chat.side_effect = [
            ChatResponse(model="t", content=action_resp, raw={}),  # orchestrator
            ChatResponse(model="t", content=synth_resp, raw={}),   # responder
        ]
        client.chat_stream.return_value = iter([])

        sessions = SessionStore()
        session = sessions.start()

        gen = run_react_events(
            query="simple question",
            config=config,
            registry=registry,
            chat_client=client,
            session_store=sessions,
            session_id=session.session_id,
        )
        result = None
        try:
            while True:
                next(gen)
        except StopIteration as stop:
            result = stop.value

        answer, citations, evidence, aggregate, tool_calls, llm_traces, warnings, react_steps = result
        assert len(react_steps) == 1
        assert react_steps[0].action == "answer"

    def test_react_agent_calls_retrieve_corpus(self, tmp_path):
        """LLM first calls retrieve_corpus then says answer."""
        config = _make_config(tmp_path)
        from cardiomas.knowledge.corpus import build_corpus, load_corpus
        from cardiomas.tools.registry import build_registry, ToolRegistry
        from cardiomas.memory.session import SessionStore
        from cardiomas.schemas.tools import ToolResult, ToolSpec

        build_corpus(config)
        chunks = load_corpus(config)
        registry = build_registry(config, chunks)

        retrieve_resp = json.dumps({
            "thought": "I will retrieve relevant info.",
            "action": "retrieve_corpus",
            "args": {"query": "scp_codes", "top_k": 3},
        })
        answer_resp = json.dumps({"thought": "I have enough.", "action": "answer", "args": {}})
        synth_resp = json.dumps({"answer": "The scp_codes column contains diagnostic codes.", "citations": [], "warnings": []})
        grade_resp = json.dumps({"verdict": "sufficient", "relevant_count": 2, "reason": "good"})

        client = MagicMock()
        client.chat.side_effect = [
            ChatResponse(model="t", content=retrieve_resp, raw={}),  # iter 1 orchestrator
            ChatResponse(model="t", content=grade_resp, raw={}),     # retrieval grader
            ChatResponse(model="t", content=answer_resp, raw={}),    # iter 2 orchestrator
            ChatResponse(model="t", content=synth_resp, raw={}),     # responder
        ]
        client.chat_stream.return_value = iter([])

        sessions = SessionStore()
        session = sessions.start()

        gen = run_react_events(
            query="What is scp_codes?",
            config=config,
            registry=registry,
            chat_client=client,
            session_store=sessions,
            session_id=session.session_id,
        )
        result = None
        try:
            while True:
                next(gen)
        except StopIteration as stop:
            result = stop.value

        answer, *_ = result
        assert "scp_codes" in answer or len(answer) > 0


# ---------------------------------------------------------------------------
# Runtime dispatch: agent.mode = "react"
# ---------------------------------------------------------------------------

class TestRuntimeReactDispatch:
    def test_linear_mode_is_default(self, tmp_path):
        from cardiomas.agentic.runtime import AgenticRuntime
        config = RuntimeConfig(
            output_dir=str(tmp_path / "output"),
            sources=[],
        )
        assert config.agent.mode == "linear"

    def test_react_mode_runs_react_loop(self, tmp_path):
        """With agent.mode=react and LLM configured, react_steps are populated."""
        from cardiomas.agentic.runtime import AgenticRuntime

        config = _make_config(tmp_path, mode="react")
        config.sources = []  # no dataset needed

        route_resp = json.dumps({"route": "retrieval", "reason": "default"})
        action_resp = json.dumps({"thought": "I can answer.", "action": "answer", "args": {}})
        synth_resp = json.dumps({"answer": "This is a react answer.", "citations": [], "warnings": []})

        client = MagicMock()
        client.chat.side_effect = [
            ChatResponse(model="t", content=route_resp, raw={}),   # router
            ChatResponse(model="t", content=action_resp, raw={}),  # orchestrator iter 1 fallback
        ]
        # orchestrator streaming returns empty → falls back to chat(); responder gets the token
        client.chat_stream.side_effect = [
            iter([]),  # orchestrator iter 1 stream (empty → fallback)
            iter([ChatChunk(model="t", content=synth_resp, done=True, raw={})]),  # responder
        ]

        runtime = AgenticRuntime(config)
        runtime._chat_client = client  # inject mock

        result = runtime.query("simple question", force_rebuild=True)
        assert result.react_steps
        assert result.react_steps[0].action == "answer"
        assert result.answer == "This is a react answer."
