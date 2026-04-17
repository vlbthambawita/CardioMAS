from __future__ import annotations

from typing import Any

import requests

from cardiomas.agentic.runtime import AgenticRuntime
from cardiomas.inference.base import ChatRequest, ChatResponse, HealthStatus, ModelInfo
from cardiomas.inference.ollama import OllamaChatClient, OllamaEmbeddingClient
from cardiomas.knowledge.corpus import build_corpus, load_corpus
from cardiomas.schemas.config import EmbeddingConfig, KnowledgeSource, LLMConfig, RuntimeConfig


class _DummyResponse:
    def __init__(self, status_code: int, payload: dict[str, Any]) -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = str(payload)

    def json(self) -> dict[str, Any]:
        return self._payload


class _StubEmbeddingClient:
    def health_check(self) -> HealthStatus:
        return HealthStatus(provider="ollama", base_url="http://localhost:11434", ok=True)

    def list_models(self) -> list[ModelInfo]:
        return [ModelInfo(name="embeddinggemma", model="embeddinggemma")]

    def embed(self, model: str, texts: list[str]) -> list[list[float]]:
        return [[float(index + 1), float(len(text))] for index, text in enumerate(texts)]


class _StubChatClient:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)

    def health_check(self) -> HealthStatus:
        return HealthStatus(
            provider="ollama",
            base_url="http://localhost:11434",
            ok=True,
            models=[ModelInfo(name="llama3.2", model="llama3.2")],
        )

    def list_models(self) -> list[ModelInfo]:
        return [ModelInfo(name="llama3.2", model="llama3.2")]

    def chat(self, request: ChatRequest) -> ChatResponse:
        content = self._responses.pop(0)
        return ChatResponse(model=request.model, content=content)


def test_ollama_clients_use_expected_endpoints(monkeypatch):
    calls: list[tuple[str, str, dict[str, Any] | None]] = []

    def fake_request(self, method: str, url: str, json=None, timeout=None):  # noqa: ANN001
        calls.append((method, url, json))
        if url.endswith("/api/tags"):
            return _DummyResponse(200, {"models": [{"name": "llama3.2", "model": "llama3.2", "details": {}}]})
        if url.endswith("/api/chat"):
            return _DummyResponse(200, {"model": "llama3.2", "message": {"content": '{"answer":"ok"}'}})
        if url.endswith("/api/embed"):
            return _DummyResponse(200, {"model": "embeddinggemma", "embeddings": [[0.1, 0.2, 0.3]]})
        raise requests.RequestException(f"Unexpected URL: {url}")

    monkeypatch.setattr(requests.Session, "request", fake_request)

    chat_client = OllamaChatClient(LLMConfig(model="llama3.2"))
    assert chat_client.health_check().ok is True
    response = chat_client.chat(ChatRequest(model="llama3.2", messages=[]))
    assert response.content == '{"answer":"ok"}'

    embedding_client = OllamaEmbeddingClient(EmbeddingConfig(model="embeddinggemma"))
    vectors = embedding_client.embed("embeddinggemma", ["hello"])
    assert vectors == [[0.1, 0.2, 0.3]]
    assert any(url.endswith("/api/chat") for _, url, _ in calls)
    assert any(url.endswith("/api/embed") for _, url, _ in calls)
    assert any(url.endswith("/api/tags") for _, url, _ in calls)


def test_build_corpus_persists_embeddings(tmp_path):
    notes_path = tmp_path / "notes.md"
    notes_path.write_text("The dataset contains NORM and AFIB labels.", encoding="utf-8")
    config = RuntimeConfig(
        output_dir=str(tmp_path / "output"),
        sources=[KnowledgeSource(kind="local_file", path=str(notes_path), label="notes")],
        embeddings=EmbeddingConfig(model="embeddinggemma"),
    )

    manifest = build_corpus(config, embedding_client=_StubEmbeddingClient())
    chunks = load_corpus(config)

    assert manifest.embedded_chunk_count == len(chunks)
    assert all(chunk.embedding for chunk in chunks)
    assert all(chunk.embedding_model == "embeddinggemma" for chunk in chunks)


def test_runtime_query_can_use_ollama_planner_and_responder(monkeypatch, tmp_path):
    notes_path = tmp_path / "notes.md"
    notes_path.write_text("Known labels in this dataset are NORM and AFIB.", encoding="utf-8")
    chat_client = _StubChatClient(
        responses=[
            '{"strategy":"single_tool","steps":[{"tool_name":"retrieve_corpus","reason":"Ground the answer","args":{}}],"notes":"Use retrieval first"}',
            '{"answer":"The dataset includes NORM and AFIB labels.","citations":[1],"warnings":[]}',
        ]
    )
    monkeypatch.setattr("cardiomas.agentic.runtime.build_chat_client", lambda config: chat_client)
    monkeypatch.setattr("cardiomas.agentic.runtime.build_embedding_client", lambda config: None)

    config = RuntimeConfig(
        output_dir=str(tmp_path / "output"),
        sources=[KnowledgeSource(kind="local_file", path=str(notes_path), label="notes")],
        llm=LLMConfig(model="llama3.2", planner_mode="ollama"),
    )

    result = AgenticRuntime(config).query("What labels are present?", force_rebuild=True)

    assert "NORM" in result.answer
    assert len(result.llm_traces) == 2
    assert all(trace.ok for trace in result.llm_traces)
    assert result.citations
    assert result.decision.notes == ["Use retrieval first"]


def test_runtime_query_falls_back_when_ollama_outputs_invalid_json(monkeypatch, tmp_path):
    notes_path = tmp_path / "notes.md"
    notes_path.write_text("Known labels in this dataset are NORM and AFIB.", encoding="utf-8")
    chat_client = _StubChatClient(responses=["not-json", "still-not-json"])
    monkeypatch.setattr("cardiomas.agentic.runtime.build_chat_client", lambda config: chat_client)
    monkeypatch.setattr("cardiomas.agentic.runtime.build_embedding_client", lambda config: None)

    config = RuntimeConfig(
        output_dir=str(tmp_path / "output"),
        sources=[KnowledgeSource(kind="local_file", path=str(notes_path), label="notes")],
        llm=LLMConfig(model="llama3.2", planner_mode="ollama"),
    )

    result = AgenticRuntime(config).query("What labels are present?", force_rebuild=True)

    assert "NORM" in result.answer or "AFIB" in result.answer
    assert any("Ollama planner failed" in warning for warning in result.warnings)
    assert any("Ollama responder failed" in warning for warning in result.warnings)
