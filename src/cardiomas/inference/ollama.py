from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import requests

from cardiomas.inference.base import (
    ChatClient,
    ChatChunk,
    ChatMessage,
    ChatRequest,
    ChatResponse,
    EmbeddingClient,
    HealthStatus,
    ModelInfo,
)
from cardiomas.schemas.config import EmbeddingConfig, LLMConfig


class OllamaError(RuntimeError):
    pass


class _OllamaTransport:
    def __init__(self, base_url: str, timeout_seconds: float) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self._session = requests.Session()

    def request_json(self, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        try:
            response = self._session.request(method=method, url=url, json=payload, timeout=self.timeout_seconds)
        except requests.RequestException as exc:
            raise OllamaError(f"Could not connect to Ollama at {self.base_url}: {exc}") from exc

        try:
            data = response.json()
        except ValueError as exc:
            raise OllamaError(f"Malformed JSON response from Ollama at {url}.") from exc

        if response.status_code >= 400:
            error_message = data.get("error") if isinstance(data, dict) else response.text
            raise OllamaError(f"Ollama returned HTTP {response.status_code}: {error_message}")

        if not isinstance(data, dict):
            raise OllamaError(f"Malformed response from Ollama at {url}: expected a JSON object.")
        return data

    def request_json_lines(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> Iterator[dict[str, Any]]:
        url = f"{self.base_url}{path}"
        try:
            response = self._session.request(
                method=method,
                url=url,
                json=payload,
                timeout=self.timeout_seconds,
                stream=True,
            )
        except requests.RequestException as exc:
            raise OllamaError(f"Could not connect to Ollama at {self.base_url}: {exc}") from exc

        if response.status_code >= 400:
            try:
                data = response.json()
            except ValueError:
                data = {}
            error_message = data.get("error") if isinstance(data, dict) else response.text
            raise OllamaError(f"Ollama returned HTTP {response.status_code}: {error_message}")

        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise OllamaError(f"Malformed streamed JSON response from Ollama at {url}.") from exc
            if not isinstance(item, dict):
                raise OllamaError(f"Malformed streamed response from Ollama at {url}: expected a JSON object.")
            yield item

    def list_models(self) -> list[ModelInfo]:
        payload = self.request_json("GET", "/api/tags")
        models = payload.get("models", [])
        if not isinstance(models, list):
            raise OllamaError("Malformed Ollama model listing: 'models' was not a list.")
        result: list[ModelInfo] = []
        for item in models:
            if not isinstance(item, dict):
                continue
            result.append(
                ModelInfo(
                    name=str(item.get("name", "")),
                    model=str(item.get("model", item.get("name", ""))),
                    details=item.get("details", {}) if isinstance(item.get("details"), dict) else {},
                )
            )
        return result

    def health_check(self) -> HealthStatus:
        try:
            models = self.list_models()
        except OllamaError as exc:
            return HealthStatus(provider="ollama", base_url=self.base_url, ok=False, error=str(exc))
        return HealthStatus(provider="ollama", base_url=self.base_url, ok=True, models=models)


class OllamaChatClient(ChatClient):
    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self._transport = _OllamaTransport(config.base_url, config.timeout_seconds)

    def health_check(self) -> HealthStatus:
        return self._transport.health_check()

    def list_models(self) -> list[ModelInfo]:
        return self._transport.list_models()

    def chat(self, request: ChatRequest) -> ChatResponse:
        options: dict[str, Any] = {
            "temperature": request.temperature,
            "repeat_penalty": self.config.repeat_penalty,
        }
        if request.max_tokens > 0:
            options["num_predict"] = request.max_tokens

        payload: dict[str, Any] = {
            "model": request.model,
            "messages": [message.model_dump(mode="json") for message in request.messages],
            "stream": False,
            "keep_alive": request.keep_alive or self.config.keep_alive,
            "options": options,
        }
        if request.json_mode:
            payload["format"] = "json"

        response = self._transport.request_json("POST", "/api/chat", payload=payload)
        message = response.get("message", {})
        if not isinstance(message, dict):
            raise OllamaError("Malformed Ollama chat response: missing message object.")
        content = message.get("content", "")
        if not isinstance(content, str):
            raise OllamaError("Malformed Ollama chat response: message content was not a string.")
        return ChatResponse(model=str(response.get("model", request.model)), content=content, raw=response)

    def chat_stream(self, request: ChatRequest) -> Iterator[ChatChunk]:
        options: dict[str, Any] = {
            "temperature": request.temperature,
            "repeat_penalty": self.config.repeat_penalty,
        }
        if request.max_tokens > 0:
            options["num_predict"] = request.max_tokens

        payload: dict[str, Any] = {
            "model": request.model,
            "messages": [message.model_dump(mode="json") for message in request.messages],
            "stream": True,
            "keep_alive": request.keep_alive or self.config.keep_alive,
            "options": options,
        }
        if request.json_mode:
            payload["format"] = "json"

        for item in self._transport.request_json_lines("POST", "/api/chat", payload=payload):
            message = item.get("message", {})
            content = ""
            if isinstance(message, dict):
                raw_content = message.get("content", "")
                if isinstance(raw_content, str):
                    content = raw_content
            yield ChatChunk(
                model=str(item.get("model", request.model)),
                content=content,
                done=bool(item.get("done", False)),
                raw=item,
            )


class OllamaEmbeddingClient(EmbeddingClient):
    def __init__(self, config: EmbeddingConfig) -> None:
        self.config = config
        self._transport = _OllamaTransport(config.base_url, config.timeout_seconds)

    def health_check(self) -> HealthStatus:
        return self._transport.health_check()

    def list_models(self) -> list[ModelInfo]:
        return self._transport.list_models()

    def embed(self, model: str, texts: list[str]) -> list[list[float]]:
        payload = {
            "model": model,
            "input": texts,
            "keep_alive": self.config.keep_alive,
            "truncate": True,
        }
        response = self._transport.request_json("POST", "/api/embed", payload=payload)
        embeddings = response.get("embeddings", [])
        if not isinstance(embeddings, list):
            raise OllamaError("Malformed Ollama embedding response: 'embeddings' was not a list.")

        normalized: list[list[float]] = []
        for item in embeddings:
            if not isinstance(item, list):
                raise OllamaError("Malformed Ollama embedding response: one embedding was not a list.")
            normalized.append([float(value) for value in item])
        return normalized


    def warmup(self) -> bool:
        """Send a single 1-token prompt to force Ollama to load the model into VRAM.

        Returns True if the model responded, False on any error.
        The response content is discarded — the only goal is triggering model load.
        """
        try:
            request = ChatRequest(
                model=self.config.model,
                messages=[ChatMessage(role="user", content="hi")],
                temperature=0.0,
                max_tokens=1,
                keep_alive=self.config.keep_alive,
            )
            self.chat(request)
            return True
        except Exception:
            return False


def build_chat_client(config: LLMConfig | None) -> OllamaChatClient | None:
    if config is None or config.provider != "ollama":
        return None
    return OllamaChatClient(config)


def build_embedding_client(config: EmbeddingConfig | None) -> OllamaEmbeddingClient | None:
    if config is None or config.provider != "ollama":
        return None
    return OllamaEmbeddingClient(config)
