from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any, Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    temperature: float = 0.1
    max_tokens: int = 800
    json_mode: bool = False
    keep_alive: str = "5m"


class ChatResponse(BaseModel):
    model: str
    content: str
    raw: dict[str, Any] = Field(default_factory=dict)


class ChatChunk(BaseModel):
    model: str
    content: str = ""
    done: bool = False
    raw: dict[str, Any] = Field(default_factory=dict)


class ModelInfo(BaseModel):
    name: str
    model: str = ""
    details: dict[str, Any] = Field(default_factory=dict)


class HealthStatus(BaseModel):
    provider: str
    base_url: str
    ok: bool
    models: list[ModelInfo] = Field(default_factory=list)
    error: str = ""


class ChatClient(ABC):
    @abstractmethod
    def health_check(self) -> HealthStatus: ...

    @abstractmethod
    def list_models(self) -> list[ModelInfo]: ...

    @abstractmethod
    def chat(self, request: ChatRequest) -> ChatResponse: ...

    def chat_stream(self, request: ChatRequest) -> Iterator[ChatChunk]:
        response = self.chat(request)
        yield ChatChunk(model=response.model, content=response.content, done=True, raw=response.raw)


class EmbeddingClient(ABC):
    @abstractmethod
    def health_check(self) -> HealthStatus: ...

    @abstractmethod
    def list_models(self) -> list[ModelInfo]: ...

    @abstractmethod
    def embed(self, model: str, texts: list[str]) -> list[list[float]]: ...
