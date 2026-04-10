import ollama
from typing import Any


class BaseAgent:
    """Base class for all agents."""

    def __init__(self, name: str, model: str = "gemma4:latest"):
        self.name = name
        self.model = model

    def think(self, prompt: str, system: str = "") -> str:
        """Send a prompt to the LLM and return the response text."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = ollama.chat(model=self.model, messages=messages)
        return response["message"]["content"].strip()

    def run(self, *args, **kwargs) -> Any:
        raise NotImplementedError
