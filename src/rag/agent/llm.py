"""LLM access behind a protocol.

Two implementations. `AnthropicClient` is the real one. `ScriptedClient`
returns queued responses, so every graph path is testable without a key, a
network call or a bill.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from rag.config.settings import LLMSettings
from rag.log import get_logger

log = get_logger(__name__)


class LLMUnavailableError(RuntimeError):
    """No API key, or the provider rejected the request."""


@dataclass(frozen=True)
class Completion:
    text: str
    model: str


class LLMClient(Protocol):
    async def complete(self, system: str, user: str, model: str) -> Completion: ...


@dataclass
class ScriptedClient:
    """Deterministic stand in. Pops one queued response per call."""

    responses: list[str] = field(default_factory=list)
    calls: list[tuple[str, str, str]] = field(default_factory=list)

    async def complete(self, system: str, user: str, model: str) -> Completion:
        self.calls.append((system, user, model))
        if not self.responses:
            raise LLMUnavailableError("ScriptedClient ran out of queued responses")
        return Completion(self.responses.pop(0), model)


class AnthropicClient:
    def __init__(self, settings: LLMSettings) -> None:
        self._settings = settings
        self._client: Any = None

    def _load(self) -> Any:
        if self._client is None:
            import anthropic

            if not self._settings.api_key:
                raise LLMUnavailableError(
                    "no API key. Set ANTHROPIC_API_KEY in .env to enable /ask and "
                    "/agent, or run with the scripted client in tests"
                )
            self._client = anthropic.AsyncAnthropic(api_key=self._settings.api_key)
        return self._client

    async def complete(self, system: str, user: str, model: str) -> Completion:
        client = self._load()
        response = await client.messages.create(
            model=model,
            max_tokens=self._settings.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            timeout=self._settings.timeout_seconds,
        )
        return Completion(_text_of(response), model)


def _text_of(response: Any) -> str:
    parts = [block.text for block in response.content if hasattr(block, "text")]
    return "".join(parts)


def build_client(settings: LLMSettings) -> LLMClient:
    return AnthropicClient(settings)
