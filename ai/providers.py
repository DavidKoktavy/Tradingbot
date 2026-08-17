"""
AI provider abstraction.

Design decisions:

- The interface is deliberately minimal: text in, text out, with a hard
  timeout. Providers get no access to the broker, portfolio, filesystem,
  or shell — there is nothing in this interface to grant it through.

- **No tool use and no code execution.** The system never sends the model
  tools that touch trading, and never executes text the model returns. The
  spec's requirement that AI-generated code be sandboxed is satisfied here
  by not having a code-execution path at all in the trading loop. Research
  code generation (Phase 22 of the spec) would be a separate offline
  process with its own isolation.

- Timeouts are enforced by the caller via `asyncio.wait_for`, and a
  timeout is a *rejection*, not a retry-until-success. An AI that is slow
  during a fast market is exactly when you least want to wait for it.

- `NullProvider` is the default in the dependency container. If the AI is
  unconfigured, the system runs on deterministic strategies alone rather
  than failing to start — the AI is an enhancement, never a dependency.
"""

from __future__ import annotations

import asyncio
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass

import structlog

log = structlog.get_logger(__name__)


class AIProviderError(Exception):
    """Provider failed to return a usable response."""


@dataclass(frozen=True)
class AIRequest:
    system_prompt: str
    user_prompt: str
    max_tokens: int = 1024
    temperature: float = 0.0  # deterministic by default; see module docstring


class AIProvider(ABC):
    """Text-in, text-out. Nothing else."""

    name: str = "abstract"

    @abstractmethod
    async def complete(self, request: AIRequest) -> str:
        """Return the model's raw text response. Raise AIProviderError on
        any failure — callers must not have to distinguish provider-
        specific exception types."""

    @property
    def is_available(self) -> bool:
        return True


class NullProvider(AIProvider):
    """Used when no AI is configured. Always declines to decide, so the
    system degrades to deterministic strategies rather than refusing to
    run."""

    name = "null"

    async def complete(self, request: AIRequest) -> str:
        return json.dumps(
            {
                "action": "HOLD",
                "symbol": "N/A",
                "confidence": 0.0,
                "reasoning": "No AI provider configured",
            }
        )

    @property
    def is_available(self) -> bool:
        return False


class AnthropicProvider(AIProvider):
    """Anthropic API provider.

    Not exercised in the test suite (no network, no key). The HTTP call is
    isolated to `_call`, so everything around it — prompt construction,
    parsing, validation, risk gating — is fully tested against fakes.
    """

    name = "anthropic"

    def __init__(self, api_key: str, model: str, *, timeout_seconds: float = 10.0) -> None:
        if not api_key:
            raise ValueError("AnthropicProvider requires an API key")
        self._api_key = api_key
        self._model = model
        self._timeout = timeout_seconds

    async def complete(self, request: AIRequest) -> str:
        try:
            return await asyncio.wait_for(self._call(request), timeout=self._timeout)
        except TimeoutError as exc:
            raise AIProviderError(f"Provider timed out after {self._timeout}s") from exc
        except AIProviderError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise AIProviderError(f"Provider call failed: {exc}") from exc

    async def _call(self, request: AIRequest) -> str:
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover
            raise AIProviderError("httpx is required for AnthropicProvider") from exc

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": self._api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": self._model,
                    "max_tokens": request.max_tokens,
                    "temperature": request.temperature,
                    "system": request.system_prompt,
                    "messages": [{"role": "user", "content": request.user_prompt}],
                },
            )
            if response.status_code != 200:
                raise AIProviderError(f"HTTP {response.status_code}: {response.text[:200]}")
            payload = response.json()

        blocks = payload.get("content", [])
        text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
        if not text:
            raise AIProviderError("Provider returned no text content")
        return text


class ScriptedProvider(AIProvider):
    """Test double: returns queued responses in order. Deterministic, no
    network. Used throughout the test suite."""

    name = "scripted"

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.requests: list[AIRequest] = []

    async def complete(self, request: AIRequest) -> str:
        self.requests.append(request)
        if not self._responses:
            raise AIProviderError("ScriptedProvider exhausted")
        return self._responses.pop(0)


class FailingProvider(AIProvider):
    """Test double that always fails, for verifying fail-closed paths."""

    name = "failing"

    def __init__(self, exc: Exception | None = None) -> None:
        self._exc = exc or AIProviderError("simulated provider failure")

    async def complete(self, request: AIRequest) -> str:
        raise self._exc


class HangingProvider(AIProvider):
    """Test double that never returns, for verifying timeout handling."""

    name = "hanging"

    async def complete(self, request: AIRequest) -> str:
        await asyncio.sleep(3600)
        return ""
