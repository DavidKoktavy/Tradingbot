"""
Schemas for AI output.

Design decisions:

- The AI returns JSON conforming to `AIDecision`, validated strictly
  (`extra="forbid"`). An unparseable or non-conforming response is a
  rejection, never a partial interpretation. Salvaging "most of" a
  malformed decision is how a hallucinated field becomes a trade.

- **The schema deliberately omits any field that could influence risk
  controls.** There is no `max_position_size`, no `override_risk`, no
  `urgency`, no `enable_live`. If the model emits one, `extra="forbid"`
  rejects the whole response. This is enforcement by absence: the model
  cannot ask for something the schema has no room to express.

- `confidence` and `risk_score` exist for logging, analysis, and the
  research loop. They are **not** consumed by the risk engine and cannot
  enlarge a position. High confidence buys the model nothing.

- `reasoning` is free text and is treated as *display and audit data
  only*. It is never parsed for instructions, never eval'd, never used to
  branch logic. It is length-capped so a runaway generation can't bloat
  the audit log.

- Symbols are validated against an explicit allowlist supplied by the
  caller, not accepted as free text. A model that hallucinates a ticker,
  or is induced by prompt injection to name one, gets rejected before any
  order object exists.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from decimal import Decimal
from enum import StrEnum

from pydantic import BaseModel, Field, field_validator, model_validator

MAX_REASONING_CHARS = 2000
_SYMBOL_PATTERN = re.compile(r"^[A-Z][A-Z0-9.\-]{0,11}$")


class AIAction(StrEnum):
    BUY = "BUY"
    SELL = "SELL"
    CLOSE = "CLOSE"
    HOLD = "HOLD"


class TimeHorizon(StrEnum):
    INTRADAY = "intraday"
    SWING = "swing"
    POSITION = "position"


class MarketRegime(StrEnum):
    TRENDING_UP = "TRENDING_UP"
    TRENDING_DOWN = "TRENDING_DOWN"
    RANGING = "RANGING"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    UNKNOWN = "UNKNOWN"


class AIDecision(BaseModel):
    """Structured AI output. Strict: unknown fields are rejected outright,
    which is what prevents the model from inventing control parameters."""

    model_config = {"extra": "forbid", "frozen": True}

    action: AIAction
    symbol: str
    confidence: float = Field(ge=0.0, le=1.0)
    strategy: str = ""
    entry: Decimal | None = None
    stop_loss: Decimal | None = None
    take_profit: Decimal | None = None
    reasoning: str = ""
    time_horizon: TimeHorizon = TimeHorizon.INTRADAY
    risk_score: float = Field(default=0.5, ge=0.0, le=1.0)

    @field_validator("symbol")
    @classmethod
    def _symbol_shape(cls, v: str) -> str:
        v = v.strip().upper()
        if not _SYMBOL_PATTERN.match(v):
            raise ValueError(f"Symbol {v!r} is not a plausible ticker")
        return v

    @field_validator("reasoning")
    @classmethod
    def _cap_reasoning(cls, v: str) -> str:
        return v[:MAX_REASONING_CHARS]

    @field_validator("entry", "stop_loss", "take_profit")
    @classmethod
    def _positive_prices(cls, v: Decimal | None) -> Decimal | None:
        if v is not None and v <= 0:
            raise ValueError("Prices must be positive")
        return v

    @model_validator(mode="after")
    def _directional_coherence(self) -> "AIDecision":
        """A BUY whose stop sits above its target is incoherent. Rather
        than silently repairing it, reject: incoherent output means the
        model's reasoning is unreliable for this decision."""
        if self.action is AIAction.BUY and self.stop_loss and self.take_profit:
            if self.stop_loss >= self.take_profit:
                raise ValueError("BUY: stop_loss must be below take_profit")
            if self.entry and not (self.stop_loss < self.entry < self.take_profit):
                raise ValueError("BUY: entry must sit between stop_loss and take_profit")
        if self.action is AIAction.SELL and self.stop_loss and self.take_profit:
            if self.stop_loss <= self.take_profit:
                raise ValueError("SELL: stop_loss must be above take_profit")
            if self.entry and not (self.take_profit < self.entry < self.stop_loss):
                raise ValueError("SELL: entry must sit between take_profit and stop_loss")
        return self

    @property
    def is_actionable(self) -> bool:
        return self.action in (AIAction.BUY, AIAction.SELL, AIAction.CLOSE)


class AIAnalysis(BaseModel):
    """Non-trading analytical output: regime views, commentary, research.
    Kept separate from AIDecision so analysis can never be mistaken for a
    trade instruction by a downstream consumer."""

    model_config = {"extra": "forbid"}

    regime: MarketRegime = MarketRegime.UNKNOWN
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    summary: str = ""
    observations: list[str] = Field(default_factory=list)

    @field_validator("summary")
    @classmethod
    def _cap(cls, v: str) -> str:
        return v[:MAX_REASONING_CHARS]


class AIRejectionReason(StrEnum):
    PROVIDER_ERROR = "PROVIDER_ERROR"
    PROVIDER_TIMEOUT = "PROVIDER_TIMEOUT"
    MALFORMED_JSON = "MALFORMED_JSON"
    SCHEMA_VIOLATION = "SCHEMA_VIOLATION"
    UNKNOWN_SYMBOL = "UNKNOWN_SYMBOL"
    SYMBOL_MISMATCH = "SYMBOL_MISMATCH"
    PRICE_IMPLAUSIBLE = "PRICE_IMPLAUSIBLE"
    MISSING_STOP = "MISSING_STOP"
    NO_MARKET_DATA = "NO_MARKET_DATA"
    NOT_ACTIONABLE = "NOT_ACTIONABLE"
    INJECTION_SUSPECTED = "INJECTION_SUSPECTED"


class AIDecisionResult(BaseModel):
    """Outcome of asking the AI for a decision. Defaults to *no decision*,
    so any failure path yields no trade rather than an accidental one."""

    accepted: bool = False
    decision: AIDecision | None = None
    reason: AIRejectionReason | None = None
    detail: str = ""
    raw_response: str = ""
    latency_ms: float = 0.0
    evaluated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @classmethod
    def reject(
        cls, reason: AIRejectionReason, detail: str = "", raw: str = ""
    ) -> "AIDecisionResult":
        return cls(accepted=False, reason=reason, detail=detail, raw_response=raw[:4000])

    @classmethod
    def accept(cls, decision: AIDecision, raw: str = "", latency_ms: float = 0.0) -> "AIDecisionResult":
        return cls(
            accepted=True, decision=decision, raw_response=raw[:4000], latency_ms=latency_ms
        )


def parse_ai_json(raw: str) -> dict:
    """Extract a JSON object from a model response.

    Tolerates markdown fences and surrounding prose, because those are
    formatting noise rather than semantic content. Does NOT tolerate
    multiple objects or malformed JSON — ambiguity is rejected.
    """
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()

    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in response")

    obj = json.loads(text[start : end + 1])
    if not isinstance(obj, dict):
        raise ValueError("Top-level JSON value is not an object")
    return obj
