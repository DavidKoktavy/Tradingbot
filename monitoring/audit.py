"""
Decision audit trail.

The requirement from the spec: it must be possible to answer *"why did the
agent make this trade?"* months later.

Design decisions:

- **One record per decision, containing the whole chain**: market state,
  features, regime, every strategy signal, the AI's raw response and
  verdict, every risk check with its outcome, the order intent, the
  approved quantity, the broker response, and the resulting fills. A trail
  split across ten tables that must be joined by timestamp is a trail
  nobody reconstructs correctly under pressure.

- **The record is written even when nothing is traded.** Rejections are
  the more interesting half of the audit trail: "why did the agent *not*
  trade" and "why did it keep rejecting" are the questions that actually
  come up. Recording only fills would hide every risk-limit interaction.

- **Records are append-only and immutable once written.** There is no
  update or delete path. The spec explicitly forbids the AI from deleting
  audit logs or hiding losing trades; the way to guarantee that is to have
  no code that can do it.

- **Serialisation never fails silently.** If a field cannot be serialised
  it is stringified rather than dropped, because a missing field in an
  audit record is worse than an ugly one.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

import structlog
from pydantic import BaseModel, Field

log = structlog.get_logger(__name__)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, BaseModel):
        return json.loads(value.model_dump_json())
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)  # never drop a field


class RiskCheckRecord(BaseModel):
    check_name: str
    approved: bool
    reason: str | None = None
    detail: str = ""


class SignalRecord(BaseModel):
    strategy: str
    direction: str
    strength: float
    rationale: str = ""
    features: dict[str, float] = Field(default_factory=dict)


class FillRecord(BaseModel):
    fill_id: str
    quantity: str
    price: str
    commission: str
    timestamp: str


class DecisionRecord(BaseModel):
    """The complete story of one decision. Immutable once created."""

    model_config = {"frozen": True}

    record_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    cycle: int = 0
    trading_mode: str = "PAPER"
    instrument: str = ""

    # Market state at decision time
    bid: str | None = None
    ask: str | None = None
    last: str | None = None
    mid: str | None = None
    data_age_seconds: float | None = None

    # Analysis
    regime: str | None = None
    regime_confidence: float | None = None
    regime_features: dict[str, float] = Field(default_factory=dict)
    signals: list[SignalRecord] = Field(default_factory=list)

    # AI
    ai_consulted: bool = False
    ai_accepted: bool = False
    ai_action: str | None = None
    ai_confidence: float | None = None
    ai_reasoning: str = ""
    ai_rejection_reason: str | None = None
    ai_raw_response: str = ""
    ai_latency_ms: float | None = None

    # Proposal
    intent_id: str | None = None
    intent_source: str | None = None
    intent_strategy: str | None = None
    intent_side: str | None = None
    requested_quantity: str | None = None
    stop_loss: str | None = None
    take_profit: str | None = None

    # Risk
    risk_checks: list[RiskCheckRecord] = Field(default_factory=list)
    risk_approved: bool = False
    risk_rejection_reason: str | None = None
    approved_quantity: str | None = None
    was_reduced: bool = False

    # Execution
    order_id: str | None = None
    broker_order_id: str | None = None
    submitted: bool = False
    submission_error: str | None = None
    final_order_state: str | None = None
    fills: list[FillRecord] = Field(default_factory=list)
    average_fill_price: str | None = None
    slippage_bps: float | None = None

    # Portfolio impact
    position_after: str | None = None
    realised_pnl_after: str | None = None
    equity_after: str | None = None

    outcome: str = "NO_ACTION"

    def explain(self) -> str:
        """Human-readable narrative. This is what an operator reads when
        asking 'why did the agent do this?'"""
        lines = [
            f"Decision {self.record_id} at {self.timestamp.isoformat()} "
            f"[{self.trading_mode}] cycle {self.cycle}",
            f"Instrument: {self.instrument}  mid={self.mid}  "
            f"data_age={self.data_age_seconds}s",
        ]
        if self.regime:
            lines.append(f"Regime: {self.regime} (confidence {self.regime_confidence})")
        for signal in self.signals:
            lines.append(
                f"  Signal [{signal.strategy}] {signal.direction} "
                f"strength={signal.strength:.2f} — {signal.rationale}"
            )
        if self.ai_consulted:
            if self.ai_accepted:
                lines.append(
                    f"  AI: {self.ai_action} confidence={self.ai_confidence} — "
                    f"{self.ai_reasoning[:200]}"
                )
            else:
                lines.append(f"  AI: rejected ({self.ai_rejection_reason})")
        if self.intent_id:
            lines.append(
                f"Proposal: {self.intent_side} {self.requested_quantity} "
                f"from {self.intent_source} (stop={self.stop_loss}, target={self.take_profit})"
            )
        for check in self.risk_checks:
            verdict = "PASS" if check.approved else f"FAIL ({check.reason})"
            lines.append(f"  Risk[{check.check_name}]: {verdict} {check.detail}")
        if self.risk_approved:
            note = " (reduced)" if self.was_reduced else ""
            lines.append(f"Risk: APPROVED {self.approved_quantity}{note}")
        elif self.intent_id:
            lines.append(f"Risk: REJECTED — {self.risk_rejection_reason}")
        if self.submitted:
            lines.append(f"Submitted as {self.broker_order_id}")
        if self.submission_error:
            lines.append(f"Submission FAILED: {self.submission_error}")
        for fill in self.fills:
            lines.append(f"  Fill {fill.quantity} @ {fill.price} (commission {fill.commission})")
        if self.slippage_bps is not None:
            lines.append(f"Slippage: {self.slippage_bps:.1f} bps")
        lines.append(f"Outcome: {self.outcome}")
        return "\n".join(lines)

    def to_json(self) -> str:
        return json.dumps(_jsonable(self.model_dump()), separators=(",", ":"))


class DecisionRecorder:
    """Append-only sink for decision records.

    Deliberately exposes no update or delete method — see module docstring.
    """

    def __init__(
        self,
        *,
        path: Path | str | None = None,
        keep_in_memory: int = 1000,
        emit_to_log: bool = True,
    ) -> None:
        self._path = Path(path) if path else None
        self._records: list[DecisionRecord] = []
        self._keep = keep_in_memory
        self._emit = emit_to_log
        self._write_failures = 0
        if self._path is not None:
            self._path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def write_failures(self) -> int:
        return self._write_failures

    def record(self, record: DecisionRecord) -> DecisionRecord:
        """Persist a record. Never raises: losing the ability to trade
        because an audit sink is full would be a worse failure than a
        gap in the trail, but the gap is counted and logged loudly."""
        self._records.append(record)
        if len(self._records) > self._keep:
            self._records = self._records[-self._keep :]

        if self._emit:
            log.info(
                "decision.recorded",
                record_id=record.record_id,
                instrument=record.instrument,
                outcome=record.outcome,
                risk_approved=record.risk_approved,
                source=record.intent_source,
            )

        if self._path is not None:
            try:
                with self._path.open("a", encoding="utf-8") as handle:
                    handle.write(record.to_json() + "\n")
            except Exception as exc:  # noqa: BLE001
                self._write_failures += 1
                log.error(
                    "decision.write_failed",
                    record_id=record.record_id,
                    error=str(exc),
                    total_failures=self._write_failures,
                )
        return record

    def recent(self, limit: int = 50) -> list[DecisionRecord]:
        return self._records[-limit:][::-1]

    def find(self, record_id: str) -> DecisionRecord | None:
        return next((r for r in self._records if r.record_id == record_id), None)

    def for_instrument(self, instrument: str) -> list[DecisionRecord]:
        return [r for r in self._records if r.instrument == instrument]

    def rejections(self) -> list[DecisionRecord]:
        return [r for r in self._records if not r.risk_approved and r.intent_id]

    def replay(self, path: Path | str | None = None) -> list[DecisionRecord]:
        """Read records back from disk. Used to answer questions about
        past sessions."""
        target = Path(path) if path else self._path
        if target is None or not target.exists():
            return []
        out = []
        with target.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(DecisionRecord.model_validate_json(line))
                except Exception as exc:  # noqa: BLE001
                    log.warning("decision.replay_skipped_line", error=str(exc))
        return out


def compute_slippage_bps(
    *, expected_price: Decimal, fill_price: Decimal, side: str
) -> float:
    """Positive = worse than expected. Sign is normalised by side so a
    single threshold works for both directions."""
    if expected_price <= 0:
        return 0.0
    diff = fill_price - expected_price
    if side.upper() == "SELL":
        diff = -diff
    return float(diff / expected_price * 10000)
