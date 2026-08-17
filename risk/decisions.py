"""
Risk decision vocabulary.

Design decisions:

- Every risk check returns a structured `RiskDecision`, never a bare bool.
  "Rejected" without a machine-readable reason is useless for the audit
  trail, for alerting, and for the AI feedback loop (which must be able to
  learn *which* limit it keeps hitting without being able to change it).

- `RiskDecision.approved` defaults to False. A check that forgets to set
  a verdict, or throws partway through, fails closed rather than open.
  This is the single most important default in the module.

- Reasons are an enum, not free text, so alerting and metrics can count
  them. Free-text detail rides alongside for humans.
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from enum import StrEnum

from pydantic import BaseModel, Field


class RejectionReason(StrEnum):
    # Hard blocks
    KILL_SWITCH_ACTIVE = "KILL_SWITCH_ACTIVE"
    TRADING_HALTED = "TRADING_HALTED"
    NOT_CONNECTED = "NOT_CONNECTED"
    RECONCILIATION_REQUIRED = "RECONCILIATION_REQUIRED"

    # Data quality
    STALE_MARKET_DATA = "STALE_MARKET_DATA"
    MISSING_MARKET_DATA = "MISSING_MARKET_DATA"
    PRICE_SANITY_FAILED = "PRICE_SANITY_FAILED"
    SPREAD_TOO_WIDE = "SPREAD_TOO_WIDE"
    INSUFFICIENT_LIQUIDITY = "INSUFFICIENT_LIQUIDITY"

    # Risk limits
    MAX_DAILY_LOSS_BREACHED = "MAX_DAILY_LOSS_BREACHED"
    MAX_DRAWDOWN_BREACHED = "MAX_DRAWDOWN_BREACHED"
    MAX_POSITION_SIZE_EXCEEDED = "MAX_POSITION_SIZE_EXCEEDED"
    MAX_GROSS_EXPOSURE_EXCEEDED = "MAX_GROSS_EXPOSURE_EXCEEDED"
    MAX_SECTOR_EXPOSURE_EXCEEDED = "MAX_SECTOR_EXPOSURE_EXCEEDED"
    MAX_CORRELATED_EXPOSURE_EXCEEDED = "MAX_CORRELATED_EXPOSURE_EXCEEDED"
    MAX_OPEN_POSITIONS_EXCEEDED = "MAX_OPEN_POSITIONS_EXCEEDED"
    MAX_RISK_PER_TRADE_EXCEEDED = "MAX_RISK_PER_TRADE_EXCEEDED"
    MAX_ORDER_RATE_EXCEEDED = "MAX_ORDER_RATE_EXCEEDED"
    INSUFFICIENT_BUYING_POWER = "INSUFFICIENT_BUYING_POWER"

    # Order integrity
    DUPLICATE_ORDER = "DUPLICATE_ORDER"
    INVALID_ORDER = "INVALID_ORDER"
    OUTSIDE_TRADING_HOURS = "OUTSIDE_TRADING_HOURS"
    ZERO_QUANTITY = "ZERO_QUANTITY"

    # Internal
    INTERNAL_ERROR = "INTERNAL_ERROR"


class RiskDecision(BaseModel):
    """Result of a risk evaluation. Defaults to rejection: a check that
    fails to set a verdict must not accidentally approve a trade."""

    approved: bool = False
    reason: RejectionReason | None = None
    detail: str = ""
    check_name: str = ""
    evaluated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @classmethod
    def approve(cls, check_name: str, detail: str = "") -> "RiskDecision":
        return cls(approved=True, check_name=check_name, detail=detail)

    @classmethod
    def reject(
        cls, check_name: str, reason: RejectionReason, detail: str = ""
    ) -> "RiskDecision":
        return cls(approved=False, check_name=check_name, reason=reason, detail=detail)


class RiskAssessment(BaseModel):
    """Aggregate verdict across every check that ran, plus the final
    approved quantity (which may be smaller than requested if the position
    sizer trimmed it)."""

    approved: bool = False
    decisions: list[RiskDecision] = Field(default_factory=list)
    approved_quantity: Decimal = Decimal("0")
    requested_quantity: Decimal = Decimal("0")
    evaluated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def rejections(self) -> list[RiskDecision]:
        return [d for d in self.decisions if not d.approved]

    @property
    def first_rejection(self) -> RiskDecision | None:
        return self.rejections[0] if self.rejections else None

    @property
    def reason(self) -> RejectionReason | None:
        first = self.first_rejection
        return first.reason if first else None

    @property
    def was_reduced(self) -> bool:
        return self.approved and self.approved_quantity < self.requested_quantity

    def summary(self) -> str:
        if self.approved:
            note = (
                f" (reduced from {self.requested_quantity})" if self.was_reduced else ""
            )
            return f"APPROVED {self.approved_quantity}{note}"
        first = self.first_rejection
        return f"REJECTED {first.reason}: {first.detail}" if first else "REJECTED"
