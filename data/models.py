"""
Normalized market-data models.

Design decision: strategies, the risk engine, and the AI layer must never
depend on ib_async types directly. Everything broker-specific is translated
into these models at the edge (broker/ibkr_client.py). This means:
  - Strategies/backtests can run against synthetic or historical data with
    zero broker code involved.
  - Swapping or upgrading the broker library only touches the translation
    layer, not the rest of the system.
  - "Never trade on stale data" is enforced against a single well-defined
    `timestamp` field, regardless of where the data came from.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum

from pydantic import BaseModel, Field, model_validator


class AssetClass(StrEnum):
    STOCK = "STOCK"
    FOREX = "FOREX"
    FUTURE = "FUTURE"
    OPTION = "OPTION"
    CRYPTO = "CRYPTO"


class Instrument(BaseModel):
    """A normalized, broker-agnostic instrument identifier."""

    symbol: str
    asset_class: AssetClass = AssetClass.STOCK
    exchange: str = "SMART"
    currency: str = "USD"

    def __str__(self) -> str:
        return f"{self.symbol}:{self.exchange}:{self.currency}"


class Bar(BaseModel):
    """A single OHLCV bar."""

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0

    @model_validator(mode="after")
    def _sane_ohlc(self) -> "Bar":
        if not (self.low <= self.open <= self.high and self.low <= self.close <= self.high):
            raise ValueError(
                f"Inconsistent OHLC bar: O={self.open} H={self.high} "
                f"L={self.low} C={self.close}"
            )
        if self.low > self.high:
            raise ValueError("low > high in bar")
        return self


class MarketSnapshot(BaseModel):
    """
    A point-in-time normalized view of a symbol's market state.

    `timestamp` is always UTC and always reflects when the *underlying*
    data was produced (not when we received/processed it), so staleness
    checks are meaningful even under processing delay.
    """

    instrument: Instrument
    timestamp: datetime
    bid: float | None = None
    ask: float | None = None
    last: float | None = None
    volume: float | None = None
    bars: list[Bar] = Field(default_factory=list)

    @model_validator(mode="after")
    def _timestamp_is_utc(self) -> "MarketSnapshot":
        if self.timestamp.tzinfo is None:
            raise ValueError("MarketSnapshot.timestamp must be timezone-aware (UTC)")
        return self

    @property
    def mid(self) -> float | None:
        if self.bid is not None and self.ask is not None:
            return (self.bid + self.ask) / 2
        return self.last

    def age_seconds(self, *, now: datetime | None = None) -> float:
        now = now or datetime.now(timezone.utc)
        return (now - self.timestamp).total_seconds()

    def is_stale(self, max_age_seconds: float, *, now: datetime | None = None) -> bool:
        return self.age_seconds(now=now) > max_age_seconds
