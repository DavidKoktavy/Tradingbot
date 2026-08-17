"""
Database schema.

Design decisions:

- **Numerics are `Numeric`, never `Float`.** Storing money as a float
  reintroduces exactly the rounding error the Decimal discipline in the
  domain models exists to prevent, and it does so at the persistence
  boundary where it is hardest to notice.

- **Audit tables are append-only by convention and by API.** The
  repository exposes no update or delete for `decisions`, `fills`, or
  `risk_events`. Orders and positions are mutable current-state tables;
  their history lives in the append-only tables.

- **Every table carries `trading_mode`.** Paper and live records will end
  up in the same database at some point, and a query that silently mixes
  them produces performance numbers that are worse than having none.

- **SQLite works for local development, PostgreSQL for production.** The
  schema avoids Postgres-specific types so the same code runs against
  both, which means the test suite exercises the real persistence layer
  rather than a mock.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class OrderRow(Base):
    __tablename__ = "orders"

    order_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    broker_order_id: Mapped[str | None] = mapped_column(String(64), index=True)
    trading_mode: Mapped[str] = mapped_column(String(16), index=True)

    instrument: Mapped[str] = mapped_column(String(64), index=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    side: Mapped[str] = mapped_column(String(8))
    order_type: Mapped[str] = mapped_column(String(16))
    time_in_force: Mapped[str] = mapped_column(String(8), default="DAY")

    quantity: Mapped[float] = mapped_column(Numeric(20, 8))
    filled_quantity: Mapped[float] = mapped_column(Numeric(20, 8), default=0)
    limit_price: Mapped[float | None] = mapped_column(Numeric(20, 8))
    stop_price: Mapped[float | None] = mapped_column(Numeric(20, 8))
    stop_loss: Mapped[float | None] = mapped_column(Numeric(20, 8))
    take_profit: Mapped[float | None] = mapped_column(Numeric(20, 8))
    average_fill_price: Mapped[float | None] = mapped_column(Numeric(20, 8))

    state: Mapped[str] = mapped_column(String(24), index=True)
    source: Mapped[str] = mapped_column(String(64))
    strategy: Mapped[str | None] = mapped_column(String(64))
    signal_id: Mapped[str | None] = mapped_column(String(64))
    error_message: Mapped[str | None] = mapped_column(Text)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    fills: Mapped[list["FillRow"]] = relationship(back_populates="order")

    __table_args__ = (Index("ix_orders_mode_created", "trading_mode", "created_at"),)


class FillRow(Base):
    """Append-only."""

    __tablename__ = "fills"

    fill_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    order_id: Mapped[str] = mapped_column(ForeignKey("orders.order_id"), index=True)
    trading_mode: Mapped[str] = mapped_column(String(16), index=True)

    quantity: Mapped[float] = mapped_column(Numeric(20, 8))
    price: Mapped[float] = mapped_column(Numeric(20, 8))
    commission: Mapped[float] = mapped_column(Numeric(20, 8), default=0)
    slippage_bps: Mapped[float | None] = mapped_column(Float)
    filled_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    recorded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    order: Mapped[OrderRow] = relationship(back_populates="fills")


class PositionRow(Base):
    """Current state, keyed by instrument + mode."""

    __tablename__ = "positions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    instrument: Mapped[str] = mapped_column(String(64), index=True)
    trading_mode: Mapped[str] = mapped_column(String(16), index=True)

    quantity: Mapped[float] = mapped_column(Numeric(20, 8), default=0)
    average_cost: Mapped[float] = mapped_column(Numeric(20, 8), default=0)
    realized_pnl: Mapped[float] = mapped_column(Numeric(20, 8), default=0)
    total_commission: Mapped[float] = mapped_column(Numeric(20, 8), default=0)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    __table_args__ = (Index("ix_positions_instrument_mode", "instrument", "trading_mode"),)


class DecisionRow(Base):
    """Append-only. The full audit record, stored as JSON so the schema
    can evolve without migrating historical rows, with the fields most
    commonly queried promoted to columns."""

    __tablename__ = "decisions"

    record_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    trading_mode: Mapped[str] = mapped_column(String(16), index=True)
    cycle: Mapped[int] = mapped_column(Integer, default=0)
    instrument: Mapped[str] = mapped_column(String(64), index=True)

    intent_source: Mapped[str | None] = mapped_column(String(64), index=True)
    intent_strategy: Mapped[str | None] = mapped_column(String(64), index=True)
    risk_approved: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    risk_rejection_reason: Mapped[str | None] = mapped_column(String(64), index=True)
    order_id: Mapped[str | None] = mapped_column(String(64), index=True)
    outcome: Mapped[str] = mapped_column(String(32), index=True)

    payload: Mapped[dict] = mapped_column(JSON)

    __table_args__ = (Index("ix_decisions_mode_ts", "trading_mode", "timestamp"),)


class RiskEventRow(Base):
    """Append-only record of limit breaches, kill-switch activations, and
    halts."""

    __tablename__ = "risk_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    trading_mode: Mapped[str] = mapped_column(String(16), index=True)
    event_type: Mapped[str] = mapped_column(String(48), index=True)
    severity: Mapped[str] = mapped_column(String(16))
    detail: Mapped[str] = mapped_column(Text)
    context: Mapped[dict] = mapped_column(JSON, default=dict)


class AccountSnapshotRow(Base):
    """Periodic account state, for equity curve reconstruction."""

    __tablename__ = "account_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    trading_mode: Mapped[str] = mapped_column(String(16), index=True)
    equity: Mapped[float] = mapped_column(Numeric(20, 8))
    cash: Mapped[float] = mapped_column(Numeric(20, 8))
    buying_power: Mapped[float] = mapped_column(Numeric(20, 8))
    realised_pnl: Mapped[float] = mapped_column(Numeric(20, 8), default=0)
    unrealised_pnl: Mapped[float | None] = mapped_column(Numeric(20, 8))
    gross_exposure: Mapped[float | None] = mapped_column(Numeric(20, 8))
    open_positions: Mapped[int] = mapped_column(Integer, default=0)


class StrategyVersionRow(Base):
    """Which strategy code and parameters were active, so a decision months
    old can be traced to the exact configuration that produced it."""

    __tablename__ = "strategy_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(64), index=True)
    version: Mapped[str] = mapped_column(String(32))
    params: Mapped[dict] = mapped_column(JSON, default=dict)
    active_from: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    trading_mode: Mapped[str] = mapped_column(String(16), index=True)
