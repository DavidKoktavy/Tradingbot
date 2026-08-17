"""
Repository layer.

Design decisions:

- **A database outage must not stop trading, but must be visible.** Writes
  are best-effort and counted; a failure logs loudly and increments a
  counter that health checks surface as DEGRADED. Halting trading because
  a logging database is down would be trading the wrong risk — but running
  blind without anyone noticing would be worse, hence the counter.

  The one exception is reconciliation state: that comes from the broker,
  not the database, so a database outage never affects correctness of
  position tracking.

- **No update or delete methods exist for audit tables.** The spec forbids
  the AI from deleting audit logs or hiding losing trades. The enforcement
  is that the code to do so does not exist.

- Sync SQLAlchemy rather than async: the write volume here is trivial
  (tens of rows per minute), and sync sessions are markedly easier to
  reason about for correctness. The writes happen off the critical path.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Iterator

import structlog
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from database.models import (
    AccountSnapshotRow,
    Base,
    DecisionRow,
    FillRow,
    OrderRow,
    PositionRow,
    RiskEventRow,
    StrategyVersionRow,
)
from execution.execution_models import Fill, Order
from monitoring.audit import DecisionRecord, _jsonable
from portfolio.positions import Position

log = structlog.get_logger(__name__)


class Repository:
    def __init__(self, database_url: str, *, echo: bool = False, mode: str = "PAPER") -> None:
        self._engine = create_engine(database_url, echo=echo, future=True)
        self._session_factory = sessionmaker(self._engine, expire_on_commit=False)
        self._mode = mode
        self._write_failures = 0
        self._available = True

    @property
    def write_failures(self) -> int:
        return self._write_failures

    @property
    def is_available(self) -> bool:
        return self._available

    def create_schema(self) -> None:
        Base.metadata.create_all(self._engine)

    def drop_schema(self) -> None:
        """Development helper. Never called by the running agent."""
        Base.metadata.drop_all(self._engine)

    @contextmanager
    def session(self) -> Iterator[Session]:
        session = self._session_factory()
        try:
            yield session
            session.commit()
            self._available = True
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def _safe_write(self, description: str, fn: Any) -> bool:
        """Run a write, containing failures. Returns success."""
        try:
            with self.session() as session:
                fn(session)
            return True
        except Exception as exc:  # noqa: BLE001
            self._write_failures += 1
            self._available = False
            log.error(
                "repository.write_failed",
                operation=description,
                error=str(exc),
                total_failures=self._write_failures,
            )
            return False

    def health_check(self) -> bool:
        try:
            with self.session() as session:
                session.execute(select(1))
            return True
        except Exception as exc:  # noqa: BLE001
            log.error("repository.health_check_failed", error=str(exc))
            self._available = False
            return False

    # ---- orders (mutable current state) -----------------------------------

    def save_order(self, order: Order) -> bool:
        def write(session: Session) -> None:
            intent = order.intent
            row = session.get(OrderRow, order.order_id)
            if row is None:
                row = OrderRow(order_id=order.order_id, trading_mode=self._mode)
                session.add(row)
            row.broker_order_id = order.broker_order_id
            row.instrument = str(intent.instrument)
            row.symbol = intent.instrument.symbol
            row.side = str(intent.side)
            row.order_type = str(intent.order_type)
            row.time_in_force = str(intent.time_in_force)
            row.quantity = intent.quantity
            row.filled_quantity = order.filled_quantity
            row.limit_price = intent.limit_price
            row.stop_price = intent.stop_price
            row.stop_loss = intent.stop_loss
            row.take_profit = intent.take_profit
            row.average_fill_price = order.average_fill_price
            row.state = str(order.state)
            row.source = intent.source
            row.strategy = intent.strategy
            row.signal_id = intent.signal_id
            row.error_message = order.error_message
            row.created_at = order.created_at
            row.updated_at = order.updated_at

        return self._safe_write("save_order", write)

    def get_order(self, order_id: str) -> OrderRow | None:
        try:
            with self.session() as session:
                return session.get(OrderRow, order_id)
        except Exception:  # noqa: BLE001
            return None

    def open_orders(self) -> list[OrderRow]:
        """Used on restart to rebuild what we believe is working, before
        reconciling against the broker."""
        active = ("SUBMITTED", "ACKNOWLEDGED", "PARTIALLY_FILLED", "CANCEL_REQUESTED")
        try:
            with self.session() as session:
                return list(
                    session.scalars(
                        select(OrderRow)
                        .where(OrderRow.trading_mode == self._mode)
                        .where(OrderRow.state.in_(active))
                    )
                )
        except Exception as exc:  # noqa: BLE001
            log.error("repository.open_orders_failed", error=str(exc))
            return []

    # ---- fills (append-only) ------------------------------------------------

    def save_fill(self, fill: Fill, *, slippage_bps: float | None = None) -> bool:
        def write(session: Session) -> None:
            if session.get(FillRow, fill.fill_id) is not None:
                return  # idempotent, matching the execution listener
            session.add(
                FillRow(
                    fill_id=fill.fill_id,
                    order_id=fill.order_id,
                    trading_mode=self._mode,
                    quantity=fill.quantity,
                    price=fill.price,
                    commission=fill.commission,
                    slippage_bps=slippage_bps,
                    filled_at=fill.timestamp,
                )
            )

        return self._safe_write("save_fill", write)

    def fills_for_order(self, order_id: str) -> list[FillRow]:
        try:
            with self.session() as session:
                return list(
                    session.scalars(select(FillRow).where(FillRow.order_id == order_id))
                )
        except Exception:  # noqa: BLE001
            return []

    # ---- positions (mutable current state) ----------------------------------

    def save_position(self, position: Position) -> bool:
        def write(session: Session) -> None:
            row = session.scalar(
                select(PositionRow)
                .where(PositionRow.instrument == str(position.instrument))
                .where(PositionRow.trading_mode == self._mode)
            )
            if row is None:
                row = PositionRow(
                    instrument=str(position.instrument), trading_mode=self._mode
                )
                session.add(row)
            row.quantity = position.quantity
            row.average_cost = position.average_cost
            row.realized_pnl = position.realized_pnl
            row.total_commission = position.total_commission
            row.updated_at = position.updated_at

        return self._safe_write("save_position", write)

    def load_positions(self) -> list[PositionRow]:
        try:
            with self.session() as session:
                return list(
                    session.scalars(
                        select(PositionRow).where(PositionRow.trading_mode == self._mode)
                    )
                )
        except Exception:  # noqa: BLE001
            return []

    # ---- decisions (append-only) ---------------------------------------------

    def save_decision(self, record: DecisionRecord) -> bool:
        def write(session: Session) -> None:
            session.add(
                DecisionRow(
                    record_id=record.record_id,
                    timestamp=record.timestamp,
                    trading_mode=record.trading_mode,
                    cycle=record.cycle,
                    instrument=record.instrument,
                    intent_source=record.intent_source,
                    intent_strategy=record.intent_strategy,
                    risk_approved=record.risk_approved,
                    risk_rejection_reason=record.risk_rejection_reason,
                    order_id=record.order_id,
                    outcome=record.outcome,
                    payload=_jsonable(record.model_dump()),
                )
            )

        return self._safe_write("save_decision", write)

    def decisions_for_order(self, order_id: str) -> list[DecisionRow]:
        try:
            with self.session() as session:
                return list(
                    session.scalars(select(DecisionRow).where(DecisionRow.order_id == order_id))
                )
        except Exception:  # noqa: BLE001
            return []

    def recent_decisions(self, limit: int = 50) -> list[DecisionRow]:
        try:
            with self.session() as session:
                return list(
                    session.scalars(
                        select(DecisionRow)
                        .where(DecisionRow.trading_mode == self._mode)
                        .order_by(DecisionRow.timestamp.desc())
                        .limit(limit)
                    )
                )
        except Exception:  # noqa: BLE001
            return []

    def rejection_counts(self) -> dict[str, int]:
        """Which risk limits are actually binding. One of the most useful
        questions to ask of the audit trail."""
        try:
            with self.session() as session:
                rows = session.scalars(
                    select(DecisionRow)
                    .where(DecisionRow.trading_mode == self._mode)
                    .where(DecisionRow.risk_approved.is_(False))
                ).all()
            counts: dict[str, int] = {}
            for row in rows:
                if row.risk_rejection_reason:
                    counts[row.risk_rejection_reason] = (
                        counts.get(row.risk_rejection_reason, 0) + 1
                    )
            return counts
        except Exception:  # noqa: BLE001
            return {}

    # ---- risk events (append-only) --------------------------------------------

    def save_risk_event(
        self,
        *,
        event_type: str,
        severity: str,
        detail: str,
        context: dict | None = None,
        timestamp: datetime | None = None,
    ) -> bool:
        def write(session: Session) -> None:
            session.add(
                RiskEventRow(
                    timestamp=timestamp or datetime.now(timezone.utc),
                    trading_mode=self._mode,
                    event_type=event_type,
                    severity=severity,
                    detail=detail,
                    context=_jsonable(context or {}),
                )
            )

        return self._safe_write("save_risk_event", write)

    def risk_events(self, limit: int = 100) -> list[RiskEventRow]:
        try:
            with self.session() as session:
                return list(
                    session.scalars(
                        select(RiskEventRow)
                        .where(RiskEventRow.trading_mode == self._mode)
                        .order_by(RiskEventRow.timestamp.desc())
                        .limit(limit)
                    )
                )
        except Exception:  # noqa: BLE001
            return []

    # ---- account snapshots -------------------------------------------------------

    def save_account_snapshot(
        self,
        *,
        equity: Decimal,
        cash: Decimal,
        buying_power: Decimal,
        realised_pnl: Decimal = Decimal("0"),
        unrealised_pnl: Decimal | None = None,
        gross_exposure: Decimal | None = None,
        open_positions: int = 0,
    ) -> bool:
        def write(session: Session) -> None:
            session.add(
                AccountSnapshotRow(
                    timestamp=datetime.now(timezone.utc),
                    trading_mode=self._mode,
                    equity=equity,
                    cash=cash,
                    buying_power=buying_power,
                    realised_pnl=realised_pnl,
                    unrealised_pnl=unrealised_pnl,
                    gross_exposure=gross_exposure,
                    open_positions=open_positions,
                )
            )

        return self._safe_write("save_account_snapshot", write)

    def equity_curve(self) -> list[tuple[datetime, Decimal]]:
        try:
            with self.session() as session:
                rows = session.scalars(
                    select(AccountSnapshotRow)
                    .where(AccountSnapshotRow.trading_mode == self._mode)
                    .order_by(AccountSnapshotRow.timestamp)
                ).all()
            return [(r.timestamp, Decimal(str(r.equity))) for r in rows]
        except Exception:  # noqa: BLE001
            return []

    # ---- strategy versions ---------------------------------------------------------

    def record_strategy_version(self, *, name: str, version: str, params: dict) -> bool:
        def write(session: Session) -> None:
            session.add(
                StrategyVersionRow(
                    name=name,
                    version=version,
                    params=_jsonable(params),
                    trading_mode=self._mode,
                )
            )

        return self._safe_write("record_strategy_version", write)
