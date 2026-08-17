"""
Trade journal: the write path from the control loop to durable storage.

Design decisions:

- **The journal is the only component that writes to the database from the
  trading path**, and every method is failure-contained. A database outage
  degrades observability, never correctness: reconciliation reads from the
  broker, so position tracking is unaffected by a dead database.

- **Writes are queued and flushed, not inline.** A synchronous database
  round-trip inside the order-submission path would put database latency
  between a signal and a fill. The queue is bounded: if the database is
  down long enough to fill it, the oldest records are dropped and the drop
  is counted, because unbounded buffering during an outage is how a
  trading process runs out of memory.

- **The audit record is built once and written to both sinks** (JSONL file
  and database). The file sink is deliberately the more durable of the
  two: it is append-only, has no schema, and survives a database that will
  not start.
"""

from __future__ import annotations

from collections import deque
from decimal import Decimal

import structlog

from data.models import MarketSnapshot
from execution.execution_models import Fill, Order
from monitoring.audit import (
    DecisionRecord,
    DecisionRecorder,
    FillRecord,
    RiskCheckRecord,
    SignalRecord,
    compute_slippage_bps,
)
from portfolio.portfolio_manager import PortfolioManager
from risk.decisions import RiskAssessment
from strategies.base import Signal

log = structlog.get_logger(__name__)


class TradeJournal:
    def __init__(
        self,
        recorder: DecisionRecorder,
        *,
        repository: object | None = None,
        portfolio: PortfolioManager | None = None,
        trading_mode: str = "PAPER",
        max_queue: int = 5000,
    ) -> None:
        self._recorder = recorder
        self._repo = repository
        self._portfolio = portfolio
        self._mode = trading_mode
        self._queue: deque[tuple[str, object]] = deque(maxlen=max_queue)
        self._dropped = 0
        self._max_queue = max_queue

    @property
    def dropped_writes(self) -> int:
        return self._dropped

    @property
    def pending(self) -> int:
        return len(self._queue)

    # ---- building records ---------------------------------------------------

    def build_record(
        self,
        *,
        instrument: str,
        cycle: int,
        snapshot: MarketSnapshot | None = None,
        regime: object | None = None,
        signals: list[Signal] | None = None,
        ai_result: object | None = None,
        intent: object | None = None,
        assessment: RiskAssessment | None = None,
        order: Order | None = None,
        submission_error: str | None = None,
        outcome: str = "NO_ACTION",
    ) -> DecisionRecord:
        """Assemble the full decision record. Every field is optional so a
        record can be written at any point the decision stopped, which is
        what makes rejections auditable."""
        fields: dict = {
            "instrument": instrument,
            "cycle": cycle,
            "trading_mode": self._mode,
            "outcome": outcome,
        }

        if snapshot is not None:
            fields.update(
                bid=str(snapshot.bid) if snapshot.bid is not None else None,
                ask=str(snapshot.ask) if snapshot.ask is not None else None,
                last=str(snapshot.last) if snapshot.last is not None else None,
                mid=str(snapshot.mid) if snapshot.mid is not None else None,
                data_age_seconds=round(snapshot.age_seconds(), 3),
            )

        if regime is not None:
            fields.update(
                regime=str(getattr(regime, "regime", "")),
                regime_confidence=getattr(regime, "confidence", None),
                regime_features={
                    k: float(v) for k, v in getattr(regime, "features", {}).items()
                },
            )

        if signals:
            fields["signals"] = [
                SignalRecord(
                    strategy=s.strategy,
                    direction=str(s.direction),
                    strength=s.strength,
                    rationale=s.rationale,
                    features={k: float(v) for k, v in s.features.items()},
                )
                for s in signals
            ]

        if ai_result is not None:
            decision = getattr(ai_result, "decision", None)
            fields.update(
                ai_consulted=True,
                ai_accepted=bool(getattr(ai_result, "accepted", False)),
                ai_rejection_reason=(
                    str(ai_result.reason) if getattr(ai_result, "reason", None) else None
                ),
                ai_raw_response=getattr(ai_result, "raw_response", "")[:4000],
                ai_latency_ms=getattr(ai_result, "latency_ms", None),
            )
            if decision is not None:
                fields.update(
                    ai_action=str(decision.action),
                    ai_confidence=decision.confidence,
                    ai_reasoning=decision.reasoning,
                )

        if intent is not None:
            fields.update(
                intent_id=intent.intent_id,
                intent_source=intent.source,
                intent_strategy=intent.strategy,
                intent_side=str(intent.side),
                requested_quantity=str(intent.quantity),
                stop_loss=str(intent.stop_loss) if intent.stop_loss else None,
                take_profit=str(intent.take_profit) if intent.take_profit else None,
            )

        if assessment is not None:
            fields.update(
                risk_checks=[
                    RiskCheckRecord(
                        check_name=d.check_name,
                        approved=d.approved,
                        reason=str(d.reason) if d.reason else None,
                        detail=d.detail,
                    )
                    for d in assessment.decisions
                ],
                risk_approved=assessment.approved,
                risk_rejection_reason=(
                    str(assessment.reason) if assessment.reason else None
                ),
                approved_quantity=str(assessment.approved_quantity),
                was_reduced=assessment.was_reduced,
            )

        if order is not None:
            fields.update(
                order_id=order.order_id,
                broker_order_id=order.broker_order_id,
                submitted=order.broker_order_id is not None,
                final_order_state=str(order.state),
                average_fill_price=(
                    str(order.average_fill_price) if order.average_fill_price else None
                ),
                fills=[
                    FillRecord(
                        fill_id=f.fill_id,
                        quantity=str(f.quantity),
                        price=str(f.price),
                        commission=str(f.commission),
                        timestamp=f.timestamp.isoformat(),
                    )
                    for f in order.fills
                ],
            )
            if order.fills and snapshot is not None and snapshot.mid is not None:
                fields["slippage_bps"] = compute_slippage_bps(
                    expected_price=Decimal(str(snapshot.mid)),
                    fill_price=order.fills[-1].price,
                    side=str(order.intent.side),
                )

        if submission_error:
            fields["submission_error"] = submission_error

        if self._portfolio is not None and intent is not None:
            position = self._portfolio.get_position(intent.instrument)
            fields.update(
                position_after=str(position.quantity),
                realised_pnl_after=str(self._portfolio.realized_pnl),
                equity_after=str(self._portfolio.account.equity),
            )

        return DecisionRecord(**fields)

    # ---- writing -------------------------------------------------------------

    def record_decision(self, record: DecisionRecord) -> None:
        """Write to the file sink immediately, queue the database write."""
        self._recorder.record(record)
        self._enqueue("decision", record)

    def record_order(self, order: Order) -> None:
        self._enqueue("order", order)

    def record_fill(self, fill: Fill, *, slippage_bps: float | None = None) -> None:
        self._enqueue("fill", (fill, slippage_bps))

    def record_risk_event(
        self, *, event_type: str, severity: str, detail: str, context: dict | None = None
    ) -> None:
        self._enqueue("risk_event", (event_type, severity, detail, context or {}))

    def _enqueue(self, kind: str, payload: object) -> None:
        if self._repo is None:
            return
        if len(self._queue) >= self._max_queue:
            # Bounded: unbounded buffering during an outage is how a
            # trading process runs out of memory.
            self._dropped += 1
            if self._dropped % 100 == 1:
                log.error("journal.queue_full_dropping", dropped=self._dropped)
        self._queue.append((kind, payload))

    def flush(self) -> int:
        """Drain the queue to the database. Never raises; failures are
        counted by the repository and surfaced by health checks."""
        if self._repo is None:
            return 0
        written = 0
        while self._queue:
            kind, payload = self._queue.popleft()
            try:
                if kind == "decision":
                    self._repo.save_decision(payload)
                elif kind == "order":
                    self._repo.save_order(payload)
                elif kind == "fill":
                    fill, slippage = payload
                    self._repo.save_fill(fill, slippage_bps=slippage)
                elif kind == "risk_event":
                    event_type, severity, detail, context = payload
                    self._repo.save_risk_event(
                        event_type=event_type,
                        severity=severity,
                        detail=detail,
                        context=context,
                    )
                written += 1
            except Exception as exc:  # noqa: BLE001
                log.error("journal.flush_failed", kind=kind, error=str(exc))
        return written

    def snapshot_account(self, prices: dict[str, Decimal] | None = None) -> None:
        """Persist an equity point for later curve reconstruction."""
        if self._repo is None or self._portfolio is None:
            return
        account = self._portfolio.account
        unrealised = None
        gross = None
        if prices:
            try:
                unrealised = self._portfolio.unrealized_pnl(prices)
                gross = self._portfolio.gross_exposure(prices)
            except Exception:  # noqa: BLE001 — missing marks are expected
                pass
        try:
            self._repo.save_account_snapshot(
                equity=account.equity,
                cash=account.cash,
                buying_power=account.buying_power,
                realised_pnl=self._portfolio.realized_pnl,
                unrealised_pnl=unrealised,
                gross_exposure=gross,
                open_positions=self._portfolio.open_position_count,
            )
        except Exception as exc:  # noqa: BLE001
            log.error("journal.snapshot_failed", error=str(exc))

    def persist_positions(self) -> None:
        if self._repo is None or self._portfolio is None:
            return
        for position in self._portfolio.positions.values():
            try:
                self._repo.save_position(position)
            except Exception as exc:  # noqa: BLE001
                log.error("journal.position_write_failed", error=str(exc))
