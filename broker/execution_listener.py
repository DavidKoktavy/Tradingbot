"""
Execution listener: broker events -> confirmed fills -> portfolio updates.

Design decisions:

- **Fills are idempotent by execution id.** IBKR can and does re-deliver
  execution reports (on reconnect, on `reqExecutions`, on duplicate
  callbacks). Applying the same fill twice would double a position and
  silently corrupt every downstream risk calculation. A seen-set keyed on
  the broker's execution id is the guard.

- **The listener never invents state.** It only acts on events the broker
  sent. If an order goes quiet, that is reconciliation's problem, not
  something to paper over with an assumed fill.

- **Order rejections are counted.** A burst of rejections usually means
  something systemic is wrong (bad contract, no permissions, margin
  problem, pacing violation), and continuing to fire orders into it makes
  things worse. Crossing the threshold trips the kill switch.

- Callbacks are wrapped so an exception in our handler cannot kill the
  broker event loop; failures are logged and the listener keeps running,
  because a dead listener means fills stop being recorded while orders
  keep executing — the worst possible state.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from decimal import Decimal
from typing import Any

import structlog

from execution.execution_models import Fill, Order, OrderState
from execution.order_store import OrderStore
from portfolio.portfolio_manager import PortfolioManager
from risk.kill_switch import KillSwitch, KillSwitchTrigger

log = structlog.get_logger(__name__)


class ExecutionListener:
    def __init__(
        self,
        store: OrderStore,
        portfolio: PortfolioManager,
        *,
        kill_switch: KillSwitch | None = None,
        rejection_threshold: int = 5,
        on_fill: Callable[[Order, Fill], Awaitable[None]] | None = None,
    ) -> None:
        self._store = store
        self._portfolio = portfolio
        self._kill_switch = kill_switch
        self._rejection_threshold = rejection_threshold
        self._on_fill = on_fill
        self._seen_execution_ids: set[str] = set()
        self._rejection_count = 0
        self._fill_count = 0

    @property
    def rejection_count(self) -> int:
        return self._rejection_count

    @property
    def fill_count(self) -> int:
        return self._fill_count

    def reset_rejection_count(self) -> None:
        self._rejection_count = 0

    async def handle_fill(self, fill: Fill) -> Order | None:
        """Apply a broker-confirmed fill. Idempotent by fill_id."""
        if fill.fill_id in self._seen_execution_ids:
            log.debug("execution.duplicate_fill_ignored", fill_id=fill.fill_id)
            return None
        self._seen_execution_ids.add(fill.fill_id)

        order = self._store.get(fill.order_id)
        if order is None:
            # A fill for an order we don't know about. Do NOT guess: flag
            # it and let reconciliation resolve the discrepancy.
            log.error(
                "execution.fill_for_unknown_order",
                fill_id=fill.fill_id,
                order_id=fill.order_id,
            )
            if self._kill_switch is not None:
                self._kill_switch.activate(
                    KillSwitchTrigger.RECONCILIATION_FAILURE,
                    f"Fill {fill.fill_id} for unknown order {fill.order_id}",
                )
            return None

        # The order object may already have the fill applied if it came
        # from the simulated gateway, which applies fills directly.
        if not any(f.fill_id == fill.fill_id for f in order.fills):
            try:
                order.apply_fill(fill)
            except Exception as exc:  # noqa: BLE001
                log.error(
                    "execution.fill_application_failed",
                    order_id=order.order_id,
                    error=str(exc),
                )
                return None

        self._portfolio.apply_fill(order, fill)
        self._fill_count += 1
        log.info(
            "execution.filled",
            order_id=order.order_id,
            fill_id=fill.fill_id,
            quantity=str(fill.quantity),
            price=str(fill.price),
            state=order.state,
            remaining=str(order.remaining_quantity),
        )

        if self._on_fill is not None:
            try:
                await self._on_fill(order, fill)
            except Exception as exc:  # noqa: BLE001
                log.error("execution.on_fill_hook_failed", error=str(exc))

        return order

    async def handle_status(
        self, broker_order_id: str, state: OrderState, *, message: str = ""
    ) -> Order | None:
        """Apply a broker status update."""
        order = self._store.get_by_broker_id(broker_order_id)
        if order is None:
            log.warning("execution.status_for_unknown_order", broker_order_id=broker_order_id)
            return None

        if state is OrderState.REJECTED:
            return await self._handle_rejection(order, message)

        if order.state is state or order.is_terminal:
            return order

        from execution.execution_models import can_transition

        if not can_transition(order.state, state):
            log.error(
                "execution.illegal_status_transition",
                order_id=order.order_id,
                current=order.state,
                received=state,
            )
            return order

        order.transition_to(state)
        log.info("execution.status", order_id=order.order_id, state=state)
        return order

    async def _handle_rejection(self, order: Order, message: str) -> Order:
        if not order.is_terminal:
            order.transition_to(OrderState.REJECTED, error_message=message)
        self._rejection_count += 1
        log.error(
            "execution.rejected",
            order_id=order.order_id,
            message=message,
            rejection_count=self._rejection_count,
        )

        if (
            self._kill_switch is not None
            and self._rejection_count >= self._rejection_threshold
        ):
            # A burst of rejections means something systemic is wrong.
            # Continuing to fire orders into it makes things worse.
            self._kill_switch.activate(
                KillSwitchTrigger.ORDER_REJECTION_STORM,
                f"{self._rejection_count} order rejections; latest: {message}",
            )
        return order

    def ib_fill_to_normalised(self, trade: Any, ib_fill: Any) -> Fill | None:
        """Translate an ib_async Fill into our normalised model.

        Returns None if the execution cannot be matched to a local order,
        rather than fabricating an order id.
        """
        execution = getattr(ib_fill, "execution", None)
        if execution is None:
            return None
        order_ref = getattr(getattr(trade, "order", None), "orderRef", "") or ""
        if not order_ref:
            return None

        commission = Decimal("0")
        report = getattr(ib_fill, "commissionReport", None)
        if report is not None and getattr(report, "commission", None):
            commission = Decimal(str(report.commission))

        return Fill(
            fill_id=str(execution.execId),
            order_id=order_ref,
            timestamp=execution.time,
            quantity=Decimal(str(execution.shares)),
            price=Decimal(str(execution.price)),
            commission=commission,
        )
