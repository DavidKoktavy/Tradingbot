"""
State reconciliation between local memory and the broker.

This is one of the most safety-critical modules in the system. The
governing principle from the spec: *never assume local state equals broker
state*, and *always reconcile after reconnect/restart*.

Design decisions:

- Reconciliation produces a `ReconciliationReport` describing every
  discrepancy, and the caller decides what to do. It never silently
  "fixes" anything by placing compensating trades — an automatic
  corrective trade based on a misunderstanding of state is precisely how
  an automated system turns a small bug into a large loss.

- The broker is treated as the authority for positions and for order
  existence. Local state is overwritten to match. The one thing we do NOT
  overwrite is our audit trail: the discrepancy is logged permanently.

- `requires_halt` is True whenever we find anything we cannot explain —
  an unknown broker position, an unknown broker order, or a local order
  the broker has never heard of. The control loop must not open new
  positions while this is True. Fail closed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal

import structlog

from data.models import Instrument
from execution.execution_models import Order, OrderState
from execution.order_store import OrderStore
from portfolio.portfolio_manager import PortfolioManager
from portfolio.positions import Position

log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class BrokerPosition:
    """A position as reported by the broker."""

    instrument: Instrument
    quantity: Decimal
    average_cost: Decimal


@dataclass(frozen=True)
class BrokerOrder:
    """An open order as reported by the broker."""

    broker_order_id: str
    instrument: Instrument
    quantity: Decimal
    filled_quantity: Decimal
    state: OrderState


@dataclass
class Discrepancy:
    kind: str
    detail: str
    instrument: str | None = None
    broker_order_id: str | None = None


@dataclass
class ReconciliationReport:
    discrepancies: list[Discrepancy] = field(default_factory=list)
    positions_corrected: int = 0
    orders_adopted: int = 0
    orders_marked_unknown: int = 0

    @property
    def is_clean(self) -> bool:
        return not self.discrepancies

    @property
    def requires_halt(self) -> bool:
        """Any unexplained state means we stop opening new positions until
        a human or a higher-level policy resolves it."""
        halting = {"UNKNOWN_BROKER_POSITION", "UNKNOWN_BROKER_ORDER", "MISSING_AT_BROKER",
                   "POSITION_MISMATCH"}
        return any(d.kind in halting for d in self.discrepancies)


class Reconciler:
    def __init__(self, order_store: OrderStore, portfolio: PortfolioManager) -> None:
        self._orders = order_store
        self._portfolio = portfolio

    def reconcile(
        self,
        *,
        broker_positions: list[BrokerPosition],
        broker_orders: list[BrokerOrder],
    ) -> ReconciliationReport:
        report = ReconciliationReport()
        self._reconcile_positions(broker_positions, report)
        self._reconcile_orders(broker_orders, report)

        log.info(
            "reconcile.complete",
            discrepancies=len(report.discrepancies),
            positions_corrected=report.positions_corrected,
            orders_adopted=report.orders_adopted,
            requires_halt=report.requires_halt,
        )
        for d in report.discrepancies:
            log.warning("reconcile.discrepancy", kind=d.kind, detail=d.detail)
        return report

    def _reconcile_positions(
        self, broker_positions: list[BrokerPosition], report: ReconciliationReport
    ) -> None:
        broker_by_key = {str(bp.instrument): bp for bp in broker_positions}

        for key, bp in broker_by_key.items():
            local = self._portfolio.positions.get(key)
            if local is None or local.is_flat:
                if bp.quantity != 0:
                    # The broker holds something we have no record of.
                    # Adopt it so risk calculations include it, but flag
                    # loudly: we do not know why it exists.
                    report.discrepancies.append(
                        Discrepancy(
                            kind="UNKNOWN_BROKER_POSITION",
                            detail=f"Broker reports {bp.quantity} of {key}, local state has none",
                            instrument=key,
                        )
                    )
                    position = self._portfolio.get_position(bp.instrument)
                    position.quantity = bp.quantity
                    position.average_cost = bp.average_cost
                    report.positions_corrected += 1
            elif local.quantity != bp.quantity:
                report.discrepancies.append(
                    Discrepancy(
                        kind="POSITION_MISMATCH",
                        detail=(
                            f"{key}: local {local.quantity} != broker {bp.quantity}; "
                            "adopting broker value"
                        ),
                        instrument=key,
                    )
                )
                local.quantity = bp.quantity
                local.average_cost = bp.average_cost
                report.positions_corrected += 1

        # Local thinks we hold something the broker doesn't report at all.
        for key, local in self._portfolio.positions.items():
            if local.is_flat or key in broker_by_key:
                continue
            report.discrepancies.append(
                Discrepancy(
                    kind="POSITION_MISMATCH",
                    detail=f"{key}: local {local.quantity} but broker reports no position; flattening local",
                    instrument=key,
                )
            )
            local.quantity = Decimal("0")
            local.average_cost = Decimal("0")
            report.positions_corrected += 1

    def _reconcile_orders(
        self, broker_orders: list[BrokerOrder], report: ReconciliationReport
    ) -> None:
        broker_by_id = {bo.broker_order_id: bo for bo in broker_orders}

        for broker_id, bo in broker_by_id.items():
            local = self._orders.get_by_broker_id(broker_id)
            if local is None:
                # An order live at the broker that we don't know about —
                # e.g. we crashed between submission and persisting the id.
                report.discrepancies.append(
                    Discrepancy(
                        kind="UNKNOWN_BROKER_ORDER",
                        detail=(
                            f"Broker order {broker_id} ({bo.quantity} {bo.instrument}) "
                            "has no local record"
                        ),
                        broker_order_id=broker_id,
                    )
                )
                report.orders_adopted += 1
                continue

            if local.state is not bo.state and not local.is_terminal:
                report.discrepancies.append(
                    Discrepancy(
                        kind="ORDER_STATE_MISMATCH",
                        detail=f"Order {local.order_id}: local {local.state} != broker {bo.state}",
                        broker_order_id=broker_id,
                    )
                )
                self._adopt_broker_state(local, bo)

        # Orders we believe are live that the broker has never heard of.
        for order in self._orders.active_orders():
            if order.broker_order_id and order.broker_order_id not in broker_by_id:
                report.discrepancies.append(
                    Discrepancy(
                        kind="MISSING_AT_BROKER",
                        detail=(
                            f"Order {order.order_id} believed active but not open at broker — "
                            "may have filled or been cancelled while disconnected"
                        ),
                        broker_order_id=order.broker_order_id,
                    )
                )
                report.orders_marked_unknown += 1

    @staticmethod
    def _adopt_broker_state(local: Order, bo: BrokerOrder) -> None:
        """Move local order state toward the broker's view where the
        transition is legal. If it isn't legal we leave it alone and let
        the discrepancy stand — forcing an illegal transition would
        corrupt the audit trail."""
        from execution.execution_models import can_transition

        if can_transition(local.state, bo.state):
            local.transition_to(bo.state)
        else:
            log.error(
                "reconcile.illegal_adoption_skipped",
                order_id=local.order_id,
                local_state=local.state,
                broker_state=bo.state,
            )
