"""
Order validator: the last gate before an order can be built for the broker.

Why this exists as a separate stage from the risk engine: they answer
different questions.

  RiskEngine:      "Should we take this risk?"       (economic)
  OrderValidator:  "Is this a well-formed order?"    (structural)

Keeping them separate means a bug in economic policy cannot produce a
malformed order, and a malformed order cannot be waved through because
the risk numbers looked fine. It also gives us a place to enforce
invariants that are about *correctness*, not risk appetite — duplicate
detection, tick-size conformance, marketable-limit direction, and the
requirement that an order can only be constructed from an assessment that
actually approved it.

The critical invariant: `build_order()` refuses to construct an `Order`
unless it is handed a `RiskAssessment` with `approved=True`, and it uses
`assessment.approved_quantity`, never the intent's requested quantity.
There is no other constructor path used by the execution engine.
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import structlog

from data.models import MarketSnapshot
from execution.execution_models import (
    Order,
    OrderIntent,
    OrderSide,
    OrderState,
    OrderType,
)
from execution.order_store import DuplicateOrderError, OrderStore
from risk.decisions import RejectionReason, RiskAssessment, RiskDecision

log = structlog.get_logger(__name__)


class OrderValidationError(Exception):
    """Raised when an order cannot be constructed. Never suppressed into
    a default order."""


class OrderValidator:
    def __init__(
        self,
        order_store: OrderStore,
        *,
        tick_size: Decimal = Decimal("0.01"),
        min_quantity: Decimal = Decimal("1"),
    ) -> None:
        self._store = order_store
        self._tick_size = tick_size
        self._min_quantity = min_quantity

    def validate(
        self,
        intent: OrderIntent,
        assessment: RiskAssessment,
        *,
        snapshot: MarketSnapshot | None = None,
        now: datetime | None = None,
    ) -> RiskDecision:
        now = now or datetime.now(timezone.utc)

        if not assessment.approved:
            return RiskDecision.reject(
                "validator",
                RejectionReason.INVALID_ORDER,
                "Risk assessment did not approve this intent",
            )

        quantity = assessment.approved_quantity
        if quantity <= 0:
            return RiskDecision.reject(
                "validator", RejectionReason.ZERO_QUANTITY, "Approved quantity is zero"
            )
        if quantity < self._min_quantity:
            return RiskDecision.reject(
                "validator",
                RejectionReason.INVALID_ORDER,
                f"Quantity {quantity} below minimum {self._min_quantity}",
            )

        try:
            self._store.check_duplicate(intent, now=now)
        except DuplicateOrderError as exc:
            return RiskDecision.reject(
                "validator", RejectionReason.DUPLICATE_ORDER, str(exc)
            )

        tick_check = self._check_tick_conformance(intent)
        if tick_check is not None:
            return tick_check

        direction_check = self._check_stop_direction(intent, snapshot)
        if direction_check is not None:
            return direction_check

        return RiskDecision.approve("validator", f"validated quantity {quantity}")

    def _check_tick_conformance(self, intent: OrderIntent) -> RiskDecision | None:
        """IBKR rejects prices that don't conform to the instrument's tick
        size. Catching it locally avoids a broker round-trip and a
        rejection that would otherwise count toward our rejection-storm
        circuit breaker."""
        for label, price in (("limit_price", intent.limit_price), ("stop_price", intent.stop_price)):
            if price is None:
                continue
            if price % self._tick_size != 0:
                return RiskDecision.reject(
                    "validator",
                    RejectionReason.INVALID_ORDER,
                    f"{label} {price} is not a multiple of tick size {self._tick_size}",
                )
        return None

    def _check_stop_direction(
        self, intent: OrderIntent, snapshot: MarketSnapshot | None
    ) -> RiskDecision | None:
        """A protective stop on the wrong side of the market triggers
        immediately at an unintended price. This is a correctness bug that
        looks like a valid order to the risk engine, which is exactly why
        it belongs here."""
        if intent.stop_price is None or snapshot is None or snapshot.mid is None:
            return None
        mid = Decimal(str(snapshot.mid))
        if intent.side is OrderSide.BUY and intent.stop_price < mid:
            return RiskDecision.reject(
                "validator",
                RejectionReason.INVALID_ORDER,
                f"BUY stop {intent.stop_price} is below market {mid} — would trigger immediately",
            )
        if intent.side is OrderSide.SELL and intent.stop_price > mid:
            return RiskDecision.reject(
                "validator",
                RejectionReason.INVALID_ORDER,
                f"SELL stop {intent.stop_price} is above market {mid} — would trigger immediately",
            )
        return None

    def build_order(self, intent: OrderIntent, assessment: RiskAssessment) -> Order:
        """Construct the Order. The ONLY path from intent to order.

        Refuses unapproved assessments, and always uses the risk-approved
        quantity rather than the requested one — if the sizer trimmed the
        trade, the trimmed size is what goes to the broker."""
        if not assessment.approved:
            raise OrderValidationError(
                "Refusing to build an order from an unapproved risk assessment"
            )
        if assessment.approved_quantity <= 0:
            raise OrderValidationError("Refusing to build an order with non-positive quantity")

        effective_intent = intent.model_copy(
            update={"quantity": assessment.approved_quantity}
        )
        order = Order(intent=effective_intent)
        order.transition_to(OrderState.VALIDATING)
        order.transition_to(OrderState.APPROVED)

        self._store.add(order)
        self._store.record_intent(intent)
        log.info(
            "order.approved",
            order_id=order.order_id,
            instrument=str(intent.instrument),
            side=intent.side,
            quantity=str(assessment.approved_quantity),
            requested_quantity=str(intent.quantity),
            source=intent.source,
        )
        return order
