"""
Order store and duplicate detection.

Design decisions:
- A single registry owns every Order object, keyed by our internal
  order_id, with a secondary index by broker_order_id. Reconciliation
  needs both directions: "what does the broker think exists that I don't
  know about" and vice versa.
- Duplicate detection is time-windowed on `OrderIntent.dedupe_key()`.
  A repeated identical signal within the window is rejected. This guards
  against the common failure where a strategy re-fires on every tick, or
  the control loop runs twice after a restart, and the account ends up
  with 3x the intended position.
- The store is deliberately in-memory here; Phase 5 adds the PostgreSQL
  repository behind the same interface. Keeping persistence out of this
  class means order-lifecycle logic is testable without a database.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import structlog

from execution.execution_models import Order, OrderIntent, OrderState

log = structlog.get_logger(__name__)


class DuplicateOrderError(Exception):
    """Raised when an intent matches a recently-seen intent. Treated as a
    rejection, not a warning."""


class OrderStore:
    def __init__(self, *, dedupe_window_seconds: float = 60.0) -> None:
        self._orders: dict[str, Order] = {}
        self._by_broker_id: dict[str, str] = {}
        self._dedupe_window = timedelta(seconds=dedupe_window_seconds)
        self._recent_intents: dict[str, datetime] = {}

    def add(self, order: Order) -> None:
        self._orders[order.order_id] = order
        if order.broker_order_id:
            self._by_broker_id[order.broker_order_id] = order.order_id

    def get(self, order_id: str) -> Order | None:
        return self._orders.get(order_id)

    def get_by_broker_id(self, broker_order_id: str) -> Order | None:
        order_id = self._by_broker_id.get(broker_order_id)
        return self._orders.get(order_id) if order_id else None

    def link_broker_id(self, order_id: str, broker_order_id: str) -> None:
        order = self._orders[order_id]
        order.broker_order_id = broker_order_id
        self._by_broker_id[broker_order_id] = order_id

    def all_orders(self) -> list[Order]:
        return list(self._orders.values())

    def active_orders(self) -> list[Order]:
        return [o for o in self._orders.values() if o.is_active]

    def orders_in_state(self, state: OrderState) -> list[Order]:
        return [o for o in self._orders.values() if o.state is state]

    # ---- duplicate detection ------------------------------------------------

    def check_duplicate(self, intent: OrderIntent, *, now: datetime | None = None) -> None:
        """Raise DuplicateOrderError if an equivalent intent was seen
        within the dedupe window."""
        now = now or datetime.now(timezone.utc)
        key = intent.dedupe_key()
        self._prune_recent(now)
        previous = self._recent_intents.get(key)
        if previous is not None:
            log.warning(
                "order.duplicate_rejected",
                dedupe_key=key,
                seconds_since_previous=(now - previous).total_seconds(),
            )
            raise DuplicateOrderError(
                f"Duplicate intent within {self._dedupe_window.total_seconds()}s: {key}"
            )

    def record_intent(self, intent: OrderIntent, *, now: datetime | None = None) -> None:
        self._recent_intents[intent.dedupe_key()] = now or datetime.now(timezone.utc)

    def _prune_recent(self, now: datetime) -> None:
        cutoff = now - self._dedupe_window
        expired = [k for k, ts in self._recent_intents.items() if ts < cutoff]
        for k in expired:
            del self._recent_intents[k]
