from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from data.models import Instrument
from execution.execution_models import Order, OrderIntent, OrderSide, OrderState
from execution.order_store import DuplicateOrderError, OrderStore


def _intent(qty: str = "100", source: str = "momentum") -> OrderIntent:
    return OrderIntent(
        instrument=Instrument(symbol="AAPL"),
        side=OrderSide.BUY,
        quantity=Decimal(qty),
        source=source,
        strategy=source,
    )


def test_duplicate_intent_within_window_rejected():
    store = OrderStore(dedupe_window_seconds=60)
    intent = _intent()
    store.check_duplicate(intent)  # first time is fine
    store.record_intent(intent)
    with pytest.raises(DuplicateOrderError):
        store.check_duplicate(_intent())  # equivalent intent, new id


def test_duplicate_allowed_after_window_expires():
    store = OrderStore(dedupe_window_seconds=60)
    now = datetime.now(timezone.utc)
    store.record_intent(_intent(), now=now)
    later = now + timedelta(seconds=61)
    store.check_duplicate(_intent(), now=later)  # should not raise


def test_different_quantity_is_not_duplicate():
    store = OrderStore()
    store.record_intent(_intent("100"))
    store.check_duplicate(_intent("200"))  # different size, allowed


def test_different_source_is_not_duplicate():
    store = OrderStore()
    store.record_intent(_intent(source="momentum"))
    store.check_duplicate(_intent(source="mean_reversion"))


def test_lookup_by_broker_id():
    store = OrderStore()
    order = Order(intent=_intent())
    store.add(order)
    store.link_broker_id(order.order_id, "IB-12345")
    assert store.get_by_broker_id("IB-12345") is order
    assert store.get(order.order_id) is order


def test_active_orders_excludes_terminal():
    store = OrderStore()
    live = Order(intent=_intent("100"))
    live.transition_to(OrderState.VALIDATING)
    live.transition_to(OrderState.APPROVED)
    live.transition_to(OrderState.SUBMITTED)

    dead = Order(intent=_intent("200"))
    dead.transition_to(OrderState.VALIDATING)
    dead.transition_to(OrderState.REJECTED)

    store.add(live)
    store.add(dead)
    assert store.active_orders() == [live]
