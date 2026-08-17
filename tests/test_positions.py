from datetime import datetime, timezone
from decimal import Decimal

import pytest

from data.models import Instrument
from execution.execution_models import Fill, OrderSide
from portfolio.positions import Position


def _fill(qty: str, price: str, commission: str = "0") -> Fill:
    return Fill(
        fill_id="f",
        order_id="o",
        timestamp=datetime.now(timezone.utc),
        quantity=Decimal(qty),
        price=Decimal(price),
        commission=Decimal(commission),
    )


@pytest.fixture
def position():
    return Position(instrument=Instrument(symbol="AAPL"))


def test_open_long(position):
    position.apply_fill(_fill("100", "50"), OrderSide.BUY)
    assert position.quantity == Decimal("100")
    assert position.average_cost == Decimal("50")
    assert position.is_long


def test_open_short(position):
    position.apply_fill(_fill("100", "50"), OrderSide.SELL)
    assert position.quantity == Decimal("-100")
    assert position.is_short
    assert position.exposure(Decimal("50")) == Decimal("5000")


def test_add_to_long_blends_cost_basis(position):
    position.apply_fill(_fill("100", "50"), OrderSide.BUY)
    position.apply_fill(_fill("100", "60"), OrderSide.BUY)
    assert position.quantity == Decimal("200")
    assert position.average_cost == Decimal("55")


def test_partial_close_books_realized_pnl(position):
    position.apply_fill(_fill("100", "50"), OrderSide.BUY)
    position.apply_fill(_fill("40", "55"), OrderSide.SELL)
    assert position.quantity == Decimal("60")
    assert position.average_cost == Decimal("50")  # unchanged on reduction
    assert position.realized_pnl == Decimal("200")  # 40 * (55-50)


def test_full_close_flattens(position):
    position.apply_fill(_fill("100", "50"), OrderSide.BUY)
    position.apply_fill(_fill("100", "45"), OrderSide.SELL)
    assert position.is_flat
    assert position.average_cost == Decimal("0")
    assert position.realized_pnl == Decimal("-500")


def test_flip_long_to_short(position):
    position.apply_fill(_fill("100", "50"), OrderSide.BUY)
    position.apply_fill(_fill("150", "55"), OrderSide.SELL)
    # 100 closed at +5 each = +500 realized; 50 short opened at 55.
    assert position.quantity == Decimal("-50")
    assert position.average_cost == Decimal("55")
    assert position.realized_pnl == Decimal("500")


def test_short_profit_realized_correctly(position):
    position.apply_fill(_fill("100", "50"), OrderSide.SELL)
    position.apply_fill(_fill("100", "45"), OrderSide.BUY)
    # Short from 50, covered at 45 -> +500
    assert position.realized_pnl == Decimal("500")
    assert position.is_flat


def test_unrealized_pnl_long_and_short(position):
    position.apply_fill(_fill("100", "50"), OrderSide.BUY)
    assert position.unrealized_pnl(Decimal("55")) == Decimal("500")
    assert position.unrealized_pnl(Decimal("45")) == Decimal("-500")


def test_unrealized_pnl_short(position):
    position.apply_fill(_fill("100", "50"), OrderSide.SELL)
    # Short 100 @ 50, price falls to 45 -> profit
    assert position.unrealized_pnl(Decimal("45")) == Decimal("500")


def test_commission_accumulates(position):
    position.apply_fill(_fill("100", "50", "1.25"), OrderSide.BUY)
    position.apply_fill(_fill("100", "50", "1.25"), OrderSide.SELL)
    assert position.total_commission == Decimal("2.50")


def test_flat_position_has_no_unrealized_pnl(position):
    assert position.unrealized_pnl(Decimal("999")) == Decimal("0")
