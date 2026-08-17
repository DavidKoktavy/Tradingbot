from datetime import datetime, timezone
from decimal import Decimal

import pytest

from data.models import Instrument
from execution.execution_models import Fill, Order, OrderIntent, OrderSide
from portfolio.portfolio_manager import AccountState, MissingPriceError, PortfolioManager


def _order(symbol: str, side: OrderSide, qty: str) -> Order:
    return Order(
        intent=OrderIntent(
            instrument=Instrument(symbol=symbol),
            side=side,
            quantity=Decimal(qty),
            source="test",
        )
    )


def _fill(order: Order, qty: str, price: str, commission: str = "0") -> Fill:
    return Fill(
        fill_id="f",
        order_id=order.order_id,
        timestamp=datetime.now(timezone.utc),
        quantity=Decimal(qty),
        price=Decimal(price),
        commission=Decimal(commission),
    )


@pytest.fixture
def portfolio():
    pm = PortfolioManager()
    pm.update_account(
        AccountState(equity=Decimal("100000"), cash=Decimal("100000"), buying_power=Decimal("200000"))
    )
    return pm


def test_apply_fill_creates_position(portfolio):
    order = _order("AAPL", OrderSide.BUY, "100")
    portfolio.apply_fill(order, _fill(order, "100", "50"))
    pos = portfolio.get_position(Instrument(symbol="AAPL"))
    assert pos.quantity == Decimal("100")
    assert portfolio.open_position_count == 1


def test_gross_exposure_counts_shorts(portfolio):
    long_order = _order("AAPL", OrderSide.BUY, "100")
    short_order = _order("MSFT", OrderSide.SELL, "50")
    portfolio.apply_fill(long_order, _fill(long_order, "100", "50"))
    portfolio.apply_fill(short_order, _fill(short_order, "50", "100"))

    prices = {"AAPL:SMART:USD": Decimal("50"), "MSFT:SMART:USD": Decimal("100")}
    # Gross: 100*50 + 50*100 = 10,000 (absolute)
    assert portfolio.gross_exposure(prices) == Decimal("10000")
    # Net: 5000 long - 5000 short = 0
    assert portfolio.net_exposure(prices) == Decimal("0")


def test_missing_price_raises_rather_than_guessing(portfolio):
    order = _order("AAPL", OrderSide.BUY, "100")
    portfolio.apply_fill(order, _fill(order, "100", "50"))
    with pytest.raises(MissingPriceError):
        portfolio.gross_exposure({})  # no mark for the open position


def test_daily_pnl_requires_start_of_day_equity():
    pm = PortfolioManager()  # never had account update
    with pytest.raises(MissingPriceError):
        pm.daily_pnl({})


def test_start_of_day_equity_set_on_first_account_update(portfolio):
    assert portfolio.start_of_day_equity == Decimal("100000")


def test_start_of_day_equity_not_reset_by_later_updates(portfolio):
    # A mid-session equity refresh must NOT silently reset the daily loss
    # baseline — that would let the system lose its daily budget twice.
    portfolio.update_account(AccountState(equity=Decimal("95000")))
    assert portfolio.start_of_day_equity == Decimal("100000")


def test_daily_pnl_reflects_drawdown(portfolio):
    portfolio.update_account(AccountState(equity=Decimal("98000")))
    assert portfolio.daily_pnl({}) == Decimal("-2000")
    assert portfolio.daily_pnl_pct({}) == Decimal("-0.02")


def test_explicit_new_session_rolls_baseline(portfolio):
    portfolio.update_account(AccountState(equity=Decimal("98000")))
    portfolio.start_new_session(Decimal("98000"))
    assert portfolio.start_of_day_equity == Decimal("98000")
    assert portfolio.daily_pnl({}) == Decimal("0")


def test_unrealized_pnl_included_in_daily_pnl(portfolio):
    order = _order("AAPL", OrderSide.BUY, "100")
    portfolio.apply_fill(order, _fill(order, "100", "50"))
    # equity unchanged at 100000, but position is up 10/share
    prices = {"AAPL:SMART:USD": Decimal("60")}
    assert portfolio.unrealized_pnl(prices) == Decimal("1000")
    assert portfolio.daily_pnl(prices) == Decimal("1000")


def test_flat_positions_excluded_from_open_count(portfolio):
    buy = _order("AAPL", OrderSide.BUY, "100")
    sell = _order("AAPL", OrderSide.SELL, "100")
    portfolio.apply_fill(buy, _fill(buy, "100", "50"))
    portfolio.apply_fill(sell, _fill(sell, "100", "55"))
    assert portfolio.open_position_count == 0
    assert portfolio.realized_pnl == Decimal("500")
