from decimal import Decimal

import pytest

from risk.position_sizer import PositionSizer


@pytest.fixture
def sizer():
    return PositionSizer(
        max_risk_per_trade=Decimal("0.01"),  # 1% of equity at risk
        max_position_size=Decimal("0.10"),  # 10% of equity notional
    )


@pytest.fixture
def uncapped_sizer():
    """Generous notional cap, so tests can observe pure risk-based sizing
    without the position-size cap binding first."""
    return PositionSizer(
        max_risk_per_trade=Decimal("0.01"),
        max_position_size=Decimal("1.00"),
    )


def test_risk_based_sizing(uncapped_sizer):
    # Equity 100k, 1% risk = $1000. Stop distance $5 -> 200 shares.
    result = uncapped_sizer.calculate(
        equity=Decimal("100000"), entry_price=Decimal("100"), stop_price=Decimal("95")
    )
    assert result.method == "risk_based"
    assert result.quantity == Decimal("200")
    assert result.risk_amount == Decimal("1000")


def test_wider_stop_gives_smaller_position(uncapped_sizer):
    tight = uncapped_sizer.calculate(
        equity=Decimal("100000"), entry_price=Decimal("100"), stop_price=Decimal("98")
    )
    wide = uncapped_sizer.calculate(
        equity=Decimal("100000"), entry_price=Decimal("100"), stop_price=Decimal("90")
    )
    assert wide.quantity < tight.quantity


def test_notional_cap_applies(sizer):
    # Very tight stop would imply a huge position; the 10% notional cap
    # must bind. 10% of 100k = 10k / $100 = 100 shares.
    result = sizer.calculate(
        equity=Decimal("100000"), entry_price=Decimal("100"), stop_price=Decimal("99.9")
    )
    assert result.quantity == Decimal("100")
    assert "capped by max_position_size" in result.detail


def test_volatility_sizing_when_no_stop(sizer):
    result = sizer.calculate(
        equity=Decimal("100000"), entry_price=Decimal("100"), atr=Decimal("2")
    )
    assert result.method == "volatility_adjusted"
    # stop distance = ATR 2 * multiple 2 = 4; 1000 / 4 = 250, capped at 100
    assert result.quantity == Decimal("100")


def test_refuses_to_size_unknown_risk(sizer):
    # No stop, no ATR -> we have no idea what we're risking. Return zero
    # rather than guessing.
    result = sizer.calculate(equity=Decimal("100000"), entry_price=Decimal("100"))
    assert result.quantity == Decimal("0")
    assert not result.is_tradeable
    assert "unknown risk" in result.detail


def test_never_exceeds_requested_quantity(sizer):
    result = sizer.calculate(
        equity=Decimal("100000"),
        entry_price=Decimal("100"),
        stop_price=Decimal("95"),
        requested_quantity=Decimal("50"),
    )
    assert result.quantity == Decimal("50")


def test_ai_cannot_inflate_size_by_requesting_more(uncapped_sizer):
    """A strategy or AI asking for 10,000 shares gets the risk-permitted
    size, not what it asked for."""
    result = uncapped_sizer.calculate(
        equity=Decimal("100000"),
        entry_price=Decimal("100"),
        stop_price=Decimal("95"),
        requested_quantity=Decimal("10000"),
    )
    assert result.quantity == Decimal("200")


def test_zero_equity_returns_zero(sizer):
    result = sizer.calculate(
        equity=Decimal("0"), entry_price=Decimal("100"), stop_price=Decimal("95")
    )
    assert result.quantity == Decimal("0")


def test_fractional_truncates_down():
    s = PositionSizer(
        max_risk_per_trade=Decimal("0.01"),
        max_position_size=Decimal("1.0"),
    )
    # 1000 risk / 3 distance = 333.33 -> 333 whole shares
    result = s.calculate(
        equity=Decimal("100000"), entry_price=Decimal("100"), stop_price=Decimal("97")
    )
    assert result.quantity == Decimal("333")


def test_zero_stop_distance_returns_zero(sizer):
    result = sizer.calculate(
        equity=Decimal("100000"), entry_price=Decimal("100"), stop_price=Decimal("100")
    )
    assert result.quantity == Decimal("0")


def test_notional_cap_binds_before_risk_budget(sizer):
    """Documents the interaction: with a 1% risk budget and a 10% notional
    cap, the tighter of the two always wins. Here risk-based sizing wants
    200 shares but the notional cap permits only 100."""
    result = sizer.calculate(
        equity=Decimal("100000"), entry_price=Decimal("100"), stop_price=Decimal("95")
    )
    assert result.quantity == Decimal("100")
    assert "capped by max_position_size" in result.detail
