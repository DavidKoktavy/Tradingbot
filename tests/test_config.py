"""Tests for app.config — in particular the LIVE-trading guard, since a
bug here is a financial-risk bug, not just a correctness bug."""

import pytest
from pydantic import ValidationError

from app.config import RiskLimits, Settings, TradingMode


def test_default_mode_is_paper():
    s = Settings()
    assert s.trading_mode is TradingMode.PAPER
    assert s.is_live is False


def test_live_mode_rejected_without_enable_flag():
    with pytest.raises(ValidationError, match="ENABLE_LIVE_TRADING"):
        Settings(trading_mode=TradingMode.LIVE, enable_live_trading=False)


def test_live_mode_rejected_without_confirmation_phrase():
    with pytest.raises(ValidationError, match="LIVE_TRADING_CONFIRMATION"):
        Settings(
            trading_mode=TradingMode.LIVE,
            enable_live_trading=True,
            live_trading_confirmation="yes please",
        )


def test_live_mode_rejected_with_wrong_confirmation_phrase():
    with pytest.raises(ValidationError):
        Settings(
            trading_mode=TradingMode.LIVE,
            enable_live_trading=True,
            live_trading_confirmation="I_UNDERSTAND",  # truncated / wrong
        )


def test_live_mode_accepted_with_both_flags_correct():
    s = Settings(
        trading_mode=TradingMode.LIVE,
        enable_live_trading=True,
        live_trading_confirmation="I_UNDERSTAND_THIS_ENABLES_REAL_ORDERS",
    )
    assert s.is_live is True


def test_paper_mode_ignores_missing_confirmation():
    # PAPER (and BACKTEST/SIMULATION) must never require the live phrase.
    s = Settings(trading_mode=TradingMode.PAPER)
    assert s.is_live is False


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_risk_per_trade", 0),
        ("max_risk_per_trade", 1.5),
        ("max_daily_loss", -0.01),
        ("max_open_positions", 0),
    ],
)
def test_risk_limits_reject_out_of_range_values(field, value):
    with pytest.raises(ValidationError):
        RiskLimits(**{field: value})


def test_risk_limits_defaults_are_sane():
    r = RiskLimits()
    assert 0 < r.max_risk_per_trade <= r.max_position_size
    assert r.max_daily_loss <= r.max_portfolio_drawdown
