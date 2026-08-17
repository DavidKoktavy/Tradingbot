from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from data.models import Bar, Instrument, MarketSnapshot


def test_bar_rejects_inconsistent_ohlc():
    with pytest.raises(ValidationError):
        Bar(timestamp=datetime.now(timezone.utc), open=10, high=9, low=8, close=9.5)


def test_bar_accepts_valid_ohlc():
    b = Bar(timestamp=datetime.now(timezone.utc), open=10, high=12, low=9, close=11)
    assert b.high >= b.open >= b.low
    assert b.high >= b.close >= b.low


def test_snapshot_requires_timezone_aware_timestamp():
    with pytest.raises(ValidationError):
        MarketSnapshot(instrument=Instrument(symbol="AAPL"), timestamp=datetime.now())  # naive


def test_snapshot_mid_prefers_bid_ask_over_last():
    snap = MarketSnapshot(
        instrument=Instrument(symbol="AAPL"),
        timestamp=datetime.now(timezone.utc),
        bid=100.0,
        ask=100.2,
        last=99.0,
    )
    assert snap.mid == pytest.approx(100.1)


def test_snapshot_mid_falls_back_to_last():
    snap = MarketSnapshot(
        instrument=Instrument(symbol="AAPL"), timestamp=datetime.now(timezone.utc), last=99.0
    )
    assert snap.mid == 99.0


def test_snapshot_staleness():
    old_ts = datetime.now(timezone.utc) - timedelta(seconds=30)
    snap = MarketSnapshot(instrument=Instrument(symbol="AAPL"), timestamp=old_ts)
    assert snap.is_stale(max_age_seconds=5)
    assert not snap.is_stale(max_age_seconds=60)
