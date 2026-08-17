from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from broker.market_data import MarketDataService, MarketDataType
from data.models import AssetClass, Instrument
from tests.fakes import FakeIB


@pytest.fixture
def instrument() -> Instrument:
    return Instrument(symbol="AAPL", exchange="SMART", currency="USD")


async def test_subscribe_and_receive_snapshot(instrument):
    ib = FakeIB()
    svc = MarketDataService(ib)
    await svc.subscribe(instrument)

    ticker = svc._tickers[svc._key(instrument)]
    ticker.bid, ticker.ask, ticker.last = 100.0, 100.2, 100.1
    await ticker.updateEvent.emit(ticker)

    snap = svc.get_snapshot(instrument)
    assert snap is not None
    assert snap.bid == 100.0
    assert snap.ask == 100.2
    assert snap.mid == pytest.approx(100.1)


async def test_no_snapshot_yet_counts_as_stale(instrument):
    ib = FakeIB()
    svc = MarketDataService(ib)
    await svc.subscribe(instrument)
    assert svc.is_stale(instrument, max_age_seconds=5) is True


async def test_fresh_snapshot_is_not_stale(instrument):
    ib = FakeIB()
    svc = MarketDataService(ib)
    await svc.subscribe(instrument)
    ticker = svc._tickers[svc._key(instrument)]
    ticker.last = 50.0
    await ticker.updateEvent.emit(ticker)
    assert svc.is_stale(instrument, max_age_seconds=5) is False


async def test_old_snapshot_is_stale(instrument):
    ib = FakeIB()
    svc = MarketDataService(ib)
    await svc.subscribe(instrument)
    key = svc._key(instrument)
    # Manually age the stored snapshot.
    snap = svc.get_snapshot(instrument)
    ticker = svc._tickers[key]
    ticker.last = 50.0
    await ticker.updateEvent.emit(ticker)
    aged = svc._latest[key].model_copy(
        update={"timestamp": datetime.now(timezone.utc) - timedelta(seconds=60)}
    )
    svc._latest[key] = aged
    assert svc.is_stale(instrument, max_age_seconds=5) is True


async def test_unsupported_asset_class_raises(instrument):
    ib = FakeIB()
    svc = MarketDataService(ib)
    fx = Instrument(symbol="EURUSD", asset_class=AssetClass.FOREX)
    with pytest.raises(NotImplementedError):
        await svc.subscribe(fx)


async def test_historical_bars_are_normalized(instrument):
    ib = FakeIB()
    ib.historical_bars = [
        SimpleNamespace(date="2026-01-05T09:30:00", open=1, high=2, low=0.5, close=1.5, volume=1000),
        SimpleNamespace(date="2026-01-05T09:31:00", open=1.5, high=2.5, low=1.0, close=2.0, volume=800),
    ]
    svc = MarketDataService(ib)
    bars = await svc.get_historical_bars(instrument, duration="1 D", bar_size="1 min")
    assert len(bars) == 2
    assert bars[0].close == 1.5
    assert bars[0].timestamp.tzinfo is not None


async def test_negative_and_nan_ticker_values_are_cleaned(instrument):
    ib = FakeIB()
    svc = MarketDataService(ib)
    await svc.subscribe(instrument)
    ticker = svc._tickers[svc._key(instrument)]
    ticker.bid = -1.0  # ib_async "no value" sentinel
    ticker.ask = float("nan")
    ticker.last = 42.0
    await ticker.updateEvent.emit(ticker)
    snap = svc.get_snapshot(instrument)
    assert snap.bid is None
    assert snap.ask is None
    assert snap.last == 42.0


# ---- market data type (delayed-data fallback fix) --------------------------


async def test_defaults_to_live_market_data_type(instrument):
    """The API's own default is LIVE, and this must match it — the bug
    this guards against is silently defaulting to something that isn't
    what the account actually has permission for."""
    ib = FakeIB()
    svc = MarketDataService(ib)
    await svc.subscribe(instrument)
    assert ib.requested_market_data_types == [1]  # MarketDataType.LIVE


async def test_can_construct_with_delayed_market_data_type(instrument):
    ib = FakeIB()
    svc = MarketDataService(ib, market_data_type=MarketDataType.DELAYED)
    await svc.subscribe(instrument)
    assert ib.requested_market_data_types == [3]  # MarketDataType.DELAYED


async def test_reqMarketDataType_called_before_reqMktData(instrument):
    """Order matters: IBKR applies the type to subsequent reqMktData
    calls on the same session, so requesting it after would have no
    effect on this subscription."""
    calls = []
    ib = FakeIB()
    original_type_call = ib.reqMarketDataType
    original_data_call = ib.reqMktData
    ib.reqMarketDataType = lambda t: (calls.append("type"), original_type_call(t))[1]
    ib.reqMktData = lambda *a, **kw: (calls.append("data"), original_data_call(*a, **kw))[1]

    svc = MarketDataService(ib)
    await svc.subscribe(instrument)
    assert calls == ["type", "data"]


async def test_set_market_data_type_changes_future_subscriptions(instrument):
    """Regression test for the real issue: a fresh IBKR paper account has
    no real-time data subscription, so a client requesting LIVE (the
    default) gets nothing at all — TWS's own quote panel falls back to
    delayed automatically for a human, but the API does not. Switching
    the type and resubscribing must actually take effect."""
    ib = FakeIB()
    svc = MarketDataService(ib)
    await svc.subscribe(instrument)
    assert ib.requested_market_data_types == [1]

    svc.set_market_data_type(MarketDataType.DELAYED)
    assert svc.market_data_type is MarketDataType.DELAYED

    await svc.unsubscribe(instrument)
    await svc.subscribe(instrument)
    assert ib.requested_market_data_types == [1, 3]


def test_market_data_type_values_match_ibkr_api_exactly():
    """These integers are dictated by IBKR's own reqMarketDataType() API
    and are not ours to choose — a mismatch here would silently request
    the wrong data type against a real account."""
    assert int(MarketDataType.LIVE) == 1
    assert int(MarketDataType.FROZEN) == 2
    assert int(MarketDataType.DELAYED) == 3
    assert int(MarketDataType.DELAYED_FROZEN) == 4


async def test_ibkr_client_exposes_market_data_type_passthrough():
    """IBKRClient must not require reaching past it into the internal
    MarketDataService — that would leak an implementation detail into
    every caller."""
    from unittest.mock import patch

    from app.config import IBKRSettings
    from broker.ibkr_client import IBKRClient

    with patch("ib_async.IB"):
        client = IBKRClient(IBKRSettings(), market_data_type=MarketDataType.DELAYED)
        assert client.market_data_type is MarketDataType.DELAYED
        client.set_market_data_type(MarketDataType.LIVE)
        assert client.market_data_type is MarketDataType.LIVE


def test_ibkr_settings_market_data_type_configurable():
    from app.config import IBKRSettings

    assert IBKRSettings().market_data_type == "live"
    assert IBKRSettings(market_data_type="delayed").market_data_type == "delayed"


def test_ibkr_settings_rejects_invalid_market_data_type():
    from pydantic import ValidationError

    from app.config import IBKRSettings

    with pytest.raises(ValidationError):
        IBKRSettings(market_data_type="not_a_real_type")


def test_smoke_test_has_delayed_fallback_and_flag():
    """Structural check that the fallback logic and CLI flag actually
    exist in the script, matching the pattern used for its other safety
    checks elsewhere in this test suite."""
    from pathlib import Path

    source = Path("scripts/smoke_test_ibkr.py").read_text()
    assert "--market-data-type" in source
    assert "MarketDataType.DELAYED" in source
    assert "_wait_for_price" in source
    # The fallback must be reported, not hidden — silently succeeding on
    # delayed data while claiming a clean live-data pass would be exactly
    # the kind of misleading "OK" this whole project is built to avoid.
    assert "DELAYED DATA" in source
    assert "NOT usable for" in source
