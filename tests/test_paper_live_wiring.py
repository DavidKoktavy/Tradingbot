"""Tests for the real IBKR wiring in `paper` mode, and confirmation that
`simulate` mode's behaviour is completely unchanged by adding it."""

import argparse
from unittest.mock import AsyncMock, patch

import pytest

from app.cli import _gateway_and_feed_for_mode, main
from app.config import Settings, TradingMode
from app.dependency_container import build_container
from broker.order_manager import IBKROrderGateway
from broker.simulated_broker import SimulatedBrokerGateway


def loop_args(**overrides):
    defaults = dict(
        symbols=["AAPL"], strategies=["ma_crossover"], cycle_seconds=5.0,
        max_cycles=1, bar_size="1 min", history_duration="2 D",
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


# ---- _gateway_and_feed_for_mode: the core routing decision ------------------


def test_simulation_mode_uses_simulated_gateway_and_empty_feed():
    from app.control_loop import MarketDataFeed

    container = build_container(Settings(), symbols=["AAPL"])
    gateway, feed = _gateway_and_feed_for_mode(
        TradingMode.SIMULATION, container, loop_args()
    )
    assert gateway is container.gateway
    assert isinstance(gateway, SimulatedBrokerGateway)
    assert isinstance(feed, MarketDataFeed)


def test_paper_mode_requires_a_connected_client():
    """Calling this in PAPER mode with no client is a programming error,
    not a silent fallback to something fake — it must raise loudly."""
    container = build_container(Settings(), symbols=["AAPL"])
    with pytest.raises(ValueError, match="already-connected ibkr_client"):
        _gateway_and_feed_for_mode(TradingMode.PAPER, container, loop_args())


def test_paper_mode_uses_real_ibkr_gateway_and_live_feed():
    from broker.live_feed import LiveMarketDataFeed

    container = build_container(Settings(), symbols=["AAPL"])

    class FakeClient:
        _ib = object()

    gateway, feed = _gateway_and_feed_for_mode(
        TradingMode.PAPER, container, loop_args(), ibkr_client=FakeClient()
    )
    assert isinstance(gateway, IBKROrderGateway)
    assert isinstance(feed, LiveMarketDataFeed)


def test_paper_mode_feed_uses_configured_bar_size_and_history():
    container = build_container(Settings(), symbols=["AAPL"])

    class FakeClient:
        _ib = object()

    _, feed = _gateway_and_feed_for_mode(
        TradingMode.PAPER,
        container,
        loop_args(bar_size="5 mins", history_duration="5 D"),
        ibkr_client=FakeClient(),
    )
    assert feed._bar_size == "5 mins"  # noqa: SLF001
    assert feed._history_duration == "5 D"  # noqa: SLF001


# ---- simulate mode: regression, must be byte-for-byte unaffected -----------


def test_simulate_command_runs_unaffected_by_paper_wiring(capsys):
    """The most important regression test in this file: adding real IBKR
    support to `paper` must not change one line of `simulate`'s
    behaviour — it should still run entirely in-process, no network."""
    code = main(["simulate", "--symbols", "AAPL", "--max-cycles", "2"])
    assert code == 0
    output = capsys.readouterr().out
    assert "Connecting to IBKR" not in output
    assert "REAL IBKR paper account" not in output
    assert "Cycles: 2" in output


def test_simulate_never_constructs_ibkr_client():
    """Structural guarantee alongside the behavioural one above: running
    simulate must not even touch the ib_async-dependent code path."""
    with patch("broker.ibkr_client.IBKRClient") as mock_client:
        main(["simulate", "--symbols", "AAPL", "--max-cycles", "1"])
        mock_client.assert_not_called()


# ---- paper mode: connection and warm-up failures fail closed and clean -----


def test_paper_mode_reports_connection_failure_cleanly(capsys):
    """A failed IBKR connection must produce a clear message and a clean
    return, never an unhandled traceback dumped at a beginner."""
    with patch("broker.ibkr_client.IBKRClient") as MockClient:
        instance = MockClient.return_value
        instance.connect = AsyncMock(side_effect=ConnectionRefusedError("no TWS"))
        main(["paper", "--symbols", "AAPL", "--max-cycles", "1"])
    output = capsys.readouterr()
    assert "Could not connect to IBKR" in output.err
    assert "smoke_test_ibkr.py" in output.err


def test_paper_mode_reports_warmup_failure_and_disconnects(capsys):
    """Regression for the exact failure this project has already hit
    live: no historical bars available (e.g. account/data issues). Must
    report clearly AND disconnect the client rather than leaving a
    dangling TWS API session."""
    with patch("broker.ibkr_client.IBKRClient") as MockClient, \
         patch("broker.live_feed.LiveMarketDataFeed") as MockFeed:
        client_instance = MockClient.return_value
        client_instance.connect = AsyncMock()
        client_instance.disconnect = AsyncMock()

        feed_instance = MockFeed.return_value
        feed_instance.start = AsyncMock(
            side_effect=RuntimeError("No historical bars returned for AAPL")
        )

        main(["paper", "--symbols", "AAPL", "--max-cycles", "1"])

        client_instance.disconnect.assert_awaited_once()
    output = capsys.readouterr()
    assert "Could not warm up market data" in output.err
    assert "smoke_test_ibkr.py" in output.err


def test_paper_mode_announces_real_account_before_starting(capsys):
    """An operator must never be surprised that autonomous order
    submission against their real paper account has begun."""
    with patch("broker.ibkr_client.IBKRClient") as MockClient, \
         patch("broker.live_feed.LiveMarketDataFeed") as MockFeed:
        client_instance = MockClient.return_value
        client_instance.connect = AsyncMock()
        client_instance.disconnect = AsyncMock()
        feed_instance = MockFeed.return_value
        feed_instance.start = AsyncMock()
        feed_instance.run_ingest = AsyncMock()

        main(["paper", "--symbols", "AAPL", "--max-cycles", "0"])
    output = capsys.readouterr().out
    assert "REAL IBKR paper account" in output
    assert "WILL decide and submit orders autonomously" in output


def test_paper_still_refuses_when_trading_mode_is_live(monkeypatch, capsys):
    """Pre-existing safety check, re-verified unaffected by this change."""
    monkeypatch.setenv("TRADING_MODE", "LIVE")
    monkeypatch.setenv("ENABLE_LIVE_TRADING", "true")
    monkeypatch.setenv("LIVE_TRADING_CONFIRMATION", "I_UNDERSTAND_THIS_ENABLES_REAL_ORDERS")
    from app.config import get_settings

    get_settings.cache_clear()
    try:
        code = main(["paper", "--symbols", "AAPL"])
        assert code == 1
        assert "will not silently downgrade" in capsys.readouterr().err
    finally:
        get_settings.cache_clear()


def test_paper_command_disconnects_on_keyboard_interrupt():
    """Ctrl+C must still result in a clean IBKR disconnect, not an
    abandoned API session."""
    with patch("broker.ibkr_client.IBKRClient") as MockClient, \
         patch("broker.live_feed.LiveMarketDataFeed") as MockFeed:
        client_instance = MockClient.return_value
        client_instance.connect = AsyncMock()
        client_instance.disconnect = AsyncMock()
        feed_instance = MockFeed.return_value
        feed_instance.start = AsyncMock()
        feed_instance.run_ingest = AsyncMock()

        with patch(
            "app.control_loop.ControlLoop.start", AsyncMock(side_effect=KeyboardInterrupt)
        ):
            main(["paper", "--symbols", "AAPL"])

        client_instance.disconnect.assert_awaited_once()
