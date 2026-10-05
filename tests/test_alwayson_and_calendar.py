"""
Tests for always-on operation and real exchange sessions.

Three separate correctness gaps, all of which only show up when the
process runs longer than a single watched session:

1. **Daily loss budget reset on restart.** The limit is measured against
   start-of-day equity. A restart that took current equity as the baseline
   forgot every loss already taken that day, so a bot that had hit its
   loss limit would cheerfully resume trading after a crash — exactly when
   it should not.

2. **Nightly IBKR restart.** IB Gateway / TWS restarts once a day. The
   fast retry budget (5 attempts, seconds) cannot span that, so a bot left
   running overnight was simply dead by morning, holding positions with no
   monitoring.

3. **Holiday- and DST-blind trading hours.** The fixed 13:30-20:00 UTC
   window is the correct Eastern session only during US standard time, and
   knows nothing about market holidays or 13:00 ET early closes.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal

import pytest

from app.config import Settings
from app.dependency_container import build_container
from broker.connection_manager import ConnectionManager
from broker.interfaces import ConnectionState
from data.exchange_calendar import ExchangeCalendar, resolve_calendar_name
from data.models import Instrument, MarketSnapshot
from database.repository import Repository
from execution.execution_models import OrderIntent, OrderSide, OrderType
from portfolio.portfolio_manager import AccountState, PortfolioManager
from risk.decisions import RejectionReason
from risk.kill_switch import KillSwitch, TradingHalt
from risk.risk_engine import RiskEngine, RiskEngineLimits
from tests.fakes import FakeIB

AAPL = Instrument(symbol="AAPL")


# ---- 1. daily loss baseline survives a restart ---------------------------


def test_restore_session_baseline_sets_an_unset_baseline():
    pm = PortfolioManager()
    pm.restore_session_baseline(Decimal("100000"))
    assert pm.start_of_day_equity == Decimal("100000")


def test_restore_session_baseline_never_overwrites_an_existing_one():
    """This is the safety property: the method must not be usable to hand
    back a loss budget that has already been spent."""
    pm = PortfolioManager()
    pm.restore_session_baseline(Decimal("100000"))
    pm.restore_session_baseline(Decimal("90000"))
    assert pm.start_of_day_equity == Decimal("100000")


def test_restore_before_update_account_wins():
    pm = PortfolioManager()
    pm.restore_session_baseline(Decimal("100000"))
    pm.update_account(
        AccountState(
            equity=Decimal("96000"),
            cash=Decimal("96000"),
            buying_power=Decimal("96000"),
        )
    )
    # The 4k already lost today is still counted against the limit.
    assert pm.start_of_day_equity == Decimal("100000")


def test_without_restore_a_restart_forgets_the_loss():
    """Documents the original behaviour, so a regression is visible."""
    pm = PortfolioManager()
    pm.update_account(
        AccountState(
            equity=Decimal("96000"),
            cash=Decimal("96000"),
            buying_power=Decimal("96000"),
        )
    )
    assert pm.start_of_day_equity == Decimal("96000")


def test_start_new_session_still_rolls_the_day():
    """The deliberate reset must keep working — it is how a new trading
    day legitimately gets a fresh budget."""
    pm = PortfolioManager()
    pm.restore_session_baseline(Decimal("100000"))
    pm.start_new_session(Decimal("96000"))
    assert pm.start_of_day_equity == Decimal("96000")


def test_daily_pnl_reflects_the_restored_baseline():
    pm = PortfolioManager()
    pm.restore_session_baseline(Decimal("100000"))
    pm.update_account(
        AccountState(
            equity=Decimal("97500"),
            cash=Decimal("97500"),
            buying_power=Decimal("97500"),
        )
    )
    assert pm.daily_pnl({}) == Decimal("-2500")


@pytest.fixture()
def repo(tmp_path):
    r = Repository(f"sqlite:///{tmp_path / 'test.db'}", mode="PAPER")
    r.create_schema()
    return r


def test_first_equity_of_day_returns_none_when_empty(repo):
    assert repo.first_equity_of_day() is None


def test_first_equity_of_day_returns_the_earliest_snapshot(repo):
    for equity in ("100000", "99000", "98000"):
        repo.save_account_snapshot(
            equity=Decimal(equity),
            cash=Decimal(equity),
            buying_power=Decimal(equity),
        )
    assert repo.first_equity_of_day() == Decimal("100000")


def test_first_equity_of_day_ignores_other_days(repo):
    repo.save_account_snapshot(
        equity=Decimal("100000"),
        cash=Decimal("100000"),
        buying_power=Decimal("100000"),
    )
    yesterday = datetime.now(timezone.utc).date() - timedelta(days=1)
    assert repo.first_equity_of_day(day=yesterday) is None


def test_first_equity_of_day_survives_a_broken_database(tmp_path):
    """Must never block startup."""
    r = Repository(f"sqlite:///{tmp_path / 'missing.db'}", mode="PAPER")
    assert r.first_equity_of_day() is None


def test_container_restores_the_baseline_from_the_repository(repo):
    repo.save_account_snapshot(
        equity=Decimal("100000"),
        cash=Decimal("100000"),
        buying_power=Decimal("100000"),
    )
    container = build_container(
        Settings(),
        symbols=["AAPL"],
        initial_equity=Decimal("96000"),  # equity after today's losses
        repository=repo,
    )
    assert container.portfolio.start_of_day_equity == Decimal("100000")


def test_container_falls_back_to_live_equity_with_no_history(repo):
    container = build_container(
        Settings(),
        symbols=["AAPL"],
        initial_equity=Decimal("96000"),
        repository=repo,
    )
    assert container.portfolio.start_of_day_equity == Decimal("96000")


def test_container_startup_survives_a_repository_that_raises():
    class _Exploding:
        def first_equity_of_day(self):
            raise RuntimeError("database on fire")

        def health_check(self) -> bool:
            return False

        write_failures = 0

    container = build_container(
        Settings(),
        symbols=["AAPL"],
        initial_equity=Decimal("96000"),
        repository=_Exploding(),
    )
    assert container.portfolio.start_of_day_equity == Decimal("96000")


# ---- 2. persistent reconnect --------------------------------------------


async def test_persistent_reconnect_is_off_by_default():
    """Fail-closed stays the default: surfacing the failure is correct
    unless an operator has deliberately asked for a daemon."""
    ib = FakeIB(fail_connect_times=99)
    cm = ConnectionManager(
        ib, host="127.0.0.1", port=7497, client_id=1, max_reconnect_attempts=2
    )
    cm._state = ConnectionState.CONNECTED
    ib._connected = False
    await ib.disconnectedEvent.emit()
    assert cm.state == ConnectionState.DISCONNECTED


async def test_persistent_reconnect_recovers_after_the_fast_budget(monkeypatch):
    import asyncio

    slept: list[float] = []

    async def _no_sleep(seconds: float) -> None:
        slept.append(seconds)

    monkeypatch.setattr(asyncio, "sleep", _no_sleep)

    # Fails the two fast attempts plus two slow ones, then succeeds.
    ib = FakeIB(fail_connect_times=4)
    cm = ConnectionManager(
        ib,
        host="127.0.0.1",
        port=7497,
        client_id=1,
        max_reconnect_attempts=2,
        persistent_reconnect=True,
        persistent_reconnect_base_delay=5.0,
    )
    cm._state = ConnectionState.CONNECTED
    ib._connected = False
    await ib.disconnectedEvent.emit()

    assert cm.is_connected
    assert len(slept) >= 2


async def test_persistent_reconnect_backoff_is_capped(monkeypatch):
    import asyncio

    slept: list[float] = []

    async def _no_sleep(seconds: float) -> None:
        slept.append(seconds)
        if len(slept) > 12:
            raise asyncio.CancelledError

    monkeypatch.setattr(asyncio, "sleep", _no_sleep)

    ib = FakeIB(fail_connect_times=10_000)
    cm = ConnectionManager(
        ib,
        host="127.0.0.1",
        port=7497,
        client_id=1,
        max_reconnect_attempts=1,
        persistent_reconnect=True,
        persistent_reconnect_base_delay=5.0,
        persistent_reconnect_max_delay=60.0,
    )
    cm._state = ConnectionState.CONNECTED
    ib._connected = False
    with pytest.raises(asyncio.CancelledError):
        await ib.disconnectedEvent.emit()

    assert max(slept) <= 60.0
    assert slept[0] == 5.0


async def test_on_reconnect_fires_after_recovery(monkeypatch):
    import asyncio

    async def _no_sleep(seconds: float) -> None:
        return None

    monkeypatch.setattr(asyncio, "sleep", _no_sleep)

    calls: list[str] = []

    async def on_reconnect() -> None:
        calls.append("re-check readiness")

    ib = FakeIB(fail_connect_times=3)
    cm = ConnectionManager(
        ib,
        host="127.0.0.1",
        port=7497,
        client_id=1,
        max_reconnect_attempts=1,
        persistent_reconnect=True,
        on_reconnect=on_reconnect,
    )
    cm._state = ConnectionState.CONNECTED
    ib._connected = False
    await ib.disconnectedEvent.emit()

    assert calls == ["re-check readiness"]


async def test_trading_is_halted_before_any_reconnect_attempt(monkeypatch):
    """Recovery must never precede the halt, or the loop could place
    orders against a half-known broker state."""
    import asyncio

    order: list[str] = []

    async def _no_sleep(seconds: float) -> None:
        order.append("sleep")

    monkeypatch.setattr(asyncio, "sleep", _no_sleep)

    async def on_disconnect() -> None:
        order.append("halt")

    async def on_reconnect() -> None:
        order.append("resume")

    ib = FakeIB(fail_connect_times=2)
    cm = ConnectionManager(
        ib,
        host="127.0.0.1",
        port=7497,
        client_id=1,
        max_reconnect_attempts=1,
        persistent_reconnect=True,
        on_disconnect=on_disconnect,
        on_reconnect=on_reconnect,
    )
    cm._state = ConnectionState.CONNECTED
    ib._connected = False
    await ib.disconnectedEvent.emit()

    assert order[0] == "halt"
    assert order[-1] == "resume"


async def test_a_failing_on_reconnect_callback_does_not_propagate(monkeypatch):
    import asyncio

    async def _no_sleep(seconds: float) -> None:
        return None

    monkeypatch.setattr(asyncio, "sleep", _no_sleep)

    async def on_reconnect() -> None:
        raise RuntimeError("readiness check failed")

    ib = FakeIB(fail_connect_times=2)
    cm = ConnectionManager(
        ib,
        host="127.0.0.1",
        port=7497,
        client_id=1,
        max_reconnect_attempts=1,
        persistent_reconnect=True,
        on_reconnect=on_reconnect,
    )
    cm._state = ConnectionState.CONNECTED
    ib._connected = False
    await ib.disconnectedEvent.emit()  # must not raise


async def test_stop_reconnecting_ends_the_loop(monkeypatch):
    import asyncio

    cm_holder: dict[str, ConnectionManager] = {}

    async def _no_sleep(seconds: float) -> None:
        # Shut it down from inside the first sleep, as a shutdown would.
        cm_holder["cm"].stop_reconnecting()

    monkeypatch.setattr(asyncio, "sleep", _no_sleep)

    ib = FakeIB(fail_connect_times=10_000)
    cm = ConnectionManager(
        ib,
        host="127.0.0.1",
        port=7497,
        client_id=1,
        max_reconnect_attempts=1,
        persistent_reconnect=True,
    )
    cm_holder["cm"] = cm
    cm._state = ConnectionState.CONNECTED
    ib._connected = False
    await ib.disconnectedEvent.emit()

    assert not cm.is_connected


async def test_explicit_disconnect_stops_reconnecting():
    ib = FakeIB()
    cm = ConnectionManager(
        ib, host="127.0.0.1", port=7497, client_id=1, persistent_reconnect=True
    )
    await cm.connect()
    await cm.disconnect()
    assert cm._stop_reconnecting is True


# ---- 3. exchange calendar -----------------------------------------------


def test_smart_maps_to_nyse():
    """SMART is IBKR's router, not a venue; the NYSE session is the
    correct conservative mapping for US equities."""
    assert resolve_calendar_name("SMART") == "XNYS"


def test_nasdaq_maps_to_xnas():
    assert resolve_calendar_name("nasdaq") == "XNAS"


def test_unknown_exchange_passes_through_uppercased():
    assert resolve_calendar_name("xjpx") == "XJPX"


def test_calendar_loads_for_a_known_exchange():
    cal = ExchangeCalendar("SMART")
    assert cal.is_available
    assert cal.name == "XNYS"


def test_unknown_exchange_fails_closed():
    cal = ExchangeCalendar("NOT-A-REAL-EXCHANGE")
    assert not cal.is_available
    assert cal.is_open(datetime(2026, 6, 15, 15, 0, tzinfo=timezone.utc)) is False


def test_naive_timestamp_fails_closed():
    """Assuming a timezone is the exact bug this module exists to fix."""
    cal = ExchangeCalendar("SMART")
    assert cal.is_open(datetime(2026, 6, 15, 15, 0)) is False


def test_open_during_a_regular_session():
    # Monday 15 June 2026, 15:00 UTC = 11:00 EDT.
    cal = ExchangeCalendar("SMART")
    assert cal.is_open(datetime(2026, 6, 15, 15, 0, tzinfo=timezone.utc))


def test_closed_on_a_weekend():
    cal = ExchangeCalendar("SMART")
    assert not cal.is_open(datetime(2026, 6, 13, 15, 0, tzinfo=timezone.utc))


def test_closed_on_christmas_day():
    """The fixed window had no concept of holidays at all."""
    cal = ExchangeCalendar("SMART")
    assert not cal.is_trading_day(date(2026, 12, 25))


def test_closed_on_independence_day_observed():
    cal = ExchangeCalendar("SMART")
    assert not cal.is_trading_day(date(2026, 7, 3))


def test_dst_summer_close_is_respected():
    """20:00 UTC is 16:00 ET only in winter. In July it is 16:00 BST /
    17:00 CEST and an hour PAST the US close — the fixed window would
    have permitted an hour of after-hours trading every summer."""
    cal = ExchangeCalendar("SMART")
    assert not cal.is_open(datetime(2026, 7, 15, 20, 30, tzinfo=timezone.utc))


def test_dst_summer_open_is_respected():
    """And conversely: 13:30 UTC in July is 09:30 BST, a full hour before
    the 09:30 EDT open — the fixed window blocked the real first hour."""
    cal = ExchangeCalendar("SMART")
    assert cal.is_open(datetime(2026, 7, 15, 14, 0, tzinfo=timezone.utc))


def test_describe_distinguishes_a_holiday_from_wrong_hours():
    """Reading an audit trail, these mean very different things: one says
    today was never going to trade, the other says the bot ran at the
    wrong time of day."""
    cal = ExchangeCalendar("SMART")
    holiday = cal.describe(datetime(2026, 12, 25, 15, 0, tzinfo=timezone.utc))
    off_hours = cal.describe(datetime(2026, 6, 15, 2, 0, tzinfo=timezone.utc))
    assert "not a trading session" in holiday
    assert "outside" in off_hours


def test_calendars_are_cached():
    assert ExchangeCalendar("SMART")._calendar is ExchangeCalendar("NYSE")._calendar


# ---- risk engine wiring -------------------------------------------------


def _engine(**kw) -> RiskEngine:
    portfolio = PortfolioManager()
    portfolio.update_account(
        AccountState(
            equity=Decimal("100000"),
            cash=Decimal("100000"),
            buying_power=Decimal("100000"),
        )
    )
    return RiskEngine(
        limits=RiskEngineLimits(),
        portfolio=portfolio,
        kill_switch=KillSwitch(),
        trading_halt=TradingHalt(),
        enforce_trading_hours=True,
        **kw,
    )


def test_engine_uses_the_calendar_when_one_is_configured():
    engine = _engine(exchange_calendar=ExchangeCalendar("SMART"))
    decision = engine._check_trading_hours(
        datetime(2026, 12, 25, 15, 0, tzinfo=timezone.utc)
    )
    assert not decision.approved
    assert decision.reason is RejectionReason.OUTSIDE_TRADING_HOURS


def test_engine_approves_a_real_session_with_the_calendar():
    engine = _engine(exchange_calendar=ExchangeCalendar("SMART"))
    assert engine._check_trading_hours(
        datetime(2026, 6, 15, 15, 0, tzinfo=timezone.utc)
    ).approved


def test_fixed_window_fallback_admits_it_is_holiday_blind():
    """Without a calendar the check still runs, but the rejection message
    says what it cannot account for, so nobody reads it as authoritative."""
    engine = _engine()
    decision = engine._check_trading_hours(
        datetime(2026, 6, 15, 2, 0, tzinfo=timezone.utc)
    )
    assert not decision.approved
    assert "fixed-window fallback" in decision.detail


def test_fixed_window_wrongly_permits_christmas_without_a_calendar():
    """Documents the gap the calendar closes. Christmas Day 2026 is a
    Friday inside the fixed window, so this is approved — which is why the
    calendar matters."""
    engine = _engine()
    assert engine._check_trading_hours(
        datetime(2026, 12, 25, 15, 0, tzinfo=timezone.utc)
    ).approved


def test_calendar_rejection_cannot_be_bypassed_by_a_signal_source():
    """The AI cannot route around a deterministic control. Checked at the
    full evaluate() boundary, not just the private method."""
    engine = _engine(exchange_calendar=ExchangeCalendar("SMART"))
    intent = OrderIntent(
        instrument=AAPL,
        side=OrderSide.BUY,
        quantity=Decimal("10"),
        order_type=OrderType.MARKET,
        source="ai_decision_engine",
    )
    christmas = datetime(2026, 12, 25, 15, 0, tzinfo=timezone.utc)
    snapshot = MarketSnapshot(
        instrument=AAPL, timestamp=christmas, bid=99.95, ask=100.05, last=100.0
    )
    assessment = engine.evaluate(
        intent,
        snapshot=snapshot,
        prices={AAPL.symbol: Decimal("100")},
        now=christmas,
    )
    assert not assessment.approved


def test_enforcement_disabled_still_short_circuits_everything():
    portfolio = PortfolioManager()
    portfolio.update_account(
        AccountState(
            equity=Decimal("100000"),
            cash=Decimal("100000"),
            buying_power=Decimal("100000"),
        )
    )
    engine = RiskEngine(
        limits=RiskEngineLimits(),
        portfolio=portfolio,
        kill_switch=KillSwitch(),
        trading_halt=TradingHalt(),
        enforce_trading_hours=False,
        exchange_calendar=ExchangeCalendar("SMART"),
    )
    assert engine._check_trading_hours(
        datetime(2026, 12, 25, 15, 0, tzinfo=timezone.utc)
    ).approved
