"""Phase 9 tests: health, metrics, alerts, dashboard."""

from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from app.config import TradingMode
from app.mode_gate import ModeGate
from data.models import Instrument, MarketSnapshot
from execution.execution_models import Fill, Order, OrderIntent, OrderSide
from execution.order_store import OrderStore
from monitoring.alerts import (
    AlertCategory,
    AlertManager,
    AlertSeverity,
    CollectingProvider,
    FailingProvider,
    LogNotificationProvider,
    alert_daily_loss,
    alert_kill_switch,
)
from monitoring.dashboard import DashboardState
from monitoring.health import (
    CheckResult,
    HealthMonitor,
    HealthStatus,
    Severity,
    ai_provider_check,
    broker_connection_check,
    kill_switch_check,
    loop_liveness_check,
    market_data_freshness_check,
    ok,
    portfolio_check,
    unhealthy,
)
from monitoring.metrics import M, MetricsRegistry
from portfolio.portfolio_manager import AccountState, PortfolioManager
from risk.kill_switch import KillSwitch, KillSwitchTrigger, TradingHalt
from risk.risk_engine import RiskEngine, RiskEngineLimits

AAPL = Instrument(symbol="AAPL")


def snap(mid: float = 100.0, age: float = 0.0) -> MarketSnapshot:
    return MarketSnapshot(
        instrument=AAPL,
        timestamp=datetime.now(timezone.utc) - timedelta(seconds=age),
        bid=mid - 0.05,
        ask=mid + 0.05,
        last=mid,
    )


class FakeFeed:
    def __init__(self, snapshots: dict[str, MarketSnapshot] | None = None) -> None:
        self._snapshots = snapshots or {}

    def snapshot(self, instrument):
        return self._snapshots.get(str(instrument))

    def history(self, instrument):
        return []


# ---- health ---------------------------------------------------------------


async def test_all_passing_is_healthy():
    monitor = HealthMonitor()
    monitor.register("a", lambda: ok("fine"))
    monitor.register("b", lambda: ok("fine"))
    report = await monitor.run()
    assert report.status is HealthStatus.HEALTHY
    assert report.can_trade


async def test_check_result_defaults_to_unhealthy():
    """A check that fails to set a verdict must not read as a pass."""
    assert CheckResult(name="x").status is HealthStatus.UNHEALTHY


async def test_raising_check_is_unhealthy_not_unknown():
    monitor = HealthMonitor()

    def explode():
        raise RuntimeError("check itself is broken")

    monitor.register("broken", explode, severity=Severity.CRITICAL)
    report = await monitor.run()
    assert report.status is HealthStatus.UNHEALTHY
    assert not report.can_trade
    assert "raised" in report.get("broken").detail


async def test_critical_failure_blocks_trading():
    monitor = HealthMonitor()
    monitor.register("broker", lambda: unhealthy("disconnected"), severity=Severity.CRITICAL)
    report = await monitor.run()
    assert not report.can_trade


async def test_degraded_failure_does_not_block_trading():
    """A flaky non-critical dependency must not stop trading, or operators
    learn to disable health checking entirely."""
    monitor = HealthMonitor()
    monitor.register("ai", lambda: unhealthy("provider down"), severity=Severity.DEGRADED)
    report = await monitor.run()
    assert report.status is HealthStatus.DEGRADED
    assert report.can_trade


async def test_hanging_check_times_out():
    import asyncio

    monitor = HealthMonitor(check_timeout_seconds=0.05)

    async def hang():
        await asyncio.sleep(10)
        return ok()

    monitor.register("slow", hang, severity=Severity.CRITICAL)
    report = await monitor.run()
    assert not report.can_trade
    assert "timed out" in report.get("slow").detail


async def test_broker_connection_check():
    class Conn:
        is_connected = True
        state = "CONNECTED"

    monitor = HealthMonitor()
    monitor.register("broker", broker_connection_check(Conn()), severity=Severity.CRITICAL)
    assert (await monitor.run()).can_trade

    Conn.is_connected = False
    Conn.state = "DISCONNECTED"
    assert not (await monitor.run()).can_trade


async def test_market_data_freshness_check():
    monitor = HealthMonitor()
    feed = FakeFeed({str(AAPL): snap()})
    monitor.register(
        "data", market_data_freshness_check(feed, [AAPL], 5.0), severity=Severity.CRITICAL
    )
    assert (await monitor.run()).can_trade

    stale_feed = FakeFeed({str(AAPL): snap(age=600)})
    monitor.register(
        "data", market_data_freshness_check(stale_feed, [AAPL], 5.0), severity=Severity.CRITICAL
    )
    assert not (await monitor.run()).can_trade


async def test_missing_market_data_is_unhealthy():
    monitor = HealthMonitor()
    monitor.register(
        "data", market_data_freshness_check(FakeFeed(), [AAPL], 5.0), severity=Severity.CRITICAL
    )
    report = await monitor.run()
    assert not report.can_trade
    assert "missing" in report.get("data").detail


async def test_kill_switch_check_is_visible_but_not_critical():
    """The kill switch already stops trading; marking it critical would
    confuse 'why can't we trade'."""
    ks = KillSwitch()
    ks.activate(KillSwitchTrigger.MANUAL, "test")
    monitor = HealthMonitor()
    monitor.register("kill_switch", kill_switch_check(ks))
    report = await monitor.run()
    assert report.status is HealthStatus.DEGRADED
    assert report.can_trade  # health doesn't double-block


async def test_ai_unavailability_is_degraded():
    class Engine:
        provider_available = False

    monitor = HealthMonitor()
    monitor.register("ai", ai_provider_check(Engine()))
    report = await monitor.run()
    assert report.can_trade
    assert "deterministic strategies only" in report.get("ai").detail


async def test_portfolio_check_flags_zero_equity():
    portfolio = PortfolioManager()
    portfolio.update_account(AccountState(equity=Decimal("0")))
    monitor = HealthMonitor()
    monitor.register("portfolio", portfolio_check(portfolio), severity=Severity.CRITICAL)
    assert not (await monitor.run()).can_trade


async def test_loop_liveness_check():
    class Stats:
        last_cycle_at = datetime.now(timezone.utc)

    monitor = HealthMonitor()
    monitor.register("loop", loop_liveness_check(Stats()), severity=Severity.CRITICAL)
    assert (await monitor.run()).can_trade

    Stats.last_cycle_at = datetime.now(timezone.utc) - timedelta(seconds=600)
    assert not (await monitor.run()).can_trade


async def test_health_monitor_does_not_mutate_trading_state():
    """Health checks report; they must not halt trading as a side effect."""
    halt = TradingHalt()
    ks = KillSwitch()
    monitor = HealthMonitor()
    monitor.register("broker", lambda: unhealthy("down"), severity=Severity.CRITICAL)
    await monitor.run()
    assert not halt.is_halted
    assert not ks.is_active


# ---- metrics ---------------------------------------------------------------


def test_counter_increments():
    m = MetricsRegistry()
    m.increment(M.ORDERS_SUBMITTED)
    m.increment(M.ORDERS_SUBMITTED, 2)
    assert m.counter(M.ORDERS_SUBMITTED) == 3


def test_counters_with_labels_are_separate():
    m = MetricsRegistry()
    m.increment(M.RISK_REJECTIONS, reason="STALE")
    m.increment(M.RISK_REJECTIONS, reason="STALE")
    m.increment(M.RISK_REJECTIONS, reason="KILL_SWITCH")
    assert m.counter(M.RISK_REJECTIONS, reason="STALE") == 2
    assert m.counter(M.RISK_REJECTIONS, reason="KILL_SWITCH") == 1


def test_gauge_overwrites():
    m = MetricsRegistry()
    m.set_gauge(M.EQUITY, 100000)
    m.set_gauge(M.EQUITY, 99000)
    assert m.gauge(M.EQUITY) == 99000


def test_histogram_percentiles():
    m = MetricsRegistry()
    for i in range(100):
        m.observe(M.ORDER_LATENCY, float(i))
    hist = m.histogram(M.ORDER_LATENCY)
    assert hist.count == 100
    assert hist.percentile(0.5) == pytest.approx(50, abs=2)
    assert hist.percentile(0.95) == pytest.approx(95, abs=2)


def test_histogram_is_bounded():
    """Instrumentation must not leak memory in a long-running process."""
    m = MetricsRegistry()
    for i in range(5000):
        m.observe("x", float(i))
    assert m.histogram("x").count <= 1000


def test_timer_records_duration():
    import time

    m = MetricsRegistry()
    with m.timer(M.CYCLE_DURATION):
        time.sleep(0.01)
    hist = m.histogram(M.CYCLE_DURATION)
    assert hist.count == 1
    assert hist.mean > 5  # at least 5ms


def test_metrics_never_raise():
    """A monitoring failure must not propagate into the trading path."""
    m = MetricsRegistry()

    class Unconvertible:
        def __float__(self):
            raise ValueError("nope")

    m.set_gauge("bad", Unconvertible())  # must not raise
    m.observe("bad", Unconvertible())  # must not raise
    assert m.gauge("bad") is None


def test_prometheus_export_format():
    m = MetricsRegistry()
    m.increment(M.ORDERS_SUBMITTED, 5)
    m.set_gauge(M.EQUITY, 100000)
    m.observe(M.ORDER_LATENCY, 12.5)
    output = m.to_prometheus()
    assert "agent_uptime_seconds" in output
    assert "orders_submitted_total 5" in output
    assert "account_equity 100000" in output
    assert "order_latency_ms_count 1" in output


def test_snapshot_contains_all_types():
    m = MetricsRegistry()
    m.increment("c")
    m.set_gauge("g", 1)
    m.observe("h", 1.0)
    snapshot = m.snapshot()
    assert "c" in snapshot["counters"]
    assert "g" in snapshot["gauges"]
    assert "h" in snapshot["histograms"]


# ---- alerts ----------------------------------------------------------------


async def test_alert_is_delivered():
    provider = CollectingProvider()
    manager = AlertManager([provider])
    await manager.raise_alert(
        key="test", category=AlertCategory.SYSTEM, severity=AlertSeverity.WARNING, title="Test"
    )
    assert len(provider.alerts) == 1
    assert provider.alerts[0].title == "Test"


async def test_duplicate_alert_suppressed_by_cooldown():
    """Alert fatigue is a safety problem: an unhandled condition firing
    every cycle must not bury the operator."""
    provider = CollectingProvider()
    manager = AlertManager([provider], cooldown_seconds=300)
    for _ in range(10):
        await manager.raise_alert(
            key="same",
            category=AlertCategory.SYSTEM,
            severity=AlertSeverity.WARNING,
            title="Repeated",
        )
    assert len(provider.alerts) == 1
    assert manager.suppressed_count("same") == 9


async def test_different_keys_not_suppressed():
    provider = CollectingProvider()
    manager = AlertManager([provider], cooldown_seconds=300)
    for i in range(3):
        await manager.raise_alert(
            key=f"key-{i}",
            category=AlertCategory.SYSTEM,
            severity=AlertSeverity.CRITICAL,
            title=f"Alert {i}",
        )
    assert len(provider.alerts) == 3


async def test_cooldown_expires():
    provider = CollectingProvider()
    manager = AlertManager([provider], cooldown_seconds=60)
    now = datetime.now(timezone.utc)
    await manager.raise_alert(
        key="k", category=AlertCategory.SYSTEM, severity=AlertSeverity.INFO,
        title="First", now=now,
    )
    await manager.raise_alert(
        key="k", category=AlertCategory.SYSTEM, severity=AlertSeverity.INFO,
        title="Second", now=now + timedelta(seconds=61),
    )
    assert len(provider.alerts) == 2


async def test_clearing_cooldown_allows_immediate_realert():
    provider = CollectingProvider()
    manager = AlertManager([provider], cooldown_seconds=3600)
    await manager.raise_alert(
        key="k", category=AlertCategory.BROKER, severity=AlertSeverity.CRITICAL, title="Down"
    )
    manager.clear_cooldown("k")  # condition resolved
    await manager.raise_alert(
        key="k", category=AlertCategory.BROKER, severity=AlertSeverity.CRITICAL, title="Down again"
    )
    assert len(provider.alerts) == 2


async def test_delivery_failure_does_not_propagate():
    """A dead notification provider must not raise into the trading loop."""
    manager = AlertManager([FailingProvider()])
    alert = await manager.raise_alert(
        key="k", category=AlertCategory.SYSTEM, severity=AlertSeverity.CRITICAL, title="Test"
    )
    assert alert is not None  # still recorded
    assert manager.delivery_failures == 1
    assert len(manager.history) == 1  # visible on the dashboard regardless


async def test_multiple_providers_all_receive():
    a, b = CollectingProvider(), CollectingProvider()
    manager = AlertManager([a, b])
    await manager.raise_alert(
        key="k", category=AlertCategory.SYSTEM, severity=AlertSeverity.INFO, title="T"
    )
    assert len(a.alerts) == 1 and len(b.alerts) == 1


async def test_one_failing_provider_does_not_block_others():
    good = CollectingProvider()
    manager = AlertManager([FailingProvider(), good])
    await manager.raise_alert(
        key="k", category=AlertCategory.SYSTEM, severity=AlertSeverity.INFO, title="T"
    )
    assert len(good.alerts) == 1


async def test_convenience_raisers():
    provider = CollectingProvider()
    manager = AlertManager([provider])
    await alert_kill_switch(manager, "DAILY_LOSS_LIMIT", "limit hit")
    await alert_daily_loss(manager, -0.025, -0.02)
    assert len(provider.alerts) == 2
    assert all(a.severity is AlertSeverity.CRITICAL for a in provider.alerts)


async def test_history_is_bounded():
    manager = AlertManager([CollectingProvider()], cooldown_seconds=0, history_size=10)
    for i in range(50):
        await manager.raise_alert(
            key=f"k{i}", category=AlertCategory.SYSTEM,
            severity=AlertSeverity.INFO, title=f"A{i}",
        )
    assert len(manager.history) == 10


async def test_default_provider_never_fails():
    manager = AlertManager()  # LogNotificationProvider by default
    await manager.raise_alert(
        key="k", category=AlertCategory.SYSTEM, severity=AlertSeverity.CRITICAL, title="T"
    )
    assert manager.delivery_failures == 0


# ---- dashboard --------------------------------------------------------------


@pytest.fixture
def dashboard_state():
    portfolio = PortfolioManager()
    portfolio.update_account(
        AccountState(
            equity=Decimal("100000"), cash=Decimal("100000"), buying_power=Decimal("200000")
        )
    )
    store = OrderStore()
    ks = KillSwitch()
    halt = TradingHalt()
    risk = RiskEngine(
        limits=RiskEngineLimits(), portfolio=portfolio, kill_switch=ks, trading_halt=halt
    )
    return DashboardState(
        mode_gate=ModeGate(TradingMode.PAPER),
        portfolio=portfolio,
        order_store=store,
        risk_engine=risk,
        kill_switch=ks,
        trading_halt=halt,
        health_monitor=HealthMonitor(),
        metrics=MetricsRegistry(),
        alert_manager=AlertManager([CollectingProvider()]),
        feed=FakeFeed({str(AAPL): snap()}),
        instruments=[AAPL],
    )


def test_snapshot_always_includes_mode(dashboard_state):
    """Ambiguity about mode is how people trade real money believing they
    are on paper."""
    snapshot = dashboard_state.full_snapshot()
    assert snapshot["mode"]["mode"] == "PAPER"
    assert snapshot["mode"]["is_live"] is False


def test_every_api_view_includes_mode(dashboard_state):
    for view in (
        dashboard_state.mode_view(),
    ):
        assert "mode" in view


def test_account_view_reports_missing_marks_rather_than_wrong_numbers(dashboard_state):
    # Open a position with no price available.
    order = Order(
        intent=OrderIntent(
            instrument=Instrument(symbol="TSLA"),
            side=OrderSide.BUY,
            quantity=Decimal("10"),
            source="test",
        )
    )
    dashboard_state.portfolio.apply_fill(
        order,
        Fill(
            fill_id="f",
            order_id=order.order_id,
            timestamp=datetime.now(timezone.utc),
            quantity=Decimal("10"),
            price=Decimal("200"),
        ),
    )
    view = dashboard_state.account_view()
    assert view["daily_pnl"] is None
    assert "pnl_unavailable_reason" in view


def test_risk_view_exposes_limits_and_utilisation(dashboard_state):
    view = dashboard_state.risk_view()
    assert view["limits"]["max_daily_loss"] == 0.02
    assert view["kill_switch_active"] is False
    assert "risk_utilisation" in view


def test_dashboard_survives_broken_component(dashboard_state):
    """A dashboard failure must never look like a trading problem."""
    class Broken:
        @property
        def account(self):
            raise RuntimeError("component exploded")

        positions = {}
        realized_pnl = Decimal("0")

    dashboard_state.portfolio = Broken()
    view = dashboard_state.account_view()
    assert "error" in view
    # And the full snapshot still renders.
    snapshot = dashboard_state.full_snapshot()
    assert snapshot["mode"]["mode"] == "PAPER"


def test_no_secrets_in_snapshot(dashboard_state):
    import json

    rendered = json.dumps(dashboard_state.full_snapshot()).lower()
    for secret in ("api_key", "password", "secret", "token", "account_id"):
        assert secret not in rendered


def test_html_shows_mode_prominently(dashboard_state):
    from monitoring.dashboard import _render_html

    html = _render_html(dashboard_state)
    assert "MODE: PAPER" in html
    assert "REAL MONEY" not in html


def test_html_warns_loudly_in_live_mode(dashboard_state):
    from app.mode_gate import LIVE_CONFIRMATION_PHRASE, ModeAuthorisation
    from monitoring.dashboard import _render_html

    dashboard_state.mode_gate = ModeGate(
        TradingMode.LIVE,
        authorisation=ModeAuthorisation(
            enable_live_trading=True,
            confirmation_phrase=LIVE_CONFIRMATION_PHRASE,
            operator_acknowledged=True,
        ),
    )
    html = _render_html(dashboard_state)
    assert "MODE: LIVE" in html
    assert "REAL MONEY" in html


def test_html_shows_kill_switch_banner(dashboard_state):
    from monitoring.dashboard import _render_html

    dashboard_state.kill_switch.activate(KillSwitchTrigger.DAILY_LOSS_LIMIT, "limit")
    html = _render_html(dashboard_state)
    assert "KILL SWITCH ACTIVE" in html


def test_dashboard_has_no_deactivation_endpoint():
    """Resuming trading must require deliberate action outside the web
    surface."""
    import inspect

    from monitoring import dashboard

    source = inspect.getsource(dashboard.create_app)
    assert "kill-switch/activate" in source
    assert "deactivate" not in source.replace(
        "Deactivation is not available via the dashboard.", ""
    ).replace("no\n    deactivation endpoint", "")


def test_dashboard_mutating_endpoints_are_limited():
    import inspect

    from monitoring import dashboard

    source = inspect.getsource(dashboard.create_app)
    post_count = source.count("@app.post")
    assert post_count == 1, "Only the kill switch may mutate state"
    for forbidden in ("@app.put", "@app.delete", "@app.patch"):
        assert forbidden not in source


# ---- dashboard HTTP layer ---------------------------------------------------


@pytest.fixture
def client(dashboard_state):
    from fastapi.testclient import TestClient

    from monitoring.dashboard import create_app

    dashboard_state.health.register("broker", lambda: ok("connected"), severity=Severity.CRITICAL)
    return TestClient(create_app(dashboard_state)), dashboard_state


def test_health_endpoint(client):
    c, _ = client
    response = c.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "HEALTHY"
    assert body["mode"] == "PAPER"


def test_all_read_endpoints_serve(client):
    c, _ = client
    for path in ("/api/snapshot", "/api/account", "/api/positions",
                 "/api/orders", "/api/risk", "/api/alerts", "/metrics", "/"):
        assert c.get(path).status_code == 200, path


def test_metrics_endpoint_is_prometheus_text(client):
    c, state = client
    state.metrics.increment(M.ORDERS_SUBMITTED, 7)
    body = c.get("/metrics").text
    assert "orders_submitted_total 7" in body
    assert "agent_uptime_seconds" in body


def test_kill_switch_endpoint_activates(client):
    c, state = client
    assert not state.kill_switch.is_active
    response = c.post("/kill-switch/activate?reason=incident")
    assert response.status_code == 200
    assert state.kill_switch.is_active


def test_no_deactivation_route_exists(client):
    c, state = client
    state.kill_switch.activate(KillSwitchTrigger.MANUAL, "test")
    for path in ("/kill-switch/deactivate", "/kill-switch/reset", "/kill-switch/clear"):
        assert c.post(path).status_code == 404
    assert state.kill_switch.is_active


def test_no_endpoint_can_raise_limits_or_enable_live(client):
    c, _ = client
    for path in ("/api/limits", "/api/risk/limits", "/mode/promote", "/api/orders/submit"):
        assert c.post(path).status_code in (404, 405), path


def test_html_dashboard_renders(client):
    c, _ = client
    body = c.get("/").text
    assert "TRADING AGENT" in body
    assert "MODE: PAPER" in body


# ---- control loop instrumentation ------------------------------------------


@pytest.fixture
def instrumented_loop():
    from app.control_loop import ControlLoop, MarketDataFeed
    from broker.order_manager import OrderManager
    from broker.simulated_broker import SimulatedBrokerGateway
    from data.models import Bar
    from execution.order_validator import OrderValidator
    from execution.reconciliation import Reconciler
    from strategies.engine import StrategyEngine
    from strategies.ma_crossover import MACrossoverParams, MACrossoverStrategy

    gateway = SimulatedBrokerGateway()
    gateway.set_snapshot(snap())
    store = OrderStore()
    portfolio = PortfolioManager()
    portfolio.update_account(
        AccountState(
            equity=Decimal("100000"), cash=Decimal("100000"), buying_power=Decimal("200000")
        )
    )
    ks, halt = KillSwitch(), TradingHalt()
    risk = RiskEngine(
        limits=RiskEngineLimits(), portfolio=portfolio, kill_switch=ks, trading_halt=halt
    )
    mode = ModeGate(TradingMode.PAPER)
    feed = MarketDataFeed()
    feed.snapshots[str(AAPL)] = snap()
    base = datetime(2024, 1, 2, tzinfo=timezone.utc)
    feed.bars[str(AAPL)] = [
        Bar(timestamp=base + timedelta(days=i), open=c, high=c * 1.01, low=c * 0.99,
            close=c, volume=200000)
        for i, c in enumerate([100.0] * 20 + [101.0])
    ]

    provider = CollectingProvider()
    metrics = MetricsRegistry()
    loop = ControlLoop(
        instruments=[AAPL],
        feed=feed,
        strategy_engine=StrategyEngine(
            [MACrossoverStrategy(MACrossoverParams(fast_period=3, slow_period=10, atr_period=5))]
        ),
        risk_engine=risk,
        validator=OrderValidator(store),
        order_manager=OrderManager(gateway, store, mode),
        order_store=store,
        portfolio=portfolio,
        reconciler=Reconciler(store, portfolio),
        kill_switch=ks,
        trading_halt=halt,
        mode_gate=mode,
        cycle_seconds=0.0,
        reconcile_every_n_cycles=0,
        metrics=metrics,
        alerts=AlertManager([provider]),
    )
    return loop, metrics, provider, portfolio, ks


async def test_loop_records_metrics(instrumented_loop):
    loop, metrics, provider, portfolio, ks = instrumented_loop
    await loop.reconcile()
    await loop.run_cycle()
    assert metrics.counter(M.CYCLES) >= 1
    assert metrics.gauge(M.EQUITY) == 100000.0
    assert metrics.gauge(M.KILL_SWITCH_ACTIVE) == 0.0


async def test_loop_records_order_metrics(instrumented_loop):
    loop, metrics, provider, portfolio, ks = instrumented_loop
    await loop.reconcile()
    await loop.run_cycle()
    assert metrics.counter(M.ORDERS_SUBMITTED, source="ma_crossover") >= 1
    assert metrics.histogram(M.ORDER_LATENCY).count >= 1


async def test_loop_records_rejection_reasons(instrumented_loop):
    loop, metrics, provider, portfolio, ks = instrumented_loop
    await loop.reconcile()
    ks.activate(KillSwitchTrigger.MANUAL, "test")
    await loop.run_cycle()
    # Kill switch stops the cycle before intents are generated.
    assert metrics.gauge(M.KILL_SWITCH_ACTIVE) == 1.0


async def test_daily_loss_breach_raises_alert(instrumented_loop):
    loop, metrics, provider, portfolio, ks = instrumented_loop
    await loop.reconcile()
    portfolio.update_account(
        AccountState(equity=Decimal("97000"), buying_power=Decimal("100000"))
    )
    await loop.run_cycle()
    assert ks.is_active
    assert any(a.category is AlertCategory.RISK_LIMIT for a in provider.alerts)


async def test_broker_disconnect_raises_alert(instrumented_loop):
    loop, metrics, provider, portfolio, ks = instrumented_loop
    await loop.reconcile()
    await loop.on_broker_disconnect()
    assert any(a.category is AlertCategory.BROKER for a in provider.alerts)
    assert metrics.gauge(M.IBKR_CONNECTED) == 0.0


async def test_dirty_reconciliation_raises_alert(instrumented_loop):
    from execution.reconciliation import BrokerPosition

    loop, metrics, provider, portfolio, ks = instrumented_loop
    loop._orders._gateway.set_position(
        BrokerPosition(instrument=AAPL, quantity=Decimal("500"), average_cost=Decimal("90"))
    )
    await loop.reconcile()
    assert any(a.category is AlertCategory.UNEXPECTED_POSITION for a in provider.alerts)


async def test_failing_alert_provider_does_not_stop_loop(instrumented_loop):
    from monitoring.alerts import FailingProvider

    loop, metrics, provider, portfolio, ks = instrumented_loop
    loop._alerts = AlertManager([FailingProvider()])
    await loop.reconcile()
    await loop.run_cycle()  # must not raise
    assert loop.stats.cycles >= 1
    assert loop.stats.consecutive_failures == 0
