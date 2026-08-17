"""Phase 8 tests: mode gating, broker submission, execution handling, and
the autonomous control loop."""

from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from app.config import TradingMode
from app.control_loop import ControlLoop, MarketDataFeed
from app.mode_gate import (
    LIVE_CONFIRMATION_PHRASE,
    IllegalModePromotion,
    LiveTradingNotAuthorised,
    ModeAuthorisation,
    ModeGate,
)
from broker.execution_listener import ExecutionListener
from broker.order_manager import OrderManager, OrderSubmissionError
from broker.simulated_broker import SimulatedBrokerGateway, SimulationConfig
from data.models import Bar, Instrument, MarketSnapshot
from execution.execution_models import (
    Fill,
    Order,
    OrderIntent,
    OrderSide,
    OrderState,
)
from execution.order_store import OrderStore
from execution.order_validator import OrderValidator
from execution.reconciliation import BrokerPosition, Reconciler
from portfolio.portfolio_manager import AccountState, PortfolioManager
from risk.decisions import RiskAssessment
from risk.kill_switch import HaltReason, KillSwitch, KillSwitchTrigger, TradingHalt
from risk.risk_engine import RiskEngine, RiskEngineLimits
from strategies.engine import StrategyEngine
from strategies.ma_crossover import MACrossoverParams, MACrossoverStrategy

AAPL = Instrument(symbol="AAPL")


def full_auth() -> ModeAuthorisation:
    return ModeAuthorisation(
        enable_live_trading=True,
        confirmation_phrase=LIVE_CONFIRMATION_PHRASE,
        operator_acknowledged=True,
    )


def snap(mid: float = 100.0, age: float = 0.0) -> MarketSnapshot:
    return MarketSnapshot(
        instrument=AAPL,
        timestamp=datetime.now(timezone.utc) - timedelta(seconds=age),
        bid=mid - 0.05,
        ask=mid + 0.05,
        last=mid,
    )


def approved_order(quantity: str = "100") -> Order:
    store = OrderStore()
    validator = OrderValidator(store)
    intent = OrderIntent(
        instrument=AAPL,
        side=OrderSide.BUY,
        quantity=Decimal(quantity),
        stop_loss=Decimal("95"),
        source="test",
    )
    assessment = RiskAssessment(
        approved=True,
        approved_quantity=Decimal(quantity),
        requested_quantity=Decimal(quantity),
    )
    return validator.build_order(intent, assessment)


# ---- mode gate -----------------------------------------------------------


def test_default_mode_is_paper():
    assert ModeGate().mode is TradingMode.PAPER


def test_cannot_construct_live_gate_without_authorisation():
    with pytest.raises(LiveTradingNotAuthorised):
        ModeGate(TradingMode.LIVE)


def test_cannot_construct_live_gate_with_partial_authorisation():
    partial = ModeAuthorisation(
        enable_live_trading=True, confirmation_phrase=LIVE_CONFIRMATION_PHRASE
    )  # operator_acknowledged missing
    with pytest.raises(LiveTradingNotAuthorised):
        ModeGate(TradingMode.LIVE, authorisation=partial)


def test_live_gate_with_full_authorisation():
    gate = ModeGate(TradingMode.LIVE, authorisation=full_auth())
    assert gate.is_live
    gate.assert_can_submit()  # must not raise


def test_wrong_confirmation_phrase_rejected():
    auth = ModeAuthorisation(
        enable_live_trading=True,
        confirmation_phrase="I_UNDERSTAND",
        operator_acknowledged=True,
    )
    with pytest.raises(LiveTradingNotAuthorised):
        ModeGate(TradingMode.LIVE, authorisation=auth)


def test_backtest_mode_refuses_broker_submission():
    gate = ModeGate(TradingMode.BACKTEST)
    with pytest.raises(LiveTradingNotAuthorised):
        gate.assert_can_submit()


def test_simulation_mode_refuses_broker_submission():
    with pytest.raises(LiveTradingNotAuthorised):
        ModeGate(TradingMode.SIMULATION).assert_can_submit()


def test_paper_mode_permits_submission():
    ModeGate(TradingMode.PAPER).assert_can_submit()


def test_promotion_must_be_one_step():
    gate = ModeGate(TradingMode.BACKTEST)
    with pytest.raises(IllegalModePromotion):
        gate.promote(TradingMode.LIVE)
    with pytest.raises(IllegalModePromotion):
        gate.promote(TradingMode.PAPER)
    gate.promote(TradingMode.SIMULATION)
    assert gate.mode is TradingMode.SIMULATION


def test_promotion_to_live_requires_authorisation():
    gate = ModeGate(TradingMode.PAPER)
    with pytest.raises(LiveTradingNotAuthorised):
        gate.promote(TradingMode.LIVE)
    assert gate.mode is TradingMode.PAPER  # unchanged


def test_promotion_to_live_with_authorisation():
    gate = ModeGate(TradingMode.PAPER)
    gate.promote(TradingMode.LIVE, authorisation=full_auth())
    assert gate.is_live


def test_promote_takes_no_performance_argument():
    """Good paper results must not be encodable as grounds for promotion."""
    import inspect

    params = inspect.signature(ModeGate.promote).parameters
    for forbidden in ("performance", "metrics", "sharpe", "results", "pnl"):
        assert forbidden not in params


def test_demotion_is_always_permitted():
    gate = ModeGate(TradingMode.LIVE, authorisation=full_auth())
    gate.demote_to_safe("operator intervention")
    assert gate.mode is TradingMode.PAPER


def test_live_is_terminal_for_promotion():
    gate = ModeGate(TradingMode.LIVE, authorisation=full_auth())
    with pytest.raises(IllegalModePromotion):
        gate.promote(TradingMode.LIVE)


# ---- order manager --------------------------------------------------------


@pytest.fixture
def sim_stack():
    gateway = SimulatedBrokerGateway()
    gateway.set_snapshot(snap())
    store = OrderStore()
    portfolio = PortfolioManager()
    portfolio.update_account(
        AccountState(equity=Decimal("100000"), buying_power=Decimal("200000"))
    )
    manager = OrderManager(gateway, store, ModeGate(TradingMode.PAPER))
    return gateway, store, portfolio, manager


async def test_submit_moves_order_to_submitted(sim_stack):
    gateway, store, portfolio, manager = sim_stack
    order = approved_order()
    store.add(order)
    await manager.submit(order)
    assert order.state is OrderState.SUBMITTED
    assert order.broker_order_id is not None


async def test_submit_blocked_in_backtest_mode(sim_stack):
    gateway, store, _, _ = sim_stack
    manager = OrderManager(gateway, store, ModeGate(TradingMode.BACKTEST))
    order = approved_order()
    store.add(order)
    with pytest.raises(LiveTradingNotAuthorised):
        await manager.submit(order)
    assert order.state is OrderState.APPROVED  # untouched


async def test_submit_refuses_non_approved_order(sim_stack):
    gateway, store, _, manager = sim_stack
    order = approved_order()
    order.transition_to(OrderState.SUBMITTED)
    store.add(order)
    with pytest.raises(OrderSubmissionError, match="only APPROVED"):
        await manager.submit(order)


async def test_submission_failure_moves_to_error_and_does_not_retry(sim_stack):
    gateway, store, _, manager = sim_stack
    gateway.config.fail_submission_next_n = 1
    order = approved_order()
    store.add(order)
    with pytest.raises(OrderSubmissionError):
        await manager.submit(order)
    assert order.state is OrderState.ERROR
    assert order.error_message


async def test_cancel_requests_rather_than_asserts(sim_stack):
    gateway, store, _, manager = sim_stack
    order = approved_order()
    store.add(order)
    await manager.submit(order)
    await manager.cancel(order)
    # CANCEL_REQUESTED, not CANCELLED — the broker decides.
    assert order.state in (OrderState.CANCEL_REQUESTED, OrderState.CANCELLED)


async def test_cancel_all_covers_active_orders(sim_stack):
    gateway, store, _, manager = sim_stack
    for _ in range(3):
        order = approved_order()
        store.add(order)
        await manager.submit(order)
    count = await manager.cancel_all()
    assert count == 3


async def test_live_submission_blocked_without_authorisation():
    """Even if a LIVE gate somehow existed unauthorised, submission is
    checked again at the boundary."""
    gate = ModeGate(TradingMode.PAPER)
    gate._mode = TradingMode.LIVE  # simulate corrupted state
    manager = OrderManager(SimulatedBrokerGateway(), OrderStore(), gate)
    order = approved_order()
    with pytest.raises(LiveTradingNotAuthorised):
        await manager.submit(order)


# ---- execution listener ----------------------------------------------------


def make_fill(order: Order, fill_id: str = "f1", qty: str = "100") -> Fill:
    return Fill(
        fill_id=fill_id,
        order_id=order.order_id,
        timestamp=datetime.now(timezone.utc),
        quantity=Decimal(qty),
        price=Decimal("100"),
        commission=Decimal("1"),
    )


async def test_fill_updates_portfolio(sim_stack):
    gateway, store, portfolio, manager = sim_stack
    listener = ExecutionListener(store, portfolio)
    order = approved_order()
    store.add(order)
    order.transition_to(OrderState.SUBMITTED)

    await listener.handle_fill(make_fill(order))
    assert portfolio.get_position(AAPL).quantity == Decimal("100")
    assert order.state is OrderState.FILLED


async def test_duplicate_fill_is_ignored(sim_stack):
    """IBKR re-delivers execution reports; applying twice would double
    the position."""
    gateway, store, portfolio, manager = sim_stack
    listener = ExecutionListener(store, portfolio)
    order = approved_order()
    store.add(order)
    order.transition_to(OrderState.SUBMITTED)

    fill = make_fill(order)
    await listener.handle_fill(fill)
    await listener.handle_fill(fill)  # same fill_id
    assert portfolio.get_position(AAPL).quantity == Decimal("100")
    assert listener.fill_count == 1


async def test_fill_for_unknown_order_trips_kill_switch(sim_stack):
    gateway, store, portfolio, manager = sim_stack
    kill_switch = KillSwitch()
    listener = ExecutionListener(store, portfolio, kill_switch=kill_switch)

    orphan = Fill(
        fill_id="orphan",
        order_id="nonexistent",
        timestamp=datetime.now(timezone.utc),
        quantity=Decimal("100"),
        price=Decimal("100"),
    )
    await listener.handle_fill(orphan)
    assert kill_switch.is_active
    assert portfolio.get_position(AAPL).is_flat  # nothing guessed


async def test_rejection_storm_trips_kill_switch(sim_stack):
    gateway, store, portfolio, manager = sim_stack
    kill_switch = KillSwitch()
    listener = ExecutionListener(
        store, portfolio, kill_switch=kill_switch, rejection_threshold=3
    )

    for i in range(3):
        order = approved_order()
        store.add(order)
        order.transition_to(OrderState.SUBMITTED)
        store.link_broker_id(order.order_id, f"IB-{i}")
        await listener.handle_status(f"IB-{i}", OrderState.REJECTED, message="no permission")

    assert listener.rejection_count == 3
    assert kill_switch.is_active
    assert kill_switch.current_event.trigger is KillSwitchTrigger.ORDER_REJECTION_STORM


async def test_partial_fill_then_completion(sim_stack):
    gateway, store, portfolio, manager = sim_stack
    listener = ExecutionListener(store, portfolio)
    order = approved_order()
    store.add(order)
    order.transition_to(OrderState.SUBMITTED)

    await listener.handle_fill(make_fill(order, "f1", "40"))
    assert order.state is OrderState.PARTIALLY_FILLED
    await listener.handle_fill(make_fill(order, "f2", "60"))
    assert order.state is OrderState.FILLED
    assert portfolio.get_position(AAPL).quantity == Decimal("100")


async def test_illegal_status_transition_is_refused(sim_stack):
    gateway, store, portfolio, manager = sim_stack
    listener = ExecutionListener(store, portfolio)
    order = approved_order()
    store.add(order)
    store.link_broker_id(order.order_id, "IB-1")
    order.transition_to(OrderState.SUBMITTED)
    await listener.handle_fill(make_fill(order))
    assert order.state is OrderState.FILLED

    # Broker sends a stale ACKNOWLEDGED after the fill.
    await listener.handle_status("IB-1", OrderState.ACKNOWLEDGED)
    assert order.state is OrderState.FILLED  # unchanged


# ---- simulated broker -------------------------------------------------------


async def test_simulated_fill_crosses_spread(sim_stack):
    gateway, store, portfolio, manager = sim_stack
    order = approved_order()
    store.add(order)
    broker_id = await gateway.place(order)
    order.transition_to(OrderState.SUBMITTED)

    fill = await gateway.try_fill(broker_id)
    assert fill is not None
    assert fill.price > Decimal("100")  # buy pays up
    assert fill.commission > 0


async def test_simulated_rejection(sim_stack):
    gateway, store, portfolio, manager = sim_stack
    gateway.config.reject_next_n = 1
    order = approved_order()
    store.add(order)
    broker_id = await gateway.place(order)
    order.transition_to(OrderState.SUBMITTED)

    fill = await gateway.try_fill(broker_id)
    assert fill is None
    assert order.state is OrderState.REJECTED


async def test_simulated_partial_fill(sim_stack):
    gateway, store, portfolio, manager = sim_stack
    gateway.config.partial_fill_probability = 1.0
    order = approved_order()
    store.add(order)
    broker_id = await gateway.place(order)
    order.transition_to(OrderState.SUBMITTED)

    fill = await gateway.try_fill(broker_id)
    assert fill is not None
    assert fill.quantity < order.intent.quantity
    assert order.state is OrderState.PARTIALLY_FILLED


async def test_no_snapshot_means_no_fill(sim_stack):
    gateway, store, portfolio, manager = sim_stack
    gateway._snapshots.clear()
    order = approved_order()
    store.add(order)
    broker_id = await gateway.place(order)
    order.transition_to(OrderState.SUBMITTED)
    assert await gateway.try_fill(broker_id) is None


# ---- control loop ------------------------------------------------------------


def make_bars(closes: list[float]) -> list[Bar]:
    base = datetime(2024, 1, 2, tzinfo=timezone.utc)
    return [
        Bar(
            timestamp=base + timedelta(days=i),
            open=c,
            high=c * 1.01,
            low=c * 0.99,
            close=c,
            volume=100000,
        )
        for i, c in enumerate(closes)
    ]


@pytest.fixture
def loop_stack():
    gateway = SimulatedBrokerGateway()
    gateway.set_snapshot(snap())
    store = OrderStore()
    portfolio = PortfolioManager()
    portfolio.update_account(
        AccountState(
            equity=Decimal("100000"), cash=Decimal("100000"), buying_power=Decimal("200000")
        )
    )
    kill_switch = KillSwitch()
    halt = TradingHalt()
    risk = RiskEngine(
        limits=RiskEngineLimits(),
        portfolio=portfolio,
        kill_switch=kill_switch,
        trading_halt=halt,
    )
    validator = OrderValidator(store)
    mode = ModeGate(TradingMode.PAPER)
    manager = OrderManager(gateway, store, mode)
    reconciler = Reconciler(store, portfolio)

    feed = MarketDataFeed()
    feed.snapshots[str(AAPL)] = snap()
    feed.bars[str(AAPL)] = make_bars([100.0] * 25 + [101.0])

    strategies = StrategyEngine(
        [MACrossoverStrategy(MACrossoverParams(fast_period=3, slow_period=10, atr_period=5))]
    )

    loop = ControlLoop(
        instruments=[AAPL],
        feed=feed,
        strategy_engine=strategies,
        risk_engine=risk,
        validator=validator,
        order_manager=manager,
        order_store=store,
        portfolio=portfolio,
        reconciler=reconciler,
        kill_switch=kill_switch,
        trading_halt=halt,
        mode_gate=mode,
        cycle_seconds=0.0,
        reconcile_every_n_cycles=0,
    )
    return loop, gateway, store, portfolio, kill_switch, halt, feed


async def test_loop_starts_halted_pending_reconciliation(loop_stack):
    loop, *_ = loop_stack
    assert not loop.can_trade
    assert HaltReason.STARTUP in loop._halt.reasons


async def test_clean_reconciliation_lifts_startup_halt(loop_stack):
    loop, *_ = loop_stack
    assert await loop.reconcile() is True
    assert loop.can_trade


async def test_dirty_reconciliation_keeps_trading_halted(loop_stack):
    loop, gateway, store, portfolio, *_ = loop_stack
    # Broker reports a position we know nothing about.
    gateway.set_position(
        BrokerPosition(instrument=AAPL, quantity=Decimal("500"), average_cost=Decimal("90"))
    )
    assert await loop.reconcile() is False
    assert not loop.can_trade


async def test_reconciliation_never_places_orders(loop_stack):
    loop, gateway, store, portfolio, *_ = loop_stack
    gateway.set_position(
        BrokerPosition(instrument=AAPL, quantity=Decimal("500"), average_cost=Decimal("90"))
    )
    await loop.reconcile()
    assert store.all_orders() == []


async def test_loop_submits_order_on_signal(loop_stack):
    loop, gateway, store, portfolio, *_ = loop_stack
    await loop.reconcile()
    await loop.run_cycle()
    assert loop.stats.orders_submitted >= 1
    assert store.all_orders()


async def test_kill_switch_stops_new_orders(loop_stack):
    loop, gateway, store, portfolio, kill_switch, *_ = loop_stack
    await loop.reconcile()
    kill_switch.activate(KillSwitchTrigger.MANUAL, "test")
    await loop.run_cycle()
    assert loop.stats.orders_submitted == 0
    assert store.all_orders() == []


async def test_stale_data_halts_trading(loop_stack):
    loop, gateway, store, portfolio, kill_switch, halt, feed = loop_stack
    await loop.reconcile()
    feed.snapshots[str(AAPL)] = snap(age=600)
    await loop.run_cycle()
    assert HaltReason.STALE_MARKET_DATA in halt.reasons
    assert loop.stats.orders_submitted == 0


async def test_loop_survives_cycle_exception(loop_stack):
    loop, *_ = loop_stack
    await loop.reconcile()

    def explode(_context):
        raise RuntimeError("boom")

    loop._strategies.evaluate = explode  # type: ignore[method-assign]
    await loop.run_cycle()  # must not raise
    assert loop.stats.consecutive_failures == 1
    assert loop.is_running is False  # never started, but loop object alive


async def test_repeated_failures_trip_kill_switch(loop_stack):
    loop, gateway, store, portfolio, kill_switch, *_ = loop_stack
    await loop.reconcile()

    def explode(_context):
        raise RuntimeError("boom")

    loop._strategies.evaluate = explode  # type: ignore[method-assign]
    for _ in range(5):
        await loop.run_cycle()
    assert kill_switch.is_active


async def test_broker_disconnect_halts_and_requires_reconcile(loop_stack):
    loop, gateway, store, portfolio, kill_switch, halt, feed = loop_stack
    await loop.reconcile()
    assert loop.can_trade

    await loop.on_broker_disconnect()
    assert not loop.can_trade
    assert HaltReason.BROKER_DISCONNECTED in halt.reasons

    await loop.on_broker_reconnect()
    assert loop.can_trade


async def test_daily_loss_breach_trips_kill_switch(loop_stack):
    loop, gateway, store, portfolio, kill_switch, *_ = loop_stack
    await loop.reconcile()
    portfolio.update_account(
        AccountState(equity=Decimal("97000"), buying_power=Decimal("100000"))
    )
    await loop.run_cycle()
    assert kill_switch.is_active
    assert kill_switch.current_event.trigger is KillSwitchTrigger.DAILY_LOSS_LIMIT


async def test_bounded_run_executes_cycles(loop_stack):
    loop, *_ = loop_stack
    stats = await loop.start(max_cycles=3)
    assert stats.cycles == 3
    assert stats.reconciliations >= 1


async def test_emergency_stop_cancels_orders(loop_stack):
    loop, gateway, store, portfolio, kill_switch, *_ = loop_stack
    await loop.reconcile()
    await loop.run_cycle()
    await loop.emergency_stop("operator pressed the button")
    assert kill_switch.is_active
    assert all(
        o.state in (OrderState.CANCEL_REQUESTED, OrderState.CANCELLED, OrderState.FILLED)
        for o in store.all_orders()
    )


async def test_submission_failure_triggers_reconcile_halt(loop_stack):
    loop, gateway, store, portfolio, kill_switch, halt, feed = loop_stack
    await loop.reconcile()
    gateway.config.fail_submission_next_n = 5
    await loop.run_cycle()
    assert loop.stats.submission_failures >= 1
    assert HaltReason.RECONCILIATION_PENDING in halt.reasons
