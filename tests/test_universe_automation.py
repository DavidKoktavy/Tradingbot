"""Automatic stock selection, buy-the-dip, and the long-only risk gate.

Ends with an end-to-end test through the real control loop: select a
universe, buy a dip, keep the held stock when it drops out of the ranking,
then exit. The project's worst bugs were only ever caught end to end."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace

import pytest

from ai.decision_engine import AIDecisionEngine
from ai.providers import NullProvider
from app.config import RiskLimits, Settings, TradingMode
from app.control_loop import ControlLoop
from app.dependency_container import build_container
from app.mode_gate import ModeGate
from broker.execution_listener import ExecutionListener
from broker.order_manager import OrderManager
from broker.scanner import IBKRScanner
from broker.simulated_broker import SimulatedBrokerGateway
from data.models import Bar, Instrument, MarketSnapshot
from execution.execution_models import OrderIntent, OrderSide, OrderType
from execution.order_store import OrderStore
from execution.order_validator import OrderValidator
from execution.reconciliation import Reconciler
from portfolio.portfolio_manager import AccountState, PortfolioManager
from risk.decisions import RejectionReason
from risk.kill_switch import KillSwitch, TradingHalt
from risk.risk_engine import RiskEngine, RiskEngineLimits
from strategies.base import SignalDirection, StrategyContext
from strategies.buy_the_dip import BuyTheDipParams, BuyTheDipStrategy
from strategies.engine import StrategyEngine
from strategies.registry import registry
from universe import ApprovedUniverse, UniverseSelector, score_candidate
from universe.refresher import UniverseRefresher

BASE = datetime(2025, 1, 2, tzinfo=UTC)


# ---- bar builders -----------------------------------------------------------


def bars_from(closes: list[float], volume: float = 2_000_000) -> list[Bar]:
    return [
        Bar(
            timestamp=BASE + timedelta(days=i),
            open=c,
            high=c * 1.005,
            low=c * 0.995,
            close=c,
            volume=volume,
        )
        for i, c in enumerate(closes)
    ]


def uptrend(n: int = 260, start: float = 100.0, end: float = 150.0) -> list[float]:
    step = (end - start) / (n - 1)
    return [start + step * i for i in range(n)]


def uptrend_with_dip(drop: float = 0.03, days: int = 3) -> list[float]:
    closes = uptrend()
    for _ in range(days):
        closes.append(closes[-1] * (1 - drop))
    return closes


def downtrend_with_dip() -> list[float]:
    closes = uptrend(start=150.0, end=100.0)
    for _ in range(3):
        closes.append(closes[-1] * 0.97)
    return closes


def snap_for(symbol: str, mid: float) -> MarketSnapshot:
    return MarketSnapshot(
        instrument=Instrument(symbol=symbol),
        timestamp=datetime.now(UTC),
        bid=round(mid - 0.02, 4),
        ask=round(mid + 0.02, 4),
        last=mid,
    )


# ---- long-only risk gate ----------------------------------------------------


def make_risk(*, allow_short: bool = False) -> tuple[RiskEngine, PortfolioManager]:
    portfolio = PortfolioManager()
    portfolio.update_account(
        AccountState(equity=Decimal(100000), cash=Decimal(100000), buying_power=Decimal(200000))
    )
    halt = TradingHalt()
    risk = RiskEngine(
        limits=RiskEngineLimits(allow_short_entries=allow_short),
        portfolio=portfolio,
        kill_switch=KillSwitch(),
        trading_halt=halt,
    )
    return risk, portfolio


AAPL = Instrument(symbol="AAPL")


def sell(qty: str) -> OrderIntent:
    return OrderIntent(
        instrument=AAPL,
        side=OrderSide.SELL,
        quantity=Decimal(qty),
        order_type=OrderType.MARKET,
        source="test",
        strategy="test",
    )


def buy(qty: str) -> OrderIntent:
    return OrderIntent(
        instrument=AAPL,
        side=OrderSide.BUY,
        quantity=Decimal(qty),
        order_type=OrderType.MARKET,
        source="test",
        strategy="test",
    )


def short_entry_decision(risk: RiskEngine, intent: OrderIntent):
    assessment = risk.evaluate(intent, snapshot=snap_for("AAPL", 100.0), prices={})
    return next(d for d in assessment.decisions if d.check_name == "short_entry")


def test_short_entries_disabled_by_default_in_config():
    assert RiskLimits().allow_short_entries is False
    assert RiskEngineLimits.from_config(RiskLimits()).allow_short_entries is False
    assert RiskEngineLimits().allow_short_entries is False


def test_opening_a_short_is_rejected():
    risk, _ = make_risk()
    d = short_entry_decision(risk, sell("10"))
    assert not d.approved
    assert d.reason is RejectionReason.SHORT_SELLING_DISABLED


def test_selling_exactly_the_long_is_allowed():
    """The exit path must never be blocked by the long-only gate."""
    risk, portfolio = make_risk()
    portfolio.get_position(AAPL).quantity = Decimal(10)
    assert short_entry_decision(risk, sell("10")).approved


def test_selling_part_of_the_long_is_allowed():
    risk, portfolio = make_risk()
    portfolio.get_position(AAPL).quantity = Decimal(10)
    assert short_entry_decision(risk, sell("4")).approved


def test_flipping_long_through_zero_is_rejected():
    risk, portfolio = make_risk()
    portfolio.get_position(AAPL).quantity = Decimal(10)
    assert not short_entry_decision(risk, sell("15")).approved


def test_adding_to_an_existing_short_is_rejected():
    risk, portfolio = make_risk()
    portfolio.get_position(AAPL).quantity = Decimal(-5)
    assert not short_entry_decision(risk, sell("1")).approved


def test_covering_an_existing_short_is_allowed():
    risk, portfolio = make_risk()
    portfolio.get_position(AAPL).quantity = Decimal(-5)
    assert short_entry_decision(risk, buy("5")).approved


def test_short_entries_allowed_when_enabled():
    risk, _ = make_risk(allow_short=True)
    assert short_entry_decision(risk, sell("10")).approved


# ---- buy_the_dip strategy ---------------------------------------------------


def ctx(closes: list[float], quantity: str = "0") -> StrategyContext:
    from portfolio.positions import Position

    bars = bars_from(closes)
    position = Position(instrument=AAPL, quantity=Decimal(quantity))
    return StrategyContext(
        instrument=AAPL,
        bars=bars,
        snapshot=snap_for("AAPL", closes[-1]),
        position=position,
        equity=Decimal(100000),
    )


def test_buy_the_dip_is_registered():
    assert "buy_the_dip" in registry


def test_buys_a_dip_in_an_uptrend():
    signal = BuyTheDipStrategy().generate_signal(ctx(uptrend_with_dip()))
    assert signal.direction is SignalDirection.LONG
    assert 0.5 <= signal.strength <= 1.0
    assert signal.features["above_trend"] == 1.0


def test_does_not_buy_a_dip_in_a_downtrend():
    signal = BuyTheDipStrategy().generate_signal(ctx(downtrend_with_dip()))
    assert signal.direction is SignalDirection.NONE
    assert "trend" in signal.rationale


def test_no_dip_no_trade():
    signal = BuyTheDipStrategy().generate_signal(ctx(uptrend()))
    assert signal.direction is SignalDirection.NONE


def test_never_shorts_a_spike():
    closes = uptrend()
    for _ in range(3):
        closes.append(closes[-1] * 1.04)
    signal = BuyTheDipStrategy().generate_signal(ctx(closes))
    assert signal.direction is not SignalDirection.SHORT


def test_exits_on_bounce():
    closes = uptrend_with_dip()
    closes.extend([closes[-1] * 1.04, closes[-1] * 1.04 * 1.04])
    signal = BuyTheDipStrategy().generate_signal(ctx(closes, quantity="10"))
    assert signal.direction is SignalDirection.FLAT
    assert "bounce" in signal.rationale


def test_exits_when_trend_breaks():
    closes = uptrend()
    closes.extend([closes[-1] * (0.9**k) for k in range(1, 6)])
    signal = BuyTheDipStrategy().generate_signal(ctx(closes, quantity="10"))
    assert signal.direction is SignalDirection.FLAT
    assert "trend broken" in signal.rationale


def test_leaves_an_existing_short_alone():
    signal = BuyTheDipStrategy().generate_signal(ctx(uptrend_with_dip(), quantity="-10"))
    assert signal.direction is SignalDirection.NONE


def test_dip_intent_is_a_buy_with_protective_stop():
    strategy = BuyTheDipStrategy()
    context = ctx(uptrend_with_dip())
    intent = strategy.generate_order_intent(strategy.generate_signal(context), context)
    assert intent is not None and intent.side is OrderSide.BUY
    assert intent.stop_loss is not None and intent.stop_loss < Decimal(str(context.last_close))


def test_insufficient_history():
    signal = BuyTheDipStrategy().generate_signal(ctx(uptrend(n=50)))
    assert signal.direction is SignalDirection.NONE


def test_invalid_params_fail_at_construction():
    with pytest.raises(ValueError):
        BuyTheDipParams(entry_rsi=80, exit_rsi=70)


# ---- approved universe ------------------------------------------------------


def test_approved_universe_dedupes_and_uppercases():
    u = ApprovedUniverse(["aapl", "MSFT", "AAPL"])
    assert u.symbols == ("AAPL", "MSFT")
    assert "aapl" in u


def test_approved_universe_rejects_garbage():
    with pytest.raises(ValueError):
        ApprovedUniverse(["AAPL", "DROP TABLE"])
    with pytest.raises(ValueError):
        ApprovedUniverse([])


def test_shipped_approved_universe_loads():
    assert len(ApprovedUniverse.from_file("approved_universe.json")) > 20


# ---- scoring ----------------------------------------------------------------


def test_dip_in_uptrend_is_eligible_and_outscores_no_dip():
    dip = score_candidate("A", bars_from(uptrend_with_dip()))
    flat = score_candidate("B", bars_from(uptrend()))
    assert dip.eligible and flat.eligible
    assert dip.score > flat.score


def test_downtrend_is_ineligible():
    s = score_candidate("A", bars_from(downtrend_with_dip()))
    assert not s.eligible
    assert any("uptrend" in r for r in s.reasons)


def test_illiquid_is_ineligible():
    s = score_candidate("A", bars_from(uptrend_with_dip(), volume=1_000))
    assert not s.eligible
    assert any("dollar volume" in r for r in s.reasons)


def test_short_history_is_ineligible():
    s = score_candidate("A", bars_from(uptrend(n=100)))
    assert not s.eligible


# ---- selector ---------------------------------------------------------------


class FakeBars:
    def __init__(self, data: dict[str, list[Bar]], fail: set[str] | None = None) -> None:
        self.data = data
        self.fail = fail or set()
        self.requests: list[str] = []

    async def get_historical_bars(self, instrument, *, duration, bar_size, end=None):
        self.requests.append(instrument.symbol)
        assert bar_size == "1 day"
        if instrument.symbol in self.fail:
            raise RuntimeError("pacing violation")
        return self.data.get(instrument.symbol, [])


class FakeScanner:
    def __init__(self, hits: list[str] | None = None, fail: bool = False) -> None:
        self.hits = hits or []
        self.fail = fail

    async def scan(self) -> list[str]:
        if self.fail:
            raise RuntimeError("no scanner permissions")
        return list(self.hits)


MARKET = {
    "DIP1": bars_from(uptrend_with_dip(drop=0.04)),
    "DIP2": bars_from(uptrend_with_dip(drop=0.02)),
    "FLAT": bars_from(uptrend()),
    "DOWN": bars_from(downtrend_with_dip()),
}
APPROVED = ApprovedUniverse(["DIP1", "DIP2", "FLAT", "DOWN"])


async def test_selector_ranks_approved_list_without_scanner():
    sel = await UniverseSelector(APPROVED, FakeBars(MARKET), top_n=2).select()
    assert sel.candidate_source == "approved_list"
    assert sel.selected == ["DIP1", "DIP2"]
    assert "DOWN" not in sel.selected


async def test_scanner_hits_are_limited_to_approved_list():
    bars = FakeBars(MARKET)
    scanner = FakeScanner(["GME", "DIP2", "PENNY", "FLAT"])
    sel = await UniverseSelector(APPROVED, bars, scanner=scanner, top_n=5).select()
    assert sel.candidate_source == "scanner"
    assert sel.scanner_rejected == ["GME", "PENNY"]
    assert set(bars.requests) == {"DIP2", "FLAT"}  # never even fetched unapproved names
    assert set(sel.selected) <= {"DIP2", "FLAT"}
    assert "GME" not in sel.symbols


async def test_scanner_failure_falls_back_to_approved_list():
    sel = await UniverseSelector(
        APPROVED, FakeBars(MARKET), scanner=FakeScanner(fail=True), top_n=1
    ).select()
    assert "scanner failed" in sel.candidate_source
    assert sel.selected == ["DIP1"]


async def test_no_approved_scanner_hits_falls_back():
    sel = await UniverseSelector(
        APPROVED, FakeBars(MARKET), scanner=FakeScanner(["GME"]), top_n=1
    ).select()
    assert "no approved scanner hits" in sel.candidate_source
    assert sel.selected == ["DIP1"]


async def test_held_symbols_always_kept():
    sel = await UniverseSelector(APPROVED, FakeBars(MARKET), top_n=1).select(held=["DOWN"])
    assert sel.selected == ["DIP1"]
    assert sel.symbols == ["DOWN", "DIP1"]


async def test_one_history_failure_does_not_stop_selection():
    sel = await UniverseSelector(APPROVED, FakeBars(MARKET, fail={"DIP1"}), top_n=1).select()
    assert sel.selected == ["DIP2"]
    failed = next(s for s in sel.scored if s.symbol == "DIP1")
    assert not failed.eligible and "history request failed" in failed.reasons[0]


async def test_max_candidates_bounds_broker_requests():
    bars = FakeBars(MARKET)
    await UniverseSelector(APPROVED, bars, top_n=1, max_candidates=2).select()
    assert len(bars.requests) == 2


# ---- IBKR scanner adapter -----------------------------------------------------


def scan_row(symbol: str):
    return SimpleNamespace(contractDetails=SimpleNamespace(contract=SimpleNamespace(symbol=symbol)))


class FakeScannerIB:
    def __init__(self, results: dict[str, list] | None = None, fail: set[str] | None = None):
        self.results = results or {}
        self.fail = fail or set()
        self.subscriptions: list = []

    async def reqScannerDataAsync(self, subscription):
        self.subscriptions.append(subscription)
        if subscription.scanCode in self.fail:
            raise RuntimeError("Error 162: no scanner permissions")
        return self.results.get(subscription.scanCode, [])


async def test_scanner_merges_and_dedupes_codes():
    ib = FakeScannerIB(
        {
            "TOP_PERC_LOSE": [scan_row("aapl"), scan_row("MSFT")],
            "MOST_ACTIVE": [scan_row("MSFT"), scan_row("NVDA")],
        }
    )
    assert await IBKRScanner(ib).scan() == ["AAPL", "MSFT", "NVDA"]
    assert ib.subscriptions[0].locationCode == "STK.US.MAJOR"
    assert ib.subscriptions[0].numberOfRows == 50


async def test_scanner_partial_failure_still_returns_hits():
    ib = FakeScannerIB({"MOST_ACTIVE": [scan_row("NVDA")]}, fail={"TOP_PERC_LOSE"})
    assert await IBKRScanner(ib).scan() == ["NVDA"]


async def test_scanner_total_failure_raises_for_selector_fallback():
    ib = FakeScannerIB(fail={"TOP_PERC_LOSE", "MOST_ACTIVE"})
    with pytest.raises(RuntimeError):
        await IBKRScanner(ib).scan()


# ---- container / AI fencing ---------------------------------------------------


def test_empty_auto_universe_does_not_fall_back_to_spy():
    container = build_container(Settings(trading_mode=TradingMode.PAPER), symbols=[])
    assert container.instruments == []


async def test_empty_allowed_set_fences_ai_rather_than_unfencing_it():
    engine = AIDecisionEngine(NullProvider(), allowed_symbols={"AAPL"})
    engine.set_allowed_symbols(set())
    assert engine._allowed == set()


# ---- end to end: select -> buy dip -> keep held -> exit -----------------------


class FakeLiveFeed:
    """Dynamic feed backed by a fake market: add_instrument 'subscribes'
    by copying that symbol's bars and snapshot in, like LiveMarketDataFeed."""

    def __init__(self, market: dict[str, list[Bar]]) -> None:
        self.market = market
        self.snapshots: dict[str, MarketSnapshot] = {}
        self.bars: dict[str, list[Bar]] = {}
        self.subscribed: list[str] = []

    def snapshot(self, instrument):
        return self.snapshots.get(str(instrument))

    def history(self, instrument):
        return self.bars.get(str(instrument), [])

    async def add_instrument(self, instrument):
        bars = self.market[instrument.symbol]
        self.bars[str(instrument)] = list(bars)
        self.snapshots[str(instrument)] = snap_for(instrument.symbol, bars[-1].close)
        self.subscribed.append(instrument.symbol)

    async def remove_instrument(self, instrument):
        self.bars.pop(str(instrument), None)
        self.snapshots.pop(str(instrument), None)
        self.subscribed.remove(instrument.symbol)

    def set_closes(self, symbol: str, closes: list[float]) -> None:
        inst = Instrument(symbol=symbol)
        self.bars[str(inst)] = bars_from(closes)
        self.snapshots[str(inst)] = snap_for(symbol, closes[-1])


@pytest.fixture
def auto_stack():
    market = dict(MARKET)
    gateway = SimulatedBrokerGateway()
    store = OrderStore()
    portfolio = PortfolioManager()
    portfolio.update_account(
        AccountState(equity=Decimal(100000), cash=Decimal(100000), buying_power=Decimal(200000))
    )
    kill_switch, halt = KillSwitch(), TradingHalt()
    risk = RiskEngine(
        limits=RiskEngineLimits(), portfolio=portfolio, kill_switch=kill_switch, trading_halt=halt
    )
    mode = ModeGate(TradingMode.PAPER)
    feed = FakeLiveFeed(market)
    ai = AIDecisionEngine(NullProvider(), allowed_symbols=set())
    loop = ControlLoop(
        instruments=[],
        feed=feed,
        strategy_engine=StrategyEngine([BuyTheDipStrategy()]),
        risk_engine=risk,
        validator=OrderValidator(store),
        order_manager=OrderManager(gateway, store, mode),
        order_store=store,
        portfolio=portfolio,
        reconciler=Reconciler(store, portfolio),
        kill_switch=kill_switch,
        trading_halt=halt,
        mode_gate=mode,
        ai_engine=ai,
        cycle_seconds=0.0,
        reconcile_every_n_cycles=0,
    )
    selector_bars = FakeBars(market)
    selector = UniverseSelector(APPROVED, selector_bars, top_n=1)
    refresher = UniverseRefresher(
        selector,
        loop,
        feed,
        portfolio=portfolio,
        order_store=store,
        ai_engine=ai,
        first_snapshot_timeout=0.0,
    )
    listener = ExecutionListener(store, portfolio)
    return SimpleNamespace(
        loop=loop,
        feed=feed,
        gateway=gateway,
        store=store,
        portfolio=portfolio,
        refresher=refresher,
        listener=listener,
        market=market,
        ai=ai,
        halt=halt,
    )


async def _fill_everything(s) -> None:
    for inst in s.loop.instruments:
        snap = s.feed.snapshot(inst)
        if snap is not None:
            s.gateway.set_snapshot(snap)
            for fill in await s.gateway.fill_all_pending():
                await s.listener.handle_fill(fill)


async def test_end_to_end_select_buy_hold_exit(auto_stack):
    s = auto_stack
    await s.loop.reconcile()

    # 1. Selection picks the deepest dip and the loop starts tracking it.
    await s.refresher.refresh()
    assert [i.symbol for i in s.loop.instruments] == ["DIP1"]
    assert s.ai._allowed == {"DIP1"}

    # 2. The strategy buys it, the risk engine approves, the order fills.
    await s.loop.run_cycle()
    assert s.loop.stats.orders_submitted == 1, s.loop.stats
    order = s.store.all_orders()[0]
    assert order.intent.side is OrderSide.BUY and order.intent.instrument.symbol == "DIP1"
    await _fill_everything(s)
    held = s.portfolio.get_position(Instrument(symbol="DIP1")).quantity
    assert held > 0

    # 3. Next day DIP1 no longer ranks; a different stock does. DIP1 must
    #    stay tracked because we hold it.
    s.market["DIP1"] = bars_from(uptrend())
    s.market["DIP2"] = bars_from(uptrend_with_dip(drop=0.035))
    await s.refresher.refresh()
    tracked = {i.symbol for i in s.loop.instruments}
    assert tracked == {"DIP1", "DIP2"}
    assert "DIP1" in s.feed.subscribed

    # 4. DIP1 bounces: the strategy exits with a SELL the long-only gate allows.
    closes = uptrend_with_dip(drop=0.04)
    closes.extend([closes[-1] * 1.05, closes[-1] * 1.05 * 1.05])
    s.feed.set_closes("DIP1", closes)
    before = len(s.store.all_orders())
    await s.loop.run_cycle()
    new = s.store.all_orders()[before:]
    exits = [o for o in new if o.intent.instrument.symbol == "DIP1"]
    assert exits and exits[0].intent.side is OrderSide.SELL
    assert exits[0].intent.quantity == held
    await _fill_everything(s)
    assert s.portfolio.get_position(Instrument(symbol="DIP1")).is_flat

    # 5. Now flat and unranked, DIP1 is dropped on the next refresh — unless
    #    a protective child order is still working, in which case it stays.
    await s.refresher.refresh()
    working = {o.intent.instrument.symbol for o in s.store.active_orders()}
    tracked = {i.symbol for i in s.loop.instruments}
    assert ("DIP1" in tracked) == ("DIP1" in working)
    assert "DIP2" in tracked


async def test_loop_refuses_to_drop_a_held_instrument(auto_stack):
    s = auto_stack
    inst = Instrument(symbol="DOWN")
    await s.feed.add_instrument(inst)
    s.loop.add_instrument(inst)
    s.portfolio.get_position(inst).quantity = Decimal(10)
    assert s.loop.remove_instrument(inst) is False
    assert inst.symbol in {i.symbol for i in s.loop.instruments}


async def test_sync_held_adopts_reconciled_positions(auto_stack):
    """After a restart, reconciliation adopts broker positions; they must
    be pulled into the loop even before the next full refresh."""
    s = auto_stack
    s.portfolio.get_position(Instrument(symbol="FLAT")).quantity = Decimal(7)
    await s.refresher.sync_held()
    assert "FLAT" in {i.symbol for i in s.loop.instruments}
    assert "FLAT" in s.ai._allowed


async def test_new_symbol_without_snapshot_is_not_added(auto_stack):
    s = auto_stack

    async def no_snapshot_add(instrument):
        s.feed.bars[str(instrument)] = list(s.market[instrument.symbol])
        s.feed.subscribed.append(instrument.symbol)

    s.feed.add_instrument = no_snapshot_add
    await s.refresher.refresh()
    assert s.loop.instruments == []
    assert s.feed.subscribed == []  # unsubscribed again


# ---- regression: strategy exits were rejected by the sizer -------------------


def test_strategy_exit_without_stop_is_approved_in_full():
    """FLAT signals produce exits with no stop_loss. The sizer refuses to
    size a trade of unknown risk, so before the fix every strategy exit
    was rejected and positions only closed via bracket children."""
    risk, portfolio = make_risk()
    portfolio.get_position(AAPL).quantity = Decimal(75)
    a = risk.evaluate(
        sell("75"), snapshot=snap_for("AAPL", 100.0), prices={"AAPL:SMART:USD": Decimal(100)}
    )
    assert a.approved, [(d.check_name, d.detail) for d in a.decisions if not d.approved]
    assert a.approved_quantity == Decimal(75)


def test_exit_is_not_blocked_by_buying_power():
    risk, portfolio = make_risk()
    portfolio.get_position(AAPL).quantity = Decimal(75)
    portfolio.update_account(
        AccountState(equity=Decimal(100000), cash=Decimal(0), buying_power=Decimal(1))
    )
    a = risk.evaluate(
        sell("75"), snapshot=snap_for("AAPL", 100.0), prices={"AAPL:SMART:USD": Decimal(100)}
    )
    assert a.approved


def test_exits_still_respect_the_kill_switch():
    risk, portfolio = make_risk()
    portfolio.get_position(AAPL).quantity = Decimal(75)
    from risk.kill_switch import KillSwitchTrigger

    risk._kill_switch.activate(KillSwitchTrigger.MANUAL, "test")
    a = risk.evaluate(
        sell("75"), snapshot=snap_for("AAPL", 100.0), prices={"AAPL:SMART:USD": Decimal(100)}
    )
    assert not a.approved


def test_new_entry_without_stop_is_still_refused_by_sizer():
    risk, _ = make_risk()
    a = risk.evaluate(buy("10"), snapshot=snap_for("AAPL", 100.0), prices={})
    assert not a.approved
    assert any(d.check_name == "position_sizing" and not d.approved for d in a.decisions)


# ---- CLI: `paper --auto-universe` end to end with a fake IBKR -----------------


class FakeIBKRClient:
    """Stands in for broker.ibkr_client.IBKRClient: scanner, history and
    snapshots from the fake market, no socket."""

    def __init__(self, settings, market_data_type=None, **_):
        self._ib = FakeScannerIB(
            {
                "TOP_PERC_LOSE": [scan_row("GME"), scan_row("DIP1"), scan_row("DIP2")],
                "MOST_ACTIVE": [scan_row("FLAT")],
            }
        )
        self.daily_requests: list[str] = []

    async def connect(self):
        pass

    async def disconnect(self):
        pass

    async def subscribe(self, instrument):
        pass

    async def unsubscribe(self, instrument):
        pass

    def get_snapshot(self, instrument):
        return snap_for(instrument.symbol, MARKET[instrument.symbol][-1].close)

    async def get_historical_bars(self, instrument, *, duration, bar_size, end=None):
        if bar_size == "1 day":
            self.daily_requests.append(instrument.symbol)
        return list(MARKET[instrument.symbol])


def test_paper_auto_universe_cli_path(monkeypatch, tmp_path, capsys):
    import json

    import broker.ibkr_client
    import broker.order_manager
    from app import cli
    from app.config import get_settings

    universe_file = tmp_path / "approved.json"
    universe_file.write_text(json.dumps({"symbols": ["DIP1", "DIP2", "FLAT", "DOWN"]}))

    monkeypatch.setenv("TRADING_MODE", "PAPER")
    get_settings.cache_clear()
    monkeypatch.setattr(broker.ibkr_client, "IBKRClient", FakeIBKRClient)
    monkeypatch.setattr(
        broker.order_manager, "IBKROrderGateway", lambda ib: SimulatedBrokerGateway()
    )

    args = cli.build_parser().parse_args(
        [
            "paper",
            "--auto-universe",
            "--use-scanner",
            "--top-n",
            "1",
            "--strategies",
            "buy_the_dip",
            "--max-cycles",
            "1",
            "--cycle-seconds",
            "0",
            "--universe-file",
            str(universe_file),
        ]
    )
    try:
        assert args.func(args) == 0
    finally:
        get_settings.cache_clear()

    out = capsys.readouterr().out
    assert "Candidate source: scanner" in out
    assert "GME" in out and "ignored" in out
    assert "Selected: ['DIP1']" in out
    assert "Starting control loop: 1 instruments" in out
    assert "Orders submitted: 1" in out


def test_simulate_refuses_auto_universe(capsys):
    from app import cli

    args = cli.build_parser().parse_args(["simulate", "--auto-universe"])
    assert args.func(args) == 2


# ---- two-step exit: cancel broker-side bracket children first -----------------


class _BracketIB:
    """IBLike that records placements and lets tests set child statuses."""

    def __init__(self) -> None:
        self.cancelled: list = []
        self.trades: list = []
        self.client = SimpleNamespace(getReqId=self._next)
        self._id = 900

    def _next(self) -> int:
        self._id += 1
        return self._id

    async def qualifyContractsAsync(self, *contracts):
        return list(contracts)

    def placeOrder(self, contract, order):
        trade = SimpleNamespace(order=order, orderStatus=SimpleNamespace(status="Submitted"))
        self.trades.append(trade)
        return trade

    def cancelOrder(self, order):
        self.cancelled.append(order)
        for t in self.trades:
            if t.order is order:
                t.orderStatus.status = "PendingCancel"


async def _bracket_gateway():
    from broker.order_manager import IBKROrderGateway
    from execution.execution_models import Order, OrderState

    ib = _BracketIB()
    gw = IBKROrderGateway(ib)
    intent = OrderIntent(
        instrument=AAPL,
        side=OrderSide.BUY,
        quantity=Decimal(75),
        order_type=OrderType.MARKET,
        stop_loss=Decimal(95),
        take_profit=Decimal(110),
        source="buy_the_dip",
        strategy="buy_the_dip",
    )
    order = Order(intent=intent)
    order.transition_to(OrderState.VALIDATING)
    order.transition_to(OrderState.APPROVED)
    await gw.place(order)
    return gw, ib


async def test_gateway_reports_and_cancels_working_children():
    gw, ib = await _bracket_gateway()
    assert gw.working_protective_orders(AAPL) == 2
    assert gw.working_protective_orders(Instrument(symbol="MSFT")) == 0
    assert await gw.cancel_protective_orders(AAPL) == 2
    assert len(ib.cancelled) == 2


async def test_pending_cancel_still_counts_as_working():
    """ib_async's isActive() is False for PendingCancel, but a pending
    cancel can still lose the race with a fill."""
    gw, ib = await _bracket_gateway()
    await gw.cancel_protective_orders(AAPL)
    assert gw.working_protective_orders(AAPL) == 2
    for t in ib.trades[1:]:
        t.orderStatus.status = "Cancelled"
    assert gw.working_protective_orders(AAPL) == 0


class _ChildAwareSimGateway(SimulatedBrokerGateway):
    """Simulated gateway with N protective children working per symbol."""

    def __init__(self) -> None:
        super().__init__()
        self.children: dict[str, int] = {}
        self.cancel_requests: list[str] = []

    def working_protective_orders(self, instrument) -> int:
        return self.children.get(str(instrument), 0)

    async def cancel_protective_orders(self, instrument) -> int:
        self.cancel_requests.append(str(instrument))
        return self.children.get(str(instrument), 0)


async def test_strategy_exit_waits_for_protective_cancel(auto_stack):
    s = auto_stack
    gw = _ChildAwareSimGateway()
    s.loop._orders._gateway = gw  # noqa: SLF001
    inst = Instrument(symbol="DIP1")
    await s.feed.add_instrument(inst)
    s.loop.add_instrument(inst)
    s.portfolio.get_position(inst).quantity = Decimal(75)
    s.portfolio.get_position(inst).average_cost = Decimal(130)
    from execution.reconciliation import BrokerPosition

    gw.set_position(
        BrokerPosition(instrument=inst, quantity=Decimal(75), average_cost=Decimal(130))
    )
    closes = uptrend_with_dip(drop=0.04)
    closes.extend([closes[-1] * 1.05, closes[-1] * 1.05 * 1.05])
    s.feed.set_closes("DIP1", closes)
    await s.loop.reconcile()

    gw.children[str(inst)] = 2
    await s.loop.run_cycle()
    assert gw.cancel_requests == [str(inst)]
    assert s.loop.stats.orders_submitted == 0  # exit deferred, not sent

    gw.children[str(inst)] = 0  # broker confirmed the cancels
    await s.loop.run_cycle()
    assert s.loop.stats.orders_submitted == 1
    exit_order = s.store.all_orders()[-1]
    assert exit_order.intent.side is OrderSide.SELL
    assert exit_order.intent.quantity == Decimal(75)
