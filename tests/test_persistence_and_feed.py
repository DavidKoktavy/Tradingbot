"""Tests for the persistence write path and the live market-data feed."""

from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from app.config import TradingMode
from app.control_loop import ControlLoop, MarketDataFeed
from app.mode_gate import ModeGate
from broker.live_feed import LiveMarketDataFeed
from broker.order_manager import OrderManager
from broker.simulated_broker import SimulatedBrokerGateway
from data.models import Bar, Instrument, MarketSnapshot
from database.repository import Repository
from execution.order_store import OrderStore
from execution.order_validator import OrderValidator
from execution.reconciliation import Reconciler
from monitoring.audit import DecisionRecorder
from monitoring.journal import TradeJournal
from portfolio.portfolio_manager import AccountState, PortfolioManager
from risk.kill_switch import KillSwitch, KillSwitchTrigger, TradingHalt
from risk.risk_engine import RiskEngine, RiskEngineLimits
from strategies.engine import StrategyEngine
from strategies.ma_crossover import MACrossoverParams, MACrossoverStrategy

AAPL = Instrument(symbol="AAPL")


def snap(mid: float = 100.0, age: float = 0.0, ts: datetime | None = None) -> MarketSnapshot:
    return MarketSnapshot(
        instrument=AAPL,
        timestamp=ts or (datetime.now(timezone.utc) - timedelta(seconds=age)),
        bid=mid - 0.05,
        ask=mid + 0.05,
        last=mid,
        volume=1000,
    )


def make_bars(closes: list[float]) -> list[Bar]:
    base = datetime(2024, 1, 2, tzinfo=timezone.utc)
    return [
        Bar(timestamp=base + timedelta(days=i), open=c, high=c * 1.01,
            low=c * 0.99, close=c, volume=200000)
        for i, c in enumerate(closes)
    ]


# ---- journal ----------------------------------------------------------------


@pytest.fixture
def journal(tmp_path):
    repo = Repository(f"sqlite:///{tmp_path}/j.db", mode="PAPER")
    repo.create_schema()
    portfolio = PortfolioManager()
    portfolio.update_account(AccountState(equity=Decimal("100000")))
    recorder = DecisionRecorder(path=tmp_path / "d.jsonl", emit_to_log=False)
    return TradeJournal(recorder, repository=repo, portfolio=portfolio), repo


def test_journal_writes_to_file_immediately(journal, tmp_path):
    j, repo = journal
    record = j.build_record(instrument="AAPL:SMART:USD", cycle=1, outcome="NO_SIGNAL")
    j.record_decision(record)
    assert (tmp_path / "d.jsonl").exists()


def test_journal_queues_database_writes(journal):
    j, repo = journal
    j.record_decision(j.build_record(instrument="AAPL:SMART:USD", cycle=1))
    assert j.pending == 1
    assert j.flush() == 1
    assert j.pending == 0
    assert len(repo.recent_decisions()) == 1


def test_journal_queue_is_bounded(tmp_path):
    """Unbounded buffering during an outage is how a trading process runs
    out of memory."""
    repo = Repository(f"sqlite:///{tmp_path}/b.db", mode="PAPER")
    repo.create_schema()
    j = TradeJournal(
        DecisionRecorder(emit_to_log=False), repository=repo, max_queue=10
    )
    for i in range(50):
        j.record_decision(j.build_record(instrument="AAPL", cycle=i))
    assert j.pending <= 10
    assert j.dropped_writes > 0


def test_journal_without_repository_still_writes_file(tmp_path):
    """The file sink is the more durable of the two: it survives a
    database that will not start."""
    recorder = DecisionRecorder(path=tmp_path / "d.jsonl", emit_to_log=False)
    j = TradeJournal(recorder, repository=None)
    j.record_decision(j.build_record(instrument="AAPL", cycle=1))
    assert (tmp_path / "d.jsonl").exists()
    assert j.pending == 0


def test_flush_failure_does_not_raise(tmp_path):
    class BrokenRepo:
        def save_decision(self, record):
            raise RuntimeError("database is down")

    j = TradeJournal(DecisionRecorder(emit_to_log=False), repository=BrokenRepo())
    j.record_decision(j.build_record(instrument="AAPL", cycle=1))
    assert j.flush() == 0  # must not raise


def test_record_captures_full_chain(journal):
    from ai.schemas import AIDecision, AIDecisionResult
    from execution.execution_models import OrderIntent, OrderSide
    from risk.decisions import RiskAssessment, RiskDecision
    from strategies.base import Signal, SignalDirection

    j, repo = journal
    signal = Signal(
        instrument=AAPL,
        direction=SignalDirection.LONG,
        strength=0.7,
        strategy="momentum",
        rationale="ROC 4%",
    )
    ai_decision = AIDecision(
        action="BUY", symbol="AAPL", confidence=0.8, entry=Decimal("100"),
        stop_loss=Decimal("96"), take_profit=Decimal("108"), reasoning="Trend up",
    )
    intent = OrderIntent(
        instrument=AAPL, side=OrderSide.BUY, quantity=Decimal("1000"),
        stop_loss=Decimal("96"), source="ai", strategy="momentum",
    )
    assessment = RiskAssessment(
        approved=True,
        approved_quantity=Decimal("100"),
        requested_quantity=Decimal("1000"),
        decisions=[RiskDecision.approve("kill_switch"), RiskDecision.approve("spread")],
    )

    record = j.build_record(
        instrument="AAPL:SMART:USD",
        cycle=7,
        snapshot=snap(),
        signals=[signal],
        ai_result=AIDecisionResult.accept(ai_decision),
        intent=intent,
        assessment=assessment,
        outcome="SUBMITTED",
    )
    text = record.explain()
    for expected in ("momentum", "ROC 4%", "Trend up", "kill_switch", "APPROVED 100", "reduced"):
        assert expected in text, expected
    assert record.was_reduced


def test_account_snapshot_persisted(journal):
    j, repo = journal
    j.snapshot_account({"AAPL:SMART:USD": Decimal("100")})
    assert len(repo.equity_curve()) == 1


# ---- loop integration --------------------------------------------------------


@pytest.fixture
def loop_with_journal(tmp_path):
    repo = Repository(f"sqlite:///{tmp_path}/loop.db", mode="PAPER")
    repo.create_schema()
    gateway = SimulatedBrokerGateway()
    gateway.set_snapshot(snap())
    store = OrderStore()
    portfolio = PortfolioManager()
    portfolio.update_account(
        AccountState(equity=Decimal("100000"), cash=Decimal("100000"),
                     buying_power=Decimal("200000"))
    )
    ks, halt = KillSwitch(), TradingHalt()
    risk = RiskEngine(limits=RiskEngineLimits(), portfolio=portfolio,
                      kill_switch=ks, trading_halt=halt)
    mode = ModeGate(TradingMode.PAPER)
    feed = MarketDataFeed()
    feed.snapshots[str(AAPL)] = snap()
    feed.bars[str(AAPL)] = make_bars([100.0] * 20 + [101.0])

    journal = TradeJournal(
        DecisionRecorder(path=tmp_path / "decisions.jsonl", emit_to_log=False),
        repository=repo,
        portfolio=portfolio,
    )
    loop = ControlLoop(
        instruments=[AAPL], feed=feed,
        strategy_engine=StrategyEngine(
            [MACrossoverStrategy(MACrossoverParams(fast_period=3, slow_period=10, atr_period=5))]
        ),
        risk_engine=risk, validator=OrderValidator(store),
        order_manager=OrderManager(gateway, store, mode), order_store=store,
        portfolio=portfolio, reconciler=Reconciler(store, portfolio),
        kill_switch=ks, trading_halt=halt, mode_gate=mode,
        cycle_seconds=0.0, reconcile_every_n_cycles=0, journal=journal,
    )
    return loop, repo, ks, portfolio


async def test_loop_persists_submitted_decision(loop_with_journal):
    loop, repo, ks, portfolio = loop_with_journal
    await loop.reconcile()
    await loop.run_cycle()
    decisions = repo.recent_decisions()
    assert decisions
    assert any(d.outcome == "SUBMITTED" for d in decisions)


async def test_loop_persists_rejections(loop_with_journal):
    """'Why did the agent not trade' must be answerable from the record.

    Uses a gross-exposure limit rather than the daily-loss limit, because
    the latter trips the kill switch and stops the cycle before an intent
    is ever evaluated — a different (also correct) path.
    """
    from risk.risk_engine import RiskEngine as RE

    loop, repo, ks, portfolio = loop_with_journal
    loop._risk = RE(
        limits=RiskEngineLimits(max_gross_exposure=Decimal("0.0001")),
        portfolio=portfolio,
        kill_switch=ks,
        trading_halt=loop._halt,
    )
    await loop.reconcile()
    await loop.run_cycle()
    counts = repo.rejection_counts()
    assert counts, "risk rejections must be persisted"
    assert "MAX_GROSS_EXPOSURE_EXCEEDED" in counts


async def test_validator_rejections_are_also_persisted(loop_with_journal):
    """A duplicate signal is rejected by the validator, not the risk
    engine, and must still appear in the audit trail."""
    loop, repo, ks, portfolio = loop_with_journal
    await loop.reconcile()
    await loop.run_cycle()  # first order goes through
    await loop.run_cycle()  # identical signal -> duplicate
    outcomes = {d.outcome for d in repo.recent_decisions()}
    assert "VALIDATOR_REJECTED" in outcomes


async def test_loop_records_no_signal_cycles(loop_with_journal):
    """A cycle where no strategy proposed anything is still recorded."""
    loop, repo, ks, portfolio = loop_with_journal
    await loop.reconcile()
    # Flat price history produces no crossover, so no signal.
    loop._feed.bars[str(AAPL)] = make_bars([100.0] * 30)
    await loop.run_cycle()
    outcomes = {d.outcome for d in repo.recent_decisions()}
    assert "NO_SIGNAL" in outcomes


async def test_stale_data_cycles_are_recorded(loop_with_journal):
    """A halt on stale data is a decision not to trade, and belongs in
    the audit trail."""
    loop, repo, ks, portfolio = loop_with_journal
    await loop.reconcile()
    loop._feed.snapshots[str(AAPL)] = snap(age=600)
    await loop.run_cycle()
    outcomes = {d.outcome for d in repo.recent_decisions()}
    assert "STALE_DATA" in outcomes


async def test_loop_persists_order_rows(loop_with_journal):
    loop, repo, ks, portfolio = loop_with_journal
    await loop.reconcile()
    await loop.run_cycle()
    orders = repo.open_orders()
    assert orders
    assert orders[0].symbol == "AAPL"
    assert orders[0].trading_mode == "PAPER"


async def test_loop_persists_risk_events(loop_with_journal):
    loop, repo, ks, portfolio = loop_with_journal
    await loop.reconcile()
    portfolio.update_account(
        AccountState(equity=Decimal("97000"), buying_power=Decimal("100000"))
    )
    await loop.run_cycle()
    events = repo.risk_events()
    assert any(e.event_type == "DAILY_LOSS_LIMIT" for e in events)


async def test_database_outage_does_not_stop_loop(tmp_path):
    """Trading continues when the audit database is unavailable."""
    broken = Repository("sqlite:////nonexistent/dir/x.db", mode="PAPER")
    gateway = SimulatedBrokerGateway()
    gateway.set_snapshot(snap())
    store = OrderStore()
    portfolio = PortfolioManager()
    portfolio.update_account(
        AccountState(equity=Decimal("100000"), buying_power=Decimal("200000"))
    )
    ks, halt = KillSwitch(), TradingHalt()
    risk = RiskEngine(limits=RiskEngineLimits(), portfolio=portfolio,
                      kill_switch=ks, trading_halt=halt)
    mode = ModeGate(TradingMode.PAPER)
    feed = MarketDataFeed()
    feed.snapshots[str(AAPL)] = snap()
    feed.bars[str(AAPL)] = make_bars([100.0] * 20 + [101.0])

    loop = ControlLoop(
        instruments=[AAPL], feed=feed,
        strategy_engine=StrategyEngine(
            [MACrossoverStrategy(MACrossoverParams(fast_period=3, slow_period=10, atr_period=5))]
        ),
        risk_engine=risk, validator=OrderValidator(store),
        order_manager=OrderManager(gateway, store, mode), order_store=store,
        portfolio=portfolio, reconciler=Reconciler(store, portfolio),
        kill_switch=ks, trading_halt=halt, mode_gate=mode,
        cycle_seconds=0.0, reconcile_every_n_cycles=0,
        journal=TradeJournal(
            DecisionRecorder(emit_to_log=False), repository=broken, portfolio=portfolio
        ),
    )
    await loop.reconcile()
    await loop.run_cycle()
    assert loop.stats.orders_submitted >= 1  # trading unaffected
    assert loop.stats.consecutive_failures == 0
    assert broken.write_failures > 0  # but the failure is visible


# ---- live feed ----------------------------------------------------------------


class FakeProvider:
    def __init__(self, bars: list[Bar] | None = None) -> None:
        self._bars = bars if bars is not None else make_bars([100.0] * 30)
        self._snapshots: dict[str, MarketSnapshot] = {}
        self.subscribed: list[str] = []

    async def subscribe(self, instrument):
        self.subscribed.append(str(instrument))

    async def unsubscribe(self, instrument):
        self.subscribed.remove(str(instrument))

    def get_snapshot(self, instrument):
        return self._snapshots.get(str(instrument))

    def set_snapshot(self, s):
        self._snapshots[str(s.instrument)] = s

    async def get_historical_bars(self, instrument, *, duration, bar_size, end=None):
        return self._bars

    def snapshot_stream(self, instrument):
        raise NotImplementedError


async def test_feed_warms_up_with_history():
    feed = LiveMarketDataFeed(FakeProvider(), [AAPL], bar_size="1 min")
    await feed.start()
    assert feed.is_warmed_up
    assert len(feed.history(AAPL)) == 30


async def test_feed_refuses_to_start_without_history():
    """Starting with no history means strategies silently emit nothing
    while appearing healthy."""
    feed = LiveMarketDataFeed(FakeProvider(bars=[]), [AAPL])
    with pytest.raises(RuntimeError, match="No historical bars"):
        await feed.start()


async def test_feed_exposes_provider_snapshot():
    provider = FakeProvider()
    feed = LiveMarketDataFeed(provider, [AAPL])
    await feed.start()
    provider.set_snapshot(snap(101.0))
    assert feed.snapshot(AAPL).mid == pytest.approx(101.0)


async def test_partial_bar_is_not_appended():
    """Appending an in-progress bar would give strategies a close price
    that changes underneath them."""
    provider = FakeProvider()
    feed = LiveMarketDataFeed(provider, [AAPL], bar_size="1 min")
    await feed.start()
    before = len(feed.history(AAPL))

    base = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
    for i in range(5):
        feed.ingest(snap(100.0 + i, ts=base + timedelta(seconds=i * 10)))
    assert len(feed.history(AAPL)) == before  # still building


async def test_completed_bar_is_appended_with_correct_ohlc():
    provider = FakeProvider()
    feed = LiveMarketDataFeed(provider, [AAPL], bar_size="1 min")
    await feed.start()
    before = len(feed.history(AAPL))

    base = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
    for price, offset in ((100.0, 0), (103.0, 15), (98.0, 30), (101.0, 45)):
        feed.ingest(snap(price, ts=base + timedelta(seconds=offset)))
    completed = feed.ingest(snap(105.0, ts=base + timedelta(seconds=61)))

    assert completed is not None
    assert len(feed.history(AAPL)) == before + 1
    assert completed.open == pytest.approx(100.0)
    assert completed.high == pytest.approx(103.0)
    assert completed.low == pytest.approx(98.0)
    assert completed.close == pytest.approx(101.0)


async def test_feed_bounds_bar_history():
    provider = FakeProvider()
    feed = LiveMarketDataFeed(provider, [AAPL], bar_size="1 min", max_bars=5)
    await feed.start()
    base = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
    for i in range(20):
        feed.ingest(snap(100.0 + i, ts=base + timedelta(minutes=i)))
    assert len(feed.history(AAPL)) <= 5


async def test_feed_ignores_snapshot_without_price():
    provider = FakeProvider()
    feed = LiveMarketDataFeed(provider, [AAPL])
    await feed.start()
    empty = MarketSnapshot(instrument=AAPL, timestamp=datetime.now(timezone.utc))
    assert feed.ingest(empty) is None


async def test_feed_does_not_hide_stale_data():
    """Staleness is the risk engine's decision; a feed that withheld stale
    data would hide an outage."""
    provider = FakeProvider()
    feed = LiveMarketDataFeed(provider, [AAPL])
    await feed.start()
    provider.set_snapshot(snap(100.0, age=600))
    stale = feed.snapshot(AAPL)
    assert stale is not None
    assert stale.is_stale(5.0)


async def test_feed_unsubscribes_on_stop():
    provider = FakeProvider()
    feed = LiveMarketDataFeed(provider, [AAPL])
    await feed.start()
    assert provider.subscribed == [str(AAPL)]
    await feed.stop()
    assert provider.subscribed == []


# ---- smoke test script safety ---------------------------------------------------


def test_smoke_test_refuses_orders_on_non_paper_ports():
    import inspect
    from pathlib import Path

    source = Path("scripts/smoke_test_ibkr.py").read_text()
    assert "PAPER_PORTS = {7497, 4002}" in source
    assert "is not a known paper port" in source
    # Order placement must be opt-in, not default.
    assert 'action="store_true"' in source
    assert "--place-test-order" in source
