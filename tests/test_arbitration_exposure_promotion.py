"""Tests for signal arbitration, sector/correlation limits, and the
strategy promotion pipeline."""

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from backtesting.metrics import MetricsResult
from backtesting.walk_forward import DegradationReport
from data.models import Instrument, MarketSnapshot
from execution.execution_models import OrderIntent, OrderSide, OrderType
from portfolio.arbitration import ArbitrationOutcome, SignalArbitrator
from portfolio.portfolio_manager import AccountState, PortfolioManager
from portfolio.positions import Position
from risk.decisions import RejectionReason
from risk.exposure_manager import (
    CorrelationMatrix,
    ExposureManager,
    InstrumentMetadata,
    MetadataRegistry,
    UNKNOWN_SECTOR,
)
from risk.kill_switch import KillSwitch, TradingHalt
from risk.risk_engine import RiskEngine, RiskEngineLimits
from strategies.base import Signal, SignalDirection
from strategies.promotion import (
    GateCriteria,
    HumanApproval,
    PaperResults,
    PromotionPipeline,
    PromotionRefused,
    PromotionStage,
    ResearchProposal,
)

AAPL = Instrument(symbol="AAPL")
MSFT = Instrument(symbol="MSFT")
XOM = Instrument(symbol="XOM")


def intent(side=OrderSide.BUY, qty="100", instrument=AAPL, source="momentum"):
    return OrderIntent(
        instrument=instrument,
        side=side,
        quantity=Decimal(qty),
        order_type=OrderType.MARKET,
        stop_loss=Decimal("95") if side is OrderSide.BUY else Decimal("105"),
        source=source,
        strategy=source,
    )


def signal(direction=SignalDirection.LONG, strategy="momentum", strength=0.7):
    return Signal(
        instrument=AAPL, direction=direction, strength=strength, strategy=strategy
    )


# ---- arbitration ------------------------------------------------------------


def test_single_proposal_passes_through():
    result = SignalArbitrator().arbitrate([(signal(), intent())])
    assert len(result.accepted) == 1


def test_opposing_directions_resolve_to_no_trade():
    """Mixed evidence resolves to no action, not to the louder strategy."""
    proposals = [
        (signal(SignalDirection.LONG, "momentum"), intent(OrderSide.BUY)),
        (signal(SignalDirection.SHORT, "mean_reversion"), intent(OrderSide.SELL)),
    ]
    result = SignalArbitrator().arbitrate(proposals)
    assert result.accepted == []
    assert all(
        d.outcome is ArbitrationOutcome.DROPPED_CONFLICT for d in result.decisions
    )


def test_conflict_does_not_pick_higher_confidence():
    """Confidence across different strategies is not calibrated to a
    common scale, so comparing it is comparing different units."""
    proposals = [
        (signal(SignalDirection.LONG, "momentum", strength=0.99), intent(OrderSide.BUY)),
        (signal(SignalDirection.SHORT, "mean_reversion", strength=0.10), intent(OrderSide.SELL)),
    ]
    assert SignalArbitrator().arbitrate(proposals).accepted == []


def test_agreeing_strategies_produce_one_order():
    """Agreement is not a reason to double the position."""
    proposals = [
        (signal(SignalDirection.LONG, "momentum"), intent(OrderSide.BUY, "100")),
        (signal(SignalDirection.LONG, "trend_following"), intent(OrderSide.BUY, "150")),
    ]
    result = SignalArbitrator().arbitrate(proposals)
    assert len(result.accepted) == 1
    assert result.accepted[0].quantity == Decimal("100")  # smallest


def test_exit_beats_entry():
    """Blocking an exit to take an entry turns a manageable loss into an
    unmanageable one."""
    proposals = [
        (signal(SignalDirection.LONG, "momentum"), intent(OrderSide.BUY, "100")),
        (
            signal(SignalDirection.FLAT, "mean_reversion"),
            intent(OrderSide.SELL, "50", source="mean_reversion"),
        ),
    ]
    result = SignalArbitrator().arbitrate(proposals)
    assert len(result.accepted) == 1
    assert result.accepted[0].source == "mean_reversion"
    assert result.accepted[0].quantity == Decimal("50")
    assert any(
        d.outcome is ArbitrationOutcome.DROPPED_SUPERSEDED_BY_EXIT
        for d in result.decisions
    )


def test_reversal_is_trimmed_to_flatten_by_default():
    """Reversing in one order doubles the effective trade size."""
    position = Position(instrument=AAPL, quantity=Decimal("100"), average_cost=Decimal("100"))
    result = SignalArbitrator().arbitrate(
        [(signal(SignalDirection.SHORT), intent(OrderSide.SELL, "250"))],
        positions={str(AAPL): position},
    )
    assert result.accepted[0].quantity == Decimal("100")
    assert result.decisions[0].outcome is ArbitrationOutcome.REDUCED_TO_EXISTING


def test_reversal_allowed_when_configured():
    position = Position(instrument=AAPL, quantity=Decimal("100"), average_cost=Decimal("100"))
    result = SignalArbitrator(allow_position_reversal=True).arbitrate(
        [(signal(SignalDirection.SHORT), intent(OrderSide.SELL, "250"))],
        positions={str(AAPL): position},
    )
    assert result.accepted[0].quantity == Decimal("250")


def test_different_instruments_do_not_conflict():
    proposals = [
        (signal(SignalDirection.LONG, "momentum"), intent(OrderSide.BUY, instrument=AAPL)),
        (signal(SignalDirection.SHORT, "momentum"), intent(OrderSide.SELL, instrument=MSFT)),
    ]
    assert len(SignalArbitrator().arbitrate(proposals).accepted) == 2


def test_arbitration_never_creates_or_enlarges():
    """It can only reduce; the risk engine remains the sole authority."""
    proposals = [
        (signal(SignalDirection.LONG, "a"), intent(OrderSide.BUY, "100")),
        (signal(SignalDirection.LONG, "b"), intent(OrderSide.BUY, "200")),
    ]
    result = SignalArbitrator().arbitrate(proposals)
    assert len(result.accepted) <= len(proposals)
    assert all(i.quantity <= Decimal("200") for i in result.accepted)


# ---- metadata and correlation -------------------------------------------------


def test_unknown_metadata_is_restrictive_not_exempt():
    registry = MetadataRegistry()
    assert registry.sector(Instrument(symbol="NOBODY")) == UNKNOWN_SECTOR


def test_known_metadata_returned():
    registry = MetadataRegistry([InstrumentMetadata(symbol="AAPL", sector="TECH")])
    assert registry.sector(AAPL) == "TECH"


def test_correlation_defaults_to_conservative_when_no_data():
    """Assuming independence is the optimistic assumption."""
    matrix = CorrelationMatrix(default_correlation=0.5)
    assert matrix.correlation("AAPL", "MSFT") == 0.5


def test_correlation_defaults_when_sample_too_small():
    matrix = CorrelationMatrix(min_observations=30, default_correlation=0.5)
    matrix.observe_prices("AAPL", [100.0 + i for i in range(10)])
    matrix.observe_prices("MSFT", [200.0 + i for i in range(10)])
    assert matrix.correlation("AAPL", "MSFT") == 0.5


def test_correlation_detects_perfect_positive():
    matrix = CorrelationMatrix(min_observations=10)
    prices = [100.0 + (i % 7) * 2 for i in range(40)]
    matrix.observe_prices("AAPL", prices)
    matrix.observe_prices("MSFT", [p * 3 for p in prices])
    assert matrix.correlation("AAPL", "MSFT") > 0.95


def test_correlation_detects_negative():
    matrix = CorrelationMatrix(min_observations=10)
    prices = [100.0 + (i % 7) * 2 for i in range(40)]
    matrix.observe_prices("AAPL", prices)
    matrix.observe_prices("HEDGE", [300.0 - p for p in prices])
    assert matrix.correlation("AAPL", "HEDGE") < -0.5


def test_self_correlation_is_one():
    assert CorrelationMatrix().correlation("AAPL", "AAPL") == 1.0


# ---- exposure manager ----------------------------------------------------------


@pytest.fixture
def exposure():
    registry = MetadataRegistry(
        [
            InstrumentMetadata(symbol="AAPL", sector="TECH"),
            InstrumentMetadata(symbol="MSFT", sector="TECH"),
            InstrumentMetadata(symbol="XOM", sector="ENERGY"),
        ]
    )
    return ExposureManager(
        metadata=registry,
        max_sector_exposure=Decimal("0.30"),
        max_unknown_sector_exposure=Decimal("0.15"),
        max_correlated_exposure=Decimal("0.40"),
    )


def test_within_sector_limit_approved(exposure):
    assessment = exposure.evaluate(
        instrument=AAPL,
        additional_quantity=Decimal("100"),
        price=Decimal("100"),
        positions={},
        prices={},
        equity=Decimal("100000"),
    )
    assert assessment.approved


def test_sector_concentration_rejected(exposure):
    """Ten positions in one sector are one position with ten tickets."""
    existing = {
        "MSFT:SMART:USD": Position(
            instrument=MSFT, quantity=Decimal("100"), average_cost=Decimal("250")
        )
    }
    assessment = exposure.evaluate(
        instrument=AAPL,
        additional_quantity=Decimal("100"),
        price=Decimal("100"),
        positions=existing,
        prices={"MSFT:SMART:USD": Decimal("250")},
        equity=Decimal("100000"),
    )
    assert not assessment.approved
    assert assessment.breaches[0].kind == "MAX_SECTOR_EXPOSURE"


def test_different_sector_not_penalised(exposure):
    existing = {
        "MSFT:SMART:USD": Position(
            instrument=MSFT, quantity=Decimal("100"), average_cost=Decimal("250")
        )
    }
    assessment = exposure.evaluate(
        instrument=XOM,
        additional_quantity=Decimal("100"),
        price=Decimal("100"),
        positions=existing,
        prices={"MSFT:SMART:USD": Decimal("250")},
        equity=Decimal("100000"),
    )
    assert assessment.approved


def test_unknown_sector_gets_tighter_limit(exposure):
    """We hold less of what we cannot reason about."""
    unknown = Instrument(symbol="MYSTERY")
    assessment = exposure.evaluate(
        instrument=unknown,
        additional_quantity=Decimal("200"),
        price=Decimal("100"),
        positions={},
        prices={},
        equity=Decimal("100000"),
    )
    assert not assessment.approved
    assert "unclassified" in assessment.breaches[0].detail


def test_correlated_cluster_rejected():
    matrix = CorrelationMatrix(min_observations=10)
    prices = [100.0 + (i % 7) * 2 for i in range(40)]
    matrix.observe_prices("AAPL", prices)
    matrix.observe_prices("MSFT", [p * 3 for p in prices])

    manager = ExposureManager(
        metadata=MetadataRegistry(
            [
                InstrumentMetadata(symbol="AAPL", sector="TECH"),
                InstrumentMetadata(symbol="MSFT", sector="OTHER"),
            ]
        ),
        correlations=matrix,
        max_sector_exposure=Decimal("1.0"),  # isolate the correlation check
        max_correlated_exposure=Decimal("0.20"),
    )
    existing = {
        "MSFT:SMART:USD": Position(
            instrument=MSFT, quantity=Decimal("100"), average_cost=Decimal("150")
        )
    }
    assessment = manager.evaluate(
        instrument=AAPL,
        additional_quantity=Decimal("100"),
        price=Decimal("100"),
        positions=existing,
        prices={"MSFT:SMART:USD": Decimal("150")},
        equity=Decimal("100000"),
    )
    assert not assessment.approved
    assert assessment.breaches[0].kind == "MAX_CORRELATED_EXPOSURE"


def test_zero_equity_rejected(exposure):
    assessment = exposure.evaluate(
        instrument=AAPL,
        additional_quantity=Decimal("100"),
        price=Decimal("100"),
        positions={},
        prices={},
        equity=Decimal("0"),
    )
    assert not assessment.approved


# ---- risk engine integration ----------------------------------------------------


def test_risk_engine_enforces_sector_limit():
    portfolio = PortfolioManager()
    portfolio.update_account(
        AccountState(equity=Decimal("100000"), buying_power=Decimal("200000"))
    )
    from execution.execution_models import Fill, Order

    order = Order(
        intent=OrderIntent(
            instrument=MSFT, side=OrderSide.BUY, quantity=Decimal("100"), source="t"
        )
    )
    portfolio.apply_fill(
        order,
        Fill(
            fill_id="f",
            order_id=order.order_id,
            timestamp=datetime.now(timezone.utc),
            quantity=Decimal("100"),
            price=Decimal("250"),
        ),
    )

    manager = ExposureManager(
        metadata=MetadataRegistry(
            [
                InstrumentMetadata(symbol="AAPL", sector="TECH"),
                InstrumentMetadata(symbol="MSFT", sector="TECH"),
            ]
        ),
        max_sector_exposure=Decimal("0.20"),
    )
    engine = RiskEngine(
        limits=RiskEngineLimits(),
        portfolio=portfolio,
        kill_switch=KillSwitch(),
        trading_halt=TradingHalt(),
        exposure_manager=manager,
    )
    snapshot = MarketSnapshot(
        instrument=AAPL, timestamp=datetime.now(timezone.utc), bid=99.95, ask=100.05, last=100.0
    )
    assessment = engine.evaluate(
        intent(qty="100"),
        snapshot=snapshot,
        prices={"MSFT:SMART:USD": Decimal("250")},
    )
    assert not assessment.approved
    assert assessment.reason is RejectionReason.MAX_SECTOR_EXPOSURE_EXCEEDED


def test_risk_engine_without_exposure_manager_still_works():
    """The constraint is optional and additive."""
    portfolio = PortfolioManager()
    portfolio.update_account(
        AccountState(equity=Decimal("100000"), buying_power=Decimal("200000"))
    )
    engine = RiskEngine(
        limits=RiskEngineLimits(),
        portfolio=portfolio,
        kill_switch=KillSwitch(),
        trading_halt=TradingHalt(),
    )
    snapshot = MarketSnapshot(
        instrument=AAPL, timestamp=datetime.now(timezone.utc), bid=99.95, ask=100.05, last=100.0
    )
    assert engine.evaluate(intent(), snapshot=snapshot, prices={}).approved


# ---- promotion pipeline -------------------------------------------------------------


def proposal(name="ai_momentum_v2", version="0.1.0") -> ResearchProposal:
    return ResearchProposal(
        name=name,
        version=version,
        hypothesis="Momentum persists longer in high-volume regimes",
        rationale="Observed in rejection analysis",
        proposed_by="ai",
        code="# inert text, never executed",
    )


def good_metrics(**over) -> MetricsResult:
    base = dict(
        n_trades=50, total_return=0.15, sharpe=1.2, max_drawdown=0.10, profit_factor=1.4
    )
    base.update(over)
    return MetricsResult(**base)


def good_degradation() -> DegradationReport:
    return DegradationReport(
        in_sample=MetricsResult(n_trades=50, sharpe=1.3, total_return=0.20),
        out_of_sample=MetricsResult(n_trades=30, sharpe=1.1, total_return=0.12),
    )


def full_approval() -> HumanApproval:
    return HumanApproval(
        approver="risk.officer@example.com",
        rationale="Reviewed OOS results and 30-day paper record; sizing is conservative.",
        reviewed_backtest=True,
        reviewed_paper_results=True,
    )


@pytest.fixture
def pipeline():
    return PromotionPipeline()


def advance_to_paper(pipeline, candidate):
    pipeline.attach_backtest(candidate.candidate_id, good_metrics())
    pipeline.promote(candidate.candidate_id, PromotionStage.BACKTEST)
    pipeline.attach_degradation(candidate.candidate_id, good_degradation())
    pipeline.promote(candidate.candidate_id, PromotionStage.VALIDATION)
    pipeline.attach_paper_results(
        candidate.candidate_id, PaperResults(days_running=30, trades=25)
    )
    pipeline.promote(candidate.candidate_id, PromotionStage.PAPER)


def test_submit_starts_in_research(pipeline):
    candidate = pipeline.submit(proposal())
    assert candidate.stage is PromotionStage.RESEARCH


def test_cannot_skip_stages(pipeline):
    candidate = pipeline.submit(proposal())
    with pytest.raises(PromotionRefused, match="cannot be skipped"):
        pipeline.promote(candidate.candidate_id, PromotionStage.LIVE)
    with pytest.raises(PromotionRefused):
        pipeline.promote(candidate.candidate_id, PromotionStage.PAPER)


def test_backtest_gate_requires_results(pipeline):
    candidate = pipeline.submit(proposal())
    with pytest.raises(PromotionRefused, match="No backtest results"):
        pipeline.promote(candidate.candidate_id, PromotionStage.BACKTEST)


def test_backtest_gate_rejects_too_few_trades(pipeline):
    candidate = pipeline.submit(proposal())
    pipeline.attach_backtest(candidate.candidate_id, good_metrics(n_trades=5))
    with pytest.raises(PromotionRefused, match="Only 5 trades"):
        pipeline.promote(candidate.candidate_id, PromotionStage.BACKTEST)


def test_backtest_gate_rejects_implausible_sharpe(pipeline):
    candidate = pipeline.submit(proposal())
    pipeline.attach_backtest(candidate.candidate_id, good_metrics(sharpe=8.0))
    with pytest.raises(PromotionRefused, match="implausibly high"):
        pipeline.promote(candidate.candidate_id, PromotionStage.BACKTEST)


def test_backtest_gate_rejects_excessive_drawdown(pipeline):
    candidate = pipeline.submit(proposal())
    pipeline.attach_backtest(candidate.candidate_id, good_metrics(max_drawdown=0.60))
    with pytest.raises(PromotionRefused, match="drawdown"):
        pipeline.promote(candidate.candidate_id, PromotionStage.BACKTEST)


def test_validation_gate_rejects_overfit(pipeline):
    """Overfitting is a gate, not advice."""
    candidate = pipeline.submit(proposal())
    pipeline.attach_backtest(candidate.candidate_id, good_metrics())
    pipeline.promote(candidate.candidate_id, PromotionStage.BACKTEST)
    pipeline.attach_degradation(
        candidate.candidate_id,
        DegradationReport(
            in_sample=MetricsResult(n_trades=50, sharpe=2.5, total_return=0.40),
            out_of_sample=MetricsResult(n_trades=30, sharpe=0.2, total_return=0.01),
        ),
    )
    with pytest.raises(PromotionRefused, match="overfitting"):
        pipeline.promote(candidate.candidate_id, PromotionStage.VALIDATION)


def test_paper_gate_requires_minimum_duration(pipeline):
    candidate = pipeline.submit(proposal())
    pipeline.attach_backtest(candidate.candidate_id, good_metrics())
    pipeline.promote(candidate.candidate_id, PromotionStage.BACKTEST)
    pipeline.attach_degradation(candidate.candidate_id, good_degradation())
    pipeline.promote(candidate.candidate_id, PromotionStage.VALIDATION)
    pipeline.attach_paper_results(
        candidate.candidate_id, PaperResults(days_running=2, trades=25)
    )
    with pytest.raises(PromotionRefused, match="days of paper trading"):
        pipeline.promote(candidate.candidate_id, PromotionStage.PAPER)


def test_live_requires_human_approval(pipeline):
    """The AI cannot approve its own strategy."""
    candidate = pipeline.submit(proposal())
    advance_to_paper(pipeline, candidate)
    with pytest.raises(PromotionRefused, match="No human approval"):
        pipeline.promote(candidate.candidate_id, PromotionStage.APPROVED)


def test_incomplete_approval_refused(pipeline):
    candidate = pipeline.submit(proposal())
    advance_to_paper(pipeline, candidate)
    partial = HumanApproval(
        approver="someone", rationale="looks fine", reviewed_backtest=True
    )  # paper results not reviewed
    with pytest.raises(PromotionRefused, match="Approval incomplete"):
        pipeline.promote(candidate.candidate_id, PromotionStage.APPROVED, approval=partial)


def test_full_path_to_live(pipeline):
    candidate = pipeline.submit(proposal())
    advance_to_paper(pipeline, candidate)
    pipeline.promote(
        candidate.candidate_id, PromotionStage.APPROVED, approval=full_approval()
    )
    pipeline.promote(candidate.candidate_id, PromotionStage.LIVE)
    assert candidate.is_live
    assert candidate.approval.approver == "risk.officer@example.com"


def test_rejection_is_permanent_for_that_version(pipeline):
    """Otherwise a candidate could be retried until noise carried it
    through a gate."""
    candidate = pipeline.submit(proposal())
    pipeline.reject(candidate.candidate_id, "Hypothesis not supported out-of-sample")
    with pytest.raises(PromotionRefused, match="previously rejected"):
        pipeline.submit(proposal())


def test_revised_version_may_be_resubmitted(pipeline):
    candidate = pipeline.submit(proposal(version="0.1.0"))
    pipeline.reject(candidate.candidate_id, "overfit")
    revised = pipeline.submit(proposal(version="0.2.0"))
    assert revised.stage is PromotionStage.RESEARCH


def test_rejected_candidate_cannot_be_promoted(pipeline):
    candidate = pipeline.submit(proposal())
    pipeline.reject(candidate.candidate_id, "bad")
    with pytest.raises(PromotionRefused, match="was rejected"):
        pipeline.promote(candidate.candidate_id, PromotionStage.BACKTEST)


def test_proposal_code_is_never_executed():
    """AI-generated code is inert text. Nothing in the module evaluates it."""
    import inspect

    from strategies import promotion

    source = inspect.getsource(promotion)
    for dangerous in ("exec(", "eval(", "compile(", "__import__", "importlib"):
        assert dangerous not in source, dangerous


def test_audit_trail_records_every_gate(pipeline):
    candidate = pipeline.submit(proposal())
    advance_to_paper(pipeline, candidate)
    pipeline.promote(
        candidate.candidate_id, PromotionStage.APPROVED, approval=full_approval()
    )
    trail = candidate.audit_trail()
    for expected in ("RESEARCH", "BACKTEST", "VALIDATION", "PAPER", "risk.officer"):
        assert expected in trail, expected


def test_failed_gate_is_recorded_in_history(pipeline):
    candidate = pipeline.submit(proposal())
    pipeline.attach_backtest(candidate.candidate_id, good_metrics(n_trades=3))
    with pytest.raises(PromotionRefused):
        pipeline.promote(candidate.candidate_id, PromotionStage.BACKTEST)
    assert any(not r.passed for r in candidate.history)
    assert "Only 3 trades" in candidate.audit_trail()


def test_criteria_are_not_ai_tunable():
    """A candidate that can lower its own bar has no bar at all."""
    import inspect

    sig = inspect.signature(PromotionPipeline.promote)
    for forbidden in ("criteria", "thresholds", "override", "force", "skip_gates"):
        assert forbidden not in sig.parameters


# ---- control loop integration ---------------------------------------------------


async def test_loop_arbitrates_conflicting_strategies():
    """Two strategies proposing opposite directions must not both reach
    the risk engine."""
    from datetime import timedelta

    from app.config import TradingMode
    from app.control_loop import ControlLoop, MarketDataFeed
    from app.mode_gate import ModeGate
    from broker.order_manager import OrderManager
    from broker.simulated_broker import SimulatedBrokerGateway
    from data.models import Bar
    from execution.order_store import OrderStore
    from execution.order_validator import OrderValidator
    from execution.reconciliation import Reconciler
    from strategies.base import Strategy
    from strategies.engine import StrategyEngine

    class Always(Strategy):
        def __init__(self, name, direction):
            super().__init__()
            self.name = name
            self._direction = direction

        @property
        def min_bars(self):
            return 1

        def calculate_features(self, context):
            return {"atr": 2.0}

        def generate_signal(self, context):
            return Signal(
                instrument=context.instrument,
                direction=self._direction,
                strength=0.8,
                strategy=self.name,
                features={"atr": 2.0},
            )

    gateway = SimulatedBrokerGateway()
    snapshot = MarketSnapshot(
        instrument=AAPL, timestamp=datetime.now(timezone.utc),
        bid=99.95, ask=100.05, last=100.0,
    )
    gateway.set_snapshot(snapshot)
    store = OrderStore()
    portfolio = PortfolioManager()
    portfolio.update_account(
        AccountState(equity=Decimal("100000"), buying_power=Decimal("200000"))
    )
    ks, halt = KillSwitch(), TradingHalt()
    feed = MarketDataFeed()
    feed.snapshots[str(AAPL)] = snapshot
    base = datetime(2024, 1, 2, tzinfo=timezone.utc)
    feed.bars[str(AAPL)] = [
        Bar(timestamp=base + timedelta(days=i), open=100, high=101, low=99,
            close=100, volume=100000)
        for i in range(30)
    ]

    loop = ControlLoop(
        instruments=[AAPL], feed=feed,
        strategy_engine=StrategyEngine(
            [Always("bull", SignalDirection.LONG), Always("bear", SignalDirection.SHORT)]
        ),
        risk_engine=RiskEngine(
            limits=RiskEngineLimits(), portfolio=portfolio,
            kill_switch=ks, trading_halt=halt,
        ),
        validator=OrderValidator(store),
        order_manager=OrderManager(gateway, store, ModeGate(TradingMode.PAPER)),
        order_store=store, portfolio=portfolio,
        reconciler=Reconciler(store, portfolio),
        kill_switch=ks, trading_halt=halt, mode_gate=ModeGate(TradingMode.PAPER),
        cycle_seconds=0.0, reconcile_every_n_cycles=0,
    )
    await loop.reconcile()
    await loop.run_cycle()

    assert loop.stats.orders_submitted == 0, "conflicting signals must not trade"
    assert store.all_orders() == []
