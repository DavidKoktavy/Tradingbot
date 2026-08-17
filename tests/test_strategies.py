import inspect
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from data.models import Bar, Instrument, MarketSnapshot
from execution.execution_models import OrderIntent, OrderSide
from portfolio.positions import Position
from strategies.base import Signal, SignalDirection, Strategy, StrategyContext
from strategies.engine import StrategyEngine
from strategies.ma_crossover import MACrossoverParams, MACrossoverStrategy
from strategies.mean_reversion import MeanReversionParams, MeanReversionStrategy
from strategies.momentum import MomentumParams, MomentumStrategy
from strategies.registry import StrategyRegistry, registry
from strategies.trend_following import TrendFollowingParams, TrendFollowingStrategy

AAPL = Instrument(symbol="AAPL")


def bars_from(closes: list[float], *, high_pad=1.0, low_pad=1.0) -> list[Bar]:
    base = datetime(2026, 1, 5, tzinfo=timezone.utc)
    return [
        Bar(
            timestamp=base + timedelta(minutes=i),
            open=c,
            high=c + high_pad,
            low=c - low_pad,
            close=c,
            volume=1000,
        )
        for i, c in enumerate(closes)
    ]


def context(closes: list[float], *, position=None, mid=None) -> StrategyContext:
    bars = bars_from(closes)
    price = mid if mid is not None else closes[-1]
    return StrategyContext(
        instrument=AAPL,
        bars=bars,
        snapshot=MarketSnapshot(
            instrument=AAPL,
            timestamp=datetime.now(timezone.utc),
            bid=price - 0.05,
            ask=price + 0.05,
            last=price,
        ),
        position=position,
        equity=Decimal("100000"),
    )


# ---- registry ------------------------------------------------------------


def test_all_four_strategies_registered():
    for name in ("ma_crossover", "momentum", "mean_reversion", "trend_following"):
        assert name in registry


def test_registry_creates_instances():
    strategy = registry.create("momentum")
    assert isinstance(strategy, MomentumStrategy)


def test_unknown_strategy_raises():
    with pytest.raises(KeyError):
        registry.get("does_not_exist")


def test_name_collision_rejected():
    r = StrategyRegistry()

    class A(MomentumStrategy):
        name = "dupe"

    class B(MomentumStrategy):
        name = "dupe"

    r.register(A)
    with pytest.raises(ValueError, match="collision"):
        r.register(B)


def test_unnamed_strategy_rejected():
    r = StrategyRegistry()

    class Unnamed(MomentumStrategy):
        name = "unnamed"

    with pytest.raises(ValueError):
        r.register(Unnamed)


# ---- parameter validation ------------------------------------------------


def test_ma_params_reject_fast_gte_slow():
    with pytest.raises(ValueError):
        MACrossoverParams(fast_period=50, slow_period=20)


def test_mean_reversion_params_reject_exit_gte_entry():
    with pytest.raises(ValueError):
        MeanReversionParams(entry_z=1.0, exit_z=2.0)


def test_unknown_param_rejected():
    with pytest.raises(Exception):
        MomentumParams(nonexistent_param=5)


# ---- insufficient history -------------------------------------------------


@pytest.mark.parametrize(
    "strategy",
    [
        MACrossoverStrategy(),
        MomentumStrategy(),
        MeanReversionStrategy(),
        TrendFollowingStrategy(),
    ],
)
def test_no_signal_without_enough_history(strategy):
    signal = strategy.generate_signal(context([100.0, 101.0, 102.0]))
    assert signal.direction is SignalDirection.NONE


# ---- MA crossover --------------------------------------------------------


def test_ma_crossover_fires_on_golden_cross():
    strategy = MACrossoverStrategy(MACrossoverParams(fast_period=3, slow_period=10, atr_period=5))
    # Flat, then one up-bar: this is the bar the cross completes on.
    closes = [100.0] * 20 + [101.0]
    signal = strategy.generate_signal(context(closes))
    assert signal.direction is SignalDirection.LONG


def test_ma_crossover_does_not_refire_while_extended():
    strategy = MACrossoverStrategy(MACrossoverParams(fast_period=3, slow_period=10, atr_period=5))
    closes = [100.0] * 20 + [101.0, 103.0, 106.0, 110.0, 115.0, 120.0]
    signal = strategy.generate_signal(context(closes))
    # Cross happened several bars ago; must not fire again.
    assert signal.direction is SignalDirection.NONE


def test_ma_crossover_death_cross_goes_short():
    strategy = MACrossoverStrategy(MACrossoverParams(fast_period=3, slow_period=10, atr_period=5))
    # Uptrend that rolls over: the death cross completes on the last bar.
    closes = [100.0 + i for i in range(15)] + [113.0, 111.0, 108.0]
    signal = strategy.generate_signal(context(closes))
    assert signal.direction is SignalDirection.SHORT


def test_ma_crossover_long_only_mode_exits_instead_of_shorting():
    strategy = MACrossoverStrategy(
        MACrossoverParams(fast_period=3, slow_period=10, atr_period=5, allow_short=False)
    )
    closes = [100.0 + i for i in range(15)] + [113.0, 111.0, 108.0]
    signal = strategy.generate_signal(context(closes))
    assert signal.direction is SignalDirection.FLAT


# ---- momentum ------------------------------------------------------------


def rising_with_pullbacks(n: int = 30) -> list[float]:
    """Uptrend with periodic pullbacks, so RSI stays below extreme levels.
    A monotonic ramp pins RSI at 100 and would be filtered out."""
    out, price = [], 100.0
    for i in range(n):
        price += 1.2 if i % 3 != 2 else -0.5
        out.append(price)
    return out


def test_momentum_long_on_strong_move():
    strategy = MomentumStrategy(
        MomentumParams(lookback=10, entry_threshold=0.02, rsi_period=5, atr_period=5,
                       rsi_overbought=90)
    )
    closes = rising_with_pullbacks()
    signal = strategy.generate_signal(context(closes))
    assert signal.direction is SignalDirection.LONG
    assert "roc" in signal.features


def test_momentum_suppressed_by_overbought_rsi():
    strategy = MomentumStrategy(
        MomentumParams(lookback=10, entry_threshold=0.02, rsi_period=5, atr_period=5,
                       rsi_overbought=60)
    )
    closes = rising_with_pullbacks()
    signal = strategy.generate_signal(context(closes))
    assert signal.direction is SignalDirection.NONE
    assert "overbought" in signal.rationale


def test_momentum_flat_when_below_threshold():
    strategy = MomentumStrategy(MomentumParams(lookback=10, entry_threshold=0.5, atr_period=5))
    closes = [100.0 + i * 0.1 for i in range(30)]
    signal = strategy.generate_signal(context(closes))
    assert signal.direction is SignalDirection.NONE


# ---- mean reversion -------------------------------------------------------


def test_mean_reversion_long_when_stretched_down():
    strategy = MeanReversionStrategy(MeanReversionParams(lookback=20, entry_z=1.5, atr_period=5))
    closes = [100.0 + (1 if i % 2 else -1) for i in range(24)] + [97.0, 95.0, 93.0, 90.0]
    signal = strategy.generate_signal(context(closes))
    assert signal.direction is SignalDirection.LONG


def test_mean_reversion_exits_when_reverted():
    strategy = MeanReversionStrategy(
        MeanReversionParams(lookback=20, entry_z=2.0, exit_z=0.5, atr_period=5)
    )
    position = Position(instrument=AAPL, quantity=Decimal("100"), average_cost=Decimal("95"))
    # Noisy series so std > 0, with the final value sitting on the mean.
    closes = [100.0 + (3 if i % 2 else -3) for i in range(29)] + [100.0]
    signal = strategy.generate_signal(context(closes, position=position))
    assert signal.direction is SignalDirection.FLAT


# ---- trend following -------------------------------------------------------


def test_trend_following_breakout_long():
    strategy = TrendFollowingStrategy(
        TrendFollowingParams(
            entry_lookback=10, exit_lookback=5, trend_ema_period=10, atr_period=5
        )
    )
    closes = [100.0] * 25 + [130.0]
    signal = strategy.generate_signal(context(closes))
    assert signal.direction is SignalDirection.LONG


def test_trend_following_channel_excludes_current_bar():
    """The breakout channel must not include the bar being tested, or a
    breakout could never be detected. This is the classic look-ahead bug."""
    strategy = TrendFollowingStrategy(
        TrendFollowingParams(entry_lookback=10, exit_lookback=5, trend_ema_period=10, atr_period=5)
    )
    closes = [100.0] * 25 + [130.0]
    features = strategy.calculate_features(context(closes))
    # channel_high is from prior bars only, so it excludes the 130 spike.
    assert features["channel_high"] < 130.0
    assert features["close"] == 130.0


def test_trend_filter_blocks_counter_trend_breakout():
    strategy = TrendFollowingStrategy(
        TrendFollowingParams(
            entry_lookback=5, exit_lookback=3, trend_ema_period=20, atr_period=5,
            use_trend_filter=True,
        )
    )
    # Strong downtrend, then a small upward pop that breaks the short channel
    # but remains below the long EMA.
    closes = [200.0 - i * 3 for i in range(30)] + [120.0]
    signal = strategy.generate_signal(context(closes))
    assert signal.direction is not SignalDirection.LONG


# ---- order intent generation ----------------------------------------------


def test_intent_carries_protective_stop_from_atr():
    strategy = MomentumStrategy(
        MomentumParams(lookback=10, entry_threshold=0.02, rsi_period=5, atr_period=5,
                       rsi_overbought=90)
    )
    ctx = context(rising_with_pullbacks())
    signal = strategy.generate_signal(ctx)
    intent = strategy.generate_order_intent(signal, ctx)
    assert intent is not None
    assert intent.side is OrderSide.BUY
    assert intent.stop_loss is not None and intent.stop_loss < Decimal(str(ctx.snapshot.mid))
    assert intent.take_profit is not None and intent.take_profit > Decimal(str(ctx.snapshot.mid))


def test_no_intent_for_non_actionable_signal():
    strategy = MomentumStrategy()
    ctx = context([100.0] * 40)
    signal = strategy.generate_signal(ctx)
    assert strategy.generate_order_intent(signal, ctx) is None


def test_does_not_stack_same_side_entry():
    strategy = MomentumStrategy(
        MomentumParams(lookback=10, entry_threshold=0.02, rsi_period=5, atr_period=5,
                       rsi_overbought=90)
    )
    existing = Position(instrument=AAPL, quantity=Decimal("100"), average_cost=Decimal("100"))
    ctx = context(rising_with_pullbacks(), position=existing)
    signal = strategy.generate_signal(ctx)
    assert strategy.generate_order_intent(signal, ctx) is None


def test_flat_signal_closes_existing_position():
    strategy = MeanReversionStrategy(
        MeanReversionParams(lookback=20, entry_z=2.0, exit_z=0.5, atr_period=5)
    )
    position = Position(instrument=AAPL, quantity=Decimal("100"), average_cost=Decimal("95"))
    closes = [100.0 + (3 if i % 2 else -3) for i in range(29)] + [100.0]
    ctx = context(closes, position=position)
    signal = strategy.generate_signal(ctx)
    intent = strategy.generate_order_intent(signal, ctx)
    assert intent is not None
    assert intent.side is OrderSide.SELL
    assert intent.quantity == Decimal("100")


def test_no_intent_without_market_snapshot():
    strategy = MomentumStrategy()
    ctx = StrategyContext(instrument=AAPL, bars=bars_from([100.0] * 40), equity=Decimal("100000"))
    signal = Signal(instrument=AAPL, direction=SignalDirection.LONG, strategy="momentum")
    assert strategy.generate_order_intent(signal, ctx) is None


# ---- structural: strategies cannot execute ---------------------------------


@pytest.mark.parametrize(
    "cls",
    [MACrossoverStrategy, MomentumStrategy, MeanReversionStrategy, TrendFollowingStrategy],
)
def test_strategy_has_no_execution_capability(cls):
    """A strategy must have no broker, order store, or risk engine access."""
    forbidden = ("submit", "place_order", "cancel", "broker", "ib", "execute")
    members = [m for m in dir(cls) if not m.startswith("__")]
    assert not [m for m in members if any(f in m.lower() for f in forbidden)]


def test_strategy_context_exposes_no_mutation_methods():
    ctx = context([100.0] * 40)
    forbidden = ("submit", "place", "execute", "apply_fill", "update_account")
    members = [m for m in dir(ctx) if not m.startswith("_")]
    assert not [m for m in members if any(f in m.lower() for f in forbidden)]


def test_strategy_module_does_not_import_broker():
    import strategies.base as base_mod
    import strategies.momentum as mom_mod

    for mod in (base_mod, mom_mod):
        src = inspect.getsource(mod)
        assert "import broker" not in src
        assert "from broker" not in src
        assert "ib_async" not in src


def test_strategy_sees_only_own_position():
    """StrategyContext carries a single Position, not the portfolio."""
    fields = StrategyContext.model_fields
    assert "position" in fields
    assert "portfolio" not in fields
    assert "positions" not in fields


# ---- engine ----------------------------------------------------------------


def test_engine_runs_all_strategies():
    engine = StrategyEngine([MomentumStrategy(), MeanReversionStrategy()])
    results = engine.evaluate(context([100.0] * 40))
    assert len(results) == 2


def test_engine_isolates_failing_strategy():
    class Exploding(Strategy):
        name = "exploding"

        @property
        def min_bars(self) -> int:
            return 1

        def calculate_features(self, context):
            return {}

        def generate_signal(self, context):
            raise RuntimeError("boom")

    engine = StrategyEngine([Exploding(), MomentumStrategy()])
    results = engine.evaluate(context([100.0] * 40))
    # The healthy strategy still produced a result.
    assert len(results) == 1


def test_engine_disables_repeatedly_failing_strategy():
    class Exploding(Strategy):
        name = "exploding2"

        @property
        def min_bars(self) -> int:
            return 1

        def calculate_features(self, context):
            return {}

        def generate_signal(self, context):
            raise RuntimeError("boom")

    engine = StrategyEngine([Exploding()], max_consecutive_failures=2)
    ctx = context([100.0] * 40)
    engine.evaluate(ctx)
    assert "exploding2" not in engine.disabled_strategies
    engine.evaluate(ctx)
    assert "exploding2" in engine.disabled_strategies
    assert engine.active_strategies == []


def test_disabled_strategy_can_be_reset_explicitly():
    class Exploding(Strategy):
        name = "exploding3"

        @property
        def min_bars(self) -> int:
            return 1

        def calculate_features(self, context):
            return {}

        def generate_signal(self, context):
            raise RuntimeError("boom")

    engine = StrategyEngine([Exploding()], max_consecutive_failures=1)
    engine.evaluate(context([100.0] * 40))
    assert engine.active_strategies == []
    engine.reset_strategy("exploding3")
    assert len(engine.active_strategies) == 1


# ---- determinism -----------------------------------------------------------


@pytest.mark.parametrize(
    "strategy",
    [MACrossoverStrategy(), MomentumStrategy(), MeanReversionStrategy(), TrendFollowingStrategy()],
)
def test_signals_are_deterministic(strategy):
    ctx = context([100.0 + (i % 7) * 1.5 for i in range(150)])
    first = strategy.generate_signal(ctx)
    second = strategy.generate_signal(ctx)
    assert first.direction == second.direction
    assert first.features == second.features
    assert first.strength == pytest.approx(second.strength)
