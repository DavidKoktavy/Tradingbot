"""
Performance analysis — the deterministic half of the learning loop.

Design decision: **this module contains no AI call.** Everything here is
plain arithmetic over trade history, for the same reason regime detection
is deterministic (ai/regime_detector.py): it must be reproducible, cheap
enough to run every session, and auditable months later by pointing at the
numbers that produced it. The AI layer (reflection.py) consumes this
module's output; it never replaces it.

The questions this answers:
  - Which strategies are actually making money, net of costs?
  - Is a strategy's performance *degrading* — not just "did it lose
    money recently" but "has its edge measurably declined versus its own
    history"?
  - Is it on a losing streak long enough to be a signal rather than noise?
  - Which risk limits keep rejecting its proposals?

None of these answers changes anything by itself. They are inputs to a
human decision, or to a `ResearchProposal` that must pass the full
promotion pipeline in strategies/promotion.py.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from decimal import Decimal

from backtesting.metrics import TradeRecord


@dataclass
class StrategyStats:
    strategy: str
    n_trades: int = 0
    n_wins: int = 0
    win_rate: float | None = None
    expectancy: float | None = None
    profit_factor: float | None = None
    average_win: float | None = None
    average_loss: float | None = None
    total_pnl: float = 0.0
    total_commission: float = 0.0
    average_bars_held: float | None = None

    def to_kelly_stats(self) -> "KellyStats | None":
        """Convert to the narrow shape KellyPositionSizer.update_stats()
        needs. Returns None when there isn't enough information to size
        with at all (no wins/losses recorded yet), matching the
        fail-closed convention used throughout this module."""
        from risk.kelly_sizer import KellyStats

        if self.average_win is None or self.average_loss is None:
            return None
        return KellyStats(
            n_trades=self.n_trades,
            n_wins=self.n_wins,
            average_win=self.average_win,
            average_loss=self.average_loss,
        )


@dataclass
class DegradationFlag:
    strategy: str
    baseline_expectancy: float
    recent_expectancy: float
    baseline_trades: int
    recent_trades: int
    detail: str


@dataclass
class StreakFlag:
    strategy: str
    consecutive_losses: int
    total_loss: float


@dataclass
class PerformanceReport:
    by_strategy: dict[str, StrategyStats] = field(default_factory=dict)
    degradation: list[DegradationFlag] = field(default_factory=list)
    streaks: list[StreakFlag] = field(default_factory=list)
    rejection_counts: dict[str, int] = field(default_factory=dict)
    total_trades: int = 0
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [f"Trades analysed: {self.total_trades}"]
        for stats in sorted(self.by_strategy.values(), key=lambda s: s.strategy):
            wr = f"{stats.win_rate:.1%}" if stats.win_rate is not None else "n/a"
            pf = f"{stats.profit_factor:.2f}" if stats.profit_factor is not None else "n/a"
            exp = f"{stats.expectancy:.2f}" if stats.expectancy is not None else "n/a"
            lines.append(
                f"  [{stats.strategy}] {stats.n_trades} trades, win_rate={wr}, "
                f"profit_factor={pf}, expectancy={exp}, total_pnl={stats.total_pnl:.2f}"
            )
        for flag in self.degradation:
            lines.append(f"  DEGRADATION [{flag.strategy}]: {flag.detail}")
        for streak in self.streaks:
            lines.append(
                f"  STREAK [{streak.strategy}]: {streak.consecutive_losses} consecutive "
                f"losses totalling {streak.total_loss:.2f}"
            )
        if self.rejection_counts:
            top = sorted(self.rejection_counts.items(), key=lambda kv: -kv[1])[:5]
            lines.append("  Top rejection reasons: " + ", ".join(f"{k}={v}" for k, v in top))
        for w in self.warnings:
            lines.append(f"  WARNING: {w}")
        return "\n".join(lines)


class PerformanceAnalyzer:
    def __init__(
        self,
        *,
        degradation_window: int = 20,
        min_trades_for_degradation: int = 40,
        streak_threshold: int = 4,
        min_trades_for_stats: int = 10,
    ) -> None:
        self._window = degradation_window
        self._min_degradation_trades = min_trades_for_degradation
        self._streak_threshold = streak_threshold
        self._min_stats_trades = min_trades_for_stats

    def analyze(
        self,
        trades: list[TradeRecord],
        *,
        rejection_counts: dict[str, int] | None = None,
    ) -> PerformanceReport:
        report = PerformanceReport(
            total_trades=len(trades), rejection_counts=rejection_counts or {}
        )
        if not trades:
            report.warnings.append("No closed trades to analyse")
            return report

        by_strategy: dict[str, list[TradeRecord]] = defaultdict(list)
        for trade in trades:
            by_strategy[trade.strategy].append(trade)

        for strategy, group in by_strategy.items():
            # Trades are assumed to already be in chronological order per
            # the source (backtest engine emits them that way); sort
            # defensively so degradation/streak logic is correct regardless.
            group = sorted(group, key=lambda t: t.exit_time)
            report.by_strategy[strategy] = self._stats(strategy, group)

            if len(group) >= self._min_degradation_trades:
                flag = self._check_degradation(strategy, group)
                if flag is not None:
                    report.degradation.append(flag)

            streak = self._check_streak(strategy, group)
            if streak is not None:
                report.streaks.append(streak)

            if len(group) < self._min_stats_trades:
                report.warnings.append(
                    f"{strategy}: only {len(group)} trades — stats below "
                    f"{self._min_stats_trades} are not meaningful"
                )

        return report

    # ---- internals ----------------------------------------------------------

    def _stats(self, strategy: str, trades: list[TradeRecord]) -> StrategyStats:
        pnls = [float(t.net_pnl) for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))

        return StrategyStats(
            strategy=strategy,
            n_trades=len(trades),
            n_wins=len(wins),
            win_rate=len(wins) / len(trades) if trades else None,
            expectancy=sum(pnls) / len(pnls) if pnls else None,
            profit_factor=(gross_profit / gross_loss) if gross_loss else None,
            average_win=(sum(wins) / len(wins)) if wins else None,
            average_loss=(sum(losses) / len(losses)) if losses else None,
            total_pnl=sum(pnls),
            total_commission=sum(float(t.commission) for t in trades),
            average_bars_held=(
                sum(t.bars_held for t in trades) / len(trades) if trades else None
            ),
        )

    def _check_degradation(
        self, strategy: str, trades: list[TradeRecord]
    ) -> DegradationFlag | None:
        """Compare the most recent window against everything before it.

        Deliberately NOT compared against a fixed baseline computed once:
        that baseline would go stale. Comparing recent-vs-prior on every
        run means degradation is always measured against the strategy's
        own most recent stable behaviour.
        """
        window = self._window
        if len(trades) < window * 2:
            return None

        recent = trades[-window:]
        baseline = trades[:-window]

        recent_exp = sum(float(t.net_pnl) for t in recent) / len(recent)
        baseline_exp = sum(float(t.net_pnl) for t in baseline) / len(baseline)

        # Flag only real degradation, not noise: a sign flip from positive
        # to negative, or a drop of more than half while staying positive.
        sign_flip = baseline_exp > 0 and recent_exp <= 0
        material_decline = (
            baseline_exp > 0 and recent_exp > 0 and recent_exp < baseline_exp * 0.5
        )
        if not (sign_flip or material_decline):
            return None

        detail = (
            f"expectancy over last {window} trades ({recent_exp:.2f}) has "
            f"{'gone negative' if sign_flip else 'fallen by more than half'} "
            f"versus the prior {len(baseline)} trades ({baseline_exp:.2f})"
        )
        return DegradationFlag(
            strategy=strategy,
            baseline_expectancy=baseline_exp,
            recent_expectancy=recent_exp,
            baseline_trades=len(baseline),
            recent_trades=len(recent),
            detail=detail,
        )

    def _check_streak(self, strategy: str, trades: list[TradeRecord]) -> StreakFlag | None:
        streak = 0
        total = 0.0
        for trade in reversed(trades):
            if trade.net_pnl <= 0:
                streak += 1
                total += float(trade.net_pnl)
            else:
                break
        if streak >= self._streak_threshold:
            return StreakFlag(strategy=strategy, consecutive_losses=streak, total_loss=total)
        return None
