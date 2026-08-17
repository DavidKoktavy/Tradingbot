"""
Backtesting engine.

Design decisions:

- **The backtest runs the real risk engine, real position sizer, real
  order validator, and real portfolio manager.** It does not reimplement
  them. A backtest whose risk logic differs from production is measuring a
  system you will never run, and the divergence is invisible precisely
  because it lives in duplicated code. The only substituted component is
  the broker, replaced by a fill simulator.

- **No look-ahead, structurally.** On bar `i`, strategies see
  `bars[0..i]` — a genuinely truncated list, not a full list with an
  index. A strategy cannot reach past the end of what it was handed. The
  resulting order executes at bar `i + latency_bars` using that bar's
  *open*, never the close of the bar that produced the signal.

- **Deterministic.** No RNG anywhere. Given the same bars, config, and
  seed-free code, the equity curve is byte-identical across runs. There is
  a test asserting this.

- **Pessimistic fills.** Orders cross the spread and pay slippage. Volume
  participation is capped, producing partial fills when an order is large
  relative to the bar, rather than assuming infinite liquidity.

- Equity is marked to market every bar, so the drawdown figure reflects
  what an operator would actually have experienced intraday-to-bar, not
  just closed-trade P&L.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal

import structlog

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
from portfolio.portfolio_manager import AccountState, PortfolioManager
from risk.decisions import RiskAssessment
from risk.kill_switch import KillSwitch, TradingHalt
from risk.risk_engine import RiskEngine, RiskEngineLimits
from strategies.base import Strategy, StrategyContext
from backtesting.costs import CostModel
from backtesting.metrics import (
    PERIODS_PER_YEAR,
    MetricsResult,
    TradeRecord,
    compute_metrics,
)

log = structlog.get_logger(__name__)


@dataclass
class _PendingOrder:
    order: Order
    execute_at_index: int


@dataclass
class _OpenTrade:
    side: OrderSide
    quantity: Decimal
    entry_price: Decimal
    entry_time: datetime
    entry_index: int
    commission: Decimal
    strategy: str


@dataclass
class BacktestResult:
    metrics: MetricsResult
    equity_curve: list[float]
    timestamps: list[datetime]
    trades: list[TradeRecord]
    rejections: dict[str, int] = field(default_factory=dict)
    orders_submitted: int = 0
    orders_filled: int = 0
    orders_partially_filled: int = 0
    bars_processed: int = 0

    def summary(self) -> str:
        lines = [
            self.metrics.summary(),
            f"Orders: {self.orders_submitted} submitted, {self.orders_filled} filled, "
            f"{self.orders_partially_filled} partial",
        ]
        if self.rejections:
            top = sorted(self.rejections.items(), key=lambda kv: -kv[1])
            lines.append("Risk rejections: " + ", ".join(f"{k}={v}" for k, v in top))
        return "\n".join(lines)


class BacktestEngine:
    def __init__(
        self,
        *,
        strategy: Strategy,
        instrument: Instrument,
        initial_equity: Decimal = Decimal("100000"),
        cost_model: CostModel | None = None,
        risk_limits: RiskEngineLimits | None = None,
        bar_size: str = "1 day",
        warmup_bars: int | None = None,
    ) -> None:
        self._strategy = strategy
        self._instrument = instrument
        self._initial_equity = initial_equity
        self._costs = cost_model or CostModel()
        self._risk_limits = risk_limits or RiskEngineLimits()
        self._bar_size = bar_size
        self._warmup = warmup_bars if warmup_bars is not None else strategy.min_bars

    def run(self, bars: list[Bar]) -> BacktestResult:
        if len(bars) <= self._warmup + self._costs.latency_bars:
            raise ValueError(
                f"Need more than {self._warmup + self._costs.latency_bars} bars; got {len(bars)}"
            )

        portfolio = PortfolioManager(start_of_day_equity=self._initial_equity)
        portfolio.update_account(
            AccountState(
                equity=self._initial_equity,
                cash=self._initial_equity,
                buying_power=self._initial_equity,
            )
        )
        risk_engine = RiskEngine(
            limits=self._risk_limits,
            portfolio=portfolio,
            kill_switch=KillSwitch(),
            trading_halt=TradingHalt(),
        )
        store = OrderStore(dedupe_window_seconds=0)  # dedupe is time-based; bars are the clock
        validator = OrderValidator(store)

        pending: list[_PendingOrder] = []
        open_trade: _OpenTrade | None = None
        trades: list[TradeRecord] = []
        equity_curve: list[float] = []
        timestamps: list[datetime] = []
        rejections: dict[str, int] = {}
        bars_in_market = 0
        traded_notional = Decimal("0")
        counters = {"submitted": 0, "filled": 0, "partial": 0}

        for i, bar in enumerate(bars):
            # 1. Execute orders whose latency has elapsed. Fills happen at
            #    THIS bar's open — information available at this point.
            still_pending: list[_PendingOrder] = []
            for p in pending:
                if p.execute_at_index > i:
                    still_pending.append(p)
                    continue
                filled_qty, fill_price, commission = self._simulate_fill(p.order, bar)
                if filled_qty <= 0:
                    p.order.transition_to(OrderState.CANCELLED)
                    continue

                fill = Fill(
                    fill_id=f"bt-{i}-{p.order.order_id[:8]}",
                    order_id=p.order.order_id,
                    timestamp=bar.timestamp,
                    quantity=filled_qty,
                    price=fill_price,
                    commission=commission,
                )
                p.order.apply_fill(fill)
                position_before = portfolio.get_position(self._instrument).quantity
                portfolio.apply_fill(p.order, fill)
                traded_notional += filled_qty * fill_price

                if p.order.state is OrderState.FILLED:
                    counters["filled"] += 1
                else:
                    counters["partial"] += 1
                    # Remaining quantity is abandoned rather than carried:
                    # a resting order across bars needs an order book model
                    # we don't have. Documented as a known limitation.
                    p.order.transition_to(OrderState.CANCELLED)

                open_trade, closed = self._update_trade(
                    open_trade,
                    p.order,
                    fill,
                    position_before,
                    portfolio.get_position(self._instrument).quantity,
                    i,
                    bar,
                )
                if closed is not None:
                    trades.append(closed)

            pending = still_pending

            # 2. Mark to market.
            position = portfolio.get_position(self._instrument)
            price = Decimal(str(bar.close))
            equity = (
                self._initial_equity
                + portfolio.realized_pnl
                + position.unrealized_pnl(price)
                - position.total_commission
            )
            portfolio.update_account(
                AccountState(equity=equity, cash=equity, buying_power=equity)
            )
            risk_engine.update_peak_equity(equity)
            equity_curve.append(float(equity))
            timestamps.append(bar.timestamp)
            if not position.is_flat:
                bars_in_market += 1

            # 3. Generate signals — only from bars up to and including i.
            if i < self._warmup or i + self._costs.latency_bars >= len(bars):
                continue

            context = StrategyContext(
                instrument=self._instrument,
                bars=bars[: i + 1],  # genuinely truncated: no look-ahead possible
                snapshot=self._snapshot_from_bar(bar),
                position=position,
                equity=equity,
            )
            try:
                signal = self._strategy.generate_signal(context)
                intent = (
                    self._strategy.generate_order_intent(signal, context)
                    if signal.is_actionable
                    else None
                )
            except Exception as exc:  # noqa: BLE001
                log.error("backtest.strategy_error", bar=i, error=str(exc))
                continue

            if intent is None:
                continue

            # 4. Same risk gate as production. `now` is SIMULATION time —
            #    the bar's timestamp — not wall clock. Evaluating staleness
            #    or trading hours against real time would reject every
            #    historical bar and silently produce an empty backtest.
            assessment = risk_engine.evaluate(
                intent,
                snapshot=context.snapshot,
                prices={str(self._instrument): price},
                now=bar.timestamp,
            )
            if not assessment.approved:
                reason = str(assessment.reason) if assessment.reason else "UNKNOWN"
                rejections[reason] = rejections.get(reason, 0) + 1
                continue

            decision = validator.validate(
                intent, assessment, snapshot=context.snapshot, now=bar.timestamp
            )
            if not decision.approved:
                reason = str(decision.reason) if decision.reason else "VALIDATOR"
                rejections[reason] = rejections.get(reason, 0) + 1
                continue

            order = validator.build_order(intent, assessment)
            order.transition_to(OrderState.SUBMITTED)
            counters["submitted"] += 1
            risk_engine.rate_limiter.record(now=bar.timestamp)
            pending.append(
                _PendingOrder(order=order, execute_at_index=i + self._costs.latency_bars)
            )

        metrics = compute_metrics(
            equity_curve=equity_curve,
            trades=trades,
            periods_per_year=PERIODS_PER_YEAR.get(self._bar_size, 252),
            bars_in_market=bars_in_market,
            total_traded_notional=float(traded_notional),
        )
        return BacktestResult(
            metrics=metrics,
            equity_curve=equity_curve,
            timestamps=timestamps,
            trades=trades,
            rejections=rejections,
            orders_submitted=counters["submitted"],
            orders_filled=counters["filled"],
            orders_partially_filled=counters["partial"],
            bars_processed=len(bars),
        )

    # ---- helpers -----------------------------------------------------------

    def _snapshot_from_bar(self, bar: Bar) -> MarketSnapshot:
        """Synthesise a quote from a bar. Uses the assumed spread, since
        OHLCV data carries no bid/ask."""
        price = Decimal(str(bar.close))
        half = self._costs.half_spread(price, None, None)
        return MarketSnapshot(
            instrument=self._instrument,
            timestamp=bar.timestamp,
            bid=float(price - half),
            ask=float(price + half),
            last=bar.close,
            volume=bar.volume,
        )

    def _simulate_fill(self, order: Order, bar: Bar) -> tuple[Decimal, Decimal, Decimal]:
        """Fill at this bar's OPEN, adjusted adversely. Returns
        (filled_quantity, price, commission)."""
        reference = Decimal(str(bar.open))
        requested = order.intent.quantity
        volume = Decimal(str(bar.volume)) if bar.volume else None

        cap = self._costs.max_fillable(volume)
        filled = min(requested, cap) if cap is not None else requested
        filled = filled.quantize(Decimal("1"))
        if filled <= 0:
            return Decimal("0"), Decimal("0"), Decimal("0")

        price = self._costs.fill_price(
            reference_price=reference,
            side=order.intent.side,
            quantity=filled,
            bar_volume=volume,
        )
        commission = self._costs.commission(filled, price)
        return filled, price, commission

    @staticmethod
    def _update_trade(
        open_trade: _OpenTrade | None,
        order: Order,
        fill: Fill,
        position_before: Decimal,
        position_after: Decimal,
        index: int,
        bar: Bar,
    ) -> tuple[_OpenTrade | None, TradeRecord | None]:
        """Track round-trips for trade statistics."""
        side = order.intent.side

        if open_trade is None:
            return (
                _OpenTrade(
                    side=side,
                    quantity=fill.quantity,
                    entry_price=fill.price,
                    entry_time=bar.timestamp,
                    entry_index=index,
                    commission=fill.commission,
                    strategy=order.intent.strategy or order.intent.source,
                ),
                None,
            )

        # Same direction: average in.
        opening_side = OrderSide.BUY if open_trade.side is OrderSide.BUY else OrderSide.SELL
        if side is opening_side:
            total = open_trade.quantity + fill.quantity
            open_trade.entry_price = (
                open_trade.entry_price * open_trade.quantity + fill.price * fill.quantity
            ) / total
            open_trade.quantity = total
            open_trade.commission += fill.commission
            return open_trade, None

        # Opposite direction: close (fully or partially).
        closed_qty = min(open_trade.quantity, fill.quantity)
        direction = Decimal("1") if open_trade.side is OrderSide.BUY else Decimal("-1")
        gross = (fill.price - open_trade.entry_price) * closed_qty * direction
        commission = open_trade.commission + fill.commission
        record = TradeRecord(
            instrument=str(order.intent.instrument),
            strategy=open_trade.strategy,
            entry_time=open_trade.entry_time.isoformat(),
            exit_time=bar.timestamp.isoformat(),
            side=open_trade.side.value,
            quantity=closed_qty,
            entry_price=open_trade.entry_price,
            exit_price=fill.price,
            gross_pnl=gross,
            commission=commission,
            net_pnl=gross - commission,
            bars_held=index - open_trade.entry_index,
        )

        remaining = open_trade.quantity - closed_qty
        if remaining > 0:
            open_trade.quantity = remaining
            open_trade.commission = Decimal("0")
            return open_trade, record
        if position_after != 0:
            # Flipped through zero: a new trade opens in the other direction.
            return (
                _OpenTrade(
                    side=side,
                    quantity=abs(position_after),
                    entry_price=fill.price,
                    entry_time=bar.timestamp,
                    entry_index=index,
                    commission=Decimal("0"),
                    strategy=order.intent.strategy or order.intent.source,
                ),
                record,
            )
        return None, record
