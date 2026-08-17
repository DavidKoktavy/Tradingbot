"""
The autonomous control loop.

Design decisions:

- **Reconciliation runs before any trading, always.** The loop starts with
  a `STARTUP` halt in place and refuses to lift it until reconciliation
  has completed cleanly. The spec's principle — never assume local state
  equals broker state, always reconcile after restart — is enforced by
  making the halt the default rather than something to remember to set.

- **A dirty reconciliation halts trading rather than "fixing" it.** If the
  reconciler reports anything unexplained, the loop keeps the halt on and
  alerts. It never places compensating trades to reach a desired state.

- **Health failures stop new trades but do not stop the loop.** The loop
  keeps running so it can continue monitoring, reconciling, and managing
  existing positions. Exiting on a health failure would leave open
  positions completely unattended, which is worse than a degraded loop.

- **Every cycle is wrapped.** An exception anywhere inside a cycle is
  caught, logged, counted, and the loop continues. Repeated consecutive
  failures trip the kill switch — a loop that fails every cycle is not a
  loop that should keep trying to trade.

- **The loop is a coordinator, not a decision-maker.** It owns no risk
  logic. Every order still goes strategy/AI -> risk -> validator ->
  order manager.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal

import structlog

from ai.decision_engine import AIDecisionEngine
from ai.macro_context import MacroContextRegistry
from ai.regime_detector import RegimeDetector
from app.mode_gate import ModeGate
from broker.order_manager import OrderManager, OrderSubmissionError
from data.models import Bar, Instrument, MarketSnapshot
from execution.execution_models import OrderIntent
from execution.order_store import OrderStore
from execution.order_validator import OrderValidator
from execution.reconciliation import Reconciler
from monitoring.alerts import (
    AlertManager,
    alert_broker_disconnect,
    alert_daily_loss,
    alert_drawdown,
    alert_kill_switch,
    alert_unexpected_position,
)
from monitoring.journal import TradeJournal
from monitoring.metrics import M, MetricsRegistry
from portfolio.arbitration import SignalArbitrator
from portfolio.portfolio_manager import AccountState, PortfolioManager
from risk.kill_switch import HaltReason, KillSwitch, KillSwitchTrigger, TradingHalt
from risk.risk_engine import RiskEngine
from strategies.base import StrategyContext
from strategies.engine import StrategyEngine

log = structlog.get_logger(__name__)


@dataclass
class LoopStats:
    cycles: int = 0
    consecutive_failures: int = 0
    intents_generated: int = 0
    intents_approved: int = 0
    intents_rejected: int = 0
    orders_submitted: int = 0
    submission_failures: int = 0
    reconciliations: int = 0
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_cycle_at: datetime | None = None

    @property
    def uptime_seconds(self) -> float:
        return (datetime.now(timezone.utc) - self.started_at).total_seconds()


@dataclass
class MarketDataFeed:
    """Minimal read interface the loop needs. Satisfied by the IBKR
    market data service and by test fakes alike."""

    snapshots: dict[str, MarketSnapshot] = field(default_factory=dict)
    bars: dict[str, list[Bar]] = field(default_factory=dict)

    def snapshot(self, instrument: Instrument) -> MarketSnapshot | None:
        return self.snapshots.get(str(instrument))

    def history(self, instrument: Instrument) -> list[Bar]:
        return self.bars.get(str(instrument), [])


class ControlLoop:
    def __init__(
        self,
        *,
        instruments: list[Instrument],
        feed: MarketDataFeed,
        strategy_engine: StrategyEngine,
        risk_engine: RiskEngine,
        validator: OrderValidator,
        order_manager: OrderManager,
        order_store: OrderStore,
        portfolio: PortfolioManager,
        reconciler: Reconciler,
        kill_switch: KillSwitch,
        trading_halt: TradingHalt,
        mode_gate: ModeGate,
        ai_engine: AIDecisionEngine | None = None,
        regime_detector: RegimeDetector | None = None,
        macro_context: MacroContextRegistry | None = None,
        metrics: MetricsRegistry | None = None,
        alerts: AlertManager | None = None,
        journal: TradeJournal | None = None,
        arbitrator: SignalArbitrator | None = None,
        cycle_seconds: float = 5.0,
        max_consecutive_failures: int = 5,
        reconcile_every_n_cycles: int = 60,
    ) -> None:
        self._instruments = instruments
        self._feed = feed
        self._strategies = strategy_engine
        self._risk = risk_engine
        self._validator = validator
        self._orders = order_manager
        self._store = order_store
        self._portfolio = portfolio
        self._reconciler = reconciler
        self._kill_switch = kill_switch
        self._halt = trading_halt
        self._mode = mode_gate
        self._ai = ai_engine
        self._regime = regime_detector or RegimeDetector()
        # Operator-supplied only; see ai/macro_context.py. Empty registry
        # by default, which is equivalent to no macro context at all.
        self._macro = macro_context or MacroContextRegistry()
        # Monitoring is optional and best-effort: a missing or failing
        # metrics/alert sink must never stop the loop from trading.
        self._metrics = metrics or MetricsRegistry()
        self._alerts = alerts or AlertManager()
        # Optional durable audit sink. Absent = in-memory only; the loop
        # trades identically either way.
        self._journal = journal
        # Resolves competing proposals before the risk engine sees them.
        # It can only drop or shrink; the risk engine remains the sole
        # authority on whether anything is permissible.
        self._arbitrator = arbitrator or SignalArbitrator()
        self._cycle_seconds = cycle_seconds
        self._max_failures = max_consecutive_failures
        self._reconcile_interval = reconcile_every_n_cycles

        self._running = False
        self.stats = LoopStats()
        # `_update_portfolio_marks` is synchronous but alerts are async;
        # conditions detected there are buffered and flushed by the cycle.
        self._pending_alerts: list[tuple[object, tuple]] = []

        # Start halted. Trading is only permitted after a clean reconcile.
        self._halt.set(HaltReason.STARTUP, "Awaiting initial reconciliation")

    @property
    def journal(self) -> TradeJournal | None:
        return self._journal

    @property
    def metrics(self) -> MetricsRegistry:
        return self._metrics

    @property
    def alerts(self) -> AlertManager:
        return self._alerts

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def can_trade(self) -> bool:
        return not self._kill_switch.is_active and not self._halt.is_halted

    async def start(self, *, max_cycles: int | None = None) -> LoopStats:
        """Run until stopped, or for a bounded number of cycles (tests)."""
        self._running = True
        log.info(
            "loop.starting",
            mode=self._mode.mode,
            instruments=[str(i) for i in self._instruments],
        )

        await self.reconcile()

        while self._running:
            if max_cycles is not None and self.stats.cycles >= max_cycles:
                break
            await self.run_cycle()
            if max_cycles is None:
                await asyncio.sleep(self._cycle_seconds)

        self._running = False
        log.info("loop.stopped", cycles=self.stats.cycles)
        return self.stats

    def stop(self) -> None:
        self._running = False

    async def run_cycle(self) -> None:
        """One iteration. Never raises: a fault is contained and counted."""
        try:
            self.stats.cycles += 1
            self.stats.last_cycle_at = datetime.now(timezone.utc)
            self._metrics.increment(M.CYCLES)
            self._metrics.set_gauge(
                M.KILL_SWITCH_ACTIVE, 1.0 if self._kill_switch.is_active else 0.0
            )

            if (
                self._reconcile_interval
                and self.stats.cycles % self._reconcile_interval == 0
            ):
                await self.reconcile()

            self._update_portfolio_marks()

            await self._flush_alerts()

            if not self.can_trade:
                log.debug(
                    "loop.trading_paused",
                    kill_switch=self._kill_switch.is_active,
                    halts=list(self._halt.reasons),
                )
                # Flush before returning: the risk event that caused the
                # halt is exactly the record an operator needs after an
                # incident, and an early return would strand it in the
                # queue until the next cycle that happens to complete.
                if self._journal is not None:
                    self._journal.flush()
                self.stats.consecutive_failures = 0
                return

            for instrument in self._instruments:
                await self._process_instrument(instrument)

            await self._flush_alerts()
            if self._journal is not None:
                self._journal.flush()
            self.stats.consecutive_failures = 0

        except Exception as exc:  # noqa: BLE001 — the loop must not die
            self.stats.consecutive_failures += 1
            log.exception(
                "loop.cycle_failed",
                error=str(exc),
                consecutive_failures=self.stats.consecutive_failures,
            )
            if self.stats.consecutive_failures >= self._max_failures:
                self._kill_switch.activate(
                    KillSwitchTrigger.SYSTEM_ERROR,
                    f"{self.stats.consecutive_failures} consecutive cycle failures: {exc}",
                )
            if self._journal is not None:
                try:
                    self._journal.flush()
                except Exception:  # noqa: BLE001
                    pass

    async def _flush_alerts(self) -> None:
        """Deliver alerts queued by synchronous code. Never raises."""
        pending, self._pending_alerts = self._pending_alerts, []
        for fn, args in pending:
            try:
                await fn(self._alerts, *args)
            except Exception as exc:  # noqa: BLE001
                log.error("loop.alert_failed", error=str(exc))

    async def _process_instrument(self, instrument: Instrument) -> None:
        snapshot = self._feed.snapshot(instrument)
        if snapshot is None:
            self._halt.set(
                HaltReason.STALE_MARKET_DATA, f"No snapshot for {instrument}"
            )
            self._journal_decision(instrument=instrument, outcome="NO_MARKET_DATA")
            return

        max_age = self._risk.limits.max_market_data_age_seconds
        if snapshot.is_stale(max_age):
            self._halt.set(
                HaltReason.STALE_MARKET_DATA,
                f"{instrument} data {snapshot.age_seconds():.1f}s old",
            )
            self._journal_decision(
                instrument=instrument, snapshot=snapshot, outcome="STALE_DATA"
            )
            return
        self._halt.clear(HaltReason.STALE_MARKET_DATA)

        bars = self._feed.history(instrument)
        position = self._portfolio.get_position(instrument)
        equity = self._portfolio.account.equity

        context = StrategyContext(
            instrument=instrument,
            bars=bars,
            snapshot=snapshot,
            position=position,
            equity=equity,
        )

        results = self._strategies.evaluate(context)
        proposals = [(s, i) for s, i in results if i is not None]
        signals = [s for s, _ in results]
        regime = self._regime.detect(bars) if bars else None
        ai_result = None

        # The AI is consulted with the strategies' output as context. Its
        # proposal is one more intent, with no special standing.
        if self._ai is not None and self._ai.provider_available:
            macro_factors = self._macro.for_instrument(instrument.symbol)
            ai_result = await self._ai.decide(
                instrument=instrument,
                snapshot=snapshot,
                regime=regime,
                signals=signals,
                position=position,
                equity=equity,
                macro_factors=macro_factors,
            )
            if ai_result.accepted and ai_result.decision is not None:
                ai_intent = AIDecisionEngine.to_order_intent(
                    ai_result.decision,
                    instrument=instrument,
                    snapshot=snapshot,
                    position=position,
                    equity=equity,
                )
                if ai_intent is not None:
                    proposals.append((None, ai_intent))

        # Resolve conflicts BEFORE the risk engine: two strategies
        # proposing opposite directions in one instrument would otherwise
        # both be evaluated and could whipsaw the position within a cycle.
        arbitration = self._arbitrator.arbitrate(
            proposals, positions=self._portfolio.positions
        )
        intents = arbitration.accepted
        for dropped in arbitration.dropped:
            self._metrics.increment(
                "arbitration_dropped_total", outcome=str(dropped.outcome)
            )

        prices = self._current_prices()

        if not intents:
            # Record the cycle even when nothing was proposed: 'why did the
            # agent not trade' is the more common audit question.
            self._journal_decision(
                instrument=instrument,
                snapshot=snapshot,
                regime=regime,
                signals=signals,
                ai_result=ai_result,
                outcome="ARBITRATION_DROPPED" if proposals else "NO_SIGNAL",
            )
            return

        for intent in intents:
            self.stats.intents_generated += 1
            await self._gate_and_submit(
                intent,
                snapshot,
                prices,
                regime=regime,
                signals=signals,
                ai_result=ai_result,
            )

    def _journal_decision(self, **kwargs) -> None:
        """Build and persist a decision record. Never raises: losing the
        audit sink must not stop trading."""
        if self._journal is None:
            return
        try:
            record = self._journal.build_record(
                instrument=str(kwargs.pop("instrument")),
                cycle=self.stats.cycles,
                **kwargs,
            )
            self._journal.record_decision(record)
        except Exception as exc:  # noqa: BLE001
            log.error("loop.journal_failed", error=str(exc))

    async def _gate_and_submit(
        self,
        intent: OrderIntent,
        snapshot: MarketSnapshot,
        prices: dict[str, Decimal],
        *,
        regime: object | None = None,
        signals: list | None = None,
        ai_result: object | None = None,
    ) -> None:
        context = {
            "instrument": intent.instrument,
            "snapshot": snapshot,
            "regime": regime,
            "signals": signals,
            "ai_result": ai_result,
            "intent": intent,
        }

        assessment = self._risk.evaluate(intent, snapshot=snapshot, prices=prices)
        if not assessment.approved:
            self.stats.intents_rejected += 1
            self._metrics.increment(
                M.RISK_REJECTIONS, reason=str(assessment.reason or "UNKNOWN")
            )
            self._journal_decision(
                **context, assessment=assessment, outcome="RISK_REJECTED"
            )
            return

        decision = self._validator.validate(intent, assessment, snapshot=snapshot)
        if not decision.approved:
            self.stats.intents_rejected += 1
            self._metrics.increment(
                M.RISK_REJECTIONS, reason=str(decision.reason or "VALIDATOR")
            )
            self._journal_decision(
                **context, assessment=assessment, outcome="VALIDATOR_REJECTED"
            )
            return

        order = self._validator.build_order(intent, assessment)
        self.stats.intents_approved += 1

        try:
            with self._metrics.timer(M.ORDER_LATENCY):
                await self._orders.submit(order)
            self._risk.rate_limiter.record()
            self.stats.orders_submitted += 1
            self._metrics.increment(M.ORDERS_SUBMITTED, source=intent.source)
            self._journal_decision(
                **context, assessment=assessment, order=order, outcome="SUBMITTED"
            )
            if self._journal is not None:
                self._journal.record_order(order)
        except OrderSubmissionError as exc:
            self.stats.submission_failures += 1
            self._metrics.increment(M.ORDERS_REJECTED, reason="SUBMISSION_FAILED")
            self._journal_decision(
                **context,
                assessment=assessment,
                order=order,
                submission_error=str(exc),
                outcome="SUBMISSION_FAILED",
            )
            log.error("loop.submission_failed", order_id=order.order_id, error=str(exc))
            # The order's fate at the broker is unknown. Reconcile before
            # doing anything else with this instrument.
            self._halt.set(
                HaltReason.RECONCILIATION_PENDING,
                f"Submission failed for {order.order_id}; state unknown",
            )

    async def reconcile(self) -> bool:
        """Reconcile against the broker. Returns True if clean.

        A dirty reconciliation keeps trading halted; it never triggers
        compensating trades.
        """
        self.stats.reconciliations += 1
        try:
            broker_positions = await self._orders._gateway.positions()  # noqa: SLF001
            broker_orders = await self._orders._gateway.open_orders()  # noqa: SLF001
        except Exception as exc:  # noqa: BLE001
            log.error("loop.reconcile_failed", error=str(exc))
            self._halt.set(HaltReason.RECONCILIATION_PENDING, f"Reconcile failed: {exc}")
            return False

        report = self._reconciler.reconcile(
            broker_positions=broker_positions, broker_orders=broker_orders
        )

        if report.requires_halt:
            log.error(
                "loop.reconcile_dirty",
                discrepancies=[d.kind for d in report.discrepancies],
            )
            self._halt.set(
                HaltReason.RECONCILIATION_PENDING,
                f"{len(report.discrepancies)} unresolved discrepancies",
            )
            await alert_unexpected_position(
                self._alerts,
                "; ".join(f"{d.kind}: {d.detail}" for d in report.discrepancies[:3]),
            )
            return False

        self._halt.clear(HaltReason.RECONCILIATION_PENDING)
        self._halt.clear(HaltReason.STARTUP)
        log.info("loop.reconcile_clean")
        return True

    def _current_prices(self) -> dict[str, Decimal]:
        prices: dict[str, Decimal] = {}
        for instrument in self._instruments:
            snapshot = self._feed.snapshot(instrument)
            if snapshot is not None and snapshot.mid is not None:
                prices[str(instrument)] = Decimal(str(snapshot.mid))
        # Any open position without a mark must also be priced, or the
        # risk engine will (correctly) refuse to evaluate exposure.
        for key, position in self._portfolio.positions.items():
            if not position.is_flat and key not in prices:
                log.warning("loop.missing_mark", instrument=key)
        return prices

    def _update_portfolio_marks(self) -> None:
        """Refresh equity from marks and update the drawdown high-water
        mark, then check the daily-loss and drawdown limits."""
        prices = self._current_prices()
        try:
            unrealised = self._portfolio.unrealized_pnl(prices)
        except Exception:  # noqa: BLE001 - missing marks handled below
            return

        account = self._portfolio.account
        equity = account.equity
        self._risk.update_peak_equity(equity)

        try:
            daily_pct = self._portfolio.daily_pnl_pct(prices)
        except Exception:  # noqa: BLE001
            return

        drawdown = self._risk.current_drawdown(equity)
        self._metrics.set_gauge(M.EQUITY, float(equity))
        self._metrics.set_gauge(M.DAILY_PNL, float(daily_pct))
        self._metrics.set_gauge(M.UNREALISED_PNL, float(unrealised))
        self._metrics.set_gauge(M.REALISED_PNL, float(self._portfolio.realized_pnl))
        self._metrics.set_gauge(M.DRAWDOWN, float(drawdown))
        self._metrics.set_gauge(M.OPEN_POSITIONS, self._portfolio.open_position_count)

        if daily_pct <= -self._risk.limits.max_daily_loss:
            self._kill_switch.activate(
                KillSwitchTrigger.DAILY_LOSS_LIMIT,
                f"Daily P&L {daily_pct:.2%} breached limit",
            )
            self._pending_alerts.append(
                (
                    alert_daily_loss,
                    (float(daily_pct), -float(self._risk.limits.max_daily_loss)),
                )
            )
            if self._journal is not None:
                self._journal.record_risk_event(
                    event_type="DAILY_LOSS_LIMIT",
                    severity="CRITICAL",
                    detail=f"Daily P&L {daily_pct:.4%} breached limit",
                    context={"daily_pnl_pct": str(daily_pct)},
                )
        if drawdown >= self._risk.limits.max_portfolio_drawdown:
            self._kill_switch.activate(
                KillSwitchTrigger.MAX_DRAWDOWN, f"Drawdown {drawdown:.2%} breached limit"
            )
            self._pending_alerts.append(
                (
                    alert_drawdown,
                    (float(drawdown), float(self._risk.limits.max_portfolio_drawdown)),
                )
            )

    async def on_broker_disconnect(self) -> None:
        """Hook for the connection manager. Stops new trading immediately
        and requires a reconcile before resuming."""
        self._halt.set(HaltReason.BROKER_DISCONNECTED, "Broker connection lost")
        self._halt.set(HaltReason.RECONCILIATION_PENDING, "Reconcile after reconnect")
        self._metrics.set_gauge(M.IBKR_CONNECTED, 0.0)
        log.error("loop.broker_disconnected")
        await alert_broker_disconnect(self._alerts, "Broker connection lost")

    async def on_broker_reconnect(self) -> None:
        self._halt.clear(HaltReason.BROKER_DISCONNECTED)
        await self.reconcile()

    async def emergency_stop(self, reason: str) -> None:
        """Trip the kill switch and apply the configured emergency policy."""
        from risk.kill_switch import EmergencyPolicy

        self._kill_switch.activate(KillSwitchTrigger.MANUAL, reason)
        await alert_kill_switch(self._alerts, "MANUAL", reason)
        policy = self._kill_switch.emergency_policy
        if policy in (EmergencyPolicy.CANCEL_ONLY, EmergencyPolicy.FLATTEN_ALL):
            await self._orders.cancel_all()
        if policy is EmergencyPolicy.FLATTEN_ALL:
            log.critical("loop.flatten_all_requested")
            # Flattening submits ordinary orders through the normal path.
            # Not implemented as an automatic action here: it requires an
            # explicit operator-configured policy and a market to trade
            # into. Documented as a known limitation.
