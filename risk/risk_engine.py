"""
The deterministic risk engine.

This is the boundary the spec calls out as non-negotiable: *the AI must
never be able to bypass deterministic risk controls*. Structurally, that
is enforced three ways here:

1. `RiskEngine.evaluate()` takes an `OrderIntent` — the only thing the AI
   layer can produce — and returns a `RiskAssessment`. There is no
   parameter, flag, or override argument that relaxes a check. An AI
   "confidence" of 0.99 changes nothing; confidence is not an input to
   this module at all, deliberately.

2. Limits are read from an immutable snapshot taken at construction. The
   engine holds no setter for them. Changing limits requires constructing
   a new engine, which happens in application wiring, not at runtime from
   a decision path.

3. Checks run in a fixed order, cheapest and most-fatal first, and the
   first rejection short-circuits. Every check that ran is recorded in
   the assessment for the audit trail.

Ordering rationale: kill switch and halts come before anything else
because if we shouldn't be trading at all, no further computation is
meaningful. Data-quality checks come before limit checks because limit
arithmetic computed on stale prices produces confident nonsense.

Failure mode: any unexpected exception inside evaluate() is caught and
converted into a rejection with INTERNAL_ERROR. A risk engine that raises
must not let an order through by virtue of having crashed.
"""

from __future__ import annotations

from datetime import datetime, time, timezone
from decimal import Decimal

import structlog

from data.models import MarketSnapshot
from execution.execution_models import OrderIntent, OrderSide, OrderType
from portfolio.portfolio_manager import MissingPriceError, PortfolioManager
from risk.decisions import RejectionReason, RiskAssessment, RiskDecision
from risk.exposure_manager import ExposureManager
from risk.kill_switch import KillSwitch, TradingHalt
from risk.position_sizer import PositionSizer
from risk.rate_limiter import OrderRateLimiter

log = structlog.get_logger(__name__)


class RiskEngineLimits:
    """Immutable snapshot of the limits this engine enforces. Constructed
    from config once; no runtime mutation path exists."""

    __slots__ = (
        "max_risk_per_trade",
        "max_daily_loss",
        "max_portfolio_drawdown",
        "max_position_size",
        "max_gross_exposure",
        "max_open_positions",
        "max_orders_per_minute",
        "max_market_data_age_seconds",
        "max_spread_pct",
        "price_sanity_band_pct",
        "target_sanity_band_pct",
    )

    def __init__(
        self,
        *,
        max_risk_per_trade: Decimal = Decimal("0.005"),
        max_daily_loss: Decimal = Decimal("0.02"),
        max_portfolio_drawdown: Decimal = Decimal("0.10"),
        max_position_size: Decimal = Decimal("0.10"),
        max_gross_exposure: Decimal = Decimal("1.00"),
        max_open_positions: int = 10,
        max_orders_per_minute: int = 20,
        max_market_data_age_seconds: float = 5.0,
        max_spread_pct: Decimal = Decimal("0.01"),
        price_sanity_band_pct: Decimal = Decimal("0.10"),
        target_sanity_band_pct: Decimal = Decimal("0.50"),
    ) -> None:
        object.__setattr__(self, "max_risk_per_trade", Decimal(str(max_risk_per_trade)))
        object.__setattr__(self, "max_daily_loss", Decimal(str(max_daily_loss)))
        object.__setattr__(self, "max_portfolio_drawdown", Decimal(str(max_portfolio_drawdown)))
        object.__setattr__(self, "max_position_size", Decimal(str(max_position_size)))
        object.__setattr__(self, "max_gross_exposure", Decimal(str(max_gross_exposure)))
        object.__setattr__(self, "max_open_positions", int(max_open_positions))
        object.__setattr__(self, "max_orders_per_minute", int(max_orders_per_minute))
        object.__setattr__(
            self, "max_market_data_age_seconds", float(max_market_data_age_seconds)
        )
        object.__setattr__(self, "max_spread_pct", Decimal(str(max_spread_pct)))
        object.__setattr__(self, "price_sanity_band_pct", Decimal(str(price_sanity_band_pct)))
        object.__setattr__(
            self, "target_sanity_band_pct", Decimal(str(target_sanity_band_pct))
        )

    @classmethod
    def from_config(cls, risk_limits: object) -> "RiskEngineLimits":
        """Build from app.config.RiskLimits without importing it, keeping
        this module usable in isolation (e.g. backtests)."""
        return cls(
            max_risk_per_trade=Decimal(str(getattr(risk_limits, "max_risk_per_trade"))),
            max_daily_loss=Decimal(str(getattr(risk_limits, "max_daily_loss"))),
            max_portfolio_drawdown=Decimal(str(getattr(risk_limits, "max_portfolio_drawdown"))),
            max_position_size=Decimal(str(getattr(risk_limits, "max_position_size"))),
            max_gross_exposure=Decimal(str(getattr(risk_limits, "max_gross_exposure"))),
            max_open_positions=int(getattr(risk_limits, "max_open_positions")),
            max_orders_per_minute=int(getattr(risk_limits, "max_orders_per_minute")),
            max_market_data_age_seconds=float(
                getattr(risk_limits, "max_market_data_age_seconds")
            ),
        )


class RiskEngine:
    def __init__(
        self,
        *,
        limits: RiskEngineLimits,
        portfolio: PortfolioManager,
        kill_switch: KillSwitch,
        trading_halt: TradingHalt,
        position_sizer: PositionSizer | None = None,
        rate_limiter: OrderRateLimiter | None = None,
        exposure_manager: ExposureManager | None = None,
        session_open: time = time(13, 30),  # 09:30 ET in UTC (standard time)
        session_close: time = time(20, 0),  # 16:00 ET in UTC
        enforce_trading_hours: bool = False,
    ) -> None:
        self._limits = limits
        self._portfolio = portfolio
        self._kill_switch = kill_switch
        self._halt = trading_halt
        self._sizer = position_sizer or PositionSizer(
            max_risk_per_trade=limits.max_risk_per_trade,
            max_position_size=limits.max_position_size,
        )
        self._rate_limiter = rate_limiter or OrderRateLimiter(
            max_orders_per_minute=limits.max_orders_per_minute
        )
        # Optional additional constraint. Absent = sector/correlation
        # limits are not enforced; present = it can only reject further.
        self._exposure = exposure_manager
        self._session_open = session_open
        self._session_close = session_close
        self._enforce_trading_hours = enforce_trading_hours
        self._peak_equity: Decimal | None = None

    @property
    def limits(self) -> RiskEngineLimits:
        return self._limits

    @property
    def exposure_manager(self) -> ExposureManager | None:
        return self._exposure

    @property
    def rate_limiter(self) -> OrderRateLimiter:
        return self._rate_limiter

    def update_peak_equity(self, equity: Decimal) -> None:
        """High-water mark for drawdown. Monotonically non-decreasing —
        it must not reset on a losing day, or max drawdown becomes
        meaningless."""
        if self._peak_equity is None or equity > self._peak_equity:
            self._peak_equity = equity

    @property
    def peak_equity(self) -> Decimal | None:
        return self._peak_equity

    def current_drawdown(self, equity: Decimal) -> Decimal:
        if not self._peak_equity:
            return Decimal("0")
        return (self._peak_equity - equity) / self._peak_equity

    # ---- main entry point ---------------------------------------------------

    def evaluate(
        self,
        intent: OrderIntent,
        *,
        snapshot: MarketSnapshot | None,
        prices: dict[str, Decimal],
        atr: Decimal | None = None,
        now: datetime | None = None,
    ) -> RiskAssessment:
        """Evaluate an order intent against every deterministic control.

        Returns an assessment; never raises. An internal failure produces
        a rejection, because a risk engine that crashes must not be
        interpreted as approval.
        """
        now = now or datetime.now(timezone.utc)
        assessment = RiskAssessment(requested_quantity=intent.quantity)

        try:
            checks = (
                lambda: self._check_kill_switch(),
                lambda: self._check_halt(),
                lambda: self._check_intent_sanity(intent),
                lambda: self._check_trading_hours(now),
                lambda: self._check_rate_limit(now),
                lambda: self._check_market_data(intent, snapshot, now),
                lambda: self._check_price_sanity(intent, snapshot),
                lambda: self._check_spread(snapshot),
                lambda: self._check_daily_loss(prices),
                lambda: self._check_drawdown(prices),
                lambda: self._check_open_positions(intent),
            )

            for check in checks:
                decision = check()
                assessment.decisions.append(decision)
                if not decision.approved:
                    assessment.approved = False
                    log.warning(
                        "risk.rejected",
                        intent_id=intent.intent_id,
                        check=decision.check_name,
                        reason=decision.reason,
                        detail=decision.detail,
                        source=intent.source,
                    )
                    return assessment

            # Sizing runs after the gates: no point sizing a trade that
            # was never going to be allowed.
            sized = self._size(intent, snapshot, atr)
            assessment.decisions.append(sized[0])
            if not sized[0].approved:
                assessment.approved = False
                return assessment
            quantity = sized[1]

            # Post-sizing limit checks, using the sized quantity.
            reference_price = self._reference_price(intent, snapshot)
            for check in (
                lambda: self._check_position_size(intent, quantity, reference_price),
                lambda: self._check_gross_exposure(intent, quantity, reference_price, prices),
                lambda: self._check_buying_power(quantity, reference_price),
                lambda: self._check_sector_and_correlation(
                    intent, quantity, reference_price, prices
                ),
            ):
                decision = check()
                assessment.decisions.append(decision)
                if not decision.approved:
                    assessment.approved = False
                    log.warning(
                        "risk.rejected",
                        intent_id=intent.intent_id,
                        check=decision.check_name,
                        reason=decision.reason,
                        detail=decision.detail,
                    )
                    return assessment

            assessment.approved = True
            assessment.approved_quantity = quantity
            log.info(
                "risk.approved",
                intent_id=intent.intent_id,
                requested=str(intent.quantity),
                approved=str(quantity),
                source=intent.source,
            )
            return assessment

        except Exception as exc:  # noqa: BLE001 — fail closed on any fault
            log.exception("risk.internal_error", intent_id=intent.intent_id, error=str(exc))
            assessment.approved = False
            assessment.decisions.append(
                RiskDecision.reject(
                    "internal", RejectionReason.INTERNAL_ERROR, f"Risk engine fault: {exc}"
                )
            )
            return assessment

    # ---- individual checks --------------------------------------------------

    def _check_kill_switch(self) -> RiskDecision:
        if self._kill_switch.is_active:
            event = self._kill_switch.current_event
            return RiskDecision.reject(
                "kill_switch",
                RejectionReason.KILL_SWITCH_ACTIVE,
                f"Kill switch active: {event.trigger if event else 'unknown'}",
            )
        return RiskDecision.approve("kill_switch")

    def _check_halt(self) -> RiskDecision:
        if self._halt.is_halted:
            return RiskDecision.reject(
                "trading_halt",
                RejectionReason.TRADING_HALTED,
                f"Trading halted: {list(self._halt.reasons)}",
            )
        return RiskDecision.approve("trading_halt")

    def _check_intent_sanity(self, intent: OrderIntent) -> RiskDecision:
        if intent.quantity <= 0:
            return RiskDecision.reject(
                "intent_sanity", RejectionReason.ZERO_QUANTITY, "Quantity must be positive"
            )
        return RiskDecision.approve("intent_sanity")

    def _check_trading_hours(self, now: datetime) -> RiskDecision:
        if not self._enforce_trading_hours:
            return RiskDecision.approve("trading_hours", "enforcement disabled")
        if now.weekday() >= 5:
            return RiskDecision.reject(
                "trading_hours", RejectionReason.OUTSIDE_TRADING_HOURS, "Weekend"
            )
        current = now.timetz().replace(tzinfo=None)
        if not (self._session_open <= current <= self._session_close):
            return RiskDecision.reject(
                "trading_hours",
                RejectionReason.OUTSIDE_TRADING_HOURS,
                f"{current} outside {self._session_open}-{self._session_close} UTC",
            )
        return RiskDecision.approve("trading_hours")

    def _check_rate_limit(self, now: datetime) -> RiskDecision:
        if self._rate_limiter.would_exceed(now=now):
            return RiskDecision.reject(
                "rate_limit",
                RejectionReason.MAX_ORDER_RATE_EXCEEDED,
                f"Exceeded {self._limits.max_orders_per_minute} orders/minute",
            )
        return RiskDecision.approve("rate_limit")

    def _check_market_data(
        self, intent: OrderIntent, snapshot: MarketSnapshot | None, now: datetime
    ) -> RiskDecision:
        if snapshot is None:
            return RiskDecision.reject(
                "market_data",
                RejectionReason.MISSING_MARKET_DATA,
                f"No market data for {intent.instrument}",
            )
        if snapshot.is_stale(self._limits.max_market_data_age_seconds, now=now):
            return RiskDecision.reject(
                "market_data",
                RejectionReason.STALE_MARKET_DATA,
                f"Data age {snapshot.age_seconds(now=now):.1f}s exceeds "
                f"{self._limits.max_market_data_age_seconds}s",
            )
        if snapshot.mid is None:
            return RiskDecision.reject(
                "market_data",
                RejectionReason.MISSING_MARKET_DATA,
                "Snapshot has no usable price",
            )
        return RiskDecision.approve("market_data")

    def _check_price_sanity(
        self, intent: OrderIntent, snapshot: MarketSnapshot | None
    ) -> RiskDecision:
        """Reject prices far away from the market. Catches fat-finger
        errors and AI-hallucinated price levels before they reach the
        broker.

        Two bands, because the fields mean different things:

        - **Executable/protective prices** (limit, stop trigger, stop
          loss) must sit close to the market. A limit far from the market
          is a typo; a stop loss far from the market is not protecting
          anything.
        - **`take_profit` is a target, not an executable price.** It is
          *supposed* to be far away — that is the entire point of a
          target. Applying the narrow band to it rejects legitimate
          trades, and (worse) rejects legitimate *exits*, which is more
          dangerous than rejecting an entry. It gets a wider band that
          still catches absurd values.
        """
        if snapshot is None or snapshot.mid is None:
            return RiskDecision.approve("price_sanity", "no market reference")
        mid = Decimal(str(snapshot.mid))

        checks = (
            ("limit", intent.limit_price, self._limits.price_sanity_band_pct),
            ("stop", intent.stop_price, self._limits.price_sanity_band_pct),
            ("stop_loss", intent.stop_loss, self._limits.price_sanity_band_pct),
            ("take_profit", intent.take_profit, self._limits.target_sanity_band_pct),
        )
        for label, price, band in checks:
            if price is None:
                continue
            deviation = abs(price - mid) / mid
            if deviation > band:
                return RiskDecision.reject(
                    "price_sanity",
                    RejectionReason.PRICE_SANITY_FAILED,
                    f"{label} price {price} deviates {deviation:.1%} from mid {mid} "
                    f"(band {band:.1%})",
                )
        return RiskDecision.approve("price_sanity")

    def _check_spread(self, snapshot: MarketSnapshot | None) -> RiskDecision:
        if snapshot is None or snapshot.bid is None or snapshot.ask is None:
            return RiskDecision.approve("spread", "no bid/ask available")
        bid, ask = Decimal(str(snapshot.bid)), Decimal(str(snapshot.ask))
        if bid <= 0 or ask <= 0 or ask < bid:
            return RiskDecision.reject(
                "spread", RejectionReason.PRICE_SANITY_FAILED, f"Crossed/invalid book {bid}/{ask}"
            )
        mid = (bid + ask) / 2
        spread_pct = (ask - bid) / mid
        if spread_pct > self._limits.max_spread_pct:
            return RiskDecision.reject(
                "spread",
                RejectionReason.SPREAD_TOO_WIDE,
                f"Spread {spread_pct:.2%} exceeds {self._limits.max_spread_pct:.2%}",
            )
        return RiskDecision.approve("spread")

    def _check_daily_loss(self, prices: dict[str, Decimal]) -> RiskDecision:
        try:
            daily_pct = self._portfolio.daily_pnl_pct(prices)
        except MissingPriceError as exc:
            # Cannot evaluate the limit -> cannot approve. Fail closed.
            return RiskDecision.reject(
                "daily_loss",
                RejectionReason.MISSING_MARKET_DATA,
                f"Cannot evaluate daily loss limit: {exc}",
            )
        if daily_pct <= -self._limits.max_daily_loss:
            return RiskDecision.reject(
                "daily_loss",
                RejectionReason.MAX_DAILY_LOSS_BREACHED,
                f"Daily P&L {daily_pct:.2%} breaches limit {-self._limits.max_daily_loss:.2%}",
            )
        return RiskDecision.approve("daily_loss", f"daily P&L {daily_pct:.2%}")

    def _check_drawdown(self, prices: dict[str, Decimal]) -> RiskDecision:
        equity = self._portfolio.account.equity
        self.update_peak_equity(equity)
        drawdown = self.current_drawdown(equity)
        if drawdown >= self._limits.max_portfolio_drawdown:
            return RiskDecision.reject(
                "drawdown",
                RejectionReason.MAX_DRAWDOWN_BREACHED,
                f"Drawdown {drawdown:.2%} breaches limit "
                f"{self._limits.max_portfolio_drawdown:.2%}",
            )
        return RiskDecision.approve("drawdown", f"drawdown {drawdown:.2%}")

    def _check_open_positions(self, intent: OrderIntent) -> RiskDecision:
        existing = self._portfolio.get_position(intent.instrument)
        # Adding to or closing an existing position doesn't open a new slot.
        if not existing.is_flat:
            return RiskDecision.approve("open_positions", "existing position")
        if self._portfolio.open_position_count >= self._limits.max_open_positions:
            return RiskDecision.reject(
                "open_positions",
                RejectionReason.MAX_OPEN_POSITIONS_EXCEEDED,
                f"{self._portfolio.open_position_count} open positions, "
                f"limit {self._limits.max_open_positions}",
            )
        return RiskDecision.approve("open_positions")

    # ---- sizing and post-sizing checks -------------------------------------

    def _reference_price(
        self, intent: OrderIntent, snapshot: MarketSnapshot | None
    ) -> Decimal:
        if intent.limit_price is not None:
            return intent.limit_price
        if snapshot is not None and snapshot.mid is not None:
            return Decimal(str(snapshot.mid))
        raise MissingPriceError(f"No reference price for {intent.instrument}")

    def _size(
        self, intent: OrderIntent, snapshot: MarketSnapshot | None, atr: Decimal | None
    ) -> tuple[RiskDecision, Decimal]:
        entry = self._reference_price(intent, snapshot)
        # Size against the PROTECTIVE stop, not the order's trigger price.
        # For a STOP entry order the trigger is the entry, not the risk.
        result = self._sizer.calculate(
            equity=self._portfolio.account.equity,
            entry_price=entry,
            stop_price=intent.stop_loss,
            atr=atr,
            requested_quantity=intent.quantity,
            strategy=intent.strategy or intent.source,
        )
        if not result.is_tradeable:
            return (
                RiskDecision.reject(
                    "position_sizing",
                    RejectionReason.MAX_RISK_PER_TRADE_EXCEEDED,
                    f"Sizer returned zero: {result.detail}",
                ),
                Decimal("0"),
            )
        return (
            RiskDecision.approve(
                "position_sizing", f"{result.method}: {result.quantity} ({result.detail})"
            ),
            result.quantity,
        )

    def _check_position_size(
        self, intent: OrderIntent, quantity: Decimal, price: Decimal
    ) -> RiskDecision:
        equity = self._portfolio.account.equity
        if equity <= 0:
            return RiskDecision.reject(
                "position_size", RejectionReason.INSUFFICIENT_BUYING_POWER, "Equity is zero"
            )
        existing = self._portfolio.get_position(intent.instrument)
        signed = quantity if intent.side is OrderSide.BUY else -quantity
        projected = abs(existing.quantity + signed)
        projected_pct = (projected * price) / equity
        if projected_pct > self._limits.max_position_size:
            return RiskDecision.reject(
                "position_size",
                RejectionReason.MAX_POSITION_SIZE_EXCEEDED,
                f"Projected position {projected_pct:.2%} of equity exceeds "
                f"{self._limits.max_position_size:.2%}",
            )
        return RiskDecision.approve("position_size", f"projected {projected_pct:.2%}")

    def _check_gross_exposure(
        self,
        intent: OrderIntent,
        quantity: Decimal,
        price: Decimal,
        prices: dict[str, Decimal],
    ) -> RiskDecision:
        equity = self._portfolio.account.equity
        if equity <= 0:
            return RiskDecision.reject(
                "gross_exposure", RejectionReason.INSUFFICIENT_BUYING_POWER, "Equity is zero"
            )
        try:
            current = self._portfolio.gross_exposure(prices)
        except MissingPriceError as exc:
            return RiskDecision.reject(
                "gross_exposure",
                RejectionReason.MISSING_MARKET_DATA,
                f"Cannot evaluate exposure: {exc}",
            )

        existing = self._portfolio.get_position(intent.instrument)
        signed = quantity if intent.side is OrderSide.BUY else -quantity
        existing_exposure = abs(existing.quantity) * price
        projected_exposure = abs(existing.quantity + signed) * price
        projected_total = current - existing_exposure + projected_exposure
        projected_pct = projected_total / equity

        if projected_pct > self._limits.max_gross_exposure:
            return RiskDecision.reject(
                "gross_exposure",
                RejectionReason.MAX_GROSS_EXPOSURE_EXCEEDED,
                f"Projected gross exposure {projected_pct:.2%} exceeds "
                f"{self._limits.max_gross_exposure:.2%}",
            )
        return RiskDecision.approve("gross_exposure", f"projected {projected_pct:.2%}")

    def _check_sector_and_correlation(
        self,
        intent: OrderIntent,
        quantity: Decimal,
        price: Decimal,
        prices: dict[str, Decimal],
    ) -> RiskDecision:
        """Sector and correlation concentration. Only ever rejects: gross
        exposure alone treats ten positions in one sector as diversified
        when they are effectively one position with ten tickets."""
        if self._exposure is None:
            return RiskDecision.approve("sector_correlation", "not configured")

        assessment = self._exposure.evaluate(
            instrument=intent.instrument,
            additional_quantity=quantity,
            price=price,
            positions=self._portfolio.positions,
            prices=prices,
            equity=self._portfolio.account.equity,
        )
        if not assessment.approved:
            breach = assessment.breaches[0]
            reason = (
                RejectionReason.MAX_SECTOR_EXPOSURE_EXCEEDED
                if breach.kind == "MAX_SECTOR_EXPOSURE"
                else RejectionReason.MAX_CORRELATED_EXPOSURE_EXCEEDED
            )
            return RiskDecision.reject("sector_correlation", reason, breach.detail)
        return RiskDecision.approve(
            "sector_correlation",
            f"correlated cluster exposure {assessment.correlated_exposure}",
        )

    def _check_buying_power(self, quantity: Decimal, price: Decimal) -> RiskDecision:
        required = quantity * price
        available = self._portfolio.account.buying_power
        if available > 0 and required > available:
            return RiskDecision.reject(
                "buying_power",
                RejectionReason.INSUFFICIENT_BUYING_POWER,
                f"Requires {required}, buying power {available}",
            )
        return RiskDecision.approve("buying_power")
