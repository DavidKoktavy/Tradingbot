"""
Dependency container.

Design decisions:

- **One place where the object graph is assembled.** Wiring scattered
  across modules makes it impossible to answer "what actually talks to
  what", which matters a great deal when the question is "can anything
  reach the broker without passing the risk engine".

- **The container is built from settings and is otherwise inert.** It
  starts nothing, connects to nothing, and submits nothing on
  construction. Side effects at import or construction time make the CLI
  dangerous — `trading_agent status` must not be able to place an order.

- **The mode gate is constructed first and injected everywhere**, so no
  component can be built with a different view of what mode it is in.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import structlog

from ai.decision_engine import AIDecisionEngine
from ai.providers import AIProvider, AnthropicProvider, NullProvider
from ai.regime_detector import RegimeDetector
from app.config import Settings, TradingMode
from app.mode_gate import ModeGate, gate_from_settings
from broker.order_manager import BrokerOrderGateway, OrderManager
from broker.simulated_broker import SimulatedBrokerGateway
from data.models import Instrument
from execution.order_store import OrderStore
from execution.order_validator import OrderValidator
from execution.reconciliation import Reconciler
from monitoring.alerts import AlertManager, LogNotificationProvider
from monitoring.audit import DecisionRecorder
from monitoring.health import (
    HealthMonitor,
    Severity,
    ai_provider_check,
    kill_switch_check,
    portfolio_check,
)
from monitoring.journal import TradeJournal
from monitoring.metrics import MetricsRegistry
from portfolio.portfolio_manager import AccountState, PortfolioManager
from risk.kill_switch import EmergencyPolicy, KillSwitch, TradingHalt
from risk.risk_engine import RiskEngine, RiskEngineLimits
import strategies  # noqa: F401 — import for side effect: registers built-in strategies
from strategies.base import Strategy
from strategies.engine import StrategyEngine
from strategies.registry import registry as strategy_registry

log = structlog.get_logger(__name__)


@dataclass
class Container:
    settings: Settings
    mode_gate: ModeGate
    portfolio: PortfolioManager
    kill_switch: KillSwitch
    trading_halt: TradingHalt
    risk_engine: RiskEngine
    order_store: OrderStore
    validator: OrderValidator
    gateway: BrokerOrderGateway
    order_manager: OrderManager
    reconciler: Reconciler
    strategy_engine: StrategyEngine
    ai_engine: AIDecisionEngine
    regime_detector: RegimeDetector
    metrics: MetricsRegistry
    alerts: AlertManager
    health: HealthMonitor
    recorder: DecisionRecorder
    journal: TradeJournal
    instruments: list[Instrument]
    repository: object | None = None


def build_ai_provider(settings: Settings) -> AIProvider:
    """Return a real provider only if fully configured. Otherwise the null
    provider, so the system runs on deterministic strategies rather than
    refusing to start."""
    key = settings.ai.anthropic_api_key
    if settings.ai.provider == "anthropic" and key and settings.ai.model:
        try:
            return AnthropicProvider(key.get_secret_value(), settings.ai.model)
        except Exception as exc:  # noqa: BLE001
            log.error("container.ai_provider_failed", error=str(exc))
    log.warning("container.ai_unconfigured", detail="Using NullProvider")
    return NullProvider()


def _database_check(repository: object):
    def check():
        from monitoring.health import degraded, ok

        healthy = repository.health_check()
        failures = getattr(repository, "write_failures", 0)
        if not healthy:
            return degraded("Database unreachable; trading continues, audit degraded")
        if failures:
            return degraded(f"{failures} failed writes since start")
        return ok("connected")

    return check


def build_container(
    settings: Settings,
    *,
    symbols: list[str] | None = None,
    strategy_names: list[str] | None = None,
    gateway: BrokerOrderGateway | None = None,
    initial_equity: Decimal = Decimal("100000"),
    audit_path: Path | str | None = None,
    repository: object | None = None,
) -> Container:
    mode_gate = gate_from_settings(settings)
    instruments = [Instrument(symbol=s.upper()) for s in (symbols or ["SPY"])]

    portfolio = PortfolioManager()
    portfolio.update_account(
        AccountState(
            equity=initial_equity, cash=initial_equity, buying_power=initial_equity
        )
    )

    kill_switch = KillSwitch(emergency_policy=EmergencyPolicy.CANCEL_ONLY)
    trading_halt = TradingHalt()

    risk_engine = RiskEngine(
        limits=RiskEngineLimits.from_config(settings.risk),
        portfolio=portfolio,
        kill_switch=kill_switch,
        trading_halt=trading_halt,
    )

    order_store = OrderStore()
    validator = OrderValidator(order_store)

    # In BACKTEST/SIMULATION we never construct a real broker gateway at
    # all: the safest way to guarantee no live order is to have no object
    # capable of sending one.
    if gateway is None:
        gateway = SimulatedBrokerGateway()
    order_manager = OrderManager(gateway, order_store, mode_gate)
    reconciler = Reconciler(order_store, portfolio)

    names = strategy_names or ["ma_crossover"]
    strategies: list[Strategy] = []
    for name in names:
        try:
            strategies.append(strategy_registry.create(name))
        except KeyError:
            log.error("container.unknown_strategy", name=name)
    strategy_engine = StrategyEngine(strategies)

    ai_engine = AIDecisionEngine(
        build_ai_provider(settings),
        allowed_symbols={i.symbol for i in instruments},
    )

    metrics = MetricsRegistry()
    alerts = AlertManager([LogNotificationProvider()])
    recorder = DecisionRecorder(path=audit_path)
    journal = TradeJournal(
        recorder,
        repository=repository,
        portfolio=portfolio,
        trading_mode=str(mode_gate.mode),
    )

    health = HealthMonitor()
    health.register("portfolio", portfolio_check(portfolio), severity=Severity.CRITICAL)
    health.register("kill_switch", kill_switch_check(kill_switch))
    health.register("ai", ai_provider_check(ai_engine))
    if repository is not None:
        # DEGRADED, not CRITICAL: a dead audit database costs observability,
        # not correctness — reconciliation reads from the broker.
        health.register("database", _database_check(repository))

    return Container(
        settings=settings,
        mode_gate=mode_gate,
        portfolio=portfolio,
        kill_switch=kill_switch,
        trading_halt=trading_halt,
        risk_engine=risk_engine,
        order_store=order_store,
        validator=validator,
        gateway=gateway,
        order_manager=order_manager,
        reconciler=reconciler,
        strategy_engine=strategy_engine,
        ai_engine=ai_engine,
        regime_detector=RegimeDetector(),
        metrics=metrics,
        alerts=alerts,
        health=health,
        recorder=recorder,
        journal=journal,
        instruments=instruments,
        repository=repository,
    )
