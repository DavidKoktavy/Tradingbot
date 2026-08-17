"""
Command-line interface.

Design decisions:

- **Every command prints the current mode banner first**, in colour, before
  doing anything. The spec requires the CLI to make the mode obvious; an
  operator who has to ask "wait, is this live?" has already been failed by
  the tool.

- **No command defaults to LIVE, and no command can promote to LIVE.**
  `paper` and `simulate` exist; there is deliberately no `live`
  subcommand. Running live requires setting the environment variables the
  config layer validates, which is a deliberate act performed outside the
  convenience of a CLI flag.

- **`kill-switch` activates only.** As with the dashboard, there is no
  deactivation command: resuming trading after an emergency stop should
  require a considered restart, not a single keystroke.

- Read-only commands (`status`, `strategies`, `positions`, `risk`) never
  construct a broker connection.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from decimal import Decimal
from pathlib import Path

import strategies  # noqa: F401 — import for side effect: registers built-in strategies
from app.config import TradingMode, get_settings
from app.dependency_container import build_container
from app.logging_config import configure_logging

RED = "\033[41;97m"
GREEN = "\033[42;30m"
YELLOW = "\033[43;30m"
RESET = "\033[0m"


def _banner(mode: TradingMode) -> str:
    colour = {
        TradingMode.LIVE: RED,
        TradingMode.PAPER: GREEN,
        TradingMode.SIMULATION: YELLOW,
        TradingMode.BACKTEST: YELLOW,
    }[mode]
    suffix = "  ***  REAL MONEY AT RISK  ***" if mode is TradingMode.LIVE else ""
    if not sys.stdout.isatty():
        return f"[ MODE: {mode} ]{suffix}"
    return f"{colour} MODE: {mode}{suffix} {RESET}"


def _print_banner(mode: TradingMode) -> None:
    print(_banner(mode))
    print()


# ---- commands --------------------------------------------------------------


def cmd_status(args: argparse.Namespace) -> int:
    settings = get_settings()
    _print_banner(settings.trading_mode)
    container = build_container(settings, symbols=args.symbols)

    report = asyncio.run(container.health.run())
    print(f"Health:        {report.status}")
    print(f"Can trade:     {report.can_trade}")
    for check in report.checks:
        print(f"  [{check.status:9}] {check.name}: {check.detail}")
    print()
    print(f"Kill switch:   {'ACTIVE' if container.kill_switch.is_active else 'inactive'}")
    print(f"Trading halt:  {container.trading_halt.is_halted}")
    print(f"Instruments:   {', '.join(str(i) for i in container.instruments)}")
    print(f"Strategies:    {', '.join(s.name for s in container.strategy_engine.active_strategies)}")
    print(f"AI provider:   {'available' if container.ai_engine.provider_available else 'unavailable (deterministic only)'}")
    return 0


def cmd_strategies(args: argparse.Namespace) -> int:
    from strategies.registry import registry

    settings = get_settings()
    _print_banner(settings.trading_mode)
    print("Registered strategies (none is claimed to be profitable):\n")
    for entry in registry.describe():
        print(f"  {entry['name']:<18} v{entry['version']:<8} {entry['class']}")
    return 0


def cmd_positions(args: argparse.Namespace) -> int:
    settings = get_settings()
    _print_banner(settings.trading_mode)
    container = build_container(settings, symbols=args.symbols)

    positions = container.portfolio.open_positions()
    if not positions:
        print("No open positions.")
        return 0
    print(f"{'Instrument':<24} {'Qty':>12} {'Avg cost':>12} {'Realised':>12}")
    for p in positions:
        print(
            f"{str(p.instrument):<24} {str(p.quantity):>12} "
            f"{str(p.average_cost):>12} {str(p.realized_pnl):>12}"
        )
    return 0


def cmd_risk(args: argparse.Namespace) -> int:
    settings = get_settings()
    _print_banner(settings.trading_mode)
    container = build_container(settings, symbols=args.symbols)
    limits = container.risk_engine.limits

    print("Configured risk limits:\n")
    for name in (
        "max_risk_per_trade",
        "max_daily_loss",
        "max_portfolio_drawdown",
        "max_position_size",
        "max_gross_exposure",
        "max_open_positions",
        "max_orders_per_minute",
        "max_market_data_age_seconds",
    ):
        print(f"  {name:<30} {getattr(limits, name)}")
    print()
    print(f"Kill switch:      {'ACTIVE' if container.kill_switch.is_active else 'inactive'}")
    print(f"Emergency policy: {container.kill_switch.emergency_policy}")
    print(f"Orders last min:  {container.risk_engine.rate_limiter.current_count}")
    return 0


def cmd_backtest(args: argparse.Namespace) -> int:
    from backtesting.engine import BacktestEngine
    from data.models import Instrument
    from strategies.registry import registry

    settings = get_settings()
    _print_banner(TradingMode.BACKTEST)

    bars = _load_bars(args.data)
    if not bars:
        print(f"No bars loaded from {args.data}", file=sys.stderr)
        return 1

    try:
        strategy = registry.create(args.strategy)
    except KeyError as exc:
        print(f"Unknown strategy: {exc}", file=sys.stderr)
        return 1

    engine = BacktestEngine(
        strategy=strategy,
        instrument=Instrument(symbol=args.symbol.upper()),
        initial_equity=Decimal(str(args.equity)),
        bar_size=args.bar_size,
    )
    result = engine.run(bars)
    print(result.summary())
    print()
    print("Reminder: backtest results are not evidence of future profitability.")
    return 0


def cmd_overfitting_check(args: argparse.Namespace) -> int:
    """Grid-search a strategy, then report the Deflated Sharpe Ratio for
    the best result — the statistically corrected answer to 'is this
    actually good, or did I just search hard enough to find noise that
    looks good?'
    """
    from backtesting.statistics import evaluate_search_overfitting
    from backtesting.walk_forward import grid_search_with_trials
    from data.models import Instrument
    from strategies.registry import registry

    settings = get_settings()
    _print_banner(TradingMode.BACKTEST)

    bars = _load_bars(args.data)
    if not bars:
        print(f"No bars loaded from {args.data}", file=sys.stderr)
        return 1
    try:
        grid = json.loads(args.grid)
    except json.JSONDecodeError as exc:
        print(f"--grid is not valid JSON: {exc}", file=sys.stderr)
        return 1
    if not isinstance(grid, dict) or not all(isinstance(v, list) for v in grid.values()):
        print("--grid must be a JSON object mapping parameter names to lists", file=sys.stderr)
        return 1

    try:
        strategy_cls = registry.get(args.strategy)
    except KeyError as exc:
        print(f"Unknown strategy: {exc}", file=sys.stderr)
        return 1
    params_cls = type(strategy_cls().params)

    print(f"Searching {len(list(__import__('itertools').product(*grid.values())))} "
          f"parameter combinations...")
    best_params, best_metrics, trials, tested = grid_search_with_trials(
        strategy_cls=strategy_cls,
        params_cls=params_cls,
        grid=grid,
        bars=bars,
        instrument=Instrument(symbol=args.symbol.upper()),
        initial_equity=Decimal(str(args.equity)),
        bar_size=args.bar_size,
    )
    if best_params is None or best_metrics is None:
        print("No valid parameter combination produced a result.", file=sys.stderr)
        return 1

    print(f"\nBest raw result ({tested} combinations tested):")
    print(best_metrics.summary())
    print(f"Best params: {best_params.model_dump()}")

    report = evaluate_search_overfitting(trials)
    print("\n=== Deflated Sharpe Ratio (corrected for search size) ===\n")
    print(report.summary())
    print()
    if report.has_result and not report.likely_genuine:
        print(
            "This result does NOT clear 95% confidence once the number of parameter "
            "combinations searched is accounted for. Reporting the raw Sharpe above "
            "without this correction would overstate how good this strategy looks."
        )
    print(
        "\nThis corrects for how hard you searched. It is not a substitute for "
        "out-of-sample testing — pair with `walk_forward.evaluate_out_of_sample` "
        "before treating this as evidence of anything."
    )
    return 0


def cmd_simulate(args: argparse.Namespace) -> int:
    settings = get_settings()
    if settings.trading_mode is TradingMode.LIVE:
        print("Refusing to run a simulation while TRADING_MODE=LIVE.", file=sys.stderr)
        return 1
    _print_banner(TradingMode.SIMULATION)
    print("SIMULATION mode uses an internal fill simulator; no broker connection is made.")
    if not args.data:
        print(
            "No --data file given: there is no market data source, so nothing will "
            "be evaluated and no orders will be submitted. Pass --data a_file.csv to "
            "replay real bars through the loop and actually see it decide and trade."
        )
        return _run_loop(args, TradingMode.SIMULATION)
    return _run_simulation_replay(args)


def _run_simulation_replay(args: argparse.Namespace) -> int:
    """Drives the control loop bar-by-bar over a historical CSV, and
    drains the simulated broker's fills after each cycle — the piece
    `ControlLoop` deliberately does NOT do itself (fill draining is the
    caller's responsibility, matching how a real broker's execution
    events arrive out-of-band from the cycle that submitted the order).
    This is the most direct, zero-setup way to watch the full
    strategy -> risk -> order -> fill -> portfolio chain actually run.
    """
    from broker.execution_listener import ExecutionListener
    from broker.order_manager import OrderManager
    from broker.simulated_broker import SimulatedBrokerGateway
    from data.models import MarketSnapshot
    from datetime import datetime, timezone

    settings = get_settings()
    bars = _load_bars(args.data)
    if not bars:
        print(f"No bars loaded from {args.data}", file=sys.stderr)
        return 1

    container = build_container(
        settings, symbols=args.symbols, strategy_names=args.strategies
    )
    instrument = container.instruments[0]
    if len(container.instruments) > 1:
        print(
            "Note: --data replay only drives the first --symbols entry "
            f"({instrument}); the rest have no data source this run.",
            file=sys.stderr,
        )

    gateway = SimulatedBrokerGateway()
    order_manager = OrderManager(gateway, container.order_store, container.mode_gate)
    listener = ExecutionListener(
        container.order_store, container.portfolio, kill_switch=container.kill_switch
    )

    from app.control_loop import ControlLoop, MarketDataFeed

    feed = MarketDataFeed()
    loop = ControlLoop(
        instruments=[instrument],
        feed=feed,
        strategy_engine=container.strategy_engine,
        risk_engine=container.risk_engine,
        validator=container.validator,
        order_manager=order_manager,
        order_store=container.order_store,
        portfolio=container.portfolio,
        reconciler=container.reconciler,
        kill_switch=container.kill_switch,
        trading_halt=container.trading_halt,
        mode_gate=container.mode_gate,
        ai_engine=container.ai_engine,
        regime_detector=container.regime_detector,
        metrics=container.metrics,
        alerts=container.alerts,
        journal=container.journal,
        cycle_seconds=0.0,
    )

    strategy = container.strategy_engine.active_strategies[0] if container.strategy_engine.active_strategies else None
    warmup = strategy.min_bars if strategy is not None else 30
    if len(bars) <= warmup:
        print(
            f"Only {len(bars)} bars, but the strategy needs {warmup} to warm up — "
            "nothing will trade. Use a longer --data file.",
            file=sys.stderr,
        )
        return 1

    print(f"Replaying {len(bars)} bars ({warmup} used for warm-up) through the live "
          f"strategy -> risk -> order -> fill chain...\n")

    async def replay() -> None:
        await loop.reconcile()
        for i in range(warmup, len(bars)):
            window = bars[: i + 1]
            bar = bars[i]
            feed.bars[str(instrument)] = window
            # ControlLoop is built for LIVE operation and checks data
            # freshness against the real wall clock (unlike
            # BacktestEngine, which explicitly accepts simulation time —
            # see backtesting/engine.py). Stamping each replayed bar with
            # its own historical timestamp would make every single one
            # look catastrophically stale and get correctly rejected,
            # which is exactly the bug this comment is here to prevent
            # reintroducing. We deliberately trade historical timestamp
            # fidelity for exercising the real live-trading gate
            # correctly: this is a demo of the LIVE decision machinery
            # using historical price shapes, not a faithful historical
            # replay — do not use this for anything resembling a backtest.
            feed.snapshots[str(instrument)] = MarketSnapshot(
                instrument=instrument,
                timestamp=datetime.now(timezone.utc),
                bid=bar.close - 0.02,
                ask=bar.close + 0.02,
                last=bar.close,
                volume=bar.volume,
            )
            gateway.set_snapshot(feed.snapshots[str(instrument)])
            await loop.run_cycle()
            for fill in await gateway.fill_all_pending():
                await listener.handle_fill(fill)
                container.journal.record_fill(fill)

    asyncio.run(replay())

    position = container.portfolio.get_position(instrument)
    print(f"Cycles: {loop.stats.cycles}  Orders submitted: {loop.stats.orders_submitted}  "
          f"Fills: {listener.fill_count}")
    print(f"Final position: {position.quantity} @ avg cost {position.average_cost}")
    print(f"Realised P&L: {position.realized_pnl}  Commission paid: {position.total_commission}")
    if container.kill_switch.is_active:
        print(f"Kill switch tripped: {container.kill_switch.current_event.trigger}")
    return 0


def cmd_paper(args: argparse.Namespace) -> int:
    settings = get_settings()
    _print_banner(settings.trading_mode)
    if settings.trading_mode is TradingMode.LIVE:
        print(
            "TRADING_MODE is LIVE. The `paper` command will not silently downgrade it.\n"
            "Set TRADING_MODE=PAPER explicitly to run paper trading.",
            file=sys.stderr,
        )
        return 1
    return _run_loop(args, TradingMode.PAPER)


def cmd_reconcile(args: argparse.Namespace) -> int:
    settings = get_settings()
    _print_banner(settings.trading_mode)
    container = build_container(settings, symbols=args.symbols)

    async def run() -> int:
        positions = await container.gateway.positions()
        orders = await container.gateway.open_orders()
        report = container.reconciler.reconcile(
            broker_positions=positions, broker_orders=orders
        )
        if report.is_clean:
            print("Reconciliation clean: local state matches broker.")
            return 0
        print(f"{len(report.discrepancies)} discrepancies found:\n")
        for d in report.discrepancies:
            print(f"  [{d.kind}] {d.detail}")
        print()
        print(f"Requires halt: {report.requires_halt}")
        print("No corrective orders were placed. Resolve manually before trading.")
        return 2

    return asyncio.run(run())


def cmd_kill_switch(args: argparse.Namespace) -> int:
    from risk.kill_switch import KillSwitchTrigger

    settings = get_settings()
    _print_banner(settings.trading_mode)
    container = build_container(settings, symbols=args.symbols)
    event = container.kill_switch.activate(KillSwitchTrigger.MANUAL, args.reason)
    print(f"Kill switch ACTIVATED: {event.trigger} — {event.detail}")
    print("New orders are blocked.")
    print()
    print(
        "Note: this activates the switch in this process only. For a running agent, "
        "use the dashboard endpoint POST /kill-switch/activate."
    )
    print("There is no CLI command to deactivate; resume requires a considered restart.")
    return 0


def cmd_explain(args: argparse.Namespace) -> int:
    """Answer 'why did the agent make this trade?' from the audit trail."""
    from monitoring.audit import DecisionRecorder

    settings = get_settings()
    _print_banner(settings.trading_mode)
    recorder = DecisionRecorder(path=args.audit_log)
    records = recorder.replay()
    if not records:
        print(f"No decision records found in {args.audit_log}")
        return 1

    if args.record_id:
        match = next((r for r in records if r.record_id.startswith(args.record_id)), None)
        if match is None:
            print(f"No record matching {args.record_id}", file=sys.stderr)
            return 1
        print(match.explain())
        return 0

    for record in records[-args.limit :]:
        print(record.explain())
        print("-" * 72)
    return 0


def cmd_macro_add(args: argparse.Namespace) -> int:
    from datetime import date, timedelta

    from ai.macro_context import MacroContextRegistry, MacroFactor

    settings = get_settings()
    _print_banner(settings.trading_mode)
    registry = MacroContextRegistry.load(args.store)
    factor = MacroFactor(
        name=args.name,
        category=args.category,
        stance=args.stance,
        description=args.description,
        affected_sectors=args.sectors,
        affected_symbols=args.symbols,
        confidence=args.confidence,
        source=args.source,
        expires_at=date.today() + timedelta(days=args.expires_in_days),
    )
    registry.add(factor)
    registry.save(args.store)
    print(f"Added '{factor.name}' [{factor.category}], expires {factor.expires_at}")
    print("This is stored as a labelled hypothesis and surfaced to the AI as context.")
    print("It does not change any risk limit, sizing rule, or trading permission.")
    return 0


def cmd_macro_list(args: argparse.Namespace) -> int:
    from ai.macro_context import MacroContextRegistry

    settings = get_settings()
    _print_banner(settings.trading_mode)
    registry = MacroContextRegistry.load(args.store)
    factors = registry.all() if args.all else registry.active()
    if not factors:
        print("No macro factors" + ("" if args.all else " currently active") + ".")
        return 0
    for f in factors:
        status = "active" if f.is_active() else "EXPIRED"
        print(f"[{status}] {f.name} — {f.category} — stance={f.stance} "
              f"(confidence {f.confidence:.2f}, expires {f.expires_at})")
        if f.description:
            print(f"    {f.description}")
        if f.affected_sectors or f.affected_symbols:
            print(f"    sectors={f.affected_sectors} symbols={f.affected_symbols}")
        if f.source:
            print(f"    source: {f.source}")
    return 0


def cmd_macro_remove(args: argparse.Namespace) -> int:
    from ai.macro_context import MacroContextRegistry

    settings = get_settings()
    _print_banner(settings.trading_mode)
    registry = MacroContextRegistry.load(args.store)
    if registry.remove(args.name):
        registry.save(args.store)
        print(f"Removed '{args.name}'.")
        return 0
    print(f"No factor named '{args.name}' found.", file=sys.stderr)
    return 1


def cmd_reflect(args: argparse.Namespace) -> int:
    """Run a backtest, then analyse the resulting trades — deterministic
    stats always, an AI hypothesis pass only if a provider is configured.

    Nothing here changes any live parameter. See ai/reflection.py.
    """
    import asyncio

    from ai.performance_analyzer import PerformanceAnalyzer
    from ai.reflection import ReflectionEngine
    from app.dependency_container import build_ai_provider
    from backtesting.engine import BacktestEngine
    from data.models import Instrument
    from strategies.registry import registry

    settings = get_settings()
    _print_banner(TradingMode.BACKTEST)

    bars = _load_bars(args.data)
    if not bars:
        print(f"No bars loaded from {args.data}", file=sys.stderr)
        return 1
    try:
        strategy = registry.create(args.strategy)
    except KeyError as exc:
        print(f"Unknown strategy: {exc}", file=sys.stderr)
        return 1

    engine = BacktestEngine(
        strategy=strategy,
        instrument=Instrument(symbol=args.symbol.upper()),
        initial_equity=Decimal(str(args.equity)),
        bar_size=args.bar_size,
    )
    result = engine.run(bars)

    report = PerformanceAnalyzer().analyze(result.trades, rejection_counts=result.rejections)
    print("=== Deterministic performance report ===")
    print()
    print(report.summary())

    reflection = ReflectionEngine(
        build_ai_provider(settings), known_strategies={args.strategy}
    )
    if not reflection.provider_available:
        print()
        print(
            "No AI provider configured — showing deterministic analysis only. "
            "This is complete and self-contained; the AI pass only adds narrative "
            "hypotheses on top of the numbers above."
        )
        return 0

    print()
    print("=== AI hypotheses (advisory only — nothing below is applied) ===")
    print()
    outcome = asyncio.run(reflection.reflect(report))
    if not outcome.accepted:
        print(f"Reflection declined: {outcome.reason} — {outcome.detail}")
        return 0
    if not outcome.hypotheses:
        print("No hypotheses proposed.")
        return 0
    for h in outcome.hypotheses:
        print(f"[{h.strategy}] {h.suggested_action} (confidence {h.confidence:.2f})")
        print(f"  Observation: {h.observation}")
        print(f"  Hypothesis:  {h.hypothesis}")
        if h.suggested_params:
            print(f"  Suggested params (NOT applied): {h.suggested_params}")
        print()
    print(
        "None of the above changes anything. To act on a hypothesis, submit it as a "
        "ResearchProposal to the PromotionPipeline (strategies/promotion.py) and take it "
        "through BACKTEST -> VALIDATION -> PAPER -> a named human's APPROVAL -> LIVE."
    )
    return 0


def cmd_migrate(args: argparse.Namespace) -> int:
    """Apply Alembic migrations using the configured database URL."""
    settings = get_settings()
    _print_banner(settings.trading_mode)
    try:
        from alembic import command
        from alembic.config import Config
    except ImportError:
        print("alembic is required: pip install alembic", file=sys.stderr)
        return 1

    config = Config("alembic.ini")
    # The URL comes from settings, never from alembic.ini, so credentials
    # stay out of source control.
    config.set_main_option(
        "sqlalchemy.url",
        settings.database_url.get_secret_value().replace(
            "postgresql+asyncpg", "postgresql+psycopg"
        ),
    )
    command.upgrade(config, args.revision)
    print(f"Migrated to {args.revision}.")
    return 0


def cmd_dashboard(args: argparse.Namespace) -> int:
    settings = get_settings()
    _print_banner(settings.trading_mode)
    try:
        import uvicorn
    except ImportError:
        print("uvicorn is required: pip install uvicorn", file=sys.stderr)
        return 1

    from monitoring.dashboard import DashboardState, create_app

    container = build_container(settings, symbols=args.symbols)
    state = DashboardState(
        mode_gate=container.mode_gate,
        portfolio=container.portfolio,
        order_store=container.order_store,
        risk_engine=container.risk_engine,
        kill_switch=container.kill_switch,
        trading_halt=container.trading_halt,
        health_monitor=container.health,
        metrics=container.metrics,
        alert_manager=container.alerts,
        strategy_engine=container.strategy_engine,
        instruments=container.instruments,
    )
    print(f"Dashboard on http://{args.host}:{args.port} (no authentication — bind to localhost)")
    uvicorn.run(create_app(state), host=args.host, port=args.port, log_level="warning")
    return 0


# ---- helpers ----------------------------------------------------------------


def _load_bars(path: str) -> list:
    """Load OHLCV bars from CSV or JSON."""
    from datetime import datetime, timezone

    from data.models import Bar

    file = Path(path)
    if not file.exists():
        return []

    rows: list[dict] = []
    if file.suffix.lower() == ".json":
        rows = json.loads(file.read_text())
    else:
        import csv

        with file.open(newline="") as handle:
            rows = list(csv.DictReader(handle))

    bars = []
    for row in rows:
        raw_ts = str(row.get("timestamp") or row.get("date") or row.get("time"))
        try:
            ts = datetime.fromisoformat(raw_ts.replace("Z", "+00:00"))
        except ValueError:
            continue
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        try:
            bars.append(
                Bar(
                    timestamp=ts,
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row.get("volume") or 0),
                )
            )
        except (KeyError, ValueError):
            continue
    return bars


def _gateway_and_feed_for_mode(mode: TradingMode, container, args, ibkr_client=None):
    """Decide which broker gateway and market data feed a run should use.

    Pulled out as its own small function so the mode-dependent decision is
    independently testable without needing a real IBKR connection: given a
    fake `ibkr_client`, this can be exercised directly.

    - SIMULATION: the in-memory `SimulatedBrokerGateway` already set up by
      `build_container()`, and an empty `MarketDataFeed()` — completely
      unchanged from before this fix.
    - PAPER: a real `IBKROrderGateway` wrapping the caller-supplied
      `ibkr_client`, and a `LiveMarketDataFeed` reading from that same
      client. `ibkr_client` must already be connected; this function does
      not connect it.
    """
    from app.control_loop import MarketDataFeed
    from broker.live_feed import LiveMarketDataFeed
    from broker.order_manager import IBKROrderGateway

    if mode is TradingMode.PAPER:
        if ibkr_client is None:
            raise ValueError("PAPER mode requires an already-connected ibkr_client")
        gateway = IBKROrderGateway(ibkr_client._ib)  # noqa: SLF001
        feed = LiveMarketDataFeed(
            ibkr_client,
            container.instruments,
            bar_size=args.bar_size,
            history_duration=args.history_duration,
        )
        return gateway, feed

    return container.gateway, MarketDataFeed()


def _run_loop(args: argparse.Namespace, mode: TradingMode) -> int:
    from app.control_loop import ControlLoop

    settings = get_settings()
    ibkr_client = None

    async def run() -> None:
        nonlocal ibkr_client
        container = build_container(
            settings, symbols=args.symbols, strategy_names=args.strategies
        )

        if mode is TradingMode.PAPER:
            from broker.ibkr_client import IBKRClient
            from broker.market_data import MarketDataType

            md_type = MarketDataType[settings.ibkr.market_data_type.upper()]
            ibkr_client = IBKRClient(settings.ibkr, market_data_type=md_type)
            print(
                f"Connecting to IBKR at {settings.ibkr.host}:{settings.ibkr.port} "
                f"(market data: {md_type.name})..."
            )
            try:
                await ibkr_client.connect()
            except Exception as exc:  # noqa: BLE001
                print(f"\nCould not connect to IBKR: {exc}", file=sys.stderr)
                print(
                    "Check that TWS/Gateway is running, logged into paper, and API "
                    "access is enabled (see `python scripts/smoke_test_ibkr.py`).",
                    file=sys.stderr,
                )
                return

        gateway, feed = _gateway_and_feed_for_mode(mode, container, args, ibkr_client)

        # Rebuild the container's order manager against the real gateway
        # for PAPER — build_container() always defaults to the simulated
        # one, and swapping it here (rather than changing that default)
        # keeps `status`/`risk`/etc. construction-only and side-effect-free
        # for every other command.
        from broker.order_manager import OrderManager

        order_manager = OrderManager(gateway, container.order_store, container.mode_gate)

        if mode is TradingMode.PAPER:
            print("Fetching historical data to warm up strategies...")
            try:
                await feed.start()
            except Exception as exc:  # noqa: BLE001
                print(f"\nCould not warm up market data: {exc}", file=sys.stderr)
                print(
                    "This is the same check `smoke_test_ibkr.py` performs — if that "
                    "script also fails, resolve it first; this loop will not run "
                    "without real history to seed strategies with.",
                    file=sys.stderr,
                )
                await ibkr_client.disconnect()
                return
            print(f"Warmed up on {len(container.instruments)} instrument(s).\n")

        loop = ControlLoop(
            instruments=container.instruments,
            feed=feed,
            strategy_engine=container.strategy_engine,
            risk_engine=container.risk_engine,
            validator=container.validator,
            order_manager=order_manager,
            order_store=container.order_store,
            portfolio=container.portfolio,
            reconciler=container.reconciler,
            kill_switch=container.kill_switch,
            trading_halt=container.trading_halt,
            mode_gate=container.mode_gate,
            ai_engine=container.ai_engine,
            regime_detector=container.regime_detector,
            metrics=container.metrics,
            alerts=container.alerts,
            journal=container.journal,
            cycle_seconds=args.cycle_seconds,
        )

        print(f"Starting control loop: {len(container.instruments)} instruments, "
              f"cycle {args.cycle_seconds}s")
        if mode is TradingMode.PAPER:
            print(
                "This is connected to your REAL IBKR paper account. It WILL decide "
                "and submit orders autonomously against it, using pretend money."
            )
        print("Press Ctrl+C to stop.\n")

        background = None
        try:
            if mode is TradingMode.PAPER:
                background = asyncio.create_task(feed.run_ingest())
            await loop.start(max_cycles=args.max_cycles)
        except KeyboardInterrupt:
            loop.stop()
        finally:
            if background is not None:
                background.cancel()
            if ibkr_client is not None:
                await ibkr_client.disconnect()

        nonlocal_stats["cycles"] = loop.stats.cycles
        nonlocal_stats["orders"] = loop.stats.orders_submitted

    nonlocal_stats = {"cycles": 0, "orders": 0}
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("\nStopped by operator.")
    print(f"\nCycles: {nonlocal_stats['cycles']}  Orders submitted: {nonlocal_stats['orders']}")
    return 0


# ---- parser -------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="trading_agent",
        description="Autonomous trading agent for Interactive Brokers. "
        "Default mode is PAPER; there is no `live` command.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--symbols", nargs="+", default=["SPY"], help="Instrument symbols")

    def add_loop_args(p: argparse.ArgumentParser) -> None:
        add_common(p)
        p.add_argument("--strategies", nargs="+", default=["ma_crossover"])
        p.add_argument("--cycle-seconds", type=float, default=5.0)
        p.add_argument("--max-cycles", type=int, default=None)
        p.add_argument(
            "--bar-size", default="1 min",
            help="Only used by `paper` (live IBKR feed). Ignored by `simulate`.",
        )
        p.add_argument(
            "--history-duration", default="2 D",
            help="How much history to warm up strategies with. Only used by `paper`.",
        )

    p = sub.add_parser("status", help="Show system status and health")
    add_common(p)
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("strategies", help="List registered strategies")
    p.set_defaults(func=cmd_strategies)

    p = sub.add_parser("positions", help="Show open positions")
    add_common(p)
    p.set_defaults(func=cmd_positions)

    p = sub.add_parser("risk", help="Show risk limits and utilisation")
    add_common(p)
    p.set_defaults(func=cmd_risk)

    p = sub.add_parser("backtest", help="Run a backtest")
    p.add_argument("--strategy", required=True)
    p.add_argument("--data", required=True, help="CSV or JSON file of OHLCV bars")
    p.add_argument("--symbol", default="SPY")
    p.add_argument("--equity", type=float, default=100000)
    p.add_argument("--bar-size", default="1 day")
    p.set_defaults(func=cmd_backtest)

    p = sub.add_parser(
        "overfitting-check",
        help="Grid-search a strategy and compute Deflated Sharpe Ratio (statistical "
             "overfitting correction, not just a bigger backtest)",
    )
    p.add_argument("--strategy", required=True)
    p.add_argument("--data", required=True, help="CSV or JSON file of OHLCV bars")
    p.add_argument("--symbol", default="SPY")
    p.add_argument("--equity", type=float, default=100000)
    p.add_argument("--bar-size", default="1 day")
    p.add_argument(
        "--grid", required=True,
        help='JSON object of parameter -> list of values, e.g. '
             '\'{"fast_period": [5,10,15], "slow_period": [20,30,40]}\'',
    )
    p.set_defaults(func=cmd_overfitting_check)

    p = sub.add_parser("simulate", help="Run with an internal fill simulator")
    add_loop_args(p)
    p.add_argument(
        "--data", default=None,
        help="CSV/JSON of OHLCV bars to replay through the loop cycle-by-cycle. "
             "Without this, there is no market data source and nothing will trade.",
    )
    p.set_defaults(func=cmd_simulate)

    p = sub.add_parser("paper", help="Run against an IBKR paper account")
    add_loop_args(p)
    p.set_defaults(func=cmd_paper)

    p = sub.add_parser("reconcile", help="Compare local state against the broker")
    add_common(p)
    p.set_defaults(func=cmd_reconcile)

    p = sub.add_parser("kill-switch", help="Activate the kill switch (no deactivation)")
    add_common(p)
    p.add_argument("--reason", default="Activated from CLI")
    p.set_defaults(func=cmd_kill_switch)

    p = sub.add_parser("explain", help="Explain decisions from the audit trail")
    p.add_argument("--audit-log", default="logs/decisions.jsonl")
    p.add_argument("--record-id", default=None, help="Explain one record (prefix match)")
    p.add_argument("--limit", type=int, default=5)
    p.set_defaults(func=cmd_explain)

    p = sub.add_parser("macro", help="Manage macro/global-event context (operator hypotheses, not live news)")
    macro_sub = p.add_subparsers(dest="macro_command", required=True)

    ma = macro_sub.add_parser("add", help="Add a macro factor")
    ma.add_argument("--name", required=True)
    ma.add_argument("--category", required=True,
                     choices=["CLIMATE", "MONETARY_POLICY", "GEOPOLITICAL",
                              "COMMODITY_SUPPLY", "REGULATORY", "OTHER"])
    ma.add_argument("--stance", default="MIXED_UNCERTAIN",
                     choices=["POSSIBLE_TAILWIND", "POSSIBLE_HEADWIND", "MIXED_UNCERTAIN"])
    ma.add_argument("--description", default="")
    ma.add_argument("--sectors", nargs="*", default=[])
    ma.add_argument("--symbols", nargs="*", default=[])
    ma.add_argument("--confidence", type=float, default=0.5)
    ma.add_argument("--source", default="", help="Citation or note on where this came from")
    ma.add_argument("--expires-in-days", type=int, required=True,
                     help="Macro theses go stale; you must set an expiry")
    ma.add_argument("--store", default="macro_context.json")
    ma.set_defaults(func=cmd_macro_add)

    ml = macro_sub.add_parser("list", help="List macro factors")
    ml.add_argument("--store", default="macro_context.json")
    ml.add_argument("--all", action="store_true", help="Include expired factors")
    ml.set_defaults(func=cmd_macro_list)

    mr = macro_sub.add_parser("remove", help="Remove a macro factor by name")
    mr.add_argument("--name", required=True)
    mr.add_argument("--store", default="macro_context.json")
    mr.set_defaults(func=cmd_macro_remove)

    p = sub.add_parser("reflect", help="Analyse past trades and generate hypotheses (advisory only)")
    p.add_argument("--strategy", required=True)
    p.add_argument("--data", required=True, help="CSV or JSON file of OHLCV bars")
    p.add_argument("--symbol", default="SPY")
    p.add_argument("--equity", type=float, default=100000)
    p.add_argument("--bar-size", default="1 day")
    p.set_defaults(func=cmd_reflect)

    p = sub.add_parser("migrate", help="Apply database migrations")
    p.add_argument("--revision", default="head")
    p.set_defaults(func=cmd_migrate)

    p = sub.add_parser("dashboard", help="Serve the monitoring dashboard")
    add_common(p)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(func=cmd_dashboard)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = get_settings()
    configure_logging(
        level=settings.log_level,
        json_output=settings.log_format == "json",
        mode=str(settings.trading_mode),
    )
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
