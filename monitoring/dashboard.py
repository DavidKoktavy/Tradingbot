"""
Monitoring dashboard (FastAPI).

Design decisions:

- **The dashboard is read-only, with exactly one exception.** Every
  endpoint is a GET that reads state. The single mutating endpoint is
  `POST /kill-switch/activate`, because an operator needs a way to stop
  the system that does not require shell access during an incident.
  Notably there is **no endpoint to deactivate** the kill switch, raise a
  limit, enable live trading, or submit an order — those require
  deliberate operator action outside the web surface, where they can be
  reviewed.

- **A dashboard failure must never affect trading.** Every handler
  degrades to partial data rather than raising, because a 500 in a
  monitoring endpoint should not look like a trading problem, and an
  exception here must not propagate into shared state.

- **The mode is displayed prominently on every response**, per the spec's
  requirement that it always be obvious whether the system is
  BACKTEST/SIMULATION/PAPER/LIVE. Ambiguity about mode is how people
  trade real money believing they are on paper.

- No secrets are ever serialised. Account identifiers, API keys, and
  connection details are omitted entirely rather than masked.
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

import structlog

log = structlog.get_logger(__name__)


class DashboardState:
    """Aggregates references to the live components. Holds no state of its
    own, so it cannot drift from reality."""

    def __init__(
        self,
        *,
        mode_gate: Any,
        portfolio: Any,
        order_store: Any,
        risk_engine: Any,
        kill_switch: Any,
        trading_halt: Any,
        health_monitor: Any,
        metrics: Any,
        alert_manager: Any,
        loop_stats: Any = None,
        strategy_engine: Any = None,
        feed: Any = None,
        instruments: list | None = None,
    ) -> None:
        self.mode_gate = mode_gate
        self.portfolio = portfolio
        self.order_store = order_store
        self.risk_engine = risk_engine
        self.kill_switch = kill_switch
        self.trading_halt = trading_halt
        self.health = health_monitor
        self.metrics = metrics
        self.alerts = alert_manager
        self.loop_stats = loop_stats
        self.strategy_engine = strategy_engine
        self.feed = feed
        self.instruments = instruments or []

    # ---- view builders (pure reads, exception-safe) ----------------------

    def _prices(self) -> dict[str, Decimal]:
        prices: dict[str, Decimal] = {}
        if self.feed is None:
            return prices
        for instrument in self.instruments:
            snapshot = self.feed.snapshot(instrument)
            if snapshot is not None and snapshot.mid is not None:
                prices[str(instrument)] = Decimal(str(snapshot.mid))
        return prices

    def mode_view(self) -> dict[str, Any]:
        return {
            "mode": str(self.mode_gate.mode),
            "is_live": self.mode_gate.is_live,
            "uses_real_broker": self.mode_gate.uses_real_broker,
        }

    def account_view(self) -> dict[str, Any]:
        try:
            account = self.portfolio.account
            prices = self._prices()
            view: dict[str, Any] = {
                "equity": str(account.equity),
                "cash": str(account.cash),
                "buying_power": str(account.buying_power),
                "maintenance_margin": str(account.maintenance_margin),
                "realised_pnl": str(self.portfolio.realized_pnl),
                "start_of_day_equity": str(self.portfolio.start_of_day_equity or "0"),
                "updated_at": account.updated_at.isoformat(),
            }
            try:
                view["unrealised_pnl"] = str(self.portfolio.unrealized_pnl(prices))
                view["daily_pnl"] = str(self.portfolio.daily_pnl(prices))
            except Exception as exc:  # noqa: BLE001
                # Missing marks: report the gap rather than a wrong number.
                view["unrealised_pnl"] = None
                view["daily_pnl"] = None
                view["pnl_unavailable_reason"] = str(exc)
            return view
        except Exception as exc:  # noqa: BLE001
            log.error("dashboard.account_view_failed", error=str(exc))
            return {"error": str(exc)}

    def positions_view(self) -> list[dict[str, Any]]:
        try:
            prices = self._prices()
            out = []
            for key, position in self.portfolio.positions.items():
                if position.is_flat:
                    continue
                price = prices.get(key)
                out.append(
                    {
                        "instrument": key,
                        "quantity": str(position.quantity),
                        "average_cost": str(position.average_cost),
                        "current_price": str(price) if price else None,
                        "market_value": str(position.market_value(price)) if price else None,
                        "exposure": str(position.exposure(price)) if price else None,
                        "unrealised_pnl": (
                            str(position.unrealized_pnl(price)) if price else None
                        ),
                        "realised_pnl": str(position.realized_pnl),
                        "commission": str(position.total_commission),
                    }
                )
            return out
        except Exception as exc:  # noqa: BLE001
            log.error("dashboard.positions_view_failed", error=str(exc))
            return []

    def orders_view(self, limit: int = 50) -> list[dict[str, Any]]:
        try:
            orders = sorted(
                self.order_store.all_orders(), key=lambda o: o.created_at, reverse=True
            )[:limit]
            return [
                {
                    "order_id": o.order_id,
                    "broker_order_id": o.broker_order_id,
                    "instrument": str(o.intent.instrument),
                    "side": str(o.intent.side),
                    "quantity": str(o.intent.quantity),
                    "filled_quantity": str(o.filled_quantity),
                    "average_fill_price": (
                        str(o.average_fill_price) if o.average_fill_price else None
                    ),
                    "order_type": str(o.intent.order_type),
                    "state": str(o.state),
                    "source": o.intent.source,
                    "strategy": o.intent.strategy,
                    "created_at": o.created_at.isoformat(),
                    "error": o.error_message,
                }
                for o in orders
            ]
        except Exception as exc:  # noqa: BLE001
            log.error("dashboard.orders_view_failed", error=str(exc))
            return []

    def risk_view(self) -> dict[str, Any]:
        try:
            limits = self.risk_engine.limits
            account = self.portfolio.account
            prices = self._prices()
            equity = account.equity

            view: dict[str, Any] = {
                "kill_switch_active": self.kill_switch.is_active,
                "kill_switch_trigger": (
                    str(self.kill_switch.current_event.trigger)
                    if self.kill_switch.current_event
                    else None
                ),
                "emergency_policy": str(self.kill_switch.emergency_policy),
                "trading_halted": self.trading_halt.is_halted,
                "halt_reasons": {str(k): v for k, v in self.trading_halt.reasons.items()},
                "open_positions": self.portfolio.open_position_count,
                "peak_equity": str(self.risk_engine.peak_equity or "0"),
                "drawdown_pct": float(self.risk_engine.current_drawdown(equity)),
                "limits": {
                    "max_risk_per_trade": float(limits.max_risk_per_trade),
                    "max_daily_loss": float(limits.max_daily_loss),
                    "max_portfolio_drawdown": float(limits.max_portfolio_drawdown),
                    "max_position_size": float(limits.max_position_size),
                    "max_gross_exposure": float(limits.max_gross_exposure),
                    "max_open_positions": limits.max_open_positions,
                    "max_orders_per_minute": limits.max_orders_per_minute,
                },
                "orders_last_minute": self.risk_engine.rate_limiter.current_count,
            }
            try:
                gross = self.portfolio.gross_exposure(prices)
                view["gross_exposure"] = str(gross)
                view["gross_exposure_pct"] = (
                    float(gross / equity) if equity else None
                )
                view["risk_utilisation"] = {
                    "exposure": (
                        float(gross / equity / limits.max_gross_exposure)
                        if equity
                        else None
                    ),
                    "positions": (
                        self.portfolio.open_position_count / limits.max_open_positions
                    ),
                }
            except Exception:  # noqa: BLE001
                view["gross_exposure"] = None
                view["gross_exposure_pct"] = None
            return view
        except Exception as exc:  # noqa: BLE001
            log.error("dashboard.risk_view_failed", error=str(exc))
            return {"error": str(exc)}

    def system_view(self) -> dict[str, Any]:
        try:
            report = self.health.last_report
            stats = self.loop_stats
            return {
                "health": str(report.status) if report else "UNKNOWN",
                "can_trade": report.can_trade if report else False,
                "checks": (
                    [
                        {
                            "name": c.name,
                            "status": str(c.status),
                            "severity": str(c.severity),
                            "detail": c.detail,
                            "latency_ms": round(c.latency_ms, 2),
                        }
                        for c in report.checks
                    ]
                    if report
                    else []
                ),
                "uptime_seconds": round(self.metrics.uptime_seconds, 1),
                "cycles": getattr(stats, "cycles", 0) if stats else 0,
                "consecutive_failures": (
                    getattr(stats, "consecutive_failures", 0) if stats else 0
                ),
                "last_cycle_at": (
                    stats.last_cycle_at.isoformat()
                    if stats and stats.last_cycle_at
                    else None
                ),
                "alert_delivery_failures": self.alerts.delivery_failures,
            }
        except Exception as exc:  # noqa: BLE001
            log.error("dashboard.system_view_failed", error=str(exc))
            return {"error": str(exc)}

    def strategies_view(self) -> dict[str, Any]:
        if self.strategy_engine is None:
            return {"active": [], "disabled": []}
        try:
            return {
                "active": [
                    {"name": s.name, "version": s.version}
                    for s in self.strategy_engine.active_strategies
                ],
                "disabled": sorted(self.strategy_engine.disabled_strategies),
            }
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)}

    def alerts_view(self, limit: int = 20) -> list[dict[str, Any]]:
        try:
            return [
                {
                    "key": a.key,
                    "category": str(a.category),
                    "severity": str(a.severity),
                    "title": a.title,
                    "detail": a.detail,
                    "raised_at": a.raised_at.isoformat(),
                }
                for a in self.alerts.recent(limit)
            ]
        except Exception as exc:  # noqa: BLE001
            return []

    def full_snapshot(self) -> dict[str, Any]:
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "mode": self.mode_view(),
            "account": self.account_view(),
            "positions": self.positions_view(),
            "risk": self.risk_view(),
            "system": self.system_view(),
            "strategies": self.strategies_view(),
            "orders": self.orders_view(limit=20),
            "alerts": self.alerts_view(),
        }


def create_app(state: DashboardState) -> Any:
    """Build the FastAPI app. Imported lazily so the trading system does
    not depend on FastAPI being installed."""
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse, PlainTextResponse

    app = FastAPI(title="Trading Agent Dashboard", version="0.1.0")

    @app.get("/health")
    async def health() -> dict[str, Any]:
        report = await state.health.run()
        return {
            "status": str(report.status),
            "can_trade": report.can_trade,
            "mode": str(state.mode_gate.mode),
            "checks": [
                {"name": c.name, "status": str(c.status), "detail": c.detail}
                for c in report.checks
            ],
        }

    @app.get("/api/snapshot")
    async def snapshot() -> dict[str, Any]:
        return state.full_snapshot()

    @app.get("/api/account")
    async def account() -> dict[str, Any]:
        return {"mode": state.mode_view(), "account": state.account_view()}

    @app.get("/api/positions")
    async def positions() -> dict[str, Any]:
        return {"mode": state.mode_view(), "positions": state.positions_view()}

    @app.get("/api/orders")
    async def orders(limit: int = 50) -> dict[str, Any]:
        return {"mode": state.mode_view(), "orders": state.orders_view(limit)}

    @app.get("/api/risk")
    async def risk() -> dict[str, Any]:
        return {"mode": state.mode_view(), "risk": state.risk_view()}

    @app.get("/api/alerts")
    async def alerts(limit: int = 20) -> dict[str, Any]:
        return {"mode": state.mode_view(), "alerts": state.alerts_view(limit)}

    @app.get("/metrics", response_class=PlainTextResponse)
    async def metrics() -> str:
        return state.metrics.to_prometheus()

    @app.post("/kill-switch/activate")
    async def activate_kill_switch(reason: str = "Activated from dashboard") -> dict[str, Any]:
        """The ONLY mutating endpoint. There is deliberately no
        deactivation endpoint: resuming trading requires deliberate
        operator action outside the web surface."""
        from risk.kill_switch import KillSwitchTrigger

        event = state.kill_switch.activate(KillSwitchTrigger.MANUAL, reason)
        log.critical("dashboard.kill_switch_activated", reason=reason)
        return {
            "activated": True,
            "trigger": str(event.trigger),
            "detail": event.detail,
            "note": "Deactivation is not available via the dashboard.",
        }

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return _render_html(state)

    return app


def _render_html(state: DashboardState) -> str:
    """Minimal server-rendered dashboard. Deliberately dependency-free:
    an operator opening this during an incident should not be waiting on a
    CDN."""
    snapshot = state.full_snapshot()
    mode = snapshot["mode"]["mode"]
    is_live = snapshot["mode"]["is_live"]
    mode_colour = "#c0392b" if is_live else "#27ae60"

    risk = snapshot["risk"]
    account = snapshot["account"]
    system = snapshot["system"]

    def rows(items: list[dict[str, Any]], columns: list[str]) -> str:
        if not items:
            return f'<tr><td colspan="{len(columns)}" class="muted">None</td></tr>'
        out = []
        for item in items:
            cells = "".join(f"<td>{item.get(c) if item.get(c) is not None else '—'}</td>"
                            for c in columns)
            out.append(f"<tr>{cells}</tr>")
        return "".join(out)

    banner = ""
    if risk.get("kill_switch_active"):
        banner = (
            '<div class="banner critical">KILL SWITCH ACTIVE — '
            f'{risk.get("kill_switch_trigger")}</div>'
        )
    elif risk.get("trading_halted"):
        banner = (
            '<div class="banner warn">TRADING HALTED — '
            f'{", ".join(risk.get("halt_reasons", {}))}</div>'
        )

    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Trading Agent — {mode}</title>
<meta http-equiv="refresh" content="10">
<style>
 body{{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;margin:0;background:#0f1115;color:#e6e6e6}}
 header{{padding:12px 20px;background:{mode_colour};color:#fff;font-weight:700;font-size:18px}}
 .banner{{padding:10px 20px;font-weight:700}}
 .critical{{background:#c0392b;color:#fff}} .warn{{background:#e67e22;color:#fff}}
 .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px;padding:20px}}
 .card{{background:#181b21;border:1px solid #262a33;border-radius:8px;padding:14px}}
 h2{{margin:0 0 10px;font-size:13px;text-transform:uppercase;letter-spacing:.08em;color:#8b93a1}}
 table{{width:100%;border-collapse:collapse;font-size:13px}}
 td,th{{padding:4px 6px;text-align:left;border-bottom:1px solid #262a33}}
 th{{color:#8b93a1;font-weight:600}}
 .muted{{color:#6b7280}} .k{{color:#8b93a1}}
 .full{{grid-column:1/-1}}
</style></head><body>
<header>TRADING AGENT — MODE: {mode}{' ⚠ REAL MONEY' if is_live else ''}</header>
{banner}
<div class="grid">
 <div class="card"><h2>Account</h2><table>
  <tr><td class="k">Equity</td><td>{account.get('equity')}</td></tr>
  <tr><td class="k">Cash</td><td>{account.get('cash')}</td></tr>
  <tr><td class="k">Buying power</td><td>{account.get('buying_power')}</td></tr>
  <tr><td class="k">Realised P&amp;L</td><td>{account.get('realised_pnl')}</td></tr>
  <tr><td class="k">Unrealised P&amp;L</td><td>{account.get('unrealised_pnl') or '—'}</td></tr>
  <tr><td class="k">Daily P&amp;L</td><td>{account.get('daily_pnl') or '—'}</td></tr>
 </table></div>

 <div class="card"><h2>Risk</h2><table>
  <tr><td class="k">Kill switch</td><td>{risk.get('kill_switch_active')}</td></tr>
  <tr><td class="k">Halted</td><td>{risk.get('trading_halted')}</td></tr>
  <tr><td class="k">Open positions</td><td>{risk.get('open_positions')} / {risk.get('limits',{}).get('max_open_positions')}</td></tr>
  <tr><td class="k">Gross exposure</td><td>{risk.get('gross_exposure') or '—'}</td></tr>
  <tr><td class="k">Drawdown</td><td>{risk.get('drawdown_pct',0):.2%}</td></tr>
  <tr><td class="k">Orders/min</td><td>{risk.get('orders_last_minute')} / {risk.get('limits',{}).get('max_orders_per_minute')}</td></tr>
 </table></div>

 <div class="card"><h2>System</h2><table>
  <tr><td class="k">Health</td><td>{system.get('health')}</td></tr>
  <tr><td class="k">Can trade</td><td>{system.get('can_trade')}</td></tr>
  <tr><td class="k">Uptime</td><td>{system.get('uptime_seconds')}s</td></tr>
  <tr><td class="k">Cycles</td><td>{system.get('cycles')}</td></tr>
  <tr><td class="k">Consecutive failures</td><td>{system.get('consecutive_failures')}</td></tr>
 </table></div>

 <div class="card full"><h2>Positions</h2><table>
  <tr><th>Instrument</th><th>Qty</th><th>Avg cost</th><th>Price</th><th>Unrealised</th><th>Exposure</th></tr>
  {rows(snapshot['positions'], ['instrument','quantity','average_cost','current_price','unrealised_pnl','exposure'])}
 </table></div>

 <div class="card full"><h2>Recent orders</h2><table>
  <tr><th>Instrument</th><th>Side</th><th>Qty</th><th>Filled</th><th>State</th><th>Source</th></tr>
  {rows(snapshot['orders'], ['instrument','side','quantity','filled_quantity','state','source'])}
 </table></div>

 <div class="card full"><h2>Recent alerts</h2><table>
  <tr><th>Severity</th><th>Title</th><th>Detail</th><th>At</th></tr>
  {rows(snapshot['alerts'], ['severity','title','detail','raised_at'])}
 </table></div>
</div></body></html>"""
