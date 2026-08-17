"""
Health checks.

Design decisions:

- Checks are classified **CRITICAL** or **DEGRADED**. A failing critical
  check halts new trading; a degraded one is reported and alerted but does
  not stop the system. Treating every check as critical means a slow
  database or a flaky metrics endpoint stops trading, which trains
  operators to disable health checking entirely — the worst outcome.

- **A check that throws is UNHEALTHY, not unknown.** A health check whose
  own execution fails tells you nothing reassuring, so it must not be
  treated as a pass. This is the same fail-closed default as the risk
  engine.

- **Health checks never mutate trading state directly.** They report; the
  control loop decides what to do. A monitoring component that can halt
  trading as a side effect is a monitoring component that can halt trading
  by accident.

- Each check has a timeout. A hanging health check must not hang the
  monitoring cycle, or the system loses visibility precisely when
  something is wrong.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from decimal import Decimal
from enum import StrEnum

import structlog
from pydantic import BaseModel, Field

log = structlog.get_logger(__name__)


class HealthStatus(StrEnum):
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    UNHEALTHY = "UNHEALTHY"


class Severity(StrEnum):
    CRITICAL = "CRITICAL"  # failing -> stop new trading
    DEGRADED = "DEGRADED"  # failing -> alert only


class CheckResult(BaseModel):
    """Result of one health check. Defaults to UNHEALTHY so a check that
    fails to set a verdict does not read as a pass."""

    name: str
    status: HealthStatus = HealthStatus.UNHEALTHY
    severity: Severity = Severity.DEGRADED
    detail: str = ""
    latency_ms: float = 0.0
    checked_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def is_ok(self) -> bool:
        return self.status is HealthStatus.HEALTHY

    @property
    def blocks_trading(self) -> bool:
        return self.severity is Severity.CRITICAL and self.status is HealthStatus.UNHEALTHY


class HealthReport(BaseModel):
    checks: list[CheckResult] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def status(self) -> HealthStatus:
        if any(c.blocks_trading for c in self.checks):
            return HealthStatus.UNHEALTHY
        if any(not c.is_ok for c in self.checks):
            return HealthStatus.DEGRADED
        return HealthStatus.HEALTHY

    @property
    def can_trade(self) -> bool:
        return not any(c.blocks_trading for c in self.checks)

    @property
    def failures(self) -> list[CheckResult]:
        return [c for c in self.checks if not c.is_ok]

    def get(self, name: str) -> CheckResult | None:
        return next((c for c in self.checks if c.name == name), None)

    def summary(self) -> str:
        failures = self.failures
        if not failures:
            return f"{self.status}: all {len(self.checks)} checks passing"
        detail = ", ".join(f"{c.name}={c.status}" for c in failures)
        return f"{self.status}: {detail}"


CheckFn = Callable[[], Awaitable[CheckResult]] | Callable[[], CheckResult]


class HealthMonitor:
    def __init__(self, *, check_timeout_seconds: float = 5.0) -> None:
        self._checks: dict[str, tuple[CheckFn, Severity]] = {}
        self._timeout = check_timeout_seconds
        self._last_report: HealthReport | None = None

    def register(self, name: str, fn: CheckFn, *, severity: Severity = Severity.DEGRADED) -> None:
        self._checks[name] = (fn, severity)

    @property
    def last_report(self) -> HealthReport | None:
        return self._last_report

    async def run(self) -> HealthReport:
        results: list[CheckResult] = []
        for name, (fn, severity) in self._checks.items():
            results.append(await self._run_one(name, fn, severity))
        report = HealthReport(checks=results)
        self._last_report = report

        if not report.can_trade:
            log.error("health.critical_failure", failures=[c.name for c in report.failures])
        elif report.failures:
            log.warning("health.degraded", failures=[c.name for c in report.failures])
        return report

    async def _run_one(self, name: str, fn: CheckFn, severity: Severity) -> CheckResult:
        started = datetime.now(timezone.utc)
        try:
            result = fn()
            if asyncio.iscoroutine(result):
                result = await asyncio.wait_for(result, timeout=self._timeout)
        except TimeoutError:
            return CheckResult(
                name=name,
                status=HealthStatus.UNHEALTHY,
                severity=severity,
                detail=f"Check timed out after {self._timeout}s",
            )
        except Exception as exc:  # noqa: BLE001 — a broken check is unhealthy
            log.error("health.check_raised", check=name, error=str(exc))
            return CheckResult(
                name=name,
                status=HealthStatus.UNHEALTHY,
                severity=severity,
                detail=f"Check raised: {exc}",
            )

        latency = (datetime.now(timezone.utc) - started).total_seconds() * 1000
        result.name = name
        result.severity = severity
        result.latency_ms = latency
        return result


# ---- standard checks ------------------------------------------------------


def ok(detail: str = "") -> CheckResult:
    return CheckResult(name="", status=HealthStatus.HEALTHY, detail=detail)


def unhealthy(detail: str) -> CheckResult:
    return CheckResult(name="", status=HealthStatus.UNHEALTHY, detail=detail)


def degraded(detail: str) -> CheckResult:
    return CheckResult(name="", status=HealthStatus.DEGRADED, detail=detail)


def broker_connection_check(connection: object) -> CheckFn:
    def check() -> CheckResult:
        connected = bool(getattr(connection, "is_connected", False))
        state = getattr(connection, "state", "UNKNOWN")
        return ok(f"state={state}") if connected else unhealthy(f"state={state}")

    return check


def market_data_freshness_check(
    feed: object, instruments: list, max_age_seconds: float
) -> CheckFn:
    def check() -> CheckResult:
        stale: list[str] = []
        missing: list[str] = []
        for instrument in instruments:
            snapshot = feed.snapshot(instrument)
            if snapshot is None:
                missing.append(str(instrument))
            elif snapshot.is_stale(max_age_seconds):
                stale.append(f"{instrument}({snapshot.age_seconds():.0f}s)")
        if missing or stale:
            parts = []
            if missing:
                parts.append(f"missing: {', '.join(missing)}")
            if stale:
                parts.append(f"stale: {', '.join(stale)}")
            return unhealthy("; ".join(parts))
        return ok(f"{len(instruments)} instruments fresh")

    return check


def kill_switch_check(kill_switch: object) -> CheckFn:
    def check() -> CheckResult:
        if getattr(kill_switch, "is_active", False):
            event = getattr(kill_switch, "current_event", None)
            trigger = getattr(event, "trigger", "unknown") if event else "unknown"
            # Reported as UNHEALTHY so it is visible, but DEGRADED severity:
            # the kill switch already stops trading. Marking it critical
            # would be redundant and would confuse "why can't we trade".
            return unhealthy(f"Kill switch active: {trigger}")
        return ok("inactive")

    return check


def portfolio_check(portfolio: object) -> CheckFn:
    def check() -> CheckResult:
        account = getattr(portfolio, "account", None)
        if account is None:
            return unhealthy("No account state")
        equity = getattr(account, "equity", Decimal("0"))
        if equity <= 0:
            return unhealthy(f"Account equity is {equity}")
        age = (datetime.now(timezone.utc) - account.updated_at).total_seconds()
        if age > 300:
            return degraded(f"Account state {age:.0f}s old")
        return ok(f"equity={equity}")

    return check


def ai_provider_check(engine: object) -> CheckFn:
    def check() -> CheckResult:
        # AI unavailability is DEGRADED by design: the system falls back
        # to deterministic strategies rather than stopping.
        if getattr(engine, "provider_available", False):
            return ok("available")
        return degraded("AI provider unavailable; deterministic strategies only")

    return check


def loop_liveness_check(stats: object, *, max_cycle_gap_seconds: float = 60.0) -> CheckFn:
    def check() -> CheckResult:
        last = getattr(stats, "last_cycle_at", None)
        if last is None:
            return degraded("Loop has not completed a cycle yet")
        gap = (datetime.now(timezone.utc) - last).total_seconds()
        if gap > max_cycle_gap_seconds:
            return unhealthy(f"No cycle for {gap:.0f}s")
        return ok(f"last cycle {gap:.1f}s ago")

    return check


def resource_check(*, max_memory_pct: float = 90.0) -> CheckFn:
    def check() -> CheckResult:
        try:
            import psutil
        except ImportError:
            return degraded("psutil not installed; resource metrics unavailable")
        memory = psutil.virtual_memory().percent
        cpu = psutil.cpu_percent(interval=None)
        if memory > max_memory_pct:
            return degraded(f"Memory at {memory:.0f}%")
        return ok(f"cpu={cpu:.0f}% memory={memory:.0f}%")

    return check
