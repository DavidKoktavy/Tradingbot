"""
Alerting.

Design decisions:

- **Alerts are deduplicated with a cooldown per alert key.** An unhandled
  condition can fire every cycle; without suppression an operator receives
  thousands of identical messages and stops reading any of them. Alert
  fatigue is a safety problem, not a UX one.

- **Delivery failure never propagates.** If the notification provider is
  down, that is logged and counted, but it must not raise into the trading
  loop. The alert is still recorded in history so it is visible on the
  dashboard.

- **CRITICAL alerts bypass the cooldown on state change.** Suppressing a
  repeat is right; suppressing a *new* critical condition because a
  similar one fired recently is not. Dedup is keyed on the specific
  condition, not the severity class.

- The provider abstraction means email/Slack/PagerDuty can be added later
  without touching any calling code, per the spec's requirement for a
  notification abstraction.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from datetime import datetime, timedelta, timezone
from enum import StrEnum

import structlog
from pydantic import BaseModel, Field

log = structlog.get_logger(__name__)


class AlertSeverity(StrEnum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class AlertCategory(StrEnum):
    RISK_LIMIT = "RISK_LIMIT"
    LARGE_LOSS = "LARGE_LOSS"
    DRAWDOWN = "DRAWDOWN"
    BROKER = "BROKER"
    ORDER_REJECTION = "ORDER_REJECTION"
    UNEXPECTED_POSITION = "UNEXPECTED_POSITION"
    DATA_OUTAGE = "DATA_OUTAGE"
    AI_FAILURE = "AI_FAILURE"
    SYSTEM = "SYSTEM"
    KILL_SWITCH = "KILL_SWITCH"
    SLIPPAGE = "SLIPPAGE"
    RECONCILIATION = "RECONCILIATION"


class Alert(BaseModel):
    key: str
    category: AlertCategory
    severity: AlertSeverity
    title: str
    detail: str = ""
    context: dict[str, str] = Field(default_factory=dict)
    raised_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def render(self) -> str:
        lines = [f"[{self.severity}] {self.title}"]
        if self.detail:
            lines.append(self.detail)
        if self.context:
            lines.extend(f"  {k}: {v}" for k, v in sorted(self.context.items()))
        return "\n".join(lines)


class NotificationProvider(ABC):
    name: str = "abstract"

    @abstractmethod
    async def send(self, alert: Alert) -> None: ...


class LogNotificationProvider(NotificationProvider):
    """Default provider: structured log output. Always available, never
    fails, and keeps alerts in the audit trail even with no external
    integration configured."""

    name = "log"

    async def send(self, alert: Alert) -> None:
        logger = {
            AlertSeverity.INFO: log.info,
            AlertSeverity.WARNING: log.warning,
            AlertSeverity.CRITICAL: log.critical,
        }[alert.severity]
        logger(
            "alert",
            key=alert.key,
            category=alert.category,
            title=alert.title,
            detail=alert.detail,
            **alert.context,
        )


class CollectingProvider(NotificationProvider):
    """Test double."""

    name = "collecting"

    def __init__(self) -> None:
        self.alerts: list[Alert] = []

    async def send(self, alert: Alert) -> None:
        self.alerts.append(alert)


class FailingProvider(NotificationProvider):
    """Test double: always fails, to verify delivery failure containment."""

    name = "failing"

    async def send(self, alert: Alert) -> None:
        raise RuntimeError("notification provider unavailable")


class AlertManager:
    def __init__(
        self,
        providers: list[NotificationProvider] | None = None,
        *,
        cooldown_seconds: float = 300.0,
        history_size: int = 500,
    ) -> None:
        self._providers = providers or [LogNotificationProvider()]
        self._cooldown = timedelta(seconds=cooldown_seconds)
        self._last_sent: dict[str, datetime] = {}
        self._history: deque[Alert] = deque(maxlen=history_size)
        self._suppressed_count: dict[str, int] = {}
        self._delivery_failures = 0

    @property
    def history(self) -> list[Alert]:
        return list(self._history)

    @property
    def delivery_failures(self) -> int:
        return self._delivery_failures

    def recent(self, limit: int = 20) -> list[Alert]:
        return list(self._history)[-limit:][::-1]

    def suppressed_count(self, key: str) -> int:
        return self._suppressed_count.get(key, 0)

    async def raise_alert(
        self,
        *,
        key: str,
        category: AlertCategory,
        severity: AlertSeverity,
        title: str,
        detail: str = "",
        context: dict[str, str] | None = None,
        now: datetime | None = None,
    ) -> Alert | None:
        """Raise an alert. Returns the Alert if delivered, or None if
        suppressed by cooldown."""
        now = now or datetime.now(timezone.utc)

        last = self._last_sent.get(key)
        if last is not None and now - last < self._cooldown:
            self._suppressed_count[key] = self._suppressed_count.get(key, 0) + 1
            return None

        alert = Alert(
            key=key,
            category=category,
            severity=severity,
            title=title,
            detail=detail,
            context=context or {},
            raised_at=now,
        )
        self._last_sent[key] = now
        self._history.append(alert)

        for provider in self._providers:
            try:
                await provider.send(alert)
            except Exception as exc:  # noqa: BLE001 — delivery must not propagate
                self._delivery_failures += 1
                log.error(
                    "alert.delivery_failed",
                    provider=provider.name,
                    key=key,
                    error=str(exc),
                )
        return alert

    def clear_cooldown(self, key: str) -> None:
        """Called when a condition resolves, so its recurrence alerts
        immediately rather than being suppressed."""
        self._last_sent.pop(key, None)
        self._suppressed_count.pop(key, None)


# ---- convenience raisers -------------------------------------------------


async def alert_kill_switch(manager: AlertManager, trigger: str, detail: str) -> None:
    await manager.raise_alert(
        key=f"kill_switch:{trigger}",
        category=AlertCategory.KILL_SWITCH,
        severity=AlertSeverity.CRITICAL,
        title=f"Kill switch activated: {trigger}",
        detail=detail,
    )


async def alert_broker_disconnect(manager: AlertManager, detail: str) -> None:
    await manager.raise_alert(
        key="broker:disconnected",
        category=AlertCategory.BROKER,
        severity=AlertSeverity.CRITICAL,
        title="Broker connection lost",
        detail=detail,
    )


async def alert_daily_loss(manager: AlertManager, pnl_pct: float, limit_pct: float) -> None:
    await manager.raise_alert(
        key="risk:daily_loss",
        category=AlertCategory.RISK_LIMIT,
        severity=AlertSeverity.CRITICAL,
        title="Daily loss limit breached",
        detail=f"Daily P&L {pnl_pct:.2%} breached limit {limit_pct:.2%}",
    )


async def alert_drawdown(manager: AlertManager, drawdown: float, limit: float) -> None:
    await manager.raise_alert(
        key="risk:drawdown",
        category=AlertCategory.DRAWDOWN,
        severity=AlertSeverity.CRITICAL,
        title="Maximum drawdown breached",
        detail=f"Drawdown {drawdown:.2%} breached limit {limit:.2%}",
    )


async def alert_order_rejected(manager: AlertManager, order_id: str, reason: str) -> None:
    await manager.raise_alert(
        key=f"order:rejected:{reason}",
        category=AlertCategory.ORDER_REJECTION,
        severity=AlertSeverity.WARNING,
        title="Order rejected by broker",
        detail=reason,
        context={"order_id": order_id},
    )


async def alert_unexpected_position(manager: AlertManager, detail: str) -> None:
    await manager.raise_alert(
        key="reconciliation:unexpected_position",
        category=AlertCategory.UNEXPECTED_POSITION,
        severity=AlertSeverity.CRITICAL,
        title="Unexpected position found at broker",
        detail=detail,
    )


async def alert_data_outage(manager: AlertManager, detail: str) -> None:
    await manager.raise_alert(
        key="data:outage",
        category=AlertCategory.DATA_OUTAGE,
        severity=AlertSeverity.CRITICAL,
        title="Market data outage",
        detail=detail,
    )


async def alert_abnormal_slippage(
    manager: AlertManager, order_id: str, slippage_bps: float
) -> None:
    await manager.raise_alert(
        key="execution:slippage",
        category=AlertCategory.SLIPPAGE,
        severity=AlertSeverity.WARNING,
        title="Abnormal slippage observed",
        detail=f"{slippage_bps:.1f} bps",
        context={"order_id": order_id},
    )
