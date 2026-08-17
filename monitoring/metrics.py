"""
Metrics.

Design decisions:

- A tiny in-process registry rather than a Prometheus client dependency.
  The metrics we need are few and well-defined, and an in-process registry
  can be asserted against in tests without a scrape endpoint. Prometheus
  text-format export is provided so a real scraper can consume it.

- **Recording a metric never raises.** A monitoring failure must not
  propagate into the trading path. Every public method is wrapped.

- Histograms keep a bounded ring of recent observations rather than
  unbounded history, so a long-running process cannot leak memory through
  its own instrumentation.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from threading import Lock

import structlog

log = structlog.get_logger(__name__)


@dataclass
class Histogram:
    name: str
    max_samples: int = 1000
    _samples: deque[float] = field(default_factory=lambda: deque(maxlen=1000))

    def observe(self, value: float) -> None:
        self._samples.append(value)

    @property
    def count(self) -> int:
        return len(self._samples)

    @property
    def mean(self) -> float | None:
        return sum(self._samples) / len(self._samples) if self._samples else None

    def percentile(self, p: float) -> float | None:
        if not self._samples:
            return None
        ordered = sorted(self._samples)
        index = min(len(ordered) - 1, int(len(ordered) * p))
        return ordered[index]

    def snapshot(self) -> dict[str, float | None]:
        return {
            "count": self.count,
            "mean": self.mean,
            "p50": self.percentile(0.5),
            "p95": self.percentile(0.95),
            "p99": self.percentile(0.99),
        }


class MetricsRegistry:
    """Thread-safe registry of counters, gauges, and histograms."""

    def __init__(self) -> None:
        self._counters: dict[str, float] = {}
        self._gauges: dict[str, float] = {}
        self._histograms: dict[str, Histogram] = {}
        self._lock = Lock()
        self._started = time.monotonic()

    def increment(self, name: str, value: float = 1.0, **labels: str) -> None:
        try:
            key = self._key(name, labels)
            with self._lock:
                self._counters[key] = self._counters.get(key, 0.0) + value
        except Exception as exc:  # noqa: BLE001 — never break the caller
            log.error("metrics.increment_failed", metric=name, error=str(exc))

    def set_gauge(self, name: str, value: float, **labels: str) -> None:
        try:
            key = self._key(name, labels)
            with self._lock:
                self._gauges[key] = float(value)
        except Exception as exc:  # noqa: BLE001
            log.error("metrics.gauge_failed", metric=name, error=str(exc))

    def observe(self, name: str, value: float, **labels: str) -> None:
        try:
            key = self._key(name, labels)
            with self._lock:
                if key not in self._histograms:
                    self._histograms[key] = Histogram(name=key)
                self._histograms[key].observe(float(value))
        except Exception as exc:  # noqa: BLE001
            log.error("metrics.observe_failed", metric=name, error=str(exc))

    def timer(self, name: str, **labels: str) -> "_Timer":
        return _Timer(self, name, labels)

    # ---- reads ------------------------------------------------------------

    def counter(self, name: str, **labels: str) -> float:
        return self._counters.get(self._key(name, labels), 0.0)

    def gauge(self, name: str, **labels: str) -> float | None:
        return self._gauges.get(self._key(name, labels))

    def histogram(self, name: str, **labels: str) -> Histogram | None:
        return self._histograms.get(self._key(name, labels))

    @property
    def uptime_seconds(self) -> float:
        return time.monotonic() - self._started

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return {
                "uptime_seconds": self.uptime_seconds,
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "histograms": {k: h.snapshot() for k, h in self._histograms.items()},
            }

    def to_prometheus(self) -> str:
        """Prometheus text exposition format."""
        lines: list[str] = []
        with self._lock:
            lines.append(f"agent_uptime_seconds {self.uptime_seconds:.3f}")
            for key, value in sorted(self._counters.items()):
                lines.append(f"{key} {value}")
            for key, value in sorted(self._gauges.items()):
                lines.append(f"{key} {value}")
            for key, hist in sorted(self._histograms.items()):
                snap = hist.snapshot()
                base = key.split("{")[0]
                suffix = key[len(base):]
                lines.append(f"{base}_count{suffix} {snap['count']}")
                for stat in ("mean", "p50", "p95", "p99"):
                    if snap[stat] is not None:
                        lines.append(f"{base}_{stat}{suffix} {snap[stat]:.6f}")
        return "\n".join(lines) + "\n"

    def reset(self) -> None:
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()

    @staticmethod
    def _key(name: str, labels: dict[str, str]) -> str:
        if not labels:
            return name
        rendered = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
        return f"{name}{{{rendered}}}"


class _Timer:
    def __init__(self, registry: MetricsRegistry, name: str, labels: dict[str, str]) -> None:
        self._registry = registry
        self._name = name
        self._labels = labels
        self._start = 0.0

    def __enter__(self) -> "_Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        elapsed_ms = (time.perf_counter() - self._start) * 1000
        self._registry.observe(self._name, elapsed_ms, **self._labels)


# Canonical metric names, matching the spec's observability section.
class M:
    UPTIME = "agent_uptime_seconds"
    IBKR_CONNECTED = "ibkr_connection_status"
    MARKET_DATA_LATENCY = "market_data_latency_ms"
    MARKET_DATA_AGE = "market_data_age_seconds"
    ORDER_LATENCY = "order_latency_ms"
    ORDERS_SUBMITTED = "orders_submitted_total"
    ORDERS_FILLED = "orders_filled_total"
    ORDERS_REJECTED = "orders_rejected_total"
    ORDERS_CANCELLED = "orders_cancelled_total"
    DAILY_PNL = "daily_pnl"
    REALISED_PNL = "realised_pnl"
    UNREALISED_PNL = "unrealised_pnl"
    EQUITY = "account_equity"
    DRAWDOWN = "drawdown_pct"
    PORTFOLIO_EXPOSURE = "portfolio_gross_exposure_pct"
    OPEN_POSITIONS = "open_positions"
    RISK_REJECTIONS = "risk_rejections_total"
    AI_DECISIONS = "ai_decisions_total"
    AI_LATENCY = "ai_latency_ms"
    STRATEGY_SIGNALS = "strategy_signals_total"
    CYCLES = "loop_cycles_total"
    CYCLE_DURATION = "loop_cycle_duration_ms"
    KILL_SWITCH_ACTIVE = "kill_switch_active"
    SLIPPAGE_BPS = "slippage_bps"


registry = MetricsRegistry()
