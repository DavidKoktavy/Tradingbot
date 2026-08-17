"""
Structured logging configuration.

Design decisions:

- **JSON by default in production**, human-readable console output only
  when explicitly configured. Trading logs are read by machines during
  incidents far more often than by humans in real time, and grep-ability
  across months of history matters more than prettiness.

- **Secrets are redacted by a processor, not by discipline.** Relying on
  every call site to remember not to log an API key guarantees that one
  eventually does. The processor scans every event's keys against a
  denylist and redacts before rendering. It also truncates very long
  values, so a runaway AI response cannot flood the log.

- **Timestamps are UTC ISO-8601 with microseconds.** Reconstructing the
  order of events across the broker, the loop, and the database requires
  more resolution than whole seconds, and local time makes cross-region
  correlation miserable.

- Logging is configured once, at startup, and never reconfigured. A
  reconfiguration mid-run would silently change what is captured.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog

# Keys whose values must never reach a log sink.
_SECRET_KEYS = frozenset(
    {
        "api_key",
        "apikey",
        "anthropic_api_key",
        "password",
        "passwd",
        "secret",
        "token",
        "access_token",
        "authorization",
        "auth",
        "credentials",
        "database_url",
        "dsn",
        "account_id",
        "account_number",
        "ibkr_account_id",
        "private_key",
        "session_id",
    }
)

_MAX_VALUE_CHARS = 4000
REDACTED = "***REDACTED***"


def redact_secrets(_logger: Any, _method: str, event_dict: dict) -> dict:
    """Redact sensitive values and truncate oversized ones."""
    for key in list(event_dict):
        lowered = key.lower()
        if any(secret in lowered for secret in _SECRET_KEYS):
            event_dict[key] = REDACTED
            continue
        value = event_dict[key]
        if isinstance(value, str) and len(value) > _MAX_VALUE_CHARS:
            event_dict[key] = value[:_MAX_VALUE_CHARS] + f"...[truncated {len(value)} chars]"
    return event_dict


def add_service_context(_logger: Any, _method: str, event_dict: dict) -> dict:
    event_dict.setdefault("service", "trading_agent")
    return event_dict


def configure_logging(
    *, level: str = "INFO", json_output: bool = True, mode: str = "PAPER"
) -> None:
    """Configure structlog. Call once at startup."""
    shared: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        # NOTE: `structlog.stdlib.add_logger_name` is deliberately absent.
        # It requires a stdlib LogRecord and raises AttributeError against
        # PrintLoggerFactory, which would make every log call after
        # configuration crash. The module name is captured by the caller's
        # own bound logger instead.
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        add_service_context,
        redact_secrets,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    renderer = (
        structlog.processors.JSONRenderer()
        if json_output
        else structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())
    )

    structlog.configure(
        processors=[*shared, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # The mode is bound globally so every single log line records whether
    # this was paper or real money. Reconstructing that later from context
    # is exactly the sort of ambiguity that causes expensive mistakes.
    structlog.contextvars.bind_contextvars(trading_mode=mode)


def bind_cycle(cycle: int) -> None:
    structlog.contextvars.bind_contextvars(cycle=cycle)


def clear_context() -> None:
    structlog.contextvars.clear_contextvars()
