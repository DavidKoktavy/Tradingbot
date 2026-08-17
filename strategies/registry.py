"""
Strategy registry.

Design decision: registration is explicit (decorator or `register()`
call), not filesystem auto-discovery by import side effect. Auto-importing
every module in a package means a syntax error or a heavy import in an
unrelated experimental strategy can take down the trading process at
startup. Explicit registration keeps the live set of strategies auditable
and matches the spec's requirement that new strategies go through a
controlled promotion process rather than appearing by virtue of existing
on disk.
"""

from __future__ import annotations

from collections.abc import Callable

import structlog

from strategies.base import Strategy, StrategyParams

log = structlog.get_logger(__name__)


class StrategyRegistry:
    def __init__(self) -> None:
        self._strategies: dict[str, type[Strategy]] = {}

    def register(self, cls: type[Strategy]) -> type[Strategy]:
        name = cls.name
        if not name or name == "unnamed":
            raise ValueError(f"{cls.__name__} must define a unique `name`")
        if name in self._strategies and self._strategies[name] is not cls:
            raise ValueError(f"Strategy name collision: {name!r} already registered")
        self._strategies[name] = cls
        # Deliberately not logged: registration happens at import time,
        # before logging is configured, so emitting here would print
        # unformatted noise ahead of every CLI command's output.
        return cls

    def get(self, name: str) -> type[Strategy]:
        if name not in self._strategies:
            raise KeyError(
                f"Unknown strategy {name!r}. Registered: {sorted(self._strategies)}"
            )
        return self._strategies[name]

    def create(self, name: str, params: StrategyParams | None = None) -> Strategy:
        return self.get(name)(params)

    def names(self) -> list[str]:
        return sorted(self._strategies)

    def describe(self) -> list[dict[str, str]]:
        return [
            {"name": n, "version": c.version, "class": c.__name__}
            for n, c in sorted(self._strategies.items())
        ]

    def __contains__(self, name: object) -> bool:
        return name in self._strategies

    def __len__(self) -> int:
        return len(self._strategies)


# Module-level default registry used by the application wiring.
registry = StrategyRegistry()


def register_strategy(cls: type[Strategy]) -> type[Strategy]:
    """Decorator form: @register_strategy above a Strategy subclass."""
    return registry.register(cls)
