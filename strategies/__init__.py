"""
Strategy package.

Importing this package registers the built-in strategies. Registration
happens via the `@register_strategy` decorator at class definition time,
so the modules must be imported for the registry to be populated.

This is an explicit, auditable list rather than filesystem auto-discovery
(see `registry.py` for why): a syntax error in an experimental strategy
must not take down the trading process at startup, and new strategies
should reach production through a deliberate change here rather than by
appearing on disk.

No strategy in this package is claimed to be profitable.
"""

from strategies.base import (
    Signal,
    SignalDirection,
    Strategy,
    StrategyContext,
    StrategyParams,
)
from strategies.ma_crossover import MACrossoverStrategy
from strategies.mean_reversion import MeanReversionStrategy
from strategies.momentum import MomentumStrategy
from strategies.registry import registry, register_strategy
from strategies.trend_following import TrendFollowingStrategy

__all__ = [
    "MACrossoverStrategy",
    "MeanReversionStrategy",
    "MomentumStrategy",
    "TrendFollowingStrategy",
    "Signal",
    "SignalDirection",
    "Strategy",
    "StrategyContext",
    "StrategyParams",
    "register_strategy",
    "registry",
]
