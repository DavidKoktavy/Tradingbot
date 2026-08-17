"""
Technical indicators.

Design decisions:

- Every indicator takes a sequence of `Bar` and returns a list aligned to
  the input, with `None` for positions where the indicator is not yet
  defined. Returning a shorter list would force every caller to do its own
  index arithmetic, and index arithmetic is exactly how look-ahead bias
  gets introduced.

- **No look-ahead by construction**: the value at index `i` is computed
  only from bars `0..i`. There is a test that asserts this by recomputing
  each indicator on progressively truncated inputs and requiring the
  historical values to be identical. Any indicator that peeks forward
  fails that test.

- Pure functions over plain floats, no pandas. These run inside the
  backtest loop where a DataFrame allocation per bar dominates runtime,
  and determinism matters more than convenience.
"""

from __future__ import annotations

from collections.abc import Sequence

from data.models import Bar


def closes(bars: Sequence[Bar]) -> list[float]:
    return [b.close for b in bars]


def sma(values: Sequence[float], period: int) -> list[float | None]:
    """Simple moving average. Value at i uses values[i-period+1 .. i]."""
    if period <= 0:
        raise ValueError("period must be positive")
    out: list[float | None] = []
    running = 0.0
    for i, v in enumerate(values):
        running += v
        if i >= period:
            running -= values[i - period]
        out.append(running / period if i >= period - 1 else None)
    return out


def ema(values: Sequence[float], period: int) -> list[float | None]:
    """Exponential moving average, seeded with the SMA of the first
    `period` values so the series is deterministic regardless of how much
    history is supplied."""
    if period <= 0:
        raise ValueError("period must be positive")
    out: list[float | None] = [None] * len(values)
    if len(values) < period:
        return out
    multiplier = 2.0 / (period + 1)
    seed = sum(values[:period]) / period
    out[period - 1] = seed
    prev = seed
    for i in range(period, len(values)):
        prev = (values[i] - prev) * multiplier + prev
        out[i] = prev
    return out


def rsi(values: Sequence[float], period: int = 14) -> list[float | None]:
    """Wilder's RSI. Returns 0-100, None until enough history."""
    if period <= 0:
        raise ValueError("period must be positive")
    out: list[float | None] = [None] * len(values)
    if len(values) <= period:
        return out

    gains, losses = 0.0, 0.0
    for i in range(1, period + 1):
        change = values[i] - values[i - 1]
        gains += max(change, 0.0)
        losses += max(-change, 0.0)
    avg_gain, avg_loss = gains / period, losses / period
    out[period] = _rsi_from(avg_gain, avg_loss)

    for i in range(period + 1, len(values)):
        change = values[i] - values[i - 1]
        avg_gain = (avg_gain * (period - 1) + max(change, 0.0)) / period
        avg_loss = (avg_loss * (period - 1) + max(-change, 0.0)) / period
        out[i] = _rsi_from(avg_gain, avg_loss)
    return out


def _rsi_from(avg_gain: float, avg_loss: float) -> float:
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def true_range(bars: Sequence[Bar]) -> list[float | None]:
    out: list[float | None] = []
    for i, bar in enumerate(bars):
        if i == 0:
            out.append(bar.high - bar.low)
            continue
        prev_close = bars[i - 1].close
        out.append(
            max(
                bar.high - bar.low,
                abs(bar.high - prev_close),
                abs(bar.low - prev_close),
            )
        )
    return out


def atr(bars: Sequence[Bar], period: int = 14) -> list[float | None]:
    """Average true range — used by the position sizer when no explicit
    stop is supplied."""
    tr = [t for t in true_range(bars)]
    out: list[float | None] = [None] * len(bars)
    if len(bars) < period:
        return out
    seed = sum(t for t in tr[:period] if t is not None) / period
    out[period - 1] = seed
    prev = seed
    for i in range(period, len(bars)):
        current = tr[i] or 0.0
        prev = (prev * (period - 1) + current) / period
        out[i] = prev
    return out


def rolling_std(values: Sequence[float], period: int) -> list[float | None]:
    out: list[float | None] = [None] * len(values)
    for i in range(period - 1, len(values)):
        window = values[i - period + 1 : i + 1]
        mean = sum(window) / period
        variance = sum((v - mean) ** 2 for v in window) / period
        out[i] = variance**0.5
    return out


def zscore(values: Sequence[float], period: int) -> list[float | None]:
    """How many standard deviations the current value sits from its own
    rolling mean. The core input to mean-reversion logic."""
    means = sma(values, period)
    stds = rolling_std(values, period)
    out: list[float | None] = []
    for i, value in enumerate(values):
        mean, std = means[i], stds[i]
        if mean is None or std is None or std == 0:
            out.append(None)
        else:
            out.append((value - mean) / std)
    return out


def roc(values: Sequence[float], period: int) -> list[float | None]:
    """Rate of change over `period` bars, as a fraction."""
    out: list[float | None] = [None] * len(values)
    for i in range(period, len(values)):
        prior = values[i - period]
        out[i] = (values[i] - prior) / prior if prior != 0 else None
    return out
