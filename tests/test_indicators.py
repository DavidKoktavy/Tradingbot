from datetime import datetime, timedelta, timezone

import pytest

from data.models import Bar
from strategies.indicators import (
    atr,
    ema,
    roc,
    rolling_std,
    rsi,
    sma,
    true_range,
    zscore,
)


def make_bars(closes: list[float]) -> list[Bar]:
    base = datetime(2026, 1, 5, tzinfo=timezone.utc)
    bars = []
    for i, c in enumerate(closes):
        bars.append(
            Bar(
                timestamp=base + timedelta(minutes=i),
                open=c,
                high=c + 1,
                low=c - 1,
                close=c,
                volume=1000,
            )
        )
    return bars


# ---- correctness ---------------------------------------------------------


def test_sma_basic():
    result = sma([1, 2, 3, 4, 5], 3)
    assert result[:2] == [None, None]
    assert result[2] == pytest.approx(2.0)
    assert result[3] == pytest.approx(3.0)
    assert result[4] == pytest.approx(4.0)


def test_sma_length_matches_input():
    values = list(range(50))
    assert len(sma(values, 10)) == len(values)


def test_ema_seeded_with_sma():
    values = [1.0] * 10
    result = ema(values, 5)
    assert result[3] is None
    assert result[4] == pytest.approx(1.0)


def test_ema_responds_faster_than_sma():
    values = [10.0] * 20 + [20.0] * 5
    e = ema(values, 10)[-1]
    s = sma(values, 10)[-1]
    assert e > s  # EMA weights recent values more


def test_rsi_all_gains_is_100():
    values = [float(i) for i in range(1, 30)]
    assert rsi(values, 14)[-1] == pytest.approx(100.0)


def test_rsi_all_losses_is_zero():
    values = [float(i) for i in range(30, 1, -1)]
    assert rsi(values, 14)[-1] == pytest.approx(0.0)


def test_rsi_within_bounds():
    values = [10, 12, 11, 13, 15, 14, 16, 18, 17, 19, 20, 18, 21, 22, 20, 23, 25]
    for v in rsi([float(x) for x in values], 14):
        if v is not None:
            assert 0 <= v <= 100


def test_true_range_uses_previous_close():
    bars = [
        Bar(timestamp=datetime(2026, 1, 5, tzinfo=timezone.utc), open=10, high=12, low=9, close=11),
        Bar(
            timestamp=datetime(2026, 1, 5, 0, 1, tzinfo=timezone.utc),
            open=14,
            high=15,
            low=14,
            close=14,
        ),
    ]
    tr = true_range(bars)
    assert tr[0] == 3  # high - low
    assert tr[1] == 4  # |high - prev_close| = |15-11|


def test_atr_positive():
    bars = make_bars([float(10 + i % 5) for i in range(40)])
    values = [v for v in atr(bars, 14) if v is not None]
    assert values and all(v > 0 for v in values)


def test_zscore_of_constant_series_is_none():
    # Zero standard deviation -> undefined, must not divide by zero.
    assert zscore([5.0] * 30, 20)[-1] is None


def test_zscore_detects_stretch():
    values = [10.0] * 29 + [20.0]
    z = zscore(values, 20)[-1]
    assert z is not None and z > 2


def test_roc_computes_fraction():
    values = [100.0] * 10 + [110.0]
    assert roc(values, 10)[-1] == pytest.approx(0.10)


def test_rolling_std_matches_manual():
    values = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
    assert rolling_std(values, 8)[-1] == pytest.approx(2.0)


def test_zero_period_rejected():
    for fn in (sma, ema, rsi):
        with pytest.raises(ValueError):
            fn([1.0, 2.0, 3.0], 0)


def test_insufficient_history_returns_all_none():
    assert all(v is None for v in ema([1.0, 2.0], 10))
    assert all(v is None for v in sma([1.0, 2.0], 10))


# ---- look-ahead bias -----------------------------------------------------


LOOKAHEAD_SERIES = [
    100.0, 102.0, 101.0, 105.0, 103.0, 108.0, 107.0, 110.0, 109.0, 112.0,
    115.0, 113.0, 118.0, 117.0, 120.0, 119.0, 122.0, 125.0, 123.0, 128.0,
    127.0, 130.0, 129.0, 133.0, 131.0, 136.0, 135.0, 138.0, 140.0, 139.0,
    142.0, 145.0, 143.0, 148.0, 147.0, 150.0, 149.0, 152.0, 155.0, 153.0,
]


@pytest.mark.parametrize(
    "fn,period",
    [(sma, 10), (ema, 10), (rsi, 14), (rolling_std, 10), (zscore, 10), (roc, 10)],
)
def test_no_lookahead_in_series_indicators(fn, period):
    """The value at index i must depend only on values 0..i.

    Verified by recomputing on progressively truncated inputs: if the
    indicator peeked forward, truncating the future would change past
    values.
    """
    full = fn(LOOKAHEAD_SERIES, period)
    for cutoff in range(period + 2, len(LOOKAHEAD_SERIES)):
        truncated = fn(LOOKAHEAD_SERIES[:cutoff], period)
        for i in range(cutoff):
            a, b = full[i], truncated[i]
            if a is None or b is None:
                assert a == b, f"{fn.__name__} index {i} defined-ness changed at cutoff {cutoff}"
            else:
                assert a == pytest.approx(b), (
                    f"{fn.__name__} value at index {i} changed when future bars were "
                    f"removed (cutoff {cutoff}): {a} != {b} — indicates look-ahead bias"
                )


@pytest.mark.parametrize("fn,period", [(atr, 14), (true_range, None)])
def test_no_lookahead_in_bar_indicators(fn, period):
    bars = make_bars(LOOKAHEAD_SERIES)
    full = fn(bars, period) if period else fn(bars)
    for cutoff in range(20, len(bars)):
        truncated = fn(bars[:cutoff], period) if period else fn(bars[:cutoff])
        for i in range(cutoff):
            a, b = full[i], truncated[i]
            if a is None or b is None:
                assert a == b
            else:
                assert a == pytest.approx(b), (
                    f"{fn.__name__} at index {i} changed with future bars removed — look-ahead"
                )
