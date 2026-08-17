#!/usr/bin/env python3
"""
Generate a sample price-history file for practising with `backtest`.

**This data is completely made up.** It is a mathematical pattern, not real
market history. It exists so you can learn how to run the commands without
needing a market-data subscription first.

Because the pattern is artificial and repeating, any strategy may look
unusually good or unusually bad on it. Results from this file mean nothing
whatsoever about real trading.

Usage:
    python scripts/make_sample_data.py
    python scripts/make_sample_data.py --output mydata.csv --days 500
"""

from __future__ import annotations

import argparse
import csv
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path


def generate(days: int, start_price: float, seed_shift: float) -> list[dict]:
    rows = []
    price = start_price
    date = datetime(2023, 1, 3, tzinfo=timezone.utc)

    for i in range(days):
        # A slow wave plus a faster wobble: produces trends that turn,
        # which is enough for a moving-average strategy to react to.
        trend = math.sin((i + seed_shift) / 45.0) * 0.9
        wobble = math.sin(i / 3.0) * 0.35
        price = max(5.0, price + trend + wobble)

        high = price * 1.012
        low = price * 0.988
        rows.append(
            {
                "timestamp": date.isoformat(),
                "open": round(price, 2),
                "high": round(high, 2),
                "low": round(low, 2),
                "close": round(price, 2),
                "volume": 200000,
            }
        )
        date += timedelta(days=1)
        # Skip weekends so the dates look like real trading days.
        while date.weekday() >= 5:
            date += timedelta(days=1)

    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="sample_data.csv")
    parser.add_argument("--days", type=int, default=400)
    parser.add_argument("--start-price", type=float, default=100.0)
    parser.add_argument("--shift", type=float, default=0.0)
    args = parser.parse_args()

    rows = generate(args.days, args.start_price, args.shift)
    path = Path(args.output)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["timestamp", "open", "high", "low", "close", "volume"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows of MADE-UP price data to {path}")
    print("This is not real market data. Results from it mean nothing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
