#!/usr/bin/env python3
"""
IBKR connectivity smoke test.

This exercises the one part of the system no unit test can cover: the
translation layer between our normalised models and ib_async, against a
real TWS or IB Gateway session.

**Run this before trusting the paper path.** Every layer above the broker
adapter is tested against fakes; this script is what tells you whether the
adapter itself works against the real API.

It is deliberately read-only by default. Order placement is opt-in behind
`--place-test-order`, refuses to run unless the port is a known paper
port, and cancels the order it places.

Usage:

    python scripts/smoke_test_ibkr.py --symbol AAPL
    python scripts/smoke_test_ibkr.py --symbol AAPL --place-test-order

Prerequisites:
  - TWS or IB Gateway running and logged into a PAPER account
  - API access enabled: Configure > API > Settings > Enable ActiveX and
    Socket Clients
  - The port matching your setup (7497 TWS paper, 4002 Gateway paper)
  - "Read-Only API" unchecked if you intend to test order placement
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import datetime, timezone
from decimal import Decimal

# Ports IBKR documents as paper-trading ports. Order placement is refused
# on anything else, because a mistyped port is the difference between a
# test order and a real one.
PAPER_PORTS = {7497, 4002}

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
WARN = "\033[33mWARN\033[0m"


class Results:
    def __init__(self) -> None:
        self.passed = 0
        self.failed = 0

    def ok(self, name: str, detail: str = "") -> None:
        self.passed += 1
        print(f"  [{PASS}] {name}" + (f" — {detail}" if detail else ""))

    def fail(self, name: str, detail: str) -> None:
        self.failed += 1
        print(f"  [{FAIL}] {name} — {detail}")

    def warn(self, name: str, detail: str) -> None:
        print(f"  [{WARN}] {name} — {detail}")


async def _wait_for_price(client, instrument, timeout_seconds: int):
    """Poll for a usable price for up to timeout_seconds. Shared between
    the initial live attempt and the delayed-data retry so both wait the
    same way and there's exactly one place this polling loop lives."""
    snapshot = None
    for _ in range(timeout_seconds):
        await asyncio.sleep(1)
        snapshot = client.get_snapshot(instrument)
        if snapshot is not None and snapshot.mid is not None:
            return snapshot
    return snapshot


async def run(args: argparse.Namespace) -> int:
    from app.config import IBKRSettings
    from broker.ibkr_client import IBKRClient
    from data.models import Instrument

    results = Results()
    instrument = Instrument(symbol=args.symbol.upper(), exchange=args.exchange)

    print(f"\nIBKR smoke test — {args.host}:{args.port} clientId={args.client_id}")
    print(f"Instrument: {instrument}\n")

    if args.port not in PAPER_PORTS:
        print(
            f"  [{WARN}] Port {args.port} is not a known paper port {sorted(PAPER_PORTS)}.\n"
            "         Order placement will be refused. Read-only checks continue.\n"
        )

    settings = IBKRSettings(host=args.host, port=args.port, client_id=args.client_id)
    from broker.market_data import MarketDataType

    initial_type = (
        MarketDataType.DELAYED if args.market_data_type == "delayed" else MarketDataType.LIVE
    )
    client = IBKRClient(settings, market_data_type=initial_type)

    # 1. Connection
    print("Connection")
    try:
        await client.connect()
        results.ok("connect", f"state={client.state}")
    except Exception as exc:
        results.fail("connect", str(exc))
        print(
            "\nCannot continue without a connection. Check that TWS/Gateway is "
            "running, API access is enabled, and the port is correct.\n"
        )
        return 1

    try:
        # 2. Contract qualification
        print("\nContract qualification")
        try:
            bars = await client.get_historical_bars(
                instrument, duration="2 D", bar_size="1 hour"
            )
            if bars:
                results.ok("historical bars", f"{len(bars)} bars, last close {bars[-1].close}")
                first, last = bars[0].timestamp, bars[-1].timestamp
                if last <= first:
                    results.fail("bar ordering", "bars are not chronologically ordered")
                else:
                    results.ok("bar ordering", f"{first.date()} → {last.date()}")
                if any(b.timestamp.tzinfo is None for b in bars):
                    results.fail("bar timezones", "some bars are timezone-naive")
                else:
                    results.ok("bar timezones", "all timezone-aware")
            else:
                results.fail("historical bars", "no bars returned")
        except Exception as exc:
            results.fail("historical bars", str(exc))

        # 3. Live market data, with an automatic fallback to delayed data.
        #
        # The TWS API defaults to LIVE and does NOT automatically fall
        # back to delayed the way TWS's own quote panel does for a human
        # manually looking something up. An account with no real-time
        # subscription gets nothing at all from the API unless DELAYED is
        # explicitly requested. This block tries live first, and only
        # switches to delayed if live genuinely produced nothing — the
        # distinction is always reported, never silently hidden, because
        # delayed data (15-20 minutes old) is fine for proving
        # connectivity but is NOT something the risk engine's staleness
        # check will ever accept for actual automated trading.
        print("\nLive market data")
        try:
            await client.subscribe(instrument)
            results.ok("subscribe", "request accepted")

            snapshot = await _wait_for_price(client, instrument, args.timeout)
            used_delayed = False

            if (snapshot is None or snapshot.mid is None) and initial_type is MarketDataType.LIVE:
                print(
                    "  No live price after "
                    f"{args.timeout}s (Error 10168 in the logs above, if present, means "
                    "your account has no real-time data subscription). Retrying with "
                    "DELAYED market data \u2014 this is normal and expected for a fresh "
                    "paper account."
                )
                from broker.market_data import MarketDataType

                client.set_market_data_type(MarketDataType.DELAYED)
                await client.unsubscribe(instrument)
                await client.subscribe(instrument)
                snapshot = await _wait_for_price(client, instrument, args.timeout)
                used_delayed = True

            if snapshot is None:
                results.fail(
                    "snapshot",
                    f"no data after retrying with delayed data either \u2014 check that "
                    "TWS is showing a quote for this symbol at all, and that the "
                    "market is open",
                )
            elif snapshot.mid is None:
                results.warn(
                    "snapshot",
                    "received but no usable price (market may be closed, or you may "
                    "lack any data access \u2014 live or delayed \u2014 for this instrument)",
                )
            else:
                label = " (DELAYED DATA)" if used_delayed else " (live)"
                results.ok(
                    "snapshot",
                    f"bid={snapshot.bid} ask={snapshot.ask} last={snapshot.last} "
                    f"mid={snapshot.mid}{label}",
                )
                if used_delayed:
                    results.warn(
                        "data mode",
                        "connectivity is proven, but delayed data is NOT usable for "
                        "actual automated trading \u2014 the risk engine's staleness "
                        "check will correctly refuse to trade on 15-20 minute old "
                        "quotes. Subscribe to real-time data before running `paper` "
                        "for real.",
                    )
                age = snapshot.age_seconds()
                if not used_delayed and age > 30:
                    results.warn("data freshness", f"snapshot is {age:.0f}s old")
                elif not used_delayed:
                    results.ok("data freshness", f"{age:.1f}s old")
                if snapshot.bid and snapshot.ask and snapshot.ask < snapshot.bid:
                    results.fail("quote sanity", "book is crossed (ask < bid)")
                else:
                    results.ok("quote sanity", "book is not crossed")
        except Exception as exc:
            results.fail("market data", str(exc))

        # 4. Account and position reads
        print("\nAccount state")
        try:
            from broker.order_manager import IBKROrderGateway

            gateway = IBKROrderGateway(client._ib)  # noqa: SLF001
            positions = await gateway.positions()
            results.ok("positions", f"{len(positions)} reported by broker")
            orders = await gateway.open_orders()
            results.ok("open orders", f"{len(orders)} reported by broker")
        except Exception as exc:
            results.fail("account reads", str(exc))

        # 5. Order placement (opt-in, paper only)
        print("\nOrder round-trip")
        if not args.place_test_order:
            print("  (skipped — pass --place-test-order to test submission)")
        elif args.port not in PAPER_PORTS:
            results.fail(
                "order placement",
                f"refused: port {args.port} is not a known paper port",
            )
        else:
            await _test_order_roundtrip(client, instrument, args, results)

    finally:
        await client.disconnect()
        print(f"\nDisconnected. state={client.state}")

    print(f"\n{'=' * 60}")
    print(f"Passed: {results.passed}   Failed: {results.failed}")
    if results.failed:
        print("\nThe broker adapter is NOT ready. Fix the failures above before")
        print("running the agent, even in paper mode.")
        return 1
    print("\nBroker adapter looks healthy. Remember: this validates connectivity")
    print("and translation only — it says nothing about strategy performance.")
    return 0


async def _test_order_roundtrip(client, instrument, args, results: Results) -> None:
    """Place a far-from-market limit order, verify state, then cancel it.

    A limit order priced well away from the market is used deliberately:
    it should not fill, so the test does not leave a position behind.
    """
    from app.config import TradingMode
    from app.mode_gate import ModeGate
    from broker.order_manager import IBKROrderGateway, OrderManager
    from execution.execution_models import OrderIntent, OrderSide, OrderType
    from execution.order_store import OrderStore
    from execution.order_validator import OrderValidator
    from risk.decisions import RiskAssessment

    snapshot = client.get_snapshot(instrument)
    if snapshot is None or snapshot.mid is None:
        results.fail("order placement", "no market price available to base a limit on")
        return

    mid = Decimal(str(snapshot.mid))
    # 20% below market: marketable only in a catastrophe, and the risk
    # engine's own sanity band would reject it in normal operation, which
    # is why this bypasses the loop and goes straight to the gateway.
    limit = (mid * Decimal("0.80")).quantize(Decimal("0.01"))

    store = OrderStore()
    validator = OrderValidator(store)
    intent = OrderIntent(
        instrument=instrument,
        side=OrderSide.BUY,
        quantity=Decimal("1"),
        order_type=OrderType.LIMIT,
        limit_price=limit,
        source="smoke_test",
    )
    order = validator.build_order(
        intent,
        RiskAssessment(
            approved=True, approved_quantity=Decimal("1"), requested_quantity=Decimal("1")
        ),
    )

    gateway = IBKROrderGateway(client._ib)  # noqa: SLF001
    manager = OrderManager(gateway, store, ModeGate(TradingMode.PAPER))

    try:
        await manager.submit(order)
        results.ok("submit", f"1 share @ {limit} (20% below market), broker id "
                             f"{order.broker_order_id}")
    except Exception as exc:
        results.fail("submit", str(exc))
        return

    await asyncio.sleep(2)
    try:
        open_orders = await gateway.open_orders()
        if any(str(o.broker_order_id) == str(order.broker_order_id) for o in open_orders):
            results.ok("order visible at broker", "appears in open orders")
        else:
            results.warn(
                "order visible at broker",
                "not in open orders — it may have been rejected; check TWS",
            )
    except Exception as exc:
        results.fail("open orders query", str(exc))

    try:
        await manager.cancel(order)
        await asyncio.sleep(2)
        remaining = await gateway.open_orders()
        if any(str(o.broker_order_id) == str(order.broker_order_id) for o in remaining):
            results.fail(
                "cancel",
                f"order {order.broker_order_id} still open — CANCEL IT MANUALLY IN TWS",
            )
        else:
            results.ok("cancel", "order no longer open at broker")
    except Exception as exc:
        results.fail(
            "cancel",
            f"{exc} — CHECK TWS AND CANCEL ORDER {order.broker_order_id} MANUALLY",
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7497, help="7497 TWS paper, 4002 Gateway paper")
    parser.add_argument("--client-id", type=int, default=99)
    parser.add_argument("--symbol", default="AAPL")
    parser.add_argument("--exchange", default="SMART")
    parser.add_argument("--timeout", type=int, default=15, help="Seconds to wait for a quote")
    parser.add_argument(
        "--market-data-type", default="live", choices=["live", "delayed"],
        help="Start with 'delayed' directly if you already know you have no real-time "
             "subscription \u2014 skips the live attempt and its wait entirely",
    )
    parser.add_argument(
        "--place-test-order",
        action="store_true",
        help="Place and cancel a 1-share limit order 20%% below market (paper ports only)",
    )
    args = parser.parse_args()

    try:
        return asyncio.run(run(args))
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
