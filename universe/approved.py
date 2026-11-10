"""
The approved universe: the operator-curated list of symbols the bot may
ever trade.

Design decisions:

- **This list is the boundary, the scanner is not.** The IBKR scanner can
  propose anything listed on a US exchange, including illiquid small caps,
  halted names and leveraged products. A scanner hit that is not on this
  list is logged and discarded. Widening what the bot can trade therefore
  requires an explicit, reviewable edit to a file in version control —
  the same principle as strategy registration and promotion.

- **Loaded and validated once, then immutable.** A malformed file fails at
  startup rather than producing a partially-loaded universe mid-session.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

_SYMBOL = re.compile(r"^[A-Z][A-Z0-9.]{0,9}$")


class ApprovedUniverse:
    """Ordered, de-duplicated, immutable set of approved symbols."""

    def __init__(self, symbols: list[str]) -> None:
        cleaned: list[str] = []
        seen: set[str] = set()
        for raw in symbols:
            symbol = str(raw).strip().upper()
            if not _SYMBOL.match(symbol):
                raise ValueError(f"Invalid symbol in approved universe: {raw!r}")
            if symbol not in seen:
                seen.add(symbol)
                cleaned.append(symbol)
        if not cleaned:
            raise ValueError("Approved universe is empty; refusing to auto-select from nothing")
        self._symbols = tuple(cleaned)
        self._set = frozenset(cleaned)

    @classmethod
    def from_file(cls, path: str | Path) -> ApprovedUniverse:
        """Load `{"symbols": ["AAPL", ...]}` (comments allowed under
        any other key) or a plain JSON list."""
        file = Path(path)
        if not file.exists():
            raise FileNotFoundError(f"Approved universe file not found: {file}")
        data = json.loads(file.read_text())
        symbols = data.get("symbols") if isinstance(data, dict) else data
        if not isinstance(symbols, list):
            raise TypeError(f"{file}: expected a list under 'symbols'")
        return cls(symbols)

    @property
    def symbols(self) -> tuple[str, ...]:
        return self._symbols

    def __contains__(self, symbol: object) -> bool:
        return isinstance(symbol, str) and symbol.upper() in self._set

    def __len__(self) -> int:
        return len(self._symbols)

    def filter(self, symbols: list[str]) -> tuple[list[str], list[str]]:
        """Split `symbols` into (approved, rejected), preserving order."""
        approved: list[str] = []
        rejected: list[str] = []
        for s in symbols:
            (approved if s.upper() in self._set else rejected).append(s.upper())
        return approved, rejected
