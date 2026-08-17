"""
Sector and correlation-aware exposure limits.

Closes the Phase 4 limitation. Gross exposure alone treats ten positions in
the same sector as diversified when they are effectively one position with
ten tickets — the exposure that actually matters is the correlated one.

Design decisions:

- **Unknown metadata is treated as the most restrictive case, not the
  least.** An instrument with no sector assignment is placed in an
  `UNKNOWN` bucket that counts against a limit, rather than being exempt.
  Exempting unknowns means a data gap silently removes a risk control.

- **Missing correlation data assumes high correlation, not zero.** Two
  instruments with no observed relationship are conservatively treated as
  correlated at `default_correlation` (0.5). Assuming independence is the
  optimistic assumption, and optimistic defaults in a risk engine are
  where losses come from.

- **Correlation is computed from returns the system has actually seen**,
  not supplied by an oracle. It is recomputed on a rolling window and
  degrades to the default when there is insufficient history. A
  correlation estimate from 12 observations is noise, so a minimum sample
  size is enforced.

- **This is an additional constraint, never a relaxation.** It can only
  reject trades the existing limits would have allowed. There is no path
  by which good diversification raises any other limit.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from decimal import Decimal

import structlog
from pydantic import BaseModel, Field

from data.models import Instrument
from portfolio.positions import Position

log = structlog.get_logger(__name__)

UNKNOWN_SECTOR = "UNKNOWN"


class InstrumentMetadata(BaseModel):
    """Static classification. Supplied by configuration, never by the AI —
    an AI that could assign sectors could evade sector limits by
    reclassifying instruments."""

    symbol: str
    sector: str = UNKNOWN_SECTOR
    industry: str = UNKNOWN_SECTOR
    asset_class: str = "EQUITY"
    beta: float | None = None


class MetadataRegistry:
    def __init__(self, entries: Sequence[InstrumentMetadata] | None = None) -> None:
        self._by_symbol: dict[str, InstrumentMetadata] = {
            e.symbol.upper(): e for e in (entries or [])
        }

    def add(self, metadata: InstrumentMetadata) -> None:
        self._by_symbol[metadata.symbol.upper()] = metadata

    def get(self, instrument: Instrument | str) -> InstrumentMetadata:
        symbol = (
            instrument.symbol if isinstance(instrument, Instrument) else str(instrument)
        ).upper()
        existing = self._by_symbol.get(symbol)
        if existing is not None:
            return existing
        # Unknown metadata is a restriction, not an exemption: the
        # instrument joins the UNKNOWN bucket, which has its own limit.
        return InstrumentMetadata(symbol=symbol, sector=UNKNOWN_SECTOR)

    def sector(self, instrument: Instrument | str) -> str:
        return self.get(instrument).sector

    @property
    def known_symbols(self) -> set[str]:
        return set(self._by_symbol)


class CorrelationMatrix:
    """Rolling correlation estimated from observed returns."""

    def __init__(
        self,
        *,
        window: int = 60,
        min_observations: int = 30,
        default_correlation: float = 0.5,
    ) -> None:
        self._window = window
        self._min_obs = min_observations
        # Conservative default: assuming independence is the optimistic
        # assumption, and optimistic defaults in a risk engine are where
        # losses come from.
        self._default = default_correlation
        self._returns: dict[str, list[float]] = {}

    def observe(self, symbol: str, return_pct: float) -> None:
        series = self._returns.setdefault(symbol.upper(), [])
        series.append(return_pct)
        if len(series) > self._window:
            del series[: len(series) - self._window]

    def observe_prices(self, symbol: str, prices: Sequence[float]) -> None:
        """Convenience: derive returns from a price series."""
        for i in range(1, len(prices)):
            prev = prices[i - 1]
            if prev:
                self.observe(symbol, (prices[i] - prev) / prev)

    def correlation(self, a: str, b: str) -> float:
        a, b = a.upper(), b.upper()
        if a == b:
            return 1.0
        xs, ys = self._returns.get(a, []), self._returns.get(b, [])
        n = min(len(xs), len(ys))
        if n < self._min_obs:
            # A correlation estimate from a handful of observations is
            # noise; fall back to the conservative default.
            return self._default
        xs, ys = xs[-n:], ys[-n:]

        mean_x, mean_y = sum(xs) / n, sum(ys) / n
        cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
        var_x = sum((x - mean_x) ** 2 for x in xs)
        var_y = sum((y - mean_y) ** 2 for y in ys)
        if var_x <= 0 or var_y <= 0:
            return self._default
        return max(-1.0, min(1.0, cov / math.sqrt(var_x * var_y)))

    def has_data(self, symbol: str) -> bool:
        return len(self._returns.get(symbol.upper(), [])) >= self._min_obs


@dataclass
class ExposureBreach:
    kind: str
    detail: str
    bucket: str = ""


@dataclass
class ExposureAssessment:
    approved: bool = False
    breaches: list[ExposureBreach] = field(default_factory=list)
    sector_exposure: dict[str, Decimal] = field(default_factory=dict)
    correlated_exposure: Decimal = Decimal("0")

    @property
    def reason(self) -> str | None:
        return self.breaches[0].detail if self.breaches else None


class ExposureManager:
    """Sector and correlation constraints. Rejects only; never relaxes."""

    def __init__(
        self,
        *,
        metadata: MetadataRegistry | None = None,
        correlations: CorrelationMatrix | None = None,
        max_sector_exposure: Decimal = Decimal("0.30"),
        max_unknown_sector_exposure: Decimal = Decimal("0.15"),
        max_correlated_exposure: Decimal = Decimal("0.40"),
        correlation_threshold: float = 0.7,
    ) -> None:
        self._metadata = metadata or MetadataRegistry()
        self._correlations = correlations or CorrelationMatrix()
        self._max_sector = max_sector_exposure
        # Unknown-sector instruments get a *tighter* limit: we cannot
        # reason about their concentration, so we hold less of them.
        self._max_unknown = max_unknown_sector_exposure
        self._max_correlated = max_correlated_exposure
        self._threshold = correlation_threshold

    @property
    def metadata(self) -> MetadataRegistry:
        return self._metadata

    @property
    def correlations(self) -> CorrelationMatrix:
        return self._correlations

    def sector_exposures(
        self, positions: dict[str, Position], prices: dict[str, Decimal]
    ) -> dict[str, Decimal]:
        out: dict[str, Decimal] = {}
        for key, position in positions.items():
            if position.is_flat:
                continue
            price = prices.get(key)
            if price is None:
                continue
            sector = self._metadata.sector(key.split(":")[0])
            out[sector] = out.get(sector, Decimal("0")) + position.exposure(price)
        return out

    def evaluate(
        self,
        *,
        instrument: Instrument,
        additional_quantity: Decimal,
        price: Decimal,
        positions: dict[str, Position],
        prices: dict[str, Decimal],
        equity: Decimal,
    ) -> ExposureAssessment:
        assessment = ExposureAssessment()
        if equity <= 0:
            assessment.breaches.append(
                ExposureBreach(kind="NO_EQUITY", detail="Account equity is zero or unset")
            )
            return assessment

        added = abs(additional_quantity) * price
        key = str(instrument)

        # --- sector concentration ---
        exposures = self.sector_exposures(positions, prices)
        sector = self._metadata.sector(instrument)
        projected = exposures.get(sector, Decimal("0")) + added
        exposures[sector] = projected
        assessment.sector_exposure = exposures

        limit = self._max_unknown if sector == UNKNOWN_SECTOR else self._max_sector
        projected_pct = projected / equity
        if projected_pct > limit:
            label = (
                "unclassified instruments" if sector == UNKNOWN_SECTOR else f"sector {sector}"
            )
            assessment.breaches.append(
                ExposureBreach(
                    kind="MAX_SECTOR_EXPOSURE",
                    bucket=sector,
                    detail=(
                        f"Projected exposure to {label} is {projected_pct:.1%} of equity, "
                        f"limit {limit:.1%}"
                    ),
                )
            )
            return assessment

        # --- correlated cluster ---
        symbol = instrument.symbol.upper()
        cluster = added
        members = [symbol]
        for other_key, position in positions.items():
            if position.is_flat or other_key == key:
                continue
            other_price = prices.get(other_key)
            if other_price is None:
                continue
            other_symbol = other_key.split(":")[0].upper()
            rho = self._correlations.correlation(symbol, other_symbol)
            if abs(rho) >= self._threshold:
                cluster += position.exposure(other_price)
                members.append(other_symbol)

        assessment.correlated_exposure = cluster
        cluster_pct = cluster / equity
        if cluster_pct > self._max_correlated and len(members) > 1:
            assessment.breaches.append(
                ExposureBreach(
                    kind="MAX_CORRELATED_EXPOSURE",
                    bucket=",".join(sorted(members)),
                    detail=(
                        f"Projected exposure to correlated cluster {sorted(members)} is "
                        f"{cluster_pct:.1%} of equity, limit {self._max_correlated:.1%}"
                    ),
                )
            )
            return assessment

        assessment.approved = True
        return assessment
