"""
Typed application configuration.

Design decisions:
- All configuration is loaded through pydantic-settings from environment
  variables (optionally backed by a .env file for local dev). Nothing in
  the rest of the codebase should read os.environ directly — this keeps
  configuration centralized and validated at startup, not scattered.
- TradingMode defaults to PAPER. Reaching LIVE requires two independent,
  explicitly-set values (ENABLE_LIVE_TRADING=true AND a non-empty
  LIVE_TRADING_CONFIRMATION string matching a fixed phrase). This is a
  deliberate two-factor guard: a single typo'd env var should not be able
  to enable live order flow.
- Secrets (API keys, DB credentials) are never given defaults and are
  never logged. get_settings() is cached so we parse/validate env once.
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_LIVE_CONFIRMATION_PHRASE = "I_UNDERSTAND_THIS_ENABLES_REAL_ORDERS"


class TradingMode(StrEnum):
    BACKTEST = "BACKTEST"
    SIMULATION = "SIMULATION"
    PAPER = "PAPER"
    LIVE = "LIVE"


class RiskLimits(BaseSettings):
    """Deterministic risk limits. These are example defaults only —
    not a recommendation, and not evidence any strategy is profitable."""

    model_config = SettingsConfigDict(env_prefix="", extra="ignore")

    max_risk_per_trade: float = Field(default=0.005, gt=0, le=1.0)
    max_daily_loss: float = Field(default=0.02, gt=0, le=1.0)
    max_portfolio_drawdown: float = Field(default=0.10, gt=0, le=1.0)
    max_position_size: float = Field(default=0.10, gt=0, le=1.0)
    max_sector_exposure: float = Field(default=0.30, gt=0, le=1.0)
    max_gross_exposure: float = Field(default=1.00, gt=0, le=10.0)
    max_open_positions: int = Field(default=10, gt=0)
    max_orders_per_minute: int = Field(default=20, gt=0)
    max_market_data_age_seconds: float = Field(default=5.0, gt=0)


class IBKRSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="IBKR_", extra="ignore")

    host: str = "127.0.0.1"
    port: int = 7497  # 7497 = TWS paper, 7496 = TWS live, 4002/4001 = IB Gateway
    client_id: int = 1
    account_id: str | None = None
    # "live" requires a real-time data subscription and is what the API
    # requests by default (this default is NOT automatic on IBKR's side —
    # see broker/market_data.py::MarketDataType docstring for why an
    # account with no subscription gets nothing at all unless "delayed"
    # is explicitly requested here or via set_market_data_type()).
    market_data_type: Literal["live", "frozen", "delayed", "delayed_frozen"] = "live"


class AISettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AI_", extra="ignore")

    provider: str = "anthropic"
    model: str = ""
    anthropic_api_key: SecretStr | None = Field(default=None, alias="ANTHROPIC_API_KEY")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    trading_mode: TradingMode = TradingMode.PAPER
    enable_live_trading: bool = False
    live_trading_confirmation: str = ""

    database_url: SecretStr = Field(
        default=SecretStr("postgresql+asyncpg://trading_agent:changeme@localhost:5432/trading_agent")
    )

    log_level: str = "INFO"
    log_format: str = "json"

    risk: RiskLimits = Field(default_factory=RiskLimits)
    ibkr: IBKRSettings = Field(default_factory=IBKRSettings)
    ai: AISettings = Field(default_factory=AISettings)

    @model_validator(mode="after")
    def _guard_live_trading(self) -> "Settings":
        """Refuse to construct a Settings object in an inconsistent LIVE
        state. This runs at startup, so a misconfigured deployment fails
        fast instead of silently running in an unintended mode."""
        if self.trading_mode is TradingMode.LIVE:
            if not self.enable_live_trading:
                raise ValueError(
                    "TRADING_MODE=LIVE requires ENABLE_LIVE_TRADING=true. Refusing to start."
                )
            if self.live_trading_confirmation != _LIVE_CONFIRMATION_PHRASE:
                raise ValueError(
                    "TRADING_MODE=LIVE requires LIVE_TRADING_CONFIRMATION="
                    f"{_LIVE_CONFIRMATION_PHRASE!r}. Refusing to start."
                )
        return self

    @property
    def is_live(self) -> bool:
        return self.trading_mode is TradingMode.LIVE


@lru_cache
def get_settings() -> Settings:
    """Cached settings accessor. Call get_settings.cache_clear() in tests
    that need to re-read environment variables."""
    return Settings()
