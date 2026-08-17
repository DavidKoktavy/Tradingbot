"""
Alembic environment.

Design decision: the database URL comes from application settings at
runtime, never from alembic.ini. Committing a connection string with
credentials to source control is the mistake this avoids, and it keeps
one source of truth for configuration.
"""
from __future__ import annotations

import os
import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import engine_from_config, pool

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from database.models import Base  # noqa: E402

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Runtime URL: env var wins, then application settings, then the ini
# placeholder. Never the other way round.
url = os.environ.get("DATABASE_URL")
if not url:
    try:
        from app.config import get_settings

        url = get_settings().database_url.get_secret_value()
    except Exception:
        url = None
if url:
    # Alembic runs migrations synchronously, but the application uses an
    # async driver. Translate the async driver name to its sync
    # equivalent rather than requiring two configured URLs, which would
    # inevitably drift apart.
    for async_driver, sync_driver in (
        ("postgresql+asyncpg", "postgresql+psycopg"),
        ("sqlite+aiosqlite", "sqlite"),
        ("mysql+aiomysql", "mysql+pymysql"),
    ):
        if url.startswith(async_driver):
            url = url.replace(async_driver, sync_driver, 1)
            break
    config.set_main_option("sqlalchemy.url", url)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            # SQLite cannot ALTER most things in place; batch mode
            # recreates the table. Harmless on PostgreSQL.
            render_as_batch=connection.dialect.name == "sqlite",
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
