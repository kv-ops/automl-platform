import logging
from logging.config import fileConfig

from sqlalchemy import engine_from_config
from sqlalchemy import pool
from sqlalchemy.engine import make_url

from alembic import context

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# add your model's MetaData object here
# for 'autogenerate' support
# from myapp import mymodel
# target_metadata = mymodel.Base.metadata

# Import all models for autogenerate support
from automl_platform.auth import Base as AuthBase
from automl_platform.models.tenant import Base as TenantBase

# Alembic supporte nativement une liste de MetaData
# Pas besoin de fusion manuelle avec to_metadata()
target_metadata = [AuthBase.metadata, TenantBase.metadata]

import os

logger = logging.getLogger("alembic.env")


def get_url():
    """Resolve the database URL used by Alembic."""

    env_url = os.getenv("AUTOML_DATABASE_URL") or os.getenv("DATABASE_URL")
    if env_url:
        logger.info("Using database URL from environment variable")
        return env_url

    fallback_url = "postgresql://automl:password@localhost:5432/automl_app"
    logger.warning(
        "AUTOML_DATABASE_URL not set; falling back to %s. "
        "Set AUTOML_DATABASE_URL (or DATABASE_URL) to avoid connection hangs during startup.",
        fallback_url,
    )
    return fallback_url

# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = get_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    from sqlalchemy import create_engine

    database_url = get_url()
    url = make_url(database_url)

    connect_args = {}
    if url.get_backend_name() == "postgresql":
        timeout_value = os.getenv("AUTOML_DB_CONNECT_TIMEOUT") or os.getenv("DB_CONNECT_TIMEOUT")
        if timeout_value:
            try:
                connect_args["connect_timeout"] = int(timeout_value)
            except ValueError:
                logger.warning(
                    "Invalid connect timeout '%s'. Falling back to driver default.",
                    timeout_value,
                )

        if "connect_timeout" not in connect_args:
            connect_args["connect_timeout"] = 10

        logger.info(
            "Creating PostgreSQL database engine with connect_timeout=%s",
            connect_args["connect_timeout"],
        )
    else:
        logger.info(
            "Creating %s database engine without connect_timeout override",
            url.get_backend_name(),
        )

    connectable = create_engine(
        database_url,
        poolclass=pool.NullPool,
        connect_args=connect_args,
        pool_pre_ping=True,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection, target_metadata=target_metadata
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
