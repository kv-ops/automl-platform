"""Shared SQLAlchemy base classes for application and audit models."""

from typing import Optional

from sqlalchemy import MetaData
from sqlalchemy.engine import make_url
from sqlalchemy.orm import declarative_base

from automl_platform.config import DatabaseConfig


def database_supports_schemas(url: Optional[str]) -> bool:
    """Return True when the database backend supports named schemas."""

    if not url:
        return False

    try:
        backend_name = make_url(url).get_backend_name()
    except Exception:  # pragma: no cover - defensive guard
        return False

    return backend_name == "postgresql"


_db_config = DatabaseConfig()
supports_schemas = database_supports_schemas(getattr(_db_config, "url", None))
supports_audit_schemas = database_supports_schemas(
    getattr(_db_config, "audit_url", None) or getattr(_db_config, "url", None)
)


def _schema_table_args(schema: str, enabled: bool) -> dict:
    return {"schema": schema} if enabled else {}


def public_table_args() -> dict:
    """Return table args for the public schema when supported."""

    return _schema_table_args("public", supports_schemas)


def audit_table_args(enabled: Optional[bool] = None) -> dict:
    """Return table args for the audit schema when supported."""

    active = supports_audit_schemas if enabled is None else enabled
    return _schema_table_args("audit", active)


def qualify(table_name: str, schema: str = "public", supports: Optional[bool] = None) -> str:
    """Qualify a table or column reference with its schema when supported."""

    if supports is None:
        supports = supports_schemas if schema == "public" else supports_audit_schemas
    return f"{schema}.{table_name}" if supports else table_name


# Metadata for application tables living in the public schema when supported
metadata = MetaData(schema=public_table_args().get("schema"))
Base = declarative_base(metadata=metadata)

# Metadata for audit tables living in the audit schema when supported
audit_metadata = MetaData(schema=audit_table_args().get("schema"))
AuditBase = declarative_base(metadata=audit_metadata)

__all__ = [
    "metadata",
    "Base",
    "audit_metadata",
    "AuditBase",
    "database_supports_schemas",
    "supports_schemas",
    "supports_audit_schemas",
    "public_table_args",
    "audit_table_args",
    "qualify",
]
