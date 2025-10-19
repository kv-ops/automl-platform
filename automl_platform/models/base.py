"""Shared SQLAlchemy base classes for application and audit models."""

from sqlalchemy import MetaData
from sqlalchemy.orm import declarative_base

# Metadata for application tables living in the public schema
metadata = MetaData(schema="public")
Base = declarative_base(metadata=metadata)

# Metadata for audit tables living in the audit schema
audit_metadata = MetaData(schema="audit")
AuditBase = declarative_base(metadata=audit_metadata)

__all__ = ["metadata", "Base", "audit_metadata", "AuditBase"]
