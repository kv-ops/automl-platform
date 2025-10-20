"""
Centralized database connections for AutoML Platform.
Manages connections to: automl_app (public + audit schemas) and automl (MLflow).
"""
from functools import lru_cache
from typing import Optional
from sqlalchemy import create_engine, Engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import NullPool
import os

from automl_platform.config import DatabaseConfig

# Singleton instances
_app_engine: Optional[Engine] = None
_audit_engine: Optional[Engine] = None
_rgpd_engine: Optional[Engine] = None
_mlflow_engine: Optional[Engine] = None

_AppSessionLocal: Optional[sessionmaker] = None
_AuditSessionLocal: Optional[sessionmaker] = None
_RGPDSessionLocal: Optional[sessionmaker] = None


def get_app_engine(url: Optional[str] = None) -> Engine:
    """
    Get SQLAlchemy engine for application database (automl_app).
    Used for: Auth, Tenants, API data.
    
    Args:
        url: Optional database URL (for testing). If None, uses DatabaseConfig.
        
    Returns:
        SQLAlchemy Engine instance
    """
    global _app_engine
    
    if url:
        # For tests: create a new engine with the provided URL
        return create_engine(url, poolclass=NullPool)
    
    if _app_engine is None:
        db_config = DatabaseConfig()
        _app_engine = create_engine(
            db_config.url,
            pool_pre_ping=True,
            pool_size=10,
            max_overflow=20,
            echo=os.getenv("SQL_ECHO", "false").lower() == "true"
        )
    
    return _app_engine


def get_audit_engine(url: Optional[str] = None) -> Engine:
    """
    Get SQLAlchemy engine for audit schema (automl_app).
    Used for: Audit logs, compliance tracking.
    
    Args:
        url: Optional database URL (for testing). If None, uses DatabaseConfig.
        
    Returns:
        SQLAlchemy Engine instance
    """
    global _audit_engine
    
    if url:
        return create_engine(url, poolclass=NullPool)
    
    if _audit_engine is None:
        db_config = DatabaseConfig()
        _audit_engine = create_engine(
            db_config.audit_url,
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=10,
            echo=os.getenv("SQL_ECHO", "false").lower() == "true"
        )
    
    return _audit_engine


def get_rgpd_engine(url: Optional[str] = None) -> Engine:
    """
    Get SQLAlchemy engine for RGPD schema (automl_app).
    Note: RGPD shares the audit schema inside automl_app.
    
    Args:
        url: Optional database URL (for testing). If None, uses DatabaseConfig.
        
    Returns:
        SQLAlchemy Engine instance
    """
    global _rgpd_engine
    
    if url:
        return create_engine(url, poolclass=NullPool)
    
    if _rgpd_engine is None:
        db_config = DatabaseConfig()
        _rgpd_engine = create_engine(
            db_config.rgpd_url,
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=10,
            echo=os.getenv("SQL_ECHO", "false").lower() == "true"
        )
    
    return _rgpd_engine


def get_mlflow_engine(url: Optional[str] = None) -> Engine:
    """
    Get SQLAlchemy engine for MLflow database (automl).
    Note: MLflow manages its own schema, this is mainly for migrations.
    
    Args:
        url: Optional database URL (for testing). If None, uses env var.
        
    Returns:
        SQLAlchemy Engine instance
    """
    global _mlflow_engine
    
    if url:
        return create_engine(url, poolclass=NullPool)
    
    if _mlflow_engine is None:
        mlflow_url = os.getenv(
            "MLFLOW_DATABASE_URL",
            os.getenv("DATABASE_URL", "").replace("automl_app", "automl")
        )
        _mlflow_engine = create_engine(
            mlflow_url,
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=10,
            echo=os.getenv("SQL_ECHO", "false").lower() == "true"
        )
    
    return _mlflow_engine


def get_app_sessionmaker(url: Optional[str] = None) -> sessionmaker:
    """
    Get sessionmaker for application database.
    
    Args:
        url: Optional database URL (for testing).
        
    Returns:
        SQLAlchemy sessionmaker
    """
    global _AppSessionLocal
    
    if url:
        engine = get_app_engine(url)
        return sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    if _AppSessionLocal is None:
        engine = get_app_engine()
        _AppSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    return _AppSessionLocal


def get_audit_sessionmaker(url: Optional[str] = None) -> sessionmaker:
    """
    Get sessionmaker for audit database.
    
    Args:
        url: Optional database URL (for testing).
        
    Returns:
        SQLAlchemy sessionmaker
    """
    global _AuditSessionLocal
    
    if url:
        engine = get_audit_engine(url)
        return sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    if _AuditSessionLocal is None:
        engine = get_audit_engine()
        _AuditSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    return _AuditSessionLocal


def get_rgpd_sessionmaker(url: Optional[str] = None) -> sessionmaker:
    """
    Get sessionmaker for RGPD database.
    
    Args:
        url: Optional database URL (for testing).
        
    Returns:
        SQLAlchemy sessionmaker
    """
    global _RGPDSessionLocal
    
    if url:
        engine = get_rgpd_engine(url)
        return sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    if _RGPDSessionLocal is None:
        engine = get_rgpd_engine()
        _RGPDSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    return _RGPDSessionLocal


def get_app_session(url: Optional[str] = None) -> Session:
    """
    Get a new session for application database.
    Use in context manager: with get_app_session() as session: ...
    
    Args:
        url: Optional database URL (for testing).
        
    Returns:
        SQLAlchemy Session instance
    """
    SessionLocal = get_app_sessionmaker(url)
    return SessionLocal()


def get_audit_session(url: Optional[str] = None) -> Session:
    """
    Get a new session for audit database.
    
    Args:
        url: Optional database URL (for testing).
        
    Returns:
        SQLAlchemy Session instance
    """
    SessionLocal = get_audit_sessionmaker(url)
    return SessionLocal()


def get_rgpd_session(url: Optional[str] = None) -> Session:
    """
    Get a new session for RGPD database.
    
    Args:
        url: Optional database URL (for testing).
        
    Returns:
        SQLAlchemy Session instance
    """
    SessionLocal = get_rgpd_sessionmaker(url)
    return SessionLocal()


def close_all_connections():
    """
    Close all database connections.
    Useful for cleanup in tests or graceful shutdown.
    """
    global _app_engine, _audit_engine, _rgpd_engine, _mlflow_engine
    global _AppSessionLocal, _AuditSessionLocal, _RGPDSessionLocal
    
    if _app_engine:
        _app_engine.dispose()
        _app_engine = None
        _AppSessionLocal = None
    
    if _audit_engine:
        _audit_engine.dispose()
        _audit_engine = None
        _AuditSessionLocal = None
    
    if _rgpd_engine:
        _rgpd_engine.dispose()
        _rgpd_engine = None
        _RGPDSessionLocal = None
    
    if _mlflow_engine:
        _mlflow_engine.dispose()
        _mlflow_engine = None


# Context managers for sessions
class AppDatabaseSession:
    """Context manager for application database sessions."""
    
    def __init__(self, url: Optional[str] = None):
        self.url = url
        self.session: Optional[Session] = None
    
    def __enter__(self) -> Session:
        self.session = get_app_session(self.url)
        return self.session
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            if exc_type:
                self.session.rollback()
            else:
                self.session.commit()
            self.session.close()


class AuditDatabaseSession:
    """Context manager for audit database sessions."""
    
    def __init__(self, url: Optional[str] = None):
        self.url = url
        self.session: Optional[Session] = None
    
    def __enter__(self) -> Session:
        self.session = get_audit_session(self.url)
        return self.session
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            if exc_type:
                self.session.rollback()
            else:
                self.session.commit()
            self.session.close()


class RGPDDatabaseSession:
    """Context manager for RGPD database sessions."""
    
    def __init__(self, url: Optional[str] = None):
        self.url = url
        self.session: Optional[Session] = None
    
    def __enter__(self) -> Session:
        self.session = get_rgpd_session(self.url)
        return self.session
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            if exc_type:
                self.session.rollback()
            else:
                self.session.commit()
            self.session.close()
