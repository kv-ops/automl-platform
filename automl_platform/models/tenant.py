"""
Modèle Tenant unifié combinant auth et infrastructure
Centralise toutes les propriétés des locataires multi-tenant

Ce modèle fusionne les définitions de:
- automl_platform.auth.Tenant (billing, plan, SSO)
- automl_platform.api.infrastructure.TenantModel (ressources, quotas)

Utilisation:
    from automl_platform.models.tenant import Tenant, Base
"""

from sqlalchemy import Column, String, Integer, DateTime, Boolean, JSON
from sqlalchemy.dialects.postgresql import UUID
from datetime import datetime
import uuid

from automl_platform.models.base import Base


class Tenant(Base):
    """
    Modèle Tenant unifié pour multi-tenant.
    Combine les champs de auth.py et infrastructure.py.
    """
    __tablename__ = 'tenants'
    __table_args__ = {'schema': 'public'}
    
    # ========================================================================
    # Clé primaire
    # ========================================================================
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # ========================================================================
    # Identification (de auth.py)
    # ========================================================================
    name = Column(String(255), unique=True, nullable=False, index=True)
    subdomain = Column(String(100), unique=True)
    
    # ========================================================================
    # Isolation infrastructure (de auth.py)
    # ========================================================================
    k8s_namespace = Column(String(100), comment="Kubernetes namespace pour isolation")
    minio_bucket = Column(String(100), comment="Bucket MinIO dédié")
    database_schema = Column(String(100), comment="Schéma PostgreSQL dédié (futur)")
    
    # ========================================================================
    # Plan et billing (de auth.py)
    # ========================================================================
    plan_type = Column(String(50), default='free', index=True)
    max_users = Column(Integer, default=5)
    max_projects = Column(Integer, default=10)
    max_storage_gb = Column(Integer, default=10)
    billing_email = Column(String(255))
    stripe_customer_id = Column(String(255))
    trial_ends_at = Column(DateTime)
    
    # ========================================================================
    # Limites ressources (de infrastructure.py)
    # ========================================================================
    max_cpu_cores = Column(Integer, default=2, comment="Nombre max de CPU cores")
    max_memory_gb = Column(Integer, default=4, comment="Mémoire max en GB")
    max_gpu_hours = Column(Integer, default=0, comment="Heures GPU mensuelles")
    max_concurrent_jobs = Column(Integer, default=1, comment="Jobs simultanés max")
    
    # ========================================================================
    # Usage actuel (de infrastructure.py)
    # ========================================================================
    current_cpu_usage = Column(Integer, default=0, comment="CPU utilisé actuellement")
    current_memory_usage = Column(Integer, default=0, comment="Mémoire utilisée (GB)")
    current_storage_usage = Column(Integer, default=0, comment="Stockage utilisé (GB)")
    gpu_hours_used = Column(Integer, default=0, comment="Heures GPU consommées ce mois")
    
    # ========================================================================
    # Sécurité (de infrastructure.py)
    # ========================================================================
    encryption_key = Column(String(255), comment="Clé de chiffrement Fernet pour données sensibles")
    api_key_hash = Column(String(255), comment="Hash de la clé API principale")
    
    # ========================================================================
    # Status
    # ========================================================================
    is_active = Column(Boolean, default=True, index=True)
    is_suspended = Column(Boolean, default=False, comment="Tenant suspendu (non-paiement, etc.)")
    
    # ========================================================================
    # Timestamps
    # ========================================================================
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # ========================================================================
    # Metadata flexible (JSON)
    # ========================================================================
    metadata_json = Column(
        JSON, 
        default=dict,
        comment="Métadonnées additionnelles flexibles (features, config custom, etc.)"
    )
    
    def __repr__(self):
        return f"<Tenant(id={self.id}, name='{self.name}', plan='{self.plan_type}')>"
    
    def to_dict(self):
        """Sérialisation pour API"""
        return {
            'id': str(self.id),
            'name': self.name,
            'subdomain': self.subdomain,
            'plan_type': self.plan_type,
            'is_active': self.is_active,
            'is_suspended': self.is_suspended,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'limits': {
                'max_users': self.max_users,
                'max_projects': self.max_projects,
                'max_cpu_cores': self.max_cpu_cores,
                'max_memory_gb': self.max_memory_gb,
                'max_storage_gb': self.max_storage_gb,
                'max_gpu_hours': self.max_gpu_hours,
                'max_concurrent_jobs': self.max_concurrent_jobs,
            },
            'usage': {
                'current_cpu_usage': self.current_cpu_usage,
                'current_memory_usage': self.current_memory_usage,
                'current_storage_usage': self.current_storage_usage,
                'gpu_hours_used': self.gpu_hours_used,
            }
        }
