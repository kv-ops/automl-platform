"""
Module de gestion des versions de modèles pour AutoML Platform
Gère le versioning sémantique, les rollbacks, et l'historique des modèles
Place dans: automl_platform/api/model_versioning.py
"""

import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import hashlib
import pickle
from enum import Enum
from dataclasses import dataclass, asdict, field
import pandas as pd
import numpy as np
from packaging import version
import sqlite3
import time
import asyncio
from scipy import stats

# Imports depuis la structure du projet (niveau parent)
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from storage import StorageManager, ModelMetadata
from config import AutoMLConfig
from monitoring import ModelMonitor, MonitoringService, DriftDetector
from metrics import calculate_metrics, compare_models_metrics
from infrastructure import TenantManager, SecurityManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VersionStatus(str, Enum):
    """Statuts possibles pour une version de modèle"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"
    DEPRECATED = "deprecated"
    FAILED = "failed"
    AB_TESTING = "ab_testing"


class PromotionStrategy(str, Enum):
    """Stratégies de promotion de versions"""
    MANUAL = "manual"
    AUTO_THRESHOLD = "auto_threshold"
    AUTO_IMPROVEMENT = "auto_improvement"
    BLUE_GREEN = "blue_green"
    CANARY = "canary"
    SHADOW = "shadow"
    AB_TESTING = "ab_testing"


@dataclass
class ModelVersion:
    """Version de modèle avec intégration complète"""
    model_id: str
    version: str
    tenant_id: str = "default"
    
    # Métadonnées
    created_at: datetime = field(default_factory=datetime.now)
    created_by: Optional[str] = None
    description: str = ""
    tags: Dict[str, str] = field(default_factory=dict)
    
    # Statut et environnement
    status: VersionStatus = VersionStatus.DEVELOPMENT
    environment: Optional[str] = None
    
    # Métriques de performance
    metrics: Dict[str, float] = field(default_factory=dict)
    validation_metrics: Dict[str, float] = field(default_factory=dict)
    production_metrics: Optional[Dict[str, float]] = None
    
    # Configuration du modèle
    algorithm: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    feature_names: List[str] = field(default_factory=list)
    preprocessing_config: Optional[Dict[str, Any]] = None
    model_type: str = "classification"  # classification/regression
    
    # Artefacts et chemins
    model_path: Optional[str] = None
    preprocessor_path: Optional[str] = None
    artifacts: Dict[str, str] = field(default_factory=dict)
    
    # Traçabilité
    parent_version: Optional[str] = None
    commit_hash: Optional[str] = None
    dataset_hash: Optional[str] = None
    pipeline_hash: Optional[str] = None
    
    # Déploiement
    deployment_config: Optional[Dict[str, Any]] = None
    endpoints: List[str] = field(default_factory=list)
    container_id: Optional[str] = None
    
    # Monitoring et drift
    drift_detected: bool = False
    last_monitoring_check: Optional[datetime] = None
    monitoring_alerts: List[Dict[str, Any]] = field(default_factory=list)
    drift_score: float = 0.0
    
    # A/B Testing
    ab_test_id: Optional[str] = None
    ab_test_group: Optional[str] = None
    ab_test_metrics: Optional[Dict[str, float]] = None
    
    # Billing et utilisation
    compute_hours_used: float = 0.0
    predictions_count: int = 0
    storage_mb_used: float = 0.0
    estimated_cost: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire"""
        data = asdict(self)
        for key in ['created_at', 'last_monitoring_check']:
            if key in data and data[key]:
                data[key] = data[key].isoformat() if isinstance(data[key], datetime) else data[key]
        if 'status' in data and isinstance(data['status'], VersionStatus):
            data['status'] = data['status'].value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelVersion':
        """Crée depuis un dictionnaire"""
        for key in ['created_at', 'last_monitoring_check']:
            if key in data and data[key] and isinstance(data[key], str):
                data[key] = datetime.fromisoformat(data[key])
        if 'status' in data and isinstance(data['status'], str):
            data['status'] = VersionStatus(data['status'])
        return cls(**data)
    
    def is_newer_than(self, other: 'ModelVersion') -> bool:
        """Compare les versions"""
        return version.parse(self.version) > version.parse(other.version)


@dataclass
class VersionComparison:
    """Comparaison détaillée entre versions"""
    version_a: ModelVersion
    version_b: ModelVersion
    
    # Comparaison des métriques
    metrics_diff: Dict[str, float] = field(default_factory=dict)
    improvement_rate: float = 0.0
    
    # Tests statistiques
    statistical_tests: Dict[str, Any] = field(default_factory=dict)
    is_significantly_better: bool = False
    confidence_level: float = 0.95
    
    # Analyse détaillée
    feature_importance_diff: Optional[Dict[str, float]] = None
    prediction_diff_analysis: Optional[Dict[str, Any]] = None
    
    # Recommandation
    recommendation: str = ""
    risk_assessment: Dict[str, Any] = field(default_factory=dict)
    
    # Coût comparatif
    cost_diff: float = 0.0
    efficiency_ratio: float = 1.0


class ModelVersionManager:
    """Gestionnaire principal des versions de modèles avec intégration complète"""
    
    def __init__(self, config: AutoMLConfig = None):
        self.config = config or AutoMLConfig()
        
        # Initialisation des composants existants
        self.storage = StorageManager(
            backend=self.config.storage.backend,
            encryption_key=self._get_encryption_key() if self.config.storage.backend != "local" else None
        )
        self.monitoring_service = MonitoringService(
            self.storage,
            billing_tracker=None  # Sera initialisé si billing activé
        )
        self.tenant_manager = TenantManager()
        self.security_manager = SecurityManager()
        
        # Base de données de versioning
        self.db_path = Path(self.config.storage.local_base_path) / "versioning.db"
        self._init_database()
        
        # Cache des versions et métriques
        self.version_cache: Dict[str, ModelVersion] = {}
        self.metrics_cache: Dict[str, Dict] = {}
        
        # Stratégies de promotion
        self.promotion_strategies: Dict[str, callable] = {
            PromotionStrategy.MANUAL: self._manual_promotion,
            PromotionStrategy.AUTO_THRESHOLD: self._auto_threshold_promotion,
            PromotionStrategy.AUTO_IMPROVEMENT: self._auto_improvement_promotion,
            PromotionStrategy.BLUE_GREEN: self._blue_green_deployment,
            PromotionStrategy.CANARY: self._canary_deployment,
            PromotionStrategy.SHADOW: self._shadow_deployment,
            PromotionStrategy.AB_TESTING: self._ab_testing_deployment
        }
        
        # Monitors actifs par modèle
        self.active_monitors: Dict[str, ModelMonitor] = {}
    
    def _get_encryption_key(self) -> Optional[bytes]:
        """Récupère la clé d'encryption pour le tenant"""
        # En production, récupérer depuis le gestionnaire de secrets
        import os
        key = os.environ.get('ENCRYPTION_KEY')
        if key:
            return key.encode()
        return None
    
    def _init_database(self):
        """Initialise la base de données de versioning étendue"""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Table des versions étendue
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_versions (
                model_id TEXT,
                version TEXT,
                tenant_id TEXT,
                created_at TIMESTAMP,
                created_by TEXT,
                status TEXT,
                metrics JSON,
                metadata JSON,
                model_type TEXT,
                deployment_config JSON,
                ab_test_id TEXT,
                compute_hours_used REAL DEFAULT 0,
                predictions_count INTEGER DEFAULT 0,
                storage_mb_used REAL DEFAULT 0,
                estimated_cost REAL DEFAULT 0,
                PRIMARY KEY (model_id, version, tenant_id)
            )
        """)
        
        # Table de l'historique des promotions
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS promotion_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id TEXT,
                version TEXT,
                tenant_id TEXT,
                from_status TEXT,
                to_status TEXT,
                promoted_at TIMESTAMP,
                promoted_by TEXT,
                reason TEXT,
                strategy TEXT,
                metrics_before JSON,
                metrics_after JSON,
                success BOOLEAN
            )
        """)
        
        # Table des comparaisons
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS version_comparisons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id TEXT,
                version_a TEXT,
                version_b TEXT,
                tenant_id TEXT,
                compared_at TIMESTAMP,
                comparison_result JSON,
                statistical_significance REAL,
                recommendation TEXT
            )
        """)
        
        # Table des rollbacks
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS rollback_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id TEXT,
                from_version TEXT,
                to_version TEXT,
                tenant_id TEXT,
                rolled_back_at TIMESTAMP,
                rolled_back_by TEXT,
                reason TEXT,
                metrics_before JSON,
                metrics_after JSON
            )
        """)
        
        # Table des tests A/B
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ab_tests (
                test_id TEXT PRIMARY KEY,
                model_id TEXT,
                version_a TEXT,
                version_b TEXT,
                tenant_id TEXT,
                started_at TIMESTAMP,
                ended_at TIMESTAMP,
                traffic_split REAL,
                winner TEXT,
                results JSON
            )
        """)
        
        conn.commit()
        conn.close()
    
    def create_version(self, 
                      model_id: str,
                      model_object: Any,
                      version_number: Optional[str] = None,
                      tenant_id: str = "default",
                      metadata: Optional[Dict[str, Any]] = None) -> ModelVersion:
        """Crée une nouvelle version avec vérification des quotas"""
        
        # Vérification des quotas tenant
        tenant_config = self.tenant_manager.get_tenant(tenant_id)
        if tenant_config:
            if not self._check_quota(tenant_id, 'models'):
                raise ValueError(f"Quota de modèles dépassé pour le tenant {tenant_id}")
        
        # Génération automatique du numéro de version
        if not version_number:
            version_number = self._generate_next_version(model_id, tenant_id)
        
        # Validation du format de version
        try:
            version.parse(version_number)
        except:
            raise ValueError(f"Format de version invalide: {version_number}")
        
        # Création de la version
        model_version = ModelVersion(
            model_id=model_id,
            version=version_number,
            tenant_id=tenant_id,
            created_at=datetime.now(),
            status=VersionStatus.DEVELOPMENT,
            model_type=metadata.get('model_type', 'classification') if metadata else 'classification',
            **(metadata if metadata else {})
        )
        
        # Calcul des hashs pour la traçabilité
        if model_object:
            model_version.pipeline_hash = self._calculate_hash(model_object)
        
        # Calcul de la taille de stockage
        import sys
        model_size_mb = sys.getsizeof(pickle.dumps(model_object)) / (1024 * 1024)
        model_version.storage_mb_used = model_size_mb
        
        # Sauvegarde du modèle via StorageManager
        try:
            # Utilisation du StorageManager existant
            storage_metadata = {
                'model_id': model_id,
                'model_type': model_version.model_type,
                'algorithm': model_version.algorithm,
                'metrics': model_version.metrics,
                'parameters': model_version.parameters,
                'feature_names': model_version.feature_names,
                'target_name': 'target',
                'dataset_hash': model_version.dataset_hash or '',
                'pipeline_hash': model_version.pipeline_hash or '',
                'tags': list(model_version.tags.keys()),
                'description': model_version.description,
                'author': model_version.created_by or 'system',
                'tenant_id': tenant_id
            }
            
            model_path = self.storage.save_model(
                model_object, 
                storage_metadata, 
                version=version_number,
                encryption=self.config.storage.backend != "local"
            )
            model_version.model_path = model_path
            
            # Enregistrement du monitor pour ce modèle
            monitor = self.monitoring_service.register_model(
                model_id=f"{model_id}_v{version_number}",
                model_type=model_version.model_type,
                tenant_id=tenant_id
            )
            self.active_monitors[f"{model_id}:{version_number}"] = monitor
            
            # Sauvegarde en base de données
            self._save_version_to_db(model_version)
            
            # Mise en cache
            cache_key = f"{tenant_id}:{model_id}:{version_number}"
            self.version_cache[cache_key] = model_version
            
            # Mise à jour de l'utilisation du tenant
            if tenant_config:
                self.tenant_manager.allocate_resources(
                    tenant_id,
                    storage=int(model_size_mb)
                )
            
            logger.info(f"Version créée: {model_id} v{version_number} pour tenant {tenant_id}")
            return model_version
            
        except Exception as e:
            logger.error(f"Erreur lors de la création de la version: {e}")
            raise
    
    def get_version(self, 
                   model_id: str, 
                   version_number: str = "latest",
                   tenant_id: str = "default") -> Optional[ModelVersion]:
        """Récupère une version avec vérification des permissions"""
        
        # Vérification des permissions tenant
        if not self._check_tenant_access(tenant_id, model_id):
            logger.warning(f"Accès refusé au modèle {model_id} pour tenant {tenant_id}")
            return None
        
        # Si "latest", trouver la dernière version
        if version_number == "latest":
            version_number = self._get_latest_version(model_id, tenant_id)
            if not version_number:
                return None
        
        # Vérification du cache
        cache_key = f"{tenant_id}:{model_id}:{version_number}"
        if cache_key in self.version_cache:
            return self.version_cache[cache_key]
        
        # Chargement depuis la base de données
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT metadata FROM model_versions
            WHERE model_id = ? AND version = ? AND tenant_id = ?
        """, (model_id, version_number, tenant_id))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            metadata = json.loads(result[0])
            model_version = ModelVersion.from_dict(metadata)
            self.version_cache[cache_key] = model_version
            return model_version
        
        return None
    
    def list_versions(self, 
                     model_id: str,
                     tenant_id: str = "default",
                     status_filter: Optional[VersionStatus] = None,
                     include_metrics: bool = True) -> List[ModelVersion]:
        """Liste les versions avec métriques optionnelles"""
        
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        query = """
            SELECT metadata, metrics FROM model_versions
            WHERE model_id = ? AND tenant_id = ?
        """
        params = [model_id, tenant_id]
        
        if status_filter:
            query += " AND JSON_EXTRACT(metadata, '$.status') = ?"
            params.append(status_filter.value)
        
        query += " ORDER BY created_at DESC"
        
        cursor.execute(query, params)
        results = cursor.fetchall()
        conn.close()
        
        versions = []
        for result in results:
            metadata = json.loads(result[0])
            version_obj = ModelVersion.from_dict(metadata)
            
            if include_metrics and result[1]:
                version_obj.metrics = json.loads(result[1])
            
            versions.append(version_obj)
        
        return versions
    
    def promote_version(self,
                       model_id: str,
                       version_number: str,
                       target_status: VersionStatus,
                       tenant_id: str = "default",
                       strategy: PromotionStrategy = PromotionStrategy.MANUAL,
                       promoted_by: Optional[str] = None,
                       reason: Optional[str] = None,
                       config: Optional[Dict[str, Any]] = None) -> bool:
        """Promeut une version avec stratégie configurable"""
        
        # Récupération de la version
        model_version = self.get_version(model_id, version_number, tenant_id)
        if not model_version:
            logger.error(f"Version non trouvée: {model_id} v{version_number}")
            return False
        
        # Validation de la promotion
        if not self._validate_promotion(model_version, target_status):
            logger.error(f"Promotion invalide de {model_version.status} vers {target_status}")
            return False
        
        # Vérification des quotas pour production
        if target_status == VersionStatus.PRODUCTION:
            if not self._check_quota(tenant_id, 'production_models'):
                logger.error(f"Quota de modèles en production dépassé pour tenant {tenant_id}")
                return False
        
        # Application de la stratégie
        promotion_func = self.promotion_strategies.get(strategy)
        if not promotion_func:
            logger.error(f"Stratégie inconnue: {strategy}")
            return False
        
        try:
            # Mesure du temps de promotion
            start_time = time.time()
            
            # Sauvegarde de l'historique
            self._save_promotion_history(
                model_version,
                target_status,
                promoted_by,
                reason,
                strategy.value
            )
            
            # Exécution de la promotion
            success = promotion_func(model_version, target_status, config or {})
            
            if success:
                old_status = model_version.status
                model_version.status = target_status
                
                # Mise à jour du temps de calcul
                promotion_time = time.time() - start_time
                model_version.compute_hours_used += promotion_time / 3600
                
                # Estimation du coût
                model_version.estimated_cost = self._estimate_version_cost(model_version)
                
                # Sauvegarde
                self._save_version_to_db(model_version)
                
                # Si promotion en production, archiver l'ancienne version
                if target_status == VersionStatus.PRODUCTION:
                    self._archive_old_production_version(model_id, tenant_id, version_number)
                    
                    # Démarrer le monitoring de production
                    self._start_production_monitoring(model_version)
                
                logger.info(f"Version {model_id} v{version_number} promue de {old_status} vers {target_status}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Erreur lors de la promotion: {e}")
            return False
    
    def _validate_promotion(self, model_version: ModelVersion, target_status: VersionStatus) -> bool:
        """Valide qu'une promotion est autorisée avec règles étendues"""
        
        # Règles de promotion standards
        allowed_transitions = {
            VersionStatus.DEVELOPMENT: [VersionStatus.STAGING, VersionStatus.ARCHIVED, VersionStatus.AB_TESTING],
            VersionStatus.STAGING: [VersionStatus.PRODUCTION, VersionStatus.DEVELOPMENT, VersionStatus.ARCHIVED, VersionStatus.AB_TESTING],
            VersionStatus.PRODUCTION: [VersionStatus.ARCHIVED, VersionStatus.DEPRECATED],
            VersionStatus.ARCHIVED: [VersionStatus.DEVELOPMENT],
            VersionStatus.DEPRECATED: [],
            VersionStatus.FAILED: [VersionStatus.ARCHIVED],
            VersionStatus.AB_TESTING: [VersionStatus.PRODUCTION, VersionStatus.STAGING, VersionStatus.ARCHIVED]
        }
        
        # Vérification des métriques minimales pour production
        if target_status == VersionStatus.PRODUCTION:
            if not model_version.metrics:
                logger.warning("Pas de métriques disponibles pour promotion en production")
                return False
            
            # Seuils minimaux selon le type de modèle
            if model_version.model_type == "classification":
                min_accuracy = self.config.min_accuracy
                if model_version.metrics.get('accuracy', 0) < min_accuracy:
                    logger.warning(f"Accuracy ({model_version.metrics.get('accuracy')}) < seuil minimum ({min_accuracy})")
                    return False
            else:  # regression
                min_r2 = self.config.min_r2
                if model_version.metrics.get('r2', -float('inf')) < min_r2:
                    logger.warning(f"R2 ({model_version.metrics.get('r2')}) < seuil minimum ({min_r2})")
                    return False
        
        return target_status in allowed_transitions.get(model_version.status, [])
    
    def _manual_promotion(self, model_version: ModelVersion, target_status: VersionStatus, config: Dict) -> bool:
        """Promotion manuelle avec validation basique"""
        if not model_version.metrics:
            logger.warning("Aucune métrique disponible pour la promotion")
        
        # Vérification optionnelle du drift
        if config.get('check_drift', False):
            if model_version.drift_detected:
                logger.warning("Drift détecté sur le modèle")
                if not config.get('force', False):
                    return False
        
        return True
    
    def _auto_threshold_promotion(self, model_version: ModelVersion, target_status: VersionStatus, config: Dict) -> bool:
        """Promotion automatique basée sur des seuils configurables"""
        
        # Récupération des seuils depuis config ou valeurs par défaut
        thresholds = config.get('thresholds', {})
        if not thresholds:
            thresholds = {
                VersionStatus.STAGING: {
                    "accuracy": 0.8,
                    "f1": 0.75,
                    "auc": 0.75
                },
                VersionStatus.PRODUCTION: {
                    "accuracy": 0.85,
                    "f1": 0.80,
                    "auc": 0.85
                }
            }
        
        required_metrics = thresholds.get(target_status, {})
        
        # Vérification des seuils
        for metric_name, threshold in required_metrics.items():
            if metric_name not in model_version.metrics:
                logger.warning(f"Métrique {metric_name} manquante")
                return False
            
            if model_version.metrics[metric_name] < threshold:
                logger.warning(f"Métrique {metric_name} ({model_version.metrics[metric_name]}) "
                             f"en dessous du seuil ({threshold})")
                return False
        
        return True
    
    def _auto_improvement_promotion(self, model_version: ModelVersion, target_status: VersionStatus, config: Dict) -> bool:
        """Promotion si amélioration significative par rapport à la version actuelle"""
        
        # Récupération de la version actuelle en production
        current_prod = self._get_current_production_version(model_version.model_id, model_version.tenant_id)
        
        if not current_prod:
            # Pas de version en production, promotion automatique
            return True
        
        # Comparaison des versions
        comparison = self.compare_versions(
            model_version.model_id,
            current_prod.version,
            model_version.version,
            model_version.tenant_id
        )
        
        # Seuil d'amélioration requis (configurable)
        min_improvement = config.get('min_improvement', 0.05)  # 5% par défaut
        
        if comparison.improvement_rate < min_improvement:
            logger.warning(f"Amélioration insuffisante: {comparison.improvement_rate:.2%} < {min_improvement:.2%}")
            return False
        
        return comparison.is_significantly_better
    
    def _blue_green_deployment(self, model_version: ModelVersion, target_status: VersionStatus, config: Dict) -> bool:
        """Déploiement blue-green avec bascule instantanée et rollback automatique"""
        
        if target_status != VersionStatus.PRODUCTION:
            return self._manual_promotion(model_version, target_status, config)
        
        try:
            # Préparation de l'environnement "green"
            logger.info(f"Préparation environnement green pour {model_version.model_id}")
            
            # Configuration du déploiement
            model_version.deployment_config = {
                "strategy": "blue_green",
                "environment": "green",
                "ready_for_switch": False,
                "health_check_url": config.get('health_check_url'),
                "rollback_threshold": config.get('rollback_threshold', 0.1)
            }
            
            # Déploiement en parallèle (utilise le DeploymentManager de infrastructure.py)
            from infrastructure import DeploymentManager
            deployment_manager = DeploymentManager(self.tenant_manager)
            
            if self.config.storage.backend == "local":
                # Déploiement Docker local
                container_id = deployment_manager.deploy_model_docker(
                    model_version.tenant_id,
                    model_version.model_path,
                    port=8001  # Port green
                )
                model_version.container_id = container_id
            else:
                # Déploiement Kubernetes
                deployment_name = deployment_manager.deploy_model_kubernetes(
                    model_version.tenant_id,
                    model_version.model_path,
                    replicas=config.get('replicas', 3)
                )
                model_version.container_id = deployment_name
            
            # Tests de santé et monitoring initial
            if self._health_check(model_version):
                # Monitoring pendant la période de test
                test_duration = config.get('test_duration_seconds', 300)  # 5 minutes par défaut
                time.sleep(test_duration)
                
                # Vérification des métriques pendant le test
                if self._check_deployment_metrics(model_version, config):
                    # Bascule blue -> green
                    model_version.deployment_config["ready_for_switch"] = True
                    model_version.deployment_config["environment"] = "blue"  # Maintenant en production
                    logger.info("Bascule blue -> green effectuée avec succès")
                    return True
                else:
                    # Rollback automatique
                    logger.warning("Métriques dégradées, rollback automatique")
                    self._cleanup_deployment(model_version)
                    return False
            
            return False
            
        except Exception as e:
            logger.error(f"Erreur déploiement blue-green: {e}")
            self._cleanup_deployment(model_version)
            return False
    
    def _canary_deployment(self, model_version: ModelVersion, target_status: VersionStatus, config: Dict) -> bool:
        """Déploiement progressif canary avec monitoring continu"""
        
        if target_status != VersionStatus.PRODUCTION:
            return self._manual_promotion(model_version, target_status, config)
        
        try:
            # Configuration canary
            canary_config = {
                "strategy": "canary",
                "initial_traffic": config.get('initial_traffic', 5),  # 5% du trafic initial
                "increment": config.get('increment', 10),  # Augmentation de 10% à chaque étape
                "error_threshold": config.get('error_threshold', 0.01),  # 1% d'erreurs max
                "rollback_on_error": config.get('rollback_on_error', True),
                "monitoring_interval": config.get('monitoring_interval', 60)  # secondes
            }
            
            model_version.deployment_config = canary_config
            
            # Déploiement initial avec trafic minimal
            from infrastructure import DeploymentManager
            deployment_manager = DeploymentManager(self.tenant_manager)
            
            if self.config.storage.backend == "local":
                container_id = deployment_manager.deploy_model_docker(
                    model_version.tenant_id,
                    model_version.model_path,
                    port=8002
                )
                model_version.container_id = container_id
            else:
                deployment_name = deployment_manager.deploy_model_kubernetes(
                    model_version.tenant_id,
                    model_version.model_path,
                    replicas=1  # Commence avec peu de replicas
                )
                model_version.container_id = deployment_name
            
            # Déploiement progressif
            current_traffic = canary_config["initial_traffic"]
            
            while current_traffic <= 100:
                logger.info(f"Canary: {current_traffic}% du trafic")
                
                # Mise à jour de la configuration de trafic
                model_version.deployment_config["current_traffic"] = current_traffic
                
                # Attente et monitoring
                time.sleep(canary_config["monitoring_interval"])
                
                # Vérification des métriques
                metrics = self._monitor_canary_metrics(model_version, current_traffic)
                
                if not metrics['healthy']:
                    if canary_config["rollback_on_error"]:
                        logger.warning(f"Métriques canary dégradées à {current_traffic}%, rollback")
                        self._cleanup_deployment(model_version)
                        return False
                    else:
                        logger.warning(f"Métriques dégradées mais rollback désactivé")
                
                # Augmentation du trafic
                if current_traffic < 100:
                    current_traffic = min(100, current_traffic + canary_config["increment"])
                else:
                    break
            
            logger.info("Déploiement canary réussi - 100% du trafic")
            return True
            
        except Exception as e:
            logger.error(f"Erreur déploiement canary: {e}")
            self._cleanup_deployment(model_version)
            return False
    
    def _shadow_deployment(self, model_version: ModelVersion, target_status: VersionStatus, config: Dict) -> bool:
        """Déploiement shadow pour comparaison sans impact sur la production"""
        
        if target_status != VersionStatus.PRODUCTION:
            return self._manual_promotion(model_version, target_status, config)
        
        try:
            # Configuration shadow
            model_version.deployment_config = {
                "strategy": "shadow",
                "shadow_duration_hours": config.get('shadow_duration_hours', 24),
                "comparison_metrics": config.get('comparison_metrics', ["latency", "accuracy", "error_rate"]),
                "async_processing": True
            }
            
            logger.info(f"Déploiement shadow de {model_version.model_id} v{model_version.version}")
            
            # Déploiement en mode shadow (pas de trafic réel)
            from infrastructure import DeploymentManager
            deployment_manager = DeploymentManager(self.tenant_manager)
            
            container_id = deployment_manager.deploy_model_docker(
                model_version.tenant_id,
                model_version.model_path,
                port=8003  # Port shadow
            )
            model_version.container_id = container_id
            
            # Simulation de la période shadow avec collecte de métriques
            shadow_start = time.time()
            shadow_duration_seconds = model_version.deployment_config["shadow_duration_hours"] * 3600
            
            shadow_results = {
                "predictions_count": 0,
                "latency_samples": [],
                "accuracy_samples": [],
                "errors": 0
            }
            
            # Monitoring pendant la période shadow
            while (time.time() - shadow_start) < shadow_duration_seconds:
                # Collecte des métriques shadow (en production, les requêtes seraient dupliquées)
                metrics = self._collect_shadow_metrics(model_version)
                
                shadow_results["predictions_count"] += metrics.get("count", 0)
                shadow_results["latency_samples"].append(metrics.get("latency", 0))
                if "accuracy" in metrics:
                    shadow_results["accuracy_samples"].append(metrics["accuracy"])
                shadow_results["errors"] += metrics.get("errors", 0)
                
                # Vérification périodique
                time.sleep(60)  # Check toutes les minutes
            
            # Analyse des résultats shadow
            analysis = self._analyze_shadow_results(model_version, shadow_results)
            
            if analysis.get("performance_acceptable", False):
                logger.info("Résultats shadow acceptables, promotion confirmée")
                self._cleanup_deployment(model_version)  # Nettoyer le déploiement shadow
                return True
            else:
                logger.warning(f"Résultats shadow insuffisants: {analysis.get('reason')}")
                self._cleanup_deployment(model_version)
                return False
            
        except Exception as e:
            logger.error(f"Erreur déploiement shadow: {e}")
            self._cleanup_deployment(model_version)
            return False
    
    def _ab_testing_deployment(self, model_version: ModelVersion, target_status: VersionStatus, config: Dict) -> bool:
        """Déploiement A/B testing avec analyse statistique"""
        
        try:
            import uuid
            
            # Configuration A/B test
            ab_config = {
                "test_id": str(uuid.uuid4()),
                "traffic_split": config.get('traffic_split', 0.5),  # 50/50 par défaut
                "min_samples": config.get('min_samples', 1000),
                "confidence_level": config.get('confidence_level', 0.95),
                "test_duration_hours": config.get('test_duration_hours', 72)  # 3 jours
            }
            
            model_version.ab_test_id = ab_config["test_id"]
            model_version.ab_test_group = "treatment"
            model_version.deployment_config = ab_config
            
            # Récupération de la version control (actuelle)
            control_version = self._get_current_production_version(
                model_version.model_id, 
                model_version.tenant_id
            )
            
            if not control_version:
                logger.warning("Pas de version control pour A/B test")
                return False
            
            control_version.ab_test_id = ab_config["test_id"]
            control_version.ab_test_group = "control"
            
            # Déploiement des deux versions
            from infrastructure import DeploymentManager
            deployment_manager = DeploymentManager(self.tenant_manager)
            
            # Déployer la nouvelle version
            container_id = deployment_manager.deploy_model_docker(
                model_version.tenant_id,
                model_version.model_path,
                port=8004
            )
            model_version.container_id = container_id
            
            # Initialisation des métriques A/B
            ab_metrics = {
                "control": {"count": 0, "successes": 0, "errors": 0, "latency": []},
                "treatment": {"count": 0, "successes": 0, "errors": 0, "latency": []}
            }
            
            # Exécution du test A/B
            test_start = time.time()
            test_duration_seconds = ab_config["test_duration_hours"] * 3600
            
            while (time.time() - test_start) < test_duration_seconds:
                # Collecte des métriques pour les deux groupes
                control_metrics = self._collect_ab_metrics(control_version)
                treatment_metrics = self._collect_ab_metrics(model_version)
                
                ab_metrics["control"]["count"] += control_metrics.get("count", 0)
                ab_metrics["control"]["successes"] += control_metrics.get("successes", 0)
                ab_metrics["treatment"]["count"] += treatment_metrics.get("count", 0)
                ab_metrics["treatment"]["successes"] += treatment_metrics.get("successes", 0)
                
                # Vérification du nombre minimal d'échantillons
                if (ab_metrics["control"]["count"] >= ab_config["min_samples"] and
                    ab_metrics["treatment"]["count"] >= ab_config["min_samples"]):
                    
                    # Analyse statistique
                    winner = self._analyze_ab_test(ab_metrics, ab_config["confidence_level"])
                    
                    if winner:
                        logger.info(f"Gagnant A/B test: {winner}")
                        
                        # Sauvegarder les résultats
                        self._save_ab_test_results(
                            ab_config["test_id"],
                            model_version.model_id,
                            control_version.version,
                            model_version.version,
                            winner,
                            ab_metrics
                        )
                        
                        # Promotion si treatment gagne
                        if winner == "treatment":
                            self._cleanup_deployment(control_version)
                            return True
                        else:
                            self._cleanup_deployment(model_version)
                            return False
                
                time.sleep(60)  # Vérification toutes les minutes
            
            # Temps écoulé sans gagnant clair
            logger.warning("A/B test terminé sans gagnant significatif")
            self._cleanup_deployment(model_version)
            return False
            
        except Exception as e:
            logger.error(f"Erreur déploiement A/B testing: {e}")
            return False
    
    def rollback_version(self,
                        model_id: str,
                        target_version: str,
                        tenant_id: str = "default",
                        reason: Optional[str] = None,
                        rolled_by: Optional[str] = None) -> bool:
        """Rollback avec sauvegarde des métriques avant/après"""
        
        try:
            # Récupération de la version actuelle en production
            current_prod = self._get_current_production_version(model_id, tenant_id)
            if not current_prod:
                logger.error("Aucune version en production à rollback")
                return False
            
            # Récupération de la version cible
            target = self.get_version(model_id, target_version, tenant_id)
            if not target:
                logger.error(f"Version cible non trouvée: {target_version}")
                return False
            
            # Capture des métriques avant rollback
            metrics_before = current_prod.metrics.copy()
            
            # Sauvegarde de l'historique de rollback
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO rollback_history 
                (model_id, from_version, to_version, tenant_id, rolled_back_at, rolled_back_by, reason, metrics_before)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                model_id,
                current_prod.version,
                target_version,
                tenant_id,
                datetime.now(),
                rolled_by,
                reason,
                json.dumps(metrics_before)
            ))
            
            conn.commit()
            
            # Effectuer le rollback
            current_prod.status = VersionStatus.ARCHIVED
            target.status = VersionStatus.PRODUCTION
            
            # Nettoyer le déploiement actuel
            self._cleanup_deployment(current_prod)
            
            # Redéployer la version cible
            from infrastructure import DeploymentManager
            deployment_manager = DeploymentManager(self.tenant_manager)
            
            container_id = deployment_manager.deploy_model_docker(
                tenant_id,
                target.model_path,
                port=8000
            )
            target.container_id = container_id
            
            # Mise à jour en base
            self._save_version_to_db(current_prod)
            self._save_version_to_db(target)
            
            # Capture des métriques après rollback (après un délai)
            time.sleep(60)  # Attendre 1 minute pour stabilisation
            metrics_after = self._collect_production_metrics(target)
            
            # Mise à jour de l'historique avec métriques après
            cursor.execute("""
                UPDATE rollback_history 
                SET metrics_after = ?
                WHERE model_id = ? AND from_version = ? AND to_version = ?
                ORDER BY rolled_back_at DESC LIMIT 1
            """, (
                json.dumps(metrics_after),
                model_id,
                current_prod.version,
                target_version
            ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Rollback effectué: {model_id} de v{current_prod.version} vers v{target_version}")
            
            # Notification du rollback
            self._notify_rollback(model_id, current_prod.version, target_version, reason)
            
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors du rollback: {e}")
            return False
    
    def compare_versions(self,
                        model_id: str,
                        version_a: str,
                        version_b: str,
                        tenant_id: str = "default") -> VersionComparison:
        """Compare deux versions avec tests statistiques avancés"""
        
        # Récupération des versions
        model_a = self.get_version(model_id, version_a, tenant_id)
        model_b = self.get_version(model_id, version_b, tenant_id)
        
        if not model_a or not model_b:
            raise ValueError("Une ou plusieurs versions non trouvées")
        
        # Création de la comparaison
        comparison = VersionComparison(
            version_a=model_a,
            version_b=model_b
        )
        
        # Calcul des différences de métriques
        for metric_name in set(model_a.metrics.keys()) | set(model_b.metrics.keys()):
            val_a = model_a.metrics.get(metric_name, 0)
            val_b = model_b.metrics.get(metric_name, 0)
            comparison.metrics_diff[metric_name] = val_b - val_a
        
        # Calcul du taux d'amélioration global
        improvements = []
        for metric, diff in comparison.metrics_diff.items():
            if metric in model_a.metrics and model_a.metrics[metric] != 0:
                improvements.append(diff / model_a.metrics[metric])
        
        if improvements:
            comparison.improvement_rate = np.mean(improvements)
        
        # Tests statistiques
        comparison.statistical_tests = self._perform_statistical_tests(model_a, model_b)
        comparison.is_significantly_better = comparison.statistical_tests.get("significant", False)
        
        # Comparaison des coûts
        comparison.cost_diff = model_b.estimated_cost - model_a.estimated_cost
        if model_a.estimated_cost > 0:
            comparison.efficiency_ratio = (model_b.metrics.get('accuracy', 0) / model_b.estimated_cost) / \
                                         (model_a.metrics.get('accuracy', 0) / model_a.estimated_cost)
        
        # Recommandation basée sur métriques et coût
        if comparison.improvement_rate > 0.1:
            comparison.recommendation = f"Version {version_b} fortement recommandée (amélioration de {comparison.improvement_rate:.1%})"
        elif comparison.improvement_rate > 0 and comparison.cost_diff <= 0:
            comparison.recommendation = f"Version {version_b} recommandée (meilleure et moins chère)"
        elif comparison.improvement_rate > 0.05 and comparison.efficiency_ratio > 1:
            comparison.recommendation = f"Version {version_b} recommandée (meilleur rapport qualité/coût)"
        else:
            comparison.recommendation = f"Version {version_a} reste préférable"
        
        # Évaluation des risques
        comparison.risk_assessment = {
            "drift_risk": max(model_a.drift_score, model_b.drift_score),
            "stability": "stable" if abs(comparison.improvement_rate) < 0.02 else "changing",
            "cost_increase": comparison.cost_diff > 0,
            "requires_retraining": model_b.drift_detected
        }
        
        # Sauvegarde de la comparaison
        self._save_comparison_to_db(comparison)
        
        return comparison
    
    def get_version_history(self,
                           model_id: str,
                           tenant_id: str = "default",
                           days_back: int = 30) -> pd.DataFrame:
        """Récupère l'historique avec métriques de billing"""
        
        conn = sqlite3.connect(str(self.db_path))
        
        query = """
            SELECT 
                version,
                status,
                created_at,
                created_by,
                JSON_EXTRACT(metrics, '$.accuracy') as accuracy,
                JSON_EXTRACT(metrics, '$.f1') as f1,
                JSON_EXTRACT(metrics, '$.auc') as auc,
                compute_hours_used,
                predictions_count,
                storage_mb_used,
                estimated_cost
            FROM model_versions
            WHERE model_id = ? 
                AND tenant_id = ?
                AND created_at > datetime('now', ?)
            ORDER BY created_at DESC
        """
        
        df = pd.read_sql_query(
            query,
            conn,
            params=(model_id, tenant_id, f'-{days_back} days')
        )
        
        conn.close()
        
        # Ajout de colonnes calculées
        if not df.empty:
            df['cost_per_prediction'] = df['estimated_cost'] / df['predictions_count'].replace(0, 1)
            df['efficiency_score'] = df['accuracy'] / df['estimated_cost'].replace(0, 1)
        
        return df
    
    def cleanup_old_versions(self,
                            model_id: str,
                            tenant_id: str = "default",
                            keep_last_n: int = 10,
                            keep_production: bool = True,
                            keep_days: int = 90) -> int:
        """Nettoie les anciennes versions avec politique de rétention"""
        
        versions = self.list_versions(model_id, tenant_id)
        
        # Tri par date de création
        versions.sort(key=lambda v: v.created_at, reverse=True)
        
        # Identification des versions à supprimer
        versions_to_delete = []
        kept_count = 0
        cutoff_date = datetime.now() - timedelta(days=keep_days)
        
        for v in versions:
            # Garder les versions en production si demandé
            if keep_production and v.status == VersionStatus.PRODUCTION:
                continue
            
            # Garder les versions récentes
            if v.created_at > cutoff_date:
                continue
            
            # Garder les N dernières versions
            if kept_count < keep_last_n:
                kept_count += 1
                continue
            
            versions_to_delete.append(v)
        
        # Suppression avec libération des ressources
        deleted_count = 0
        total_storage_freed = 0
        
        for v in versions_to_delete:
            if self._delete_version(v):
                deleted_count += 1
                total_storage_freed += v.storage_mb_used
                
                # Libération des ressources tenant
                self.tenant_manager.release_resources(
                    tenant_id,
                    storage=int(v.storage_mb_used)
                )
        
        logger.info(f"Supprimé {deleted_count} versions pour {model_id}, "
                   f"libéré {total_storage_freed:.2f} MB")
        
        return deleted_count
    
    # Méthodes helper privées
    
    def _check_quota(self, tenant_id: str, resource_type: str) -> bool:
        """Vérifie les quotas du tenant"""
        tenant_config = self.tenant_manager.get_tenant(tenant_id)
        if not tenant_config:
            return True  # Pas de restriction si tenant non trouvé
        
        if resource_type == 'models':
            current_models = len(self.list_versions('*', tenant_id))
            return current_models < tenant_config.max_models
        elif resource_type == 'production_models':
            prod_models = len([v for v in self.list_versions('*', tenant_id) 
                              if v.status == VersionStatus.PRODUCTION])
            return prod_models < 3  # Limite fixe pour production
        
        return True
    
    def _check_tenant_access(self, tenant_id: str, model_id: str) -> bool:
        """Vérifie l'accès du tenant au modèle"""
        # Vérifier que le modèle appartient au tenant
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT COUNT(*) FROM model_versions
            WHERE model_id = ? AND tenant_id = ?
        """, (model_id, tenant_id))
        
        count = cursor.fetchone()[0]
        conn.close()
        
        return count > 0
    
    def _generate_next_version(self, model_id: str, tenant_id: str) -> str:
        """Génère le prochain numéro de version sémantique"""
        versions = self.list_versions(model_id, tenant_id)
        
        if not versions:
            return "1.0.0"
        
        # Trouver la version la plus récente
        latest = max(versions, key=lambda v: version.parse(v.version))
        
        # Parser la version
        v = version.parse(latest.version)
        
        # Incrémenter selon le type de changement
        # Pour simplifier, on incrémente toujours la version patch
        if hasattr(v, 'base_version'):
            parts = v.base_version.split('.')
            parts[-1] = str(int(parts[-1]) + 1)
            return '.'.join(parts)
        
        return "1.0.1"
    
    def _get_latest_version(self, model_id: str, tenant_id: str) -> Optional[str]:
        """Récupère le numéro de la dernière version"""
        versions = self.list_versions(model_id, tenant_id)
        
        if not versions:
            return None
        
        latest = max(versions, key=lambda v: version.parse(v.version))
        return latest.version
    
    def _get_current_production_version(self, model_id: str, tenant_id: str) -> Optional[ModelVersion]:
        """Récupère la version actuellement en production"""
        versions = self.list_versions(model_id, tenant_id, VersionStatus.PRODUCTION)
        
        if versions:
            return versions[0]  # Devrait n'y avoir qu'une seule version en production
        
        return None
    
    def _archive_old_production_version(self, model_id: str, tenant_id: str, new_version: str):
        """Archive l'ancienne version en production"""
        current = self._get_current_production_version(model_id, tenant_id)
        
        if current and current.version != new_version:
            current.status = VersionStatus.ARCHIVED
            self._save_version_to_db(current)
            
            # Nettoyer le déploiement
            self._cleanup_deployment(current)
    
    def _save_version_to_db(self, model_version: ModelVersion):
        """Sauvegarde une version en base de données"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO model_versions
            (model_id, version, tenant_id, created_at, created_by, status, metrics, metadata,
             model_type, deployment_config, ab_test_id, compute_hours_used, predictions_count,
             storage_mb_used, estimated_cost)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            model_version.model_id,
            model_version.version,
            model_version.tenant_id,
            model_version.created_at,
            model_version.created_by,
            model_version.status.value if isinstance(model_version.status, VersionStatus) else model_version.status,
            json.dumps(model_version.metrics),
            json.dumps(model_version.to_dict()),
            model_version.model_type,
            json.dumps(model_version.deployment_config) if model_version.deployment_config else None,
            model_version.ab_test_id,
            model_version.compute_hours_used,
            model_version.predictions_count,
            model_version.storage_mb_used,
            model_version.estimated_cost
        ))
        
        conn.commit()
        conn.close()
    
    def _save_promotion_history(self,
                               model_version: ModelVersion,
                               target_status: VersionStatus,
                               promoted_by: Optional[str],
                               reason: Optional[str],
                               strategy: str):
        """Sauvegarde l'historique des promotions"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO promotion_history
            (model_id, version, tenant_id, from_status, to_status, promoted_at, promoted_by, 
             reason, strategy, metrics_before)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            model_version.model_id,
            model_version.version,
            model_version.tenant_id,
            model_version.status.value if isinstance(model_version.status, VersionStatus) else model_version.status,
            target_status.value,
            datetime.now(),
            promoted_by,
            reason,
            strategy,
            json.dumps(model_version.metrics)
        ))
        
        conn.commit()
        conn.close()
    
    def _save_comparison_to_db(self, comparison: VersionComparison):
        """Sauvegarde une comparaison en base de données"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        comparison_data = {
            "metrics_diff": comparison.metrics_diff,
            "improvement_rate": comparison.improvement_rate,
            "is_significantly_better": comparison.is_significantly_better,
            "recommendation": comparison.recommendation,
            "cost_diff": comparison.cost_diff,
            "efficiency_ratio": comparison.efficiency_ratio,
            "risk_assessment": comparison.risk_assessment
        }
        
        cursor.execute("""
            INSERT INTO version_comparisons
            (model_id, version_a, version_b, tenant_id, compared_at, comparison_result,
             statistical_significance, recommendation)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            comparison.version_a.model_id,
            comparison.version_a.version,
            comparison.version_b.version,
            comparison.version_a.tenant_id,
            datetime.now(),
            json.dumps(comparison_data),
            comparison.confidence_level,
            comparison.recommendation
        ))
        
        conn.commit()
        conn.close()
    
    def _save_ab_test_results(self, test_id: str, model_id: str, 
                             version_a: str, version_b: str,
                             winner: str, results: Dict):
        """Sauvegarde les résultats d'un test A/B"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO ab_tests
            (test_id, model_id, version_a, version_b, tenant_id, started_at, ended_at,
             traffic_split, winner, results)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            test_id,
            model_id,
            version_a,
            version_b,
            "default",  # À récupérer du contexte
            datetime.now() - timedelta(hours=72),  # Approximation
            datetime.now(),
            0.5,  # Split par défaut
            winner,
            json.dumps(results)
        ))
        
        conn.commit()
        conn.close()
    
    def _delete_version(self, model_version: ModelVersion) -> bool:
        """Supprime une version et ses artefacts"""
        try:
            # Suppression du modèle physique via StorageManager
            self.storage.delete_model(
                model_version.model_id,
                model_version.version,
                model_version.tenant_id
            )
            
            # Suppression de la base de données
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute("""
                DELETE FROM model_versions
                WHERE model_id = ? AND version = ? AND tenant_id = ?
            """, (
                model_version.model_id,
                model_version.version,
                model_version.tenant_id
            ))
            
            conn.commit()
            conn.close()
            
            # Suppression du cache
            cache_key = f"{model_version.tenant_id}:{model_version.model_id}:{model_version.version}"
            if cache_key in self.version_cache:
                del self.version_cache[cache_key]
            
            # Nettoyage du déploiement si existant
            self._cleanup_deployment(model_version)
            
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de la suppression: {e}")
            return False
    
    def _calculate_hash(self, obj: Any) -> str:
        """Calcule le hash SHA256 d'un objet"""
        try:
            obj_bytes = pickle.dumps(obj)
            return hashlib.sha256(obj_bytes).hexdigest()
        except:
            return ""
    
    def _estimate_version_cost(self, model_version: ModelVersion) -> float:
        """Estime le coût d'une version basé sur l'utilisation"""
        # Tarifs simplifiés (à adapter selon votre modèle de pricing)
        cost_per_compute_hour = 0.10
        cost_per_gb_month = 0.023
        cost_per_1k_predictions = 0.001
        
        compute_cost = model_version.compute_hours_used * cost_per_compute_hour
        storage_cost = (model_version.storage_mb_used / 1024) * cost_per_gb_month
        prediction_cost = (model_version.predictions_count / 1000) * cost_per_1k_predictions
        
        return round(compute_cost + storage_cost + prediction_cost, 4)
    
    def _health_check(self, model_version: ModelVersion) -> bool:
        """Effectue un health check sur une version déployée"""
        if not model_version.deployment_config:
            return True  # Pas de déploiement à vérifier
        
        # Vérification basique - en production, faire un appel HTTP au endpoint
        if model_version.container_id:
            try:
                # Simulation d'un health check
                import requests
                health_url = model_version.deployment_config.get('health_check_url', 
                                                                 'http://localhost:8000/health')
                response = requests.get(health_url, timeout=5)
                return response.status_code == 200
            except:
                return False
        
        # Vérification que le modèle peut être chargé
        try:
            model, _ = self.storage.load_model(
                model_version.model_id,
                model_version.version,
                model_version.tenant_id
            )
            return model is not None
        except:
            return False
    
    def _check_deployment_metrics(self, model_version: ModelVersion, config: Dict) -> bool:
        """Vérifie les métriques d'un déploiement"""
        threshold = config.get('rollback_threshold', 0.1)
        
        # Récupération des métriques via monitoring
        monitor_key = f"{model_version.model_id}:{model_version.version}"
        if monitor_key in self.active_monitors:
            monitor = self.active_monitors[monitor_key]
            summary = monitor.get_performance_summary(last_n_days=1)
            
            if summary.get('metrics'):
                # Vérification des seuils
                accuracy = summary['metrics'].get('avg_accuracy', 0)
                if accuracy < (1 - threshold):
                    return False
        
        return True
    
    def _monitor_canary_metrics(self, model_version: ModelVersion, traffic_percentage: int) -> Dict:
        """Monitore les métriques pendant un déploiement canary"""
        metrics = {
            'healthy': True,
            'error_rate': 0,
            'latency_p99': 0,
            'accuracy': 1.0
        }
        
        monitor_key = f"{model_version.model_id}:{model_version.version}"
        if monitor_key in self.active_monitors:
            monitor = self.active_monitors[monitor_key]
            summary = monitor.get_performance_summary(last_n_days=1)
            
            if summary.get('metrics'):
                metrics['accuracy'] = summary['metrics'].get('avg_accuracy', 1.0)
                
                # Vérification des seuils
                if model_version.deployment_config:
                    error_threshold = model_version.deployment_config.get('error_threshold', 0.01)
                    if metrics.get('error_rate', 0) > error_threshold:
                        metrics['healthy'] = False
        
        return metrics
    
    def _collect_shadow_metrics(self, model_version: ModelVersion) -> Dict:
        """Collecte les métriques en mode shadow"""
        metrics = {
            'count': 0,
            'latency': 0,
            'errors': 0
        }
        
        monitor_key = f"{model_version.model_id}:{model_version.version}"
        if monitor_key in self.active_monitors:
            monitor = self.active_monitors[monitor_key]
            
            # Simulation de collecte de métriques
            # En production, les requêtes seraient réellement dupliquées
            metrics['count'] = np.random.randint(10, 100)
            metrics['latency'] = np.random.uniform(10, 100)  # ms
            metrics['errors'] = np.random.randint(0, 2)
            
            # Si des vraies prédictions sont disponibles
            if monitor.prediction_history:
                metrics['count'] = len(monitor.prediction_history)
        
        return metrics
    
    def _analyze_shadow_results(self, model_version: ModelVersion, shadow_results: Dict) -> Dict:
        """Analyse les résultats d'un déploiement shadow"""
        analysis = {
            'performance_acceptable': True,
            'reason': '',
            'metrics': {}
        }
        
        if shadow_results['predictions_count'] == 0:
            analysis['performance_acceptable'] = False
            analysis['reason'] = "Pas assez de données shadow"
            return analysis
        
        # Calcul des métriques moyennes
        avg_latency = np.mean(shadow_results['latency_samples']) if shadow_results['latency_samples'] else 0
        error_rate = shadow_results['errors'] / shadow_results['predictions_count'] if shadow_results['predictions_count'] > 0 else 0
        
        analysis['metrics'] = {
            'avg_latency_ms': avg_latency,
            'error_rate': error_rate,
            'total_predictions': shadow_results['predictions_count']
        }
        
        # Vérification des seuils
        if avg_latency > 1000:  # 1 seconde
            analysis['performance_acceptable'] = False
            analysis['reason'] = f"Latence trop élevée: {avg_latency:.0f}ms"
        elif error_rate > 0.01:  # 1% d'erreurs
            analysis['performance_acceptable'] = False
            analysis['reason'] = f"Taux d'erreur trop élevé: {error_rate:.2%}"
        
        if shadow_results['accuracy_samples']:
            avg_accuracy = np.mean(shadow_results['accuracy_samples'])
            analysis['metrics']['avg_accuracy'] = avg_accuracy
            if avg_accuracy < 0.8:
                analysis['performance_acceptable'] = False
                analysis['reason'] = f"Accuracy insuffisante: {avg_accuracy:.2%}"
        
        return analysis
    
    def _collect_ab_metrics(self, model_version: ModelVersion) -> Dict:
        """Collecte les métriques pour un test A/B"""
        metrics = {
            'count': 0,
            'successes': 0,
            'errors': 0,
            'latency': []
        }
        
        monitor_key = f"{model_version.model_id}:{model_version.version}"
        if monitor_key in self.active_monitors:
            monitor = self.active_monitors[monitor_key]
            
            # Simulation de métriques A/B
            # En production, collecte réelle depuis le monitoring
            metrics['count'] = np.random.randint(50, 200)
            metrics['successes'] = int(metrics['count'] * np.random.uniform(0.7, 0.95))
            metrics['errors'] = np.random.randint(0, 5)
            metrics['latency'] = [np.random.uniform(10, 100) for _ in range(10)]
        
        return metrics
    
    def _analyze_ab_test(self, ab_metrics: Dict, confidence_level: float) -> Optional[str]:
        """Analyse statistique d'un test A/B"""
        from scipy import stats
        
        # Extraction des données
        control_count = ab_metrics['control']['count']
        control_successes = ab_metrics['control']['successes']
        treatment_count = ab_metrics['treatment']['count']
        treatment_successes = ab_metrics['treatment']['successes']
        
        if control_count == 0 or treatment_count == 0:
            return None
        
        # Calcul des taux de conversion
        control_rate = control_successes / control_count
        treatment_rate = treatment_successes / treatment_count
        
        # Test de proportion z-test
        pooled_rate = (control_successes + treatment_successes) / (control_count + treatment_count)
        se = np.sqrt(pooled_rate * (1 - pooled_rate) * (1/control_count + 1/treatment_count))
        
        if se == 0:
            return None
        
        z_score = (treatment_rate - control_rate) / se
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        
        # Détermination du gagnant
        alpha = 1 - confidence_level
        if p_value < alpha:
            if treatment_rate > control_rate:
                return "treatment"
            else:
                return "control"
        
        return None  # Pas de différence significative
    
    def _collect_production_metrics(self, model_version: ModelVersion) -> Dict:
        """Collecte les métriques de production"""
        metrics = {}
        
        monitor_key = f"{model_version.model_id}:{model_version.version}"
        if monitor_key in self.active_monitors:
            monitor = self.active_monitors[monitor_key]
            summary = monitor.get_performance_summary(last_n_days=1)
            
            if summary.get('metrics'):
                metrics = summary['metrics']
        
        return metrics
    
    def _cleanup_deployment(self, model_version: ModelVersion):
        """Nettoie un déploiement (Docker/K8s)"""
        if not model_version.container_id:
            return
        
        try:
            from infrastructure import DeploymentManager
            
            if self.config.storage.backend == "local":
                # Nettoyage Docker
                import docker
                client = docker.from_env()
                try:
                    container = client.containers.get(model_version.container_id)
                    container.stop()
                    container.remove()
                    logger.info(f"Container {model_version.container_id} nettoyé")
                except:
                    pass
            else:
                # Nettoyage Kubernetes
                # Implémenter selon votre configuration K8s
                pass
            
            model_version.container_id = None
            
        except Exception as e:
            logger.error(f"Erreur lors du nettoyage du déploiement: {e}")
    
    def _start_production_monitoring(self, model_version: ModelVersion):
        """Démarre le monitoring de production pour une version"""
        monitor_key = f"{model_version.model_id}:{model_version.version}"
        
        if monitor_key not in self.active_monitors:
            monitor = self.monitoring_service.register_model(
                model_id=f"{model_version.model_id}_v{model_version.version}",
                model_type=model_version.model_type,
                tenant_id=model_version.tenant_id
            )
            self.active_monitors[monitor_key] = monitor
        
        logger.info(f"Monitoring de production démarré pour {model_version.model_id} v{model_version.version}")
    
    def _notify_rollback(self, model_id: str, from_version: str, to_version: str, reason: Optional[str]):
        """Envoie une notification de rollback"""
        # Intégration avec le système de notification (Slack, email, etc.)
        notification = {
            'type': 'rollback',
            'model_id': model_id,
            'from_version': from_version,
            'to_version': to_version,
            'reason': reason,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.warning(f"ROLLBACK NOTIFICATION: {notification}")
        
        # En production, envoyer via les canaux configurés
        # Par exemple, utiliser MonitoringIntegration.send_to_slack()
    
    def _perform_statistical_tests(self, model_a: ModelVersion, model_b: ModelVersion) -> Dict:
        """Effectue des tests statistiques entre deux versions"""
        tests = {
            'significant': False,
            'test_type': None,
            'p_value': 1.0,
            'effect_size': 0
        }
        
        # Si pas assez de données, pas de test
        if not model_a.metrics or not model_b.metrics:
            return tests
        
        # Test simple sur l'accuracy (ou métrique principale)
        if 'accuracy' in model_a.metrics and 'accuracy' in model_b.metrics:
            # Simulation d'un test statistique
            # En production, utiliser les vraies distributions
            diff = abs(model_b.metrics['accuracy'] - model_a.metrics['accuracy'])
            
            # Seuil arbitraire pour la démo
            if diff > 0.05:  # 5% de différence
                tests['significant'] = True
                tests['test_type'] = 'accuracy_comparison'
                tests['p_value'] = 0.01  # Simulé
                tests['effect_size'] = diff
        
        return tests
    
    def export_version_to_docker(self, model_id: str, version_number: str, 
                                tenant_id: str = "default") -> str:
        """Export une version spécifique vers Docker"""
        model_version = self.get_version(model_id, version_number, tenant_id)
        if not model_version:
            raise ValueError(f"Version {version_number} non trouvée")
        
        # Utilise la méthode du StorageManager
        return self.storage.export_model_to_docker(
            model_id, 
            version_number, 
            tenant_id
        )
    
    def export_version_to_onnx(self, model_id: str, version_number: str,
                              tenant_id: str = "default") -> str:
        """Export une version vers ONNX"""
        model_version = self.get_version(model_id, version_number, tenant_id)
        if not model_version:
            raise ValueError(f"Version {version_number} non trouvée")
        
        return self.storage.export_model_to_onnx(
            model_id,
            version_number,
            tenant_id
        )
    
    def get_deployment_status(self, model_id: str, version_number: str,
                             tenant_id: str = "default") -> Dict:
        """Récupère le statut de déploiement d'une version"""
        model_version = self.get_version(model_id, version_number, tenant_id)
        if not model_version:
            return {'status': 'not_found'}
        
        status = {
            'model_id': model_id,
            'version': version_number,
            'deployment_status': model_version.status.value,
            'container_id': model_version.container_id,
            'deployment_config': model_version.deployment_config,
            'endpoints': model_version.endpoints,
            'health': 'unknown'
        }
        
        # Vérification de santé si déployé
        if model_version.container_id:
            status['health'] = 'healthy' if self._health_check(model_version) else 'unhealthy'
        
        # Métriques de monitoring si disponibles
        monitor_key = f"{model_id}:{version_number}"
        if monitor_key in self.active_monitors:
            monitor = self.active_monitors[monitor_key]
            status['monitoring'] = monitor.get_performance_summary(last_n_days=1)
        
        return status


# Export des classes principales
__all__ = [
    'ModelVersionManager',
    'ModelVersion',
    'VersionStatus',
    'PromotionStrategy',
    'VersionComparison'
]
