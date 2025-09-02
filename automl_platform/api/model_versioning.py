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
import shutil
import sqlite3

# Imports depuis votre structure
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from storage import StorageManager, ModelMetadata
from config import AutoMLConfig, load_config
from monitoring import ModelMonitor, MonitoringService
from metrics import calculate_metrics, compare_models_metrics
from infrastructure import TenantManager

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


class PromotionStrategy(str, Enum):
    """Stratégies de promotion de versions"""
    MANUAL = "manual"
    AUTO_THRESHOLD = "auto_threshold"
    AUTO_IMPROVEMENT = "auto_improvement"
    BLUE_GREEN = "blue_green"
    CANARY = "canary"
    SHADOW = "shadow"


@dataclass
class ModelVersion:
    """Représentation d'une version de modèle"""
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
    
    # Artefacts
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
    
    # Monitoring
    drift_detected: bool = False
    last_monitoring_check: Optional[datetime] = None
    monitoring_alerts: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire"""
        data = asdict(self)
        # Conversion des datetime en string
        for key in ['created_at', 'last_monitoring_check']:
            if key in data and data[key]:
                data[key] = data[key].isoformat() if isinstance(data[key], datetime) else data[key]
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelVersion':
        """Crée depuis un dictionnaire"""
        # Conversion des strings en datetime
        for key in ['created_at', 'last_monitoring_check']:
            if key in data and data[key] and isinstance(data[key], str):
                data[key] = datetime.fromisoformat(data[key])
        return cls(**data)
    
    def is_newer_than(self, other: 'ModelVersion') -> bool:
        """Compare les versions"""
        return version.parse(self.version) > version.parse(other.version)


@dataclass
class VersionComparison:
    """Résultat de comparaison entre versions"""
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


class ModelVersionManager:
    """Gestionnaire principal des versions de modèles"""
    
    def __init__(self, config: AutoMLConfig = None):
        self.config = config or load_config()
        self.storage = StorageManager(backend=self.config.storage.backend)
        self.monitoring = MonitoringService(self.storage)
        self.tenant_manager = TenantManager()
        
        # Base de données de versioning
        self.db_path = Path(self.config.storage.local_base_path) / "versioning.db"
        self._init_database()
        
        # Cache des versions
        self.version_cache: Dict[str, ModelVersion] = {}
        
        # Stratégies de promotion
        self.promotion_strategies: Dict[str, callable] = {
            PromotionStrategy.MANUAL: self._manual_promotion,
            PromotionStrategy.AUTO_THRESHOLD: self._auto_threshold_promotion,
            PromotionStrategy.AUTO_IMPROVEMENT: self._auto_improvement_promotion,
            PromotionStrategy.BLUE_GREEN: self._blue_green_deployment,
            PromotionStrategy.CANARY: self._canary_deployment,
            PromotionStrategy.SHADOW: self._shadow_deployment
        }
    
    def _init_database(self):
        """Initialise la base de données de versioning"""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Table des versions
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
                metrics_before JSON,
                metrics_after JSON
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
                comparison_result JSON
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
                reason TEXT
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
        """Crée une nouvelle version de modèle"""
        
        # Génération automatique du numéro de version si non fourni
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
            **metadata if metadata else {}
        )
        
        # Calcul des hashs pour la traçabilité
        if model_object:
            model_version.pipeline_hash = self._calculate_hash(model_object)
        
        # Sauvegarde du modèle
        try:
            # Préparation des métadonnées pour le storage
            storage_metadata = ModelMetadata(
                model_id=model_id,
                version=version_number,
                model_type=model_version.algorithm or "unknown",
                algorithm=model_version.algorithm,
                created_at=model_version.created_at.isoformat(),
                updated_at=datetime.now().isoformat(),
                metrics=model_version.metrics,
                parameters=model_version.parameters,
                feature_names=model_version.feature_names,
                target_name="target",
                dataset_hash=model_version.dataset_hash or "",
                pipeline_hash=model_version.pipeline_hash or "",
                tags=list(model_version.tags.keys()),
                description=model_version.description,
                author=model_version.created_by or "system",
                tenant_id=tenant_id
            )
            
            # Sauvegarde physique
            model_path = self.storage.save_model(model_object, storage_metadata)
            model_version.model_path = model_path
            
            # Sauvegarde en base de données
            self._save_version_to_db(model_version)
            
            # Mise en cache
            cache_key = f"{tenant_id}:{model_id}:{version_number}"
            self.version_cache[cache_key] = model_version
            
            logger.info(f"Version créée: {model_id} v{version_number} pour tenant {tenant_id}")
            return model_version
            
        except Exception as e:
            logger.error(f"Erreur lors de la création de la version: {e}")
            raise
    
    def get_version(self, 
                   model_id: str, 
                   version_number: str = "latest",
                   tenant_id: str = "default") -> Optional[ModelVersion]:
        """Récupère une version spécifique"""
        
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
                     status_filter: Optional[VersionStatus] = None) -> List[ModelVersion]:
        """Liste toutes les versions d'un modèle"""
        
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        query = """
            SELECT metadata FROM model_versions
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
            versions.append(ModelVersion.from_dict(metadata))
        
        return versions
    
    def promote_version(self,
                       model_id: str,
                       version_number: str,
                       target_status: VersionStatus,
                       tenant_id: str = "default",
                       strategy: PromotionStrategy = PromotionStrategy.MANUAL,
                       promoted_by: Optional[str] = None,
                       reason: Optional[str] = None) -> bool:
        """Promeut une version vers un nouveau statut"""
        
        # Récupération de la version
        model_version = self.get_version(model_id, version_number, tenant_id)
        if not model_version:
            logger.error(f"Version non trouvée: {model_id} v{version_number}")
            return False
        
        # Validation de la promotion
        if not self._validate_promotion(model_version, target_status):
            logger.error(f"Promotion invalide de {model_version.status} vers {target_status}")
            return False
        
        # Application de la stratégie de promotion
        promotion_func = self.promotion_strategies.get(strategy)
        if not promotion_func:
            logger.error(f"Stratégie inconnue: {strategy}")
            return False
        
        try:
            # Sauvegarde de l'historique
            self._save_promotion_history(
                model_version,
                target_status,
                promoted_by,
                reason
            )
            
            # Exécution de la promotion
            success = promotion_func(model_version, target_status)
            
            if success:
                # Mise à jour du statut
                old_status = model_version.status
                model_version.status = target_status
                
                # Sauvegarde
                self._save_version_to_db(model_version)
                
                # Si promotion en production, archiver l'ancienne version
                if target_status == VersionStatus.PRODUCTION:
                    self._archive_old_production_version(model_id, tenant_id, version_number)
                
                logger.info(f"Version {model_id} v{version_number} promue de {old_status} vers {target_status}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Erreur lors de la promotion: {e}")
            return False
    
    def _validate_promotion(self, model_version: ModelVersion, target_status: VersionStatus) -> bool:
        """Valide qu'une promotion est autorisée"""
        
        # Règles de promotion
        allowed_transitions = {
            VersionStatus.DEVELOPMENT: [VersionStatus.STAGING, VersionStatus.ARCHIVED],
            VersionStatus.STAGING: [VersionStatus.PRODUCTION, VersionStatus.DEVELOPMENT, VersionStatus.ARCHIVED],
            VersionStatus.PRODUCTION: [VersionStatus.ARCHIVED, VersionStatus.DEPRECATED],
            VersionStatus.ARCHIVED: [VersionStatus.DEVELOPMENT],
            VersionStatus.DEPRECATED: [],
            VersionStatus.FAILED: [VersionStatus.ARCHIVED]
        }
        
        return target_status in allowed_transitions.get(model_version.status, [])
    
    def _manual_promotion(self, model_version: ModelVersion, target_status: VersionStatus) -> bool:
        """Promotion manuelle simple"""
        # Vérifications de base
        if not model_version.metrics:
            logger.warning("Aucune métrique disponible pour la promotion")
        
        return True
    
    def _auto_threshold_promotion(self, model_version: ModelVersion, target_status: VersionStatus) -> bool:
        """Promotion automatique basée sur des seuils"""
        
        # Définition des seuils selon le statut cible
        thresholds = {
            VersionStatus.STAGING: {
                "accuracy": 0.8,
                "f1": 0.75
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
    
    def _auto_improvement_promotion(self, model_version: ModelVersion, target_status: VersionStatus) -> bool:
        """Promotion si amélioration par rapport à la version actuelle"""
        
        # Récupération de la version actuelle en production
        current_prod = self._get_current_production_version(model_version.model_id, model_version.tenant_id)
        
        if not current_prod:
            # Pas de version en production, promotion automatique
            return True
        
        # Comparaison des métriques
        comparison = self.compare_versions(
            model_version.model_id,
            current_prod.version,
            model_version.version,
            model_version.tenant_id
        )
        
        return comparison.is_significantly_better
    
    def _blue_green_deployment(self, model_version: ModelVersion, target_status: VersionStatus) -> bool:
        """Déploiement blue-green avec bascule instantanée"""
        
        if target_status != VersionStatus.PRODUCTION:
            return self._manual_promotion(model_version, target_status)
        
        try:
            # Préparation de l'environnement "green"
            logger.info(f"Préparation environnement green pour {model_version.model_id}")
            
            # Déploiement en parallèle
            model_version.deployment_config = {
                "strategy": "blue_green",
                "environment": "green",
                "ready_for_switch": False
            }
            
            # Tests de santé
            if self._health_check(model_version):
                # Bascule
                model_version.deployment_config["ready_for_switch"] = True
                logger.info("Bascule blue -> green effectuée")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Erreur déploiement blue-green: {e}")
            return False
    
    def _canary_deployment(self, model_version: ModelVersion, target_status: VersionStatus) -> bool:
        """Déploiement progressif canary"""
        
        if target_status != VersionStatus.PRODUCTION:
            return self._manual_promotion(model_version, target_status)
        
        try:
            # Configuration canary
            canary_config = {
                "strategy": "canary",
                "initial_traffic": 5,  # 5% du trafic initial
                "increment": 10,  # Augmentation de 10% à chaque étape
                "error_threshold": 0.01,  # 1% d'erreurs max
                "rollback_on_error": True
            }
            
            model_version.deployment_config = canary_config
            
            # Simulation du déploiement progressif
            current_traffic = canary_config["initial_traffic"]
            
            while current_traffic <= 100:
                logger.info(f"Canary: {current_traffic}% du trafic")
                
                # Vérification des métriques (simulé)
                if not self._monitor_canary_metrics(model_version, current_traffic):
                    logger.warning("Métriques canary dégradées, rollback")
                    return False
                
                current_traffic += canary_config["increment"]
            
            logger.info("Déploiement canary réussi")
            return True
            
        except Exception as e:
            logger.error(f"Erreur déploiement canary: {e}")
            return False
    
    def _shadow_deployment(self, model_version: ModelVersion, target_status: VersionStatus) -> bool:
        """Déploiement shadow pour comparaison sans impact"""
        
        if target_status != VersionStatus.PRODUCTION:
            return self._manual_promotion(model_version, target_status)
        
        try:
            # Configuration shadow
            model_version.deployment_config = {
                "strategy": "shadow",
                "shadow_duration_hours": 24,
                "comparison_metrics": ["latency", "accuracy", "error_rate"]
            }
            
            logger.info(f"Déploiement shadow de {model_version.model_id} v{model_version.version}")
            
            # Simulation de la période shadow
            # En production, cela impliquerait de router le trafic en double
            
            # Analyse des résultats shadow
            shadow_results = self._analyze_shadow_results(model_version)
            
            if shadow_results.get("performance_acceptable", False):
                logger.info("Résultats shadow acceptables, promotion confirmée")
                return True
            
            logger.warning("Résultats shadow insuffisants")
            return False
            
        except Exception as e:
            logger.error(f"Erreur déploiement shadow: {e}")
            return False
    
    def rollback_version(self,
                        model_id: str,
                        target_version: str,
                        tenant_id: str = "default",
                        reason: Optional[str] = None,
                        rolled_by: Optional[str] = None) -> bool:
        """Effectue un rollback vers une version précédente"""
        
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
            
            # Sauvegarde de l'historique de rollback
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO rollback_history 
                (model_id, from_version, to_version, tenant_id, rolled_back_at, rolled_back_by, reason)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                model_id,
                current_prod.version,
                target_version,
                tenant_id,
                datetime.now(),
                rolled_by,
                reason
            ))
            
            conn.commit()
            conn.close()
            
            # Effectuer le rollback
            current_prod.status = VersionStatus.ARCHIVED
            target.status = VersionStatus.PRODUCTION
            
            self._save_version_to_db(current_prod)
            self._save_version_to_db(target)
            
            logger.info(f"Rollback effectué: {model_id} de v{current_prod.version} vers v{target_version}")
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors du rollback: {e}")
            return False
    
    def compare_versions(self,
                        model_id: str,
                        version_a: str,
                        version_b: str,
                        tenant_id: str = "default") -> VersionComparison:
        """Compare deux versions de modèle"""
        
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
        
        # Tests statistiques (simplifiés)
        comparison.is_significantly_better = comparison.improvement_rate > 0.05
        
        # Recommandation
        if comparison.improvement_rate > 0.1:
            comparison.recommendation = f"Version {version_b} fortement recommandée (amélioration de {comparison.improvement_rate:.1%})"
        elif comparison.improvement_rate > 0:
            comparison.recommendation = f"Version {version_b} légèrement meilleure"
        else:
            comparison.recommendation = f"Version {version_a} reste préférable"
        
        # Sauvegarde de la comparaison
        self._save_comparison_to_db(comparison)
        
        return comparison
    
    def get_version_history(self,
                           model_id: str,
                           tenant_id: str = "default",
                           days_back: int = 30) -> pd.DataFrame:
        """Récupère l'historique des versions sous forme de DataFrame"""
        
        conn = sqlite3.connect(str(self.db_path))
        
        query = """
            SELECT 
                version,
                status,
                created_at,
                created_by,
                JSON_EXTRACT(metrics, '$.accuracy') as accuracy,
                JSON_EXTRACT(metrics, '$.f1') as f1,
                JSON_EXTRACT(metrics, '$.auc') as auc
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
        
        return df
    
    def cleanup_old_versions(self,
                            model_id: str,
                            tenant_id: str = "default",
                            keep_last_n: int = 10,
                            keep_production: bool = True) -> int:
        """Nettoie les anciennes versions"""
        
        versions = self.list_versions(model_id, tenant_id)
        
        # Tri par date de création
        versions.sort(key=lambda v: v.created_at, reverse=True)
        
        # Identification des versions à supprimer
        versions_to_delete = []
        kept_count = 0
        
        for v in versions:
            if keep_production and v.status == VersionStatus.PRODUCTION:
                continue
            
            if kept_count >= keep_last_n:
                versions_to_delete.append(v)
            else:
                kept_count += 1
        
        # Suppression
        deleted_count = 0
        for v in versions_to_delete:
            if self._delete_version(v):
                deleted_count += 1
        
        logger.info(f"Supprimé {deleted_count} versions pour {model_id}")
        return deleted_count
    
    # Méthodes privées d'aide
    
    def _generate_next_version(self, model_id: str, tenant_id: str) -> str:
        """Génère le prochain numéro de version"""
        
        versions = self.list_versions(model_id, tenant_id)
        
        if not versions:
            return "1.0.0"
        
        # Trouver la version la plus récente
        latest = max(versions, key=lambda v: version.parse(v.version))
        
        # Incrémenter la version mineure
        v = version.parse(latest.version)
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
            return versions[0]  # Should only have one production version
        
        return None
    
    def _archive_old_production_version(self, model_id: str, tenant_id: str, new_version: str):
        """Archive l'ancienne version en production"""
        
        current = self._get_current_production_version(model_id, tenant_id)
        
        if current and current.version != new_version:
            current.status = VersionStatus.ARCHIVED
            self._save_version_to_db(current)
    
    def _save_version_to_db(self, model_version: ModelVersion):
        """Sauvegarde une version en base de données"""
        
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO model_versions
            (model_id, version, tenant_id, created_at, created_by, status, metrics, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            model_version.model_id,
            model_version.version,
            model_version.tenant_id,
            model_version.created_at,
            model_version.created_by,
            model_version.status.value,
            json.dumps(model_version.metrics),
            json.dumps(model_version.to_dict())
        ))
        
        conn.commit()
        conn.close()
    
    def _save_promotion_history(self,
                               model_version: ModelVersion,
                               target_status: VersionStatus,
                               promoted_by: Optional[str],
                               reason: Optional[str]):
        """Sauvegarde l'historique des promotions"""
        
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO promotion_history
            (model_id, version, tenant_id, from_status, to_status, promoted_at, promoted_by, reason, metrics_before)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            model_version.model_id,
            model_version.version,
            model_version.tenant_id,
            model_version.status.value,
            target_status.value,
            datetime.now(),
            promoted_by,
            reason,
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
            "recommendation": comparison.recommendation
        }
        
        cursor.execute("""
            INSERT INTO version_comparisons
            (model_id, version_a, version_b, tenant_id, compared_at, comparison_result)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            comparison.version_a.model_id,
            comparison.version_a.version,
            comparison.version_b.version,
            comparison.version_a.tenant_id,
            datetime.now(),
            json.dumps(comparison_data)
        ))
        
        conn.commit()
        conn.close()
    
    def _delete_version(self, model_version: ModelVersion) -> bool:
        """Supprime une version"""
        
        try:
            # Suppression du modèle physique
            # (à implémenter selon votre storage)
            
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
            
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de la suppression: {e}")
            return False
    
    def _calculate_hash(self, obj: Any) -> str:
        """Calcule le hash d'un objet"""
        
        try:
            obj_bytes = pickle.dumps(obj)
            return hashlib.sha256(obj_bytes).hexdigest()
        except:
            return ""
    
    def _health_check(self, model_version: ModelVersion) -> bool:
        """Effectue un health check sur une version"""
        
        # Vérifications de base
        if not model_version.model_path:
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
    
    def _monitor_canary_metrics(self, model_version: ModelVersion, traffic_percentage: int) -> bool:
        """Monitore les métriques pendant un déploiement canary"""
        
        # Simulation - en production, cela interrogerait les métriques réelles
        if model_version.metrics.get("error_rate", 0) > 0.01:
            return False
        
        if model_version.metrics.get("latency_p99", 0) > 1000:  # 1 seconde
            return False
        
        return True
    
    def _analyze_shadow_results(self, model_version: ModelVersion) -> Dict[str, Any]:
        """Analyse les résultats d'un déploiement shadow"""
        
        # Simulation - en production, cela comparerait les prédictions shadow
        return {
            "performance_acceptable": True,
            "latency_difference": 10,  # ms
            "accuracy_difference": 0.02,
            "error_rate_difference": 0.001
        }


# Export des classes principales
__all__ = [
    'ModelVersionManager',
    'ModelVersion',
    'VersionStatus',
    'PromotionStrategy',
    'VersionComparison'
]
