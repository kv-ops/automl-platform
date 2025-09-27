"""
Module de prédiction par batch pour AutoML Platform
Gère les prédictions asynchrones sur de gros volumes de données
Place dans: automl_platform/api/batch_inference.py
"""

import asyncio
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from pathlib import Path
import json
import uuid
from enum import Enum
from pydantic import BaseModel, Field

try:
    import ray
    from ray import serve
    RAY_AVAILABLE = True
except ImportError:
    ray = None  # type: ignore[assignment]
    serve = None  # type: ignore[assignment]
    RAY_AVAILABLE = False

if not RAY_AVAILABLE:
    class _RayStub:
        """Fallback implementation when Ray is unavailable."""

        def remote(self, func=None, **kwargs):  # type: ignore[override]
            if func is None:
                def decorator(f):
                    return f
                return decorator
            return func

        def is_initialized(self):
            return False

        def init(self, *args, **kwargs):
            raise ImportError("ray is not installed")

        def get(self, *args, **kwargs):
            raise ImportError("ray is not installed")

        def kill(self, *args, **kwargs):
            raise ImportError("ray is not installed")

        def available_resources(self):
            return {}

        def shutdown(self):
            return None

    ray = _RayStub()  # type: ignore[assignment]

try:
    import dask.dataframe as dd
    DASK_AVAILABLE = True
except ImportError:
    dd = None  # type: ignore[assignment]
    DASK_AVAILABLE = False
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import pickle
import joblib
from sqlalchemy import create_engine
from redis import Redis

try:
    import boto3
    BOTO3_AVAILABLE = True
except ImportError:
    boto3 = None  # type: ignore[assignment]
    BOTO3_AVAILABLE = False

try:
    from azure.storage.blob import BlobServiceClient
    AZURE_BLOB_AVAILABLE = True
except ImportError:
    BlobServiceClient = None  # type: ignore[assignment]
    AZURE_BLOB_AVAILABLE = False
import psutil
import traceback
import time
from io import BytesIO

# Import des modules internes depuis votre structure existante
from ..storage import StorageManager, ModelMetadata
from ..monitoring import ModelMonitor, MonitoringService
from ..data_prep import EnhancedDataPreprocessor
from ..model_selection import get_available_models
from ..metrics import calculate_metrics
from .infrastructure import TenantManager, ResourceMonitor
from .feature_store import FeatureStore
from ..config import AutoMLConfig, load_config

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BatchStatus(str, Enum):
    """Statuts possibles pour un job de batch"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PARTIAL = "partial"


class BatchPriority(str, Enum):
    """Niveaux de priorité pour les jobs"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class BatchJobConfig(BaseModel):
    """Configuration d'un job de batch inference"""
    job_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    model_id: str
    model_version: Optional[str] = "latest"
    tenant_id: str = "default"
    
    # Sources et destinations
    input_source: str  # path, s3, azure, database, feature_store
    input_config: Dict[str, Any]
    output_destination: str  # path, s3, azure, database
    output_config: Dict[str, Any]
    
    # Paramètres de traitement
    batch_size: int = Field(default=1000, ge=1, le=100000)
    priority: BatchPriority = BatchPriority.NORMAL
    max_workers: int = Field(default=4, ge=1, le=32)
    timeout_seconds: int = Field(default=3600, ge=60)
    retry_count: int = Field(default=3, ge=0, le=10)
    
    # Options de traitement
    enable_monitoring: bool = True
    enable_caching: bool = True
    enable_drift_detection: bool = True
    enable_quality_checks: bool = True
    
    # Configuration du prétraitement
    preprocessing_config: Optional[Dict[str, Any]] = None
    postprocessing_config: Optional[Dict[str, Any]] = None
    
    # Notifications
    notification_config: Optional[Dict[str, Any]] = None
    
    # Planification
    scheduled_time: Optional[datetime] = None
    
    # Métadonnées
    tags: Dict[str, str] = Field(default_factory=dict)
    created_by: Optional[str] = None
    project_id: Optional[str] = None


class BatchJobResult(BaseModel):
    """Résultat d'un job de batch inference"""
    job_id: str
    status: BatchStatus
    tenant_id: str = "default"
    
    # Timing
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    
    # Métriques de traitement
    total_records: int = 0
    processed_records: int = 0
    failed_records: int = 0
    success_rate: float = 0.0
    
    # Résultats
    output_location: Optional[str] = None
    error_messages: List[str] = Field(default_factory=list)
    
    # Métriques détaillées
    metrics: Dict[str, Any] = Field(default_factory=dict)
    drift_detected: bool = False
    quality_score: Optional[float] = None
    
    # Coûts et ressources
    cost_estimate: Optional[float] = None
    resources_used: Dict[str, Any] = Field(default_factory=dict)


class DataLoader:
    """Chargeur de données multi-sources"""
    
    def __init__(self, config: AutoMLConfig):
        self.config = config
        self.storage_manager = StorageManager(backend=config.storage.backend)
        self.feature_store = None
        if config.feature_store.enabled:
            self.feature_store = FeatureStore(config.feature_store.to_dict())
        
        # Clients cloud
        self.s3_client = None
        self.azure_client = None
        self._init_cloud_clients()
    
    def _init_cloud_clients(self):
        """Initialise les clients cloud si configurés"""
        if self.config.storage.backend == "s3":
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=self.config.storage.access_key,
                aws_secret_access_key=self.config.storage.secret_key
            )
        
        # Azure Blob Storage
        if self.config.storage.backend == "azure":
            self.azure_client = BlobServiceClient(
                account_url=f"https://{self.config.storage.endpoint}",
                credential=self.config.storage.secret_key
            )
    
    async def load_data(self, source: str, config: Dict[str, Any]) -> pd.DataFrame:
        """Charge les données depuis différentes sources"""
        try:
            if source == "file":
                return await self._load_from_file(config["path"])
            elif source == "s3":
                return await self._load_from_s3(config["bucket"], config["key"])
            elif source == "azure":
                return await self._load_from_azure(config["container"], config["blob"])
            elif source == "database":
                return await self._load_from_database(config["connection_string"], config["query"])
            elif source == "feature_store":
                return await self._load_from_feature_store(config["feature_names"], config["entity_ids"])
            elif source == "storage":
                return await self._load_from_storage(config["dataset_id"], config.get("tenant_id", "default"))
            else:
                raise ValueError(f"Source non supportée: {source}")
        except Exception as e:
            logger.error(f"Erreur lors du chargement des données: {e}")
            raise
    
    async def _load_from_file(self, path: str) -> pd.DataFrame:
        """Charge depuis un fichier local"""
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Fichier non trouvé: {path}")
        
        if file_path.suffix == '.csv':
            return pd.read_csv(path)
        elif file_path.suffix == '.parquet':
            return pd.read_parquet(path)
        elif file_path.suffix == '.json':
            return pd.read_json(path)
        elif file_path.suffix == '.feather':
            return pd.read_feather(path)
        else:
            raise ValueError(f"Format non supporté: {file_path.suffix}")
    
    async def _load_from_s3(self, bucket: str, key: str) -> pd.DataFrame:
        """Charge depuis S3"""
        if not self.s3_client:
            raise ValueError("Client S3 non configuré")
        
        response = self.s3_client.get_object(Bucket=bucket, Key=key)
        data = response['Body'].read()
        
        if key.endswith('.csv'):
            return pd.read_csv(BytesIO(data))
        elif key.endswith('.parquet'):
            return pd.read_parquet(BytesIO(data))
        else:
            return pd.read_json(BytesIO(data))
    
    async def _load_from_azure(self, container: str, blob: str) -> pd.DataFrame:
        """Charge depuis Azure Blob Storage"""
        if not self.azure_client:
            raise ValueError("Client Azure non configuré")
        
        blob_client = self.azure_client.get_blob_client(container=container, blob=blob)
        data = blob_client.download_blob().readall()
        
        if blob.endswith('.csv'):
            return pd.read_csv(BytesIO(data))
        elif blob.endswith('.parquet'):
            return pd.read_parquet(BytesIO(data))
        else:
            return pd.read_json(BytesIO(data))
    
    async def _load_from_database(self, connection_string: str, query: str) -> pd.DataFrame:
        """Charge depuis une base de données"""
        engine = create_engine(connection_string)
        return pd.read_sql_query(query, engine)
    
    async def _load_from_feature_store(self, feature_names: List[str], entity_ids: List[str]) -> pd.DataFrame:
        """Charge depuis le feature store"""
        if not self.feature_store:
            raise ValueError("Feature store non configuré")
        
        return self.feature_store.read_features(feature_names, entity_ids)
    
    async def _load_from_storage(self, dataset_id: str, tenant_id: str) -> pd.DataFrame:
        """Charge depuis le storage manager"""
        return self.storage_manager.load_dataset(dataset_id, tenant_id)


class DataSaver:
    """Sauvegarde des résultats multi-destinations"""
    
    def __init__(self, config: AutoMLConfig):
        self.config = config
        self.storage_manager = StorageManager(backend=config.storage.backend)
        
        # Clients cloud
        self.s3_client = None
        self.azure_client = None
        self._init_cloud_clients()
    
    def _init_cloud_clients(self):
        """Initialise les clients cloud"""
        if self.config.storage.backend == "s3":
            self.s3_client = boto3.client('s3')
        
        if self.config.storage.backend == "azure":
            self.azure_client = BlobServiceClient(
                account_url=f"https://{self.config.storage.endpoint}",
                credential=self.config.storage.secret_key
            )
    
    async def save_data(self, data: pd.DataFrame, destination: str, 
                       config: Dict[str, Any], tenant_id: str = "default") -> str:
        """Sauvegarde les données vers différentes destinations"""
        try:
            if destination == "file":
                return await self._save_to_file(data, config["path"])
            elif destination == "s3":
                return await self._save_to_s3(data, config["bucket"], config["key"])
            elif destination == "azure":
                return await self._save_to_azure(data, config["container"], config["blob"])
            elif destination == "database":
                return await self._save_to_database(data, config["connection_string"], config["table"])
            elif destination == "storage":
                return await self._save_to_storage(data, config["dataset_id"], tenant_id)
            else:
                raise ValueError(f"Destination non supportée: {destination}")
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde: {e}")
            raise
    
    async def _save_to_file(self, data: pd.DataFrame, path: str) -> str:
        """Sauvegarde vers un fichier local"""
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        if file_path.suffix == '.csv':
            data.to_csv(path, index=False)
        elif file_path.suffix == '.parquet':
            data.to_parquet(path, index=False)
        elif file_path.suffix == '.json':
            data.to_json(path, orient='records')
        else:
            data.to_feather(path)
        
        return path
    
    async def _save_to_s3(self, data: pd.DataFrame, bucket: str, key: str) -> str:
        """Sauvegarde vers S3"""
        if not self.s3_client:
            raise ValueError("Client S3 non configuré")
        
        buffer = BytesIO()
        if key.endswith('.csv'):
            data.to_csv(buffer, index=False)
        elif key.endswith('.parquet'):
            data.to_parquet(buffer, index=False)
        else:
            data.to_json(buffer, orient='records')
        
        buffer.seek(0)
        self.s3_client.put_object(Bucket=bucket, Key=key, Body=buffer.getvalue())
        return f"s3://{bucket}/{key}"
    
    async def _save_to_azure(self, data: pd.DataFrame, container: str, blob: str) -> str:
        """Sauvegarde vers Azure"""
        if not self.azure_client:
            raise ValueError("Client Azure non configuré")
        
        blob_client = self.azure_client.get_blob_client(container=container, blob=blob)
        
        buffer = BytesIO()
        if blob.endswith('.csv'):
            data.to_csv(buffer, index=False)
        elif blob.endswith('.parquet'):
            data.to_parquet(buffer, index=False)
        else:
            data.to_json(buffer, orient='records')
        
        buffer.seek(0)
        blob_client.upload_blob(buffer.getvalue(), overwrite=True)
        return f"azure://{container}/{blob}"
    
    async def _save_to_database(self, data: pd.DataFrame, connection_string: str, table: str) -> str:
        """Sauvegarde vers une base de données"""
        engine = create_engine(connection_string)
        data.to_sql(table, engine, if_exists='append', index=False)
        return f"database://{table}"
    
    async def _save_to_storage(self, data: pd.DataFrame, dataset_id: str, tenant_id: str) -> str:
        """Sauvegarde via le storage manager"""
        return self.storage_manager.save_dataset(data, dataset_id, tenant_id)


@ray.remote
class BatchWorker:
    """Worker Ray pour le traitement parallèle"""
    
    def __init__(self, model_id: str, model_version: str, tenant_id: str, config_dict: Dict):
        self.model_id = model_id
        self.model_version = model_version
        self.tenant_id = tenant_id
        self.config = AutoMLConfig(**config_dict)
        self.model = None
        self.preprocessor = None
        self._load_model()
    
    def _load_model(self):
        """Charge le modèle et le préprocesseur"""
        storage = StorageManager(backend=self.config.storage.backend)
        model_data = storage.load_model(self.model_id, self.model_version, self.tenant_id)
        
        if model_data:
            self.model = model_data[0]
            # Charger le préprocesseur si disponible
            try:
                preprocessor_id = f"{self.model_id}_preprocessor"
                preprocessor_data = storage.load_model(preprocessor_id, self.model_version, self.tenant_id)
                if preprocessor_data:
                    self.preprocessor = preprocessor_data[0]
            except:
                logger.warning(f"Pas de préprocesseur trouvé pour {self.model_id}")
    
    def predict_batch(self, data: pd.DataFrame) -> pd.DataFrame:
        """Effectue les prédictions sur un batch"""
        try:
            # Prétraitement si disponible
            if self.preprocessor:
                data_processed = self.preprocessor.transform(data)
            else:
                data_processed = data
            
            # Prédiction
            predictions = self.model.predict(data_processed)
            
            # Ajout des prédictions au DataFrame
            data['prediction'] = predictions
            
            # Probabilités si disponibles
            if hasattr(self.model, 'predict_proba'):
                try:
                    probas = self.model.predict_proba(data_processed)
                    if len(probas.shape) > 1:
                        for i in range(probas.shape[1]):
                            data[f'proba_class_{i}'] = probas[:, i]
                except:
                    pass
            
            # Ajout de métadonnées
            data['model_id'] = self.model_id
            data['model_version'] = self.model_version
            data['prediction_timestamp'] = datetime.now()
            
            return data
            
        except Exception as e:
            logger.error(f"Erreur lors de la prédiction: {e}")
            data['prediction_error'] = str(e)
            return data


class BatchInferenceEngine:
    """Moteur principal de batch inference"""
    
    def __init__(self, config: AutoMLConfig = None):
        self.config = config or load_config()
        self.storage_manager = StorageManager(backend=self.config.storage.backend)
        self.monitoring_service = MonitoringService(self.storage_manager)
        self.tenant_manager = TenantManager()
        self.data_loader = DataLoader(self.config)
        self.data_saver = DataSaver(self.config)
        
        # Cache Redis si disponible
        try:
            self.redis_client = Redis(
                host=self.config.feature_store.redis_config.get("host", "localhost"),
                port=self.config.feature_store.redis_config.get("port", 6379)
            )
            self.redis_client.ping()
        except:
            self.redis_client = None
            logger.warning("Redis non disponible, cache désactivé")
        
        # Jobs actifs
        self.active_jobs: Dict[str, BatchJobResult] = {}
        
        # Executor pour les tâches asynchrones
        self.executor = ProcessPoolExecutor(max_workers=self.config.worker.max_workers)
        
        # Initialisation Ray
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True, num_cpus=self.config.worker.max_workers)
    
    async def submit_job(self, job_config: BatchJobConfig) -> BatchJobResult:
        """Soumet un nouveau job de batch inference"""
        logger.info(f"Soumission du job {job_config.job_id} pour le tenant {job_config.tenant_id}")
        
        # Vérification des quotas
        if self.config.billing.enabled:
            tenant = self.tenant_manager.get_tenant(job_config.tenant_id)
            if tenant:
                max_concurrent = tenant.max_concurrent_jobs
                current_jobs = sum(1 for j in self.active_jobs.values() 
                                 if j.tenant_id == job_config.tenant_id and 
                                 j.status == BatchStatus.PROCESSING)
                if current_jobs >= max_concurrent:
                    raise ValueError(f"Limite de jobs concurrents atteinte ({max_concurrent})")
        
        # Création du résultat initial
        result = BatchJobResult(
            job_id=job_config.job_id,
            status=BatchStatus.PENDING,
            tenant_id=job_config.tenant_id,
            start_time=datetime.now()
        )
        
        # Stockage en cache
        self.active_jobs[job_config.job_id] = result
        self._cache_job_status(job_config.job_id, result)
        
        # Lancement asynchrone du traitement
        asyncio.create_task(self._process_job(job_config, result))
        
        return result
    
    async def _process_job(self, job_config: BatchJobConfig, result: BatchJobResult):
        """Traite un job de batch inference"""
        monitor = None
        
        try:
            # Mise à jour du statut
            result.status = BatchStatus.PROCESSING
            self._cache_job_status(job_config.job_id, result)
            
            # Allocation des ressources
            if not self.tenant_manager.allocate_resources(
                job_config.tenant_id,
                cpu=job_config.max_workers,
                memory=4  # GB par worker
            ):
                raise ValueError("Ressources insuffisantes")
            
            # Enregistrement du monitoring
            if job_config.enable_monitoring:
                monitor = self.monitoring_service.register_model(
                    job_config.model_id,
                    "classification",  # À déterminer dynamiquement
                    tenant_id=job_config.tenant_id
                )
            
            # Chargement des données
            logger.info(f"Chargement des données pour le job {job_config.job_id}")
            data = await self.data_loader.load_data(
                job_config.input_source,
                job_config.input_config
            )
            result.total_records = len(data)
            
            # Vérification de la qualité des données
            if job_config.enable_quality_checks:
                preprocessor = EnhancedDataPreprocessor(self.config.to_dict())
                quality_report = preprocessor.check_data_quality(data)
                result.quality_score = quality_report.get("quality_score", 100)
                
                if not quality_report.get("valid", True):
                    logger.warning(f"Problèmes de qualité détectés: {quality_report.get('issues')}")
            
            # Détection de drift
            if job_config.enable_drift_detection and monitor:
                drift_results = monitor.check_drift(data)
                result.drift_detected = drift_results.get("drift_detected", False)
                if result.drift_detected:
                    logger.warning(f"Drift détecté pour le job {job_config.job_id}")
            
            # Prétraitement global si nécessaire
            if job_config.preprocessing_config:
                preprocessor = EnhancedDataPreprocessor(job_config.preprocessing_config)
                data = preprocessor.fit_transform(data)
            
            # Division en batches
            batches = self._create_batches(data, job_config.batch_size)
            logger.info(f"Traitement de {len(batches)} batches")
            
            # Création des workers Ray
            workers = []
            for i in range(min(job_config.max_workers, len(batches))):
                worker = BatchWorker.remote(
                    job_config.model_id,
                    job_config.model_version,
                    job_config.tenant_id,
                    self.config.to_dict()
                )
                workers.append(worker)
            
            # Distribution du travail
            futures = []
            for i, batch in enumerate(batches):
                worker = workers[i % len(workers)]
                futures.append(worker.predict_batch.remote(batch))
            
            # Collecte des résultats avec timeout
            results_list = []
            start_batch_time = time.time()
            
            for i, future in enumerate(futures):
                try:
                    remaining_time = job_config.timeout_seconds - (time.time() - start_batch_time)
                    if remaining_time <= 0:
                        raise TimeoutError("Timeout global atteint")
                    
                    batch_result = ray.get(future, timeout=min(60, remaining_time))
                    results_list.append(batch_result)
                    result.processed_records += len(batch_result)
                    
                    # Mise à jour périodique du cache
                    if i % 10 == 0:
                        self._cache_job_status(job_config.job_id, result)
                        
                except Exception as e:
                    logger.error(f"Erreur batch {i}: {e}")
                    result.failed_records += job_config.batch_size
                    result.error_messages.append(f"Batch {i}: {str(e)}")
                    
                    # Retry si configuré
                    if job_config.retry_count > 0:
                        for retry in range(job_config.retry_count):
                            try:
                                logger.info(f"Retry {retry+1} pour batch {i}")
                                batch_result = ray.get(future, timeout=30)
                                results_list.append(batch_result)
                                result.processed_records += len(batch_result)
                                result.failed_records -= job_config.batch_size
                                break
                            except:
                                continue
            
            # Consolidation des résultats
            if results_list:
                final_results = pd.concat(results_list, ignore_index=True)
                
                # Post-traitement
                if job_config.postprocessing_config:
                    final_results = self._postprocess_results(
                        final_results,
                        job_config.postprocessing_config
                    )
                
                # Sauvegarde des résultats
                output_location = await self.data_saver.save_data(
                    final_results,
                    job_config.output_destination,
                    job_config.output_config,
                    job_config.tenant_id
                )
                result.output_location = output_location
                
                # Logging des prédictions pour monitoring
                if monitor and job_config.enable_monitoring:
                    prediction_cols = [col for col in final_results.columns if 'prediction' in col.lower()]
                    if prediction_cols:
                        monitor.log_prediction(
                            final_results.drop(columns=prediction_cols),
                            final_results[prediction_cols[0]].values,
                            prediction_time=result.duration_seconds
                        )
                
                # Calcul des métriques finales
                result.success_rate = result.processed_records / result.total_records if result.total_records > 0 else 0
                result.metrics = self._calculate_job_metrics(final_results)
                
                # Estimation des ressources utilisées
                result.resources_used = {
                    "cpu_hours": (time.time() - start_batch_time) / 3600 * job_config.max_workers,
                    "memory_gb": job_config.max_workers * 4,
                    "storage_mb": final_results.memory_usage(deep=True).sum() / 1024**2
                }
            
            # Statut final
            if result.success_rate >= 0.95:
                result.status = BatchStatus.COMPLETED
            elif result.success_rate > 0:
                result.status = BatchStatus.PARTIAL
            else:
                result.status = BatchStatus.FAILED
            
        except Exception as e:
            logger.error(f"Erreur fatale job {job_config.job_id}: {e}")
            result.status = BatchStatus.FAILED
            result.error_messages.append(str(e))
            
        finally:
            # Libération des ressources
            if self.tenant_manager:
                self.tenant_manager.release_resources(
                    job_config.tenant_id,
                    cpu=job_config.max_workers,
                    memory=4
                )
            
            # Finalisation
            result.end_time = datetime.now()
            result.duration_seconds = (result.end_time - result.start_time).total_seconds()
            
            # Estimation du coût
            result.cost_estimate = self._estimate_cost(result, job_config)
            
            # Mise à jour finale en cache
            self._cache_job_status(job_config.job_id, result)
            
            # Notifications
            if job_config.notification_config:
                await self._send_notifications(job_config, result)
            
            # Nettoyage des workers Ray
            for worker in workers:
                try:
                    ray.kill(worker)
                except:
                    pass
    
    def _create_batches(self, data: pd.DataFrame, batch_size: int) -> List[pd.DataFrame]:
        """Divise les données en batches optimisés"""
        # Utilisation de Dask pour les gros datasets
        if len(data) > 100000:
            ddf = dd.from_pandas(data, npartitions=max(1, len(data) // batch_size))
            return [partition.compute() for partition in ddf.to_delayed()]
        else:
            n_batches = (len(data) + batch_size - 1) // batch_size
            return [data.iloc[i*batch_size:(i+1)*batch_size] for i in range(n_batches)]
    
    def _postprocess_results(self, data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Post-traitement avancé des résultats"""
        # Filtrage par seuil de confiance
        if "confidence_threshold" in config:
            threshold = config["confidence_threshold"]
            proba_cols = [col for col in data.columns if col.startswith('proba_')]
            if proba_cols:
                max_proba = data[proba_cols].max(axis=1)
                data = data[max_proba >= threshold]
        
        # Ajout de métadonnées enrichies
        if config.get("add_metadata", True):
            data["batch_id"] = str(uuid.uuid4())
            data["processing_timestamp"] = datetime.now()
            data["confidence_score"] = data[[col for col in data.columns if col.startswith('proba_')]].max(axis=1) if any(col.startswith('proba_') for col in data.columns) else 1.0
        
        # Agrégation si demandée
        if "aggregation" in config:
            agg_config = config["aggregation"]
            data = data.groupby(agg_config["group_by"]).agg(agg_config["operations"]).reset_index()
        
        # Tri des résultats
        if "sort_by" in config:
            data = data.sort_values(by=config["sort_by"], ascending=config.get("ascending", True))
        
        return data
    
    def _calculate_job_metrics(self, results: pd.DataFrame) -> Dict[str, Any]:
        """Calcule les métriques détaillées du job"""
        metrics = {
            "total_predictions": len(results),
            "columns_generated": list(results.columns),
            "memory_usage_mb": results.memory_usage(deep=True).sum() / 1024**2
        }
        
        # Analyse des prédictions
        if 'prediction' in results.columns:
            metrics["unique_predictions"] = results['prediction'].nunique()
            metrics["prediction_distribution"] = results['prediction'].value_counts().to_dict()
            
            # Statistiques descriptives
            if pd.api.types.is_numeric_dtype(results['prediction']):
                metrics["prediction_stats"] = {
                    "mean": float(results['prediction'].mean()),
                    "std": float(results['prediction'].std()),
                    "min": float(results['prediction'].min()),
                    "max": float(results['prediction'].max()),
                    "median": float(results['prediction'].median())
                }
        
        # Analyse des probabilités
        proba_cols = [col for col in results.columns if col.startswith('proba_')]
        if proba_cols:
            metrics["probability_stats"] = {}
            for col in proba_cols:
                metrics["probability_stats"][col] = {
                    "mean": float(results[col].mean()),
                    "std": float(results[col].std()),
                    "min": float(results[col].min()),
                    "max": float(results[col].max())
                }
        
        # Taux d'erreur
        if 'prediction_error' in results.columns:
            metrics["error_rate"] = results['prediction_error'].notna().sum() / len(results)
            metrics["error_messages"] = results[results['prediction_error'].notna()]['prediction_error'].value_counts().to_dict()
        
        return metrics
    
    def _estimate_cost(self, result: BatchJobResult, job_config: BatchJobConfig) -> float:
        """Estime le coût du job basé sur l'utilisation des ressources"""
        # Coûts unitaires (à ajuster selon votre modèle de tarification)
        cost_per_cpu_hour = 0.05
        cost_per_gb_hour = 0.01
        cost_per_1k_predictions = 0.01
        cost_per_gb_storage = 0.023
        
        # Calcul du coût
        cpu_hours = result.resources_used.get("cpu_hours", 0)
        memory_gb_hours = result.resources_used.get("memory_gb", 0) * (result.duration_seconds or 0) / 3600
        predictions = result.processed_records
        storage_gb = result.resources_used.get("storage_mb", 0) / 1024
        
        # Multiplicateur selon la priorité
        priority_multiplier = {
            BatchPriority.LOW: 0.8,
            BatchPriority.NORMAL: 1.0,
            BatchPriority.HIGH: 1.5,
            BatchPriority.CRITICAL: 2.0
        }
        
        base_cost = (
            cpu_hours * cost_per_cpu_hour +
            memory_gb_hours * cost_per_gb_hour +
            (predictions / 1000) * cost_per_1k_predictions +
            storage_gb * cost_per_gb_storage
        )
        
        total_cost = base_cost * priority_multiplier.get(job_config.priority, 1.0)
        
        return round(total_cost, 4)
    
    async def _send_notifications(self, job_config: BatchJobConfig, result: BatchJobResult):
        """Envoie les notifications configurées"""
        if not job_config.notification_config:
            return
        
        notification_data = {
            "job_id": result.job_id,
            "status": result.status,
            "tenant_id": result.tenant_id,
            "success_rate": result.success_rate,
            "total_records": result.total_records,
            "processed_records": result.processed_records,
            "duration_seconds": result.duration_seconds,
            "cost_estimate": result.cost_estimate,
            "output_location": result.output_location
        }
        
        # Email
        if "email" in job_config.notification_config:
            # Implémenter l'envoi d'email
            pass
        
        # Webhook
        if "webhook" in job_config.notification_config:
            import aiohttp
            try:
                async with aiohttp.ClientSession() as session:
                    await session.post(
                        job_config.notification_config["webhook"],
                        json=notification_data,
                        timeout=aiohttp.ClientTimeout(total=10)
                    )
            except:
                logger.error("Erreur envoi webhook")
        
        # Slack
        if "slack" in job_config.notification_config and self.config.monitoring.slack_webhook_url:
            import aiohttp
            message = {
                "text": f"Job Batch {result.job_id} terminé",
                "attachments": [{
                    "color": "good" if result.status == BatchStatus.COMPLETED else "danger",
                    "fields": [
                        {"title": "Statut", "value": result.status, "short": True},
                        {"title": "Taux de succès", "value": f"{result.success_rate:.2%}", "short": True},
                        {"title": "Records traités", "value": f"{result.processed_records}/{result.total_records}", "short": True},
                        {"title": "Durée", "value": f"{result.duration_seconds:.1f}s", "short": True}
                    ]
                }]
            }
            try:
                async with aiohttp.ClientSession() as session:
                    await session.post(
                        self.config.monitoring.slack_webhook_url,
                        json=message,
                        timeout=aiohttp.ClientTimeout(total=10)
                    )
            except:
                logger.error("Erreur envoi Slack")
    
    def _cache_job_status(self, job_id: str, result: BatchJobResult):
        """Met à jour le statut en cache Redis"""
        if self.redis_client:
            try:
                self.redis_client.setex(
                    f"batch_job:{job_id}",
                    3600 * 24,  # TTL de 24h
                    json.dumps(result.dict(), default=str)
                )
            except Exception as e:
                logger.error(f"Erreur cache Redis: {e}")
    
    async def get_job_status(self, job_id: str) -> Optional[BatchJobResult]:
        """Récupère le statut d'un job"""
        # Vérification en mémoire
        if job_id in self.active_jobs:
            return self.active_jobs[job_id]
        
        # Vérification en cache Redis
        if self.redis_client:
            try:
                cached = self.redis_client.get(f"batch_job:{job_id}")
                if cached:
                    data = json.loads(cached)
                    return BatchJobResult(**data)
            except Exception as e:
                logger.error(f"Erreur récupération cache: {e}")
        
        return None
    
    async def cancel_job(self, job_id: str) -> bool:
        """Annule un job en cours"""
        if job_id in self.active_jobs:
            job = self.active_jobs[job_id]
            if job.status in [BatchStatus.PENDING, BatchStatus.PROCESSING]:
                job.status = BatchStatus.CANCELLED
                job.end_time = datetime.now()
                self._cache_job_status(job_id, job)
                logger.info(f"Job {job_id} annulé")
                return True
        return False
    
    async def list_jobs(self, 
                       tenant_id: Optional[str] = None,
                       status: Optional[BatchStatus] = None,
                       limit: int = 100) -> List[BatchJobResult]:
        """Liste les jobs selon les critères"""
        jobs = list(self.active_jobs.values())
        
        # Filtrage par tenant
        if tenant_id:
            jobs = [j for j in jobs if j.tenant_id == tenant_id]
        
        # Filtrage par statut
        if status:
            jobs = [j for j in jobs if j.status == status]
        
        # Tri par date de début décroissante
        jobs.sort(key=lambda x: x.start_time, reverse=True)
        
        return jobs[:limit]
    
    async def cleanup_old_jobs(self, days: int = 7):
        """Nettoie les anciens jobs"""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        jobs_to_remove = []
        for job_id, job in self.active_jobs.items():
            if job.end_time and job.end_time < cutoff_date:
                jobs_to_remove.append(job_id)
        
        for job_id in jobs_to_remove:
            del self.active_jobs[job_id]
            if self.redis_client:
                try:
                    self.redis_client.delete(f"batch_job:{job_id}")
                except:
                    pass
        
        logger.info(f"Nettoyage de {len(jobs_to_remove)} anciens jobs")
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Récupère les métriques système"""
        active_count = len([j for j in self.active_jobs.values() 
                          if j.status == BatchStatus.PROCESSING])
        pending_count = len([j for j in self.active_jobs.values() 
                           if j.status == BatchStatus.PENDING])
        
        return {
            "active_jobs": active_count,
            "pending_jobs": pending_count,
            "total_jobs": len(self.active_jobs),
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage('/').percent,
            "ray_resources": ray.available_resources() if ray.is_initialized() else {}
        }
    
    def shutdown(self):
        """Arrêt propre du moteur"""
        logger.info("Arrêt du moteur de batch inference")
        
        # Arrêt de Ray
        if ray.is_initialized():
            ray.shutdown()
        
        # Fermeture de l'executor
        self.executor.shutdown(wait=True)
        
        # Fermeture Redis
        if self.redis_client:
            self.redis_client.close()


class BatchScheduler:
    """Planificateur de jobs batch avec support cron"""
    
    def __init__(self, engine: BatchInferenceEngine):
        self.engine = engine
        self.scheduled_jobs: Dict[str, BatchJobConfig] = {}
        self.scheduler_task = None
        self.running = False
    
    async def schedule_job(self, job_config: BatchJobConfig):
        """Planifie un job pour exécution future"""
        if job_config.scheduled_time and job_config.scheduled_time > datetime.now():
            self.scheduled_jobs[job_config.job_id] = job_config
            logger.info(f"Job {job_config.job_id} planifié pour {job_config.scheduled_time}")
        else:
            # Exécution immédiate
            await self.engine.submit_job(job_config)
    
    async def start_scheduler(self):
        """Démarre le planificateur"""
        if not self.scheduler_task:
            self.running = True
            self.scheduler_task = asyncio.create_task(self._scheduler_loop())
            logger.info("Planificateur batch démarré")
    
    async def stop_scheduler(self):
        """Arrête le planificateur"""
        self.running = False
        if self.scheduler_task:
            self.scheduler_task.cancel()
            try:
                await self.scheduler_task
            except asyncio.CancelledError:
                pass
            logger.info("Planificateur batch arrêté")
    
    async def _scheduler_loop(self):
        """Boucle principale du planificateur"""
        while self.running:
            try:
                now = datetime.now()
                jobs_to_run = []
                
                # Vérification des jobs planifiés
                for job_id, job_config in self.scheduled_jobs.items():
                    if job_config.scheduled_time <= now:
                        jobs_to_run.append(job_config)
                
                # Exécution des jobs
                for job_config in jobs_to_run:
                    del self.scheduled_jobs[job_config.job_id]
                    await self.engine.submit_job(job_config)
                    logger.info(f"Job planifié {job_config.job_id} lancé")
                
                # Attente avant prochaine vérification
                await asyncio.sleep(30)  # Vérification toutes les 30 secondes
                
            except Exception as e:
                logger.error(f"Erreur dans le planificateur: {e}")
                await asyncio.sleep(60)


# Export des classes principales
__all__ = [
    'BatchInferenceEngine',
    'BatchJobConfig',
    'BatchJobResult',
    'BatchStatus',
    'BatchPriority',
    'BatchScheduler',
    'DataLoader',
    'DataSaver',
    'BatchWorker'
]
