"""
MLOps Service - Sections modifiées pour intégrer les optimisations
==================================================================
Dans le fichier automl_platform/mlops_service.py, modifier les sections suivantes :
"""

# ============= SECTION 1: Imports (ajouter après les imports existants) =============

from .config import AutoMLConfig
from .monitoring import DriftDetector, ModelMonitor
from .storage import StorageService

# Import optimization components (AJOUTER CES LIGNES)
from .pipeline_cache import PipelineCache, CacheConfig, warm_cache, monitor_cache_health
from .incremental_learning import IncrementalLearner
from .distributed_training import DistributedTrainer

logger = logging.getLogger(__name__)


# ============= SECTION 2: Modifier la classe MLflowRegistry =============

class MLflowRegistry:
    """MLflow-based model registry and versioning with caching"""
    
    def __init__(self, config: AutoMLConfig, tracking_uri: Optional[str] = None):
        self.config = config
        
        # Initialize pipeline cache (AJOUTER CES LIGNES)
        self.pipeline_cache = None
        if hasattr(config, 'enable_cache') and config.enable_cache:
            cache_config = CacheConfig(
                backend=getattr(config, 'cache_backend', 'redis'),
                redis_host=getattr(config, 'redis_host', 'localhost'),
                ttl_seconds=getattr(config, 'cache_ttl', 3600)
            )
            self.pipeline_cache = PipelineCache(cache_config)
            logger.info("Pipeline cache enabled for model registry")
        
        # Reste du code __init__ existant...
        if MLFLOW_AVAILABLE:
            # Code existant...
    
    # Modifier la méthode get_production_model
    def get_production_model(self, model_name: str, use_cache: bool = True) -> Optional[Any]:
        """Get current production model with caching"""
        
        # Check cache first (AJOUTER CES LIGNES)
        if use_cache and self.pipeline_cache:
            cache_key = f"prod_model_{model_name}"
            cached_model = self.pipeline_cache.get_pipeline(cache_key)
            if cached_model:
                logger.debug(f"Production model {model_name} loaded from cache")
                return cached_model
        
        if not MLFLOW_AVAILABLE:
            return None
        
        try:
            # Code existant pour récupérer le modèle...
            versions = self.client.get_latest_versions(
                model_name, 
                stages=[ModelStage.PRODUCTION.value]
            )
            
            if not versions:
                return None
            
            latest_version = versions[0]
            model_uri = f"models:/{model_name}/{latest_version.version}"
            model = mlflow.pyfunc.load_model(model_uri)
            
            # Cache the model (AJOUTER CES LIGNES)
            if use_cache and self.pipeline_cache and model:
                self.pipeline_cache.set_pipeline(
                    f"prod_model_{model_name}",
                    model,
                    ttl=3600 * 24  # Cache for 24 hours
                )
                logger.debug(f"Production model {model_name} cached")
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to load production model: {e}")
            return None
    
    # Ajouter une nouvelle méthode
    def invalidate_model_cache(self, model_name: str):
        """Invalidate cached model"""
        if self.pipeline_cache:
            cache_key = f"prod_model_{model_name}"
            self.pipeline_cache.invalidate(cache_key, reason="model_updated")
            logger.info(f"Cache invalidated for model {model_name}")


# ============= SECTION 3: Modifier la classe RetrainingService =============

class RetrainingService:
    """Automated model retraining based on drift and performance with optimizations"""
    
    def __init__(self, config: AutoMLConfig, 
                 registry: MLflowRegistry,
                 monitor: ModelMonitor):
        self.config = config
        self.registry = registry
        self.monitor = monitor
        
        # Retraining thresholds
        self.drift_threshold = 0.5
        self.performance_degradation_threshold = 0.1
        self.min_data_points = 1000
        
        # Schedule configuration
        self.check_frequency = timedelta(days=1)
        self.last_check = datetime.utcnow()
        
        # Initialize optimization components (AJOUTER CES LIGNES)
        self.incremental_learner = None
        if hasattr(config, 'incremental_learning') and config.incremental_learning:
            self.incremental_learner = IncrementalLearner(
                max_memory_mb=getattr(config, 'max_memory_mb', 1000)
            )
        
        self.distributed_trainer = None
        if hasattr(config, 'distributed_training') and config.distributed_training:
            self.distributed_trainer = DistributedTrainer(
                backend=getattr(config, 'distributed_backend', 'ray'),
                n_workers=getattr(config, 'n_workers', 4)
            )
    
    async def retrain_model(self, model_name: str, 
                           X_train: pd.DataFrame, 
                           y_train: pd.Series,
                           use_incremental: bool = False,
                           use_distributed: bool = False) -> ModelVersion:
        """Retrain a model with new data using optimizations"""
        
        logger.info(f"Starting retraining for {model_name}")
        
        # Invalidate cache for this model (AJOUTER CES LIGNES)
        if self.registry.pipeline_cache:
            self.registry.invalidate_model_cache(model_name)
        
        # Get current production model configuration
        prod_versions = self.registry.client.get_latest_versions(
            model_name,
            stages=[ModelStage.PRODUCTION.value]
        )
        
        if not prod_versions:
            raise ValueError(f"No production model found for {model_name}")
        
        current_version = prod_versions[0]
        run = self.registry.client.get_run(current_version.run_id)
        params = run.data.params
        
        # Use incremental learning for large datasets (AJOUTER CES LIGNES)
        if use_incremental and self.incremental_learner and len(X_train) > 10000:
            logger.info("Using incremental retraining")
            task = detect_task(y_train)
            models = self.incremental_learner.train_incremental(X_train, y_train, task)
            best_model = self.incremental_learner.get_best_model()
        
        # Use distributed training if enabled (AJOUTER CES LIGNES)
        elif use_distributed and self.distributed_trainer:
            logger.info("Using distributed retraining")
            from .model_selection import get_available_models
            models = get_available_models(detect_task(y_train))
            results = self.distributed_trainer.train_distributed(
                X_train, y_train, models, {}
            )
            best_result = max(results, key=lambda x: x['cv_score'])
            best_model = best_result['pipeline']
        else:
            # Standard retraining with orchestrator
            from .orchestrator import AutoMLOrchestrator
            
            retrain_config = self.config
            retrain_config.algorithms = [params.get('algorithm', 'RandomForestClassifier')]
            retrain_config.hpo_n_iter = 20
            
            orchestrator = AutoMLOrchestrator(retrain_config)
            orchestrator.fit(X_train, y_train)
            best_model = orchestrator.best_pipeline
        
        # Get metrics
        from .metrics import calculate_metrics, detect_task
        y_pred = best_model.predict(X_train)
        metrics = calculate_metrics(y_train, y_pred, None, detect_task(y_train))
        
        # Register new version
        new_version = self.registry.register_model(
            model=best_model,
            model_name=model_name,
            metrics=metrics,
            params=params,
            X_sample=X_train.head(100),
            y_sample=y_train.head(100),
            description=f"Automated retraining - {datetime.utcnow()}",
            tags={
                "retrained": "true",
                "trigger": "automated",
                "incremental": str(use_incremental),
                "distributed": str(use_distributed)
            }
        )
        
        # Promote to staging for validation
        self.registry.promote_model(model_name, new_version.version, ModelStage.STAGING)
        
        logger.info(f"Retraining completed for {model_name}, new version: {new_version.version}")
        
        return new_version


# ============= SECTION 4: Modifier la classe ModelExporter =============

class ModelExporter:
    """Export models to various formats for deployment with caching"""
    
    def __init__(self, config: AutoMLConfig = None):
        self.config = config
        
        # Initialize cache for exported models (AJOUTER CES LIGNES)
        self.export_cache = None
        if config and hasattr(config, 'enable_cache') and config.enable_cache:
            cache_config = CacheConfig(
                backend='disk',  # Use disk for exported models
                disk_cache_dir='/tmp/exported_models_cache',
                ttl_seconds=3600 * 24 * 7,  # Cache for 1 week
                use_mmap=True
            )
            self.export_cache = PipelineCache(cache_config)
    
    def export_to_onnx(self, model: Any, 
                      sample_input: np.ndarray,
                      output_path: str,
                      use_cache: bool = True) -> bool:
        """Export model to ONNX format with caching"""
        
        # Check cache first (AJOUTER CES LIGNES)
        if use_cache and self.export_cache:
            cache_key = f"onnx_{hash(str(model))}_{sample_input.shape}"
            cached_export = self.export_cache.get_pipeline(cache_key)
            if cached_export:
                logger.info(f"Using cached ONNX export")
                with open(output_path, 'wb') as f:
                    f.write(cached_export)
                return True
        
        # Code existant pour l'export...
        if not ONNX_AVAILABLE:
            logger.error("ONNX not available")
            return False
        
        try:
            n_features = sample_input.shape[1]
            initial_type = [('float_input', FloatTensorType([None, n_features]))]
            onnx_model = convert_sklearn(model, initial_types=initial_type)
            
            onnx_bytes = onnx_model.SerializeToString()
            
            # Cache the export (AJOUTER CES LIGNES)
            if use_cache and self.export_cache:
                self.export_cache.set_pipeline(cache_key, onnx_bytes)
            
            with open(output_path, "wb") as f:
                f.write(onnx_bytes)
            
            logger.info(f"Model exported to ONNX: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export to ONNX: {e}")
            return False
