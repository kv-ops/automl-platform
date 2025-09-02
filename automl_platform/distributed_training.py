"""
Distributed Training Module for AutoML Platform
Supports Ray and Dask for distributed training across multiple nodes
Place in: automl_platform/distributed_training.py
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import time
from datetime import datetime
import os
import pickle
import json
from pathlib import Path

# Ray support
try:
    import ray
    from ray import tune
    from ray.tune.sklearn import TuneSearchCV, TuneGridSearchCV
    from ray.util.joblib import register_ray
    from ray.train.sklearn import SklearnTrainer
    from ray.air.config import ScalingConfig
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

# Dask support
try:
    import dask
    import dask.dataframe as dd
    import dask.array as da
    from dask.distributed import Client, as_completed, wait
    from dask_ml.model_selection import GridSearchCV as DaskGridSearchCV
    from dask_ml.model_selection import RandomizedSearchCV as DaskRandomizedSearchCV
    from dask_ml.preprocessing import StandardScaler as DaskStandardScaler
    from dask_ml.linear_model import LogisticRegression as DaskLogisticRegression
    from dask_ml.ensemble import RandomForestClassifier as DaskRandomForestClassifier
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False

# Spark support (optional)
try:
    from pyspark.sql import SparkSession
    from pyspark.ml import Pipeline as SparkPipeline
    from pyspark.ml.classification import RandomForestClassifier as SparkRFC
    from pyspark.ml.regression import RandomForestRegressor as SparkRFR
    from pyspark.ml.feature import VectorAssembler
    SPARK_AVAILABLE = True
except ImportError:
    SPARK_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class DistributedConfig:
    """Configuration for distributed training."""
    backend: str = "ray"  # ray, dask, spark
    num_workers: int = 4
    num_cpus_per_worker: int = 2
    num_gpus_per_worker: float = 0.0
    memory_per_worker_gb: int = 4
    
    # Ray specific
    ray_address: Optional[str] = None  # Ray cluster address
    ray_dashboard_port: int = 8265
    ray_object_store_memory: Optional[int] = None
    
    # Dask specific
    dask_scheduler_address: Optional[str] = None
    dask_dashboard_port: int = 8787
    dask_threads_per_worker: int = 1
    
    # Spark specific
    spark_master: str = "local[*]"
    spark_app_name: str = "AutoML_Distributed"
    spark_executor_memory: str = "4g"
    spark_executor_cores: int = 2
    
    # Training configuration
    chunk_size: int = 10000
    max_partitions: int = 100
    use_gpu: bool = False
    timeout_minutes: int = 60
    
    # Optimization
    enable_auto_scaling: bool = True
    min_workers: int = 1
    max_workers: int = 10
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}


class RayDistributedTrainer:
    """Ray-based distributed training."""
    
    def __init__(self, config: DistributedConfig):
        """Initialize Ray distributed trainer."""
        if not RAY_AVAILABLE:
            raise ImportError("Ray not installed. Install with: pip install ray[default]")
        
        self.config = config
        self.is_initialized = False
        
    def initialize(self):
        """Initialize Ray cluster."""
        if self.is_initialized:
            return
        
        try:
            # Initialize Ray
            if self.config.ray_address:
                # Connect to existing cluster
                ray.init(address=self.config.ray_address)
                logger.info(f"Connected to Ray cluster at {self.config.ray_address}")
            else:
                # Start local cluster
                ray.init(
                    num_cpus=self.config.num_workers * self.config.num_cpus_per_worker,
                    num_gpus=int(self.config.num_workers * self.config.num_gpus_per_worker),
                    object_store_memory=self.config.ray_object_store_memory,
                    dashboard_port=self.config.ray_dashboard_port,
                    ignore_reinit_error=True
                )
                logger.info("Started local Ray cluster")
            
            # Register Ray with joblib for sklearn
            register_ray()
            
            self.is_initialized = True
            
            # Log cluster info
            nodes = ray.nodes()
            logger.info(f"Ray cluster has {len(nodes)} nodes")
            for node in nodes:
                logger.info(f"Node: {node['NodeManagerAddress']}, "
                          f"CPUs: {node['Resources'].get('CPU', 0)}, "
                          f"GPUs: {node['Resources'].get('GPU', 0)}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Ray: {e}")
            raise
    
    def train_distributed(self, 
                         X: Union[pd.DataFrame, np.ndarray],
                         y: Union[pd.Series, np.ndarray],
                         models: Dict[str, Any],
                         param_grids: Dict[str, Dict],
                         cv: Any,
                         scoring: str = 'accuracy') -> Dict[str, Any]:
        """
        Train models in parallel using Ray.
        
        Args:
            X: Training features
            y: Training labels
            models: Dictionary of models to train
            param_grids: Hyperparameter grids for each model
            cv: Cross-validation splitter
            scoring: Scoring metric
            
        Returns:
            Dictionary with results for each model
        """
        self.initialize()
        
        # Put data in Ray object store for efficient sharing
        X_id = ray.put(X)
        y_id = ray.put(y)
        
        # Define remote training function
        @ray.remote(num_cpus=self.config.num_cpus_per_worker,
                   num_gpus=self.config.num_gpus_per_worker)
        def train_model(model_name: str, model: Any, param_grid: Dict,
                       X_ref: Any, y_ref: Any, cv_splitter: Any, metric: str):
            """Remote function to train a single model."""
            import time
            from sklearn.model_selection import GridSearchCV
            
            start_time = time.time()
            
            try:
                # Get data from object store
                X_train = ray.get(X_ref)
                y_train = ray.get(y_ref)
                
                # Perform grid search
                if param_grid:
                    search = GridSearchCV(
                        model,
                        param_grid,
                        cv=cv_splitter,
                        scoring=metric,
                        n_jobs=1,  # Use 1 job since Ray handles parallelism
                        verbose=0
                    )
                    search.fit(X_train, y_train)
                    best_model = search.best_estimator_
                    best_params = search.best_params_
                    best_score = search.best_score_
                else:
                    # No hyperparameter tuning
                    from sklearn.model_selection import cross_val_score
                    scores = cross_val_score(model, X_train, y_train, 
                                           cv=cv_splitter, scoring=metric)
                    best_model = model.fit(X_train, y_train)
                    best_params = {}
                    best_score = scores.mean()
                
                training_time = time.time() - start_time
                
                return {
                    'model_name': model_name,
                    'best_model': best_model,
                    'best_params': best_params,
                    'best_score': best_score,
                    'training_time': training_time,
                    'status': 'success'
                }
                
            except Exception as e:
                return {
                    'model_name': model_name,
                    'status': 'failed',
                    'error': str(e)
                }
        
        # Submit parallel training tasks
        futures = []
        for model_name, model in models.items():
            param_grid = param_grids.get(model_name, {})
            future = train_model.remote(
                model_name, model, param_grid,
                X_id, y_id, cv, scoring
            )
            futures.append(future)
        
        # Collect results
        results = {}
        for future in ray.get(futures):
            results[future['model_name']] = future
        
        return results
    
    def hyperparameter_tune(self,
                           model: Any,
                           X: Union[pd.DataFrame, np.ndarray],
                           y: Union[pd.Series, np.ndarray],
                           param_distributions: Dict,
                           n_trials: int = 100,
                           scoring: str = 'accuracy') -> Dict[str, Any]:
        """
        Hyperparameter tuning using Ray Tune.
        
        Args:
            model: Model to tune
            X: Training features
            y: Training labels
            param_distributions: Parameter distributions for tuning
            n_trials: Number of trials
            scoring: Scoring metric
            
        Returns:
            Best parameters and results
        """
        self.initialize()
        
        # Use Ray Tune for hyperparameter optimization
        tune_search = TuneSearchCV(
            model,
            param_distributions,
            n_trials=n_trials,
            scoring=scoring,
            cv=5,
            n_jobs=self.config.num_workers,
            verbose=1
        )
        
        tune_search.fit(X, y)
        
        return {
            'best_params': tune_search.best_params_,
            'best_score': tune_search.best_score_,
            'best_estimator': tune_search.best_estimator_,
            'cv_results': tune_search.cv_results_
        }
    
    def shutdown(self):
        """Shutdown Ray cluster."""
        if self.is_initialized:
            ray.shutdown()
            self.is_initialized = False
            logger.info("Ray cluster shut down")


class DaskDistributedTrainer:
    """Dask-based distributed training."""
    
    def __init__(self, config: DistributedConfig):
        """Initialize Dask distributed trainer."""
        if not DASK_AVAILABLE:
            raise ImportError("Dask not installed. Install with: pip install dask[complete] dask-ml")
        
        self.config = config
        self.client = None
        
    def initialize(self):
        """Initialize Dask cluster."""
        if self.client is not None:
            return
        
        try:
            if self.config.dask_scheduler_address:
                # Connect to existing cluster
                self.client = Client(self.config.dask_scheduler_address)
                logger.info(f"Connected to Dask cluster at {self.config.dask_scheduler_address}")
            else:
                # Start local cluster
                from dask.distributed import LocalCluster
                
                cluster = LocalCluster(
                    n_workers=self.config.num_workers,
                    threads_per_worker=self.config.dask_threads_per_worker,
                    processes=True,
                    memory_limit=f"{self.config.memory_per_worker_gb}GB",
                    dashboard_address=f":{self.config.dask_dashboard_port}"
                )
                self.client = Client(cluster)
                logger.info("Started local Dask cluster")
            
            # Log cluster info
            logger.info(f"Dask dashboard: {self.client.dashboard_link}")
            logger.info(f"Workers: {len(self.client.scheduler_info()['workers'])}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Dask: {e}")
            raise
    
    def train_distributed(self,
                         X: Union[pd.DataFrame, np.ndarray],
                         y: Union[pd.Series, np.ndarray],
                         models: Dict[str, Any],
                         param_grids: Dict[str, Dict],
                         cv: Any,
                         scoring: str = 'accuracy') -> Dict[str, Any]:
        """
        Train models in parallel using Dask.
        
        Args:
            X: Training features
            y: Training labels
            models: Dictionary of models to train
            param_grids: Hyperparameter grids for each model
            cv: Cross-validation splitter
            scoring: Scoring metric
            
        Returns:
            Dictionary with results for each model
        """
        self.initialize()
        
        # Convert to Dask arrays if needed
        if not isinstance(X, da.Array):
            X_da = da.from_array(X, chunks=(self.config.chunk_size, -1))
        else:
            X_da = X
        
        if not isinstance(y, da.Array):
            y_da = da.from_array(y, chunks=self.config.chunk_size)
        else:
            y_da = y
        
        # Train models in parallel
        futures = {}
        
        for model_name, model in models.items():
            param_grid = param_grids.get(model_name, {})
            
            # Submit training task
            future = self.client.submit(
                self._train_single_model,
                model_name, model, param_grid,
                X_da, y_da, cv, scoring,
                pure=False
            )
            futures[model_name] = future
        
        # Gather results
        results = {}
        for model_name, future in futures.items():
            try:
                result = future.result(timeout=self.config.timeout_minutes * 60)
                results[model_name] = result
            except Exception as e:
                results[model_name] = {
                    'model_name': model_name,
                    'status': 'failed',
                    'error': str(e)
                }
        
        return results
    
    def _train_single_model(self, model_name: str, model: Any, param_grid: Dict,
                          X: da.Array, y: da.Array, cv: Any, scoring: str) -> Dict:
        """Train a single model (runs on worker)."""
        import time
        
        start_time = time.time()
        
        try:
            # Use Dask-ML for grid search if available
            if param_grid:
                search = DaskGridSearchCV(
                    model,
                    param_grid,
                    cv=cv,
                    scoring=scoring,
                    n_jobs=-1
                )
                
                # Compute arrays before fitting
                X_computed = X.compute()
                y_computed = y.compute()
                
                search.fit(X_computed, y_computed)
                best_model = search.best_estimator_
                best_params = search.best_params_
                best_score = search.best_score_
            else:
                # No hyperparameter tuning
                X_computed = X.compute()
                y_computed = y.compute()
                
                from sklearn.model_selection import cross_val_score
                scores = cross_val_score(model, X_computed, y_computed,
                                       cv=cv, scoring=scoring)
                best_model = model.fit(X_computed, y_computed)
                best_params = {}
                best_score = scores.mean()
            
            training_time = time.time() - start_time
            
            return {
                'model_name': model_name,
                'best_model': best_model,
                'best_params': best_params,
                'best_score': best_score,
                'training_time': training_time,
                'status': 'success'
            }
            
        except Exception as e:
            return {
                'model_name': model_name,
                'status': 'failed',
                'error': str(e)
            }
    
    def process_large_dataset(self,
                            file_path: str,
                            target_column: str,
                            chunk_size: int = None) -> Tuple[dd.DataFrame, dd.Series]:
        """
        Process large dataset using Dask DataFrame.
        
        Args:
            file_path: Path to dataset file
            target_column: Name of target column
            chunk_size: Size of chunks for processing
            
        Returns:
            Dask DataFrame and Series for features and target
        """
        chunk_size = chunk_size or self.config.chunk_size
        
        # Read data in chunks
        if file_path.endswith('.csv'):
            df = dd.read_csv(file_path, blocksize=f"{chunk_size}KB")
        elif file_path.endswith('.parquet'):
            df = dd.read_parquet(file_path, engine='pyarrow')
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
        
        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Persist in memory for faster access
        X = X.persist()
        y = y.persist()
        
        logger.info(f"Loaded dataset with {len(X.columns)} features")
        logger.info(f"Partitions: {X.npartitions}")
        
        return X, y
    
    def shutdown(self):
        """Shutdown Dask cluster."""
        if self.client:
            self.client.close()
            self.client = None
            logger.info("Dask cluster shut down")


class SparkDistributedTrainer:
    """Spark-based distributed training (optional)."""
    
    def __init__(self, config: DistributedConfig):
        """Initialize Spark distributed trainer."""
        if not SPARK_AVAILABLE:
            raise ImportError("PySpark not installed. Install with: pip install pyspark")
        
        self.config = config
        self.spark = None
        
    def initialize(self):
        """Initialize Spark session."""
        if self.spark is not None:
            return
        
        try:
            self.spark = SparkSession.builder \
                .appName(self.config.spark_app_name) \
                .master(self.config.spark_master) \
                .config("spark.executor.memory", self.config.spark_executor_memory) \
                .config("spark.executor.cores", self.config.spark_executor_cores) \
                .config("spark.sql.adaptive.enabled", "true") \
                .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
                .getOrCreate()
            
            logger.info(f"Started Spark session: {self.spark.sparkContext.applicationId}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Spark: {e}")
            raise
    
    def train_distributed(self,
                         df_path: str,
                         target_column: str,
                         task: str = 'classification') -> Any:
        """
        Train model using Spark ML.
        
        Args:
            df_path: Path to dataset
            target_column: Target column name
            task: Task type (classification/regression)
            
        Returns:
            Trained Spark ML model
        """
        self.initialize()
        
        # Read data
        df = self.spark.read.parquet(df_path) if df_path.endswith('.parquet') else \
             self.spark.read.csv(df_path, header=True, inferSchema=True)
        
        # Prepare features
        feature_columns = [col for col in df.columns if col != target_column]
        assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
        
        # Create model
        if task == 'classification':
            model = SparkRFC(featuresCol="features", labelCol=target_column,
                           numTrees=100, maxDepth=10)
        else:
            model = SparkRFR(featuresCol="features", labelCol=target_column,
                           numTrees=100, maxDepth=10)
        
        # Create pipeline
        pipeline = SparkPipeline(stages=[assembler, model])
        
        # Train model
        model_fitted = pipeline.fit(df)
        
        return model_fitted
    
    def shutdown(self):
        """Shutdown Spark session."""
        if self.spark:
            self.spark.stop()
            self.spark = None
            logger.info("Spark session stopped")


class DistributedTrainingOrchestrator:
    """Orchestrator for distributed training across different backends."""
    
    def __init__(self, config: DistributedConfig):
        """Initialize distributed training orchestrator."""
        self.config = config
        self.trainer = None
        
        # Select backend
        if config.backend == "ray":
            self.trainer = RayDistributedTrainer(config)
        elif config.backend == "dask":
            self.trainer = DaskDistributedTrainer(config)
        elif config.backend == "spark":
            self.trainer = SparkDistributedTrainer(config)
        else:
            raise ValueError(f"Unsupported backend: {config.backend}")
        
        logger.info(f"Initialized distributed training with {config.backend}")
    
    def train(self,
             X: Union[pd.DataFrame, np.ndarray],
             y: Union[pd.Series, np.ndarray],
             models: Dict[str, Any],
             param_grids: Dict[str, Dict] = None,
             cv: Any = None,
             scoring: str = 'accuracy') -> Dict[str, Any]:
        """
        Train models using distributed computing.
        
        Args:
            X: Training features
            y: Training labels
            models: Dictionary of models to train
            param_grids: Hyperparameter grids
            cv: Cross-validation splitter
            scoring: Scoring metric
            
        Returns:
            Training results
        """
        param_grids = param_grids or {}
        
        if cv is None:
            from sklearn.model_selection import KFold
            cv = KFold(n_splits=5, shuffle=True, random_state=42)
        
        # Initialize trainer
        self.trainer.initialize()
        
        # Train models
        results = self.trainer.train_distributed(
            X, y, models, param_grids, cv, scoring
        )
        
        # Log results
        for model_name, result in results.items():
            if result['status'] == 'success':
                logger.info(f"{model_name}: Score={result['best_score']:.4f}, "
                          f"Time={result['training_time']:.2f}s")
            else:
                logger.error(f"{model_name}: Failed - {result.get('error', 'Unknown error')}")
        
        return results
    
    def optimize_resource_allocation(self,
                                   dataset_size: int,
                                   n_features: int,
                                   n_models: int) -> DistributedConfig:
        """
        Optimize resource allocation based on workload.
        
        Args:
            dataset_size: Number of samples
            n_features: Number of features
            n_models: Number of models to train
            
        Returns:
            Optimized configuration
        """
        # Estimate memory requirements (MB)
        estimated_memory = (dataset_size * n_features * 8) / (1024 * 1024)
        
        # Determine optimal workers
        if dataset_size < 10000:
            optimal_workers = 2
        elif dataset_size < 100000:
            optimal_workers = 4
        elif dataset_size < 1000000:
            optimal_workers = 8
        else:
            optimal_workers = 16
        
        # Adjust for available resources
        import multiprocessing
        max_workers = min(optimal_workers, multiprocessing.cpu_count())
        
        # Calculate memory per worker
        memory_per_worker = max(2, int(estimated_memory / max_workers) + 1)
        
        # Update configuration
        self.config.num_workers = max_workers
        self.config.memory_per_worker_gb = memory_per_worker
        
        # Adjust chunk size based on dataset
        if dataset_size > 1000000:
            self.config.chunk_size = 50000
        elif dataset_size > 100000:
            self.config.chunk_size = 10000
        else:
            self.config.chunk_size = 1000
        
        logger.info(f"Optimized configuration: workers={max_workers}, "
                   f"memory_per_worker={memory_per_worker}GB, "
                   f"chunk_size={self.config.chunk_size}")
        
        return self.config
    
    def shutdown(self):
        """Shutdown distributed training."""
        if self.trainer:
            self.trainer.shutdown()


# Example usage
def main():
    """Example of distributed training."""
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import KFold
    
    # Generate sample data
    X, y = make_classification(n_samples=10000, n_features=20, 
                              n_classes=2, random_state=42)
    
    # Configure distributed training
    config = DistributedConfig(
        backend="ray",  # or "dask"
        num_workers=4,
        num_cpus_per_worker=2,
        memory_per_worker_gb=2
    )
    
    # Create orchestrator
    orchestrator = DistributedTrainingOrchestrator(config)
    
    # Define models
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=100),
        'LogisticRegression': LogisticRegression(max_iter=1000)
    }
    
    # Define parameter grids
    param_grids = {
        'RandomForest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, None]
        },
        'LogisticRegression': {
            'C': [0.1, 1.0, 10.0]
        }
    }
    
    # Train models
    results = orchestrator.train(
        X, y,
        models=models,
        param_grids=param_grids,
        cv=KFold(n_splits=5),
        scoring='accuracy'
    )
    
    # Print results
    for model_name, result in results.items():
        if result['status'] == 'success':
            print(f"{model_name}:")
            print(f"  Best Score: {result['best_score']:.4f}")
            print(f"  Best Params: {result['best_params']}")
            print(f"  Training Time: {result['training_time']:.2f}s")
    
    # Shutdown
    orchestrator.shutdown()


if __name__ == "__main__":
    main()
