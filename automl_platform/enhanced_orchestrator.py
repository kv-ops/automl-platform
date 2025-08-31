"""
Enhanced AutoML Orchestrator with Storage, Monitoring, and LLM Integration
Following DataRobot, Akkio, and H2O.ai best practices
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import get_scorer
import time
import json
from pathlib import Path
import joblib
import logging
from datetime import datetime
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
import warnings
warnings.filterwarnings('ignore')

from .data_prep import DataPreprocessor, handle_imbalance, validate_data
from .model_selection import (
    get_available_models, get_param_grid, get_cv_splitter,
    tune_model, try_optuna
)
from .metrics import calculate_metrics, detect_task
from .config import AutoMLConfig, load_config
from .storage import StorageManager, ModelMetadata, FeatureStore
from .monitoring import (
    MonitoringService, ModelMonitor, DataQualityMonitor,
    AlertManager, DriftDetector
)

# LLM Integration imports
from .llm import AutoMLLLMAssistant
from .data_quality_agent import IntelligentDataQualityAgent
from .prompts import PromptTemplates

logger = logging.getLogger(__name__)


class EnhancedAutoMLOrchestrator:
    """
    Enhanced AutoML orchestrator with production features and LLM integration.
    Combines approaches from DataRobot, Akkio, and H2O.ai.
    """
    
    def __init__(self, config: AutoMLConfig = None):
        """Initialize enhanced orchestrator with all components."""
        self.config = config or load_config()
        
        # Initialize storage if enabled
        if self.config.storage.backend != "none":
            self.storage_manager = StorageManager(
                backend=self.config.storage.backend,
                endpoint=self.config.storage.endpoint,
                access_key=self.config.storage.access_key,
                secret_key=self.config.storage.secret_key,
                secure=self.config.storage.secure,
                region=self.config.storage.region
            )
            self.feature_store = FeatureStore(self.storage_manager)
        else:
            self.storage_manager = None
            self.feature_store = None
        
        # Initialize monitoring if enabled
        if self.config.monitoring.enabled:
            self.monitoring_service = MonitoringService(self.storage_manager)
            self.quality_monitor = DataQualityMonitor()
            self.alert_manager = AlertManager(self._get_alert_config())
        else:
            self.monitoring_service = None
            self.quality_monitor = None
            self.alert_manager = None
        
        # Initialize LLM components if enabled
        if self.config.llm.enabled:
            try:
                self.llm_assistant = AutoMLLLMAssistant(self.config.llm.__dict__)
                self.quality_agent = IntelligentDataQualityAgent(self.llm_assistant.llm)
                logger.info("LLM components initialized successfully")
            except Exception as e:
                logger.warning(f"LLM initialization failed: {e}. Continuing without LLM features.")
                self.llm_assistant = None
                self.quality_agent = None
        else:
            self.llm_assistant = None
            self.quality_agent = None
        
        # Core components
        self.preprocessor = DataPreprocessor(self.config.to_dict())
        self.leaderboard = []
        self.best_pipeline = None
        self.task = None
        self.feature_importance = {}
        
        # Tracking
        self.experiment_id = None
        self.model_registry = {}
        self.training_history = []
        self.dataset_hash = None
        
        # Performance tracking
        self.start_time = None
        self.end_time = None
        self.total_models_trained = 0
        
        # LLM-generated insights
        self.llm_insights = {}
        self.feature_suggestions = []
        self.cleaning_report = None
        
        logger.info(f"Enhanced orchestrator initialized with {self.config.environment} environment")
    
    def _get_alert_config(self) -> Dict:
        """Get alert configuration from monitoring config."""
        return {
            "accuracy_threshold": self.config.monitoring.accuracy_alert_threshold,
            "drift_threshold": self.config.monitoring.drift_alert_threshold,
            "error_rate_threshold": self.config.monitoring.error_rate_threshold,
            "latency_threshold": self.config.monitoring.latency_threshold,
            "quality_score_threshold": self.config.monitoring.quality_score_threshold
        }
    
    async def analyze_with_llm(self, df: pd.DataFrame, target_column: str = None):
        """
        Akkio-style conversational data analysis using LLM.
        
        Args:
            df: DataFrame to analyze
            target_column: Target column for supervised learning
        
        Returns:
            Dict with analysis results and recommendations
        """
        if not self.quality_agent:
            logger.warning("LLM not configured, skipping intelligent analysis")
            return {}
        
        # DataRobot-style quality assessment
        assessment = self.quality_agent.assess(df, target_column)
        
        # Generate quality report
        quality_report = self.quality_agent.get_quality_report(assessment)
        
        # Store results
        self.llm_insights['quality_assessment'] = assessment
        self.llm_insights['quality_report'] = quality_report
        
        logger.info(f"LLM Analysis Complete - Quality Score: {assessment.quality_score:.1f}/100")
        
        return {
            'quality_score': assessment.quality_score,
            'alerts': assessment.alerts,
            'warnings': assessment.warnings,
            'recommendations': assessment.recommendations,
            'report': quality_report
        }
    
    async def suggest_features_with_llm(self, df: pd.DataFrame, target: str) -> List[Dict]:
        """
        Get LLM-powered feature engineering suggestions.
        Similar to DataRobot's Feature Discovery.
        
        Args:
            df: Input dataframe
            target: Target column name
        
        Returns:
            List of feature suggestions with code
        """
        if not self.llm_assistant:
            logger.warning("LLM not configured, skipping feature suggestions")
            return []
        
        # Get feature suggestions
        suggestions = await self.llm_assistant.suggest_features(
            df, target, self.task or "auto"
        )
        
        # Store suggestions
        self.feature_suggestions = suggestions
        
        # Log top suggestions
        for i, suggestion in enumerate(suggestions[:3], 1):
            logger.info(f"Feature Suggestion {i}: {suggestion['name']} (importance: {suggestion['importance']})")
        
        return suggestions
    
    async def clean_data_with_llm(self, df: pd.DataFrame, instructions: str = None) -> pd.DataFrame:
        """
        Akkio-style conversational data cleaning.
        
        Args:
            df: DataFrame to clean
            instructions: Natural language cleaning instructions
        
        Returns:
            Cleaned DataFrame
        """
        if not self.quality_agent:
            logger.warning("LLM not configured, using standard cleaning")
            return df
        
        if instructions:
            # User-provided instructions
            cleaned_df, response = await self.quality_agent.clean(instructions, df)
            logger.info(f"LLM Cleaning Response: {response[:200]}...")
        else:
            # Auto-clean based on assessment
            assessment = self.quality_agent.assess(df)
            
            # Apply recommended cleaning for critical issues
            cleaned_df = df.copy()
            for alert in assessment.alerts:
                if alert.get('severity') == 'critical':
                    # Generate and apply cleaning code
                    cleaning_prompt = f"Fix this issue: {alert['message']}"
                    cleaned_df, _ = await self.quality_agent.clean(cleaning_prompt, cleaned_df)
        
        # Store cleaning report
        self.cleaning_report = {
            'original_shape': df.shape,
            'cleaned_shape': cleaned_df.shape,
            'actions_taken': len(assessment.alerts) if not instructions else 1
        }
        
        return cleaned_df
    
    def fit(self, 
            X: pd.DataFrame, 
            y: pd.Series,
            task: Optional[str] = None,
            experiment_name: Optional[str] = None,
            reference_data: Optional[pd.DataFrame] = None,
            use_llm_features: bool = None,
            use_llm_cleaning: bool = None) -> 'EnhancedAutoMLOrchestrator':
        """
        Run enhanced AutoML pipeline with LLM integration.
        
        Args:
            X: Training features
            y: Target variable
            task: Task type (auto-detected if None)
            experiment_name: Name for this experiment
            reference_data: Reference data for drift detection
            use_llm_features: Whether to use LLM for feature engineering
            use_llm_cleaning: Whether to use LLM for data cleaning
        
        Returns:
            Self for chaining
        """
        self.start_time = time.time()
        self.experiment_id = experiment_name or f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"Starting enhanced AutoML pipeline: {self.experiment_id}")
        
        # Use config defaults if not specified
        if use_llm_features is None:
            use_llm_features = self.config.llm.enable_feature_suggestions if self.config.llm.enabled else False
        if use_llm_cleaning is None:
            use_llm_cleaning = self.config.llm.enable_data_cleaning if self.config.llm.enabled else False
        
        # LLM-powered data quality analysis (async)
        if self.quality_agent:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If loop is already running (e.g., in Jupyter), create task
                asyncio.create_task(self.analyze_with_llm(X, y.name))
            else:
                # Otherwise run synchronously
                loop.run_until_complete(self.analyze_with_llm(X, y.name))
        
        # Data quality check (standard + LLM if available)
        if self.quality_monitor:
            quality_report = self.quality_monitor.check_data_quality(X)
            logger.info(f"Data quality score: {quality_report['quality_score']:.1f}")
            
            if quality_report['quality_score'] < self.config.monitoring.min_quality_score:
                logger.warning(f"Low data quality detected: {quality_report['issues']}")
                
                # LLM-powered cleaning if enabled
                if use_llm_cleaning and self.quality_agent:
                    logger.info("Applying LLM-powered data cleaning...")
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        X = asyncio.create_task(self.clean_data_with_llm(X))
                    else:
                        X = loop.run_until_complete(self.clean_data_with_llm(X))
                
                # Trigger alert
                if self.alert_manager:
                    self.alert_manager.check_alerts({
                        'quality_score': quality_report['quality_score']
                    })
        
        # Calculate dataset hash for versioning
        data_str = pd.concat([X, y], axis=1).to_csv(index=False)
        self.dataset_hash = hashlib.sha256(data_str.encode()).hexdigest()[:16]
        
        # Save dataset to storage
        if self.storage_manager:
            try:
                dataset_path = self.storage_manager.save_dataset(
                    pd.concat([X, y.rename('target')], axis=1),
                    f"{self.experiment_id}_dataset",
                    tenant_id=self.config.tenant_id
                )
                logger.info(f"Dataset saved: {dataset_path}")
            except Exception as e:
                logger.error(f"Failed to save dataset: {e}")
        
        # Validate data
        validation = validate_data(X)
        if not validation['valid']:
            logger.warning(f"Data quality issues: {validation['issues']}")
        
        # Detect task
        if task is None or task == 'auto':
            self.task = detect_task(y)
        else:
            self.task = task
        
        logger.info(f"Task detected: {self.task}")
        
        # LLM-powered feature engineering
        if use_llm_features and self.llm_assistant:
            logger.info("Generating LLM-powered feature suggestions...")
            loop = asyncio.get_event_loop()
            if loop.is_running():
                suggestions = asyncio.create_task(self.suggest_features_with_llm(X, y.name))
            else:
                suggestions = loop.run_until_complete(self.suggest_features_with_llm(X, y.name))
            
            # Apply top suggestions automatically
            if suggestions and self.config.llm.enable_feature_suggestions:
                X = self._apply_feature_suggestions(X, suggestions[:5])
        
        # Standard feature engineering with caching
        if self.config.enable_auto_feature_engineering and self.feature_store:
            X = self._engineer_features_with_cache(X, y)
        
        # Get available models
        models = self._get_models_to_train()
        logger.info(f"Testing {len(models)} models")
        
        # Setup reference data for drift detection
        if self.monitoring_service and reference_data is None:
            reference_data = X.copy()
        
        # Get CV splitter and scoring
        cv = get_cv_splitter(self.task, self.config.cv_folds, self.config.random_state)
        scoring = self._determine_scoring()
        
        # Train models with parallel processing if enabled
        if self.config.worker.enabled and len(models) > 1:
            self._train_models_parallel(models, X, y, cv, scoring, reference_data)
        else:
            self._train_models_sequential(models, X, y, cv, scoring, reference_data)
        
        # Sort leaderboard
        self.leaderboard.sort(key=lambda x: x['cv_score'], reverse=True)
        
        # Select and save best model
        if self.leaderboard:
            self._select_best_model(X, y, reference_data)
            self._save_experiment_results()
            
            # Generate LLM explanation of best model
            if self.llm_assistant:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(self._generate_model_explanation())
                else:
                    loop.run_until_complete(self._generate_model_explanation())
        
        self.end_time = time.time()
        training_time = self.end_time - self.start_time
        
        logger.info(f"Training completed in {training_time:.2f} seconds")
        logger.info(f"Total models trained: {self.total_models_trained}")
        
        # Generate monitoring report
        if self.monitoring_service:
            self._generate_monitoring_report()
        
        # Generate LLM report if enabled
        if self.llm_assistant and self.config.llm.enable_report_generation:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(self._generate_llm_report())
            else:
                loop.run_until_complete(self._generate_llm_report())
        
        return self
    
    def _apply_feature_suggestions(self, X: pd.DataFrame, suggestions: List[Dict]) -> pd.DataFrame:
        """Apply LLM-suggested features to dataframe."""
        X_enhanced = X.copy()
        
        for suggestion in suggestions:
            try:
                # Execute feature engineering code
                local_vars = {"df": X_enhanced, "pd": pd, "np": np}
                exec(suggestion['code'], {}, local_vars)
                X_enhanced = local_vars.get("df", X_enhanced)
                logger.info(f"Applied feature: {suggestion['name']}")
            except Exception as e:
                logger.warning(f"Failed to apply feature {suggestion['name']}: {e}")
        
        return X_enhanced
    
    async def _generate_model_explanation(self):
        """Generate LLM explanation of the best model."""
        if not self.llm_assistant or not self.leaderboard:
            return
        
        best_model = self.leaderboard[0]
        
        explanation = await self.llm_assistant.explain_model(
            model_name=best_model['model'],
            metrics=best_model['metrics'],
            feature_importance=self.feature_importance
        )
        
        self.llm_insights['model_explanation'] = explanation
        logger.info("Model explanation generated via LLM")
    
    async def _generate_llm_report(self):
        """Generate comprehensive report using LLM."""
        if not self.llm_assistant:
            return
        
        experiment_data = {
            'experiment_id': self.experiment_id,
            'best_model': self.leaderboard[0]['model'] if self.leaderboard else None,
            'metrics': self.leaderboard[0]['metrics'] if self.leaderboard else {},
            'top_features': list(self.feature_importance.keys())[:10] if self.feature_importance else [],
            'training_time': self.end_time - self.start_time if self.end_time else None,
            'models_trained': self.total_models_trained
        }
        
        report = await self.llm_assistant.generate_report(
            experiment_data,
            format="markdown"
        )
        
        # Save report
        if self.config.output_dir:
            report_path = Path(self.config.output_dir) / self.experiment_id / "llm_report.md"
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(report)
            logger.info(f"LLM report saved to {report_path}")
        
        self.llm_insights['report'] = report
    
    async def chat_with_assistant(self, message: str, context: Dict[str, Any] = None) -> str:
        """
        Chat with LLM assistant about the experiment.
        Akkio-style conversational interface.
        
        Args:
            message: User message
            context: Additional context
        
        Returns:
            Assistant response
        """
        if not self.llm_assistant:
            return "LLM assistant not configured. Please enable LLM in configuration."
        
        # Add experiment context
        if context is None:
            context = {}
        
        context.update({
            'experiment_id': self.experiment_id,
            'task': self.task,
            'models_trained': self.total_models_trained,
            'best_model': self.leaderboard[0] if self.leaderboard else None,
            'feature_suggestions': self.feature_suggestions[:3] if self.feature_suggestions else [],
            'quality_score': self.llm_insights.get('quality_assessment', {}).get('quality_score')
        })
        
        response = await self.llm_assistant.chat(message, context)
        
        return response
    
    def _get_models_to_train(self) -> Dict[str, Any]:
        """Get models to train based on configuration."""
        if self.config.algorithms == ['all']:
            models = get_available_models(self.task)
        else:
            all_models = get_available_models(self.task)
            models = {k: v for k, v in all_models.items() 
                     if k in self.config.algorithms}
        
        # Filter excluded models
        for excluded in self.config.exclude_algorithms:
            models.pop(excluded, None)
        
        # Add advanced models if enabled
        if self.config.include_neural_networks:
            models.update(self._get_neural_models())
        
        if self.config.include_time_series and self.task == 'timeseries':
            models.update(self._get_timeseries_models())
        
        # Limit number of models if specified
        if self.config.max_models_to_train and len(models) > self.config.max_models_to_train:
            # Prioritize certain models
            priority_models = ['XGBClassifier', 'XGBRegressor', 'LGBMClassifier', 
                             'LGBMRegressor', 'RandomForestClassifier', 'RandomForestRegressor']
            
            selected_models = {}
            for name in priority_models:
                if name in models:
                    selected_models[name] = models[name]
            
            # Add remaining models up to limit
            for name, model in models.items():
                if len(selected_models) >= self.config.max_models_to_train:
                    break
                if name not in selected_models:
                    selected_models[name] = model
            
            models = selected_models
        
        return models
    
    def _get_neural_models(self) -> Dict[str, Any]:
        """Get neural network models if available."""
        models = {}
        
        try:
            # TabNet
            if self.task == 'classification':
                from pytorch_tabnet.tab_model import TabNetClassifier
                models['TabNetClassifier'] = TabNetClassifier(
                    n_d=8, n_a=8, n_steps=3,
                    gamma=1.3, n_independent=2, n_shared=2,
                    seed=self.config.random_state,
                    verbose=0
                )
            else:
                from pytorch_tabnet.tab_model import TabNetRegressor
                models['TabNetRegressor'] = TabNetRegressor(
                    n_d=8, n_a=8, n_steps=3,
                    gamma=1.3, n_independent=2, n_shared=2,
                    seed=self.config.random_state,
                    verbose=0
                )
        except ImportError:
            logger.debug("TabNet not available")
        
        return models
    
    def _get_timeseries_models(self) -> Dict[str, Any]:
        """Get time series models if available."""
        models = {}
        
        try:
            from prophet import Prophet
            models['Prophet'] = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False
            )
        except ImportError:
            logger.debug("Prophet not available")
        
        try:
            from statsmodels.tsa.arima.model import ARIMA
            models['ARIMA'] = ARIMA
        except ImportError:
            logger.debug("ARIMA not available")
        
        return models
    
    def _determine_scoring(self) -> str:
        """Determine scoring metric based on task."""
        if self.config.scoring == 'auto':
            if self.task == 'classification':
                if len(np.unique(self.y_train)) == 2:
                    return 'roc_auc'
                else:
                    return 'f1_weighted'
            else:
                return 'neg_mean_squared_error'
        return self.config.scoring
    
    def _engineer_features_with_cache(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Engineer features with caching."""
        feature_hash = hashlib.sha256(X.to_csv().encode()).hexdigest()[:16]
        feature_set_name = f"features_{self.experiment_id}_{feature_hash}"
        
        # Try to load from cache
        try:
            logger.info("Checking feature store cache...")
            return self.feature_store.load_features(feature_set_name)
        except:
            logger.info("Generating new features...")
            
            # Generate features (simplified example)
            X_engineered = X.copy()
            
            # Polynomial features for numerical columns
            if self.config.create_polynomial:
                from sklearn.preprocessing import PolynomialFeatures
                numeric_cols = X.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    poly = PolynomialFeatures(degree=self.config.polynomial_degree, include_bias=False)
                    poly_features = poly.fit_transform(X[numeric_cols])
                    poly_df = pd.DataFrame(
                        poly_features[:, len(numeric_cols):],
                        columns=[f"poly_{i}" for i in range(poly_features.shape[1] - len(numeric_cols))],
                        index=X.index
                    )
                    X_engineered = pd.concat([X_engineered, poly_df], axis=1)
            
            # Save to cache
            if self.feature_store:
                self.feature_store.save_features(
                    X_engineered,
                    feature_set_name,
                    metadata={'original_shape': list(X.shape)}
                )
            
            return X_engineered
    
    def _train_models_sequential(self, models: Dict, X: pd.DataFrame, y: pd.Series,
                                cv: Any, scoring: str, reference_data: pd.DataFrame = None):
        """Train models sequentially."""
        for model_name, base_model in models.items():
            self._train_single_model(
                model_name, base_model, X, y, cv, scoring, reference_data
            )
    
    def _train_models_parallel(self, models: Dict, X: pd.DataFrame, y: pd.Series,
                              cv: Any, scoring: str, reference_data: pd.DataFrame = None):
        """Train models in parallel."""
        with ThreadPoolExecutor(max_workers=self.config.worker.max_workers) as executor:
            futures = {
                executor.submit(
                    self._train_single_model,
                    model_name, base_model, X, y, cv, scoring, reference_data
                ): model_name
                for model_name, base_model in models.items()
            }
            
            for future in as_completed(futures):
                model_name = futures[future]
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Failed to train {model_name}: {e}")
    
    def _train_single_model(self, model_name: str, base_model: Any,
                          X: pd.DataFrame, y: pd.Series,
                          cv: Any, scoring: str,
                          reference_data: pd.DataFrame = None):
        """Train a single model with monitoring."""
        logger.info(f"Training {model_name}")
        start_time = time.time()
        
        try:
            # Create pipeline
            pipeline = Pipeline([
                ('preprocessor', DataPreprocessor(self.config.to_dict())),
                ('model', base_model)
            ])
            
            # Handle imbalance
            if self.task == 'classification' and self.config.handle_imbalance:
                if hasattr(base_model, 'class_weight'):
                    base_model.set_params(class_weight='balanced')
            
            # Hyperparameter optimization
            params = self._optimize_hyperparameters(
                model_name, base_model, X, y, cv, scoring, pipeline
            )
            
            # Cross-validate
            scores = cross_val_score(pipeline, X, y, cv=cv, 
                                    scoring=scoring, n_jobs=-1)
            
            # Fit final model
            pipeline.fit(X, y)
            y_pred = pipeline.predict(X)
            
            if self.task == 'classification' and hasattr(pipeline, 'predict_proba'):
                y_proba = pipeline.predict_proba(X)
            else:
                y_proba = None
            
            # Calculate metrics
            metrics = calculate_metrics(y, y_pred, y_proba, self.task)
            
            # Training time
            training_time = time.time() - start_time
            
            # Create result
            result = {
                'model': model_name,
                'cv_score': scores.mean(),
                'cv_std': scores.std(),
                'metrics': metrics,
                'params': params,
                'training_time': training_time,
                'pipeline': pipeline,
                'timestamp': datetime.now().isoformat()
            }
            
            # Save model to storage
            if self.storage_manager:
                model_id = f"{self.experiment_id}_{model_name}"
                version = self._get_next_version(model_id)
                
                metadata = {
                    'model_id': model_id,
                    'model_type': self.task,
                    'algorithm': model_name,
                    'metrics': metrics,
                    'parameters': params,
                    'feature_names': list(X.columns),
                    'target_name': y.name or 'target',
                    'dataset_hash': self.dataset_hash,
                    'pipeline_hash': hashlib.sha256(str(pipeline).encode()).hexdigest()[:16],
                    'tags': [self.experiment_id, self.task, model_name],
                    'description': f"Model trained in experiment {self.experiment_id}",
                    'author': self.config.user_id or 'automl',
                    'tenant_id': self.config.tenant_id,
                    'llm_insights': self.llm_insights.get('model_explanation', '')
                }
                
                model_path = self.storage_manager.save_model(pipeline, metadata, version)
                result['model_path'] = model_path
                
                # Register model
                self.model_registry[model_id] = {
                    'version': version,
                    'path': model_path,
                    'metrics': metrics
                }
            
            # Setup monitoring
            if self.monitoring_service and reference_data is not None:
                monitor = self.monitoring_service.register_model(
                    model_id=f"{self.experiment_id}_{model_name}",
                    model_type=self.task,
                    reference_data=reference_data
                )
                
                # Log initial predictions for monitoring
                monitor.log_prediction(X, y_pred, y, training_time)
            
            # Add to leaderboard
            self.leaderboard.append(result)
            self.total_models_trained += 1
            
            logger.info(f"{model_name}: CV Score = {scores.mean():.4f} (+/- {scores.std():.4f})")
            
        except Exception as e:
            logger.error(f"Failed to train {model_name}: {e}")
            self.total_models_trained += 1
    
    def _optimize_hyperparameters(self, model_name: str, base_model: Any,
                                 X: pd.DataFrame, y: pd.Series,
                                 cv: Any, scoring: str,
                                 pipeline: Pipeline) -> Dict:
        """Optimize hyperparameters for a model."""
        params = {}
        
        # Try Optuna for important models
        if self.config.hpo_method == 'optuna' and model_name in [
            'RandomForestClassifier', 'RandomForestRegressor',
            'XGBClassifier', 'XGBRegressor',
            'LGBMClassifier', 'LGBMRegressor',
            'GradientBoostingClassifier', 'GradientBoostingRegressor'
        ]:
            # Prepare data for Optuna
            from sklearn.model_selection import train_test_split
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=self.config.random_state,
                stratify=y if self.task == 'classification' else None
            )
            
            # Fit preprocessor
            temp_preprocessor = DataPreprocessor(self.config.to_dict())
            X_train_preprocessed = temp_preprocessor.fit_transform(X_train, y_train)
            X_val_preprocessed = temp_preprocessor.transform(X_val)
            
            # Combine for Optuna
            X_preprocessed = np.vstack([X_train_preprocessed, X_val_preprocessed])
            y_combined = pd.concat([y_train, y_val])
            
            tuned_model, params = try_optuna(
                model_name, X_preprocessed, y_combined, self.task,
                cv, scoring, n_trials=self.config.hpo_n_iter
            )
            
            if tuned_model is not None:
                base_model.set_params(**params)
                pipeline.set_params(model=base_model)
        
        # Fallback to grid search
        elif self.config.hpo_method in ['grid', 'random']:
            param_grid = get_param_grid(model_name)
            if param_grid:
                param_grid = {f'model__{k}': v for k, v in param_grid.items()}
                tuned_model, params = tune_model(
                    pipeline, X, y, param_grid, cv, scoring,
                    self.config.hpo_n_iter
                )
                if params:
                    pipeline = tuned_model
                    params = {k.replace('model__', ''): v for k, v in params.items()}
        
        return params
    
    def _get_next_version(self, model_id: str) -> str:
        """Get next version number for a model."""
        if self.storage_manager:
            try:
                models = self.storage_manager.list_models(self.config.tenant_id)
                versions = [m['version'] for m in models if m.get('model_id') == model_id]
                if versions:
                    latest = sorted(versions, key=lambda x: tuple(map(int, x.split('.'))))[-1]
                    major, minor, patch = map(int, latest.split('.'))
                    return f"{major}.{minor}.{patch + 1}"
            except:
                pass
        return "1.0.0"
    
    def _select_best_model(self, X: pd.DataFrame, y: pd.Series, reference_data: pd.DataFrame = None):
        """Select and configure best model."""
        self.best_pipeline = self.leaderboard[0]['pipeline']
        
        # Calculate feature importance
        try:
            self._calculate_feature_importance(X, y)
        except Exception as e:
            logger.warning(f"Could not calculate feature importance: {e}")
        
        # Setup production monitoring for best model
        if self.monitoring_service and reference_data is not None:
            best_model_id = f"{self.experiment_id}_best"
            monitor = self.monitoring_service.register_model(
                model_id=best_model_id,
                model_type=self.task,
                reference_data=reference_data
            )
            
            # Store monitor reference
            self.best_model_monitor = monitor
    
    def _save_experiment_results(self):
        """Save experiment results and metadata."""
        if not self.storage_manager:
            return
        
        experiment_metadata = {
            'experiment_id': self.experiment_id,
            'task': self.task,
            'dataset_hash': self.dataset_hash,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'total_models_trained': self.total_models_trained,
            'best_model': self.leaderboard[0]['model'] if self.leaderboard else None,
            'best_score': self.leaderboard[0]['cv_score'] if self.leaderboard else None,
            'config': self.config.to_dict(),
            'leaderboard': [
                {
                    'model': r['model'],
                    'cv_score': r['cv_score'],
                    'metrics': r['metrics'],
                    'training_time': r['training_time']
                }
                for r in self.leaderboard[:10]  # Top 10 models
            ],
            'llm_insights': self.llm_insights,
            'feature_suggestions': self.feature_suggestions[:5] if self.feature_suggestions else [],
            'cleaning_report': self.cleaning_report
        }
        
        # Save as JSON artifact
        experiment_path = Path(self.config.output_dir) / self.experiment_id
        experiment_path.mkdir(parents=True, exist_ok=True)
        
        with open(experiment_path / 'experiment_metadata.json', 'w') as f:
            json.dump(experiment_metadata, f, indent=2, default=str)
        
        logger.info(f"Experiment results saved to {experiment_path}")
    
    def _generate_monitoring_report(self):
        """Generate comprehensive monitoring report."""
        if not self.monitoring_service:
            return
        
        report = self.monitoring_service.create_global_dashboard()
        
        # Add LLM insights if available
        if self.llm_insights:
            report['llm_analysis'] = {
                'quality_score': self.llm_insights.get('quality_assessment', {}).get('quality_score'),
                'feature_suggestions_count': len(self.feature_suggestions),
                'model_explanation_available': 'model_explanation' in self.llm_insights
            }
        
        # Save report
        report_path = Path(self.config.monitoring.report_output_dir) / self.experiment_id
        report_path.mkdir(parents=True, exist_ok=True)
        
        with open(report_path / 'monitoring_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Check for alerts
        if self.alert_manager:
            active_alerts = self.alert_manager.get_active_alerts()
            if active_alerts:
                logger.warning(f"Active alerts: {active_alerts}")
                
                # Send notifications if configured
                for alert in active_alerts:
                    self._send_alert_notification(alert)
    
    def _send_alert_notification(self, alert: Dict):
        """Send alert notifications based on configuration."""
        if 'slack' in self.config.monitoring.alert_channels and self.config.monitoring.slack_webhook_url:
            from .monitoring import MonitoringIntegration
            MonitoringIntegration.send_to_slack(alert, self.config.monitoring.slack_webhook_url)
        
        if 'email' in self.config.monitoring.alert_channels and self.config.monitoring.email_smtp_host:
            from .monitoring import MonitoringIntegration
            smtp_config = {
                'host': self.config.monitoring.email_smtp_host,
                'port': self.config.monitoring.email_smtp_port,
                'from_email': self.config.monitoring.email_from,
                'username': os.getenv('SMTP_USERNAME'),
                'password': os.getenv('SMTP_PASSWORD')
            }
            MonitoringIntegration.send_to_email(
                alert, smtp_config, self.config.monitoring.email_recipients
            )
    
    def predict(self, X: pd.DataFrame, track: bool = True) -> np.ndarray:
        """Make predictions with optional monitoring."""
        if self.best_pipeline is None:
            raise ValueError("No model trained yet")
        
        start_time = time.time()
        predictions = self.best_pipeline.predict(X)
        prediction_time = time.time() - start_time
        
        # Track predictions if monitoring is enabled
        if track and hasattr(self, 'best_model_monitor'):
            self.best_model_monitor.log_prediction(
                X, predictions, None, prediction_time
            )
            
            # Check for drift
            drift_results = self.best_model_monitor.check_drift(X)
            if drift_results['drift_detected']:
                logger.warning(f"Data drift detected: {drift_results['drifted_features']}")
        
        return predictions
    
    def predict_proba(self, X: pd.DataFrame, track: bool = True) -> np.ndarray:
        """Get probability predictions with optional monitoring."""
        if self.best_pipeline is None:
            raise ValueError("No model trained yet")
        if not hasattr(self.best_pipeline, 'predict_proba'):
            raise ValueError("Model doesn't support probability predictions")
        
        start_time = time.time()
        predictions = self.best_pipeline.predict_proba(X)
        prediction_time = time.time() - start_time
        
        # Track predictions if monitoring is enabled
        if track and hasattr(self, 'best_model_monitor'):
            self.best_model_monitor.log_prediction(
                X, predictions[:, 1] if predictions.shape[1] == 2 else predictions,
                None, prediction_time
            )
        
        return predictions
    
    def get_leaderboard(self, top_n: Optional[int] = None) -> pd.DataFrame:
        """Get enhanced leaderboard as DataFrame."""
        if not self.leaderboard:
            return pd.DataFrame()
        
        data = []
        for result in self.leaderboard[:top_n]:
            row = {
                'model': result['model'],
                'cv_score': result['cv_score'],
                'cv_std': result['cv_std'],
                'training_time': result['training_time'],
                'timestamp': result.get('timestamp', ''),
                'model_path': result.get('model_path', '')
            }
            
            # Add metrics
            for metric_name, metric_value in result['metrics'].items():
                row[metric_name] = metric_value
            
            data.append(row)
        
        return pd.DataFrame(data)
    
    def save_pipeline(self, filepath: str = None) -> str:
        """Save best pipeline with enhanced metadata."""
        if self.best_pipeline is None:
            raise ValueError("No pipeline to save")
        
        if filepath is None:
            filepath = str(Path(self.config.output_dir) / f"{self.experiment_id}_best_model.pkl")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save pipeline
        joblib.dump(self.best_pipeline, filepath)
        
        # Save enhanced metadata including LLM insights
        metadata = {
            'experiment_id': self.experiment_id,
            'task': self.task,
            'best_model': self.leaderboard[0]['model'] if self.leaderboard else None,
            'cv_score': self.leaderboard[0]['cv_score'] if self.leaderboard else None,
            'metrics': self.leaderboard[0]['metrics'] if self.leaderboard else None,
            'feature_importance': self.feature_importance,
            'dataset_hash': self.dataset_hash,
            'model_registry': self.model_registry,
            'training_time': self.end_time - self.start_time if self.end_time else None,
            'config': self.config.to_dict(),
            'llm_insights': self.llm_insights,
            'feature_suggestions': self.feature_suggestions,
            'cleaning_report': self.cleaning_report
        }
        
        metadata_path = filepath.with_suffix('.meta.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Pipeline saved to {filepath}")
        return str(filepath)
    
    def load_pipeline(self, filepath: str) -> None:
        """Load pipeline with enhanced metadata."""
        filepath = Path(filepath)
        
        # Load pipeline
        self.best_pipeline = joblib.load(filepath)
        
        # Load metadata
        metadata_path = filepath.with_suffix('.meta.json')
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                self.task = metadata.get('task')
                self.feature_importance = metadata.get('feature_importance', {})
                self.experiment_id = metadata.get('experiment_id')
                self.dataset_hash = metadata.get('dataset_hash')
                self.model_registry = metadata.get('model_registry', {})
                self.llm_insights = metadata.get('llm_insights', {})
                self.feature_suggestions = metadata.get('feature_suggestions', [])
                self.cleaning_report = metadata.get('cleaning_report')
        
        logger.info(f"Pipeline loaded from {filepath}")
    
    def _calculate_feature_importance(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Calculate feature importance using permutation."""
        from sklearn.inspection import permutation_importance
        
        try:
            X_transformed = self.best_pipeline.named_steps['preprocessor'].transform(X)
            
            result = permutation_importance(
                self.best_pipeline.named_steps['model'],
                X_transformed, y,
                n_repeats=5,
                random_state=self.config.random_state,
                n_jobs=-1
            )
            
            self.feature_importance = {
                'importances_mean': result.importances_mean.tolist(),
                'importances_std': result.importances_std.tolist(),
                'feature_names': list(X.columns)
            }
            
        except Exception as e:
            logger.warning(f"Could not calculate feature importance: {e}")
    
    def get_model_info(self, model_id: str = None) -> Dict:
        """Get detailed information about a model."""
        if model_id is None:
            # Return info about best model
            if not self.leaderboard:
                return {}
            
            return {
                'model': self.leaderboard[0]['model'],
                'metrics': self.leaderboard[0]['metrics'],
                'parameters': self.leaderboard[0].get('params', {}),
                'feature_importance': self.feature_importance,
                'training_time': self.leaderboard[0]['training_time'],
                'llm_explanation': self.llm_insights.get('model_explanation', '')
            }
        else:
            # Return info about specific model from registry
            return self.model_registry.get(model_id, {})
    
    def get_monitoring_metrics(self) -> Dict:
        """Get current monitoring metrics."""
        if not self.monitoring_service:
            return {}
        
        metrics = self.monitoring_service.create_global_dashboard()
        
        # Add LLM metrics if available
        if self.llm_assistant:
            metrics['llm_usage'] = self.llm_assistant.get_usage_stats()
        
        return metrics
    
    def get_llm_insights(self) -> Dict:
        """Get all LLM-generated insights."""
        return {
            'quality_assessment': self.llm_insights.get('quality_assessment'),
            'feature_suggestions': self.feature_suggestions,
            'model_explanation': self.llm_insights.get('model_explanation'),
            'cleaning_report': self.cleaning_report,
            'report': self.llm_insights.get('report')
        }
    
    def retrain(self, X: pd.DataFrame, y: pd.Series,
               model_name: str = None) -> 'EnhancedAutoMLOrchestrator':
        """Retrain model with new data."""
        logger.info(f"Retraining model with new data (shape: {X.shape})")
        
        # If specific model requested, train only that
        if model_name:
            self.config.algorithms = [model_name]
        elif self.leaderboard:
            # Retrain only the best model
            self.config.algorithms = [self.leaderboard[0]['model']]
        
        # Reduce HPO iterations for retraining
        original_hpo_iter = self.config.hpo_n_iter
        self.config.hpo_n_iter = min(10, original_hpo_iter)
        
        # Run training
        self.fit(X, y, task=self.task, 
                experiment_name=f"{self.experiment_id}_retrain")
        
        # Restore original config
        self.config.hpo_n_iter = original_hpo_iter
        
        return self


# Convenience function
def create_automl_pipeline(config_path: str = None,
                          environment: str = None,
                          enable_llm: bool = None) -> EnhancedAutoMLOrchestrator:
    """
    Create an enhanced AutoML pipeline with configuration.
    
    Args:
        config_path: Path to configuration file
        environment: Environment name (development/production)
        enable_llm: Whether to enable LLM features
    
    Returns:
        Configured orchestrator instance
    """
    config = load_config(config_path, environment)
    
    # Override LLM setting if specified
    if enable_llm is not None:
        config.llm.enabled = enable_llm
    
    return EnhancedAutoMLOrchestrator(config)


# Example usage
if __name__ == "__main__":
    import os
    
    # Create sample data
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=15,
                              n_redundant=5, n_classes=2, random_state=42)
    X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(20)])
    y = pd.Series(y, name="target")
    
    # Create orchestrator with production config
    config = AutoMLConfig()
    config.environment = "production"
    config.apply_environment_config()
    
    # Enable key features
    config.storage.backend = "local"  # or "minio" for production
    config.monitoring.enabled = True
    config.worker.enabled = False  # Set to True for parallel training
    config.llm.enabled = True  # Enable LLM features
    config.llm.api_key = os.getenv("OPENAI_API_KEY")  # Set your API key
    
    # Create orchestrator
    orchestrator = EnhancedAutoMLOrchestrator(config)
    
    # Train models with LLM enhancements
    orchestrator.fit(
        X, y, 
        experiment_name="demo_experiment_with_llm",
        use_llm_features=True,
        use_llm_cleaning=True
    )
    
    # Get leaderboard
    leaderboard = orchestrator.get_leaderboard()
    print("\nLeaderboard:")
    print(leaderboard.head())
    
    # Get LLM insights
    insights = orchestrator.get_llm_insights()
    print("\nLLM Insights Available:")
    for key in insights.keys():
        if insights[key]:
            print(f"  - {key}: ✓")
    
    # Chat with assistant about results
    loop = asyncio.get_event_loop()
    response = loop.run_until_complete(
        orchestrator.chat_with_assistant("Why did the best model perform well?")
    )
    print(f"\nAssistant Response: {response[:200]}...")
    
    # Make predictions
    X_test = X.iloc[:10]
    predictions = orchestrator.predict(X_test)
    print(f"\nPredictions: {predictions}")
    
    # Get monitoring metrics
    metrics = orchestrator.get_monitoring_metrics()
    print(f"\nMonitoring metrics available: {list(metrics.keys())}")
    
    # Save best model
    model_path = orchestrator.save_pipeline()
    print(f"\nModel saved to: {model_path}")
