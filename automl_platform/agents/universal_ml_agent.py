"""
Universal ML Agent for AutoML Platform
=======================================
The main orchestrator that handles any ML problem without templates.
Fully integrated with existing agents.
"""

import asyncio
import logging
import pandas as pd
import numpy as np
import json
import aiohttp
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
import hashlib
import pickle
from pathlib import Path

from .intelligent_context_detector import IntelligentContextDetector, MLContext
from .intelligent_config_generator import IntelligentConfigGenerator, OptimalConfig
from .adaptive_template_system import AdaptiveTemplateSystem
from .data_cleaning_orchestrator import DataCleaningOrchestrator
from .agent_config import AgentConfig
from .yaml_config_handler import YAMLConfigHandler

# Import existing agents for full integration
from .profiler_agent import ProfilerAgent
from .validator_agent import ValidatorAgent
from .cleaner_agent import CleanerAgent
from .controller_agent import ControllerAgent

logger = logging.getLogger(__name__)


@dataclass
class MLPipelineResult:
    """Result of the ML pipeline execution"""
    success: bool
    cleaned_data: Optional[pd.DataFrame]
    config_used: OptimalConfig
    context_detected: MLContext
    cleaning_report: Dict[str, Any]
    performance_metrics: Dict[str, float]
    execution_time: float
    model_artifacts: Optional[Dict[str, Any]] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    yaml_config_path: Optional[str] = None  # Path to saved YAML config


class UniversalMLAgent:
    """
    The Universal ML Agent - handles ANY ML problem without templates.
    This is the pinnacle of the Agent-First approach.
    Fully integrated with existing OpenAI agents.
    """
    
    def __init__(self, config: Optional[AgentConfig] = None):
        """
        Initialize the Universal ML Agent
        
        Args:
            config: Optional agent configuration
        """
        self.config = config or AgentConfig()
        
        # Initialize intelligent components
        self.context_detector = IntelligentContextDetector()
        self.config_generator = IntelligentConfigGenerator()
        self.adaptive_templates = AdaptiveTemplateSystem()
        
        # Initialize cleaning orchestrator with existing agents
        self.cleaning_orchestrator = DataCleaningOrchestrator(self.config)
        
        # Individual agents for direct access if needed
        self.profiler = ProfilerAgent(self.config)
        self.validator = ValidatorAgent(self.config)
        self.cleaner = CleanerAgent(self.config)
        self.controller = ControllerAgent(self.config)
        
        # YAML handler for configuration persistence
        self.yaml_handler = YAMLConfigHandler()
        
        # Learning and caching
        self.knowledge_base = KnowledgeBase()
        self.cache_dir = Path(self.config.cache_dir) / "universal_agent"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Performance tracking
        self.execution_history = []
        self.agent_metrics = {
            "profiler_calls": 0,
            "validator_calls": 0,
            "cleaner_calls": 0,
            "controller_calls": 0,
            "context_detections": 0,
            "config_generations": 0
        }
    
    async def understand_problem(
        self,
        df: pd.DataFrame,
        target_col: Optional[str] = None,
        user_hints: Optional[Dict[str, Any]] = None
    ) -> MLContext:
        """
        Understand the ML problem from the data.
        No templates, no configuration - pure intelligence.
        """
        logger.info("🧠 Universal Agent: Understanding the problem...")
        self.agent_metrics["context_detections"] += 1
        
        # Check if we've seen similar data before
        data_hash = self._compute_data_hash(df)
        cached_context = self.knowledge_base.get_cached_context(data_hash)
        
        if cached_context:
            logger.info("📚 Found similar problem in knowledge base")
            return cached_context
        
        # First, use profiler agent to get deep insights
        logger.info("📊 Getting data profile from ProfilerAgent...")
        profile_report = await self.profiler.analyze(df)
        self.agent_metrics["profiler_calls"] += 1
        
        # Detect context intelligently with profile information
        context = await self.context_detector.detect_ml_context(
            df, target_col, user_hints
        )
        
        # Enhance context with profiler insights
        context.detected_patterns.extend([
            pattern for pattern in profile_report.get("quality_issues", [])
        ])
        
        # Store in knowledge base
        self.knowledge_base.store_context(data_hash, context)
        
        logger.info(f"✅ Problem understood: {context.problem_type} "
                   f"(confidence: {context.confidence:.1%})")
        
        return context
    
    async def validate_with_standards(
        self,
        df: pd.DataFrame,
        context: MLContext
    ) -> Dict[str, Any]:
        """
        Validate data against industry standards using ValidatorAgent
        """
        logger.info("🔍 Validating against industry standards...")
        self.agent_metrics["validator_calls"] += 1
        
        # Prepare profile report for validator
        profile_report = await self.profiler.analyze(df)
        
        # Update config with detected context
        self.config.user_context['secteur_activite'] = context.business_sector or 'general'
        
        # Use validator agent
        validation_report = await self.validator.validate(df, profile_report)
        
        logger.info(f"✅ Validation complete: {len(validation_report.get('issues', []))} issues found")
        
        return validation_report
    
    async def search_ml_best_practices(
        self,
        problem_type: str,
        data_characteristics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Search for ML best practices in real-time.
        Enhanced to use ValidatorAgent's web search capabilities.
        """
        logger.info(f"🔍 Searching best practices for {problem_type}...")
        
        best_practices = {
            'recommended_approaches': [],
            'recent_innovations': [],
            'common_pitfalls': [],
            'benchmark_scores': {},
            'sources': []
        }
        
        # Check knowledge base first
        cached_practices = self.knowledge_base.get_best_practices(problem_type)
        if cached_practices:
            return cached_practices
        
        # Use ValidatorAgent's web search functionality
        search_queries = [
            f"{problem_type} state of the art ML algorithms {datetime.now().year}",
            f"{problem_type} best practices machine learning",
            f"{problem_type} feature engineering techniques",
            f"{problem_type} common mistakes to avoid ML"
        ]
        
        # Perform searches through validator
        for query in search_queries:
            try:
                # Use validator's internal search method
                results = await self.validator._web_search(query)
                best_practices['sources'].extend(results.get('urls', []))
                
                # Parse results for insights
                for result in results.get('results', []):
                    if 'state of the art' in result.get('snippet', '').lower():
                        best_practices['recent_innovations'].append(result['snippet'])
                    elif 'best practice' in result.get('snippet', '').lower():
                        best_practices['recommended_approaches'].append(result['snippet'])
                    elif 'mistake' in result.get('snippet', '').lower():
                        best_practices['common_pitfalls'].append(result['snippet'])
            except Exception as e:
                logger.warning(f"Search failed for query '{query}': {e}")
        
        # Add problem-specific recommendations (enhanced)
        if problem_type == 'fraud_detection':
            best_practices['recommended_approaches'].extend([
                "Use ensemble of tree-based models (XGBoost, LightGBM) for robustness",
                "Implement real-time feature engineering for velocity checks",
                "Apply SMOTE or ADASYN for severe class imbalance",
                "Use Isolation Forest for anomaly-based fraud detection",
                "Implement graph-based features for network fraud patterns"
            ])
            best_practices['benchmark_scores'] = {
                'industry_average_auc': 0.95,
                'state_of_art_auc': 0.99,
                'minimum_acceptable': 0.90
            }
        elif problem_type == 'churn_prediction':
            best_practices['recommended_approaches'].extend([
                "Create RFM (Recency, Frequency, Monetary) features",
                "Use survival analysis for time-to-churn prediction",
                "Apply SMOTE for class imbalance handling",
                "Implement uplift modeling for intervention strategies",
                "Use SHAP values for interpretability"
            ])
            best_practices['benchmark_scores'] = {
                'industry_average_f1': 0.75,
                'state_of_art_f1': 0.85,
                'minimum_acceptable': 0.65
            }
        elif problem_type == 'sales_forecasting':
            best_practices['recommended_approaches'].extend([
                "Use Prophet or ARIMA for time series baseline",
                "Implement LightGBM with lag features for complex patterns",
                "Create seasonal and holiday features",
                "Use ensemble of statistical and ML models",
                "Implement hierarchical forecasting for multi-level data"
            ])
            best_practices['benchmark_scores'] = {
                'industry_average_mape': 0.15,
                'state_of_art_mape': 0.08,
                'maximum_acceptable': 0.25
            }
        
        # Store in knowledge base
        self.knowledge_base.store_best_practices(problem_type, best_practices)
        
        return best_practices
    
    async def generate_optimal_config(
        self,
        understanding: MLContext,
        best_practices: Dict[str, Any],
        user_preferences: Optional[Dict[str, Any]] = None,
        constraints: Optional[Dict[str, Any]] = None
    ) -> OptimalConfig:
        """
        Generate optimal configuration based on understanding and best practices.
        """
        logger.info("⚙️ Generating optimal configuration...")
        self.agent_metrics["config_generations"] += 1
        
        # Prepare context for config generator
        context = {
            'problem_type': understanding.problem_type,
            'detected_patterns': understanding.detected_patterns,
            'business_sector': understanding.business_sector,
            'temporal_aspect': understanding.temporal_aspect,
            'imbalance_detected': understanding.imbalance_detected,
            'confidence': understanding.confidence
        }
        
        # Add insights from best practices
        if best_practices.get('recommended_approaches'):
            context['recommended_approaches'] = best_practices['recommended_approaches']
        
        # Generate configuration
        config = await self.config_generator.generate_config(
            df=pd.DataFrame(),  # Will be provided later
            context=context,
            constraints=constraints,
            user_preferences=user_preferences
        )
        
        # Get adaptive improvements from template system
        adaptive_config = await self.adaptive_templates.get_configuration(
            df=pd.DataFrame(),
            context=context,
            agent_config=config.to_dict()
        )
        
        # Update config with adaptive improvements
        for key, value in adaptive_config.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        logger.info(f"✅ Configuration generated: {len(config.algorithms)} algorithms selected")
        
        return config
    
    async def execute_intelligent_cleaning(
        self,
        df: pd.DataFrame,
        context: MLContext,
        config: OptimalConfig,
        target_col: Optional[str] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Execute intelligent data cleaning using integrated agents
        """
        logger.info("🧹 Starting intelligent data cleaning...")
        
        # Prepare user context for cleaning orchestrator
        user_context = {
            'secteur_activite': context.business_sector or config.task,
            'target_variable': target_col,
            'contexte_metier': f"ML Pipeline for {context.problem_type}",
            'detected_problem_type': context.problem_type,
            'ml_confidence': context.confidence
        }
        
        # Convert OptimalConfig to cleaning config
        cleaning_config = {
            'preprocessing': config.preprocessing,
            'feature_engineering': config.feature_engineering,
            'algorithms': config.algorithms,
            'task': config.task,
            'primary_metric': config.primary_metric
        }
        
        # Execute cleaning with full orchestration
        cleaned_df, cleaning_report = await self.cleaning_orchestrator.clean_dataset(
            df=df,
            user_context=user_context,
            cleaning_config=cleaning_config,
            use_intelligence=True  # Enable Agent-First features
        )
        
        # Enhance report with agent metrics
        cleaning_report['agent_metrics'] = self.agent_metrics
        
        logger.info(f"✅ Data cleaning complete. Quality improved by "
                   f"{cleaning_report.get('quality_metrics', {}).get('improvement', 0):.1f} points")
        
        return cleaned_df, cleaning_report
    
    async def execute_with_continuous_learning(
        self,
        df: pd.DataFrame,
        config: OptimalConfig,
        context: MLContext,
        target_col: Optional[str] = None,
        validation_df: Optional[pd.DataFrame] = None
    ) -> MLPipelineResult:
        """
        Execute the ML pipeline with continuous learning and adaptation.
        Fully integrated with existing agents.
        """
        logger.info("🚀 Executing ML pipeline with continuous learning...")
        
        start_time = datetime.now()
        errors = []
        warnings = []
        yaml_config_path = None
        
        try:
            # Phase 1: Intelligent Data Cleaning (using integrated agents)
            logger.info("🧹 Phase 1: Intelligent Data Cleaning")
            cleaned_df, cleaning_report = await self.execute_intelligent_cleaning(
                df, context, config, target_col
            )
            
            self.agent_metrics["cleaner_calls"] += 1
            
            # Phase 2: Validation & Quality Control (using ControllerAgent)
            logger.info("🎯 Phase 2: Quality Control")
            control_report = await self.controller.validate(
                cleaned_df=cleaned_df,
                original_df=df,
                transformations=cleaning_report.get('transformations', [])
            )
            self.agent_metrics["controller_calls"] += 1
            
            if not control_report.get('validation_passed', False):
                warnings.extend(control_report.get('warnings', []))
                errors.extend(control_report.get('issues', []))
            
            # Phase 3: Save YAML Configuration
            logger.info("💾 Phase 3: Saving Configuration")
            yaml_config_path = self.yaml_handler.save_cleaning_config(
                transformations=cleaning_report.get('transformations', []),
                validation_sources=cleaning_report.get('validation_sources', []),
                user_context={
                    'secteur_activite': context.business_sector,
                    'target_variable': target_col,
                    'detected_problem_type': context.problem_type,
                    'ml_confidence': context.confidence
                },
                metrics=control_report.get('metrics', {})
            )
            
            # Phase 4: Feature Engineering
            logger.info("🔧 Phase 4: Feature Engineering")
            engineered_df = await self._apply_feature_engineering(
                cleaned_df, config.feature_engineering
            )
            
            # Phase 5: Model Training (simulated for now)
            logger.info("🎯 Phase 5: Model Training")
            model_artifacts = await self._train_models(
                engineered_df, target_col, config
            )
            
            # Phase 6: Evaluation
            logger.info("📊 Phase 6: Evaluation")
            performance_metrics = await self._evaluate_models(
                model_artifacts, validation_df or engineered_df, target_col, config
            )
            
            # Phase 7: Ensemble (if configured)
            if config.ensemble_config.get('enabled'):
                logger.info("🤝 Phase 7: Creating Ensemble")
                ensemble_model = await self._create_ensemble(
                    model_artifacts, config.ensemble_config
                )
                model_artifacts['ensemble'] = ensemble_model
            
            # Learn from this execution
            self._learn_from_execution(config, performance_metrics, context)
            
            # Update adaptive templates with results
            await self.adaptive_templates.learn_from_execution(
                context={
                    'problem_type': context.problem_type,
                    'n_samples': len(df),
                    'n_features': len(df.columns)
                },
                config=config.to_dict(),
                performance=performance_metrics
            )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"✅ Pipeline completed in {execution_time:.1f} seconds")
            logger.info(f"📈 Best {config.primary_metric}: {max(performance_metrics.values()):.3f}")
            
            return MLPipelineResult(
                success=True,
                cleaned_data=cleaned_df,
                config_used=config,
                context_detected=context,
                cleaning_report=cleaning_report,
                performance_metrics=performance_metrics,
                execution_time=execution_time,
                model_artifacts=model_artifacts,
                errors=errors,
                warnings=warnings,
                yaml_config_path=yaml_config_path
            )
            
        except Exception as e:
            logger.error(f"❌ Pipeline failed: {str(e)}")
            errors.append(str(e))
            
            return MLPipelineResult(
                success=False,
                cleaned_data=None,
                config_used=config,
                context_detected=context,
                cleaning_report={},
                performance_metrics={},
                execution_time=(datetime.now() - start_time).total_seconds(),
                model_artifacts=None,
                errors=errors,
                warnings=warnings,
                yaml_config_path=yaml_config_path
            )
    
    async def automl_without_templates(
        self,
        df: pd.DataFrame,
        target_col: Optional[str] = None,
        user_hints: Optional[Dict[str, Any]] = None,
        constraints: Optional[Dict[str, Any]] = None
    ) -> MLPipelineResult:
        """
        Complete AutoML without any templates - pure intelligence.
        This is the main entry point for the Universal Agent.
        """
        logger.info("=" * 50)
        logger.info("🤖 UNIVERSAL ML AGENT - NO TEMPLATES NEEDED")
        logger.info("=" * 50)
        
        # Step 1: Understand the problem (using ProfilerAgent + ContextDetector)
        problem_understanding = await self.understand_problem(df, target_col, user_hints)
        
        # Step 2: Validate against standards (using ValidatorAgent)
        validation_report = await self.validate_with_standards(df, problem_understanding)
        
        # Step 3: Search for best practices (using ValidatorAgent's web search)
        best_practices = await self.search_ml_best_practices(
            problem_type=problem_understanding.problem_type,
            data_characteristics={
                'n_samples': len(df),
                'n_features': len(df.columns),
                'has_temporal': problem_understanding.temporal_aspect,
                'has_imbalance': problem_understanding.imbalance_detected
            }
        )
        
        # Step 4: Generate optimal configuration (using ConfigGenerator + AdaptiveTemplates)
        config = await self.generate_optimal_config(
            understanding=problem_understanding,
            best_practices=best_practices,
            user_preferences=user_hints,
            constraints=constraints
        )
        
        # Step 5: Execute with continuous learning (using all agents)
        result = await self.execute_with_continuous_learning(
            df=df,
            config=config,
            context=problem_understanding,
            target_col=target_col
        )
        
        # Store execution in history
        self.execution_history.append({
            'timestamp': datetime.now(),
            'problem_type': problem_understanding.problem_type,
            'success': result.success,
            'performance': result.performance_metrics,
            'execution_time': result.execution_time,
            'agent_metrics': self.agent_metrics.copy()
        })
        
        # Log summary
        logger.info("=" * 50)
        logger.info("📊 EXECUTION SUMMARY")
        logger.info(f"Problem Type: {problem_understanding.problem_type}")
        logger.info(f"Confidence: {problem_understanding.confidence:.1%}")
        logger.info(f"Success: {result.success}")
        logger.info(f"Execution Time: {result.execution_time:.1f}s")
        logger.info(f"Agent Calls: {sum(self.agent_metrics.values())}")
        if result.yaml_config_path:
            logger.info(f"Config saved: {result.yaml_config_path}")
        logger.info("=" * 50)
        
        return result
    
    async def _apply_feature_engineering(
        self,
        df: pd.DataFrame,
        feature_config: Dict[str, Any]
    ) -> pd.DataFrame:
        """Apply feature engineering based on configuration"""
        engineered_df = df.copy()
        
        # Implement basic feature engineering
        if feature_config.get('datetime_features'):
            for col in engineered_df.select_dtypes(include=['datetime64']).columns:
                engineered_df[f'{col}_year'] = engineered_df[col].dt.year
                engineered_df[f'{col}_month'] = engineered_df[col].dt.month
                engineered_df[f'{col}_day'] = engineered_df[col].dt.day
                engineered_df[f'{col}_dayofweek'] = engineered_df[col].dt.dayofweek
        
        logger.info(f"Applied {len(feature_config)} feature engineering steps")
        
        return engineered_df
    
    async def _train_models(
        self,
        df: pd.DataFrame,
        target_col: str,
        config: OptimalConfig
    ) -> Dict[str, Any]:
        """Train models based on configuration"""
        model_artifacts = {}
        
        # This would implement actual model training
        # For now, return placeholder
        for algo in config.algorithms:
            logger.info(f"Training {algo}...")
            await asyncio.sleep(0.1)  # Simulate training
            model_artifacts[algo] = {'trained': True, 'algorithm': algo}
        
        return model_artifacts
    
    async def _evaluate_models(
        self,
        model_artifacts: Dict[str, Any],
        df: pd.DataFrame,
        target_col: str,
        config: OptimalConfig
    ) -> Dict[str, float]:
        """Evaluate trained models"""
        performance_metrics = {}
        
        # Simulate evaluation based on problem type and algorithms
        for algo in model_artifacts:
            if config.task == 'classification':
                if algo == 'XGBoost':
                    performance_metrics[algo] = 0.92
                elif algo == 'LightGBM':
                    performance_metrics[algo] = 0.91
                elif algo == 'RandomForest':
                    performance_metrics[algo] = 0.89
                elif algo == 'LogisticRegression':
                    performance_metrics[algo] = 0.85
                else:
                    performance_metrics[algo] = 0.80 + np.random.random() * 0.15
            elif config.task == 'regression':
                # Lower is better for regression metrics
                if algo == 'LightGBM':
                    performance_metrics[algo] = 0.08
                elif algo == 'XGBoost':
                    performance_metrics[algo] = 0.09
                else:
                    performance_metrics[algo] = 0.10 + np.random.random() * 0.05
            else:
                performance_metrics[algo] = 0.70 + np.random.random() * 0.20
        
        return performance_metrics
    
    async def _create_ensemble(
        self,
        model_artifacts: Dict[str, Any],
        ensemble_config: Dict[str, Any]
    ) -> Any:
        """Create ensemble from trained models"""
        logger.info(f"Creating {ensemble_config.get('method', 'voting')} ensemble")
        
        # This would implement actual ensemble creation
        return {'type': 'ensemble', 'base_models': list(model_artifacts.keys())}
    
    def _compute_data_hash(self, df: pd.DataFrame) -> str:
        """Compute hash of dataframe for caching"""
        # Use shape, column names, and dtypes for hash
        hash_input = f"{df.shape}_{list(df.columns)}_{df.dtypes.tolist()}"
        return hashlib.md5(hash_input.encode()).hexdigest()
    
    def _learn_from_execution(
        self,
        config: OptimalConfig,
        performance: Dict[str, float],
        context: MLContext
    ):
        """Learn from execution results"""
        # Update config generator with results
        self.config_generator.learn_from_results(
            config, performance, 0  # execution time would be tracked
        )
        
        # Update knowledge base with successful patterns
        if performance and max(performance.values()) > 0.9:
            self.knowledge_base.store_successful_pattern(
                config.task, config.to_dict(), performance
            )
            logger.info(f"📚 Stored successful pattern (performance: {max(performance.values()):.3f})")
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get summary of all executions"""
        if not self.execution_history:
            return {"message": "No executions performed yet"}
        
        return {
            "total_executions": len(self.execution_history),
            "success_rate": sum(1 for e in self.execution_history if e['success']) / len(self.execution_history),
            "average_execution_time": sum(e['execution_time'] for e in self.execution_history) / len(self.execution_history),
            "total_agent_calls": sum(e['agent_metrics'].values() for e in self.execution_history),
            "problem_types_handled": list(set(e['problem_type'] for e in self.execution_history)),
            "last_execution": self.execution_history[-1]['timestamp']
        }


class KnowledgeBase:
    """
    Knowledge base for storing and retrieving learned patterns.
    This enables continuous learning and improvement.
    """
    
    def __init__(self, storage_path: Optional[Path] = None):
        """Initialize knowledge base"""
        self.storage_path = storage_path or Path("./knowledge_base")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # In-memory caches
        self.context_cache = {}
        self.best_practices_cache = {}
        self.successful_patterns = []
        
        # Load existing knowledge
        self._load_knowledge()
    
    def get_cached_context(self, data_hash: str) -> Optional[MLContext]:
        """Get cached context for data hash"""
        return self.context_cache.get(data_hash)
    
    def store_context(self, data_hash: str, context: MLContext):
        """Store context in cache"""
        self.context_cache[data_hash] = context
        self._save_knowledge()
    
    def get_best_practices(self, problem_type: str) -> Optional[Dict[str, Any]]:
        """Get best practices for problem type"""
        return self.best_practices_cache.get(problem_type)
    
    def store_best_practices(self, problem_type: str, practices: Dict[str, Any]):
        """Store best practices"""
        self.best_practices_cache[problem_type] = practices
        self._save_knowledge()
    
    def store_successful_pattern(
        self,
        task: str,
        config: Dict[str, Any],
        performance: Dict[str, float]
    ):
        """Store successful configuration pattern"""
        pattern = {
            'timestamp': datetime.now().isoformat(),
            'task': task,
            'config': config,
            'performance': performance,
            'success_score': max(performance.values()) if performance else 0
        }
        self.successful_patterns.append(pattern)
        
        # Keep only recent successful patterns (last 100)
        self.successful_patterns = self.successful_patterns[-100:]
        self._save_knowledge()
    
    def get_similar_successful_patterns(
        self,
        task: str,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Get similar successful patterns for a task"""
        task_patterns = [
            p for p in self.successful_patterns
            if p['task'] == task
        ]
        
        # Sort by success score
        task_patterns.sort(key=lambda x: x['success_score'], reverse=True)
        
        return task_patterns[:limit]
    
    def _load_knowledge(self):
        """Load knowledge from disk"""
        knowledge_file = self.storage_path / "knowledge.pkl"
        if knowledge_file.exists():
            try:
                with open(knowledge_file, 'rb') as f:
                    knowledge = pickle.load(f)
                    self.context_cache = knowledge.get('contexts', {})
                    self.best_practices_cache = knowledge.get('best_practices', {})
                    self.successful_patterns = knowledge.get('patterns', [])
                logger.info(f"Loaded knowledge base with {len(self.successful_patterns)} patterns")
            except Exception as e:
                logger.warning(f"Failed to load knowledge base: {e}")
    
    def _save_knowledge(self):
        """Save knowledge to disk"""
        knowledge_file = self.storage_path / "knowledge.pkl"
        try:
            knowledge = {
                'contexts': self.context_cache,
                'best_practices': self.best_practices_cache,
                'patterns': self.successful_patterns
            }
            with open(knowledge_file, 'wb') as f:
                pickle.dump(knowledge, f)
        except Exception as e:
            logger.warning(f"Failed to save knowledge base: {e}")
