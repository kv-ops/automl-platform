"""
Data Cleaning Orchestrator - Main coordination for OpenAI agents
Integrated with Agent-First intelligent modules
"""

import pandas as pd
import numpy as np
import asyncio
import logging
from logging.handlers import RotatingFileHandler
import time
import yaml
from typing import Tuple, Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime
import json

from .agent_config import AgentConfig, AgentType
from .profiler_agent import ProfilerAgent
from .validator_agent import ValidatorAgent
from .cleaner_agent import CleanerAgent
from .controller_agent import ControllerAgent

# Import intelligent modules for Agent-First approach
from .intelligent_context_detector import IntelligentContextDetector
from .intelligent_config_generator import IntelligentConfigGenerator
from .adaptive_template_system import AdaptiveTemplateSystem

logger = logging.getLogger(__name__)


class DataCleaningOrchestrator:
    """
    Orchestrates the intelligent data cleaning process using OpenAI agents
    Now integrated with Agent-First approach for template-free operation
    """
    
    def __init__(self, config: Optional[AgentConfig] = None, automl_config: Optional[Dict] = None):
        """
        Initialize orchestrator with configuration
        
        Args:
            config: Agent configuration
            automl_config: AutoML platform configuration
        """
        self.config = config or AgentConfig()
        self.automl_config = automl_config or {}
        
        # Validate configuration
        self.config.validate()
        
        # Initialize agents
        self.profiler = ProfilerAgent(self.config)
        self.validator = ValidatorAgent(self.config)
        self.cleaner = CleanerAgent(self.config)
        self.controller = ControllerAgent(self.config)
        
        # Initialize intelligent modules (Agent-First)
        self.context_detector = IntelligentContextDetector()
        self.config_generator = IntelligentConfigGenerator()
        self.adaptive_templates = AdaptiveTemplateSystem()
        
        # Tracking
        self.execution_history = []
        self.total_cost = 0.0
        self.start_time = None
        
        # Performance metrics
        self.performance_metrics = {
            "cleaning_time_per_agent": {},
            "total_api_calls": 0,
            "total_tokens_used": 0,
            "validation_success_rate": 0.0,
            "retry_count": 0,
            "intelligence_used": False  # Track if Agent-First was used
        }
        
        # Results storage
        self.cleaning_report = {}
        self.validation_sources = []
        self.transformations_applied = []
        self.ml_context = None  # Store detected ML context
        
        # Setup logging with file handler
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging configuration with file output"""
        log_file = self.config.log_file or "./logs/automl.log"
        log_dir = Path(log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
        )
        
        # File handler with rotation
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(getattr(logging, self.config.log_level))
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.INFO)
        
        # Configure logger
        logger = logging.getLogger(__name__)
        logger.setLevel(getattr(logging, self.config.log_level))
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    async def clean_dataset(
        self, 
        df: pd.DataFrame, 
        user_context: Dict[str, Any],
        cleaning_config: Optional[Dict] = None,
        use_intelligence: bool = True  # New parameter for Agent-First
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Main pipeline for intelligent data cleaning
        
        Args:
            df: Input dataframe
            user_context: User context (sector, target variable, etc.)
            cleaning_config: Optional cleaning configuration
            use_intelligence: Whether to use Agent-First intelligence
            
        Returns:
            Tuple of (cleaned_dataframe, cleaning_report)
        """
        self.start_time = time.time()
        
        # Update user context
        self.config.user_context.update(user_context)
        
        logger.info(f"Starting intelligent cleaning for dataset with shape {df.shape}")
        logger.info(f"User context: {user_context}")
        logger.info(f"Agent-First mode: {'ENABLED' if use_intelligence else 'DISABLED'}")
        
        try:
            # NEW: Agent-First approach - Detect ML context if enabled
            if use_intelligence:
                self.performance_metrics["intelligence_used"] = True
                self.ml_context = await self.context_detector.detect_ml_context(
                    df, 
                    target_col=user_context.get("target_variable"),
                    user_hints=user_context
                )
                logger.info(f"🎯 Detected ML problem: {self.ml_context.problem_type} "
                           f"(confidence: {self.ml_context.confidence:.1%})")
                
                # Generate optimal configuration dynamically
                if not cleaning_config:
                    optimal_config = await self.config_generator.generate_config(
                        df=df,
                        context={
                            'problem_type': self.ml_context.problem_type,
                            'business_sector': self.ml_context.business_sector,
                            'temporal_aspect': self.ml_context.temporal_aspect,
                            'imbalance_detected': self.ml_context.imbalance_detected
                        },
                        user_preferences=user_context
                    )
                    
                    # Convert to cleaning config
                    cleaning_config = {
                        'preprocessing': optimal_config.preprocessing,
                        'algorithms': optimal_config.algorithms,
                        'task': optimal_config.task
                    }
                    logger.info(f"✨ Generated optimal configuration automatically")
                
                # Get adaptive configuration
                cleaning_config = await self.adaptive_templates.get_configuration(
                    df=df,
                    context={
                        'problem_type': self.ml_context.problem_type,
                        'n_samples': len(df),
                        'n_features': len(df.columns),
                        'imbalance_detected': self.ml_context.imbalance_detected,
                        'temporal_aspect': self.ml_context.temporal_aspect,
                        'business_sector': self.ml_context.business_sector
                    },
                    agent_config=cleaning_config
                )
            
            # Check dataset size and chunk if necessary
            df_chunks = self._chunk_dataset(df) if self._needs_chunking(df) else [df]
            
            # Process each chunk
            cleaned_chunks = []
            for i, chunk in enumerate(df_chunks):
                logger.info(f"Processing chunk {i+1}/{len(df_chunks)}")
                cleaned_chunk = await self._process_chunk(chunk, i)
                cleaned_chunks.append(cleaned_chunk)
            
            # Combine chunks
            cleaned_df = pd.concat(cleaned_chunks, ignore_index=True) if len(cleaned_chunks) > 1 else cleaned_chunks[0]
            
            # Generate final report
            self.cleaning_report = self._generate_final_report(df, cleaned_df)
            
            # NEW: Learn from execution if Agent-First was used
            if use_intelligence and self.ml_context:
                quality_score = await self._evaluate_cleaning_quality(cleaned_df)
                self.adaptive_templates.learn_from_execution(
                    context={
                        'problem_type': self.ml_context.problem_type,
                        'n_samples': len(df),
                        'n_features': len(df.columns)
                    },
                    config=cleaning_config,
                    performance={'quality_score': quality_score}
                )
                logger.info(f"📚 Learned from execution (quality: {quality_score:.2f})")
            
            # Save configuration if enabled
            if self.config.save_yaml_config:
                self._save_yaml_config()
            
            # Check cost limits
            if self.config.track_usage and self.total_cost > self.config.max_cost_per_dataset:
                logger.warning(f"Cost limit exceeded: ${self.total_cost:.2f}")
            
            elapsed_time = time.time() - self.start_time
            logger.info(f"Cleaning completed in {elapsed_time:.2f} seconds")
            
            return cleaned_df, self.cleaning_report
            
        except Exception as e:
            logger.error(f"Error in cleaning pipeline: {e}")
            # Fallback to basic cleaning
            return await self._fallback_cleaning(df, user_context)
    
    async def _process_chunk(self, df_chunk: pd.DataFrame, chunk_id: int) -> pd.DataFrame:
        """
        Process a single chunk through all agents with retry logic
        """
        max_retries = self.config.max_retries
        retry_delay = self.config.retry_delay
        
        for attempt in range(max_retries):
            try:
                # Step 1: Profile the data
                logger.info(f"[Chunk {chunk_id}] Step 1: Profiling data...")
                start_time = time.time()
                profile_report = await self.profiler.analyze(df_chunk)
                self.performance_metrics["cleaning_time_per_agent"]["profiler"] = time.time() - start_time
                self.performance_metrics["total_api_calls"] += 1
                
                # Add ML context to profile if available
                if self.ml_context:
                    profile_report["ml_context"] = {
                        "problem_type": self.ml_context.problem_type,
                        "confidence": self.ml_context.confidence
                    }
                
                self.execution_history.append({
                    "step": "profiling",
                    "chunk": chunk_id,
                    "timestamp": datetime.now().isoformat(),
                    "report": profile_report,
                    "duration": self.performance_metrics["cleaning_time_per_agent"]["profiler"]
                })
                
                # Step 2: Validate against sector standards
                logger.info(f"[Chunk {chunk_id}] Step 2: Validating against sector standards...")
                start_time = time.time()
                validation_task = asyncio.create_task(
                    self.validator.validate(df_chunk, profile_report)
                )
                
                # Step 3: Clean data based on profile and validation
                validation_report = await validation_task
                self.performance_metrics["cleaning_time_per_agent"]["validator"] = time.time() - start_time
                self.performance_metrics["total_api_calls"] += 1
                
                self.validation_sources.extend(validation_report.get("sources", []))
                
                # Track validation success
                if validation_report.get("valid", False):
                    self.performance_metrics["validation_success_rate"] += 1
                
                logger.info(f"[Chunk {chunk_id}] Step 3: Applying intelligent cleaning...")
                start_time = time.time()
                cleaned_df, transformations = await self.cleaner.clean(
                    df_chunk, 
                    profile_report, 
                    validation_report
                )
                self.performance_metrics["cleaning_time_per_agent"]["cleaner"] = time.time() - start_time
                self.performance_metrics["total_api_calls"] += 1
                
                self.transformations_applied.extend(transformations)
                
                # Step 4: Final validation
                logger.info(f"[Chunk {chunk_id}] Step 4: Final quality control...")
                start_time = time.time()
                control_report = await self.controller.validate(
                    cleaned_df,
                    df_chunk,
                    transformations
                )
                self.performance_metrics["cleaning_time_per_agent"]["controller"] = time.time() - start_time
                self.performance_metrics["total_api_calls"] += 1
                
                self.execution_history.append({
                    "step": "complete",
                    "chunk": chunk_id,
                    "timestamp": datetime.now().isoformat(),
                    "control_report": control_report,
                    "total_duration": sum(self.performance_metrics["cleaning_time_per_agent"].values())
                })
                
                # Estimate cost
                self._update_cost_estimate()
                
                return cleaned_df
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for chunk {chunk_id}: {e}")
                self.performance_metrics["retry_count"] += 1
                
                if attempt < max_retries - 1:
                    # Exponential backoff
                    if self.config.exponential_backoff:
                        wait_time = retry_delay * (2 ** attempt)
                    else:
                        wait_time = retry_delay
                    
                    logger.info(f"Retrying in {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"All retries exhausted for chunk {chunk_id}")
                    # Return original chunk if all retries fail
                    return df_chunk
    
    async def _evaluate_cleaning_quality(self, df: pd.DataFrame) -> float:
        """Evaluate the quality of cleaned data"""
        try:
            # Basic quality metrics
            missing_ratio = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
            duplicate_ratio = df.duplicated().mean()
            
            # Calculate quality score
            quality_score = 100.0
            quality_score -= missing_ratio * 50  # Penalize missing values
            quality_score -= duplicate_ratio * 30  # Penalize duplicates
            
            # Check for constant columns
            constant_cols = sum(1 for col in df.columns if df[col].nunique() == 1)
            quality_score -= constant_cols * 5
            
            return max(0, min(100, quality_score))
        except:
            return 50.0  # Default middle score if evaluation fails
    
    def _needs_chunking(self, df: pd.DataFrame) -> bool:
        """Check if dataset needs chunking"""
        memory_usage_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
        return memory_usage_mb > self.config.chunk_size_mb
    
    def _chunk_dataset(self, df: pd.DataFrame) -> List[pd.DataFrame]:
        """Split dataset into chunks"""
        memory_usage_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
        n_chunks = int(np.ceil(memory_usage_mb / self.config.chunk_size_mb))
        
        logger.info(f"Splitting dataset ({memory_usage_mb:.2f} MB) into {n_chunks} chunks")
        
        chunks = np.array_split(df, n_chunks)
        return chunks
    
    def _update_cost_estimate(self):
        """Update cost estimate based on API usage"""
        # Rough estimation based on GPT-4 pricing
        tokens_per_call = 1000
        cost_per_1k_tokens_input = 0.03
        cost_per_1k_tokens_output = 0.06
        
        estimated_tokens = self.performance_metrics["total_api_calls"] * tokens_per_call
        self.performance_metrics["total_tokens_used"] = estimated_tokens
        
        # Estimate cost (input + output)
        estimated_cost = (estimated_tokens / 1000) * (cost_per_1k_tokens_input + cost_per_1k_tokens_output) / 2
        self.total_cost = min(estimated_cost, self.config.max_cost_per_dataset)
        
        if self.total_cost >= self.config.max_cost_per_dataset * 0.8:
            logger.warning(f"Approaching cost limit: ${self.total_cost:.2f} / ${self.config.max_cost_per_dataset:.2f}")
    
    def _generate_final_report(self, original_df: pd.DataFrame, cleaned_df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive cleaning report with performance metrics"""
        # Calculate validation success rate
        if len(self.execution_history) > 0:
            total_validations = sum(1 for h in self.execution_history if "validation" in h.get("step", ""))
            if total_validations > 0:
                self.performance_metrics["validation_success_rate"] = (
                    self.performance_metrics["validation_success_rate"] / total_validations
                ) * 100
        
        report = {
            "metadata": {
                "processing_date": datetime.now().isoformat(),
                "industry": self.config.user_context.get("secteur_activite"),
                "target_variable": self.config.user_context.get("target_variable"),
                "original_shape": original_df.shape,
                "cleaned_shape": cleaned_df.shape,
                "execution_time": time.time() - self.start_time if self.start_time else 0,
                "total_cost": self.total_cost,
                "agent_first_used": self.performance_metrics["intelligence_used"]
            },
            "transformations": self.transformations_applied,
            "validation_sources": list(set(self.validation_sources)),
            "quality_metrics": {
                "rows_removed": len(original_df) - len(cleaned_df),
                "columns_removed": len(original_df.columns) - len(cleaned_df.columns),
                "missing_before": original_df.isnull().sum().sum(),
                "missing_after": cleaned_df.isnull().sum().sum(),
                "duplicates_removed": original_df.duplicated().sum() - cleaned_df.duplicated().sum()
            },
            "performance_metrics": self.performance_metrics,
            "execution_history": self.execution_history
        }
        
        # Add ML context if detected
        if self.ml_context:
            report["ml_context"] = {
                "problem_type": self.ml_context.problem_type,
                "confidence": self.ml_context.confidence,
                "business_sector": self.ml_context.business_sector,
                "reasoning": self.ml_context.reasoning
            }
        
        # Add column-specific changes
        column_changes = []
        for col in original_df.columns:
            if col in cleaned_df.columns:
                before_dtype = str(original_df[col].dtype)
                after_dtype = str(cleaned_df[col].dtype)
                if before_dtype != after_dtype:
                    column_changes.append({
                        "column": col,
                        "before_dtype": before_dtype,
                        "after_dtype": after_dtype
                    })
        
        report["column_changes"] = column_changes
        
        # Save report if configured
        if self.config.save_reports:
            report_path = Path(self.config.output_dir) / f"cleaning_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_path, "w") as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Report saved to {report_path}")
        
        return report
    
    def _save_yaml_config(self):
        """Save cleaning configuration to YAML"""
        from .yaml_config_handler import YAMLConfigHandler
        
        handler = YAMLConfigHandler()
        
        # Add ML context to user context if available
        user_context = self.config.user_context.copy()
        if self.ml_context:
            user_context["detected_problem_type"] = self.ml_context.problem_type
            user_context["ml_confidence"] = self.ml_context.confidence
        
        yaml_path = handler.save_cleaning_config(
            transformations=self.transformations_applied,
            validation_sources=self.validation_sources,
            user_context=user_context,
            metrics={
                "initial_quality": self.cleaning_report.get("metadata", {}).get("initial_quality", 0),
                "final_quality": self.cleaning_report.get("metadata", {}).get("final_quality", 0),
                "agent_first_used": self.performance_metrics["intelligence_used"]
            }
        )
        
        logger.info(f"YAML configuration saved to {yaml_path}")
    
    async def _fallback_cleaning(self, df: pd.DataFrame, user_context: Dict) -> Tuple[pd.DataFrame, Dict]:
        """Fallback to traditional cleaning if agents fail"""
        logger.warning("Falling back to traditional data cleaning")
        
        # Import the existing data prep module
        from automl_platform.data_prep import EnhancedDataPreprocessor
        
        # Create traditional preprocessor
        prep_config = {
            'handle_outliers': True,
            'outlier_method': 'iqr',
            'scaling_method': 'robust',
            'enable_quality_checks': True,
            'enable_drift_detection': False
        }
        
        preprocessor = EnhancedDataPreprocessor(prep_config)
        
        # Basic cleaning
        cleaned_df = df.copy()
        
        # Remove duplicates
        cleaned_df = cleaned_df.drop_duplicates()
        
        # Handle missing values
        for col in cleaned_df.columns:
            if cleaned_df[col].dtype == 'object':
                cleaned_df[col].fillna('missing', inplace=True)
            else:
                cleaned_df[col].fillna(cleaned_df[col].median(), inplace=True)
        
        # Generate basic report
        report = {
            "metadata": {
                "fallback": True,
                "reason": "Agent processing failed",
                "processing_date": datetime.now().isoformat()
            },
            "quality_metrics": {
                "rows_removed": len(df) - len(cleaned_df),
                "duplicates_removed": df.duplicated().sum()
            }
        }
        
        return cleaned_df, report
    
    def estimate_cost(self, df: pd.DataFrame) -> float:
        """Estimate cleaning cost based on dataset size"""
        # Rough estimation based on tokens
        estimated_tokens = (df.shape[0] * df.shape[1] * 10) / 1000  # Approximate
        
        # GPT-4 pricing (approximate)
        cost_per_1k_tokens = 0.03  # Input
        estimated_cost = estimated_tokens * cost_per_1k_tokens * 4  # 4 agents
        
        return min(estimated_cost, self.config.max_cost_per_dataset)
    
    async def validate_only(self, df: pd.DataFrame, user_context: Dict) -> Dict[str, Any]:
        """Run only validation without cleaning"""
        self.config.user_context.update(user_context)
        
        # Profile first
        profile_report = await self.profiler.analyze(df)
        
        # Then validate
        validation_report = await self.validator.validate(df, profile_report)
        
        return validation_report
    
    async def get_cleaning_suggestions(self, df: pd.DataFrame, user_context: Dict) -> List[Dict]:
        """Get cleaning suggestions without applying them"""
        self.config.user_context.update(user_context)
        
        # Profile the data
        profile_report = await self.profiler.analyze(df)
        
        # Get suggestions from cleaner
        suggestions = await self.cleaner.suggest_transformations(df, profile_report)
        
        return suggestions
    
    async def detect_ml_context(self, df: pd.DataFrame, target_col: Optional[str] = None) -> Dict[str, Any]:
        """
        Detect ML context for a dataset (Agent-First feature)
        
        Args:
            df: Input dataframe
            target_col: Target column name
            
        Returns:
            Dictionary with detected ML context
        """
        context = await self.context_detector.detect_ml_context(df, target_col)
        
        return {
            "problem_type": context.problem_type,
            "confidence": context.confidence,
            "business_sector": context.business_sector,
            "temporal_aspect": context.temporal_aspect,
            "imbalance_detected": context.imbalance_detected,
            "reasoning": context.reasoning,
            "alternatives": context.alternative_interpretations
        }
