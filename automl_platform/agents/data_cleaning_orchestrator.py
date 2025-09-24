"""
Data Cleaning Orchestrator - Main coordination for OpenAI agents
"""

import pandas as pd
import numpy as np
import asyncio
import logging
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

logger = logging.getLogger(__name__)


class DataCleaningOrchestrator:
    """
    Orchestrates the intelligent data cleaning process using OpenAI agents
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
        
        # Tracking
        self.execution_history = []
        self.total_cost = 0.0
        self.start_time = None
        
        # Results storage
        self.cleaning_report = {}
        self.validation_sources = []
        self.transformations_applied = []
        
        # Setup logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging configuration"""
        if self.config.log_file:
            logging.basicConfig(
                level=getattr(logging, self.config.log_level),
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.FileHandler(self.config.log_file),
                    logging.StreamHandler()
                ]
            )
    
    async def clean_dataset(
        self, 
        df: pd.DataFrame, 
        user_context: Dict[str, Any],
        cleaning_config: Optional[Dict] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Main pipeline for intelligent data cleaning
        
        Args:
            df: Input dataframe
            user_context: User context (sector, target variable, etc.)
            cleaning_config: Optional cleaning configuration
            
        Returns:
            Tuple of (cleaned_dataframe, cleaning_report)
        """
        self.start_time = time.time()
        
        # Update user context
        self.config.user_context.update(user_context)
        
        logger.info(f"Starting intelligent cleaning for dataset with shape {df.shape}")
        logger.info(f"User context: {user_context}")
        
        try:
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
        Process a single chunk through all agents
        """
        try:
            # Step 1: Profile the data
            logger.info(f"[Chunk {chunk_id}] Step 1: Profiling data...")
            profile_report = await self.profiler.analyze(df_chunk)
            self.execution_history.append({
                "step": "profiling",
                "chunk": chunk_id,
                "timestamp": datetime.now().isoformat(),
                "report": profile_report
            })
            
            # Step 2: Validate against sector standards (can run in parallel)
            logger.info(f"[Chunk {chunk_id}] Step 2: Validating against sector standards...")
            validation_task = asyncio.create_task(
                self.validator.validate(df_chunk, profile_report)
            )
            
            # Step 3: Clean data based on profile and validation
            validation_report = await validation_task
            self.validation_sources.extend(validation_report.get("sources", []))
            
            logger.info(f"[Chunk {chunk_id}] Step 3: Applying intelligent cleaning...")
            cleaned_df, transformations = await self.cleaner.clean(
                df_chunk, 
                profile_report, 
                validation_report
            )
            self.transformations_applied.extend(transformations)
            
            # Step 4: Final validation
            logger.info(f"[Chunk {chunk_id}] Step 4: Final quality control...")
            control_report = await self.controller.validate(
                cleaned_df,
                df_chunk,
                transformations
            )
            
            self.execution_history.append({
                "step": "complete",
                "chunk": chunk_id,
                "timestamp": datetime.now().isoformat(),
                "control_report": control_report
            })
            
            return cleaned_df
            
        except Exception as e:
            logger.error(f"Error processing chunk {chunk_id}: {e}")
            # Return original chunk if processing fails
            return df_chunk
    
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
    
    def _generate_final_report(self, original_df: pd.DataFrame, cleaned_df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive cleaning report"""
        report = {
            "metadata": {
                "processing_date": datetime.now().isoformat(),
                "industry": self.config.user_context.get("secteur_activite"),
                "target_variable": self.config.user_context.get("target_variable"),
                "original_shape": original_df.shape,
                "cleaned_shape": cleaned_df.shape,
                "execution_time": time.time() - self.start_time if self.start_time else 0,
                "total_cost": self.total_cost
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
            "execution_history": self.execution_history
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
        # Format transformations properly
        formatted_transformations = []
        for t in self.transformations_applied:
            transformation = {
                "column": t.get("column"),
                "action": t.get("action"),
                "params": t.get("params", {})
            }
            # Add rationale if present
            if "rationale" in t:
                transformation["rationale"] = t["rationale"]
            formatted_transformations.append(transformation)
        
        # Format validation sources
        validation_sources = list(set(self.validation_sources))
        if not validation_sources:
            validation_sources = ["Standards sectoriels identifiés automatiquement"]
        
        config_data = {
            "metadata": {
                "industry": self.config.user_context.get("secteur_activite", "general"),
                "target_variable": self.config.user_context.get("target_variable", "unknown"),
                "processing_date": datetime.now().strftime("%Y-%m-%d"),
                "business_context": self.config.user_context.get("contexte_metier", ""),
                "language": self.config.user_context.get("language", "fr")
            },
            "transformations": formatted_transformations,
            "validation_sources": validation_sources,
            "quality_metrics": {
                "initial_score": self.cleaning_report.get("metadata", {}).get("initial_quality", 0),
                "final_score": self.cleaning_report.get("metadata", {}).get("final_quality", 0),
                "improvement": self.cleaning_report.get("metadata", {}).get("quality_improvement", 0)
            },
            "execution_summary": {
                "total_transformations": len(formatted_transformations),
                "execution_time_seconds": self.cleaning_report.get("metadata", {}).get("execution_time", 0),
                "cost_estimate": self.total_cost
            }
        }
        
        yaml_path = Path(self.config.output_dir) / f"cleaning_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
        
        # Ensure the directory exists
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(yaml_path, "w", encoding='utf-8') as f:
            yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        
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
