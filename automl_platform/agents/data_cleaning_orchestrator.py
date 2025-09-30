"""
Data Cleaning Orchestrator - Main coordination for OpenAI agents
Integrated with Agent-First intelligent modules
NOW WITH CLAUDE SDK FOR STRATEGIC CLEANING DECISIONS
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

import importlib.util
_anthropic_spec = importlib.util.find_spec("anthropic")
if _anthropic_spec is not None:
    from anthropic import AsyncAnthropic
else:
    AsyncAnthropic = None

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
    ENHANCED WITH CLAUDE SDK FOR STRATEGIC DECISIONS
    """
    
    def __init__(self, config: Optional[AgentConfig] = None, automl_config: Optional[Dict] = None, use_claude: bool = True):
        """
        Initialize orchestrator with configuration
        
        Args:
            config: Agent configuration
            automl_config: AutoML platform configuration
            use_claude: Whether to use Claude SDK for strategic decisions
        """
        self.config = config or AgentConfig()
        self.automl_config = automl_config or {}
        self.use_claude = use_claude and AsyncAnthropic is not None
        
        # Validate configuration
        self.config.validate()
        
        # Initialize agents
        self.profiler = ProfilerAgent(self.config)
        self.validator = ValidatorAgent(self.config)
        self.cleaner = CleanerAgent(self.config)
        self.controller = ControllerAgent(self.config)
        
        # Initialize intelligent modules (Agent-First)
        self.context_detector = IntelligentContextDetector()
        self.config_generator = IntelligentConfigGenerator(use_claude=use_claude)
        self.adaptive_templates = AdaptiveTemplateSystem()
        
        # Initialize Claude client if available
        if self.use_claude:
            self.claude_client = AsyncAnthropic()
            self.claude_model = "claude-sonnet-4-20250514"
            logger.info("💎 Claude SDK enabled for strategic cleaning decisions")
        else:
            self.claude_client = None
            if use_claude:
                logger.warning("⚠️ Claude SDK requested but not available")
            else:
                logger.info("📋 Using rule-based cleaning orchestration")
        
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
            "intelligence_used": False,
            "claude_decisions": 0
        }
        
        # Results storage
        self.cleaning_report = {}
        self.validation_sources = []
        self.transformations_applied = []
        self.ml_context = None
        
        # Setup logging with file handler
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging configuration with file output"""
        log_file = self.config.log_file or "./logs/automl.log"
        log_dir = Path(log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
        )
        
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,
            backupCount=5
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(getattr(logging, self.config.log_level))
        
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.INFO)
        
        logger = logging.getLogger(__name__)
        logger.setLevel(getattr(logging, self.config.log_level))
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    async def determine_cleaning_mode_with_claude(
        self,
        df: pd.DataFrame,
        user_context: Dict[str, Any],
        ml_context: Any
    ) -> Dict[str, Any]:
        """
        Use Claude to intelligently determine the best cleaning mode
        STRATEGIC DECISION MAKING WITH CLAUDE
        """
        if not self.use_claude:
            # Fallback to rule-based
            return await self._determine_best_mode_rule_based(df, user_context, ml_context)
        
        logger.info("💎 Using Claude to determine optimal cleaning mode...")
        self.performance_metrics["claude_decisions"] += 1
        
        # Prepare context for Claude
        data_summary = {
            'shape': df.shape,
            'missing_ratio': float(df.isnull().sum().sum() / (df.shape[0] * df.shape[1])),
            'duplicate_ratio': float(df.duplicated().mean()),
            'numeric_cols': int(len(df.select_dtypes(include=[np.number]).columns)),
            'categorical_cols': int(len(df.select_dtypes(include=['object']).columns)),
            'high_cardinality_cols': int(sum(1 for col in df.columns if df[col].nunique() > 20)),
        }
        
        prompt = f"""Analyze this data cleaning scenario and recommend the best approach.

Data Summary:
{json.dumps(data_summary, indent=2)}

ML Context:
- Problem Type: {ml_context.problem_type if ml_context else 'unknown'}
- Business Sector: {ml_context.business_sector if ml_context else 'unknown'}
- Confidence: {ml_context.confidence if ml_context else 0}

User Context:
{json.dumps(user_context, indent=2)}

Available Cleaning Modes:
1. AUTOMATED: Agents make all decisions automatically (fastest, least control)
2. INTERACTIVE: User approves each transformation (slowest, most control)
3. HYBRID: Critical decisions need approval, routine ones automated

Consider:
- Data quality issues (missing: {data_summary['missing_ratio']:.1%}, duplicates: {data_summary['duplicate_ratio']:.1%})
- Business criticality of the problem
- Need for compliance/auditability
- Time/resource constraints

Respond ONLY with valid JSON:
{{
  "recommended_mode": "automated|interactive|hybrid",
  "confidence": 0.0-1.0,
  "reasoning": "detailed explanation why this mode",
  "key_considerations": ["point1", "point2", "point3"],
  "estimated_time_minutes": number,
  "risk_level": "low|medium|high"
}}"""
        
        try:
            response = await self.claude_client.messages.create(
                model=self.claude_model,
                max_tokens=1500,
                system="You are an expert data engineer helping choose optimal data cleaning strategies.",
                messages=[{"role": "user", "content": prompt}]
            )
            
            response_text = response.content[0].text.strip()
            decision = json.loads(response_text)
            
            logger.info(f"💎 Claude recommends: {decision['recommended_mode']} mode")
            logger.info(f"   Confidence: {decision['confidence']:.1%}")
            logger.info(f"   Reasoning: {decision['reasoning'][:150]}...")
            
            return decision
            
        except Exception as e:
            logger.warning(f"⚠️ Claude mode determination failed: {e}, using rule-based fallback")
            return await self._determine_best_mode_rule_based(df, user_context, ml_context)
    
    async def _determine_best_mode_rule_based(
        self,
        df: pd.DataFrame,
        user_context: Dict[str, Any],
        ml_context: Any
    ) -> Dict[str, Any]:
        """Fallback rule-based mode determination"""
        missing_ratio = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
        
        # Critical sectors require more oversight
        critical_sectors = ['finance', 'healthcare', 'banking', 'insurance']
        is_critical = user_context.get('secteur_activite', '').lower() in critical_sectors
        
        if is_critical or missing_ratio > 0.3:
            mode = 'interactive'
            estimated_time = 25
            risk = 'high'
        elif missing_ratio > 0.15 or len(df) > 100000:
            mode = 'hybrid'
            estimated_time = 15
            risk = 'medium'
        else:
            mode = 'automated'
            estimated_time = 8
            risk = 'low'
        
        return {
            'recommended_mode': mode,
            'confidence': 0.7,
            'reasoning': f"Rule-based: {risk} risk sector, {missing_ratio:.1%} missing data",
            'key_considerations': ['Data quality', 'Sector criticality', 'Dataset size'],
            'estimated_time_minutes': estimated_time,
            'risk_level': risk
        }
    
    async def recommend_cleaning_approach_with_claude(
        self,
        df: pd.DataFrame,
        profile_report: Dict[str, Any],
        ml_context: Any
    ) -> str:
        """
        Generate rich cleaning recommendations with Claude
        STRATEGIC GUIDANCE FOR USERS
        """
        if not self.use_claude:
            return self._generate_basic_recommendation(df, profile_report, ml_context)
        
        logger.info("💎 Generating cleaning recommendations with Claude...")
        self.performance_metrics["claude_decisions"] += 1
        
        prompt = f"""Generate actionable data cleaning recommendations for this ML project.

Data Profile:
{json.dumps(profile_report, indent=2, default=str)[:2000]}

ML Context:
- Problem: {ml_context.problem_type if ml_context else 'unknown'}
- Sector: {ml_context.business_sector if ml_context else 'unknown'}
- Confidence: {ml_context.confidence if ml_context else 0:.1%}

Provide a structured recommendation covering:
1. Most critical issues to address first (prioritized)
2. Recommended cleaning strategy with trade-offs
3. Potential risks and how to mitigate them
4. Expected impact on model performance

Be specific, actionable, and concise (3-4 paragraphs max)."""
        
        try:
            response = await self.claude_client.messages.create(
                model=self.claude_model,
                max_tokens=1000,
                system="You are an expert data engineer providing cleaning guidance.",
                messages=[{"role": "user", "content": prompt}]
            )
            
            recommendation = response.content[0].text.strip()
            logger.info("💎 Generated rich cleaning recommendations")
            return recommendation
            
        except Exception as e:
            logger.warning(f"⚠️ Claude recommendation failed: {e}")
            return self._generate_basic_recommendation(df, profile_report, ml_context)
    
    def _generate_basic_recommendation(
        self,
        df: pd.DataFrame,
        profile_report: Dict[str, Any],
        ml_context: Any
    ) -> str:
        """Fallback basic recommendation"""
        missing_ratio = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
        
        rec = f"Data Cleaning Recommendation:\n\n"
        rec += f"Dataset: {df.shape[0]} rows, {df.shape[1]} columns\n"
        rec += f"Missing data: {missing_ratio:.1%}\n"
        rec += f"Problem type: {ml_context.problem_type if ml_context else 'unknown'}\n\n"
        
        if missing_ratio > 0.2:
            rec += "Priority: Address high missing data ratio with advanced imputation.\n"
        
        rec += "Approach: Automated cleaning with agent-driven decisions.\n"
        
        return rec
    
    async def clean_dataset(
        self, 
        df: pd.DataFrame, 
        user_context: Dict[str, Any],
        cleaning_config: Optional[Dict] = None,
        use_intelligence: bool = True
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Main pipeline for intelligent data cleaning
        NOW ENHANCED WITH CLAUDE FOR STRATEGIC DECISIONS
        
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
        logger.info(f"Claude enhancement: {'ENABLED' if self.use_claude else 'DISABLED'}")
        
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
                
                # CLAUDE: Determine best cleaning mode
                if self.use_claude:
                    mode_decision = await self.determine_cleaning_mode_with_claude(
                        df, user_context, self.ml_context
                    )
                    logger.info(f"💎 Cleaning mode: {mode_decision['recommended_mode']}")
                    logger.info(f"   {mode_decision['reasoning']}")
                    
                    # Get cleaning recommendations
                    profile_report = await self.profiler.analyze(df)
                    recommendations = await self.recommend_cleaning_approach_with_claude(
                        df, profile_report, self.ml_context
                    )
                    logger.info(f"💎 Recommendations:\n{recommendations}")
                
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
            logger.info(f"Claude decisions made: {self.performance_metrics['claude_decisions']}")
            
            return cleaned_df, self.cleaning_report
            
        except Exception as e:
            logger.error(f"Error in cleaning pipeline: {e}")
            return await self._fallback_cleaning(df, user_context)
    
    async def _process_chunk(self, df_chunk: pd.DataFrame, chunk_id: int) -> pd.DataFrame:
        """Process a single chunk through all agents with retry logic"""
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
                    if self.config.exponential_backoff:
                        wait_time = retry_delay * (2 ** attempt)
                    else:
                        wait_time = retry_delay
                    
                    logger.info(f"Retrying in {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"All retries exhausted for chunk {chunk_id}")
                    return df_chunk
    
    async def _evaluate_cleaning_quality(self, df: pd.DataFrame) -> float:
        """Evaluate the quality of cleaned data"""
        try:
            missing_ratio = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
            duplicate_ratio = df.duplicated().mean()
            
            quality_score = 100.0
            quality_score -= missing_ratio * 50
            quality_score -= duplicate_ratio * 30
            
            constant_cols = sum(1 for col in df.columns if df[col].nunique() == 1)
            quality_score -= constant_cols * 5
            
            return max(0, min(100, quality_score))
        except:
            return 50.0
    
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
        tokens_per_call = 1000
        cost_per_1k_tokens_input = 0.03
        cost_per_1k_tokens_output = 0.06
        
        estimated_tokens = self.performance_metrics["total_api_calls"] * tokens_per_call
        self.performance_metrics["total_tokens_used"] = estimated_tokens
        
        estimated_cost = (estimated_tokens / 1000) * (cost_per_1k_tokens_input + cost_per_1k_tokens_output) / 2
        self.total_cost = min(estimated_cost, self.config.max_cost_per_dataset)
        
        if self.total_cost >= self.config.max_cost_per_dataset * 0.8:
            logger.warning(f"Approaching cost limit: ${self.total_cost:.2f} / ${self.config.max_cost_per_dataset:.2f}")
    
    def _generate_final_report(self, original_df: pd.DataFrame, cleaned_df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive cleaning report with performance metrics"""
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
                "agent_first_used": self.performance_metrics["intelligence_used"],
                "claude_enhanced": self.use_claude,
                "claude_decisions": self.performance_metrics["claude_decisions"]
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
        
        if self.ml_context:
            report["ml_context"] = {
                "problem_type": self.ml_context.problem_type,
                "confidence": self.ml_context.confidence,
                "business_sector": self.ml_context.business_sector,
                "reasoning": self.ml_context.reasoning
            }
        
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
                "agent_first_used": self.performance_metrics["intelligence_used"],
                "claude_enhanced": self.use_claude
            }
        )
        
        logger.info(f"YAML configuration saved to {yaml_path}")
    
    async def _fallback_cleaning(self, df: pd.DataFrame, user_context: Dict) -> Tuple[pd.DataFrame, Dict]:
        """Fallback to traditional cleaning if agents fail"""
        logger.warning("Falling back to traditional data cleaning")
        
        from automl_platform.data_prep import EnhancedDataPreprocessor
        
        prep_config = {
            'handle_outliers': True,
            'outlier_method': 'iqr',
            'scaling_method': 'robust',
            'enable_quality_checks': True,
            'enable_drift_detection': False
        }
        
        preprocessor = EnhancedDataPreprocessor(prep_config)
        
        cleaned_df = df.copy()
        cleaned_df = cleaned_df.drop_duplicates()
        
        for col in cleaned_df.columns:
            if cleaned_df[col].dtype == 'object':
                cleaned_df[col].fillna('missing', inplace=True)
            else:
                cleaned_df[col].fillna(cleaned_df[col].median(), inplace=True)
        
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
        estimated_tokens = (df.shape[0] * df.shape[1] * 10) / 1000
        cost_per_1k_tokens = 0.03
        estimated_cost = estimated_tokens * cost_per_1k_tokens * 4
        
        return min(estimated_cost, self.config.max_cost_per_dataset)
    
    async def validate_only(self, df: pd.DataFrame, user_context: Dict) -> Dict[str, Any]:
        """Run only validation without cleaning"""
        self.config.user_context.update(user_context)
        
        profile_report = await self.profiler.analyze(df)
        validation_report = await self.validator.validate(df, profile_report)
        
        return validation_report
    
    async def get_cleaning_suggestions(self, df: pd.DataFrame, user_context: Dict) -> List[Dict]:
        """Get cleaning suggestions without applying them"""
        self.config.user_context.update(user_context)
        
        profile_report = await self.profiler.analyze(df)
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
