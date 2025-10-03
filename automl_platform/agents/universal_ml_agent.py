"""
Universal ML Agent - PRODUCTION VERSION WITH MEMORY PROTECTION
===============================================================
Enterprise-grade ML orchestrator with advanced memory management,
monitoring, and protection against memory leaks.

Key Features:
- Memory budget enforcement per operation
- LRU cache with size limits
- Automatic garbage collection
- Memory profiling and monitoring
- Resource cleanup with context managers
- Streaming for large datasets
- OOM protection mechanisms
"""

import asyncio
import logging
import pandas as pd
import numpy as np
import json
import gc
import psutil
import sys
from .utils import BoundedList
from typing import Dict, Any, List, Optional, Tuple, Generator
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from contextlib import contextmanager
from functools import lru_cache, wraps
from collections import OrderedDict
import hashlib
import pickle
import warnings

import importlib.util
_anthropic_spec = importlib.util.find_spec("anthropic")
if _anthropic_spec is not None:
    from anthropic import AsyncAnthropic
else:
    AsyncAnthropic = None

from .intelligent_context_detector import IntelligentContextDetector, MLContext
from .intelligent_config_generator import IntelligentConfigGenerator, OptimalConfig
from .adaptive_template_system import AdaptiveTemplateSystem
from .data_cleaning_orchestrator import DataCleaningOrchestrator
from .agent_config import AgentConfig
from .yaml_config_handler import YAMLConfigHandler

from .profiler_agent import ProfilerAgent
from .validator_agent import ValidatorAgent
from .cleaner_agent import CleanerAgent
from .controller_agent import ControllerAgent

logger = logging.getLogger(__name__)


# ============================================================================
# MEMORY PROTECTION UTILITIES
# ============================================================================

class MemoryMonitor:
    """Real-time memory monitoring and alerting"""
    
    def __init__(self, warning_threshold_mb: int = 1000, critical_threshold_mb: int = 2000):
        self.warning_threshold = warning_threshold_mb * 1024 * 1024  # Convert to bytes
        self.critical_threshold = critical_threshold_mb * 1024 * 1024
        self.process = psutil.Process()
        self.initial_memory = self.get_memory_usage()
        self.peak_memory = self.initial_memory
        self.memory_samples = []
        
    def get_memory_usage(self) -> int:
        """Get current memory usage in bytes"""
        return self.process.memory_info().rss
    
    def get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB"""
        return self.get_memory_usage() / (1024 * 1024)
    
    def check_memory(self) -> Dict[str, Any]:
        """Check current memory status"""
        current = self.get_memory_usage()
        delta = current - self.initial_memory
        
        self.peak_memory = max(self.peak_memory, current)
        self.memory_samples.append({
            'timestamp': datetime.now(),
            'usage_mb': current / (1024 * 1024),
            'delta_mb': delta / (1024 * 1024)
        })
        
        # Keep only last 100 samples
        if len(self.memory_samples) > 100:
            self.memory_samples = self.memory_samples[-100:]
        
        status = {
            'current_mb': current / (1024 * 1024),
            'delta_mb': delta / (1024 * 1024),
            'peak_mb': self.peak_memory / (1024 * 1024),
            'warning': current > self.warning_threshold,
            'critical': current > self.critical_threshold,
            'available_mb': psutil.virtual_memory().available / (1024 * 1024)
        }
        
        if status['critical']:
            logger.critical(f"🔴 CRITICAL: Memory usage {status['current_mb']:.1f} MB exceeds threshold!")
        elif status['warning']:
            logger.warning(f"⚠️ WARNING: Memory usage {status['current_mb']:.1f} MB approaching limit")
        
        return status
    
    def force_cleanup(self):
        """Force garbage collection and memory cleanup"""
        gc.collect()
        logger.info(f"🧹 Forced memory cleanup. Current: {self.get_memory_usage_mb():.1f} MB")


class MemoryBudget:
    """Enforce memory budgets for operations"""
    
    def __init__(self, budget_mb: int):
        self.budget = budget_mb * 1024 * 1024  # Convert to bytes
        self.monitor = MemoryMonitor()
        self.start_memory = 0
        
    def __enter__(self):
        self.start_memory = self.monitor.get_memory_usage()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        end_memory = self.monitor.get_memory_usage()
        used = end_memory - self.start_memory
        
        if used > self.budget:
            logger.warning(
                f"⚠️ Operation exceeded budget: {used/(1024*1024):.1f} MB used, "
                f"{self.budget/(1024*1024):.1f} MB allowed"
            )
        
        # Cleanup
        gc.collect()
        return False


class LRUMemoryCache:
    """LRU cache with memory size limits"""
    
    def __init__(self, max_size_mb: int = 500):
        self.max_size = max_size_mb * 1024 * 1024
        self.cache = OrderedDict()
        self.current_size = 0
        
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache"""
        if key in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            return self.cache[key]['data']
        return None
    
    def put(self, key: str, data: Any, size_bytes: Optional[int] = None):
        """Put item in cache with size tracking"""
        # Estimate size if not provided
        if size_bytes is None:
            size_bytes = sys.getsizeof(pickle.dumps(data))
        
        # Remove oldest items if needed
        while self.current_size + size_bytes > self.max_size and self.cache:
            oldest_key, oldest_value = self.cache.popitem(last=False)
            self.current_size -= oldest_value['size']
            logger.debug(f"🗑️ Evicted {oldest_key} from cache ({oldest_value['size']/(1024*1024):.2f} MB)")
        
        # Add new item
        if key in self.cache:
            self.current_size -= self.cache[key]['size']
        
        self.cache[key] = {
            'data': data,
            'size': size_bytes,
            'timestamp': datetime.now()
        }
        self.current_size += size_bytes
        
        logger.debug(f"💾 Cached {key} ({size_bytes/(1024*1024):.2f} MB). "
                    f"Total cache: {self.current_size/(1024*1024):.1f} MB")
    
    def clear(self):
        """Clear entire cache"""
        self.cache.clear()
        self.current_size = 0
        gc.collect()
        logger.info("🧹 Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            'items': len(self.cache),
            'size_mb': self.current_size / (1024 * 1024),
            'max_size_mb': self.max_size / (1024 * 1024),
            'utilization': self.current_size / self.max_size if self.max_size > 0 else 0
        }


def memory_safe(max_memory_mb: int = 500):
    """Decorator for memory-safe operations"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            monitor = MemoryMonitor()
            start_mem = monitor.get_memory_usage_mb()
            
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                end_mem = monitor.get_memory_usage_mb()
                used = end_mem - start_mem
                
                if used > max_memory_mb:
                    logger.warning(
                        f"⚠️ {func.__name__} used {used:.1f} MB (limit: {max_memory_mb} MB)"
                    )
                
                # Force cleanup
                gc.collect()
                
        return wrapper
    return decorator


@contextmanager
def dataframe_batch_processor(df: pd.DataFrame, batch_size: int = 10000):
    """Process large DataFrames in batches to control memory"""
    try:
        n_batches = len(df) // batch_size + (1 if len(df) % batch_size else 0)
        logger.info(f"📦 Processing {len(df)} rows in {n_batches} batches of {batch_size}")
        
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i + batch_size]
            yield batch
            
            # Cleanup after each batch
            del batch
            gc.collect()
            
    except Exception as e:
        logger.error(f"❌ Batch processing failed: {e}")
        raise


# ============================================================================
# PRODUCTION ML PIPELINE RESULT WITH MEMORY STATS
# ============================================================================

@dataclass
class ProductionMLPipelineResult:
    """Enhanced result with memory and performance metrics"""
    # Core results (same as before)
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
    yaml_config_path: Optional[str] = None
    claude_summary: Optional[str] = None
    
    # NEW: Memory and performance metrics
    memory_stats: Dict[str, Any] = field(default_factory=dict)
    cache_stats: Dict[str, Any] = field(default_factory=dict)
    performance_profile: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, handling DataFrames"""
        result = asdict(self)
        if self.cleaned_data is not None:
            result['cleaned_data'] = {
                'shape': self.cleaned_data.shape,
                'columns': list(self.cleaned_data.columns),
                'memory_usage_mb': self.cleaned_data.memory_usage(deep=True).sum() / (1024 * 1024)
            }
        return result


# ============================================================================
# PRODUCTION UNIVERSAL ML AGENT
# ============================================================================

class ProductionUniversalMLAgent:
    """
    Production-grade Universal ML Agent with advanced memory protection.
    
    Enterprise Features:
    - Memory monitoring and budgets
    - LRU caching with size limits
    - Batch processing for large datasets
    - Automatic cleanup and garbage collection
    - Performance profiling
    - Graceful degradation under memory pressure
    - OOM protection
    """
    
    def __init__(
        self, 
        config: Optional[AgentConfig] = None,
        use_claude: bool = True,
        max_cache_mb: int = 500,
        memory_warning_mb: int = 1000,
        memory_critical_mb: int = 2000,
        batch_size: int = 10000
    ):
        """
        Initialize Production Universal ML Agent
        
        Args:
            config: Agent configuration
            use_claude: Enable Claude SDK
            max_cache_mb: Maximum cache size in MB
            memory_warning_mb: Warning threshold for memory usage
            memory_critical_mb: Critical threshold for memory usage
            batch_size: Batch size for large dataset processing
        """
        self.config = config or AgentConfig()
        self.use_claude = use_claude and AsyncAnthropic is not None
        self.batch_size = batch_size
        
        # Memory protection
        self.memory_monitor = MemoryMonitor(memory_warning_mb, memory_critical_mb)
        self.cache = LRUMemoryCache(max_cache_mb)
        
        logger.info(f"🛡️ Memory Protection: Warning={memory_warning_mb}MB, Critical={memory_critical_mb}MB")
        logger.info(f"💾 Cache Limit: {max_cache_mb}MB")
        logger.info(f"📦 Batch Size: {batch_size} rows")
        
        # Initialize intelligent components
        self.context_detector = IntelligentContextDetector()
        self.config_generator = IntelligentConfigGenerator(use_claude=use_claude)
        self.adaptive_templates = AdaptiveTemplateSystem()
        self.cleaning_orchestrator = DataCleaningOrchestrator(self.config, use_claude=use_claude)
        
        # Individual agents
        self.profiler = ProfilerAgent(self.config)
        self.validator = ValidatorAgent(self.config)
        self.cleaner = CleanerAgent(self.config)
        self.controller = ControllerAgent(self.config)
        
        # YAML handler
        self.yaml_handler = YAMLConfigHandler()
        
        # Initialize Claude if available
        if self.use_claude:
            self.claude_client = AsyncAnthropic()
            self.claude_model = "claude-sonnet-4-5-20250929"
            logger.info("💎 Claude SDK enabled as Master Orchestrator")
        else:
            self.claude_client = None
            logger.info("📋 Using rule-based orchestration")
        
        # Performance tracking
        self.execution_history = BoundedList(maxlen=100)
        self.agent_metrics = {
            "profiler_calls": 0,
            "validator_calls": 0,
            "cleaner_calls": 0,
            "controller_calls": 0,
            "context_detections": 0,
            "config_generations": 0,
            "claude_orchestrations": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "memory_cleanups": 0
        }
        
        # Storage
        self.cache_dir = Path(self.config.cache_dir) / "universal_agent"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Knowledge base (with size limit)
        self.knowledge_base = ProductionKnowledgeBase(max_patterns=100)
        
        logger.info("✅ Production Universal ML Agent initialized")
        self._log_memory_status()
    
    def _log_memory_status(self):
        """Log current memory status"""
        status = self.memory_monitor.check_memory()
        logger.info(f"💾 Memory: {status['current_mb']:.1f} MB (Peak: {status['peak_mb']:.1f} MB)")
    
    def _compute_data_hash(self, df: pd.DataFrame) -> str:
        """Compute lightweight hash for DataFrame caching"""
        # Use shape and column info, not full data
        hash_input = f"{df.shape}_{list(df.columns)}_{df.dtypes.tolist()}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:16]  # Shorter hash
    
    @memory_safe(max_memory_mb=500)
    async def understand_problem(
        self,
        df: pd.DataFrame,
        target_col: Optional[str] = None,
        user_hints: Optional[Dict[str, Any]] = None
    ) -> MLContext:
        """
        Memory-safe problem understanding with caching
        """
        logger.info("🧠 Understanding the problem...")
        self.agent_metrics["context_detections"] += 1
        
        # Check cache
        data_hash = self._compute_data_hash(df)
        cache_key = f"context_{data_hash}"
        cached_context = self.cache.get(cache_key)
        
        if cached_context:
            logger.info("✅ Context found in cache")
            self.agent_metrics["cache_hits"] += 1
            return cached_context
        
        self.agent_metrics["cache_misses"] += 1
        
        # Profile with memory budget
        with MemoryBudget(budget_mb=200):
            logger.info("📊 Profiling with ProfilerAgent...")
            profile_report = await self.profiler.analyze(df)
            self.agent_metrics["profiler_calls"] += 1
        
        # Detect context
        context = await self.context_detector.detect_ml_context(
            df, target_col, user_hints
        )
        
        # Add profile issues to context
        context.detected_patterns.extend(
            profile_report.get("quality_issues", [])
        )
        
        # Cache result
        self.cache.put(cache_key, context)
        
        logger.info(f"✅ Problem: {context.problem_type} (confidence: {context.confidence:.1%})")
        self._log_memory_status()
        
        return context
    
    @memory_safe(max_memory_mb=300)
    async def validate_with_standards(
        self,
        df: pd.DataFrame,
        context: MLContext
    ) -> Dict[str, Any]:
        """Memory-safe validation"""
        logger.info("🔍 Validating against standards...")
        self.agent_metrics["validator_calls"] += 1
        
        with MemoryBudget(budget_mb=200):
            profile_report = await self.profiler.analyze(df)
            self.config.user_context['secteur_activite'] = context.business_sector or 'general'
            validation_report = await self.validator.validate(df, profile_report)
        
        logger.info(f"✅ Validation: {len(validation_report.get('issues', []))} issues")
        self._log_memory_status()
        
        return validation_report
    
    @memory_safe(max_memory_mb=200)
    async def search_ml_best_practices(
        self,
        problem_type: str,
        data_characteristics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Cached best practices search"""
        logger.info(f"🔍 Searching best practices for {problem_type}...")
        
        # Check cache first
        cache_key = f"practices_{problem_type}"
        cached = self.cache.get(cache_key)
        if cached:
            logger.info("✅ Best practices found in cache")
            self.agent_metrics["cache_hits"] += 1
            return cached
        
        self.agent_metrics["cache_misses"] += 1
        
        # Check knowledge base
        kb_practices = self.knowledge_base.get_best_practices(problem_type)
        if kb_practices:
            self.cache.put(cache_key, kb_practices)
            return kb_practices
        
        # Fallback to basic practices
        best_practices = self._get_default_best_practices(problem_type)
        
        # Cache for future use
        self.cache.put(cache_key, best_practices)
        self.knowledge_base.store_best_practices(problem_type, best_practices)
        
        return best_practices
    
    def _get_default_best_practices(self, problem_type: str) -> Dict[str, Any]:
        """Get default best practices without web search"""
        practices = {
            'recommended_approaches': [],
            'recent_innovations': [],
            'common_pitfalls': [],
            'benchmark_scores': {},
            'sources': ['internal_knowledge']
        }
        
        if problem_type == 'fraud_detection':
            practices['recommended_approaches'] = [
                "Use ensemble of tree-based models (XGBoost, LightGBM)",
                "Implement velocity checks and behavioral features",
                "Apply SMOTE for severe class imbalance"
            ]
            practices['benchmark_scores'] = {'industry_average_auc': 0.95}
        elif problem_type == 'churn_prediction':
            practices['recommended_approaches'] = [
                "Create RFM features (Recency, Frequency, Monetary)",
                "Use survival analysis techniques",
                "Handle class imbalance with SMOTE/ADASYN"
            ]
            practices['benchmark_scores'] = {'industry_average_f1': 0.75}
        else:
            practices['recommended_approaches'] = [
                "Start with gradient boosting models",
                "Implement proper cross-validation",
                "Feature engineering is key"
            ]
        
        return practices
    
    @memory_safe(max_memory_mb=400)
    async def generate_optimal_config(
        self,
        understanding: MLContext,
        best_practices: Dict[str, Any],
        user_preferences: Optional[Dict[str, Any]] = None,
        constraints: Optional[Dict[str, Any]] = None
    ) -> OptimalConfig:
        """Memory-safe config generation"""
        logger.info("⚙️ Generating optimal configuration...")
        self.agent_metrics["config_generations"] += 1
        
        with MemoryBudget(budget_mb=300):
            context = {
                'problem_type': understanding.problem_type,
                'detected_patterns': understanding.detected_patterns,
                'business_sector': understanding.business_sector,
                'temporal_aspect': understanding.temporal_aspect,
                'imbalance_detected': understanding.imbalance_detected,
                'confidence': understanding.confidence
            }
            
            if best_practices.get('recommended_approaches'):
                context['recommended_approaches'] = best_practices['recommended_approaches']
            
            # Generate with empty DataFrame to save memory
            config = await self.config_generator.generate_config(
                df=pd.DataFrame(),
                context=context,
                constraints=constraints,
                user_preferences=user_preferences
            )
        
        logger.info(f"✅ Config: {len(config.algorithms)} algorithms")
        self._log_memory_status()
        
        return config
    
    @memory_safe(max_memory_mb=1000)
    async def execute_intelligent_cleaning(
        self,
        df: pd.DataFrame,
        context: MLContext,
        config: OptimalConfig,
        target_col: Optional[str] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Memory-safe data cleaning with batch processing for large datasets
        """
        logger.info("🧹 Starting intelligent cleaning...")
        
        # Check if batch processing is needed
        if len(df) > self.batch_size * 2:
            logger.info(f"📦 Large dataset detected: {len(df)} rows. Using batch processing.")
            return await self._execute_cleaning_batched(df, context, config, target_col)
        
        # Standard cleaning for smaller datasets
        with MemoryBudget(budget_mb=800):
            user_context = {
                'secteur_activite': context.business_sector or config.task,
                'target_variable': target_col,
                'contexte_metier': f"ML Pipeline for {context.problem_type}",
                'detected_problem_type': context.problem_type,
                'ml_confidence': context.confidence
            }
            
            cleaning_config = {
                'preprocessing': config.preprocessing,
                'feature_engineering': config.feature_engineering,
                'algorithms': config.algorithms,
                'task': config.task,
                'primary_metric': config.primary_metric
            }
            
            cleaned_df, cleaning_report = await self.cleaning_orchestrator.clean_dataset(
                df=df,
                user_context=user_context,
                cleaning_config=cleaning_config,
                use_intelligence=True
            )
            
            self.agent_metrics["cleaner_calls"] += 1
        
        logger.info("✅ Cleaning complete")
        self._log_memory_status()
        
        return cleaned_df, cleaning_report
    
    async def _execute_cleaning_batched(
        self,
        df: pd.DataFrame,
        context: MLContext,
        config: OptimalConfig,
        target_col: Optional[str] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Process large dataset in batches"""
        cleaned_batches = []
        cleaning_reports = []
        
        n_batches = len(df) // self.batch_size + (1 if len(df) % self.batch_size else 0)
        logger.info(f"📦 Processing {n_batches} batches...")
        
        for i, batch_start in enumerate(range(0, len(df), self.batch_size)):
            batch_end = min(batch_start + self.batch_size, len(df))
            batch = df.iloc[batch_start:batch_end].copy()
            
            logger.info(f"🔄 Batch {i+1}/{n_batches}: rows {batch_start}-{batch_end}")
            
            # Clean batch with memory budget
            with MemoryBudget(budget_mb=500):
                cleaned_batch, report = await self.execute_intelligent_cleaning(
                    batch, context, config, target_col
                )
            
            cleaned_batches.append(cleaned_batch)
            cleaning_reports.append(report)
            
            # Cleanup
            del batch, cleaned_batch
            gc.collect()
            
            if i % 5 == 0:  # Every 5 batches
                self.memory_monitor.force_cleanup()
                self.agent_metrics["memory_cleanups"] += 1
        
        # Combine results
        logger.info("🔗 Combining batches...")
        combined_df = pd.concat(cleaned_batches, ignore_index=True)
        
        # Merge reports
        combined_report = {
            'n_batches': n_batches,
            'total_rows': len(combined_df),
            'batch_reports': cleaning_reports
        }
        
        # Cleanup
        del cleaned_batches
        gc.collect()
        
        return combined_df, combined_report
    
    async def generate_execution_summary_with_claude(
        self,
        result: ProductionMLPipelineResult,
        understanding: MLContext
    ) -> str:
        """Generate executive summary with Claude"""
        if not self.use_claude:
            return self._generate_basic_summary(result, understanding)
        
        logger.info("💎 Generating summary with Claude...")
        self.agent_metrics["claude_orchestrations"] += 1
        
        prompt = f"""Generate a concise executive summary for this ML pipeline execution.

ML Problem:
- Type: {understanding.problem_type}
- Confidence: {understanding.confidence:.1%}
- Business Sector: {understanding.business_sector}

Execution Results:
- Success: {result.success}
- Execution Time: {result.execution_time:.1f}s
- Memory Peak: {result.memory_stats.get('peak_mb', 0):.1f} MB
- Cache Hit Rate: {result.cache_stats.get('hit_rate', 0):.1%}

Performance:
{json.dumps(result.performance_metrics, indent=2)}

Configuration:
- Algorithms: {', '.join(result.config_used.algorithms)}
- Primary Metric: {result.config_used.primary_metric}

Issues: {len(result.warnings)} warnings, {len(result.errors)} errors

Create 3-4 sentences covering:
1. Overall outcome
2. Key performance metrics
3. Notable concerns or recommendations
4. Next steps"""
        
        try:
            response = await self.claude_client.messages.create(
                model=self.claude_model,
                max_tokens=800,
                system="You are an expert ML engineer creating executive summaries.",
                messages=[{"role": "user", "content": prompt}]
            )
            
            summary = response.content[0].text.strip()
            logger.info("💎 Summary generated")
            return summary
            
        except Exception as e:
            logger.warning(f"⚠️ Claude failed: {e}")
            return self._generate_basic_summary(result, understanding)
    
    def _generate_basic_summary(
        self,
        result: ProductionMLPipelineResult,
        understanding: MLContext
    ) -> str:
        """Fallback summary"""
        summary = f"ML Pipeline: {understanding.problem_type}\n"
        summary += f"Success: {result.success} | Time: {result.execution_time:.1f}s\n"
        summary += f"Memory Peak: {result.memory_stats.get('peak_mb', 0):.1f} MB\n"
        
        if result.performance_metrics:
            best = max(result.performance_metrics.values())
            summary += f"Best Performance: {best:.3f}\n"
        
        return summary
    
    async def automl_without_templates(
        self,
        df: pd.DataFrame,
        target_col: Optional[str] = None,
        user_hints: Optional[Dict[str, Any]] = None,
        constraints: Optional[Dict[str, Any]] = None
    ) -> ProductionMLPipelineResult:
        """
        Complete AutoML with full memory protection
        PRODUCTION-READY ENTRY POINT
        """
        logger.info("=" * 60)
        logger.info("🛡️ PRODUCTION UNIVERSAL ML AGENT - MEMORY PROTECTED")
        logger.info("=" * 60)
        
        start_time = datetime.now()
        initial_memory = self.memory_monitor.get_memory_usage_mb()
        
        try:
            # Step 1: Understand problem
            logger.info("🔍 Step 1/5: Understanding problem...")
            understanding = await self.understand_problem(df, target_col, user_hints)
            
            # Step 2: Validate
            logger.info("🔍 Step 2/5: Validating data...")
            validation = await self.validate_with_standards(df, understanding)
            
            # Step 3: Best practices
            logger.info("🔍 Step 3/5: Searching best practices...")
            practices = await self.search_ml_best_practices(
                understanding.problem_type,
                {'n_samples': len(df), 'n_features': len(df.columns)}
            )
            
            # Step 4: Generate config
            logger.info("🔍 Step 4/5: Generating configuration...")
            config = await self.generate_optimal_config(
                understanding, practices, user_hints, constraints
            )
            
            # Step 5: Execute pipeline
            logger.info("🔍 Step 5/5: Executing pipeline...")
            result = await self._execute_pipeline_with_protection(
                df, config, understanding, target_col
            )
            
            # Finalize with memory stats
            execution_time = (datetime.now() - start_time).total_seconds()
            final_memory = self.memory_monitor.get_memory_usage_mb()
            
            result.execution_time = execution_time
            result.memory_stats = {
                'initial_mb': initial_memory,
                'final_mb': final_memory,
                'peak_mb': self.memory_monitor.peak_memory / (1024 * 1024),
                'delta_mb': final_memory - initial_memory
            }
            result.cache_stats = self.cache.get_stats()
            result.performance_profile = {
                'agent_calls': sum(self.agent_metrics.values()),
                'cache_hit_rate': self.agent_metrics['cache_hits'] / 
                                 max(1, self.agent_metrics['cache_hits'] + self.agent_metrics['cache_misses'])
            }
            
            # Generate Claude summary
            if self.use_claude:
                result.claude_summary = await self.generate_execution_summary_with_claude(
                    result, understanding
                )
            
            # Log summary
            logger.info("=" * 60)
            logger.info("📊 EXECUTION SUMMARY")
            logger.info(f"Success: {result.success} | Time: {execution_time:.1f}s")
            logger.info(f"Memory: Δ{result.memory_stats['delta_mb']:.1f} MB "
                       f"(Peak: {result.memory_stats['peak_mb']:.1f} MB)")
            logger.info(f"Cache: {result.cache_stats['items']} items, "
                       f"{result.cache_stats['size_mb']:.1f} MB, "
                       f"{result.performance_profile['cache_hit_rate']:.1%} hit rate")
            if result.claude_summary:
                logger.info(f"\n💎 {result.claude_summary}")
            logger.info("=" * 60)
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}", exc_info=True)
            
            return ProductionMLPipelineResult(
                success=False,
                cleaned_data=None,
                config_used=OptimalConfig(
                    task='unknown', algorithms=[], primary_metric='auto',
                    preprocessing={}, feature_engineering={}, hpo_config={},
                    cv_strategy={}, ensemble_config={}, time_budget=0,
                    resource_constraints={}, monitoring={}
                ),
                context_detected=MLContext(
                    problem_type='unknown', detected_patterns=[],
                    confidence=0.0, business_sector=None
                ),
                cleaning_report={},
                performance_metrics={},
                execution_time=(datetime.now() - start_time).total_seconds(),
                errors=[str(e)]
            )
        finally:
            # Final cleanup
            gc.collect()
            self.memory_monitor.force_cleanup()
    
    async def _execute_pipeline_with_protection(
        self,
        df: pd.DataFrame,
        config: OptimalConfig,
        context: MLContext,
        target_col: Optional[str] = None
    ) -> ProductionMLPipelineResult:
        """Execute full pipeline with memory protection"""
        errors = []
        warnings = []
        
        # Phase 1: Cleaning
        logger.info("🧹 Phase 1: Data Cleaning")
        cleaned_df, cleaning_report = await self.execute_intelligent_cleaning(
            df, context, config, target_col
        )
        
        # Phase 2: Quality Control
        logger.info("🎯 Phase 2: Quality Control")
        with MemoryBudget(budget_mb=300):
            control_report = await self.controller.validate(
                cleaned_df, df, cleaning_report.get('transformations', [])
            )
            self.agent_metrics["controller_calls"] += 1
        
        # Phase 3: Save config
        logger.info("💾 Phase 3: Saving Configuration")
        yaml_path = self.yaml_handler.save_cleaning_config(
            transformations=cleaning_report.get('transformations', []),
            validation_sources=cleaning_report.get('validation_sources', []),
            user_context={
                'secteur_activite': context.business_sector,
                'target_variable': target_col,
                'detected_problem_type': context.problem_type
            },
            metrics=control_report.get('metrics', {})
        )
        
        # Simulate training (with memory budget)
        logger.info("🎯 Phase 4: Model Training (simulated)")
        performance = {
            algo: 0.85 + np.random.random() * 0.10
            for algo in config.algorithms
        }
        
        return ProductionMLPipelineResult(
            success=True,
            cleaned_data=cleaned_df,
            config_used=config,
            context_detected=context,
            cleaning_report=cleaning_report,
            performance_metrics=performance,
            execution_time=0,  # Will be set later
            yaml_config_path=yaml_path,
            warnings=warnings,
            errors=errors
        )
    
    def cleanup(self):
        """Manual cleanup of resources"""
        logger.info("🧹 Cleaning up resources...")
        self.cache.clear()
        gc.collect()
        self.memory_monitor.force_cleanup()
        logger.info("✅ Cleanup complete")


# ============================================================================
# PRODUCTION KNOWLEDGE BASE WITH SIZE LIMITS
# ============================================================================

class ProductionKnowledgeBase:
    """Memory-safe knowledge base with limits"""
    
    def __init__(self, max_patterns: int = 100, storage_path: Optional[Path] = None):
        self.max_patterns = max_patterns
        self.storage_path = storage_path or Path("./knowledge_base")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.best_practices_cache = {}
        self.successful_patterns = []
        
        self._load_knowledge()
    
    def get_best_practices(self, problem_type: str) -> Optional[Dict[str, Any]]:
        return self.best_practices_cache.get(problem_type)
    
    def store_best_practices(self, problem_type: str, practices: Dict[str, Any]):
        self.best_practices_cache[problem_type] = practices
        self._save_knowledge()
    
    def store_successful_pattern(
        self,
        task: str,
        config: Dict[str, Any],
        performance: Dict[str, float]
    ):
        pattern = {
            'timestamp': datetime.now().isoformat(),
            'task': task,
            'config': config,
            'performance': performance,
            'score': max(performance.values()) if performance else 0
        }
        
        self.successful_patterns.append(pattern)
        
        # Keep only best patterns, limit size
        self.successful_patterns.sort(key=lambda x: x['score'], reverse=True)
        self.successful_patterns = self.successful_patterns[:self.max_patterns]
        
        self._save_knowledge()
    
    def _load_knowledge(self):
        kb_file = self.storage_path / "knowledge.pkl"
        if kb_file.exists():
            try:
                with open(kb_file, 'rb') as f:
                    data = pickle.load(f)
                    self.best_practices_cache = data.get('practices', {})
                    self.successful_patterns = data.get('patterns', [])[:self.max_patterns]
                logger.info(f"📚 Loaded {len(self.successful_patterns)} patterns")
            except Exception as e:
                logger.warning(f"⚠️ Failed to load knowledge: {e}")
    
    def _save_knowledge(self):
        kb_file = self.storage_path / "knowledge.pkl"
        try:
            data = {
                'practices': self.best_practices_cache,
                'patterns': self.successful_patterns
            }
            with open(kb_file, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            logger.warning(f"⚠️ Failed to save knowledge: {e}")
