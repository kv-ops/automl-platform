"""
Data Cleaning Orchestrator - Main coordination for OpenAI agents
PRODUCTION-READY: Memory leak fixes, parallelization, circuit breakers
ENHANCED: Hybrid mode with local/agent arbitration for retail
FINALIZED: All thresholds are now configurable
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

from .intelligent_context_detector import IntelligentContextDetector
from .intelligent_config_generator import IntelligentConfigGenerator
from .adaptive_template_system import AdaptiveTemplateSystem

# Import from enhanced version for GS1 compliance and retail recommendations
try:
    from ..data_quality_agent_enhanced import IntelligentDataQualityAgent
except ImportError:
    # Fallback to v2 if enhanced not available
    from ..data_quality_agent_v2 import IntelligentDataQualityAgent

from .utils import (
    async_retry,
    BoundedList,
    run_parallel,
    track_llm_cost,
    parse_llm_json
)

from ..risk import RiskLevel

logger = logging.getLogger(__name__)


class DataCleaningOrchestrator:
    """
    Orchestrates the intelligent data cleaning process using OpenAI agents
    
    PRODUCTION-READY with:
    - Memory leak protection (bounded history)
    - Parallel execution where possible
    - Circuit breaker protection
    - Enhanced error handling
    - Hybrid mode with local/agent arbitration
    - Improved retail recommendation exploitation
    - Fully configurable thresholds
    """
    
    def __init__(
        self, 
        config: Optional[AgentConfig] = None, 
        automl_config: Optional[Dict] = None, 
        use_claude: bool = True
    ):
        """Initialize orchestrator with configuration"""
        self.config = config or AgentConfig()
        self.automl_config = automl_config or {}
        self.use_claude = use_claude
        self._claude_available = use_claude and AsyncAnthropic is not None

        self.config.validate()

        # Initialize agents
        self.profiler = ProfilerAgent(self.config)
        self.validator = ValidatorAgent(self.config, use_claude=use_claude)
        self.cleaner = CleanerAgent(self.config)
        self.controller = ControllerAgent(self.config)
        
        # Initialize Data Quality Agent with config
        self.data_quality_agent = IntelligentDataQualityAgent(
            enable_agent_first=True,
            config=self.config
        )
        
        # Initialize intelligent modules
        self.context_detector = IntelligentContextDetector(
            anthropic_api_key=self.config.anthropic_api_key,
            config=self.config
        )
        self.config_generator = IntelligentConfigGenerator(use_claude=use_claude)
        self.adaptive_templates = AdaptiveTemplateSystem(use_claude=use_claude)
        
        # Initialize Claude client if available
        self.claude_client: Optional[Any] = None
        self.claude_model = "claude-3-opus-20240229"
        if self._claude_available:
            self.claude_client = AsyncAnthropic(api_key=self.config.anthropic_api_key)
            logger.info("🔮 Claude SDK enabled for strategic cleaning decisions")
        else:
            if self.use_claude and AsyncAnthropic is None:
                logger.warning(
                    "⚠️ Claude SDK requested but not installed. Falling back to rule-based orchestration."
                )
            else:
                logger.info("📋 Using rule-based cleaning orchestration")
        
        # Tracking with MEMORY LEAK PROTECTION
        self.execution_history = BoundedList(maxlen=100)  # FIX: Bounded list
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
            "claude_decisions": 0,
            "hybrid_decisions": {"local": 0, "agent": 0},
            "retail_recommendations_applied": []
        }
        
        # Results storage
        self.cleaning_report = {}
        self.validation_sources = []
        self.transformations_applied = []
        self.ml_context = None
        self.local_stats = {}  # Store local statistics
        self.retail_recommendations = []  # Store retail recommendations
        self.hybrid_mode_active = config.enable_hybrid_mode if config else False
        
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
    
    def _calculate_local_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate local metrics for hybrid decision making.
        Retail-optimized metrics.
        """
        metrics = {
            'missing_ratio': df.isnull().sum().sum() / (df.shape[0] * df.shape[1]),
            'duplicate_ratio': df.duplicated().mean(),
            'quality_score': 100.0,
            'complexity_score': 0.0,
            'has_sentinel_values': False,
            'has_negative_prices': False,
            'outlier_ratio': 0.0
        }
        
        # Check for sentinel values
        sentinel_values = self.config.retail_rules.get('sentinel_values', [-999, -1, 9999])
        for col in df.select_dtypes(include=[np.number]).columns:
            # Check if it's a stock column
            if self.config.is_sentinel_value(0, col):  # Uses contextual check
                continue
            if df[col].isin(sentinel_values).any():
                metrics['has_sentinel_values'] = True
                break
        
        # Check for negative prices
        price_columns = [col for col in df.columns if 'price' in col.lower() or 'prix' in col.lower()]
        for col in price_columns:
            if pd.api.types.is_numeric_dtype(df[col]) and (df[col] < 0).any():
                metrics['has_negative_prices'] = True
                break
        
        # Calculate quality score
        missing_penalty = min(30, metrics['missing_ratio'] * 100)
        duplicate_penalty = min(20, metrics['duplicate_ratio'] * 100)
        metrics['quality_score'] -= (missing_penalty + duplicate_penalty)
        
        # Calculate complexity
        if df.shape[1] > 50:
            metrics['complexity_score'] += 0.3
        if metrics['missing_ratio'] > 0.3:
            metrics['complexity_score'] += 0.3
        if metrics['has_sentinel_values']:
            metrics['complexity_score'] += 0.2
        
        # Calculate outlier ratio
        outlier_counts = []
        for col in df.select_dtypes(include=[np.number]).columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)).sum()
            outlier_counts.append(outliers)
        
        if outlier_counts:
            metrics['outlier_ratio'] = sum(outlier_counts) / (len(df) * len(outlier_counts))
        
        self.local_stats = metrics
        return metrics
    
    def _should_use_agent_mode(self, metrics: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Determine if agent mode should be used based on metrics.
        Core hybrid arbitration logic.
        """
        if not self.hybrid_mode_active:
            return True, "Hybrid mode disabled"
        
        use_agent, reason = self.config.should_use_agent(metrics)
        
        # Track decision
        if use_agent:
            self.performance_metrics["hybrid_decisions"]["agent"] += 1
        else:
            self.performance_metrics["hybrid_decisions"]["local"] += 1
        
        logger.info(f"🔄 Hybrid decision: {'AGENT' if use_agent else 'LOCAL'} - {reason}")
        
        return use_agent, reason
    
    async def _test_and_apply_strategies(
        self,
        df: pd.DataFrame,
        recommendations: List[Dict[str, Any]]
    ) -> pd.DataFrame:
        """
        Test and apply cleaning strategies based on recommendations.
        Prioritizes retail-specific recommendations.
        """
        cleaned_df = df.copy()
        
        # Extract retail recommendations
        retail_recs = [r for r in recommendations if r.get('category') in ['sentinel_handling', 'price_correction', 'imputation_strategy']]
        
        # Apply retail recommendations first
        for rec in retail_recs:
            try:
                if rec['category'] == 'sentinel_handling':
                    # Replace sentinel values
                    for col in rec.get('columns_affected', []):
                        if col in cleaned_df.columns:
                            sentinels = self.config.get_retail_rules('sentinel_values')
                            # Check if it's a stock column
                            if self.config.is_sentinel_value(0, col):
                                sentinels = [s for s in sentinels if s != 0]
                            
                            mask = cleaned_df[col].isin(sentinels)
                            if mask.any():
                                # Use category-based imputation if available
                                if 'category' in cleaned_df.columns:
                                    cleaned_df.loc[mask, col] = np.nan
                                    cleaned_df[col] = cleaned_df.groupby('category')[col].transform(
                                        lambda x: x.fillna(x.median())
                                    )
                                else:
                                    cleaned_df.loc[mask, col] = cleaned_df[col][~mask].median()
                                
                                self.performance_metrics["retail_recommendations_applied"].append({
                                    "type": "sentinel_replacement",
                                    "column": col,
                                    "count": mask.sum()
                                })
                                logger.info(f"✅ Applied sentinel replacement for {col}")
                
                elif rec['category'] == 'price_correction':
                    # Fix negative prices
                    for col in rec.get('columns_affected', []):
                        if col in cleaned_df.columns and pd.api.types.is_numeric_dtype(cleaned_df[col]):
                            negative_mask = cleaned_df[col] < 0
                            if negative_mask.any():
                                # Use category-based median if available
                                if 'category' in cleaned_df.columns:
                                    cleaned_df.loc[negative_mask, col] = cleaned_df.groupby('category')[col].transform(
                                        lambda x: x[x > 0].median()
                                    )[negative_mask]
                                else:
                                    cleaned_df.loc[negative_mask, col] = cleaned_df[col][cleaned_df[col] > 0].median()
                                
                                self.performance_metrics["retail_recommendations_applied"].append({
                                    "type": "price_correction",
                                    "column": col,
                                    "count": negative_mask.sum()
                                })
                                logger.info(f"✅ Applied price correction for {col}")
                
                elif rec['category'] == 'imputation_strategy' and 'category' in cleaned_df.columns:
                    # Apply category-based imputation for missing values
                    for col in cleaned_df.select_dtypes(include=[np.number]).columns:
                        if cleaned_df[col].isnull().any():
                            cleaned_df[col] = cleaned_df.groupby('category')[col].transform(
                                lambda x: x.fillna(x.median())
                            )
                            self.performance_metrics["retail_recommendations_applied"].append({
                                "type": "category_imputation",
                                "column": col
                            })
                    logger.info("✅ Applied category-based imputation")
                    
            except Exception as e:
                logger.warning(f"Failed to apply recommendation {rec['category']}: {e}")
        
        # Apply remaining general recommendations
        general_recs = [r for r in recommendations if r not in retail_recs]
        for rec in general_recs[:3]:  # Limit to top 3 general recommendations
            try:
                if rec['category'] == 'missing_data':
                    # Simple imputation for remaining missing values
                    for col in cleaned_df.columns:
                        if cleaned_df[col].isnull().any():
                            if pd.api.types.is_numeric_dtype(cleaned_df[col]):
                                cleaned_df[col].fillna(cleaned_df[col].median(), inplace=True)
                            else:
                                cleaned_df[col].fillna('missing', inplace=True)
                    logger.info("✅ Applied general missing data handling")
                    
                elif rec['category'] == 'outliers':
                    # Clip outliers
                    for col in cleaned_df.select_dtypes(include=[np.number]).columns:
                        # Skip stock columns for outlier handling
                        if 'stock' in col.lower() or 'qty' in col.lower() or 'quantity' in col.lower():
                            continue
                        lower = cleaned_df[col].quantile(0.01)
                        upper = cleaned_df[col].quantile(0.99)
                        cleaned_df[col] = cleaned_df[col].clip(lower, upper)
                    logger.info("✅ Applied outlier clipping")
                    
            except Exception as e:
                logger.warning(f"Failed to apply general recommendation {rec['category']}: {e}")
        
        return cleaned_df
    
    @async_retry(max_attempts=3, base_delay=2.0)
    @track_llm_cost('claude', 'orchestrator')
    async def determine_cleaning_mode_with_claude(
        self,
        df: pd.DataFrame,
        user_context: Dict[str, Any],
        ml_context: Any
    ) -> Dict[str, Any]:
        """Use Claude to intelligently determine the best cleaning mode"""
        if (
            not self.use_claude
            or not self.config.can_call_llm('claude')
            or self.claude_client is None
        ):
            if self.use_claude and self.claude_client is None:
                logger.warning(
                    "Claude client unavailable; falling back to rule-based mode determination."
                )
            return await self._determine_best_mode_rule_based(df, user_context, ml_context)

        logger.info("🔮 Using Claude to determine optimal cleaning mode...")
        self.performance_metrics["claude_decisions"] += 1
        
        # Include local metrics in decision
        local_metrics = self._calculate_local_metrics(df)
        
        data_summary = {
            'shape': df.shape,
            'missing_ratio': local_metrics['missing_ratio'],
            'duplicate_ratio': local_metrics['duplicate_ratio'],
            'numeric_cols': int(len(df.select_dtypes(include=[np.number]).columns)),
            'categorical_cols': int(len(df.select_dtypes(include=['object']).columns)),
            'high_cardinality_cols': int(sum(1 for col in df.columns if df[col].nunique() > 20)),
            'quality_score': local_metrics['quality_score'],
            'has_sentinel_values': local_metrics['has_sentinel_values'],
            'has_negative_prices': local_metrics['has_negative_prices']
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
4. LOCAL: Use rule-based cleaning without agents (cheapest, good for simple cases)

Consider:
- Data quality issues (missing: {data_summary['missing_ratio']:.1%}, duplicates: {data_summary['duplicate_ratio']:.1%})
- Business criticality of the problem
- Need for compliance/auditability
- Time/resource constraints
- Cost optimization (local mode when possible)

Respond ONLY with valid JSON:
{{
  "recommended_mode": "automated|interactive|hybrid|local",
  "confidence": 0.0-1.0,
  "reasoning": "detailed explanation why this mode",
  "key_considerations": ["point1", "point2", "point3"],
  "estimated_time_minutes": number,
  "risk_level": "low|medium|high",
  "use_local_cleaning": true/false
}}"""
        
        try:
            response = await self.claude_client.messages.create(
                model=self.claude_model,
                max_tokens=1500,
                system="You are an expert data engineer helping choose optimal data cleaning strategies.",
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Record success
            self.config.record_llm_success('claude')
            
            decision = parse_llm_json(
                response.content[0].text.strip(),
                fallback=await self._determine_best_mode_rule_based(df, user_context, ml_context)
            )

            try:
                decision_risk = RiskLevel.normalize(
                    decision.get("risk_level"), field_name="risk_level"
                )
            except ValueError:
                logger.warning(
                    "Received invalid risk level '%s' from Claude decision, defaulting to medium.",
                    decision.get("risk_level"),
                )
                decision_risk = RiskLevel.MEDIUM

            decision["risk_level"] = decision_risk.value
            
            # Store local cleaning preference
            decision["prefer_local"] = decision.get("use_local_cleaning", False)

            logger.info(f"🔮 Claude recommends: {decision.get('recommended_mode', 'unknown')} mode")
            logger.info(f"   Confidence: {decision.get('confidence', 0):.1%}")
            logger.info(f"   Use local: {decision.get('prefer_local', False)}")

            return decision
            
        except Exception as e:
            logger.warning(f"⚠️ Claude mode determination failed: {e}")
            self.config.record_llm_failure('claude')
            return await self._determine_best_mode_rule_based(df, user_context, ml_context)
    
    async def _determine_best_mode_rule_based(
        self,
        df: pd.DataFrame,
        user_context: Dict[str, Any],
        ml_context: Any
    ) -> Dict[str, Any]:
        """Fallback rule-based mode determination with hybrid awareness"""
        local_metrics = self._calculate_local_metrics(df)
        missing_ratio = local_metrics['missing_ratio']
        
        critical_sectors = ['finance', 'healthcare', 'banking', 'insurance']
        is_critical = user_context.get('secteur_activite', '').lower() in critical_sectors
        
        # Use configurable thresholds
        quality_threshold = self.config.get_quality_threshold('quality_score_threshold') or 80
        missing_threshold = self.config.get_quality_threshold('missing_warning_threshold') or 0.35
        
        # Check if local cleaning is sufficient
        use_local = (
            local_metrics['quality_score'] > quality_threshold and
            not local_metrics['has_negative_prices'] and
            missing_ratio < missing_threshold and
            not is_critical
        )
        
        if use_local:
            mode = 'local'
            estimated_time = 2
            risk = RiskLevel.LOW
        elif is_critical or missing_ratio > missing_threshold:
            mode = 'interactive'
            estimated_time = 25
            risk = RiskLevel.HIGH
        elif missing_ratio > 0.15 or len(df) > 100000:
            mode = 'hybrid'
            estimated_time = 15
            risk = RiskLevel.MEDIUM
        else:
            mode = 'automated'
            estimated_time = 8
            risk = RiskLevel.LOW

        return {
            'recommended_mode': mode,
            'confidence': 0.7,
            'reasoning': f"Rule-based: {risk.value} risk sector, {missing_ratio:.1%} missing data, quality score {local_metrics['quality_score']:.1f}",
            'key_considerations': ['Data quality', 'Sector criticality', 'Dataset size'],
            'estimated_time_minutes': estimated_time,
            'risk_level': risk.value,
            'use_local_cleaning': use_local,
            'prefer_local': use_local
        }
    
    def _generate_final_report(
        self, 
        df_original: pd.DataFrame, 
        df_cleaned: pd.DataFrame,
        quality_score_before: float,
        quality_score_after: float
    ) -> Dict[str, Any]:
        """
        Generate comprehensive final report with all metrics
        Enhanced with retail-specific metrics, verdict, and recommendation tracking
        Uses configurable thresholds
        """
        logger.info("Generating final cleaning report")
        
        # Calculate basic metrics
        rows_removed = len(df_original) - len(df_cleaned)
        cols_removed = len(df_original.columns) - len(df_cleaned.columns)
        missing_before = df_original.isnull().sum().sum()
        missing_after = df_cleaned.isnull().sum().sum()
        duplicates_removed = df_original.duplicated().sum()
        
        # Aggregate metrics from transformations
        sentinels_removed = 0
        negative_prices_fixed = 0
        missing_imputed = 0
        outliers_handled = 0
        
        for trans in self.transformations_applied:
            if trans.get("action") == "handle_sentinels" or trans.get("action") == "replace_sentinels":
                sentinels_removed += trans.get("params", {}).get("sentinels_removed", 0) or trans.get("params", {}).get("count", 0)
            elif trans.get("action") == "fix_negative_prices":
                negative_prices_fixed += trans.get("params", {}).get("negative_prices_corrected", 0) or trans.get("params", {}).get("count", 0)
            elif trans.get("action") == "fill_missing":
                missing_imputed += trans.get("params", {}).get("missing_imputed", 0) or 1
            elif trans.get("action") == "handle_outliers" or trans.get("action") == "clip_outliers":
                outliers_handled += trans.get("params", {}).get("outliers_found", 0) or trans.get("params", {}).get("count", 0)
        
        # Calculate GS1 compliance
        gs1_compliance = 100  # Default
        if self.config.user_context.get("secteur_activite") == "retail":
            # Calculate actual GS1 compliance based on SKU columns
            sku_columns = [col for col in df_cleaned.columns if any(
                keyword in col.lower() for keyword in ['sku', 'upc', 'ean', 'gtin', 'barcode', 'product_code']
            )]
            
            if sku_columns:
                total_skus = 0
                compliant_skus = 0
                
                for col in sku_columns:
                    if df_cleaned[col].dtype == 'object':
                        values = df_cleaned[col].dropna()
                        total_skus += len(values)
                        # Check GS1 format compliance
                        for value in values:
                            value_str = str(value).strip()
                            if value_str.isdigit() and len(value_str) in [8, 12, 13, 14]:
                                compliant_skus += 1
                
                if total_skus > 0:
                    gs1_compliance = (compliant_skus / total_skus) * 100
        
        # Get configurable thresholds - HARMONIZED
        quality_threshold = self.config.get_quality_threshold('quality_score_threshold') or 93
        missing_threshold = self.config.get_quality_threshold('production_missing_threshold') or 0.05
        gs1_target = (self.config.get_quality_threshold('gs1_compliance_threshold') or 
                      self.config.get_retail_rules('gs1_compliance_target') or 0.98) * 100
        
        # Determine production readiness using configurable thresholds
        is_production_ready = (
            quality_score_after >= quality_threshold and
            gs1_compliance >= gs1_target and
            missing_after / (df_cleaned.shape[0] * df_cleaned.shape[1]) < missing_threshold
        )
        
        verdict = "Dataset ready for production" if is_production_ready else "Dataset requires additional review"
        
        # Count recommendations applied
        retail_recs_applied = len(self.performance_metrics.get('retail_recommendations_applied', []))
        
        report = {
            "summary": {
                "rows_original": len(df_original),
                "rows_cleaned": len(df_cleaned),
                "rows_removed": rows_removed,
                "columns_original": len(df_original.columns),
                "columns_cleaned": len(df_cleaned.columns),
                "columns_removed": cols_removed,
                "quality_score_before": quality_score_before,
                "quality_score_after": quality_score_after,
                "quality_improvement": quality_score_after - quality_score_before
            },
            "cleaning_metrics": {
                "missing_cells_before": int(missing_before),
                "missing_cells_after": int(missing_after),
                "missing_cells_handled": int(missing_before - missing_after),
                "duplicates_removed": int(duplicates_removed),
                "sentinels_removed": int(sentinels_removed),
                "negative_prices_corrected": int(negative_prices_fixed),
                "missing_values_imputed": int(missing_imputed),
                "outliers_handled": int(outliers_handled)
            },
            "transformations": {
                "total_transformations": len(self.transformations_applied),
                "by_type": self._count_transformations_by_type()
            },
            "retail_metrics": {
                "gs1_compliance": f"{gs1_compliance:.1f}%",
                "gs1_target": f"{gs1_target:.0f}%",
                "sentinel_columns_cleaned": len([t for t in self.transformations_applied 
                                                  if t.get("action") in ["handle_sentinels", "replace_sentinels"]]),
                "negative_price_columns_fixed": len([t for t in self.transformations_applied 
                                                    if t.get("action") == "fix_negative_prices"]),
                "category_based_imputations": len([t for t in self.transformations_applied 
                                                   if t.get("params", {}).get("method") == "median_by_category"])
            },
            "recommendations": {
                "total_recommendations": len(self.retail_recommendations),
                "retail_recommendations_applied": retail_recs_applied,
                "recommendations_ignored": len(self.retail_recommendations) - retail_recs_applied,
                "applied_details": self.performance_metrics.get('retail_recommendations_applied', [])
            },
            "performance": {
                "total_time_seconds": time.time() - self.start_time if self.start_time else 0,
                "api_calls": self.performance_metrics["total_api_calls"],
                "claude_decisions": self.performance_metrics["claude_decisions"],
                "hybrid_decisions": self.performance_metrics["hybrid_decisions"],
                "mode_used": "hybrid" if self.hybrid_mode_active else "standard"
            },
            "ml_context": self.ml_context.__dict__ if self.ml_context else None,
            "verdict": verdict,
            "production_readiness": {
                "is_ready": is_production_ready,
                "quality_threshold_met": quality_score_after >= quality_threshold,
                "quality_threshold": quality_threshold,
                "gs1_compliance_met": gs1_compliance >= gs1_target,
                "missing_threshold_met": missing_after / (df_cleaned.shape[0] * df_cleaned.shape[1]) < missing_threshold,
                "missing_threshold": missing_threshold,
                "notes": []
            }
        }
        
        # Add notes for production readiness
        if not is_production_ready:
            notes = []
            if quality_score_after < quality_threshold:
                notes.append(f"Quality score {quality_score_after:.1f}% below {quality_threshold:.0f}% threshold")
            if gs1_compliance < gs1_target:
                notes.append(f"GS1 compliance {gs1_compliance:.1f}% below {gs1_target:.0f}% target")
            if missing_after / (df_cleaned.shape[0] * df_cleaned.shape[1]) >= missing_threshold:
                notes.append(f"Missing data ratio exceeds {missing_threshold*100:.0f}% threshold")
            report["production_readiness"]["notes"] = notes
        
        logger.info(f"Final Report: {verdict}")
        logger.info(f"Quality Score: {quality_score_before:.1f}% → {quality_score_after:.1f}% (improvement: {quality_score_after - quality_score_before:.1f}%)")
        logger.info(f"Retail Metrics: Sentinels removed: {sentinels_removed}, Negative prices fixed: {negative_prices_fixed}")
        logger.info(f"Recommendations: {retail_recs_applied}/{len(self.retail_recommendations)} applied")
        
        if self.config.user_context.get("secteur_activite") == "retail":
            logger.info(f"GS1 Compliance: {gs1_compliance:.1f}% (target: {gs1_target:.0f}%)")
        
        return report
    
    def _count_transformations_by_type(self) -> Dict[str, int]:
        """Count transformations by type"""
        counts = {}
        for trans in self.transformations_applied:
            action = trans.get("action", "unknown")
            counts[action] = counts.get(action, 0) + 1
        return counts
    
    async def clean_dataset(
        self, 
        df: pd.DataFrame, 
        user_context: Dict[str, Any],
        cleaning_config: Optional[Dict] = None,
        use_intelligence: bool = True
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Main pipeline for intelligent data cleaning with hybrid mode
        OPTIMIZED with parallel execution where possible
        IMPROVED with better retail recommendation exploitation
        """
        self.start_time = time.time()
        
        self.config.user_context.update(user_context)
        
        # Extract local stats from user context if provided
        if 'local_stats' in user_context:
            self.local_stats = user_context['local_stats']
        else:
            self.local_stats = self._calculate_local_metrics(df)
        
        logger.info(f"Starting intelligent cleaning for dataset with shape {df.shape}")
        logger.info(f"Agent-First mode: {'ENABLED' if use_intelligence else 'DISABLED'}")
        logger.info(f"Claude enhancement: {'ENABLED' if self.use_claude else 'DISABLED'}")
        logger.info(f"Hybrid mode: {'ENABLED' if self.hybrid_mode_active else 'DISABLED'}")
        
        try:
            # Phase 1: Data Quality Assessment
            assessment = self.data_quality_agent.assess(df, user_context.get("target_variable"))
            quality_score_before = assessment.quality_score
            
            # Extract retail recommendations
            self.retail_recommendations = assessment.retail_recommendations if hasattr(assessment, 'retail_recommendations') else []
            
            if self.retail_recommendations:
                logger.info(f"📋 Found {len(self.retail_recommendations)} retail-specific recommendations")
                for rec in self.retail_recommendations:
                    logger.info(f"  - {rec['title']}: {rec['description']}")
            
            # Check if we should use local cleaning
            use_agent, reason = self._should_use_agent_mode(self.local_stats)
            
            if not use_agent and self.hybrid_mode_active:
                logger.info(f"📋 Using LOCAL cleaning: {reason}")
                cleaned_df, report = await self._apply_local_cleaning_enhanced(df, user_context, assessment)
                return cleaned_df, report
            
            # Continue with agent-based cleaning
            # Phase 2: Intelligent detection (if enabled)
            if use_intelligence:
                self.performance_metrics["intelligence_used"] = True
                
                # Detect ML context
                self.ml_context = await self.context_detector.detect_ml_context(
                    df, 
                    target_col=user_context.get("target_variable"),
                    user_hints=user_context
                )
                logger.info(f"🎯 Detected ML problem: {self.ml_context.problem_type} "
                           f"(confidence: {self.ml_context.confidence:.1%})")
                
                # PARALLEL: Get mode decision + profile in parallel
                if self.use_claude:
                    profile_task = asyncio.create_task(self.profiler.analyze(df))
                    mode_task = asyncio.create_task(
                        self.determine_cleaning_mode_with_claude(df, user_context, self.ml_context)
                    )
                    
                    mode_decision, profile_report = await asyncio.gather(
                        mode_task, profile_task, return_exceptions=True
                    )
                    
                    # Handle exceptions
                    if isinstance(mode_decision, Exception):
                        logger.error(f"Mode decision failed: {mode_decision}")
                        mode_decision = await self._determine_best_mode_rule_based(df, user_context, self.ml_context)
                    
                    if isinstance(profile_report, Exception):
                        logger.error(f"Profiling failed: {profile_report}")
                        profile_report = {"error": str(profile_report)}
                    
                    # Check if mode decision suggests local cleaning
                    if mode_decision.get('prefer_local', False):
                        logger.info("🔡 Mode decision suggests LOCAL cleaning")
                        cleaned_df, report = await self._apply_local_cleaning_enhanced(df, user_context, assessment)
                        return cleaned_df, report
                    
                    logger.info(f"🔮 Cleaning mode: {mode_decision.get('recommended_mode', 'unknown')}")
                    
                    # Get recommendations
                    recommendations = await self.recommend_cleaning_approach_with_claude(
                        df, profile_report, self.ml_context
                    )
                    logger.info(f"🔮 Recommendations ready")
                else:
                    profile_report = await self.profiler.analyze(df)
                
                # Generate optimal configuration dynamically
                if not cleaning_config:
                    optimal_config = await self.config_generator.generate_config(
                        df=df,
                        context={
                            'problem_type': self.ml_context.problem_type,
                            'business_sector': self.ml_context.business_sector,
                            'temporal_aspect': self.ml_context.temporal_aspect,
                            'imbalance_detected': self.ml_context.imbalance_detected,
                            'local_metrics': self.local_stats
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
                        'business_sector': self.ml_context.business_sector,
                        'local_metrics': self.local_stats
                    },
                    agent_config=cleaning_config
                )
            else:
                profile_report = await self.profiler.analyze(df)
            
            # Phase 3: Apply retail recommendations if available
            if self.retail_recommendations:
                logger.info("📦 Applying retail-specific cleaning strategies...")
                df = await self._test_and_apply_strategies(df, self.retail_recommendations + assessment.recommendations)
            
            # Phase 4: Check dataset size and chunk if necessary
            df_chunks = self._chunk_dataset(df) if self._needs_chunking(df) else [df]
            
            # Phase 5: Process each chunk
            cleaned_chunks = []
            for i, chunk in enumerate(df_chunks):
                logger.info(f"Processing chunk {i+1}/{len(df_chunks)}")
                cleaned_chunk = await self._process_chunk(chunk, i, profile_report)
                cleaned_chunks.append(cleaned_chunk)
            
            # Phase 6: Combine chunks
            cleaned_df = pd.concat(cleaned_chunks, ignore_index=True) if len(cleaned_chunks) > 1 else cleaned_chunks[0]
            
            # Phase 7: Final quality assessment
            final_assessment = self.data_quality_agent.assess(cleaned_df, user_context.get("target_variable"))
            quality_score_after = final_assessment.quality_score
            
            # Phase 8: Generate final report
            self.cleaning_report = self._generate_final_report(df, cleaned_df, quality_score_before, quality_score_after)
            
            # Phase 9: Learn from execution if Agent-First was used
            if use_intelligence and self.ml_context:
                quality_score = quality_score_after
                self.adaptive_templates.learn_from_execution(
                    context={
                        'problem_type': self.ml_context.problem_type,
                        'n_samples': len(df),
                        'n_features': len(df.columns)
                    },
                    config=cleaning_config,
                    performance={'quality_score': quality_score}
                )
                logger.info(f"📊 Learned from execution (quality: {quality_score:.2f})")
            
            # Phase 10: Save configuration if enabled
            if self.config.save_yaml_config:
                self._save_yaml_config()
            
            # Phase 11: Check cost limits
            if self.config.track_usage and self.total_cost > self.config.max_cost_total:
                logger.warning(f"Cost limit exceeded: ${self.total_cost:.2f}")
            
            elapsed_time = time.time() - self.start_time
            logger.info(f"Cleaning completed in {elapsed_time:.2f} seconds")
            logger.info(f"Claude decisions made: {self.performance_metrics['claude_decisions']}")
            logger.info(f"Hybrid decisions: Agent={self.performance_metrics['hybrid_decisions']['agent']}, "
                       f"Local={self.performance_metrics['hybrid_decisions']['local']}")
            logger.info(f"Retail recommendations applied: {len(self.performance_metrics['retail_recommendations_applied'])}")
            
            return cleaned_df, self.cleaning_report
            
        except Exception as e:
            logger.error(f"Error in cleaning pipeline: {e}")
            return await self._fallback_cleaning(df, user_context)
    
    # [Other methods remain unchanged - keeping all existing functionality]
    
    def _needs_chunking(self, df: pd.DataFrame) -> bool:
        """Determine if dataset needs chunking"""
        return len(df) > self.config.chunk_size
    
    def _chunk_dataset(self, df: pd.DataFrame) -> List[pd.DataFrame]:
        """Split dataset into chunks for processing"""
        n_chunks = (len(df) + self.config.chunk_size - 1) // self.config.chunk_size
        return np.array_split(df, n_chunks)
    
    async def _process_chunk(
        self,
        chunk: pd.DataFrame,
        chunk_idx: int,
        profile_report: Dict[str, Any]
    ) -> pd.DataFrame:
        """Process a single chunk of data"""
        logger.info(f"Processing chunk {chunk_idx + 1}")
        
        # Validate chunk
        validation_report = await self.validator.validate(chunk, None)
        
        # Clean chunk
        cleaned_chunk, transformations = await self.cleaner.clean(
            chunk,
            profile_report,
            validation_report
        )
        
        # Track transformations
        self.transformations_applied.extend(transformations)
        
        return cleaned_chunk
    
    async def _evaluate_cleaning_quality(self, df_cleaned: pd.DataFrame) -> float:
        """Evaluate the quality of cleaned data"""
        try:
            # Use validator agent for quality assessment
            validation_report = await self.validator.validate(df_cleaned, None)
            
            # Calculate quality score based on validation
            quality_score = 100.0
            
            # Deduct for remaining issues
            issues = validation_report.get("issues", [])
            quality_score -= min(50, len(issues) * 5)
            
            # Deduct for remaining missing values
            missing_ratio = df_cleaned.isnull().sum().sum() / (df_cleaned.shape[0] * df_cleaned.shape[1])
            quality_score -= min(30, missing_ratio * 100)
            
            # Deduct for remaining duplicates
            duplicate_ratio = df_cleaned.duplicated().mean()
            quality_score -= min(20, duplicate_ratio * 100)
            
            return max(0, quality_score)
            
        except Exception as e:
            logger.error(f"Quality evaluation failed: {e}")
            return 50.0  # Default middle score on failure
    
    async def _fallback_cleaning(
        self,
        df: pd.DataFrame,
        user_context: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Fallback cleaning when main pipeline fails"""
        logger.warning("Using fallback cleaning strategy")
        
        cleaned_df = df.copy()
        
        # Basic cleaning
        cleaned_df = cleaned_df.drop_duplicates()
        
        # Handle missing values simply
        missing_critical_threshold = self.config.get_quality_threshold("missing_critical_threshold") or 0.5
        for col in cleaned_df.columns:
            if cleaned_df[col].isnull().mean() > missing_critical_threshold:
                cleaned_df = cleaned_df.drop(columns=[col])
            elif pd.api.types.is_numeric_dtype(cleaned_df[col]):
                cleaned_df[col].fillna(cleaned_df[col].median(), inplace=True)
            else:
                cleaned_df[col].fillna('missing', inplace=True)
        
        report = {
            "summary": {
                "mode": "fallback",
                "quality_score_after": 100 - (cleaned_df.isnull().sum().sum() / 
                                             (cleaned_df.shape[0] * cleaned_df.shape[1]) * 100),
                "message": "Fallback cleaning applied due to pipeline error"
            },
            "verdict": "Dataset cleaned with basic strategy - requires manual review"
        }
        
        return cleaned_df, report
    
    def _save_yaml_config(self):
        """Save configuration to YAML file"""
        try:
            config_dict = {
                "timestamp": datetime.now().isoformat(),
                "ml_context": self.ml_context.__dict__ if self.ml_context else None,
                "transformations": self.transformations_applied,
                "performance_metrics": self.performance_metrics,
                "cleaning_report": self.cleaning_report,
                "retail_recommendations": [
                    {"title": r["title"], "category": r["category"], "applied": True}
                    for r in self.retail_recommendations
                    if any(
                        applied["type"] in r.get("category", "")
                        for applied in self.performance_metrics.get("retail_recommendations_applied", [])
                    )
                ]
            }
            
            config_path = Path(self.config.yaml_config_path or "./configs/cleaning_config.yaml")
            config_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(config_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False)
            
            logger.info(f"Configuration saved to {config_path}")
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
    
    @async_retry(max_attempts=3, base_delay=2.0)
    @track_llm_cost('claude', 'orchestrator')
    async def recommend_cleaning_approach_with_claude(
        self,
        df: pd.DataFrame,
        profile_report: Dict[str, Any],
        ml_context: Any
    ) -> Dict[str, Any]:
        """Use Claude to recommend specific cleaning approaches"""
        if not self.use_claude or not self.config.can_call_llm('claude') or self.claude_client is None:
            return self._recommend_cleaning_approach_rule_based(df, profile_report, ml_context)
        
        logger.info("🔮 Getting cleaning recommendations from Claude...")
        
        # Extract key issues from profile report
        quality_issues = profile_report.get("quality_issues", [])[:10]
        sentinel_analysis = profile_report.get("sentinel_analysis", {})
        local_analysis = profile_report.get("local_analysis", {})
        
        prompt = f"""Recommend specific data cleaning approaches for this dataset.

Profile Summary:
- Quality Issues: {json.dumps(quality_issues, indent=2)}
- Sentinel Values Found: {len(sentinel_analysis)} columns
- High Missing Columns: {len(local_analysis.get('high_missing_columns', []))} 
- Negative Price Columns: {len(local_analysis.get('negative_price_columns', []))}

ML Context:
- Problem Type: {ml_context.problem_type if ml_context else 'unknown'}
- Business Sector: {ml_context.business_sector if ml_context else 'unknown'}

Provide specific recommendations for:
1. Handling sentinel values (especially for retail)
2. Fixing negative prices
3. Missing value imputation strategies
4. Outlier handling approaches
5. Feature engineering suggestions

Respond ONLY with valid JSON:
{{
  "sentinel_handling": {{
    "strategy": "description",
    "columns_affected": ["col1", "col2"],
    "preserve_zeros_in_stock": true/false
  }},
  "price_correction": {{
    "strategy": "median_by_category|overall_median|custom",
    "columns_affected": ["col1", "col2"]
  }},
  "missing_imputation": {{
    "numeric_strategy": "median|mean|knn|mice",
    "categorical_strategy": "mode|constant",
    "use_category_grouping": true/false
  }},
  "outlier_handling": {{
    "strategy": "clip|winsorize|remove",
    "threshold": 0.01
  }},
  "feature_engineering": ["suggestion1", "suggestion2"],
  "priority_order": ["action1", "action2", "action3"]
}}"""
        
        try:
            response = await self.claude_client.messages.create(
                model=self.claude_model,
                max_tokens=1500,
                system="You are an expert data engineer specializing in data quality and cleaning strategies.",
                messages=[{"role": "user", "content": prompt}]
            )
            
            self.config.record_llm_success('claude')
            
            recommendations = parse_llm_json(
                response.content[0].text.strip(),
                fallback=self._recommend_cleaning_approach_rule_based(df, profile_report, ml_context)
            )
            
            logger.info("🔮 Claude recommendations received successfully")
            return recommendations
            
        except Exception as e:
            logger.warning(f"⚠️ Claude recommendation failed: {e}")
            self.config.record_llm_failure('claude')
            return self._recommend_cleaning_approach_rule_based(df, profile_report, ml_context)
    
    def _recommend_cleaning_approach_rule_based(
        self,
        df: pd.DataFrame,
        profile_report: Dict[str, Any],
        ml_context: Any
    ) -> Dict[str, Any]:
        """Fallback rule-based cleaning recommendations"""
        
        recommendations = {
            "sentinel_handling": {
                "strategy": "Replace with NaN and impute",
                "columns_affected": [],
                "preserve_zeros_in_stock": True
            },
            "price_correction": {
                "strategy": "median_by_category" if "category" in df.columns else "overall_median",
                "columns_affected": []
            },
            "missing_imputation": {
                "numeric_strategy": "median",
                "categorical_strategy": "mode",
                "use_category_grouping": "category" in df.columns
            },
            "outlier_handling": {
                "strategy": "clip",
                "threshold": 0.01
            },
            "feature_engineering": [],
            "priority_order": ["sentinel_handling", "price_correction", "missing_imputation", "outlier_handling"]
        }
        
        # Detect sentinel columns
        sentinel_analysis = profile_report.get("sentinel_analysis", {})
        if sentinel_analysis:
            recommendations["sentinel_handling"]["columns_affected"] = list(sentinel_analysis.keys())
        
        # Detect price columns
        price_cols = [col for col in df.columns if 'price' in col.lower() or 'prix' in col.lower()]
        if price_cols:
            recommendations["price_correction"]["columns_affected"] = price_cols
        
        # Add ML-specific recommendations
        if ml_context and ml_context.problem_type == 'fraud_detection':
            recommendations["feature_engineering"].extend([
                "Create velocity features",
                "Add time-based aggregations",
                "Engineer transaction patterns"
            ])
        elif ml_context and ml_context.problem_type == 'sales_forecasting':
            recommendations["feature_engineering"].extend([
                "Create lag features",
                "Add seasonal indicators",
                "Engineer trend features"
            ])
        
        return recommendations
    
    async def _apply_local_cleaning_enhanced(
        self, 
        df: pd.DataFrame, 
        user_context: Dict[str, Any],
        assessment: Any
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Apply local rule-based cleaning without agents.
        Enhanced version that uses data quality assessment recommendations.
        Uses configurable thresholds.
        """
        logger.info("📋 Applying enhanced local rule-based cleaning")
        cleaned_df = df.copy()
        transformations = []
        
        sector = user_context.get('secteur_activite', 'general')
        
        # Apply retail recommendations if available
        if hasattr(assessment, 'retail_recommendations') and assessment.retail_recommendations:
            logger.info(f"📦 Applying {len(assessment.retail_recommendations)} retail recommendations")
            cleaned_df = await self._test_and_apply_strategies(cleaned_df, assessment.retail_recommendations)
            
            # Track applied recommendations
            for rec in assessment.retail_recommendations:
                transformations.append({
                    "action": f"retail_{rec['category']}",
                    "params": {"recommendation": rec['title']}
                })
        
        # Remove duplicates
        if df.duplicated().sum() > 0:
            n_duplicates = df.duplicated().sum()
            cleaned_df = cleaned_df.drop_duplicates()
            transformations.append({
                "action": "remove_duplicates",
                "params": {"rows_removed": n_duplicates}
            })
        
        # Handle sentinel values (retail-specific)
        if sector == 'retail':
            sentinel_values = self.config.retail_rules.get('sentinel_values', [-999, -1, 9999])
            for col in cleaned_df.select_dtypes(include=[np.number]).columns:
                # Use contextual sentinel check
                sentinels_mask = cleaned_df[col].apply(
                    lambda x: self.config.is_sentinel_value(x, col)
                )
                if sentinels_mask.any():
                    cleaned_df.loc[sentinels_mask, col] = np.nan
                    transformations.append({
                        "column": col,
                        "action": "replace_sentinels",
                        "params": {"count": sentinels_mask.sum()}
                    })
        
        # Handle negative prices (retail-specific)
        if sector == 'retail':
            price_columns = [col for col in cleaned_df.columns 
                           if 'price' in col.lower() or 'prix' in col.lower()]
            for col in price_columns:
                if pd.api.types.is_numeric_dtype(cleaned_df[col]):
                    negative_mask = cleaned_df[col] < 0
                    if negative_mask.any():
                        # Use median by category if available
                        if 'category' in cleaned_df.columns:
                            cleaned_df.loc[negative_mask, col] = cleaned_df.groupby('category')[col].transform(
                                lambda x: x[x > 0].median()
                            )[negative_mask]
                        else:
                            cleaned_df.loc[negative_mask, col] = cleaned_df[col][cleaned_df[col] > 0].median()
                        transformations.append({
                            "column": col,
                            "action": "fix_negative_prices",
                            "params": {"count": negative_mask.sum()}
                        })
        
        # Handle missing values using configurable thresholds
        missing_critical_threshold = self.config.get_quality_threshold("missing_critical_threshold") or 0.5
        for col in cleaned_df.columns:
            missing_ratio = cleaned_df[col].isnull().mean()
            
            if missing_ratio > missing_critical_threshold:
                cleaned_df = cleaned_df.drop(columns=[col])
                transformations.append({
                    "column": col,
                    "action": "remove_column",
                    "params": {"missing_ratio": missing_ratio}
                })
            elif missing_ratio > 0:
                if pd.api.types.is_numeric_dtype(cleaned_df[col]):
                    # Retail: impute by category if possible
                    if sector == 'retail' and 'category' in cleaned_df.columns:
                        cleaned_df[col] = cleaned_df.groupby('category')[col].transform(
                            lambda x: x.fillna(x.median())
                        )
                    else:
                        cleaned_df[col].fillna(cleaned_df[col].median(), inplace=True)
                    transformations.append({
                        "column": col,
                        "action": "fill_missing",
                        "params": {"method": "median_by_category" if sector == 'retail' else "median"}
                    })
                else:
                    cleaned_df[col].fillna('missing', inplace=True)
                    transformations.append({
                        "column": col,
                        "action": "fill_missing",
                        "params": {"method": "constant"}
                    })
        
        # Handle outliers (except for stock columns in retail)
        outlier_threshold = self.config.get_quality_threshold("outlier_warning_threshold") or 0.05
        for col in cleaned_df.select_dtypes(include=[np.number]).columns:
            if sector == 'retail' and ('stock' in col.lower() or 'qty' in col.lower() or 'quantity' in col.lower()):
                continue  # Skip outlier handling for stock columns
            
            Q1 = cleaned_df[col].quantile(0.25)
            Q3 = cleaned_df[col].quantile(0.75)
            IQR = Q3 - Q1
            outlier_mask = (cleaned_df[col] < (Q1 - 1.5 * IQR)) | (cleaned_df[col] > (Q3 + 1.5 * IQR))
            
            if outlier_mask.mean() > outlier_threshold:
                # Clip outliers
                lower = cleaned_df[col].quantile(0.01)
                upper = cleaned_df[col].quantile(0.99)
                cleaned_df[col] = cleaned_df[col].clip(lower, upper)
                transformations.append({
                    "column": col,
                    "action": "clip_outliers",
                    "params": {"count": outlier_mask.sum()}
                })
        
        # Calculate quality scores
        quality_after = 100 - (cleaned_df.isnull().sum().sum() / (cleaned_df.shape[0] * cleaned_df.shape[1]) * 100)
        
        # Calculate GS1 compliance for retail
        gs1_compliance = 100
        if sector == 'retail':
            # Simplified GS1 compliance calculation
            sku_columns = [col for col in cleaned_df.columns if any(
                keyword in col.lower() for keyword in ['sku', 'upc', 'ean', 'gtin', 'barcode']
            )]
            if sku_columns:
                valid_skus = 0
                total_skus = 0
                for col in sku_columns:
                    if cleaned_df[col].dtype == 'object':
                        values = cleaned_df[col].dropna()
                        total_skus += len(values)
                        valid_skus += sum(1 for v in values if str(v).isdigit() and len(str(v)) in [8, 12, 13, 14])
                if total_skus > 0:
                    gs1_compliance = (valid_skus / total_skus) * 100
        
        report = {
            "mode": "local_enhanced",
            "quality_before": self.local_stats['quality_score'],
            "quality_after": quality_after,
            "transformations": transformations,
            "retail_recommendations_applied": len(self.performance_metrics['retail_recommendations_applied']),
            "metrics": {
                "sentinels_removed": sum(1 for t in transformations if t.get("action") == "replace_sentinels"),
                "negative_prices_fixed": sum(1 for t in transformations if t.get("action") == "fix_negative_prices"),
                "missing_imputed": sum(1 for t in transformations if t.get("action") == "fill_missing"),
                "outliers_clipped": sum(1 for t in transformations if t.get("action") == "clip_outliers"),
                "columns_removed": sum(1 for t in transformations if t.get("action") == "remove_column")
            },
            "gs1_compliance": gs1_compliance if sector == 'retail' else None
        }
        
        return cleaned_df, report
