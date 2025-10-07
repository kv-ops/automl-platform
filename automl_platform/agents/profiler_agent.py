"""
Profiler Agent - Analyzes dataset quality and statistics using OpenAI Assistant
Enhanced with configurable thresholds and retail-specific sentinel detection
FINALIZED: Uses configuration for all thresholds
"""

import pandas as pd
import numpy as np
import json
import logging
import asyncio
import importlib.util
from typing import Dict, Any, Optional, TYPE_CHECKING
import time

from .agent_config import AgentConfig, AgentType
from .prompts.profiler_prompts import PROFILER_SYSTEM_PROMPT, PROFILER_USER_PROMPT

_openai_spec = importlib.util.find_spec("openai")
if _openai_spec is not None:
    from openai import AsyncOpenAI  # type: ignore
else:
    AsyncOpenAI = None  # type: ignore[assignment]

if TYPE_CHECKING:  # pragma: no cover - typing helper
    from openai import AsyncOpenAI as _AsyncOpenAIType

logger = logging.getLogger(__name__)


class ProfilerAgent:
    """
    Agent responsible for profiling datasets and generating quality reports
    Uses OpenAI Assistant with Code Interpreter
    Enhanced with configurable missing threshold and retail-specific rules
    FINALIZED: All thresholds from configuration
    """
    
    def __init__(self, config: AgentConfig):
        """Initialize Profiler Agent"""
        self.config = config
        if AsyncOpenAI is not None and config.openai_api_key:
            self.client = AsyncOpenAI(api_key=config.openai_api_key)
        else:
            self.client = None
            if AsyncOpenAI is None:
                logger.warning(
                    "AsyncOpenAI client unavailable because the 'openai' package is not installed. "
                    "The ProfilerAgent will fall back to local profiling routines."
                )
            else:
                logger.warning("OpenAI API key missing; ProfilerAgent will use local profiling only.")
        self.assistant = None
        self.assistant_id = config.get_assistant_id(AgentType.PROFILER)
        
        # FIXED: Lazy initialization - no automatic init
        self._init_lock = asyncio.Lock()
        self._initialized = False

    
    async def _initialize_assistant(self):
        """Create or retrieve OpenAI Assistant"""
        try:
            if self.client is None:
                logger.debug("ProfilerAgent _initialize_assistant skipped because client is unavailable.")
                return

            if self.assistant_id:
                # Retrieve existing assistant
                self.assistant = await self.client.beta.assistants.retrieve(
                    assistant_id=self.assistant_id
                )
                logger.info(f"Retrieved existing Profiler assistant: {self.assistant_id}")
            else:
                # Create new assistant with enhanced prompts including thresholds
                missing_warning = self.config.get_quality_threshold('missing_warning_threshold') or 0.35
                missing_critical = self.config.get_quality_threshold('missing_critical_threshold') or 0.50
                outlier_warning = self.config.get_quality_threshold('outlier_warning_threshold') or 0.05
                outlier_critical = self.config.get_quality_threshold('outlier_critical_threshold') or 0.15
                
                enhanced_prompt = PROFILER_SYSTEM_PROMPT + f"\n\nConfigured thresholds:\n"
                enhanced_prompt += f"- Missing warning: {missing_warning*100}%\n"
                enhanced_prompt += f"- Missing critical: {missing_critical*100}%\n"
                enhanced_prompt += f"- Outlier warning: {outlier_warning*100}%\n"
                enhanced_prompt += f"- Outlier critical: {outlier_critical*100}%"
                
                self.assistant = await self.client.beta.assistants.create(
                    name="Data Profiler Agent",
                    instructions=enhanced_prompt,
                    model=self.config.openai_model,
                    tools=self.config.get_agent_tools(AgentType.PROFILER)
                )
                self.assistant_id = self.assistant.id
                self.config.save_assistant_id(AgentType.PROFILER, self.assistant_id)
                logger.info(f"Created new Profiler assistant: {self.assistant_id}")
                
        except Exception as e:
            logger.error(f"Failed to initialize assistant: {e}")

    async def _ensure_assistant_initialized(self):
        """Thread-safe initialization with double-check locking"""
        if self.client is None or self._initialized:
            return

        async with self._init_lock:
            if self._initialized:  # Double-check
                return
            
            await self._initialize_assistant()
            self._initialized = True
    
    def _is_stock_column(self, col_name: str) -> bool:
        """Check if column is likely a stock/quantity column"""
        stock_keywords = ['stock', 'quantity', 'qty', 'inventory', 'count', 'units', 'items']
        col_lower = col_name.lower()
        return any(keyword in col_lower for keyword in stock_keywords)
    
    def _get_effective_sentinel_values(self, col_name: str) -> list:
        """Get sentinel values for a column, excluding 0 for stock columns"""
        sentinels = self.config.get_retail_rules('sentinel_values') or [-999, -1, 9999]
        sentinels = sentinels.copy()
        
        # Exclude 0 from sentinels for stock/quantity columns
        if self._is_stock_column(col_name) and 0 in sentinels:
            sentinels.remove(0)
            logger.debug(f"Column '{col_name}' identified as stock/quantity - excluding 0 from sentinels")
        
        return sentinels
                
    async def analyze(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze dataset and generate comprehensive profile report
        Enhanced with configurable thresholds and retail-specific checks
        
        Args:
            df: Input dataframe
            
        Returns:
            Dictionary containing profiling results
        """
        try:
            if self.client is None:
                logger.info("ProfilerAgent using basic profiling because OpenAI client is unavailable.")
                return self._basic_profiling(df)

            # Ensure assistant is initialized
            await self._ensure_assistant_initialized()
            
            # Prepare data summary for the assistant
            data_summary = self._prepare_data_summary(df)
            
            # Create a thread
            thread = await self.client.beta.threads.create()
            
            # Add message with data summary and enhanced context
            thresholds_info = {
                'missing_warning_threshold': self.config.get_quality_threshold('missing_warning_threshold') or 0.35,
                'missing_critical_threshold': self.config.get_quality_threshold('missing_critical_threshold') or 0.50,
                'outlier_warning_threshold': self.config.get_quality_threshold('outlier_warning_threshold') or 0.05,
                'outlier_critical_threshold': self.config.get_quality_threshold('outlier_critical_threshold') or 0.15,
                'high_cardinality_threshold': self.config.get_quality_threshold('high_cardinality_threshold') or 0.90,
                'sentinel_values': self.config.get_retail_rules('sentinel_values') or [-999, -1, 9999]
            }
            
            message_content = PROFILER_USER_PROMPT.format(
                data_summary=json.dumps(data_summary, indent=2),
                thresholds=json.dumps(thresholds_info, indent=2),
                sector=self.config.user_context.get("secteur_activite", "general"),
                target=self.config.user_context.get("target_variable", "unknown")
            )
            
            await self.client.beta.threads.messages.create(
                thread_id=thread.id,
                role="user",
                content=message_content
            )
            
            # Run the assistant
            run = await self.client.beta.threads.runs.create(
                thread_id=thread.id,
                assistant_id=self.assistant_id
            )
            
            # Wait for completion with timeout
            result = await self._wait_for_run_completion(thread.id, run.id)
            
            # Parse and return results
            return self._parse_profiling_results(result, df)
            
        except Exception as e:
            logger.error(f"Error in profiling: {e}")
            # Fallback to basic profiling
            return self._basic_profiling(df)
    
    def _prepare_data_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Prepare data summary for the assistant with enhanced metrics"""
        missing_warning = self.config.get_quality_threshold('missing_warning_threshold') or 0.35
        missing_critical = self.config.get_quality_threshold('missing_critical_threshold') or 0.50
        outlier_warning = self.config.get_quality_threshold('outlier_warning_threshold') or 0.05
        outlier_critical = self.config.get_quality_threshold('outlier_critical_threshold') or 0.15
        high_cardinality = self.config.get_quality_threshold('high_cardinality_threshold') or 0.90
        sentinel_values = self.config.get_retail_rules('sentinel_values') or [-999, -1, 9999]
        
        summary = {
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "sample_data": df.head(10).to_dict(orient='records'),
            "basic_stats": {},
            "sentinel_analysis": {},
            "thresholds_configured": {
                "missing_warning_threshold": missing_warning,
                "missing_critical_threshold": missing_critical,
                "outlier_warning_threshold": outlier_warning,
                "outlier_critical_threshold": outlier_critical,
                "high_cardinality_threshold": high_cardinality,
                "sentinel_values": sentinel_values
            }
        }
        
        # Add basic statistics with enhanced checks
        for col in df.columns:
            effective_sentinels = self._get_effective_sentinel_values(col)
            
            col_stats = {
                "dtype": str(df[col].dtype),
                "null_count": int(df[col].isnull().sum()),
                "null_percentage": float(df[col].isnull().mean() * 100),
                "unique_count": int(df[col].nunique()),
                "unique_percentage": float(df[col].nunique() / len(df) * 100),
                "high_missing_warning": df[col].isnull().mean() > missing_warning,
                "high_missing_critical": df[col].isnull().mean() > missing_critical,
                "is_stock_column": self._is_stock_column(col)
            }
            
            # Check for sentinel values in numeric columns
            if pd.api.types.is_numeric_dtype(df[col]):
                sentinel_count = df[col].isin(effective_sentinels).sum()
                if sentinel_count > 0:
                    summary["sentinel_analysis"][col] = {
                        "count": int(sentinel_count),
                        "percentage": float(sentinel_count / len(df) * 100),
                        "values_found": list(df[col][df[col].isin(effective_sentinels)].unique()),
                        "is_stock_column": self._is_stock_column(col),
                        "zero_excluded": self._is_stock_column(col) and 0 in sentinel_values
                    }
                
                col_stats.update({
                    "mean": float(df[col].mean()) if not df[col].isnull().all() else None,
                    "std": float(df[col].std()) if not df[col].isnull().all() else None,
                    "min": float(df[col].min()) if not df[col].isnull().all() else None,
                    "max": float(df[col].max()) if not df[col].isnull().all() else None,
                    "q25": float(df[col].quantile(0.25)) if not df[col].isnull().all() else None,
                    "q50": float(df[col].quantile(0.50)) if not df[col].isnull().all() else None,
                    "q75": float(df[col].quantile(0.75)) if not df[col].isnull().all() else None,
                    "has_negative": bool((df[col] < 0).any()) if not df[col].isnull().all() else False
                })
                
                # Check for negative prices specifically
                if 'price' in col.lower() or 'prix' in col.lower():
                    negative_count = (df[col] < 0).sum()
                    if negative_count > 0:
                        col_stats["negative_price_count"] = int(negative_count)
                        col_stats["negative_price_percentage"] = float(negative_count / len(df) * 100)
            
            # Add categorical statistics
            elif df[col].dtype == 'object':
                value_counts = df[col].value_counts().head(10)
                col_stats["top_values"] = value_counts.to_dict()
                col_stats["mode"] = df[col].mode()[0] if not df[col].mode().empty else None
                # Check for high cardinality
                col_stats["high_cardinality"] = df[col].nunique() / len(df) > high_cardinality
            
            summary["basic_stats"][col] = col_stats
        
        # Add correlation matrix for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            summary["correlations"] = corr_matrix.to_dict()
        
        # Add duplicate information
        summary["duplicates"] = {
            "total": int(df.duplicated().sum()),
            "percentage": float(df.duplicated().mean() * 100)
        }
        
        # Add complexity score for hybrid decision
        summary["complexity_score"] = self._calculate_complexity_score(df, summary)
        
        return summary
    
    def _calculate_complexity_score(self, df: pd.DataFrame, summary: Dict[str, Any]) -> float:
        """Calculate complexity score for hybrid mode decision"""
        score = 0.0
        
        missing_warning = self.config.get_quality_threshold('missing_warning_threshold') or 0.35
        missing_critical = self.config.get_quality_threshold('missing_critical_threshold') or 0.50
        high_cardinality_threshold = self.config.get_quality_threshold('high_cardinality_threshold') or 0.90
        
        # High missing values increase complexity
        missing_ratio = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
        if missing_ratio > missing_critical:
            score += 0.3
        elif missing_ratio > missing_warning:
            score += 0.15
        
        # Many columns increase complexity
        if df.shape[1] > 50:
            score += 0.2
        elif df.shape[1] > 20:
            score += 0.1
        
        # Sentinel values increase complexity
        if summary.get("sentinel_analysis"):
            score += 0.2
        
        # High cardinality increases complexity
        high_card_count = sum(1 for col in df.columns 
                             if df[col].nunique() > high_cardinality_threshold * len(df))
        if high_card_count > 5:
            score += 0.2
        
        return min(1.0, score)
    
    async def _wait_for_run_completion(self, thread_id: str, run_id: str) -> Dict[str, Any]:
        """Wait for assistant run to complete"""
        start_time = time.time()
        
        while time.time() - start_time < self.config.openai_timeout_seconds:
            run_status = await self.client.beta.threads.runs.retrieve(
                thread_id=thread_id,
                run_id=run_id
            )
            
            if run_status.status == 'completed':
                # Get messages
                messages = await self.client.beta.threads.messages.list(
                    thread_id=thread_id
                )
                
                # Extract assistant's response
                for message in messages.data:
                    if message.role == 'assistant':
                        return {"content": message.content[0].text.value}
                
            elif run_status.status == 'failed':
                raise Exception(f"Run failed: {run_status.last_error}")
            
            # Wait before checking again
            await asyncio.sleep(2)
        
        raise TimeoutError("Assistant run timed out")
    
    def _parse_profiling_results(self, result: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
        """Parse results from the assistant"""
        try:
            # Try to extract JSON from the response
            content = result.get("content", "")
            
            # Look for JSON blocks in the content
            import re
            json_pattern = r'\{[\s\S]*\}'
            json_matches = re.findall(json_pattern, content)
            
            if json_matches:
                # Parse the largest JSON block
                for match in sorted(json_matches, key=len, reverse=True):
                    try:
                        parsed_result = json.loads(match)
                        # Enhance with local analysis
                        parsed_result["local_analysis"] = self._enhance_with_local_analysis(df)
                        parsed_result["thresholds_used"] = {
                            "missing_warning_threshold": self.config.get_quality_threshold('missing_warning_threshold') or 0.35,
                            "missing_critical_threshold": self.config.get_quality_threshold('missing_critical_threshold') or 0.50,
                            "outlier_warning_threshold": self.config.get_quality_threshold('outlier_warning_threshold') or 0.05,
                            "outlier_critical_threshold": self.config.get_quality_threshold('outlier_critical_threshold') or 0.15,
                            "high_cardinality_threshold": self.config.get_quality_threshold('high_cardinality_threshold') or 0.90,
                            "sentinel_values": self.config.get_retail_rules('sentinel_values') or [-999, -1, 9999]
                        }
                        return parsed_result
                    except json.JSONDecodeError:
                        continue
            
            # If no valid JSON, create structured report from content
            return self._structure_text_report(content, df)
            
        except Exception as e:
            logger.error(f"Failed to parse profiling results: {e}")
            return self._basic_profiling(df)
    
    def _enhance_with_local_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Add local analysis to complement agent results"""
        missing_warning = self.config.get_quality_threshold('missing_warning_threshold') or 0.35
        missing_critical = self.config.get_quality_threshold('missing_critical_threshold') or 0.50
        high_cardinality_threshold = self.config.get_quality_threshold('high_cardinality_threshold') or 0.90
        
        analysis = {
            "high_missing_columns": [],
            "sentinel_columns": [],
            "negative_price_columns": [],
            "recommended_for_local": True,
            "threshold_exceeded": [],
            "high_cardinality_columns": []
        }
        
        # Check high missing columns against configured thresholds
        for col in df.columns:
            missing_pct = df[col].isnull().mean()
            if missing_pct > missing_critical:
                analysis["high_missing_columns"].append({
                    "column": col,
                    "missing_percentage": float(missing_pct * 100),
                    "threshold": float(missing_critical * 100),
                    "severity": "critical"
                })
                analysis["threshold_exceeded"].append(f"{col}: {missing_pct*100:.1f}% missing (>{missing_critical*100:.0f}% critical)")
            elif missing_pct > missing_warning:
                analysis["high_missing_columns"].append({
                    "column": col,
                    "missing_percentage": float(missing_pct * 100),
                    "threshold": float(missing_warning * 100),
                    "severity": "warning"
                })
                analysis["threshold_exceeded"].append(f"{col}: {missing_pct*100:.1f}% missing (>{missing_warning*100:.0f}% warning)")
        
        # Check sentinel values with context-aware detection
        for col in df.select_dtypes(include=[np.number]).columns:
            effective_sentinels = self._get_effective_sentinel_values(col)
            sentinel_count = df[col].isin(effective_sentinels).sum()
            if sentinel_count > 0:
                analysis["sentinel_columns"].append({
                    "column": col,
                    "count": int(sentinel_count),
                    "is_stock": self._is_stock_column(col),
                    "sentinels_checked": effective_sentinels
                })
        
        # Check negative prices
        price_columns = [col for col in df.columns if 'price' in col.lower() or 'prix' in col.lower()]
        for col in price_columns:
            if pd.api.types.is_numeric_dtype(df[col]) and (df[col] < 0).any():
                analysis["negative_price_columns"].append({
                    "column": col,
                    "count": int((df[col] < 0).sum())
                })
        
        # Check high cardinality
        for col in df.columns:
            cardinality_ratio = df[col].nunique() / len(df)
            if cardinality_ratio > high_cardinality_threshold:
                analysis["high_cardinality_columns"].append({
                    "column": col,
                    "unique_ratio": float(cardinality_ratio),
                    "unique_count": int(df[col].nunique())
                })
        
        # Determine if local cleaning is recommended
        complexity_indicators = [
            len(analysis["high_missing_columns"]) > 3,
            len(analysis["sentinel_columns"]) > 5,
            df.shape[1] > 50,
            df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) > 0.4
        ]
        
        analysis["recommended_for_local"] = not any(complexity_indicators)
        
        return analysis
    
    def _structure_text_report(self, content: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Structure text report into dictionary with enhanced metrics"""
        report = {
            "summary": {
                "total_rows": len(df),
                "total_columns": len(df.columns),
                "memory_usage_mb": df.memory_usage(deep=True).sum() / (1024 * 1024),
                "missing_warning_threshold": self.config.get_quality_threshold('missing_warning_threshold') or 0.35,
                "missing_critical_threshold": self.config.get_quality_threshold('missing_critical_threshold') or 0.50,
                "thresholds_configured": {
                    "missing_warning_threshold": self.config.get_quality_threshold('missing_warning_threshold') or 0.35,
                    "missing_critical_threshold": self.config.get_quality_threshold('missing_critical_threshold') or 0.50,
                    "outlier_warning_threshold": self.config.get_quality_threshold('outlier_warning_threshold') or 0.05,
                    "outlier_critical_threshold": self.config.get_quality_threshold('outlier_critical_threshold') or 0.15,
                    "high_cardinality_threshold": self.config.get_quality_threshold('high_cardinality_threshold') or 0.90,
                    "sentinel_values": self.config.get_retail_rules('sentinel_values') or [-999, -1, 9999]
                }
            },
            "quality_issues": [],
            "anomalies": [],
            "recommendations": [],
            "raw_analysis": content,
            "local_analysis": self._enhance_with_local_analysis(df)
        }
        
        # Extract issues from content using simple pattern matching
        lines = content.split('\n')
        current_section = None
        
        for line in lines:
            line_lower = line.lower()
            
            if 'issue' in line_lower or 'problem' in line_lower:
                current_section = 'issues'
            elif 'anomal' in line_lower or 'outlier' in line_lower:
                current_section = 'anomalies'
            elif 'recommend' in line_lower or 'suggest' in line_lower:
                current_section = 'recommendations'
            elif current_section and line.strip().startswith('-'):
                item = line.strip('- ').strip()
                if current_section == 'issues':
                    report["quality_issues"].append(item)
                elif current_section == 'anomalies':
                    report["anomalies"].append(item)
                elif current_section == 'recommendations':
                    report["recommendations"].append(item)
        
        return report
    
    def _basic_profiling(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Fallback basic profiling without OpenAI - Enhanced with configurable thresholds"""
        logger.info("Using basic profiling fallback")
        
        missing_warning = self.config.get_quality_threshold('missing_warning_threshold') or 0.35
        missing_critical = self.config.get_quality_threshold('missing_critical_threshold') or 0.50
        outlier_warning = self.config.get_quality_threshold('outlier_warning_threshold') or 0.05
        outlier_critical = self.config.get_quality_threshold('outlier_critical_threshold') or 0.15
        high_cardinality_threshold = self.config.get_quality_threshold('high_cardinality_threshold') or 0.90
        sentinel_values = self.config.get_retail_rules('sentinel_values') or [-999, -1, 9999]
        
        report = {
            "summary": {
                "total_rows": len(df),
                "total_columns": len(df.columns),
                "memory_usage_mb": df.memory_usage(deep=True).sum() / (1024 * 1024),
                "duplicates": df.duplicated().sum(),
                "missing_cells": df.isnull().sum().sum(),
                "missing_warning_threshold": missing_warning,
                "missing_critical_threshold": missing_critical,
                "thresholds_configured": {
                    "missing_warning_threshold": missing_warning,
                    "missing_critical_threshold": missing_critical,
                    "outlier_warning_threshold": outlier_warning,
                    "outlier_critical_threshold": outlier_critical,
                    "high_cardinality_threshold": high_cardinality_threshold,
                    "sentinel_values": sentinel_values
                }
            },
            "columns": {},
            "quality_issues": [],
            "anomalies": [],
            "sentinel_analysis": {},
            "local_analysis": self._enhance_with_local_analysis(df)
        }
        
        # Analyze each column with enhanced checks
        for col in df.columns:
            effective_sentinels = self._get_effective_sentinel_values(col)
            
            col_report = {
                "dtype": str(df[col].dtype),
                "missing": int(df[col].isnull().sum()),
                "missing_percent": float(df[col].isnull().mean() * 100),
                "unique": int(df[col].nunique()),
                "unique_percent": float(df[col].nunique() / len(df) * 100),
                "is_stock_column": self._is_stock_column(col)
            }
            
            # Check for issues with configurable thresholds
            if col_report["missing_percent"] > missing_critical * 100:
                report["quality_issues"].append(f"Column '{col}' has {col_report['missing_percent']:.1f}% missing values (critical, exceeds {missing_critical*100:.0f}%)")
            elif col_report["missing_percent"] > missing_warning * 100:
                report["quality_issues"].append(f"Column '{col}' has {col_report['missing_percent']:.1f}% missing values (warning, exceeds {missing_warning*100:.0f}%)")
            
            if col_report["unique"] == 1:
                report["quality_issues"].append(f"Column '{col}' is constant")
            
            # Detect outliers for numeric columns
            if pd.api.types.is_numeric_dtype(df[col]):
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
                
                if outliers > 0:
                    outlier_percent = (outliers / len(df)) * 100
                    if outlier_percent > outlier_critical * 100:
                        report["anomalies"].append(f"Column '{col}' has {outlier_percent:.1f}% outliers (critical, exceeds {outlier_critical*100:.0f}%)")
                    elif outlier_percent > outlier_warning * 100:
                        report["anomalies"].append(f"Column '{col}' has {outlier_percent:.1f}% outliers (warning, exceeds {outlier_warning*100:.0f}%)")
                
                col_report.update({
                    "mean": float(df[col].mean()) if not df[col].isnull().all() else None,
                    "std": float(df[col].std()) if not df[col].isnull().all() else None,
                    "min": float(df[col].min()) if not df[col].isnull().all() else None,
                    "max": float(df[col].max()) if not df[col].isnull().all() else None,
                    "outliers": int(outliers)
                })
                
                # Check for sentinel values with context
                sentinels_found = df[col].isin(effective_sentinels).sum()
                if sentinels_found > 0:
                    report["sentinel_analysis"][col] = {
                        "count": int(sentinels_found),
                        "values": list(df[col][df[col].isin(effective_sentinels)].unique()),
                        "is_stock": self._is_stock_column(col),
                        "sentinels_used": effective_sentinels
                    }
                    report["quality_issues"].append(f"Column '{col}' contains {sentinels_found} sentinel values (checked: {effective_sentinels})")
                
                # Check for negative prices
                if 'price' in col.lower() or 'prix' in col.lower():
                    negative_count = (df[col] < 0).sum()
                    if negative_count > 0:
                        report["quality_issues"].append(f"Column '{col}' has {negative_count} negative prices")
                        col_report["negative_prices"] = int(negative_count)
            
            report["columns"][col] = col_report
        
        # Check for high cardinality
        for col in df.select_dtypes(include=['object']).columns:
            cardinality_ratio = df[col].nunique() / len(df)
            if cardinality_ratio > high_cardinality_threshold:
                report["quality_issues"].append(f"Column '{col}' has high cardinality ({df[col].nunique()} unique values, {cardinality_ratio*100:.1f}% unique)")
        
        return report
    
    async def generate_quality_score(self, df: pd.DataFrame) -> float:
        """Generate a quality score for the dataset with configurable threshold awareness"""
        profile = await self.analyze(df)
        
        missing_warning = self.config.get_quality_threshold('missing_warning_threshold') or 0.35
        missing_critical = self.config.get_quality_threshold('missing_critical_threshold') or 0.50
        
        score = 100.0
        
        # Deduct for missing values (scaled by threshold)
        missing_percent = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
        if missing_percent > missing_critical * 100:
            score -= min(40, missing_percent)  # Higher penalty for exceeding critical threshold
        elif missing_percent > missing_warning * 100:
            score -= min(30, missing_percent)  # Medium penalty for exceeding warning threshold
        else:
            score -= min(20, missing_percent)  # Lower penalty below threshold
        
        # Deduct for duplicates
        duplicate_percent = (df.duplicated().sum() / len(df)) * 100
        score -= min(20, duplicate_percent)
        
        # Deduct for quality issues
        issues = profile.get("quality_issues", [])
        score -= min(30, len(issues) * 5)
        
        # Deduct for anomalies
        anomalies = profile.get("anomalies", [])
        score -= min(20, len(anomalies) * 3)
        
        # Deduct for sentinel values
        sentinel_analysis = profile.get("sentinel_analysis", {})
        if sentinel_analysis:
            score -= min(10, len(sentinel_analysis) * 2)
        
        return max(0, score)
