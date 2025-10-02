"""
Profiler Agent - Analyzes dataset quality and statistics using OpenAI Assistant
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
                # Create new assistant
                self.assistant = await self.client.beta.assistants.create(
                    name="Data Profiler Agent",
                    instructions=PROFILER_SYSTEM_PROMPT,
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
                
    async def analyze(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze dataset and generate comprehensive profile report
        
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
            
            # Add message with data summary
            message_content = PROFILER_USER_PROMPT.format(
                data_summary=json.dumps(data_summary, indent=2),
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
        """Prepare data summary for the assistant"""
        summary = {
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "sample_data": df.head(10).to_dict(orient='records'),
            "basic_stats": {}
        }
        
        # Add basic statistics
        for col in df.columns:
            col_stats = {
                "dtype": str(df[col].dtype),
                "null_count": int(df[col].isnull().sum()),
                "null_percentage": float(df[col].isnull().mean() * 100),
                "unique_count": int(df[col].nunique()),
                "unique_percentage": float(df[col].nunique() / len(df) * 100)
            }
            
            # Add numeric statistics
            if pd.api.types.is_numeric_dtype(df[col]):
                col_stats.update({
                    "mean": float(df[col].mean()) if not df[col].isnull().all() else None,
                    "std": float(df[col].std()) if not df[col].isnull().all() else None,
                    "min": float(df[col].min()) if not df[col].isnull().all() else None,
                    "max": float(df[col].max()) if not df[col].isnull().all() else None,
                    "q25": float(df[col].quantile(0.25)) if not df[col].isnull().all() else None,
                    "q50": float(df[col].quantile(0.50)) if not df[col].isnull().all() else None,
                    "q75": float(df[col].quantile(0.75)) if not df[col].isnull().all() else None
                })
            
            # Add categorical statistics
            elif df[col].dtype == 'object':
                value_counts = df[col].value_counts().head(10)
                col_stats["top_values"] = value_counts.to_dict()
                col_stats["mode"] = df[col].mode()[0] if not df[col].mode().empty else None
            
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
        
        return summary
    
    async def _wait_for_run_completion(self, thread_id: str, run_id: str) -> Dict[str, Any]:
        """Wait for assistant run to complete"""
        start_time = time.time()
        
        while time.time() - start_time < self.config.timeout_seconds:
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
                        return json.loads(match)
                    except json.JSONDecodeError:
                        continue
            
            # If no valid JSON, create structured report from content
            return self._structure_text_report(content, df)
            
        except Exception as e:
            logger.error(f"Failed to parse profiling results: {e}")
            return self._basic_profiling(df)
    
    def _structure_text_report(self, content: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Structure text report into dictionary"""
        report = {
            "summary": {
                "total_rows": len(df),
                "total_columns": len(df.columns),
                "memory_usage_mb": df.memory_usage(deep=True).sum() / (1024 * 1024)
            },
            "quality_issues": [],
            "anomalies": [],
            "recommendations": [],
            "raw_analysis": content
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
        """Fallback basic profiling without OpenAI"""
        logger.info("Using basic profiling fallback")
        
        report = {
            "summary": {
                "total_rows": len(df),
                "total_columns": len(df.columns),
                "memory_usage_mb": df.memory_usage(deep=True).sum() / (1024 * 1024),
                "duplicates": df.duplicated().sum(),
                "missing_cells": df.isnull().sum().sum()
            },
            "columns": {},
            "quality_issues": [],
            "anomalies": []
        }
        
        # Analyze each column
        for col in df.columns:
            col_report = {
                "dtype": str(df[col].dtype),
                "missing": int(df[col].isnull().sum()),
                "missing_percent": float(df[col].isnull().mean() * 100),
                "unique": int(df[col].nunique()),
                "unique_percent": float(df[col].nunique() / len(df) * 100)
            }
            
            # Check for issues
            if col_report["missing_percent"] > 50:
                report["quality_issues"].append(f"Column '{col}' has {col_report['missing_percent']:.1f}% missing values")
            
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
                    if outlier_percent > 5:
                        report["anomalies"].append(f"Column '{col}' has {outlier_percent:.1f}% outliers")
                
                col_report.update({
                    "mean": float(df[col].mean()) if not df[col].isnull().all() else None,
                    "std": float(df[col].std()) if not df[col].isnull().all() else None,
                    "min": float(df[col].min()) if not df[col].isnull().all() else None,
                    "max": float(df[col].max()) if not df[col].isnull().all() else None,
                    "outliers": int(outliers)
                })
            
            report["columns"][col] = col_report
        
        # Check for high cardinality
        for col in df.select_dtypes(include=['object']).columns:
            cardinality = df[col].nunique()
            if cardinality > 100:
                report["quality_issues"].append(f"Column '{col}' has high cardinality ({cardinality} unique values)")
        
        return report
    
    async def generate_quality_score(self, df: pd.DataFrame) -> float:
        """Generate a quality score for the dataset"""
        profile = await self.analyze(df)
        
        score = 100.0
        
        # Deduct for missing values
        missing_percent = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
        score -= min(30, missing_percent)
        
        # Deduct for duplicates
        duplicate_percent = (df.duplicated().sum() / len(df)) * 100
        score -= min(20, duplicate_percent)
        
        # Deduct for quality issues
        issues = profile.get("quality_issues", [])
        score -= min(30, len(issues) * 5)
        
        # Deduct for anomalies
        anomalies = profile.get("anomalies", [])
        score -= min(20, len(anomalies) * 3)
        
        return max(0, score)
