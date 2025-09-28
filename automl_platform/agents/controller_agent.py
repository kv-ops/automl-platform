"""
Controller Agent - Validates cleaning results and generates final reports
"""

import pandas as pd
import numpy as np
import json
import logging
import asyncio
import importlib.util
from typing import Dict, Any, List, Optional, TYPE_CHECKING
import time
from datetime import datetime

from .agent_config import AgentConfig, AgentType
from .prompts.controller_prompts import CONTROLLER_SYSTEM_PROMPT, CONTROLLER_USER_PROMPT

_openai_spec = importlib.util.find_spec("openai")
if _openai_spec is not None:
    from openai import AsyncOpenAI  # type: ignore
else:
    AsyncOpenAI = None  # type: ignore[assignment]

if TYPE_CHECKING:  # pragma: no cover
    from openai import AsyncOpenAI as _AsyncOpenAIType

logger = logging.getLogger(__name__)


class ControllerAgent:
    """
    Agent responsible for final validation and quality control
    Uses OpenAI Assistant with Code Interpreter
    """
    
    def __init__(self, config: AgentConfig):
        """Initialize Controller Agent"""
        self.config = config
        if AsyncOpenAI is not None and config.openai_api_key:
            self.client = AsyncOpenAI(api_key=config.openai_api_key)
        else:
            self.client = None
            if AsyncOpenAI is None:
                logger.warning(
                    "AsyncOpenAI client unavailable because the 'openai' package is not installed. "
                    "ControllerAgent will rely on local validation metrics."
                )
            else:
                logger.warning("OpenAI API key missing; ControllerAgent will rely on local validation metrics only.")
        self.assistant = None
        self.assistant_id = config.get_assistant_id(AgentType.CONTROLLER)
        
        # Quality metrics tracking
        self.quality_metrics = {}
        
        # Assistant initialization tracking
        self._initialization_task: Optional[asyncio.Task] = None
        self._initialization_lock: Optional[asyncio.Lock] = None

        if self.client is not None:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop is not None and loop.is_running():
                self._initialization_task = loop.create_task(self._initialize_assistant())
    
    async def _initialize_assistant(self):
        """Create or retrieve OpenAI Assistant"""
        try:
            if self.client is None:
                logger.debug("ControllerAgent _initialize_assistant skipped because client is unavailable.")
                return

            if self.assistant_id:
                self.assistant = await self.client.beta.assistants.retrieve(
                    assistant_id=self.assistant_id
                )
                logger.info(f"Retrieved existing Controller assistant: {self.assistant_id}")
            else:
                self.assistant = await self.client.beta.assistants.create(
                    name="Data Controller Agent",
                    instructions=CONTROLLER_SYSTEM_PROMPT,
                    model=self.config.model,
                    tools=self.config.get_agent_tools(AgentType.CONTROLLER)
                )
                self.assistant_id = self.assistant.id
                self.config.save_assistant_id(AgentType.CONTROLLER, self.assistant_id)
                logger.info(f"Created new Controller assistant: {self.assistant_id}")
                
        except Exception as e:
            logger.error(f"Failed to initialize controller assistant: {e}")

    async def _ensure_assistant_initialized(self):
        if self.client is None or self.assistant:
            return

        if self._initialization_lock is None:
            self._initialization_lock = asyncio.Lock()

        async with self._initialization_lock:
            if self.assistant:
                return

            if self._initialization_task is not None:
                await self._initialization_task
                self._initialization_task = None
            else:
                await self._initialize_assistant()
    
    async def validate(
        self,
        cleaned_df: pd.DataFrame,
        original_df: pd.DataFrame,
        transformations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Validate cleaned data and generate final quality report
        
        Args:
            cleaned_df: Cleaned dataframe
            original_df: Original dataframe
            transformations: List of applied transformations
            
        Returns:
            Dictionary containing validation results and metrics
        """
        try:
            if self.client is None:
                logger.info("ControllerAgent using basic validation because OpenAI client is unavailable.")
                metrics = self._calculate_quality_metrics(cleaned_df, original_df)
                self.quality_metrics = metrics
                return self._basic_validation(cleaned_df, original_df, metrics)

            # Ensure assistant is initialized
            await self._ensure_assistant_initialized()
            
            # Calculate quality metrics
            metrics = self._calculate_quality_metrics(cleaned_df, original_df)
            
            # Prepare validation context
            validation_context = self._prepare_validation_context(
                cleaned_df, original_df, transformations, metrics
            )
            
            # Create a thread
            thread = await self.client.beta.threads.create()
            
            # Create validation request
            message_content = CONTROLLER_USER_PROMPT.format(
                original_summary=json.dumps(validation_context["original_summary"], indent=2),
                cleaned_summary=json.dumps(validation_context["cleaned_summary"], indent=2),
                transformations=json.dumps(transformations, indent=2),
                metrics=json.dumps(metrics, indent=2),
                sector=self.config.user_context.get("secteur_activite", "general"),
                target_variable=self.config.user_context.get("target_variable", "unknown")
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
            
            # Wait for completion
            result = await self._wait_for_run_completion(thread.id, run.id)
            
            # Parse and enhance results
            control_report = self._parse_control_results(result, metrics)
            
            # Store metrics
            self.quality_metrics = metrics
            
            return control_report
            
        except Exception as e:
            logger.error(f"Error in validation: {e}")
            return self._basic_validation(cleaned_df, original_df, metrics)
    
    def _calculate_quality_metrics(
        self,
        cleaned_df: pd.DataFrame,
        original_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Calculate comprehensive quality metrics"""
        metrics = {
            "data_quality": {},
            "transformation_impact": {},
            "statistical_changes": {},
            "integrity_checks": {}
        }
        
        # Data quality metrics
        metrics["data_quality"] = {
            "completeness": {
                "original": 1 - (original_df.isnull().sum().sum() / (original_df.shape[0] * original_df.shape[1])),
                "cleaned": 1 - (cleaned_df.isnull().sum().sum() / (cleaned_df.shape[0] * cleaned_df.shape[1]))
            },
            "duplicates": {
                "original": original_df.duplicated().sum(),
                "cleaned": cleaned_df.duplicated().sum()
            },
            "rows": {
                "original": len(original_df),
                "cleaned": len(cleaned_df),
                "removed": len(original_df) - len(cleaned_df)
            },
            "columns": {
                "original": len(original_df.columns),
                "cleaned": len(cleaned_df.columns),
                "removed": len(original_df.columns) - len([c for c in original_df.columns if c in cleaned_df.columns])
            }
        }
        
        # Transformation impact
        metrics["transformation_impact"] = {
            "rows_affected": len(original_df) - len(cleaned_df),
            "completeness_improvement": 
                metrics["data_quality"]["completeness"]["cleaned"] - 
                metrics["data_quality"]["completeness"]["original"],
            "duplicate_reduction": 
                metrics["data_quality"]["duplicates"]["original"] - 
                metrics["data_quality"]["duplicates"]["cleaned"]
        }
        
        # Statistical changes for numeric columns
        for col in cleaned_df.select_dtypes(include=[np.number]).columns:
            if col in original_df.columns and pd.api.types.is_numeric_dtype(original_df[col]):
                try:
                    metrics["statistical_changes"][col] = {
                        "mean_change": float(cleaned_df[col].mean() - original_df[col].mean()),
                        "std_change": float(cleaned_df[col].std() - original_df[col].std()),
                        "min_change": float(cleaned_df[col].min() - original_df[col].min()),
                        "max_change": float(cleaned_df[col].max() - original_df[col].max())
                    }
                except:
                    pass
        
        # Integrity checks
        metrics["integrity_checks"] = self._perform_integrity_checks(cleaned_df)
        
        # Calculate overall quality score
        metrics["quality_score"] = self._calculate_quality_score(metrics)
        
        return metrics
    
    def _perform_integrity_checks(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform data integrity checks"""
        checks = {
            "no_empty_dataframe": len(df) > 0,
            "no_all_null_columns": not any(df[col].isnull().all() for col in df.columns),
            "no_duplicate_columns": len(df.columns) == len(set(df.columns)),
            "reasonable_missing_ratio": (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) < 0.5,
            "no_constant_columns": not any(df[col].nunique() == 1 for col in df.columns)
        }
        
        # Type consistency checks
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                # Check for inf values
                if np.isinf(df[col]).any():
                    checks[f"{col}_no_inf_values"] = False
        
        return checks
    
    def _calculate_quality_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate overall quality score"""
        score = 100.0
        
        # Completeness (30 points)
        completeness = metrics["data_quality"]["completeness"]["cleaned"]
        score -= (1 - completeness) * 30
        
        # Duplicates (20 points)
        if metrics["data_quality"]["rows"]["cleaned"] > 0:
            duplicate_ratio = metrics["data_quality"]["duplicates"]["cleaned"] / metrics["data_quality"]["rows"]["cleaned"]
            score -= duplicate_ratio * 20
        
        # Data loss (20 points)
        if metrics["data_quality"]["rows"]["original"] > 0:
            row_loss_ratio = metrics["data_quality"]["rows"]["removed"] / metrics["data_quality"]["rows"]["original"]
            score -= min(20, row_loss_ratio * 40)  # Penalize heavy data loss
        
        # Integrity checks (30 points)
        integrity_checks = metrics["integrity_checks"]
        failed_checks = sum(1 for v in integrity_checks.values() if not v)
        score -= (failed_checks / len(integrity_checks)) * 30 if integrity_checks else 0
        
        return max(0, min(100, score))
    
    def _prepare_validation_context(
        self,
        cleaned_df: pd.DataFrame,
        original_df: pd.DataFrame,
        transformations: List[Dict[str, Any]],
        metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Prepare context for validation"""
        context = {
            "original_summary": {
                "shape": original_df.shape,
                "columns": list(original_df.columns),
                "dtypes": {col: str(dtype) for col, dtype in original_df.dtypes.items()},
                "missing_values": original_df.isnull().sum().to_dict(),
                "duplicates": original_df.duplicated().sum()
            },
            "cleaned_summary": {
                "shape": cleaned_df.shape,
                "columns": list(cleaned_df.columns),
                "dtypes": {col: str(dtype) for col, dtype in cleaned_df.dtypes.items()},
                "missing_values": cleaned_df.isnull().sum().to_dict(),
                "duplicates": cleaned_df.duplicated().sum()
            },
            "transformations_count": len(transformations),
            "columns_affected": list(set(t.get("column", "") for t in transformations if "column" in t))
        }
        
        return context
    
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
            
            await asyncio.sleep(2)
        
        raise TimeoutError("Assistant run timed out")
    
    def _parse_control_results(self, result: Dict[str, Any], metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Parse control results from assistant"""
        control_report = {
            "validation_passed": True,
            "quality_score": metrics["quality_score"],
            "issues": [],
            "warnings": [],
            "recommendations": [],
            "metrics": metrics,
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            content = result.get("content", "")
            
            # Try to extract JSON
            import re
            json_pattern = r'\{[\s\S]*\}'
            json_matches = re.findall(json_pattern, content)
            
            if json_matches:
                for match in sorted(json_matches, key=len, reverse=True):
                    try:
                        parsed = json.loads(match)
                        control_report.update(parsed)
                        break
                    except json.JSONDecodeError:
                        continue
            
            # Extract information from text
            lines = content.split('\n')
            for line in lines:
                line_lower = line.lower()
                
                if "fail" in line_lower or "error" in line_lower:
                    control_report["validation_passed"] = False
                    if line.strip() and line.strip() not in control_report["issues"]:
                        control_report["issues"].append(line.strip())
                
                elif "warning" in line_lower:
                    if line.strip() and line.strip() not in control_report["warnings"]:
                        control_report["warnings"].append(line.strip())
                
                elif "recommend" in line_lower or "suggest" in line_lower:
                    if line.strip() and line.strip() not in control_report["recommendations"]:
                        control_report["recommendations"].append(line.strip())
            
            # Add automatic warnings based on metrics
            if metrics["transformation_impact"]["rows_affected"] > len(control_report.get("original_df", [])) * 0.1:
                control_report["warnings"].append(
                    f"More than 10% of rows were removed during cleaning"
                )
            
            if metrics["data_quality"]["completeness"]["cleaned"] < 0.8:
                control_report["warnings"].append(
                    f"Data completeness is below 80% ({metrics['data_quality']['completeness']['cleaned']:.1%})"
                )
            
        except Exception as e:
            logger.error(f"Failed to parse control results: {e}")
        
        return control_report
    
    def _basic_validation(
        self,
        cleaned_df: pd.DataFrame,
        original_df: pd.DataFrame,
        metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Basic validation fallback"""
        logger.info("Using basic validation fallback")
        
        report = {
            "validation_passed": True,
            "quality_score": metrics["quality_score"],
            "issues": [],
            "warnings": [],
            "recommendations": [],
            "metrics": metrics,
            "timestamp": datetime.now().isoformat()
        }
        
        # Check for critical issues
        if len(cleaned_df) == 0:
            report["validation_passed"] = False
            report["issues"].append("Cleaned dataframe is empty")
        
        if len(cleaned_df.columns) == 0:
            report["validation_passed"] = False
            report["issues"].append("No columns remaining after cleaning")
        
        # Check data loss
        row_loss = (len(original_df) - len(cleaned_df)) / len(original_df)
        if row_loss > 0.5:
            report["validation_passed"] = False
            report["issues"].append(f"Lost more than 50% of rows ({row_loss:.1%})")
        elif row_loss > 0.2:
            report["warnings"].append(f"Lost {row_loss:.1%} of rows")
        
        # Check column loss
        original_cols = set(original_df.columns)
        cleaned_cols = set(cleaned_df.columns)
        lost_cols = original_cols - cleaned_cols
        
        if len(lost_cols) > len(original_cols) * 0.3:
            report["warnings"].append(f"Lost {len(lost_cols)} columns: {list(lost_cols)[:5]}")
        
        # Check for remaining issues
        if cleaned_df.duplicated().sum() > 0:
            report["warnings"].append(f"Still contains {cleaned_df.duplicated().sum()} duplicate rows")
        
        missing_ratio = cleaned_df.isnull().sum().sum() / (cleaned_df.shape[0] * cleaned_df.shape[1])
        if missing_ratio > 0.2:
            report["warnings"].append(f"Still contains {missing_ratio:.1%} missing values")
        
        # Add recommendations
        if report["quality_score"] < 70:
            report["recommendations"].append("Consider reviewing transformation parameters")
        
        if len(report["warnings"]) > 3:
            report["recommendations"].append("Multiple quality issues detected - manual review recommended")
        
        return report
    
    async def generate_compliance_report(
        self,
        cleaned_df: pd.DataFrame,
        sector: str
    ) -> Dict[str, Any]:
        """Generate sector-specific compliance report"""
        compliance_report = {
            "sector": sector,
            "compliant": True,
            "checks": {},
            "recommendations": []
        }
        
        if sector == "finance":
            # Financial sector checks
            checks = {
                "has_transaction_id": any("transaction" in col.lower() or "id" in col.lower() 
                                        for col in cleaned_df.columns),
                "has_amount": any("amount" in col.lower() or "value" in col.lower() 
                                 for col in cleaned_df.columns),
                "has_date": any("date" in col.lower() or "time" in col.lower() 
                               for col in cleaned_df.columns),
                "no_negative_amounts": True
            }
            
            # Check for negative amounts
            for col in cleaned_df.select_dtypes(include=[np.number]).columns:
                if "amount" in col.lower():
                    if (cleaned_df[col] < 0).any():
                        checks["no_negative_amounts"] = False
            
            compliance_report["checks"] = checks
            compliance_report["compliant"] = all(checks.values())
            
        elif sector == "sante":
            # Healthcare sector checks
            checks = {
                "has_patient_id": any("patient" in col.lower() or "id" in col.lower() 
                                     for col in cleaned_df.columns),
                "has_date": any("date" in col.lower() for col in cleaned_df.columns),
                "no_pii_in_clear": not any(col.lower() in ["ssn", "social_security", "nom", "name"] 
                                          for col in cleaned_df.columns)
            }
            
            compliance_report["checks"] = checks
            compliance_report["compliant"] = all(checks.values())
            
            if not checks["no_pii_in_clear"]:
                compliance_report["recommendations"].append(
                    "Consider anonymizing or encrypting PII columns"
                )
        
        return compliance_report
    
    def generate_summary_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate summary statistics for the cleaned data"""
        summary = {
            "numeric_columns": {},
            "categorical_columns": {},
            "overall": {
                "total_rows": len(df),
                "total_columns": len(df.columns),
                "memory_usage_mb": df.memory_usage(deep=True).sum() / (1024 * 1024),
                "completeness": 1 - (df.isnull().sum().sum() / (df.shape[0] * df.shape[1]))
            }
        }
        
        # Numeric columns
        for col in df.select_dtypes(include=[np.number]).columns:
            summary["numeric_columns"][col] = {
                "mean": float(df[col].mean()),
                "std": float(df[col].std()),
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "q25": float(df[col].quantile(0.25)),
                "q50": float(df[col].quantile(0.50)),
                "q75": float(df[col].quantile(0.75)),
                "missing": int(df[col].isnull().sum()),
                "zeros": int((df[col] == 0).sum())
            }
        
        # Categorical columns
        for col in df.select_dtypes(include=['object']).columns:
            value_counts = df[col].value_counts()
            summary["categorical_columns"][col] = {
                "unique": int(df[col].nunique()),
                "most_frequent": value_counts.index[0] if len(value_counts) > 0 else None,
                "frequency": int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                "missing": int(df[col].isnull().sum())
            }
        
        return summary
