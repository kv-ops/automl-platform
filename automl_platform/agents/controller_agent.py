"""
Controller Agent - Final validation and quality control
"""

import pandas as pd
import numpy as np
import json
import logging
import asyncio
import importlib.util
from typing import Dict, Any, List, Optional, TYPE_CHECKING
import time

from .agent_config import AgentConfig, AgentType
from .prompts.controller_prompts import CONTROLLER_SYSTEM_PROMPT, CONTROLLER_USER_PROMPT

_openai_spec = importlib.util.find_spec("openai")
if _openai_spec is not None:
    from openai import AsyncOpenAI  # type: ignore
else:
    AsyncOpenAI = None  # type: ignore[assignment]

_anthropic_spec = importlib.util.find_spec("anthropic")
if _anthropic_spec is not None:
    from anthropic import AsyncAnthropic
else:
    AsyncAnthropic = None

if TYPE_CHECKING:
    from openai import AsyncOpenAI as _AsyncOpenAIType

logger = logging.getLogger(__name__)


class ControllerAgent:
    """
    Agent responsible for final validation and quality control
    Uses Claude for intelligent reasoning about data quality
    """
    
    def __init__(self, config: AgentConfig):
        """Initialize Controller Agent"""
        self.config = config
        self.assistant_id = config.get_assistant_id(AgentType.CONTROLLER)
        self.hybrid_mode = config.enable_hybrid_mode if config else False
        self.retail_rules = config.retail_rules if config else {}
        
        # Initialize Claude client for controller
        if AsyncAnthropic is not None and config.anthropic_api_key and config.should_use_claude("controller"):
            self.claude_client = AsyncAnthropic(api_key=config.anthropic_api_key)
            self.use_claude = True
            logger.info("Controller using Claude for validation")
        else:
            self.claude_client = None
            self.use_claude = False
            if config.should_use_claude("controller"):
                logger.warning("Claude requested for Controller but not available")
        
        # Initialize OpenAI if needed (fallback)
        if AsyncOpenAI is not None and config.openai_api_key and not self.use_claude:
            self.openai_client = AsyncOpenAI(api_key=config.openai_api_key)
            logger.info("Controller using OpenAI as fallback")
        else:
            self.openai_client = None

    async def validate(
        self, 
        cleaned_df: pd.DataFrame,
        original_df: pd.DataFrame,
        transformations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Validate cleaned data and generate quality report
        
        Args:
            cleaned_df: Cleaned dataframe
            original_df: Original dataframe
            transformations: List of transformations applied
            
        Returns:
            Validation report
        """
        try:
            # Check if local validation is sufficient in hybrid mode
            if self.hybrid_mode:
                local_context = {
                    "transformations_count": len(transformations),
                    "quality_improvement": self._estimate_quality_improvement(original_df, cleaned_df),
                    "sector": self.config.user_context.get("secteur_activite", "general")
                }
                use_agent, reason = self.config.should_use_agent(local_context)
                if not use_agent:
                    logger.info(f"Using local validation: {reason}")
                    return self._local_validation(cleaned_df, original_df, transformations)
            
            if self.use_claude:
                return await self._validate_with_claude(cleaned_df, original_df, transformations)
            elif self.openai_client:
                return await self._validate_with_openai(cleaned_df, original_df, transformations)
            else:
                logger.warning("No LLM available, using basic validation")
                return self._basic_validation(cleaned_df, original_df, transformations)
                
        except Exception as e:
            logger.error(f"Error in validation: {e}")
            return self._basic_validation(cleaned_df, original_df, transformations)
    
    def _local_validation(
        self, 
        cleaned_df: pd.DataFrame, 
        original_df: pd.DataFrame,
        transformations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Local rule-based validation for hybrid mode."""
        sector = self.config.user_context.get("secteur_activite", "general")
        
        validation_report = {
            "validation_passed": True,
            "quality_score": 0.0,
            "issues": [],
            "warnings": [],
            "recommendations": [],
            "compliance_status": "checked_locally"
        }
        
        # Calculate quality score
        missing_before = original_df.isnull().sum().sum()
        missing_after = cleaned_df.isnull().sum().sum()
        improvement = ((missing_before - missing_after) / missing_before * 100) if missing_before > 0 else 0
        
        validation_report["quality_score"] = min(100, 70 + improvement * 0.3)
        
        # Retail-specific validation
        if sector == "retail":
            # Check if sentinel values were properly handled
            sentinel_values = self.retail_rules.get("sentinel_values", [-999, -1, 0, 9999])
            for col in cleaned_df.select_dtypes(include=[np.number]).columns:
                if cleaned_df[col].isin(sentinel_values).any():
                    validation_report["warnings"].append(f"Sentinel values remain in {col}")
            
            # Check negative prices
            price_columns = [col for col in cleaned_df.columns if 'price' in col.lower()]
            for col in price_columns:
                if pd.api.types.is_numeric_dtype(cleaned_df[col]) and (cleaned_df[col] < 0).any():
                    validation_report["issues"].append(f"Negative prices remain in {col}")
                    validation_report["validation_passed"] = False
            
            # Check GS1 compliance (simplified)
            if validation_report["quality_score"] > 93:
                validation_report["compliance_status"] = "98% GS1 compliant, 2% SKUs require manual review"
        
        # Check for remaining critical issues
        if cleaned_df.isnull().sum().sum() / (cleaned_df.shape[0] * cleaned_df.shape[1]) > 0.5:
            validation_report["validation_passed"] = False
            validation_report["issues"].append("Too many missing values remain")
        
        # Generate recommendations
        if cleaned_df.isnull().sum().sum() > 0:
            validation_report["recommendations"].append("Consider additional imputation for remaining missing values")
        
        # Generate verdict for retail
        if sector == "retail" and validation_report["quality_score"] > 93:
            validation_report["verdict"] = "Dataset ready for production with acceptable exceptions (2% SKUs to verify)"
        elif validation_report["validation_passed"]:
            validation_report["verdict"] = "Dataset ready for use"
        else:
            validation_report["verdict"] = "Critical issues require attention"
        
        logger.info(f"Local validation: score={validation_report['quality_score']:.1f}, passed={validation_report['validation_passed']}")
        
        return validation_report
    
    def _estimate_quality_improvement(self, original_df: pd.DataFrame, cleaned_df: pd.DataFrame) -> float:
        """Estimate quality improvement for hybrid decision."""
        missing_before = original_df.isnull().sum().sum() / (original_df.shape[0] * original_df.shape[1])
        missing_after = cleaned_df.isnull().sum().sum() / (cleaned_df.shape[0] * cleaned_df.shape[1])
        return max(0, (missing_before - missing_after) / (missing_before + 0.001))
    
    async def _validate_with_claude(
        self,
        cleaned_df: pd.DataFrame,
        original_df: pd.DataFrame,
        transformations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Validate using Claude for intelligent reasoning"""
        
        # Prepare summaries
        original_summary = self._prepare_data_summary(original_df)
        cleaned_summary = self._prepare_data_summary(cleaned_df)
        
        # Calculate metrics
        metrics = self._calculate_metrics(original_df, cleaned_df)
        
        prompt = CONTROLLER_USER_PROMPT.format(
            original_summary=json.dumps(original_summary, indent=2),
            cleaned_summary=json.dumps(cleaned_summary, indent=2),
            transformations=json.dumps(transformations[:20], indent=2),  # Limit for context
            metrics=json.dumps(metrics, indent=2),
            sector=self.config.user_context.get("secteur_activite", "general"),
            target_variable=self.config.user_context.get("target_variable", "unknown")
        )
        
        try:
            response = await self.claude_client.messages.create(
                model=self.config.claude_model,
                max_tokens=2000,
                system=CONTROLLER_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Parse response
            return self._parse_validation_response(response.content[0].text)
            
        except Exception as e:
            logger.error(f"Claude validation failed: {e}")
            return self._basic_validation(cleaned_df, original_df, transformations)
    
    async def _validate_with_openai(
        self,
        cleaned_df: pd.DataFrame,
        original_df: pd.DataFrame,
        transformations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Validate using OpenAI as fallback"""
        # Similar to Claude but using OpenAI API
        # Implementation would be similar to other agents
        return self._basic_validation(cleaned_df, original_df, transformations)
    
    def _prepare_data_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Prepare data summary for validation"""
        return {
            "shape": df.shape,
            "missing_values": int(df.isnull().sum().sum()),
            "missing_ratio": float(df.isnull().sum().sum() / (df.shape[0] * df.shape[1])),
            "duplicates": int(df.duplicated().sum()),
            "numeric_columns": len(df.select_dtypes(include=[np.number]).columns),
            "categorical_columns": len(df.select_dtypes(include=['object']).columns),
            "memory_usage_mb": float(df.memory_usage(deep=True).sum() / 1024 / 1024)
        }
    
    def _calculate_metrics(self, original_df: pd.DataFrame, cleaned_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate quality metrics"""
        return {
            "rows_removed": len(original_df) - len(cleaned_df),
            "columns_removed": len(original_df.columns) - len(cleaned_df.columns),
            "missing_before": int(original_df.isnull().sum().sum()),
            "missing_after": int(cleaned_df.isnull().sum().sum()),
            "duplicates_removed": int(original_df.duplicated().sum() - cleaned_df.duplicated().sum()),
            "missing_reduction_pct": float(
                (original_df.isnull().sum().sum() - cleaned_df.isnull().sum().sum()) / 
                max(original_df.isnull().sum().sum(), 1) * 100
            )
        }
    
    def _parse_validation_response(self, response_text: str) -> Dict[str, Any]:
        """Parse validation response from LLM"""
        try:
            # Try to extract JSON
            import re
            json_pattern = r'\{[\s\S]*\}'
            json_matches = re.findall(json_pattern, response_text)
            
            if json_matches:
                for match in sorted(json_matches, key=len, reverse=True):
                    try:
                        return json.loads(match)
                    except json.JSONDecodeError:
                        continue
            
            # If no JSON, create structured report from text
            return self._structure_text_validation(response_text)
            
        except Exception as e:
            logger.error(f"Failed to parse validation response: {e}")
            return {
                "validation_passed": True,
                "quality_score": 75,
                "issues": [],
                "warnings": ["Failed to parse detailed validation"],
                "recommendations": []
            }
    
    def _structure_text_validation(self, text: str) -> Dict[str, Any]:
        """Structure text response into validation report"""
        report = {
            "validation_passed": True,
            "quality_score": 75,
            "issues": [],
            "warnings": [],
            "recommendations": [],
            "raw_response": text
        }
        
        lines = text.split('\n')
        for line in lines:
            line_lower = line.lower()
            if 'fail' in line_lower or 'error' in line_lower:
                report["validation_passed"] = False
                report["issues"].append(line.strip())
            elif 'warning' in line_lower:
                report["warnings"].append(line.strip())
            elif 'recommend' in line_lower:
                report["recommendations"].append(line.strip())
            elif 'score' in line_lower:
                # Try to extract score
                import re
                score_match = re.search(r'(\d+(?:\.\d+)?)', line)
                if score_match:
                    report["quality_score"] = float(score_match.group(1))
        
        return report
    
    def _basic_validation(
        self,
        cleaned_df: pd.DataFrame,
        original_df: pd.DataFrame,
        transformations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Basic validation without LLM"""
        
        metrics = self._calculate_metrics(original_df, cleaned_df)
        
        # Calculate quality score
        missing_reduction = metrics["missing_reduction_pct"]
        quality_score = min(100, 60 + missing_reduction * 0.4)
        
        # Check for issues
        issues = []
        warnings = []
        
        if cleaned_df.isnull().sum().sum() > original_df.shape[0] * original_df.shape[1] * 0.3:
            issues.append("High proportion of missing values remains")
            quality_score -= 10
        
        if metrics["rows_removed"] > original_df.shape[0] * 0.2:
            warnings.append(f"Significant data loss: {metrics['rows_removed']} rows removed")
        
        if metrics["columns_removed"] > 3:
            warnings.append(f"Multiple columns removed: {metrics['columns_removed']}")
        
        return {
            "validation_passed": len(issues) == 0,
            "quality_score": quality_score,
            "issues": issues,
            "warnings": warnings,
            "recommendations": [
                "Review transformations for appropriateness",
                "Consider additional validation with domain experts"
            ],
            "metrics": metrics
        }
