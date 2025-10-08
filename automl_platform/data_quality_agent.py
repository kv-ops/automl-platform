"""
Data Quality Agent - Enhanced with Agent-First Integration
Conversational data cleaning inspired by Akkio's GPT-4 approach
Integrated with Universal ML Agent for intelligent context detection
Enhanced with retail-specific recommendations
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging
import json
import asyncio
import os
from datetime import datetime
from dataclasses import dataclass, field, asdict
from enum import Enum
import re

from .risk import RiskLevel


logger = logging.getLogger(__name__)


@dataclass
class DataQualityAssessment:
    """DataRobot-style quality assessment with visual alerts."""

    quality_score: float  # 0-100
    alerts: List[Dict[str, Any]] = field(default_factory=list)  # Critical issues requiring attention
    warnings: List[Dict[str, Any]] = field(default_factory=list)  # Non-critical issues
    recommendations: List[Dict[str, Any]] = field(default_factory=list)  # Suggested improvements
    retail_recommendations: List[Dict[str, Any]] = field(default_factory=list)  # NEW: Retail-specific recommendations
    statistics: Dict[str, Any] = field(default_factory=dict)  # Statistical summary
    drift_risk: str = "low"  # "low", "medium", "high"
    target_leakage_risk: str = "low"  # Align with tests expecting string levels
    visualization_data: Dict[str, Any] = field(default_factory=dict)  # Data for visual quality assessment
    ml_context: Optional[Dict[str, Any]] = None  # Agent-First ML context
    retail_metrics: Optional[Dict[str, Any]] = None  # Retail-specific metrics

    def __post_init__(self) -> None:
        self.drift_risk = RiskLevel.normalize(self.drift_risk, field_name="drift_risk")
        self.target_leakage_risk = RiskLevel.normalize(
            self.target_leakage_risk, field_name="target_leakage_risk"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable representation of the assessment."""

        data = asdict(self)
        data["drift_risk"] = self.drift_risk.value
        data["target_leakage_risk"] = self.target_leakage_risk.value
        return data


class AkkioStyleCleaningAgent:
    """
    Conversational data cleaning agent inspired by Akkio's GPT-4 chatbot.
    Enhanced with Agent-First context awareness.
    """
    
    def __init__(self, llm_provider, enable_agent_first: bool = False, config=None):
        self.llm = llm_provider
        self.conversation_history = []
        self.cleaning_actions = []
        self.undo_stack = []
        self.enable_agent_first = enable_agent_first
        self.ml_context = None
        self.context_detector = None
        self.config = config  # AgentConfig for thresholds
        
        if enable_agent_first:
            self._init_agent_first()
    
    def _init_agent_first(self):
        """Initialize Agent-First components."""
        try:
            from automl_platform.agents import IntelligentContextDetector
            self.context_detector = IntelligentContextDetector()
            logger.info("Agent-First context detector initialized for cleaning agent")
        except ImportError:
            logger.warning("Agent-First components not available for cleaning agent")
            self.enable_agent_first = False
    
    async def chat_clean(self, user_message: str, df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
        """
        Akkio-style conversational cleaning with Agent-First context awareness.
        User can say things like:
        - "Remove outliers from the price column"
        - "Fill missing values with median"
        - "Combine first_name and last_name columns"
        - "Format dates to YYYY-MM-DD"
        - "Prepare this for fraud detection" (Agent-First)
        - "Replace sentinel values and fix negative prices" (Retail)
        """
        
        # Add to conversation history
        self.conversation_history.append({"role": "user", "message": user_message})
        
        # Detect ML context if Agent-First is enabled
        if self.enable_agent_first and self.context_detector:
            if "prepare" in user_message.lower() or "detection" in user_message.lower() or "prediction" in user_message.lower():
                self.ml_context = await self._detect_ml_context(df, user_message)
        
        # Analyze user intent with context awareness
        intent = await self._analyze_cleaning_intent(user_message, df)
        
        # Generate cleaning code
        cleaning_code = await self._generate_cleaning_code(intent, df)
        
        # Preview changes
        preview = self._preview_changes(df, cleaning_code)
        
        # Generate response with ML context if available
        context_info = ""
        if self.ml_context:
            context_info = f"""
**ML Context Detected:**
- Problem Type: {self.ml_context.get('problem_type', 'Unknown')}
- Confidence: {self.ml_context.get('confidence', 0):.1%}
- Business Sector: {self.ml_context.get('business_sector', 'General')}

"""
        
        response = f"""
{context_info}I understand you want to {intent['action']}. Here's what I'll do:

**Action:** {intent['description']}
**Affected:** {preview['affected_rows']} rows, {preview['affected_columns']} columns

```python
{cleaning_code}
```

**Preview of changes:**
{preview['sample_changes']}

Shall I apply these changes?
"""
        
        # Store for potential execution
        self.cleaning_actions.append({
            "timestamp": datetime.now(),
            "intent": intent,
            "code": cleaning_code,
            "preview": preview,
            "ml_context": self.ml_context
        })
        
        return df, response
    
    async def _detect_ml_context(self, df: pd.DataFrame, user_message: str) -> Dict[str, Any]:
        """Detect ML context using Agent-First approach."""
        if not self.context_detector:
            return {}
        
        try:
            # Extract target column from message if mentioned
            target_col = None
            for col in df.columns:
                if col.lower() in user_message.lower():
                    target_col = col
                    break
            
            # Detect context
            context = await self.context_detector.detect_ml_context(df, target_col)
            
            return {
                "problem_type": context.problem_type,
                "confidence": context.confidence,
                "business_sector": context.business_sector,
                "temporal_aspect": context.temporal_aspect,
                "imbalance_detected": context.imbalance_detected
            }
        except Exception as e:
            logger.error(f"ML context detection failed: {e}")
            return {}
    
    async def _analyze_cleaning_intent(self, message: str, df: pd.DataFrame) -> Dict:
        """Analyze user's cleaning intent using LLM with ML context awareness."""
        
        context_prompt = ""
        if self.ml_context:
            context_prompt = f"""
ML Context:
- Problem Type: {self.ml_context.get('problem_type')}
- Business Sector: {self.ml_context.get('business_sector')}
- Imbalance: {self.ml_context.get('imbalance_detected')}

Consider this context when analyzing the cleaning request.
"""
        
        # Check for retail-specific keywords
        retail_keywords = ["sentinel", "negative price", "stock", "category median"]
        is_retail_request = any(keyword in message.lower() for keyword in retail_keywords)
        
        prompt = f"""
Analyze this data cleaning request:
User message: "{message}"

Dataset info:
- Columns: {list(df.columns)}
- Shape: {df.shape}
- Types: {df.dtypes.to_dict()}
{'- Retail context detected' if is_retail_request else ''}

{context_prompt}

Extract:
1. Action type (remove, fill, transform, combine, format, filter, prepare_for_ml, handle_sentinels, fix_prices)
2. Target columns
3. Specific parameters
4. Safety concerns
5. ML-specific preparations if applicable
6. Retail-specific handling if needed

Return as JSON.
"""
        
        response = await self.llm.generate(prompt, temperature=0.1)
        
        try:
            intent = json.loads(response.content)
            
            # Add ML-specific intents if context detected
            if self.ml_context and intent.get("action") == "prepare_for_ml":
                intent["ml_preparations"] = self._get_ml_preparations(self.ml_context)
            
            # Add retail-specific intents
            if is_retail_request:
                intent["retail_handling"] = self._get_retail_preparations()
                
        except:
            intent = {
                "action": "unknown",
                "description": message,
                "columns": [],
                "parameters": {}
            }
        
        return intent
    
    def _get_ml_preparations(self, ml_context: Dict[str, Any]) -> List[str]:
        """Get ML-specific data preparations based on context."""
        preparations = []
        
        problem_type = ml_context.get('problem_type', '')
        
        if problem_type == 'fraud_detection':
            preparations = [
                "Create velocity features",
                "Handle class imbalance with SMOTE",
                "Engineer time-based features",
                "Normalize monetary amounts",
                "Replace sentinel values with NaN"
            ]
        elif problem_type == 'churn_prediction':
            preparations = [
                "Create RFM features",
                "Handle missing customer data",
                "Engineer engagement metrics",
                "Balance classes if needed"
            ]
        elif problem_type == 'sales_forecasting':
            preparations = [
                "Create lag features",
                "Add seasonal indicators",
                "Handle missing time periods",
                "Engineer trend features",
                "Fix negative prices using category median"
            ]
        
        return preparations
    
    def _get_retail_preparations(self) -> List[str]:
        """Get retail-specific data preparations."""
        return [
            "Replace sentinel values (-999, -1, 9999) with NaN",
            "Fix negative prices using median by category",
            "Preserve zero values in stock/quantity columns",
            "Impute missing values using category-based statistics",
            "Handle high cardinality SKU columns"
        ]
    
    async def _generate_cleaning_code(self, intent: Dict, df: pd.DataFrame) -> str:
        """Generate Python code for the cleaning action with ML awareness."""
        
        ml_context_prompt = ""
        if self.ml_context and intent.get("ml_preparations"):
            ml_context_prompt = f"""
Include these ML-specific preparations:
{json.dumps(intent['ml_preparations'])}
"""
        
        retail_context_prompt = ""
        if intent.get("retail_handling"):
            retail_context_prompt = f"""
Include these retail-specific preparations:
{json.dumps(intent['retail_handling'])}
"""
        
        prompt = f"""
Generate pandas code for this data cleaning task:
Intent: {json.dumps(intent)}

{ml_context_prompt}
{retail_context_prompt}

Requirements:
- Use df as the dataframe variable
- Handle edge cases
- Preserve data types where possible
- Add error handling
- Make it efficient
- Include ML-specific preprocessing if applicable
- Include retail-specific handling (sentinels, negative prices) if applicable

Return only executable Python code.
"""
        
        response = await self.llm.generate(prompt, temperature=0.1)
        
        # Extract code
        code = response.content
        if "```python" in code:
            code = code.split("```python")[1].split("```")[0]
        
        return code.strip()
    
    def _preview_changes(self, df: pd.DataFrame, code: str) -> Dict:
        """Preview changes without applying them."""
        
        # Create a copy for preview
        df_preview = df.copy()
        
        try:
            # Execute code on preview
            local_vars = {"df": df_preview, "pd": pd, "np": np}
            exec(code, {}, local_vars)
            df_after = local_vars.get("df", df_preview)
            
            # Calculate changes
            changes = {
                "affected_rows": (df != df_after).any(axis=1).sum() if df.shape == df_after.shape else abs(len(df) - len(df_after)),
                "affected_columns": list(set(df.columns) ^ set(df_after.columns)) or 
                                   [col for col in df.columns if not df[col].equals(df_after[col])],
                "sample_changes": df_after.head(3).to_dict('records'),
                "shape_before": df.shape,
                "shape_after": df_after.shape
            }
            
        except Exception as e:
            changes = {
                "error": str(e),
                "affected_rows": 0,
                "affected_columns": [],
                "sample_changes": {}
            }
        
        return changes
    
    def apply_cleaning(self, df: pd.DataFrame, action_index: int = -1) -> pd.DataFrame:
        """Apply a cleaning action from history."""
        
        if not self.cleaning_actions:
            return df
        
        action = self.cleaning_actions[action_index]
        
        # Save state for undo
        self.undo_stack.append(df.copy())
        
        # Apply cleaning
        try:
            local_vars = {"df": df.copy(), "pd": pd, "np": np}
            exec(action["code"], {}, local_vars)
            df_cleaned = local_vars.get("df", df)
            
            logger.info(f"Applied cleaning: {action['intent']['description']}")
            
            # Log ML context if present
            if action.get("ml_context"):
                logger.info(f"ML Context: {action['ml_context']['problem_type']}")
            
            return df_cleaned
            
        except Exception as e:
            logger.error(f"Failed to apply cleaning: {e}")
            return df
    
    def undo_last_action(self, df: pd.DataFrame) -> pd.DataFrame:
        """Undo the last cleaning action."""
        
        if self.undo_stack:
            return self.undo_stack.pop()
        return df
    
    def get_cleaning_suggestions(self, df: pd.DataFrame) -> List[str]:
        """Get cleaning suggestions based on ML context and retail indicators."""
        suggestions = []
        
        # ML context suggestions
        if self.ml_context:
            problem_type = self.ml_context.get('problem_type', '')
            
            if problem_type:
                suggestions.append(f"Detected {problem_type} problem. Consider:")
                suggestions.extend(self._get_ml_preparations(self.ml_context))
        
        # Retail-specific suggestions
        sentinel_values = self.config.get_retail_rules('sentinel_values') if self.config else [-999, -1, 9999]
        for col in df.select_dtypes(include=[np.number]).columns:
            # Check if it's a stock column
            stock_keywords = ['stock', 'quantity', 'qty', 'inventory', 'count', 'units']
            is_stock = any(keyword in col.lower() for keyword in stock_keywords)
            
            # Don't treat 0 as sentinel for stock columns
            effective_sentinels = sentinel_values.copy()
            if is_stock and 0 in effective_sentinels:
                effective_sentinels.remove(0)
            
            if df[col].isin(effective_sentinels).any():
                suggestions.append(f"Replace sentinel values in '{col}' with NaN and impute by category median")
                break
        
        # Check for negative prices
        price_columns = [col for col in df.columns if 'price' in col.lower() or 'prix' in col.lower()]
        for col in price_columns:
            if pd.api.types.is_numeric_dtype(df[col]) and (df[col] < 0).any():
                suggestions.append(f"Fix negative prices in '{col}' using median by category")
                break
        
        # General suggestions
        missing_cols = df.columns[df.isnull().any()].tolist()
        if missing_cols:
            suggestions.append(f"Handle missing values in: {', '.join(missing_cols[:5])}")
        
        # Check for duplicates
        if df.duplicated().any():
            suggestions.append(f"Remove {df.duplicated().sum()} duplicate rows")
        
        return suggestions


class DataRobotStyleQualityMonitor:
    """
    DataRobot-style Data Quality Assessment with visual indicators.
    Enhanced with Agent-First ML context awareness and retail-specific checks.
    """
    
    def __init__(self, enable_agent_first: bool = False, config=None):
        self.config = config  # AgentConfig for thresholds
        
        # Use thresholds from config if available, otherwise defaults
        if config:
            self.quality_thresholds = {
                "missing_critical": config.get_quality_threshold("missing_critical_threshold") or 0.5,
                "missing_warning": config.get_quality_threshold("missing_warning_threshold") or 0.35,
                "outlier_critical": config.get_quality_threshold("outlier_critical_threshold") or 0.15,
                "outlier_warning": config.get_quality_threshold("outlier_warning_threshold") or 0.05,
                "cardinality_high": config.get_quality_threshold("high_cardinality_threshold") or 0.9,
                "correlation_high": config.get_quality_threshold("correlation_high_threshold") or 0.95,
                "imbalance_severe": config.get_quality_threshold("imbalance_severe_threshold") or 20
            }
            self.sentinel_values = config.get_retail_rules("sentinel_values") or [-999, -1, 9999]
        else:
            self.quality_thresholds = {
                "missing_critical": 0.5,
                "missing_warning": 0.35,
                "outlier_critical": 0.15,
                "outlier_warning": 0.05,
                "cardinality_high": 0.9,
                "correlation_high": 0.95,
                "imbalance_severe": 20
            }
            self.sentinel_values = [-999, -1, 9999]
        
        self.enable_agent_first = enable_agent_first
        self.context_detector = None
        
        if enable_agent_first:
            self._init_agent_first()
    
    def _init_agent_first(self):
        """Initialize Agent-First components."""
        try:
            from automl_platform.agents import IntelligentContextDetector
            self.context_detector = IntelligentContextDetector()
            logger.info("Agent-First context detector initialized for quality monitor")
        except ImportError:
            logger.warning("Agent-First components not available for quality monitor")
            self.enable_agent_first = False
    
    def _is_stock_column(self, col_name: str) -> bool:
        """Check if column is likely a stock/quantity column."""
        stock_keywords = ['stock', 'quantity', 'qty', 'inventory', 'count', 'units']
        col_lower = col_name.lower()
        return any(keyword in col_lower for keyword in stock_keywords)
    
    def _get_effective_sentinel_values(self, col_name: str) -> list:
        """Get sentinel values for a column, excluding 0 for stock columns."""
        sentinels = self.sentinel_values.copy()
        
        # Exclude 0 from sentinels for stock/quantity columns
        if self._is_stock_column(col_name):
            if 0 in sentinels:
                sentinels.remove(0)
        
        return sentinels
    
    def _calculate_gs1_compliance(self, df: pd.DataFrame) -> float:
        """
        Calculate GS1 compliance for retail data.
        Based on percentage of SKU conformity rather than sentinel columns.
        """
        # Look for SKU/product code columns
        sku_columns = [col for col in df.columns if any(
            keyword in col.lower() for keyword in ['sku', 'upc', 'ean', 'gtin', 'barcode', 'product_code']
        )]
        
        if not sku_columns:
            # No SKU columns found, assume compliance
            return 100.0
        
        total_skus = 0
        compliant_skus = 0
        
        for col in sku_columns:
            if df[col].dtype == 'object':
                column_values = df[col].dropna()
                total_skus += len(column_values)
                
                # Check GS1 format compliance (simplified)
                # GS1 formats: EAN-13 (13 digits), UPC-A (12 digits), EAN-8 (8 digits), GTIN-14 (14 digits)
                for value in column_values:
                    value_str = str(value).strip()
                    # Check if it's a valid GS1 format (digits only, correct length)
                    if value_str.isdigit() and len(value_str) in [8, 12, 13, 14]:
                        compliant_skus += 1
        
        if total_skus == 0:
            return 100.0
        
        return (compliant_skus / total_skus) * 100
    
    def _calculate_local_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate local statistics for hybrid decision making."""
        stats = {
            'missing_ratio': df.isnull().sum().sum() / (df.shape[0] * df.shape[1]),
            'quality_score': 100.0,
            'has_sentinel_values': False,
            'has_negative_prices': False,
            'sentinel_columns': [],
            'negative_price_columns': []
        }
        
        # Check for sentinel values
        for col in df.select_dtypes(include=[np.number]).columns:
            effective_sentinels = self._get_effective_sentinel_values(col)
            if df[col].isin(effective_sentinels).any():
                stats['has_sentinel_values'] = True
                stats['sentinel_columns'].append(col)
        
        # Check for negative prices
        price_columns = [col for col in df.columns if 'price' in col.lower() or 'prix' in col.lower()]
        for col in price_columns:
            if pd.api.types.is_numeric_dtype(df[col]) and (df[col] < 0).any():
                stats['has_negative_prices'] = True
                stats['negative_price_columns'].append(col)
        
        # Calculate quality score
        missing_penalty = min(30, stats['missing_ratio'] * 100)
        stats['quality_score'] -= missing_penalty
        
        return stats
    
    async def assess_quality_with_context(self, df: pd.DataFrame, target_column: Optional[str] = None) -> DataQualityAssessment:
        """
        Comprehensive quality assessment with ML context detection.
        Agent-First enhanced version.
        """
        # Get standard assessment
        assessment = self.assess_quality(df, target_column)
        
        # Add ML context if Agent-First enabled
        if self.enable_agent_first and self.context_detector:
            try:
                context = await self.context_detector.detect_ml_context(df, target_column)
                assessment.ml_context = {
                    "problem_type": context.problem_type,
                    "confidence": context.confidence,
                    "business_sector": context.business_sector,
                    "temporal_aspect": context.temporal_aspect,
                    "imbalance_detected": context.imbalance_detected
                }
                
                # Add context-specific recommendations
                context_recommendations = self._get_context_recommendations(assessment.ml_context)
                assessment.recommendations.extend(context_recommendations)
                
            except Exception as e:
                logger.error(f"ML context detection failed: {e}")
        
        return assessment
    
    def assess_quality(self, df: pd.DataFrame, target_column: Optional[str] = None) -> DataQualityAssessment:
        """
        Comprehensive quality assessment following DataRobot's approach.
        Enhanced with retail-specific checks and improved GS1 compliance calculation.
        
        Returns visual-ready quality metrics with alerts and recommendations.
        """
        
        # Calculate local stats for potential hybrid decision and retail checks
        local_stats = self._calculate_local_stats(df)
        
        alerts = []
        warnings = []
        recommendations = []
        retail_recommendations = []  # NEW: Separate retail recommendations
        
        # Calculate quality score (starts at 100)
        quality_score = 100.0
        
        # 1. Missing values assessment with configurable threshold
        missing_report = self._assess_missing_values(df)
        quality_score -= missing_report["penalty"]
        alerts.extend(missing_report["alerts"])
        warnings.extend(missing_report["warnings"])
        
        # 2. Outliers assessment  
        outlier_report = self._assess_outliers(df)
        quality_score -= outlier_report["penalty"]
        warnings.extend(outlier_report["warnings"])
        
        # 3. Data types and cardinality
        dtype_report = self._assess_data_types(df)
        quality_score -= dtype_report["penalty"]
        warnings.extend(dtype_report["warnings"])
        
        # 4. Duplicates
        duplicate_report = self._assess_duplicates(df)
        quality_score -= duplicate_report["penalty"]
        if duplicate_report["count"] > 0:
            warnings.append({
                "type": "duplicates",
                "message": f"Found {duplicate_report['count']} duplicate rows",
                "severity": "medium",
                "action": "Remove duplicates"
            })
        
        # 5. Sentinel values assessment (retail-specific)
        sentinel_report = self._assess_sentinel_values(df)
        quality_score -= sentinel_report["penalty"]
        warnings.extend(sentinel_report["warnings"])
        
        # 6. Negative prices assessment (retail-specific)
        price_report = self._assess_negative_prices(df)
        quality_score -= price_report["penalty"]
        alerts.extend(price_report["alerts"])
        
        # 7. Target-specific assessments
        if target_column and target_column in df.columns:
            target_report = self._assess_target(df, target_column)
            quality_score -= target_report["penalty"]
            alerts.extend(target_report["alerts"])

            # Check for leakage
            leakage_risk = self._detect_target_leakage(df, target_column)
        else:
            leakage_risk = RiskLevel.NONE
        
        # 8. Statistical anomalies
        stat_report = self._assess_statistical_anomalies(df)
        warnings.extend(stat_report["warnings"])
        
        # 9. Generate recommendations based on issues
        recommendations = self._generate_recommendations(
            alerts + warnings,
            missing_report,
            outlier_report,
            dtype_report,
            local_stats
        )
        
        # 10. Generate retail-specific recommendations
        retail_recommendations = self._generate_retail_recommendations(
            local_stats,
            sentinel_report,
            price_report
        )
        
        # 11. Calculate drift risk
        drift_risk = self._calculate_drift_risk(df)
        
        # Ensure score is 0-100
        quality_score = max(0, min(100, quality_score))
        
        # Prepare visualization data
        viz_data = {
            "missing_heatmap": missing_report["column_missing_pct"],
            "outlier_columns": outlier_report["outlier_columns"],
            "dtype_distribution": dtype_report["type_counts"],
            "quality_breakdown": {
                "missing": missing_report["penalty"],
                "outliers": outlier_report["penalty"],
                "types": dtype_report["penalty"],
                "duplicates": duplicate_report["penalty"],
                "sentinels": sentinel_report["penalty"],
                "negative_prices": price_report["penalty"]
            }
        }
        
        # Calculate GS1 compliance for retail
        gs1_compliance = self._calculate_gs1_compliance(df) if self.config and self.config.user_context.get("secteur_activite") == "retail" else 100.0
        
        # Retail metrics with improved GS1 calculation
        retail_metrics = {
            "sentinel_columns": local_stats["sentinel_columns"],
            "negative_price_columns": local_stats["negative_price_columns"],
            "has_category_column": "category" in df.columns,
            "gs1_compliance": gs1_compliance,  # Now based on SKU conformity
            "sku_columns_found": [col for col in df.columns if any(
                keyword in col.lower() for keyword in ['sku', 'upc', 'ean', 'gtin', 'barcode', 'product_code']
            )]
        }
        
        return DataQualityAssessment(
            quality_score=quality_score,
            alerts=alerts,
            warnings=warnings,
            recommendations=recommendations,
            retail_recommendations=retail_recommendations,  # NEW: Include retail recommendations
            statistics={
                "rows": len(df),
                "columns": len(df.columns),
                "missing_cells": df.isnull().sum().sum(),
                "duplicate_rows": duplicate_report["count"],
                "numeric_columns": len(df.select_dtypes(include=[np.number]).columns),
                "categorical_columns": len(df.select_dtypes(include=['object']).columns),
                "sentinel_values_found": len(local_stats["sentinel_columns"]),
                "negative_prices_found": len(local_stats["negative_price_columns"])
            },
            drift_risk=drift_risk,
            target_leakage_risk=leakage_risk,
            visualization_data=viz_data,
            retail_metrics=retail_metrics,
            ml_context=None  # Will be filled by assess_quality_with_context if Agent-First enabled
        )
    
    def _generate_retail_recommendations(
        self,
        local_stats: Dict[str, Any],
        sentinel_report: Dict[str, Any],
        price_report: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate retail-specific recommendations."""
        retail_recs = []
        
        # Sentinel value recommendations
        if local_stats.get("has_sentinel_values"):
            retail_recs.append({
                "priority": "high",
                "category": "sentinel_handling",
                "title": "Replace Sentinel Values",
                "description": f"Sentinel values detected in {len(local_stats['sentinel_columns'])} columns",
                "actions": [
                    f"Replace sentinels in: {', '.join(local_stats['sentinel_columns'][:3])}",
                    "Use category-based median imputation if category column exists",
                    "Preserve legitimate zero values in stock/quantity columns"
                ],
                "columns_affected": local_stats['sentinel_columns']
            })
        
        # Negative price recommendations
        if local_stats.get("has_negative_prices"):
            retail_recs.append({
                "priority": "high",
                "category": "price_correction",
                "title": "Fix Negative Prices",
                "description": f"Negative prices found in {len(local_stats['negative_price_columns'])} columns",
                "actions": [
                    f"Correct prices in: {', '.join(local_stats['negative_price_columns'])}",
                    "Use median by category if category column available",
                    "Otherwise use overall median for price correction"
                ],
                "columns_affected": local_stats['negative_price_columns']
            })
        
        # Category-based imputation recommendation
        if "category" in local_stats.get("columns", []):
            retail_recs.append({
                "priority": "medium",
                "category": "imputation_strategy",
                "title": "Use Category-Based Imputation",
                "description": "Category column detected - leverage for better imputation",
                "actions": [
                    "Group by category for missing value imputation",
                    "Use category-specific medians for numeric features",
                    "Apply category-specific modes for categorical features"
                ]
            })
        
        return retail_recs
    
    def _assess_sentinel_values(self, df: pd.DataFrame) -> Dict:
        """Assess sentinel values in the dataset (retail-specific)."""
        warnings = []
        penalty = 0
        
        for col in df.select_dtypes(include=[np.number]).columns:
            effective_sentinels = self._get_effective_sentinel_values(col)
            sentinel_count = df[col].isin(effective_sentinels).sum()
            
            if sentinel_count > 0:
                sentinel_pct = (sentinel_count / len(df)) * 100
                warnings.append({
                    "type": "sentinel_values",
                    "column": col,
                    "message": f"Column '{col}' contains {sentinel_count} sentinel values {effective_sentinels}",
                    "severity": "medium",
                    "action": "Replace sentinels with NaN and impute by category median",
                    "is_stock": self._is_stock_column(col)
                })
                penalty += 2  # Small penalty per column with sentinels
        
        return {
            "warnings": warnings,
            "penalty": min(10, penalty)  # Cap at 10 points
        }
    
    def _assess_negative_prices(self, df: pd.DataFrame) -> Dict:
        """Assess negative prices in the dataset (retail-specific)."""
        alerts = []
        penalty = 0
        
        price_columns = [col for col in df.columns if 'price' in col.lower() or 'prix' in col.lower()]
        
        for col in price_columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                negative_count = (df[col] < 0).sum()
                
                if negative_count > 0:
                    negative_pct = (negative_count / len(df)) * 100
                    alerts.append({
                        "type": "negative_prices",
                        "column": col,
                        "message": f"Column '{col}' has {negative_count} negative prices ({negative_pct:.1f}%)",
                        "severity": "high",
                        "action": "Fix negative prices using median by category"
                    })
                    penalty += 5  # Higher penalty for price issues
        
        return {
            "alerts": alerts,
            "penalty": min(15, penalty)  # Cap at 15 points
        }
    
    def _assess_missing_values(self, df: pd.DataFrame) -> Dict:
        """Assess missing values with DataRobot-style alerts and configurable threshold."""
        
        alerts = []
        warnings = []
        penalty = 0
        
        column_missing_pct = {}
        
        for col in df.columns:
            missing_pct = (df[col].isnull().sum() / len(df)) * 100
            column_missing_pct[col] = missing_pct
            
            if missing_pct > self.quality_thresholds["missing_critical"] * 100:
                alerts.append({
                    "type": "missing_critical",
                    "column": col,
                    "message": f"Column '{col}' has {missing_pct:.1f}% missing values (exceeds {self.quality_thresholds['missing_critical']*100:.0f}% critical threshold)",
                    "severity": "critical",
                    "action": "Consider dropping or advanced imputation"
                })
                penalty += 15
                
            elif missing_pct > self.quality_thresholds["missing_warning"] * 100:
                warnings.append({
                    "type": "missing_warning",
                    "column": col,
                    "message": f"Column '{col}' has {missing_pct:.1f}% missing values (exceeds {self.quality_thresholds['missing_warning']*100:.0f}% warning threshold)",
                    "severity": "medium",
                    "action": "Recommend imputation strategy (median by category for retail)"
                })
                penalty += 5
        
        return {
            "alerts": alerts,
            "warnings": warnings,
            "penalty": penalty,
            "column_missing_pct": column_missing_pct
        }
    
    def _get_context_recommendations(self, ml_context: Dict[str, Any]) -> List[Dict]:
        """Generate recommendations based on detected ML context."""
        recommendations = []
        
        problem_type = ml_context.get('problem_type', '')
        business_sector = ml_context.get('business_sector', '')
        
        # Retail-specific recommendations
        if business_sector == 'retail':
            recommendations.append({
                "priority": "high",
                "category": "retail_preparation",
                "title": "Retail Data Preparation",
                "description": "Detected retail sector. Special data quality checks recommended.",
                "actions": [
                    "Replace sentinel values (-999, -1, 9999) with NaN and impute by median of category",
                    "Verify and correct negative prices using category-based medians",
                    "Preserve zero values in stock/quantity columns as they are legitimate",
                    "Check SKU validity and aim for 98% GS1 compliance"
                ]
            })
        
        if problem_type == 'fraud_detection':
            recommendations.append({
                "priority": "high",
                "category": "ml_preparation",
                "title": "Fraud Detection Preparation",
                "description": "Detected fraud detection use case. Special preparations recommended.",
                "actions": [
                    "Create velocity and frequency features",
                    "Apply SMOTE or ADASYN for class imbalance",
                    "Engineer IP and device-based features",
                    "Implement time-window aggregations"
                ]
            })
        elif problem_type == 'churn_prediction':
            recommendations.append({
                "priority": "high",
                "category": "ml_preparation",
                "title": "Churn Prediction Preparation",
                "description": "Detected churn prediction use case. Customer analytics preparations recommended.",
                "actions": [
                    "Create RFM (Recency, Frequency, Monetary) features",
                    "Calculate customer lifetime value",
                    "Engineer engagement and activity metrics",
                    "Handle class imbalance with appropriate techniques"
                ]
            })
        elif problem_type == 'sales_forecasting':
            recommendations.append({
                "priority": "high",
                "category": "ml_preparation",
                "title": "Sales Forecasting Preparation",
                "description": "Detected forecasting use case. Time series preparations recommended.",
                "actions": [
                    "Create lag features (1, 7, 30 days)",
                    "Add seasonal decomposition",
                    "Engineer moving averages and trends",
                    "Handle missing time periods appropriately"
                ]
            })
        
        return recommendations
    
    def _generate_recommendations(self, issues: List[Dict], 
                                 missing_report: Dict,
                                 outlier_report: Dict,
                                 dtype_report: Dict,
                                 local_stats: Dict) -> List[Dict]:
        """Generate actionable recommendations based on issues found."""
        
        recommendations = []
        
        # Missing data recommendations
        if missing_report["penalty"] > 10:
            recommendations.append({
                "priority": "high",
                "category": "missing_data",
                "title": "Address Missing Data",
                "description": f"Significant missing data detected (>{self.quality_thresholds['missing_warning']*100:.0f}% in some columns). Consider advanced imputation techniques.",
                "actions": [
                    "Use KNN or MICE imputation for numeric features",
                    "Use mode or create 'Unknown' category for categorical features",
                    "For retail: use median by category for numeric features",
                    f"Drop columns with >{self.quality_thresholds['missing_critical']*100:.0f}% missing values"
                ]
            })
        
        # Outlier recommendations
        if outlier_report["penalty"] > 5:
            recommendations.append({
                "priority": "medium",
                "category": "outliers",
                "title": "Handle Outliers",
                "description": "Multiple columns contain outliers that may affect model performance.",
                "actions": [
                    "Apply robust scaling or winsorization",
                    "Use tree-based models that are robust to outliers",
                    "Investigate outliers for data quality issues",
                    "For prices: verify outliers are not data entry errors"
                ]
            })
        
        # Data type recommendations
        if dtype_report["penalty"] > 5:
            recommendations.append({
                "priority": "medium",
                "category": "data_types",
                "title": "Optimize Data Types",
                "description": "Data type issues detected that may impact model performance.",
                "actions": [
                    "Encode high-cardinality categorical variables",
                    "Convert date strings to datetime objects",
                    "Use appropriate numeric types to reduce memory usage"
                ]
            })
        
        # General best practices
        recommendations.append({
            "priority": "low",
            "category": "best_practices",
            "title": "AutoML Best Practices",
            "description": "General recommendations for optimal AutoML performance.",
            "actions": [
                "Ensure target variable is properly defined",
                "Create train/validation/test splits before training",
                "Document data preprocessing steps for reproducibility",
                "Set up monitoring for production deployment"
            ]
        })
        
        return recommendations
    
    # Keep all other methods unchanged...
    def _assess_outliers(self, df: pd.DataFrame) -> Dict:
        """Assess outliers using IQR method with configurable thresholds."""
        
        warnings = []
        penalty = 0
        outlier_columns = {}
        
        for col in df.select_dtypes(include=[np.number]).columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            outliers = ((df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)).sum()
            outlier_pct = (outliers / len(df)) * 100
            
            if outliers > 0:
                outlier_columns[col] = outlier_pct
                
                if outlier_pct > self.quality_thresholds["outlier_critical"] * 100:
                    warnings.append({
                        "type": "outliers",
                        "column": col,
                        "message": f"Column '{col}' has {outlier_pct:.1f}% outliers",
                        "severity": "high",
                        "action": "Review outliers for data quality issues"
                    })
                    penalty += 5
                elif outlier_pct > self.quality_thresholds["outlier_warning"] * 100:
                    warnings.append({
                        "type": "outliers",
                        "column": col,
                        "message": f"Column '{col}' has {outlier_pct:.1f}% outliers",
                        "severity": "medium",
                        "action": "Consider outlier handling strategy"
                    })
                    penalty += 2
        
        return {
            "warnings": warnings,
            "penalty": penalty,
            "outlier_columns": outlier_columns
        }
    
    def _assess_data_types(self, df: pd.DataFrame) -> Dict:
        """Assess data type issues."""
        
        warnings = []
        penalty = 0
        
        type_counts = {
            "numeric": len(df.select_dtypes(include=[np.number]).columns),
            "categorical": len(df.select_dtypes(include=['object']).columns),
            "datetime": len(df.select_dtypes(include=['datetime64']).columns),
            "boolean": len(df.select_dtypes(include=['bool']).columns)
        }
        
        for col in df.columns:
            # Check high cardinality
            if df[col].dtype == 'object':
                unique_ratio = df[col].nunique() / len(df)
                
                if unique_ratio > self.quality_thresholds["cardinality_high"]:
                    warnings.append({
                        "type": "high_cardinality",
                        "column": col,
                        "message": f"Column '{col}' has very high cardinality ({df[col].nunique()} unique values)",
                        "severity": "medium",
                        "action": "Consider if this is an ID field or needs encoding"
                    })
                    penalty += 3
        
        return {
            "warnings": warnings,
            "penalty": penalty,
            "type_counts": type_counts
        }
    
    def _assess_duplicates(self, df: pd.DataFrame) -> Dict:
        """Assess duplicate rows."""
        
        dup_count = df.duplicated().sum()
        penalty = 0
        
        if dup_count > 0:
            dup_pct = (dup_count / len(df)) * 100
            if dup_pct > 10:
                penalty = 10
            elif dup_pct > 5:
                penalty = 5
            else:
                penalty = 2
        
        return {
            "count": dup_count,
            "percentage": (dup_count / len(df)) * 100 if len(df) > 0 else 0,
            "penalty": penalty
        }
    
    def _assess_target(self, df: pd.DataFrame, target_column: str) -> Dict:
        """Assess target variable quality."""
        
        alerts = []
        penalty = 0
        
        # Check class imbalance for classification
        if df[target_column].dtype in ['object', 'category'] or df[target_column].nunique() < 20:
            value_counts = df[target_column].value_counts()
            if len(value_counts) > 1:
                imbalance_ratio = value_counts.iloc[0] / value_counts.iloc[-1]
                
                if imbalance_ratio > self.quality_thresholds["imbalance_severe"]:
                    alerts.append({
                        "type": "class_imbalance",
                        "message": f"Severe class imbalance detected ({imbalance_ratio:.1f}:1 ratio)",
                        "severity": "high",
                        "action": "Consider resampling techniques or class weights"
                    })
                    penalty += 10
        
        # Check for missing target
        target_missing = df[target_column].isnull().sum()
        if target_missing > 0:
            alerts.append({
                "type": "target_missing",
                "message": f"Target column has {target_missing} missing values",
                "severity": "critical",
                "action": "Remove rows with missing target values"
            })
            penalty += 15
        
        return {
            "alerts": alerts,
            "penalty": penalty
        }
    
    def _detect_target_leakage(
        self, 
        df: pd.DataFrame, 
        target_column: str
    ) -> RiskLevel:
        """
        Detect potential target leakage through correlations.
    
        Returns:
            RiskLevel based on max correlation found:
            - HIGH: correlation >= 0.95 (near perfect)
            - MEDIUM: correlation >= 0.85 (very high)
            - LOW: correlation >= 0.70 (high)
            - NONE: correlation < 0.70
        """
        if target_column not in df.columns:
            return RiskLevel.NONE
            
        target_series = df[target_column]
        max_correlation = 0.0

        if pd.api.types.is_numeric_dtype(target_series):
            target_numeric = pd.Series(target_series, index=df.index)
        else:
            # Factorize categorical targets but preserve the original index so that
            # pandas doesn't misalign during correlation when the dataframe has a
            # custom index (e.g. filtered views or domain identifiers).
            encoded_target = pd.factorize(target_series)[0]
            target_numeric = pd.Series(encoded_target, index=df.index, dtype=float)
            if target_series.isna().any():
                target_numeric = target_numeric.mask(target_series.isna())

        # Check for perfect correlation
        for col in df.columns:
            if col == target_column or not pd.api.types.is_numeric_dtype(df[col]):
                continue

            try:
                feature_series = pd.Series(df[col], index=df.index)
                corr = feature_series.corr(target_numeric)
                if pd.notna(corr) and abs(corr) > self.quality_thresholds["correlation_high"]:
                    max_correlation = max(max_correlation, abs(corr))
            except Exception:
                continue
        
        # Return graduated risk levels
        if max_correlation >= 0.95:
            return RiskLevel.HIGH
        elif max_correlation >= 0.85:
            return RiskLevel.MEDIUM
        elif max_correlation >= 0.70:
            return RiskLevel.LOW
        else:
            return RiskLevel.NONE
        
    def _assess_statistical_anomalies(self, df: pd.DataFrame) -> Dict:
        """Detect statistical anomalies in the data."""
        
        warnings = []
        
        for col in df.select_dtypes(include=[np.number]).columns:
            # Check for zero variance
            if df[col].std() == 0:
                warnings.append({
                    "type": "zero_variance",
                    "column": col,
                    "message": f"Column '{col}' has zero variance (constant values)",
                    "severity": "high",
                    "action": "Remove constant column"
                })
            
            # Check for extreme skewness
            skewness = df[col].skew()
            if abs(skewness) > 2:
                warnings.append({
                    "type": "high_skewness",
                    "column": col,
                    "message": f"Column '{col}' is highly skewed (skewness: {skewness:.2f})",
                    "severity": "low",
                    "action": "Consider log or box-cox transformation"
                })
        
        return {"warnings": warnings}
    
    def _calculate_drift_risk(self, df: pd.DataFrame) -> RiskLevel:
        """Calculate risk of data drift."""
        
        risk_score = 0
        
        # High cardinality increases drift risk
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].nunique() > 100:
                risk_score += 1
        
        # Many numeric features increase drift risk
        if len(df.select_dtypes(include=[np.number]).columns) > 50:
            risk_score += 2
        
        # Time-based features increase drift risk
        date_columns = df.select_dtypes(include=['datetime64']).columns
        if len(date_columns) > 0:
            risk_score += 2
        
        if risk_score >= 4:
            return RiskLevel.HIGH
        elif risk_score >= 2:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW


# Convenience class combining both approaches
class IntelligentDataQualityAgent:
    """
    Combined agent using both Akkio-style chat cleaning and DataRobot-style assessment.
    Enhanced with Agent-First capabilities for intelligent ML context detection.
    """
    
    def __init__(self, llm_provider=None, enable_agent_first: bool = False, config=None):
        self.config = config  # AgentConfig for thresholds
        self.cleaning_agent = AkkioStyleCleaningAgent(llm_provider, enable_agent_first, config) if llm_provider else None
        self.quality_monitor = DataRobotStyleQualityMonitor(enable_agent_first, config)
        self.enable_agent_first = enable_agent_first
        self.universal_agent = None
        
        if enable_agent_first:
            self._init_universal_agent()
    
    def _init_universal_agent(self):
        """Initialize Universal ML Agent for complete Agent-First support."""
        try:
            from automl_platform.agents import UniversalMLAgent, AgentConfig
            
            agent_config = AgentConfig(
                openai_api_key=os.getenv("OPENAI_API_KEY")
            )
            self.universal_agent = UniversalMLAgent(agent_config)
            logger.info("Universal ML Agent initialized for data quality agent")
        except ImportError:
            logger.warning("Universal ML Agent not available")
        except Exception as e:
            logger.error(f"Failed to initialize Universal ML Agent: {e}")
    
    def assess(self, df: pd.DataFrame, target_column: Optional[str] = None) -> DataQualityAssessment:
        """Perform DataRobot-style quality assessment."""
        return self.quality_monitor.assess_quality(df, target_column)
    
    async def assess_with_context(self, df: pd.DataFrame, target_column: Optional[str] = None) -> DataQualityAssessment:
        """Perform quality assessment with ML context detection (Agent-First)."""
        if self.enable_agent_first:
            return await self.quality_monitor.assess_quality_with_context(df, target_column)
        else:
            return self.assess(df, target_column)
    
    async def clean(self, message: str, df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
        """Perform Akkio-style conversational cleaning."""
        if not self.cleaning_agent:
            raise ValueError("LLM provider required for conversational cleaning")
        return await self.cleaning_agent.chat_clean(message, df)
    
    async def auto_clean_for_ml(self, df: pd.DataFrame, target_column: Optional[str] = None) -> pd.DataFrame:
        """
        Automatically clean data based on detected ML context (Agent-First).
        """
        if not self.enable_agent_first or not self.universal_agent:
            raise ValueError("Agent-First mode required for automatic ML cleaning")
        
        try:
            # Use Universal Agent for complete pipeline
            result = await self.universal_agent.automl_without_templates(
                df=df,
                target_col=target_column
            )
            
            if result.success and result.cleaned_data is not None:
                logger.info(f"Auto-cleaning successful for {result.context_detected.problem_type}")
                return result.cleaned_data
            else:
                logger.warning("Auto-cleaning failed, returning original data")
                return df
                
        except Exception as e:
            logger.error(f"Auto-cleaning failed: {e}")
            return df
    
    def get_quality_report(self, assessment: DataQualityAssessment) -> str:
        """Generate a formatted quality report with ML context if available."""
        
        report = f"""
# Data Quality Assessment Report

## Overall Quality Score: {assessment.quality_score:.1f}/100
"""
        
        # Add ML context if available
        if assessment.ml_context:
            report += f"""
## ML Context Detected
- **Problem Type:** {assessment.ml_context.get('problem_type', 'Unknown')}
- **Confidence:** {assessment.ml_context.get('confidence', 0):.1%}
- **Business Sector:** {assessment.ml_context.get('business_sector', 'General')}
- **Temporal Data:** {'Yes' if assessment.ml_context.get('temporal_aspect') else 'No'}
- **Imbalance Detected:** {'Yes' if assessment.ml_context.get('imbalance_detected') else 'No'}
"""
        
        # Add retail metrics if available
        if hasattr(assessment, 'retail_metrics') and assessment.retail_metrics:
            report += f"""
## Retail Data Quality
- **Sentinel Values Found:** {len(assessment.retail_metrics.get('sentinel_columns', []))} columns
- **Negative Prices Found:** {len(assessment.retail_metrics.get('negative_price_columns', []))} columns
- **Category Column Present:** {'Yes' if assessment.retail_metrics.get('has_category_column') else 'No'}
- **GS1 Compliance:** {assessment.retail_metrics.get('gs1_compliance', 0):.1f}%
- **SKU Columns:** {', '.join(assessment.retail_metrics.get('sku_columns_found', [])) or 'None'}
"""
        
        report += f"""
## Critical Alerts ({len(assessment.alerts)})
"""
        for alert in assessment.alerts:
            report += f"- âš ï¸ **{alert['message']}** - {alert['action']}\n"
        
        report += f"\n## Warnings ({len(assessment.warnings)})\n"
        for warning in assessment.warnings[:5]:  # Limit to top 5
            report += f"- âš¡ {warning['message']}\n"
        
        # Add retail-specific recommendations if available
        if hasattr(assessment, 'retail_recommendations') and assessment.retail_recommendations:
            report += f"\n## Retail-Specific Recommendations ({len(assessment.retail_recommendations)})\n"
            for rec in assessment.retail_recommendations:
                report += f"\n### {rec['title']} (Priority: {rec['priority']})\n"
                report += f"{rec['description']}\n"
                report += "**Actions:**\n"
                for action in rec.get('actions', []):
                    report += f"- {action}\n"
        
        report += f"\n## Key Statistics\n"
        for key, value in assessment.statistics.items():
            report += f"- {key.replace('_', ' ').title()}: {value}\n"
        
        report += f"\n## Risk Assessment\n"
        report += f"- Data Drift Risk: **{assessment.drift_risk}**\n"
        leakage_risk = assessment.target_leakage_risk
        if leakage_risk == RiskLevel.NONE:
            leakage_display = "None (no target assessed)"
        else:
            leakage_display = leakage_risk.capitalize()

        report += f"- Target Leakage Risk: **{leakage_display}**\n"
        
        report += f"\n## Top Recommendations\n"
        for i, rec in enumerate(assessment.recommendations[:3], 1):
            report += f"\n### {i}. {rec['title']} (Priority: {rec['priority']})\n"
            report += f"{rec['description']}\n"
            report += "**Actions:**\n"
            for action in rec['actions']:
                report += f"- {action}\n"
        
        return report
    
    def get_cleaning_suggestions(self, df: pd.DataFrame) -> List[str]:
        """Get intelligent cleaning suggestions based on data quality."""
        if self.cleaning_agent:
            return self.cleaning_agent.get_cleaning_suggestions(df)
        return []


# Example usage
if __name__ == "__main__":
    # Create sample data with quality issues including retail-specific problems
    np.random.seed(42)
    df = pd.DataFrame({
        'numeric_clean': np.random.randn(100),
        'numeric_outliers': np.concatenate([np.random.randn(95), [100, -100, 200, -200, 300]]),
        'numeric_missing': np.concatenate([np.random.randn(60), [np.nan] * 40]),  # 40% missing
        'price': np.concatenate([np.random.uniform(10, 100, 90), [-999] * 5, [-5, -10, -15, -20, -25]]),  # Sentinels and negatives
        'stock_quantity': np.concatenate([np.random.randint(0, 100, 95), [0, 0, 0, -999, -999]]),  # Mix of legitimate zeros and sentinels
        'category': np.random.choice(['Electronics', 'Clothing', 'Food'], 100),
        'categorical': np.random.choice(['A', 'B', 'C'], 100),
        'sku': [f'SKU{i:013d}' for i in range(100)],  # Valid GS1-13 format
        'high_cardinality': [f'ID_{i:04d}' for i in range(100)],
        'constant': [1] * 100,
        'target': np.random.choice([0, 1], 100, p=[0.9, 0.1])  # Imbalanced
    })
    
    # Add some duplicates
    df = pd.concat([df, df.iloc[:5]], ignore_index=True)
    
    # Initialize agent with Agent-First enabled and config
    from .agent_config import AgentConfig
    config = AgentConfig()
    config.user_context["secteur_activite"] = "retail"
    
    agent = IntelligentDataQualityAgent(enable_agent_first=True, config=config)
    
    # Perform assessment
    assessment = agent.assess(df, target_column='target')
    
    # Print report
    print(agent.get_quality_report(assessment))
    
    # Show quality score
    print(f"\n{'='*50}")
    print(f"Quality Score: {assessment.quality_score:.1f}/100")
    print(f"Alerts: {len(assessment.alerts)}")
    print(f"Warnings: {len(assessment.warnings)}")
    print(f"Recommendations: {len(assessment.recommendations)}")
    print(f"Retail Recommendations: {len(assessment.retail_recommendations)}")
    
    # Test Agent-First context detection
    import asyncio
    
    async def test_agent_first():
        assessment_with_context = await agent.assess_with_context(df, target_column='target')
        if assessment_with_context.ml_context:
            print(f"\n{'='*50}")
            print("Agent-First ML Context Detected:")
            print(f"Problem Type: {assessment_with_context.ml_context['problem_type']}")
            print(f"Confidence: {assessment_with_context.ml_context['confidence']:.1%}")
    
    # Run if OpenAI API key is available
    if os.getenv("OPENAI_API_KEY"):
        asyncio.run(test_agent_first())
