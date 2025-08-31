"""
Data Quality Agent - Conversational data cleaning inspired by Akkio's GPT-4 approach
References: Akkio's chat-based cleaning and DataRobot's Data Quality Assessment
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
import logging
import json
import asyncio
from datetime import datetime
from dataclasses import dataclass, asdict
import re

logger = logging.getLogger(__name__)


@dataclass
class DataQualityAssessment:
    """DataRobot-style quality assessment with visual alerts."""
    quality_score: float  # 0-100
    alerts: List[Dict[str, Any]]  # Critical issues requiring attention
    warnings: List[Dict[str, Any]]  # Non-critical issues
    recommendations: List[Dict[str, Any]]  # Suggested improvements
    statistics: Dict[str, Any]  # Statistical summary
    drift_risk: str  # "low", "medium", "high"
    target_leakage_risk: bool
    visualization_data: Dict[str, Any]  # Data for visual quality assessment


class AkkioStyleCleaningAgent:
    """
    Conversational data cleaning agent inspired by Akkio's GPT-4 chatbot.
    Allows users to clean data through natural dialogue.
    """
    
    def __init__(self, llm_provider):
        self.llm = llm_provider
        self.conversation_history = []
        self.cleaning_actions = []
        self.undo_stack = []
        
    async def chat_clean(self, user_message: str, df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
        """
        Akkio-style conversational cleaning.
        User can say things like:
        - "Remove outliers from the price column"
        - "Fill missing values with median"
        - "Combine first_name and last_name columns"
        - "Format dates to YYYY-MM-DD"
        """
        
        # Add to conversation history
        self.conversation_history.append({"role": "user", "message": user_message})
        
        # Analyze user intent
        intent = await self._analyze_cleaning_intent(user_message, df)
        
        # Generate cleaning code
        cleaning_code = await self._generate_cleaning_code(intent, df)
        
        # Preview changes
        preview = self._preview_changes(df, cleaning_code)
        
        # Generate response
        response = f"""
I understand you want to {intent['action']}. Here's what I'll do:

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
            "preview": preview
        })
        
        return df, response
    
    async def _analyze_cleaning_intent(self, message: str, df: pd.DataFrame) -> Dict:
        """Analyze user's cleaning intent using LLM."""
        
        prompt = f"""
Analyze this data cleaning request:
User message: "{message}"

Dataset info:
- Columns: {list(df.columns)}
- Shape: {df.shape}
- Types: {df.dtypes.to_dict()}

Extract:
1. Action type (remove, fill, transform, combine, format, filter)
2. Target columns
3. Specific parameters
4. Safety concerns

Return as JSON.
"""
        
        response = await self.llm.generate(prompt, temperature=0.1)
        
        try:
            intent = json.loads(response.content)
        except:
            intent = {
                "action": "unknown",
                "description": message,
                "columns": [],
                "parameters": {}
            }
        
        return intent
    
    async def _generate_cleaning_code(self, intent: Dict, df: pd.DataFrame) -> str:
        """Generate Python code for the cleaning action."""
        
        prompt = f"""
Generate pandas code for this data cleaning task:
Intent: {json.dumps(intent)}

Requirements:
- Use df as the dataframe variable
- Handle edge cases
- Preserve data types where possible
- Add error handling
- Make it efficient

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
            return df_cleaned
            
        except Exception as e:
            logger.error(f"Failed to apply cleaning: {e}")
            return df
    
    def undo_last_action(self, df: pd.DataFrame) -> pd.DataFrame:
        """Undo the last cleaning action."""
        
        if self.undo_stack:
            return self.undo_stack.pop()
        return df


class DataRobotStyleQualityMonitor:
    """
    DataRobot-style Data Quality Assessment with visual indicators.
    Provides comprehensive quality metrics and recommendations.
    """
    
    def __init__(self):
        self.quality_thresholds = {
            "missing_critical": 0.5,  # >50% missing is critical
            "missing_warning": 0.2,   # >20% missing is warning
            "outlier_critical": 0.15, # >15% outliers is critical
            "cardinality_high": 0.9,  # >90% unique is too high
            "correlation_high": 0.95, # >0.95 correlation suggests redundancy
            "imbalance_severe": 20    # >20:1 class ratio is severe
        }
    
    def assess_quality(self, df: pd.DataFrame, target_column: Optional[str] = None) -> DataQualityAssessment:
        """
        Comprehensive quality assessment following DataRobot's approach.
        
        Returns visual-ready quality metrics with alerts and recommendations.
        """
        
        alerts = []
        warnings = []
        recommendations = []
        
        # Calculate quality score (starts at 100)
        quality_score = 100.0
        
        # 1. Missing values assessment
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
        
        # 5. Target-specific assessments
        if target_column and target_column in df.columns:
            target_report = self._assess_target(df, target_column)
            quality_score -= target_report["penalty"]
            alerts.extend(target_report["alerts"])
            
            # Check for leakage
            leakage_risk = self._detect_target_leakage(df, target_column)
        else:
            leakage_risk = False
        
        # 6. Statistical anomalies
        stat_report = self._assess_statistical_anomalies(df)
        warnings.extend(stat_report["warnings"])
        
        # 7. Generate recommendations based on issues
        recommendations = self._generate_recommendations(
            alerts + warnings,
            missing_report,
            outlier_report,
            dtype_report
        )
        
        # 8. Calculate drift risk
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
                "duplicates": duplicate_report["penalty"]
            }
        }
        
        return DataQualityAssessment(
            quality_score=quality_score,
            alerts=alerts,
            warnings=warnings,
            recommendations=recommendations,
            statistics={
                "rows": len(df),
                "columns": len(df.columns),
                "missing_cells": df.isnull().sum().sum(),
                "duplicate_rows": duplicate_report["count"],
                "numeric_columns": len(df.select_dtypes(include=[np.number]).columns),
                "categorical_columns": len(df.select_dtypes(include=['object']).columns)
            },
            drift_risk=drift_risk,
            target_leakage_risk=leakage_risk,
            visualization_data=viz_data
        )
    
    def _assess_missing_values(self, df: pd.DataFrame) -> Dict:
        """Assess missing values with DataRobot-style alerts."""
        
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
                    "message": f"Column '{col}' has {missing_pct:.1f}% missing values",
                    "severity": "critical",
                    "action": "Consider dropping or advanced imputation"
                })
                penalty += 15
                
            elif missing_pct > self.quality_thresholds["missing_warning"] * 100:
                warnings.append({
                    "type": "missing_warning",
                    "column": col,
                    "message": f"Column '{col}' has {missing_pct:.1f}% missing values",
                    "severity": "medium",
                    "action": "Recommend imputation strategy"
                })
                penalty += 5
        
        return {
            "alerts": alerts,
            "warnings": warnings,
            "penalty": penalty,
            "column_missing_pct": column_missing_pct
        }
    
    def _assess_outliers(self, df: pd.DataFrame) -> Dict:
        """Assess outliers using IQR method."""
        
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
    
    def _detect_target_leakage(self, df: pd.DataFrame, target_column: str) -> bool:
        """Detect potential target leakage."""
        
        # Check for perfect correlation
        for col in df.columns:
            if col != target_column and pd.api.types.is_numeric_dtype(df[col]):
                if pd.api.types.is_numeric_dtype(df[target_column]):
                    corr = df[col].corr(df[target_column])
                    if abs(corr) > self.quality_thresholds["correlation_high"]:
                        return True
        
        # Check for columns with target in name
        target_keywords = ['target', 'label', 'y', 'outcome', 'result']
        for col in df.columns:
            if col != target_column:
                if any(keyword in col.lower() for keyword in target_keywords):
                    return True
        
        return False
    
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
    
    def _calculate_drift_risk(self, df: pd.DataFrame) -> str:
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
            return "high"
        elif risk_score >= 2:
            return "medium"
        else:
            return "low"
    
    def _generate_recommendations(self, issues: List[Dict], 
                                 missing_report: Dict,
                                 outlier_report: Dict,
                                 dtype_report: Dict) -> List[Dict]:
        """Generate actionable recommendations based on issues found."""
        
        recommendations = []
        
        # Missing data recommendations
        if missing_report["penalty"] > 10:
            recommendations.append({
                "priority": "high",
                "category": "missing_data",
                "title": "Address Missing Data",
                "description": "Significant missing data detected. Consider advanced imputation techniques or removing highly incomplete features.",
                "actions": [
                    "Use KNN or MICE imputation for numeric features",
                    "Use mode or create 'Unknown' category for categorical features",
                    "Drop columns with >50% missing values"
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
                    "Investigate outliers for data quality issues"
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


# Convenience class combining both approaches
class IntelligentDataQualityAgent:
    """
    Combined agent using both Akkio-style chat cleaning and DataRobot-style assessment.
    """
    
    def __init__(self, llm_provider=None):
        self.cleaning_agent = AkkioStyleCleaningAgent(llm_provider) if llm_provider else None
        self.quality_monitor = DataRobotStyleQualityMonitor()
        
    def assess(self, df: pd.DataFrame, target_column: Optional[str] = None) -> DataQualityAssessment:
        """Perform DataRobot-style quality assessment."""
        return self.quality_monitor.assess_quality(df, target_column)
    
    async def clean(self, message: str, df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
        """Perform Akkio-style conversational cleaning."""
        if not self.cleaning_agent:
            raise ValueError("LLM provider required for conversational cleaning")
        return await self.cleaning_agent.chat_clean(message, df)
    
    def get_quality_report(self, assessment: DataQualityAssessment) -> str:
        """Generate a formatted quality report."""
        
        report = f"""
# Data Quality Assessment Report

## Overall Quality Score: {assessment.quality_score:.1f}/100

## Critical Alerts ({len(assessment.alerts)})
"""
        for alert in assessment.alerts:
            report += f"- ⚠️ **{alert['message']}** - {alert['action']}\n"
        
        report += f"\n## Warnings ({len(assessment.warnings)})\n"
        for warning in assessment.warnings[:5]:  # Limit to top 5
            report += f"- ⚡ {warning['message']}\n"
        
        report += f"\n## Key Statistics\n"
        for key, value in assessment.statistics.items():
            report += f"- {key.replace('_', ' ').title()}: {value}\n"
        
        report += f"\n## Risk Assessment\n"
        report += f"- Data Drift Risk: **{assessment.drift_risk}**\n"
        report += f"- Target Leakage Risk: **{'Yes' if assessment.target_leakage_risk else 'No'}**\n"
        
        report += f"\n## Top Recommendations\n"
        for i, rec in enumerate(assessment.recommendations[:3], 1):
            report += f"\n### {i}. {rec['title']} (Priority: {rec['priority']})\n"
            report += f"{rec['description']}\n"
            report += "**Actions:**\n"
            for action in rec['actions']:
                report += f"- {action}\n"
        
        return report


# Example usage
if __name__ == "__main__":
    # Create sample data with quality issues
    np.random.seed(42)
    df = pd.DataFrame({
        'numeric_clean': np.random.randn(100),
        'numeric_outliers': np.concatenate([np.random.randn(95), [100, -100, 200, -200, 300]]),
        'numeric_missing': np.concatenate([np.random.randn(70), [np.nan] * 30]),
        'categorical': np.random.choice(['A', 'B', 'C'], 100),
        'high_cardinality': [f'ID_{i}' for i in range(100)],
        'constant': [1] * 100,
        'target': np.random.choice([0, 1], 100, p=[0.9, 0.1])  # Imbalanced
    })
    
    # Add some duplicates
    df = pd.concat([df, df.iloc[:5]], ignore_index=True)
    
    # Initialize agent
    agent = IntelligentDataQualityAgent()
    
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
