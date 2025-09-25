"""
Intelligent Data Cleaning Interface
Combines OpenAI Agents with Data Quality Agent for comprehensive cleaning
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import asyncio
import logging
from datetime import datetime

from ..data_quality_agent import (
    DataRobotStyleQualityMonitor,
    IntelligentDataQualityAgent,
    DataQualityAssessment
)
from .data_cleaning_orchestrator import DataCleaningOrchestrator
from .agent_config import AgentConfig

logger = logging.getLogger(__name__)


class IntelligentDataCleaner:
    """
    Intelligent interface combining:
    - DataRobot-style quality assessment
    - OpenAI agents for automated cleaning
    - Akkio-style conversational interface
    """
    
    def __init__(self, config: Optional[AgentConfig] = None, llm_provider=None):
        """
        Initialize intelligent cleaner with both systems
        
        Args:
            config: Configuration for OpenAI agents
            llm_provider: LLM provider for conversational cleaning (optional)
        """
        self.config = config or AgentConfig()
        
        # Initialize both systems
        self.orchestrator = DataCleaningOrchestrator(self.config) if config else None
        self.quality_agent = IntelligentDataQualityAgent(llm_provider)
        self.monitor = DataRobotStyleQualityMonitor()
        
        # Track cleaning history
        self.cleaning_history = []
        self.quality_scores = {}
    
    async def smart_clean(
        self, 
        df: pd.DataFrame, 
        user_context: Dict[str, Any],
        mode: str = "auto"
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Smart cleaning that chooses the best approach based on data quality
        
        Args:
            df: Input dataframe
            user_context: User context (sector, target, etc.)
            mode: "auto", "agents", "conversational", or "hybrid"
            
        Returns:
            Tuple of (cleaned_dataframe, comprehensive_report)
        """
        logger.info(f"Starting smart cleaning in {mode} mode")
        
        # Step 1: Always start with quality assessment
        assessment = self.quality_agent.assess(df, user_context.get("target_variable"))
        
        # Store initial quality
        self.quality_scores["before"] = assessment.quality_score
        
        # Step 2: Choose cleaning strategy based on mode and quality
        if mode == "auto":
            mode = self._determine_best_mode(assessment, user_context)
            logger.info(f"Auto-selected mode: {mode}")
        
        # Step 3: Apply cleaning based on mode
        if mode == "agents":
            cleaned_df, report = await self._clean_with_agents(df, user_context, assessment)
            
        elif mode == "conversational":
            cleaned_df, report = await self._clean_conversationally(df, assessment)
            
        elif mode == "hybrid":
            cleaned_df, report = await self._clean_hybrid(df, user_context, assessment)
            
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        # Step 4: Final quality assessment
        final_assessment = self.quality_agent.assess(
            cleaned_df, 
            user_context.get("target_variable")
        )
        self.quality_scores["after"] = final_assessment.quality_score
        
        # Step 5: Generate comprehensive report
        comprehensive_report = self._generate_comprehensive_report(
            assessment, 
            final_assessment, 
            report,
            mode
        )
        
        # Track history
        self.cleaning_history.append({
            "timestamp": datetime.now(),
            "mode": mode,
            "quality_improvement": final_assessment.quality_score - assessment.quality_score,
            "report": comprehensive_report
        })
        
        return cleaned_df, comprehensive_report
    
    def _determine_best_mode(
        self, 
        assessment: DataQualityAssessment, 
        user_context: Dict[str, Any]
    ) -> str:
        """
        Determine the best cleaning mode based on data quality
        
        Decision logic:
        - High quality (>80): Use conversational for fine-tuning
        - Medium quality (50-80): Use agents for systematic cleaning
        - Low quality (<50): Use hybrid for comprehensive approach
        - Sector-specific: Always use agents for validation
        """
        quality_score = assessment.quality_score
        has_critical_alerts = len(assessment.alerts) > 0
        sector = user_context.get("secteur_activite")
        
        # If sector is specified and has critical issues, use agents
        if sector and (has_critical_alerts or quality_score < 70):
            return "agents"
        
        # For very poor quality, use hybrid approach
        if quality_score < 50:
            return "hybrid"
        
        # For medium quality, use agents
        if quality_score < 80:
            return "agents"
        
        # For high quality, use conversational for fine-tuning
        return "conversational"
    
    async def _clean_with_agents(
        self,
        df: pd.DataFrame,
        user_context: Dict[str, Any],
        assessment: DataQualityAssessment
    ) -> Tuple[pd.DataFrame, Dict]:
        """Clean using OpenAI agents"""
        if not self.orchestrator:
            raise ValueError("OpenAI agents not configured")
        
        logger.info("Cleaning with OpenAI agents")
        
        # Add quality assessment to context
        user_context["quality_assessment"] = {
            "score": assessment.quality_score,
            "critical_issues": [a["message"] for a in assessment.alerts],
            "warnings": [w["message"] for w in assessment.warnings[:5]]
        }
        
        # Run agent cleaning
        cleaned_df, agent_report = await self.orchestrator.clean_dataset(
            df, 
            user_context
        )
        
        return cleaned_df, agent_report
    
    async def _clean_conversationally(
        self,
        df: pd.DataFrame,
        assessment: DataQualityAssessment
    ) -> Tuple[pd.DataFrame, Dict]:
        """Clean using conversational interface"""
        logger.info("Cleaning conversationally based on assessment")
        
        cleaned_df = df.copy()
        conversations = []
        
        # Generate cleaning prompts from assessment
        cleaning_prompts = self._generate_cleaning_prompts(assessment)
        
        # Apply each cleaning suggestion
        for prompt in cleaning_prompts[:5]:  # Limit to top 5 actions
            try:
                cleaned_df, response = await self.quality_agent.clean(
                    prompt, 
                    cleaned_df
                )
                conversations.append({
                    "prompt": prompt,
                    "response": response
                })
            except Exception as e:
                logger.warning(f"Failed to apply: {prompt}. Error: {e}")
        
        report = {
            "method": "conversational",
            "conversations": conversations,
            "actions_applied": len(conversations)
        }
        
        return cleaned_df, report
    
    async def _clean_hybrid(
        self,
        df: pd.DataFrame,
        user_context: Dict[str, Any],
        assessment: DataQualityAssessment
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Hybrid cleaning: Agents for systematic issues, conversational for fine-tuning
        """
        logger.info("Hybrid cleaning approach")
        
        # Phase 1: Use agents for major issues
        if assessment.quality_score < 70 and self.orchestrator:
            df, agent_report = await self._clean_with_agents(
                df, 
                user_context, 
                assessment
            )
        else:
            agent_report = {"skipped": "Quality sufficient"}
        
        # Phase 2: Re-assess
        mid_assessment = self.quality_agent.assess(
            df, 
            user_context.get("target_variable")
        )
        
        # Phase 3: Use conversational for remaining issues
        if mid_assessment.quality_score < 90:
            df, conv_report = await self._clean_conversationally(
                df, 
                mid_assessment
            )
        else:
            conv_report = {"skipped": "Quality sufficient"}
        
        report = {
            "method": "hybrid",
            "agent_phase": agent_report,
            "conversational_phase": conv_report,
            "mid_quality_score": mid_assessment.quality_score
        }
        
        return df, report
    
    def _generate_cleaning_prompts(
        self, 
        assessment: DataQualityAssessment
    ) -> List[str]:
        """Generate cleaning prompts from quality assessment"""
        prompts = []
        
        # Handle critical alerts first
        for alert in assessment.alerts:
            if "missing" in alert["message"].lower():
                col = self._extract_column_name(alert["message"])
                if col:
                    prompts.append(f"Fill missing values in {col} with appropriate method")
            
            elif "imbalance" in alert["message"].lower():
                prompts.append("Apply SMOTE to handle class imbalance")
        
        # Handle warnings
        for warning in assessment.warnings[:3]:
            if "outlier" in warning["message"].lower():
                col = self._extract_column_name(warning["message"])
                if col:
                    prompts.append(f"Remove outliers from {col} using IQR method")
            
            elif "duplicate" in warning["message"].lower():
                prompts.append("Remove duplicate rows")
            
            elif "high_cardinality" in warning.get("type", ""):
                col = warning.get("column")
                if col:
                    prompts.append(f"Reduce cardinality of {col} by grouping rare categories")
        
        return prompts
    
    def _extract_column_name(self, message: str) -> Optional[str]:
        """Extract column name from message"""
        import re
        
        # Pattern: Column 'name' or column name
        pattern = r"[Cc]olumn ['\"]?(\w+)['\"]?"
        match = re.search(pattern, message)
        
        if match:
            return match.group(1)
        return None
    
    def _generate_comprehensive_report(
        self,
        initial_assessment: DataQualityAssessment,
        final_assessment: DataQualityAssessment,
        cleaning_report: Dict,
        mode: str
    ) -> Dict[str, Any]:
        """Generate comprehensive cleaning report"""
        
        quality_improvement = final_assessment.quality_score - initial_assessment.quality_score
        
        report = {
            "summary": {
                "mode": mode,
                "initial_quality": initial_assessment.quality_score,
                "final_quality": final_assessment.quality_score,
                "improvement": quality_improvement,
                "success": quality_improvement > 0
            },
            "initial_issues": {
                "alerts": len(initial_assessment.alerts),
                "warnings": len(initial_assessment.warnings)
            },
            "final_issues": {
                "alerts": len(final_assessment.alerts),
                "warnings": len(final_assessment.warnings)
            },
            "resolved_issues": {
                "alerts": len(initial_assessment.alerts) - len(final_assessment.alerts),
                "warnings": len(initial_assessment.warnings) - len(final_assessment.warnings)
            },
            "cleaning_details": cleaning_report,
            "recommendations": final_assessment.recommendations[:3],
            "risks": {
                "drift_risk": final_assessment.drift_risk,
                "leakage_risk": final_assessment.target_leakage_risk
            }
        }
        
        # Add performance metrics
        if mode == "agents" and "execution_history" in cleaning_report:
            report["performance"] = {
                "steps_executed": len(cleaning_report.get("execution_history", [])),
                "transformations_applied": len(cleaning_report.get("transformations", []))
            }
        
        return report
    
    def get_cleaning_summary(self) -> Dict[str, Any]:
        """Get summary of all cleaning operations performed"""
        if not self.cleaning_history:
            return {"message": "No cleaning operations performed yet"}
        
        total_operations = len(self.cleaning_history)
        avg_improvement = sum(
            h["quality_improvement"] for h in self.cleaning_history
        ) / total_operations
        
        modes_used = {}
        for h in self.cleaning_history:
            mode = h["mode"]
            modes_used[mode] = modes_used.get(mode, 0) + 1
        
        return {
            "total_operations": total_operations,
            "average_quality_improvement": avg_improvement,
            "modes_used": modes_used,
            "last_operation": self.cleaning_history[-1]["timestamp"],
            "best_improvement": max(
                h["quality_improvement"] for h in self.cleaning_history
            )
        }
    
    async def recommend_cleaning_approach(
        self,
        df: pd.DataFrame,
        user_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Recommend the best cleaning approach without applying it"""
        
        # Assess quality
        assessment = self.quality_agent.assess(
            df, 
            user_context.get("target_variable")
        )
        
        # Determine best mode
        recommended_mode = self._determine_best_mode(assessment, user_context)
        
        # Generate recommendation
        recommendation = {
            "recommended_mode": recommended_mode,
            "current_quality": assessment.quality_score,
            "reasoning": self._get_mode_reasoning(recommended_mode, assessment),
            "estimated_time": self._estimate_cleaning_time(df, recommended_mode),
            "key_issues": [
                alert["message"] for alert in assessment.alerts[:3]
            ] + [
                warning["message"] for warning in assessment.warnings[:2]
            ]
        }
        
        return recommendation
    
    def _get_mode_reasoning(
        self, 
        mode: str, 
        assessment: DataQualityAssessment
    ) -> str:
        """Get reasoning for mode selection"""
        
        reasonings = {
            "agents": f"OpenAI agents recommended due to systematic issues (quality: {assessment.quality_score:.1f}/100) and need for sector-specific validation.",
            "conversational": f"Conversational cleaning recommended for fine-tuning already good quality data (quality: {assessment.quality_score:.1f}/100).",
            "hybrid": f"Hybrid approach recommended due to multiple complex issues requiring both systematic and interactive cleaning (quality: {assessment.quality_score:.1f}/100)."
        }
        
        return reasonings.get(mode, "Mode selected based on data characteristics.")
    
    def _estimate_cleaning_time(self, df: pd.DataFrame, mode: str) -> str:
        """Estimate cleaning time based on data size and mode"""
        
        rows = len(df)
        cols = len(df.columns)
        
        # Base time estimates (in seconds)
        if mode == "agents":
            time_estimate = 30 + (rows / 1000) * 5 + cols * 2
        elif mode == "conversational":
            time_estimate = 20 + (rows / 1000) * 3 + cols * 1
        else:  # hybrid
            time_estimate = 50 + (rows / 1000) * 8 + cols * 3
        
        if time_estimate < 60:
            return f"~{int(time_estimate)} seconds"
        elif time_estimate < 3600:
            return f"~{int(time_estimate / 60)} minutes"
        else:
            return f"~{time_estimate / 3600:.1f} hours"


# Convenience function for easy access
async def smart_clean_data(
    df: pd.DataFrame,
    user_context: Dict[str, Any],
    config: Optional[AgentConfig] = None,
    mode: str = "auto"
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Convenience function for smart data cleaning
    
    Args:
        df: DataFrame to clean
        user_context: Context including sector, target variable, etc.
        config: Optional configuration for OpenAI agents
        mode: Cleaning mode ("auto", "agents", "conversational", "hybrid")
    
    Returns:
        Tuple of (cleaned_df, report)
    """
    cleaner = IntelligentDataCleaner(config)
    return await cleaner.smart_clean(df, user_context, mode)


# Example usage
if __name__ == "__main__":
    import asyncio
    
    # Create sample data
    np.random.seed(42)
    sample_df = pd.DataFrame({
        'numeric': np.random.randn(100),
        'categorical': np.random.choice(['A', 'B', 'C'], 100),
        'target': np.random.choice([0, 1], 100)
    })
    
    # Add some quality issues
    sample_df.loc[:10, 'numeric'] = np.nan
    
    # Define context
    context = {
        "secteur_activite": "finance",
        "target_variable": "target",
        "contexte_metier": "Risk prediction"
    }
    
    async def demo():
        # Create intelligent cleaner
        cleaner = IntelligentDataCleaner()
        
        # Get recommendation
        recommendation = await cleaner.recommend_cleaning_approach(sample_df, context)
        print(f"Recommended approach: {recommendation['recommended_mode']}")
        print(f"Reasoning: {recommendation['reasoning']}")
        
        # Clean data
        cleaned_df, report = await cleaner.smart_clean(sample_df, context, mode="auto")
        print(f"Quality improvement: {report['summary']['improvement']:.1f} points")
        
    # Run demo
    asyncio.run(demo())
