"""
Intelligent Data Cleaning Interface
Combines OpenAI Agents with Data Quality Agent for comprehensive cleaning
NOW ENHANCED WITH CLAUDE SDK FOR STRATEGIC INSIGHTS
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import asyncio
import logging
from datetime import datetime
import json
import importlib.util

from ..data_quality_agent import (
    DataRobotStyleQualityMonitor,
    IntelligentDataQualityAgent,
    DataQualityAssessment
)
from .data_cleaning_orchestrator import DataCleaningOrchestrator
from .agent_config import AgentConfig

# Claude SDK for strategic insights
_anthropic_spec = importlib.util.find_spec("anthropic")
if _anthropic_spec is not None:
    from anthropic import AsyncAnthropic
else:
    AsyncAnthropic = None

logger = logging.getLogger(__name__)


class IntelligentDataCleaner:
    """
    Intelligent interface combining:
    - DataRobot-style quality assessment
    - OpenAI agents for automated cleaning
    - Akkio-style conversational interface
    - Claude SDK for strategic cleaning insights and recommendations
    """
    
    def __init__(
        self, 
        config: Optional[AgentConfig] = None, 
        llm_provider=None,
        use_claude: bool = True
    ):
        """
        Initialize intelligent cleaner with both systems
        
        Args:
            config: Configuration for agents
            llm_provider: LLM provider for conversational cleaning (optional)
            use_claude: Whether to use Claude SDK for strategic insights
        """
        self.config = config or AgentConfig()
        self.use_claude = use_claude and AsyncAnthropic is not None
        
        # Initialize both systems
        self.orchestrator = DataCleaningOrchestrator(
            self.config, 
            use_claude=use_claude
        ) if config else None
        
        self.quality_agent = IntelligentDataQualityAgent(llm_provider)
        self.monitor = DataRobotStyleQualityMonitor()
        
        # Initialize Claude client for strategic insights
        if self.use_claude:
            self.claude_client = AsyncAnthropic(
                api_key=self.config.anthropic_api_key
            )
            self.claude_model = self.config.claude_model
            logger.info("💎 Claude SDK enabled for strategic cleaning insights")
        else:
            self.claude_client = None
            if use_claude:
                logger.warning("⚠️ Claude SDK requested but not available")
            else:
                logger.info("📋 Using rule-based cleaning recommendations")
        
        # Track cleaning history
        self.cleaning_history = []
        self.quality_scores = {}
        
        # Metrics tracking
        self.metrics = {
            "total_cleanings": 0,
            "claude_insights_generated": 0,
            "claude_mode_selections": 0,
            "claude_recommendations": 0,
            "rule_based_fallbacks": 0
        }
    
    async def smart_clean(
        self, 
        df: pd.DataFrame, 
        user_context: Dict[str, Any],
        mode: str = "auto"
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Smart cleaning that chooses the best approach based on data quality
        ENHANCED WITH CLAUDE FOR STRATEGIC DECISIONS
        
        Args:
            df: Input dataframe
            user_context: User context (sector, target, etc.)
            mode: "auto", "agents", "conversational", or "hybrid"
            
        Returns:
            Tuple of (cleaned_dataframe, comprehensive_report)
        """
        self.metrics["total_cleanings"] += 1
        logger.info(f"🚀 Starting smart cleaning in {mode} mode")
        
        # Step 1: Always start with quality assessment
        assessment = self.quality_agent.assess(df, user_context.get("target_variable"))
        self.quality_scores["before"] = assessment.quality_score
        
        # Step 2: Choose cleaning strategy with Claude if available
        if mode == "auto":
            if self.use_claude:
                mode = await self._claude_determine_best_mode(df, assessment, user_context)
                self.metrics["claude_mode_selections"] += 1
            else:
                mode = self._determine_best_mode(assessment, user_context)
            
            logger.info(f"💡 Auto-selected mode: {mode}")
        
        # Step 3: Generate strategic cleaning plan with Claude if available
        if self.use_claude:
            cleaning_strategy = await self._claude_generate_cleaning_strategy(
                df, assessment, user_context, mode
            )
            self.metrics["claude_insights_generated"] += 1
            logger.info(f"💎 Claude Strategy:\n{cleaning_strategy.get('summary', 'N/A')}")
        else:
            cleaning_strategy = self._generate_basic_strategy(df, assessment, mode)
        
        # Step 4: Apply cleaning based on mode
        if mode == "agents":
            cleaned_df, report = await self._clean_with_agents(
                df, user_context, assessment, cleaning_strategy
            )
        elif mode == "conversational":
            cleaned_df, report = await self._clean_conversationally(
                df, assessment, cleaning_strategy
            )
        elif mode == "hybrid":
            cleaned_df, report = await self._clean_hybrid(
                df, user_context, assessment, cleaning_strategy
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        # Step 5: Final quality assessment
        final_assessment = self.quality_agent.assess(
            cleaned_df, 
            user_context.get("target_variable")
        )
        self.quality_scores["after"] = final_assessment.quality_score
        
        # Step 6: Generate comprehensive report with Claude insights
        comprehensive_report = await self._generate_comprehensive_report(
            assessment, 
            final_assessment, 
            report,
            mode,
            cleaning_strategy
        )
        
        # Track history
        self.cleaning_history.append({
            "timestamp": datetime.now(),
            "mode": mode,
            "quality_improvement": final_assessment.quality_score - assessment.quality_score,
            "report": comprehensive_report,
            "used_claude": self.use_claude
        })
        
        return cleaned_df, comprehensive_report
    
    async def _claude_determine_best_mode(
        self,
        df: pd.DataFrame,
        assessment: DataQualityAssessment,
        user_context: Dict[str, Any]
    ) -> str:
        """Use Claude to intelligently determine the best cleaning mode"""
        logger.info("💎 Using Claude to determine optimal cleaning mode...")
        
        data_summary = {
            'shape': df.shape,
            'quality_score': assessment.quality_score,
            'alerts_count': len(assessment.alerts),
            'warnings_count': len(assessment.warnings),
            'missing_ratio': df.isnull().sum().sum() / (df.shape[0] * df.shape[1]),
            'duplicate_ratio': df.duplicated().mean()
        }
        
        critical_issues = [alert["message"] for alert in assessment.alerts[:3]]
        
        prompt = f"""Analyze this data cleaning scenario and recommend the optimal approach.

Dataset Summary:
{json.dumps(data_summary, indent=2)}

User Context:
- Sector: {user_context.get('secteur_activite', 'general')}
- Target: {user_context.get('target_variable', 'unknown')}
- Business Context: {user_context.get('contexte_metier', 'N/A')}

Critical Issues:
{json.dumps(critical_issues, indent=2)}

Available Cleaning Modes:
1. AGENTS: Fully automated using OpenAI agents (fastest, systematic)
2. CONVERSATIONAL: Interactive refinement (slowest, most control)
3. HYBRID: Agents for major issues + conversational for fine-tuning (balanced)

Respond ONLY with valid JSON:
{{
  "recommended_mode": "agents|conversational|hybrid",
  "confidence": 0.0-1.0,
  "reasoning": "2-3 sentence explanation",
  "estimated_time_minutes": number,
  "key_considerations": ["point1", "point2"]
}}"""
        
        try:
            response = await self.claude_client.messages.create(
                model=self.claude_model,
                max_tokens=1000,
                temperature=0.3,
                system="You are an expert data engineer choosing optimal cleaning strategies.",
                messages=[{"role": "user", "content": prompt}]
            )
            
            response_text = response.content[0].text.strip()
            decision = json.loads(response_text)
            
            mode = decision.get('recommended_mode', 'agents')
            logger.info(f"💎 Claude recommends: {mode}")
            logger.info(f"   Confidence: {decision.get('confidence', 0):.1%}")
            
            return mode
            
        except Exception as e:
            logger.warning(f"⚠️ Claude mode selection failed: {e}")
            self.metrics["rule_based_fallbacks"] += 1
            return self._determine_best_mode(assessment, user_context)
    
    async def _claude_generate_cleaning_strategy(
        self,
        df: pd.DataFrame,
        assessment: DataQualityAssessment,
        user_context: Dict[str, Any],
        mode: str
    ) -> Dict[str, Any]:
        """Generate strategic cleaning plan with Claude"""
        logger.info("💎 Generating cleaning strategy with Claude...")
        
        issues_summary = {
            'critical': [alert["message"] for alert in assessment.alerts[:5]],
            'warnings': [warning["message"] for warning in assessment.warnings[:5]],
            'recommendations': assessment.recommendations[:3]
        }
        
        prompt = f"""Create a strategic data cleaning plan for this dataset.

Mode: {mode}
Quality Score: {assessment.quality_score:.1f}/100
Sector: {user_context.get('secteur_activite', 'general')}

Issues Identified:
{json.dumps(issues_summary, indent=2)}

Respond with JSON:
{{
  "summary": "2-3 sentence strategic overview",
  "priorities": [
    {{"issue": "...", "priority": "critical|high|medium", "approach": "...", "impact": "..."}}
  ],
  "risks": [
    {{"risk": "...", "likelihood": "high|medium|low", "mitigation": "..."}}
  ],
  "expected_improvement": {{
    "quality_score_gain": number,
    "key_metrics": {{"metric": value}}
  }},
  "success_criteria": ["criterion1", "criterion2"]
}}"""
        
        try:
            response = await self.claude_client.messages.create(
                model=self.claude_model,
                max_tokens=2000,
                temperature=0.3,
                system="You are an expert data cleaning strategist.",
                messages=[{"role": "user", "content": prompt}]
            )
            
            response_text = response.content[0].text.strip()
            strategy = json.loads(response_text)
            
            logger.info(f"💎 Strategy generated with {len(strategy.get('priorities', []))} priorities")
            
            return strategy
            
        except Exception as e:
            logger.warning(f"⚠️ Claude strategy generation failed: {e}")
            self.metrics["rule_based_fallbacks"] += 1
            return self._generate_basic_strategy(df, assessment, mode)
    
    def _determine_best_mode(
        self, 
        assessment: DataQualityAssessment, 
        user_context: Dict[str, Any]
    ) -> str:
        """Determine the best cleaning mode (RULE-BASED FALLBACK)"""
        quality_score = assessment.quality_score
        has_critical_alerts = len(assessment.alerts) > 0
        sector = user_context.get("secteur_activite")
        
        if sector and (has_critical_alerts or quality_score < 70):
            return "agents"
        if quality_score < 50:
            return "hybrid"
        if quality_score < 80:
            return "agents"
        return "conversational"
    
    def _generate_basic_strategy(
        self,
        df: pd.DataFrame,
        assessment: DataQualityAssessment,
        mode: str
    ) -> Dict[str, Any]:
        """Generate basic strategy (RULE-BASED FALLBACK)"""
        return {
            "summary": f"Rule-based {mode} cleaning for quality {assessment.quality_score:.1f}/100",
            "priorities": [
                {
                    "issue": alert["message"],
                    "priority": "critical",
                    "approach": "Automated correction",
                    "impact": "High"
                }
                for alert in assessment.alerts[:3]
            ],
            "risks": [
                {
                    "risk": "Data loss from aggressive cleaning",
                    "likelihood": "medium",
                    "mitigation": "Backup and validate results"
                }
            ],
            "expected_improvement": {
                "quality_score_gain": 20,
                "key_metrics": {"completeness": 0.9}
            },
            "success_criteria": ["Quality score > 80", "No critical issues"]
        }
    
    async def _clean_with_agents(
        self,
        df: pd.DataFrame,
        user_context: Dict[str, Any],
        assessment: DataQualityAssessment,
        strategy: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, Dict]:
        """Clean using OpenAI agents with strategic guidance"""
        if not self.orchestrator:
            raise ValueError("OpenAI agents not configured")
        
        logger.info("🤖 Cleaning with OpenAI agents")
        
        user_context["quality_assessment"] = {
            "score": assessment.quality_score,
            "critical_issues": [a["message"] for a in assessment.alerts],
            "warnings": [w["message"] for w in assessment.warnings[:5]]
        }
        
        user_context["cleaning_strategy"] = strategy.get("summary", "")
        
        cleaned_df, agent_report = await self.orchestrator.clean_dataset(
            df, 
            user_context
        )
        
        agent_report["strategy_used"] = strategy
        
        return cleaned_df, agent_report
    
    async def _clean_conversationally(
        self,
        df: pd.DataFrame,
        assessment: DataQualityAssessment,
        strategy: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, Dict]:
        """Clean using conversational interface"""
        logger.info("💬 Cleaning conversationally")
        
        cleaned_df = df.copy()
        conversations = []
        
        cleaning_prompts = self._generate_strategic_prompts(strategy, assessment)
        
        for prompt in cleaning_prompts[:5]:
            try:
                cleaned_df, response = await self.quality_agent.clean(prompt, cleaned_df)
                conversations.append({"prompt": prompt, "response": response})
            except Exception as e:
                logger.warning(f"Failed: {prompt}. Error: {e}")
        
        report = {
            "method": "conversational",
            "conversations": conversations,
            "actions_applied": len(conversations),
            "strategy_used": strategy
        }
        
        return cleaned_df, report
    
    async def _clean_hybrid(
        self,
        df: pd.DataFrame,
        user_context: Dict[str, Any],
        assessment: DataQualityAssessment,
        strategy: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, Dict]:
        """Hybrid cleaning approach"""
        logger.info("🔀 Hybrid cleaning")
        
        if assessment.quality_score < 70 and self.orchestrator:
            df, agent_report = await self._clean_with_agents(
                df, user_context, assessment, strategy
            )
        else:
            agent_report = {"skipped": "Quality sufficient"}
        
        mid_assessment = self.quality_agent.assess(
            df, 
            user_context.get("target_variable")
        )
        
        if mid_assessment.quality_score < 90:
            df, conv_report = await self._clean_conversationally(
                df, mid_assessment, strategy
            )
        else:
            conv_report = {"skipped": "Quality sufficient"}
        
        report = {
            "method": "hybrid",
            "agent_phase": agent_report,
            "conversational_phase": conv_report,
            "mid_quality_score": mid_assessment.quality_score,
            "strategy_used": strategy
        }
        
        return df, report
    
    def _generate_strategic_prompts(
        self,
        strategy: Dict[str, Any],
        assessment: DataQualityAssessment
    ) -> List[str]:
        """Generate cleaning prompts from strategic priorities"""
        prompts = []
        
        for priority in strategy.get("priorities", [])[:5]:
            issue = priority.get("issue", "")
            approach = priority.get("approach", "")
            if issue and approach:
                prompts.append(f"{approach} for: {issue}")
        
        if not prompts:
            prompts = self._generate_cleaning_prompts(assessment)
        
        return prompts
    
    def _generate_cleaning_prompts(
        self, 
        assessment: DataQualityAssessment
    ) -> List[str]:
        """Generate cleaning prompts from assessment"""
        prompts = []
        
        for alert in assessment.alerts:
            if "missing" in alert["message"].lower():
                col = self._extract_column_name(alert["message"])
                if col:
                    prompts.append(f"Fill missing values in {col}")
        
        for warning in assessment.warnings[:3]:
            if "outlier" in warning["message"].lower():
                col = self._extract_column_name(warning["message"])
                if col:
                    prompts.append(f"Remove outliers from {col}")
        
        return prompts
    
    def _extract_column_name(self, message: str) -> Optional[str]:
        """Extract a plausible column name from a quality message."""

        import re

        # Match quoted column names first, then bare identifiers while skipping
        # generic words like "mentioned" that aren't column names.
        pattern = re.compile(r"[Cc]olumn\s+(?:['\"](?P<quoted>[^'\"]+)['\"]|(?P<bare>[A-Za-z0-9_]+))")
        stop_words = {"mentioned", "mention", "here", "there", "issues", "issue", "data", "value", "values"}

        for match in pattern.finditer(message):
            column = match.group("quoted") or match.group("bare")
            if column and column.lower() not in stop_words:
                return column

        return None
    
    async def _generate_comprehensive_report(
        self,
        initial_assessment: DataQualityAssessment,
        final_assessment: DataQualityAssessment,
        cleaning_report: Dict,
        mode: str,
        strategy: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate comprehensive report with Claude insights"""
        quality_improvement = final_assessment.quality_score - initial_assessment.quality_score
        
        report = {
            "summary": {
                "mode": mode,
                "initial_quality": initial_assessment.quality_score,
                "final_quality": final_assessment.quality_score,
                "improvement": quality_improvement,
                "success": quality_improvement > 0,
                "used_claude": self.use_claude
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
            },
            "strategy": strategy.get("summary", "Rule-based") if strategy else "Rule-based"
        }
        
        if mode == "agents" and "execution_history" in cleaning_report:
            report["performance"] = {
                "steps_executed": len(cleaning_report.get("execution_history", [])),
                "transformations_applied": len(cleaning_report.get("transformations", []))
            }
        
        if self.use_claude and strategy:
            report["strategic_insights"] = {
                "priorities_addressed": len(strategy.get("priorities", [])),
                "risks_identified": len(strategy.get("risks", [])),
                "expected_vs_actual": {
                    "expected": strategy.get("expected_improvement", {}).get("quality_score_gain", 0),
                    "actual": quality_improvement
                }
            }
        
        if self.use_claude:
            try:
                final_insights = await self._claude_generate_final_insights(
                    report, initial_assessment, final_assessment
                )
                report["claude_insights"] = final_insights
                self.metrics["claude_recommendations"] += 1
            except Exception as e:
                logger.warning(f"⚠️ Claude insights failed: {e}")
                self.metrics["rule_based_fallbacks"] += 1
        
        return report
    
    async def _claude_generate_final_insights(
        self,
        report: Dict[str, Any],
        initial_assessment: DataQualityAssessment,
        final_assessment: DataQualityAssessment
    ) -> str:
        """Generate final insights with Claude"""
        prompt = f"""Analyze this data cleaning execution.

Initial Quality: {initial_assessment.quality_score:.1f}/100
Final Quality: {final_assessment.quality_score:.1f}/100
Improvement: {report['summary']['improvement']:.1f} points

Issues Resolved:
- Alerts: {report['resolved_issues']['alerts']}
- Warnings: {report['resolved_issues']['warnings']}

Provide 3-4 paragraph analysis covering:
1. Overall success assessment
2. Key achievements
3. Remaining concerns
4. Next steps for ML readiness"""
        
        try:
            response = await self.claude_client.messages.create(
                model=self.claude_model,
                max_tokens=1000,
                temperature=0.3,
                system="You are an expert data quality analyst.",
                messages=[{"role": "user", "content": prompt}]
            )
            
            return response.content[0].text.strip()
            
        except Exception as e:
            logger.warning(f"⚠️ Claude insights failed: {e}")
            return "Cleaning completed. Review metrics for details."
    
    def get_cleaning_summary(self) -> Dict[str, Any]:
        """Get summary of all cleaning operations"""
        if not self.cleaning_history:
            return {"message": "No operations yet"}
        
        total = len(self.cleaning_history)
        avg_improvement = sum(h["quality_improvement"] for h in self.cleaning_history) / total
        
        modes_used = {}
        for h in self.cleaning_history:
            mode = h["mode"]
            modes_used[mode] = modes_used.get(mode, 0) + 1
        
        claude_usage = sum(1 for h in self.cleaning_history if h.get("used_claude", False))
        
        return {
            "total_operations": total,
            "average_quality_improvement": avg_improvement,
            "modes_used": modes_used,
            "last_operation": self.cleaning_history[-1]["timestamp"],
            "best_improvement": max(h["quality_improvement"] for h in self.cleaning_history),
            "claude_usage": {
                "operations_with_claude": claude_usage,
                "percentage": (claude_usage / total) * 100 if total > 0 else 0
            },
            "metrics": self.metrics
        }
    
    async def recommend_cleaning_approach(
        self,
        df: pd.DataFrame,
        user_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Recommend the best cleaning approach"""
        assessment = self.quality_agent.assess(df, user_context.get("target_variable"))
        
        if self.use_claude:
            recommended_mode = await self._claude_determine_best_mode(
                df, assessment, user_context
            )
            reasoning = "Claude strategic analysis"
        else:
            recommended_mode = self._determine_best_mode(assessment, user_context)
            reasoning = self._get_mode_reasoning(recommended_mode, assessment)
        
        recommendation = {
            "recommended_mode": recommended_mode,
            "current_quality": assessment.quality_score,
            "reasoning": reasoning,
            "estimated_time": self._estimate_cleaning_time(df, recommended_mode),
            "key_issues": [alert["message"] for alert in assessment.alerts[:3]] + 
                         [warning["message"] for warning in assessment.warnings[:2]],
            "used_claude": self.use_claude
        }
        
        if self.use_claude:
            try:
                strategy_preview = await self._claude_generate_cleaning_strategy(
                    df, assessment, user_context, recommended_mode
                )
                recommendation["strategic_preview"] = strategy_preview.get("summary", "")
            except Exception as e:
                logger.warning(f"⚠️ Preview failed: {e}")
        
        return recommendation
    
    def _get_mode_reasoning(self, mode: str, assessment: DataQualityAssessment) -> str:
        """Get reasoning for mode selection"""
        reasonings = {
            "agents": f"Agents recommended for systematic issues (quality: {assessment.quality_score:.1f}/100)",
            "conversational": f"Conversational for fine-tuning (quality: {assessment.quality_score:.1f}/100)",
            "hybrid": f"Hybrid for complex issues (quality: {assessment.quality_score:.1f}/100)"
        }
        return reasonings.get(mode, "Selected based on data characteristics")
    
    def _estimate_cleaning_time(self, df: pd.DataFrame, mode: str) -> str:
        """Estimate cleaning time"""
        rows, cols = len(df), len(df.columns)
        
        if mode == "agents":
            time_estimate = 30 + (rows / 1000) * 5 + cols * 2
        elif mode == "conversational":
            time_estimate = 20 + (rows / 1000) * 3 + cols * 1
        else:
            time_estimate = 50 + (rows / 1000) * 8 + cols * 3
        
        if time_estimate < 60:
            return f"~{int(time_estimate)} seconds"
        elif time_estimate < 3600:
            return f"~{int(time_estimate / 60)} minutes"
        else:
            return f"~{time_estimate / 3600:.1f} hours"
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get usage metrics"""
        return self.metrics.copy()


# Convenience function
async def smart_clean_data(
    df: pd.DataFrame,
    user_context: Dict[str, Any],
    config: Optional[AgentConfig] = None,
    mode: str = "auto",
    use_claude: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Convenience function for smart data cleaning
    
    Args:
        df: DataFrame to clean
        user_context: Context (sector, target, etc.)
        config: Optional configuration
        mode: Cleaning mode
        use_claude: Use Claude SDK for insights
    
    Returns:
        Tuple of (cleaned_df, report)
    """
    cleaner = IntelligentDataCleaner(config, use_claude=use_claude)
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
        # Create intelligent cleaner with Claude
        cleaner = IntelligentDataCleaner(use_claude=True)
        
        # Get recommendation
        recommendation = await cleaner.recommend_cleaning_approach(sample_df, context)
        print(f"💡 Recommended: {recommendation['recommended_mode']}")
        print(f"📊 Reasoning: {recommendation['reasoning']}")
        
        # Clean data
        cleaned_df, report = await cleaner.smart_clean(sample_df, context, mode="auto")
        
        print(f"\n✅ Improvement: {report['summary']['improvement']:.1f} points")
        print(f"📈 Final quality: {report['summary']['final_quality']:.1f}/100")
        
        if report.get('claude_insights'):
            print(f"\n💎 Claude Insights:\n{report['claude_insights']}")
        
    # Run demo
    asyncio.run(demo())
