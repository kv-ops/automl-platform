"""
Adaptive Template System for AutoML Platform
============================================
Templates become optional hints that the agent can override.
The system learns from successful executions.

ARCHITECTURE:
- Rule-based core (proven, fast, predictable)
- Optional Claude enhancement for semantic adaptation
- Graceful degradation if Claude unavailable
"""

import logging
import json
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
import pandas as pd
import numpy as np
import importlib.util

# Optional Claude SDK import
_anthropic_spec = importlib.util.find_spec("anthropic")
if _anthropic_spec is not None:
    from anthropic import AsyncAnthropic
else:
    AsyncAnthropic = None

logger = logging.getLogger(__name__)


@dataclass
class AdaptiveTemplate:
    """An adaptive template that can be overridden"""
    name: str
    base_config: Dict[str, Any]
    learned_adaptations: List[Dict[str, Any]] = field(default_factory=list)
    usage_count: int = 0
    success_rate: float = 0.0
    last_used: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        if self.last_used:
            data['last_used'] = self.last_used.isoformat()
        data['created_at'] = self.created_at.isoformat()
        return data


class AdaptiveTemplateSystem:
    """
    Templates are no longer rigid configurations but suggestions.
    The agent can override them based on data and context.
    
    NEW: Optional Claude enhancement for semantic pattern matching
    """
    
    def __init__(
        self, 
        template_dir: Optional[Path] = None,
        use_claude: bool = False,
        anthropic_api_key: Optional[str] = None
    ):
        """
        Initialize the adaptive template system
        
        Args:
            template_dir: Directory containing template files
            use_claude: Whether to use Claude for semantic enhancement
            anthropic_api_key: Optional Anthropic API key
        """
        self.template_dir = template_dir or Path("./templates/adaptive")
        self.template_dir.mkdir(parents=True, exist_ok=True)
        
        # Storage for templates and learned patterns
        self.templates: Dict[str, AdaptiveTemplate] = {}
        self.learned_patterns: Dict[str, List[Dict[str, Any]]] = {}
        
        # Learning metrics
        self.adaptation_history = []
        self.performance_by_template = {}
        
        # Claude enhancement (optional)
        self.use_claude = use_claude and AsyncAnthropic is not None
        if self.use_claude:
            self.claude_client = AsyncAnthropic(api_key=anthropic_api_key)
            self.claude_model = "claude-sonnet-4-20250514"
            logger.info("💎 Claude enhancement enabled for semantic adaptation")
        else:
            self.claude_client = None
            if use_claude:
                logger.warning("⚠️ Claude requested but unavailable, using rule-based only")
            else:
                logger.info("📋 Using pure rule-based adaptation (optimal)")
        
        # Metrics tracking
        self.metrics = {
            "total_adaptations": 0,
            "rule_based_adaptations": 0,
            "claude_enhanced_adaptations": 0,
            "claude_fallbacks": 0
        }
        
        # Load existing templates
        self._load_templates()
        
        # Load learned patterns
        self._load_learned_patterns()
    
    async def get_configuration(
        self,
        df: pd.DataFrame,
        context: Dict[str, Any],
        agent_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Get configuration with intelligent adaptation.
        
        The template is just a starting point - the agent adapts it.
        Claude enhancement is optional and transparent.
        """
        self.metrics["total_adaptations"] += 1
        problem_type = context.get('problem_type', 'unknown')
        
        logger.info(f"🎯 Getting adaptive configuration for {problem_type}")
        
        base_config = None
        
        # Step 1: Check learned patterns first (with optional Claude enhancement)
        if problem_type in self.learned_patterns:
            logger.info(f"📚 Found {len(self.learned_patterns[problem_type])} learned patterns")
            
            if self.use_claude:
                # Try semantic matching with Claude
                base_config = await self._claude_select_best_pattern(problem_type, context)
                if base_config:
                    logger.info("💎 Using Claude-selected pattern")
                    self.metrics["claude_enhanced_adaptations"] += 1
            
            # Fallback to rule-based if Claude didn't work
            if not base_config:
                base_config = self._select_best_learned_pattern(problem_type, context)
                if base_config:
                    logger.info("✅ Using rule-based pattern selection")
                    self.metrics["rule_based_adaptations"] += 1
        
        # Step 2: Fall back to template if exists
        if not base_config and problem_type in self.templates:
            logger.info(f"📋 Using template '{problem_type}' as hint")
            template = self.templates[problem_type]
            base_config = template.base_config.copy()
            template.usage_count += 1
            template.last_used = datetime.now()
            self.metrics["rule_based_adaptations"] += 1
        
        # Step 3: Generate from scratch if no template
        if not base_config:
            logger.info("🆕 No template found - generating from scratch")
            base_config = await self._generate_adaptive_config(df, context)
            self.metrics["rule_based_adaptations"] += 1
        
        # Step 4: ALWAYS adapt to current data and context
        adapted_config = await self._adapt_to_current_data(base_config, df, context)
        
        # Step 5: Apply agent overrides if provided
        if agent_config:
            logger.info("🤖 Applying agent overrides")
            adapted_config = self._apply_agent_overrides(adapted_config, agent_config)
        
        # Step 6: Learn from this adaptation
        self._record_adaptation(problem_type, base_config, adapted_config, context)
        
        return adapted_config
    
    async def _claude_select_best_pattern(
        self,
        problem_type: str,
        context: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Use Claude for semantic pattern matching (OPTIONAL ENHANCEMENT)
        
        This provides better understanding of contextual similarity
        beyond simple numerical comparisons.
        """
        if not self.use_claude:
            return None
        
        patterns = self.learned_patterns.get(problem_type, [])
        if not patterns:
            return None
        
        try:
            # Prepare patterns for Claude
            patterns_summary = []
            for i, pattern in enumerate(patterns[:5]):  # Top 5 to save tokens
                patterns_summary.append({
                    'id': i,
                    'context': pattern.get('context', {}),
                    'performance': pattern.get('success_score', 0),
                    'config_summary': self._summarize_config(pattern.get('config', {}))
                })
            
            prompt = f"""Analyze which learned pattern best matches the current ML context.

Current Context:
{json.dumps(context, indent=2)}

Available Patterns:
{json.dumps(patterns_summary, indent=2)}

Task: Determine semantic similarity considering:
1. Problem complexity and scale
2. Business domain alignment
3. Data characteristics similarity
4. Historical performance

Respond ONLY with JSON:
{{
  "best_pattern_id": 0-4 or null,
  "confidence": 0.0-1.0,
  "reasoning": "brief explanation",
  "semantic_similarity": 0.0-1.0
}}"""
            
            response = await self.claude_client.messages.create(
                model=self.claude_model,
                max_tokens=800,
                temperature=0.3,
                system="You are an expert at matching ML problem patterns based on semantic similarity.",
                messages=[{"role": "user", "content": prompt}]
            )
            
            response_text = response.content[0].text.strip()
            result = json.loads(response_text)
            
            best_id = result.get('best_pattern_id')
            confidence = result.get('confidence', 0)
            similarity = result.get('semantic_similarity', 0)
            
            # Use Claude's selection if confident
            if best_id is not None and confidence > 0.7 and similarity > 0.7:
                logger.info(f"💎 Claude selected pattern {best_id} "
                          f"(confidence: {confidence:.2f}, similarity: {similarity:.2f})")
                logger.info(f"   Reasoning: {result.get('reasoning', 'N/A')[:100]}")
                return patterns[best_id]['config']
            else:
                logger.info(f"💎 Claude confidence too low ({confidence:.2f}), "
                          f"falling back to rule-based")
                self.metrics["claude_fallbacks"] += 1
                return None
                
        except Exception as e:
            logger.warning(f"⚠️ Claude pattern selection failed: {e}, using rule-based")
            self.metrics["claude_fallbacks"] += 1
            return None
    
    def _summarize_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create concise summary of config for Claude"""
        return {
            'task': config.get('task', 'unknown'),
            'algorithms': config.get('algorithms', [])[:3],  # First 3
            'has_preprocessing': bool(config.get('preprocessing')),
            'has_feature_engineering': bool(config.get('feature_engineering')),
            'primary_metric': config.get('primary_metric', 'unknown')
        }
    
    def _select_best_learned_pattern(
        self,
        problem_type: str,
        context: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Select the best learned pattern for the context (RULE-BASED)
        
        This is the proven, reliable method that always works.
        """
        patterns = self.learned_patterns.get(problem_type, [])
        
        if not patterns:
            return None
        
        # Score patterns based on similarity to current context
        scored_patterns = []
        for pattern in patterns:
            score = self._calculate_pattern_similarity(pattern, context)
            scored_patterns.append((score, pattern))
        
        # Sort by score and get best
        scored_patterns.sort(key=lambda x: x[0], reverse=True)
        
        if scored_patterns and scored_patterns[0][0] > 0.7:  # Similarity threshold
            best_pattern = scored_patterns[0][1]
            logger.info(f"Found similar pattern with score {scored_patterns[0][0]:.2f}")
            return best_pattern['config']
        
        return None
    
    def _calculate_pattern_similarity(
        self,
        pattern: Dict[str, Any],
        context: Dict[str, Any]
    ) -> float:
        """
        Calculate similarity between a learned pattern and current context (RULE-BASED)
        
        This is fast, predictable, and well-tested.
        """
        score = 0.0
        weights = {
            'data_size': 0.2,
            'n_features': 0.2,
            'imbalance': 0.2,
            'temporal': 0.1,
            'sector': 0.3
        }
        
        pattern_ctx = pattern.get('context', {})
        
        # Compare data size
        if 'n_samples' in pattern_ctx and 'n_samples' in context:
            size_ratio = min(pattern_ctx['n_samples'], context['n_samples']) / \
                        max(pattern_ctx['n_samples'], context['n_samples'])
            score += weights['data_size'] * size_ratio
        
        # Compare number of features
        if 'n_features' in pattern_ctx and 'n_features' in context:
            feature_ratio = min(pattern_ctx['n_features'], context['n_features']) / \
                           max(pattern_ctx['n_features'], context['n_features'])
            score += weights['n_features'] * feature_ratio
        
        # Compare imbalance
        if 'imbalance_detected' in pattern_ctx and 'imbalance_detected' in context:
            if pattern_ctx['imbalance_detected'] == context['imbalance_detected']:
                score += weights['imbalance']
        
        # Compare temporal aspect
        if 'temporal_aspect' in pattern_ctx and 'temporal_aspect' in context:
            if pattern_ctx['temporal_aspect'] == context['temporal_aspect']:
                score += weights['temporal']
        
        # Compare business sector
        if 'business_sector' in pattern_ctx and 'business_sector' in context:
            if pattern_ctx['business_sector'] == context['business_sector']:
                score += weights['sector']
        
        return score
    
    async def _generate_adaptive_config(
        self,
        df: pd.DataFrame,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate configuration from scratch adaptively (RULE-BASED)"""
        
        config = {
            'task': 'auto',
            'algorithms': [],
            'preprocessing': {},
            'hpo': {},
            'monitoring': {}
        }
        
        problem_type = context.get('problem_type', 'unknown')
        
        # Task detection
        if problem_type in ['churn_prediction', 'fraud_detection']:
            config['task'] = 'classification'
        elif problem_type in ['sales_forecasting', 'demand_prediction']:
            config['task'] = 'regression'
        elif problem_type == 'customer_segmentation':
            config['task'] = 'clustering'
        
        # Algorithm selection based on data size
        n_samples = len(df)
        if n_samples < 1000:
            config['algorithms'] = ['RandomForest', 'XGBoost', 'LogisticRegression']
        elif n_samples < 100000:
            config['algorithms'] = ['XGBoost', 'LightGBM', 'CatBoost', 'RandomForest']
        else:
            config['algorithms'] = ['LightGBM', 'XGBoost', 'NeuralNetwork']
        
        # Preprocessing based on data quality
        missing_ratio = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
        if missing_ratio > 0.1:
            config['preprocessing']['handle_missing'] = {
                'strategy': 'iterative_impute'
            }
        else:
            config['preprocessing']['handle_missing'] = {
                'strategy': 'simple_impute'
            }
        
        # HPO based on time budget
        config['hpo'] = {
            'method': 'optuna' if n_samples > 5000 else 'random',
            'n_iter': min(50, max(10, n_samples // 1000))
        }
        
        return config
    
    async def _adapt_to_current_data(
        self,
        base_config: Dict[str, Any],
        df: pd.DataFrame,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Adapt configuration to current data characteristics (RULE-BASED)
        This is where the intelligence happens - override template decisions.
        
        Rule-based is optimal here: fast, predictable, and well-tested.
        """
        adapted = base_config.copy()
        
        logger.info("🔄 Adapting configuration to current data...")
        
        # Adapt algorithms based on actual data
        n_samples = len(df)
        n_features = len(df.columns)
        
        # Override algorithm selection if needed
        if n_samples < 500 and 'NeuralNetwork' in adapted.get('algorithms', []):
            logger.info("⚠️ Removing NeuralNetwork - too few samples")
            adapted['algorithms'] = [a for a in adapted['algorithms'] if a != 'NeuralNetwork']
            if 'LogisticRegression' not in adapted['algorithms']:
                adapted['algorithms'].append('LogisticRegression')
        
        # Adapt preprocessing based on actual missing values
        missing_ratio = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
        if missing_ratio > 0.3:
            logger.info(f"⚠️ High missing ratio ({missing_ratio:.1%}) - adapting strategy")
            if 'preprocessing' not in adapted:
                adapted['preprocessing'] = {}
            adapted['preprocessing']['handle_missing'] = {
                'strategy': 'drop_columns',
                'threshold': 0.5
            }
        
        # Adapt for imbalance
        if context.get('imbalance_detected'):
            logger.info("⚖️ Imbalance detected - adding SMOTE")
            adapted['preprocessing']['handle_imbalance'] = {
                'method': 'SMOTE',
                'sampling_strategy': 'auto'
            }
            # Also adapt metrics
            adapted['primary_metric'] = 'f1'
        
        # Adapt for temporal data
        if context.get('temporal_aspect'):
            logger.info("📅 Temporal data detected - adding time features")
            adapted['preprocessing']['create_time_features'] = True
            adapted['preprocessing']['lag_features'] = [1, 7, 30]
        
        # Adapt HPO iterations based on time and data
        if n_samples > 100000:
            logger.info("📊 Large dataset - reducing HPO iterations")
            if 'hpo' in adapted:
                adapted['hpo']['n_iter'] = min(adapted['hpo'].get('n_iter', 50), 20)
        
        # Add monitoring for critical applications
        if context.get('problem_type') in ['fraud_detection', 'credit_scoring']:
            logger.info("🚨 Critical application - enabling real-time monitoring")
            adapted['monitoring'] = {
                'real_time': True,
                'drift_detection': True,
                'alert_threshold': 0.05
            }
        
        return adapted
    
    def _apply_agent_overrides(
        self,
        config: Dict[str, Any],
        agent_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply specific overrides from the agent"""
        
        # Agent decisions have highest priority
        for key, value in agent_config.items():
            if value is not None:  # Only override if explicitly set
                logger.info(f"🤖 Agent override: {key} = {value}")
                config[key] = value
        
        return config
    
    def _record_adaptation(
        self,
        problem_type: str,
        base_config: Dict[str, Any],
        adapted_config: Dict[str, Any],
        context: Dict[str, Any]
    ):
        """Record the adaptation for learning"""
        adaptation = {
            'timestamp': datetime.now().isoformat(),
            'problem_type': problem_type,
            'base_config': base_config,
            'adapted_config': adapted_config,
            'context': context,
            'changes': self._compute_config_diff(base_config, adapted_config),
            'used_claude': self.metrics["claude_enhanced_adaptations"] > 0
        }
        
        self.adaptation_history.append(adaptation)
        
        # Keep only recent history
        self.adaptation_history = self.adaptation_history[-1000:]
    
    def _compute_config_diff(
        self,
        base: Dict[str, Any],
        adapted: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compute the difference between base and adapted config"""
        diff = {}
        
        # Check for added keys
        for key in adapted:
            if key not in base:
                diff[f"added_{key}"] = adapted[key]
            elif adapted[key] != base.get(key):
                diff[f"modified_{key}"] = {
                    'from': base[key],
                    'to': adapted[key]
                }
        
        # Check for removed keys
        for key in base:
            if key not in adapted:
                diff[f"removed_{key}"] = base[key]
        
        return diff
    
    def learn_from_execution(
        self,
        context: Dict[str, Any],
        config: Dict[str, Any],
        performance: Dict[str, float]
    ):
        """
        Learn from execution results.
        Store successful patterns for future use.
        """
        problem_type = context.get('problem_type', 'unknown')
        
        # Calculate success score
        success_score = max(performance.values()) if performance else 0.0
        
        logger.info(f"📈 Learning from execution: {problem_type} -> score: {success_score:.3f}")
        
        # Store if successful
        if success_score > 0.8:  # Success threshold
            pattern = {
                'timestamp': datetime.now().isoformat(),
                'context': context,
                'config': config,
                'performance': performance,
                'success_score': success_score
            }
            
            if problem_type not in self.learned_patterns:
                self.learned_patterns[problem_type] = []
            
            self.learned_patterns[problem_type].append(pattern)
            
            # Keep only best patterns (top 10)
            self.learned_patterns[problem_type].sort(
                key=lambda x: x['success_score'],
                reverse=True
            )
            self.learned_patterns[problem_type] = self.learned_patterns[problem_type][:10]
            
            logger.info(f"✅ Stored successful pattern for {problem_type}")
            
            # Save to disk
            self._save_learned_patterns()
        
        # Update template performance if one was used
        if problem_type in self.templates:
            if problem_type not in self.performance_by_template:
                self.performance_by_template[problem_type] = []
            
            self.performance_by_template[problem_type].append(success_score)
            
            # Update template success rate
            template = self.templates[problem_type]
            scores = self.performance_by_template[problem_type][-20:]  # Last 20
            template.success_rate = sum(scores) / len(scores)
    
    def add_template(
        self,
        name: str,
        config: Dict[str, Any],
        description: Optional[str] = None
    ) -> AdaptiveTemplate:
        """Add a new template to the system"""
        template = AdaptiveTemplate(
            name=name,
            base_config=config
        )
        
        self.templates[name] = template
        self._save_template(template)
        
        logger.info(f"Added new template: {name}")
        
        return template
    
    def get_template_stats(self) -> Dict[str, Any]:
        """Get statistics about template usage and performance"""
        stats = {
            'total_templates': len(self.templates),
            'total_learned_patterns': sum(len(p) for p in self.learned_patterns.values()),
            'total_adaptations': len(self.adaptation_history),
            'template_performance': {},
            'claude_metrics': self.metrics.copy() if self.use_claude else None
        }
        
        for name, template in self.templates.items():
            stats['template_performance'][name] = {
                'usage_count': template.usage_count,
                'success_rate': template.success_rate,
                'last_used': template.last_used.isoformat() if template.last_used else None
            }
        
        # Add Claude effectiveness metrics if enabled
        if self.use_claude and self.metrics["total_adaptations"] > 0:
            stats['claude_effectiveness'] = {
                'enhancement_rate': self.metrics["claude_enhanced_adaptations"] / self.metrics["total_adaptations"],
                'fallback_rate': self.metrics["claude_fallbacks"] / max(1, self.metrics["claude_enhanced_adaptations"] + self.metrics["claude_fallbacks"])
            }
        
        return stats
    
    def _load_templates(self):
        """Load templates from disk"""
        template_files = self.template_dir.glob("*.yaml")
        
        for file in template_files:
            try:
                with open(file, 'r') as f:
                    data = yaml.safe_load(f)
                    template = AdaptiveTemplate(
                        name=file.stem,
                        base_config=data.get('config', data)
                    )
                    self.templates[file.stem] = template
                    logger.info(f"Loaded template: {file.stem}")
            except Exception as e:
                logger.warning(f"Failed to load template {file}: {e}")
    
    def _save_template(self, template: AdaptiveTemplate):
        """Save template to disk"""
        file_path = self.template_dir / f"{template.name}.yaml"
        try:
            with open(file_path, 'w') as f:
                yaml.dump(template.to_dict(), f)
        except Exception as e:
            logger.warning(f"Failed to save template: {e}")
    
    def _load_learned_patterns(self):
        """Load learned patterns from disk"""
        patterns_file = self.template_dir / "learned_patterns.json"
        if patterns_file.exists():
            try:
                with open(patterns_file, 'r') as f:
                    self.learned_patterns = json.load(f)
                logger.info(f"Loaded {sum(len(p) for p in self.learned_patterns.values())} learned patterns")
            except Exception as e:
                logger.warning(f"Failed to load learned patterns: {e}")
    
    def _save_learned_patterns(self):
        """Save learned patterns to disk"""
        patterns_file = self.template_dir / "learned_patterns.json"
        try:
            with open(patterns_file, 'w') as f:
                json.dump(self.learned_patterns, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save learned patterns: {e}")
