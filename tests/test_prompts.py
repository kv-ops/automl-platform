"""
Tests for prompts module
========================
Tests for prompt templates and optimization.
"""

import pytest
import json
from unittest.mock import Mock, patch
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from automl_platform.prompts import (
    PromptTemplates,
    PromptOptimizer
)


class TestPromptTemplates:
    """Tests for PromptTemplates class"""
    
    def test_get_prompt_data_quality(self):
        """Test data quality analysis prompt generation"""
        prompt = PromptTemplates.get_prompt(
            'DATA_QUALITY_ANALYSIS',
            dataset_summary='1000 rows, 20 columns, 5% missing values'
        )
        
        assert 'You are an expert data scientist' in prompt
        assert '1000 rows, 20 columns, 5% missing values' in prompt
        assert 'Data Quality Issues' in prompt
        assert 'Statistical Anomalies' in prompt
        assert 'Recommended Actions' in prompt
        assert 'Risk Assessment' in prompt
        assert 'Format your response as structured JSON' in prompt
    
    def test_get_prompt_column_analysis(self):
        """Test column analysis prompt"""
        prompt = PromptTemplates.get_prompt(
            'COLUMN_ANALYSIS',
            columns_info='[col1: numeric, col2: categorical]',
            target_info='binary classification target',
            task_type='classification'
        )
        
        assert 'Analyze these columns' in prompt
        assert '[col1: numeric, col2: categorical]' in prompt
        assert 'binary classification target' in prompt
        assert 'classification' in prompt
        assert 'Data quality' in prompt
        assert 'Relationship with target' in prompt
        assert 'Transformation suggestions' in prompt
    
    def test_get_prompt_feature_generation(self):
        """Test feature generation prompt"""
        prompt = PromptTemplates.get_prompt(
            'FEATURE_GENERATION',
            dataset_info='Sales data with 10 features',
            target_column='revenue',
            task_type='regression',
            domain='e-commerce'
        )
        
        assert 'feature engineering expert' in prompt
        assert 'Sales data with 10 features' in prompt
        assert 'revenue' in prompt
        assert 'regression' in prompt
        assert 'e-commerce' in prompt
        assert 'Statistical Features' in prompt
        assert 'Interaction Features' in prompt
        assert 'Domain-Specific Features' in prompt
        assert 'Advanced Transformations' in prompt
        assert 'Python code using pandas/numpy' in prompt
    
    def test_get_prompt_auto_feature_code(self):
        """Test auto feature code generation prompt"""
        prompt = PromptTemplates.get_prompt(
            'AUTO_FEATURE_CODE',
            feature_tasks='Create ratio features, log transformations'
        )
        
        assert 'Generate Python code' in prompt
        assert 'Create ratio features, log transformations' in prompt
        assert 'pandas and numpy' in prompt
        assert 'Handle edge cases' in prompt
        assert 'def engineer_features(df):' in prompt
    
    def test_get_prompt_model_recommendation(self):
        """Test model recommendation prompt"""
        prompt = PromptTemplates.get_prompt(
            'MODEL_RECOMMENDATION',
            n_samples=10000,
            n_features=50,
            task_type='classification',
            feature_types='numeric and categorical',
            class_balance='imbalanced',
            missing_ratio=10,
            memory_size='100MB',
            max_time=3600,
            interpretability=True,
            environment='cloud'
        )
        
        assert 'Recommend ML models' in prompt
        assert '10000' in str(prompt)
        assert 'classification' in prompt
        assert 'imbalanced' in prompt
        assert 'interpretability' in str(prompt)
        assert 'cloud' in prompt
        assert 'Recommend 5-7 models' in prompt
        assert 'hyperparameters' in prompt
    
    def test_get_prompt_hyperparameter_suggestion(self):
        """Test hyperparameter suggestion prompt"""
        prompt = PromptTemplates.get_prompt(
            'HYPERPARAMETER_SUGGESTION',
            model_name='XGBoost',
            dataset_properties={'n_samples': 5000, 'n_features': 20},
            current_metrics={'accuracy': 0.85, 'f1': 0.83}
        )
        
        assert 'XGBoost' in prompt
        assert 'Critical Hyperparameters' in prompt
        assert 'Search Strategy' in prompt
        assert 'Format as JSON' in prompt
    
    def test_get_prompt_model_interpretation(self):
        """Test model interpretation prompt"""
        prompt = PromptTemplates.get_prompt(
            'MODEL_INTERPRETATION',
            model_type='RandomForest',
            metrics={'accuracy': 0.92, 'precision': 0.90},
            feature_importance={'feature1': 0.3, 'feature2': 0.2},
            confusion_matrix=[[50, 10], [5, 35]]
        )
        
        assert 'RandomForest' in prompt
        assert 'Executive Summary' in prompt
        assert 'Performance Analysis' in prompt
        assert 'Feature Insights' in prompt
        assert 'non-technical language' in prompt
    
    def test_get_prompt_shap_explanation(self):
        """Test SHAP explanation prompt"""
        prompt = PromptTemplates.get_prompt(
            'SHAP_EXPLANATION',
            stakeholder_type='business executive',
            shap_values={'feature1': 0.5, 'feature2': -0.3},
            prediction=1,
            actual=1
        )
        
        assert 'business executive' in prompt
        assert 'SHAP values' in prompt
        assert 'What drove this specific prediction' in prompt
        assert 'analogies and simple language' in prompt
    
    def test_get_prompt_error_analysis(self):
        """Test error pattern analysis prompt"""
        prompt = PromptTemplates.get_prompt(
            'ERROR_PATTERN_ANALYSIS',
            error_stats={'mae': 0.15, 'rmse': 0.25},
            worst_cases=[{'id': 1, 'error': 2.5}],
            error_features={'feature1': [1, 2, 3]}
        )
        
        assert 'Analyze these prediction errors' in prompt
        assert 'Error Patterns' in prompt
        assert 'Root Causes' in prompt
        assert 'Improvement Strategies' in prompt
        assert 'Priority Actions' in prompt
    
    def test_get_prompt_drift_analysis(self):
        """Test drift analysis prompt"""
        prompt = PromptTemplates.get_prompt(
            'DRIFT_ANALYSIS',
            reference_stats={'mean': 10, 'std': 2},
            current_stats={'mean': 12, 'std': 3},
            drift_metrics={'psi': 0.15},
            time_period='last 30 days',
            urgency_level='high'
        )
        
        assert 'data drift report' in prompt
        assert 'Drift Severity' in prompt
        assert 'Business Impact' in prompt
        assert 'Root Causes' in prompt
        assert 'high' in prompt
    
    def test_get_prompt_executive_report(self):
        """Test executive report generation prompt"""
        prompt = PromptTemplates.get_prompt(
            'EXECUTIVE_REPORT',
            experiment_summary='Classification model achieving 95% accuracy',
            business_context='Customer churn prediction',
            results={'accuracy': 0.95, 'roi_estimate': '2.5x'},
            output_format='markdown'
        )
        
        assert 'executive report' in prompt
        assert 'Customer churn prediction' in prompt
        assert 'Executive Summary' in prompt
        assert 'ROI/Impact estimate' in prompt
        assert 'markdown' in prompt
        assert 'Professional, confident, action-oriented' in prompt
    
    def test_get_prompt_technical_report(self):
        """Test technical report generation prompt"""
        prompt = PromptTemplates.get_prompt(
            'TECHNICAL_REPORT',
            experiment_details={'model': 'XGBoost', 'cv_folds': 5}
        )
        
        assert 'detailed technical report' in prompt
        assert 'Data Analysis' in prompt
        assert 'Model Development' in prompt
        assert 'Statistical tests' in prompt
        assert 'Reproducibility' in prompt
        assert 'Code snippets' in prompt
    
    def test_get_prompt_data_cleaning_dialogue(self):
        """Test data cleaning dialogue prompt"""
        prompt = PromptTemplates.get_prompt(
            'DATA_CLEANING_DIALOGUE',
            data_state='DataFrame with 1000 rows, 20 columns',
            user_request='Remove outliers from price column',
            action_history=['Removed duplicates', 'Filled missing values']
        )
        
        assert 'data cleaning assistant' in prompt
        assert 'Akkio\'s GPT-4 agent' in prompt
        assert 'Remove outliers from price column' in prompt
        assert 'Understanding Confirmation' in prompt
        assert 'Proposed Actions' in prompt
        assert 'Code Generation' in prompt
        assert 'Impact Preview' in prompt
    
    def test_get_prompt_question_answering(self):
        """Test question answering prompt"""
        prompt = PromptTemplates.get_prompt(
            'QUESTION_ANSWERING',
            question='Which model performed best?',
            experiment_context='Binary classification on customer data',
            models=['RF', 'XGBoost', 'LogisticRegression'],
            best_metrics={'model': 'XGBoost', 'accuracy': 0.95},
            stage='evaluation'
        )
        
        assert 'Which model performed best?' in prompt
        assert 'Binary classification' in prompt
        assert 'XGBoost' in prompt
        assert 'Direct answer to the question' in prompt
        assert 'Supporting evidence' in prompt
    
    def test_get_prompt_automl_code_generation(self):
        """Test AutoML code generation prompt"""
        prompt = PromptTemplates.get_prompt(
            'AUTOML_CODE_GENERATION',
            task_description='Binary classification for fraud detection',
            data_format='CSV with 50 features',
            requirements='High precision, explainable',
            constraints='Real-time inference needed',
            framework='scikit-learn'
        )
        
        assert 'Generate production-ready AutoML code' in prompt
        assert 'fraud detection' in prompt
        assert 'CSV with 50 features' in prompt
        assert 'High precision' in prompt
        assert 'Real-time inference' in prompt
        assert 'scikit-learn' in prompt
        assert 'Type hints' in prompt
        assert 'Error handling' in prompt
        assert 'Unit test examples' in prompt
    
    def test_get_prompt_custom_transformer(self):
        """Test custom transformer generation prompt"""
        prompt = PromptTemplates.get_prompt(
            'CUSTOM_TRANSFORMER',
            transformation_name='RatioFeatures',
            transformation_description='Create ratio features from numeric columns',
            data_example={'col1': [1, 2], 'col2': [4, 8]},
            expected_output={'ratio_col1_col2': [0.25, 0.25]}
        )
        
        assert 'custom scikit-learn transformer' in prompt
        assert 'RatioFeatures' in prompt
        assert 'BaseEstimator, TransformerMixin' in prompt
        assert 'fit' in prompt
        assert 'transform' in prompt
        assert 'get_feature_names_out' in prompt
    
    def test_get_prompt_missing_parameter(self):
        """Test error when required parameter is missing"""
        with pytest.raises(ValueError, match='Missing required parameter'):
            PromptTemplates.get_prompt(
                'DATA_QUALITY_ANALYSIS'
                # Missing dataset_summary parameter
            )
    
    def test_get_prompt_invalid_name(self):
        """Test error for invalid prompt name"""
        with pytest.raises(ValueError, match="Prompt 'INVALID_PROMPT' not found"):
            PromptTemplates.get_prompt('INVALID_PROMPT')
    
    def test_list_prompts(self):
        """Test listing available prompts"""
        prompts = PromptTemplates.list_prompts()
        
        assert isinstance(prompts, list)
        assert len(prompts) > 10  # Should have many prompts
        
        # Check key prompts are present
        expected_prompts = [
            'DATA_QUALITY_ANALYSIS',
            'COLUMN_ANALYSIS',
            'FEATURE_GENERATION',
            'AUTO_FEATURE_CODE',
            'MODEL_RECOMMENDATION',
            'HYPERPARAMETER_SUGGESTION',
            'MODEL_INTERPRETATION',
            'SHAP_EXPLANATION',
            'ERROR_PATTERN_ANALYSIS',
            'DRIFT_ANALYSIS',
            'EXECUTIVE_REPORT',
            'TECHNICAL_REPORT',
            'DATA_CLEANING_DIALOGUE',
            'QUESTION_ANSWERING',
            'AUTOML_CODE_GENERATION',
            'CUSTOM_TRANSFORMER'
        ]
        
        for prompt in expected_prompts:
            assert prompt in prompts


class TestPromptOptimizer:
    """Tests for PromptOptimizer class"""
    
    def test_optimize_for_gpt35(self):
        """Test optimization for GPT-3.5"""
        original_prompt = "Provide a detailed analysis. For each item, include statistics. Include comprehensive results."
        
        optimized = PromptOptimizer.optimize_for_model(
            original_prompt,
            model='gpt-3.5-turbo',
            max_tokens=1000
        )
        
        # Should add structure markers
        assert '**' in optimized or 'Format' in optimized
        # Check for structure improvements
        assert len(optimized.split('\n')) >= len(original_prompt.split('\n'))
    
    def test_optimize_for_gpt4(self):
        """Test optimization for GPT-4"""
        original_prompt = "Analyze this problem and provide a solution."
        
        optimized = PromptOptimizer.optimize_for_model(
            original_prompt,
            model='gpt-4',
            max_tokens=2000
        )
        
        # Should add reasoning chain
        assert 'step' in optimized.lower()
        assert len(optimized) > len(original_prompt)
    
    def test_optimize_for_claude(self):
        """Test optimization for Claude"""
        original_prompt = "Generate a solution for this task."
        
        optimized = PromptOptimizer.optimize_for_model(
            original_prompt,
            model='claude-3',
            max_tokens=4000
        )
        
        # Should add detailed context
        assert len(optimized) > len(original_prompt)
        assert 'comprehensive' in optimized or 'aspects' in optimized or 'balance' in optimized.lower()
    
    def test_compress_prompt_long(self):
        """Test prompt compression for token limits"""
        long_prompt = """
        Provide a detailed analysis including but not limited to the following aspects.
        For example, you should consider various factors such as performance metrics.
        
        
        Extra whitespace here.
        
        
        More content including but not limited to additional analysis.
        """
        
        # Small token limit should trigger compression
        optimized = PromptOptimizer.optimize_for_model(
            long_prompt,
            model='gpt-3.5-turbo',
            max_tokens=50  # Very small limit to force compression
        )
        
        # Should be compressed
        assert len(optimized) <= len(long_prompt)
        assert ('E.g.' in optimized or 'e.g.' in optimized) or 'including' in optimized
        assert 'including but not limited to' not in optimized or optimized.count('including but not limited to') < long_prompt.count('including but not limited to')
        # Extra whitespace should be removed
        assert '\n\n\n' not in optimized
    
    def test_add_structure_markers(self):
        """Test adding structure markers"""
        prompt = "Provide analysis. For each item, describe it. Include results."
        
        result = PromptOptimizer._add_structure_markers(prompt)
        
        # Should have formatting improvements
        assert '**' in result or 'Format' in result or result != prompt
    
    def test_add_reasoning_chain(self):
        """Test adding chain-of-thought reasoning"""
        prompt = "Calculate the result."
        
        result = PromptOptimizer._add_reasoning_chain(prompt)
        
        assert 'step' in result.lower()
        assert len(result) > len(prompt)
    
    def test_add_detailed_context(self):
        """Test adding detailed context for Claude"""
        prompt = "Solve this problem."
        
        result = PromptOptimizer._add_detailed_context(prompt)
        
        assert len(result) > len(prompt)
        assert 'comprehensive' in result or 'aspects' in result or 'balance' in result.lower()
    
    def test_compress_prompt_removes_whitespace(self):
        """Test that compression removes extra whitespace"""
        prompt = """
        
        
        Text with     lots    of    spaces.
        
        
        And multiple empty lines.
        
        
        """
        
        result = PromptOptimizer._compress_prompt(prompt)
        
        # Should not have multiple consecutive newlines
        assert '\n\n\n' not in result
        # Should still have some content
        assert 'Text' in result
        assert 'multiple empty lines' in result
    
    def test_estimate_tokens(self):
        """Test token estimation"""
        # Test with known string length
        text = "a" * 100  # 100 characters
        
        estimated = PromptOptimizer._estimate_tokens(text)
        
        # Should be approximately 100/4 = 25 tokens
        assert 20 <= estimated <= 30
    
    def test_add_examples_basic(self):
        """Test adding few-shot examples"""
        prompt = "Classify the sentiment of text."
        examples = [
            {'input': 'Great product!', 'output': 'positive'},
            {'input': 'Terrible service', 'output': 'negative'},
            {'input': 'It was okay', 'output': 'neutral'},
            {'input': 'Amazing!', 'output': 'positive'}
        ]
        
        enhanced = PromptOptimizer.add_examples(prompt, examples, max_examples=3)
        
        assert 'Examples:' in enhanced
        assert 'Example 1:' in enhanced
        assert 'Example 2:' in enhanced
        assert 'Example 3:' in enhanced
        assert 'Example 4:' not in enhanced  # Should respect max_examples
        assert 'Great product!' in enhanced
        assert 'positive' in enhanced
        assert 'Terrible service' in enhanced
        assert 'negative' in enhanced
    
    def test_add_examples_empty(self):
        """Test adding examples with empty list"""
        prompt = "Original prompt"
        
        enhanced = PromptOptimizer.add_examples(prompt, [], max_examples=3)
        
        assert enhanced == prompt  # Should return original
    
    def test_add_constraints_basic(self):
        """Test adding constraints to prompt"""
        prompt = "Generate a machine learning model."
        constraints = {
            'max_training_time': '1 hour',
            'memory_limit': '4GB',
            'interpretability': 'required',
            'accuracy_target': '> 0.90'
        }
        
        enhanced = PromptOptimizer.add_constraints(prompt, constraints)
        
        assert 'Constraints:' in enhanced
        assert 'max_training_time: 1 hour' in enhanced
        assert 'memory_limit: 4GB' in enhanced
        assert 'interpretability: required' in enhanced
        assert 'accuracy_target: > 0.90' in enhanced
    
    def test_add_constraints_empty(self):
        """Test adding constraints with empty dict"""
        prompt = "Original prompt"
        
        enhanced = PromptOptimizer.add_constraints(prompt, {})
        
        assert enhanced == prompt  # Should return original
    
    def test_format_for_json_output(self):
        """Test formatting prompt for JSON output"""
        prompt = "Analyze the data and provide results."
        
        formatted = PromptOptimizer.format_for_json_output(prompt)
        
        assert 'Output Format:' in formatted
        assert 'valid JSON' in formatted
        assert 'ONLY with valid JSON' in formatted
        assert 'Do not include any explanatory text' in formatted
        assert 'properly formatted' in formatted
    
    def test_combined_optimization(self):
        """Test combining multiple optimization techniques"""
        prompt = "Analyze data"
        examples = [
            {'input': 'data1', 'output': 'result1'},
            {'input': 'data2', 'output': 'result2'}
        ]
        constraints = {
            'time_limit': '5 minutes',
            'output_format': 'JSON'
        }
        
        # Apply multiple optimizations
        optimized = PromptOptimizer.optimize_for_model(prompt, 'gpt-4', 2000)
        optimized = PromptOptimizer.add_examples(optimized, examples, max_examples=2)
        optimized = PromptOptimizer.add_constraints(optimized, constraints)
        optimized = PromptOptimizer.format_for_json_output(optimized)
        
        # Should contain elements from all optimizations
        assert 'step' in optimized.lower()  # From GPT-4 optimization
        assert 'Examples:' in optimized  # From examples
        assert 'Constraints:' in optimized  # From constraints
        assert 'valid JSON' in optimized  # From JSON formatting


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
