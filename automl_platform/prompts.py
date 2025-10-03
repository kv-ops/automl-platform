"""
Optimized Prompt Templates for AutoML LLM Integration
Inspired by DataRobot and Akkio's approaches
"""

from typing import Dict, Any, List
import json

from .risk import RiskLevel


_NON_NEUTRAL_RISK_LEVELS = "|".join(
    level.value for level in (RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH)
)


class PromptTemplates:
    """Collection of optimized prompts for different AutoML tasks."""
    
    # ========== Data Analysis Prompts ==========
    
    DATA_QUALITY_ANALYSIS = f"""
You are an expert data scientist performing a comprehensive data quality assessment.

Analyze this dataset:
{{dataset_summary}}

Provide a detailed analysis covering:

1. **Data Quality Issues**:
   - Missing values patterns (random, systematic, or informative)
   - Outliers and anomalies
   - Data type mismatches
   - Duplicate records
   - Inconsistent formatting

2. **Statistical Anomalies**:
   - Unusual distributions (multimodal, heavily skewed)
   - Correlation anomalies
   - Class imbalance issues
   - Target leakage indicators

3. **Recommended Actions**:
   - Specific cleaning steps with priority
   - Imputation strategies for missing values
   - Outlier handling approach
   - Feature transformations needed

4. **Risk Assessment**:
   - Data issues that could impact model performance
   - Potential biases in the data
   - Compliance concerns (PII, sensitive data)

Format your response as structured JSON:
{{
    "quality_score": 0-100,
    "critical_issues": [...],
    "warnings": [...],
    "recommendations": [...],
    "risk_level": "{_NON_NEUTRAL_RISK_LEVELS}"
}}
"""

    COLUMN_ANALYSIS = """
Analyze these columns for feature engineering potential:

Columns: {columns_info}
Target: {target_info}
Task Type: {task_type}

For each column, identify:
1. Data quality (missing %, outliers)
2. Relationship with target (correlation, mutual information)
3. Transformation suggestions (log, polynomial, binning)
4. Interaction potential with other columns
5. Whether to keep, transform, or drop

Prioritize features by their potential impact on model performance.
"""

    # ========== Feature Engineering Prompts ==========
    
    FEATURE_GENERATION = """
You are a feature engineering expert. Create advanced features for this dataset:

Dataset Overview:
{dataset_info}

Target Variable: {target_column}
Task Type: {task_type}
Domain: {domain}

Generate features in these categories:

1. **Statistical Features**:
   - Aggregations (mean, std, skew, kurtosis)
   - Rolling statistics for temporal data
   - Percentile-based features

2. **Interaction Features**:
   - Multiplication/division of numeric features
   - Polynomial combinations (degree 2-3)
   - Conditional features (if-then logic)

3. **Domain-Specific Features**:
   - Based on {domain} best practices
   - Business logic transformations
   - Expert knowledge encoding

4. **Advanced Transformations**:
   - Target encoding for high-cardinality categoricals
   - Clustering-based features
   - PCA/embedding features

For each feature, provide:
- Name and description
- Python code using pandas/numpy
- Expected impact (high/medium/low)
- Computational cost

Limit to top 10 most impactful features.
"""

    AUTO_FEATURE_CODE = """
Generate Python code for these feature engineering tasks:
{feature_tasks}

Requirements:
- Use pandas and numpy
- Handle edge cases (division by zero, null values)
- Add inline comments
- Optimize for performance
- Return a function that takes df and returns df with new features

Template:
```python
def engineer_features(df):
    \"\"\"Auto-generated feature engineering.\"\"\"
    df = df.copy()
    
    # Your feature engineering code here
    
    return df
```
"""

    # ========== Model Selection Prompts ==========
    
    MODEL_RECOMMENDATION = """
Recommend ML models for this dataset:

Data Characteristics:
- Samples: {n_samples}
- Features: {n_features}
- Task: {task_type}
- Feature Types: {feature_types}
- Class Balance: {class_balance}
- Missing Data: {missing_ratio}%
- Dataset Size: {memory_size}

Constraints:
- Max Training Time: {max_time}
- Interpretability Required: {interpretability}
- Production Environment: {environment}

Recommend 5-7 models ranked by suitability:

For each model provide:
1. Model name and family
2. Why it's suitable for this data
3. Recommended hyperparameters
4. Expected performance characteristics
5. Training time estimate
6. Pros and cons
7. Implementation tips

Consider:
- Linear vs non-linear relationships
- Feature interactions
- Scalability requirements
- Ensemble potential
"""

    HYPERPARAMETER_SUGGESTION = """
Suggest optimal hyperparameters for {model_name}:

Dataset Properties:
{dataset_properties}

Performance Metrics So Far:
{current_metrics}

Based on the data characteristics and model type, suggest:

1. **Critical Hyperparameters** (high impact):
   - Parameter name, suggested range, and why

2. **Secondary Hyperparameters** (moderate impact):
   - Fine-tuning parameters

3. **Search Strategy**:
   - Grid search vs random vs Bayesian
   - Number of iterations recommended
   - Early stopping criteria

4. **Specific Recommendations**:
   - For preventing overfitting
   - For handling class imbalance
   - For computational efficiency

Format as JSON with parameter names and values.
"""

    # ========== Model Explanation Prompts ==========
    
    MODEL_INTERPRETATION = """
Explain this model's behavior in business terms:

Model: {model_type}
Performance Metrics: {metrics}
Top Features: {feature_importance}
Confusion Matrix: {confusion_matrix}

Provide:

1. **Executive Summary** (2-3 sentences)
   - What the model does well
   - Key limitations

2. **Performance Analysis**:
   - Interpret each metric in business context
   - Compare to baseline/industry standards
   - Statistical significance of results

3. **Feature Insights**:
   - Why top features make business sense
   - Unexpected important features
   - Missing features that might help

4. **Recommendations**:
   - How to improve the model
   - When to retrain
   - Deployment considerations

5. **Risk Assessment**:
   - Potential failure modes
   - Bias concerns
   - Monitoring requirements

Use non-technical language suitable for stakeholders.
"""

    SHAP_EXPLANATION = """
Interpret these SHAP values for a {stakeholder_type}:

Feature SHAP Values: {shap_values}
Prediction: {prediction}
Actual: {actual}

Explain:
1. What drove this specific prediction
2. Which features pushed toward/against the prediction  
3. Any concerning patterns
4. Confidence in this explanation

Use analogies and simple language appropriate for {stakeholder_type}.
"""

    # ========== Error Analysis Prompts ==========
    
    ERROR_PATTERN_ANALYSIS = """
Analyze these prediction errors to find patterns:

Error Statistics: {error_stats}
Worst Predictions: {worst_cases}
Feature Values for Errors: {error_features}

Identify:

1. **Error Patterns**:
   - Systematic vs random errors
   - Specific regions where model fails
   - Feature combinations causing errors

2. **Root Causes**:
   - Data quality issues
   - Model limitations
   - Feature engineering gaps
   - Training data biases

3. **Improvement Strategies**:
   - Additional features needed
   - Different algorithms to try
   - Data collection improvements
   - Ensemble strategies

4. **Priority Actions**:
   Rank improvements by expected impact

Provide specific, actionable recommendations.
"""

    DRIFT_ANALYSIS = """
Analyze this data drift report:

Reference Distribution: {reference_stats}
Current Distribution: {current_stats}
Drift Metrics: {drift_metrics}
Time Period: {time_period}

Assess:

1. **Drift Severity**:
   - Which features drifted most
   - Is drift gradual or sudden
   - Statistical significance

2. **Business Impact**:
   - How drift affects predictions
   - Risk to model performance
   - Business metrics at risk

3. **Root Causes**:
   - Natural drift vs data issues
   - Seasonal patterns
   - External factors

4. **Recommendations**:
   - Retrain immediately?
   - Adjust monitoring thresholds?
   - Collect new training data?
   - Feature engineering changes?

Priority: {urgency_level}
"""

    # ========== Report Generation Prompts ==========
    
    EXECUTIVE_REPORT = """
Generate an executive report for this AutoML experiment:

Experiment Summary:
{experiment_summary}

Business Context:
{business_context}

Results:
{results}

Create a professional report with:

1. **Executive Summary** (1 paragraph)
   - Business problem addressed
   - Key outcome
   - ROI/Impact estimate

2. **Methodology** (brief, non-technical)
   - Data used
   - Approach taken
   - Validation method

3. **Key Findings**:
   - Model performance vs baseline
   - Important insights discovered
   - Confidence in results

4. **Recommendations**:
   - Deployment strategy
   - Expected benefits
   - Risks and mitigations
   - Next steps

5. **Appendix** (optional technical details)

Tone: Professional, confident, action-oriented
Length: 2-3 pages equivalent
Format: {output_format}
"""

    TECHNICAL_REPORT = """
Generate a detailed technical report:

Full Experiment Details:
{experiment_details}

Include:

1. **Data Analysis**:
   - Dataset characteristics
   - Quality assessment
   - Feature engineering performed

2. **Model Development**:
   - Algorithms tested
   - Hyperparameter optimization
   - Cross-validation strategy
   - Learning curves

3. **Results**:
   - Detailed metrics table
   - Feature importance analysis
   - Error analysis
   - Statistical tests

4. **Production Readiness**:
   - Performance benchmarks
   - Scalability analysis
   - Monitoring plan
   - API specifications

5. **Reproducibility**:
   - Environment details
   - Random seeds
   - Data versioning
   - Code snippets

Format: Technical documentation style
Include: Code examples, equations, visualizations descriptions
"""

    # ========== Interactive Prompts ==========
    
    DATA_CLEANING_DIALOGUE = """
You are an intelligent data cleaning assistant like Akkio's GPT-4 agent.

Current Data State:
{data_state}

User Request: {user_request}

Previous Actions: {action_history}

Respond with:

1. **Understanding Confirmation**:
   "I understand you want to [rephrase request]"

2. **Proposed Actions**:
   - Step-by-step plan
   - Potential risks/impacts
   - Alternative approaches

3. **Code Generation**:
   ```python
   # Clean, commented code
   ```

4. **Impact Preview**:
   - Rows affected
   - Data shape change
   - Quality improvement estimate

5. **Next Suggestions**:
   - Related cleaning tasks
   - Logical next steps

Be conversational but precise. Ask for clarification if needed.
"""

    QUESTION_ANSWERING = """
Answer this question about the AutoML results:

Question: {question}

Context:
- Experiment: {experiment_context}
- Models Trained: {models}
- Best Performance: {best_metrics}
- Current Stage: {stage}

Provide:
1. Direct answer to the question
2. Supporting evidence from the results
3. Additional relevant insights
4. Suggested follow-up actions

Adapt complexity to user's apparent technical level.
"""

    # ========== Code Generation Prompts ==========
    
    AUTOML_CODE_GENERATION = """
Generate production-ready AutoML code for:

Task Description: {task_description}
Data Format: {data_format}
Requirements: {requirements}
Constraints: {constraints}

Generate complete Python code with:

1. **Imports and Setup**
2. **Data Loading and Validation**
3. **Feature Engineering Pipeline**
4. **Model Training with HPO**
5. **Evaluation and Metrics**
6. **Model Persistence**
7. **Prediction Pipeline**
8. **Error Handling**
9. **Logging and Monitoring**
10. **Documentation**

Use best practices:
- Type hints
- Docstrings
- Error handling
- Logging
- Configuration management
- Unit test examples

Framework: {framework}
Style: Production-ready, maintainable
"""

    CUSTOM_TRANSFORMER = """
Create a custom scikit-learn transformer for:

Transformation Need: {transformation_description}
Input Data Example: {data_example}
Expected Output: {expected_output}

Generate:

```python
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

class Custom{transformation_name}Transformer(BaseEstimator, TransformerMixin):
    \"\"\"
    {Brief description}
    
    Parameters
    ----------
    {parameters}
    
    Attributes
    ----------
    {attributes}
    \"\"\"
    
    def __init__(self, ...):
        # Initialization
        
    def fit(self, X, y=None):
        # Fitting logic
        return self
        
    def transform(self, X):
        # Transformation logic
        return X_transformed
        
    def get_feature_names_out(self, input_features=None):
        # Feature names for pipeline
        return feature_names
```

Include:
- Parameter validation
- Error handling  
- Preservation of DataFrame structure
- Inverse transform if applicable
"""

    @classmethod
    def get_prompt(cls, prompt_name: str, **kwargs) -> str:
        """Get a formatted prompt template."""
        prompt = getattr(cls, prompt_name, None)
        if prompt is None:
            raise ValueError(f"Prompt '{prompt_name}' not found")
        
        # Format with provided kwargs
        try:
            return prompt.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Missing required parameter for prompt '{prompt_name}': {e}")
    
    @classmethod
    def list_prompts(cls) -> List[str]:
        """List all available prompt templates."""
        return [
            attr for attr in dir(cls)
            if not attr.startswith('_') and 
            isinstance(getattr(cls, attr), str) and
            attr.isupper()
        ]


class PromptOptimizer:
    """Optimize prompts based on context and constraints."""
    
    @staticmethod
    def optimize_for_model(prompt: str, model: str, max_tokens: int) -> str:
        """Optimize prompt for specific model and token limits."""
        
        # Model-specific optimizations
        if "gpt-3.5" in model.lower():
            # GPT-3.5 works better with structured prompts
            prompt = PromptOptimizer._add_structure_markers(prompt)
        elif "gpt-4" in model.lower():
            # GPT-4 handles complex reasoning better
            prompt = PromptOptimizer._add_reasoning_chain(prompt)
        elif "claude" in model.lower():
            # Claude prefers detailed context
            prompt = PromptOptimizer._add_detailed_context(prompt)
        
        # Token limit optimization
        if PromptOptimizer._estimate_tokens(prompt) > max_tokens * 0.5:
            prompt = PromptOptimizer._compress_prompt(prompt)
        
        return prompt
    
    @staticmethod
    def _add_structure_markers(prompt: str) -> str:
        """Add structure markers for better parsing."""
        markers = [
            ("Provide", "**Response Format:**\nProvide"),
            ("For each", "**For each item:**\n"),
            ("Include", "**Required Elements:**\nInclude"),
        ]
        
        for old, new in markers:
            prompt = prompt.replace(old, new)
        
        return prompt
    
    @staticmethod
    def _add_reasoning_chain(prompt: str) -> str:
        """Add chain-of-thought reasoning."""
        if "step" not in prompt.lower():
            prompt += "\n\nThink through this step-by-step before providing your final answer."
        return prompt
    
    @staticmethod
    def _add_detailed_context(prompt: str) -> str:
        """Add more context for Claude."""
        context_addition = """
        
Consider all aspects of this problem and provide a comprehensive response.
Balance technical accuracy with practical applicability.
"""
        return prompt + context_addition
    
    @staticmethod
    def _compress_prompt(prompt: str) -> str:
        """Compress prompt to fit token limits."""
        # Remove extra whitespace
        lines = [line.strip() for line in prompt.split('\n')]
        prompt = '\n'.join(line for line in lines if line)
        
        # Abbreviate common phrases
        replacements = [
            ("For example", "E.g."),
            ("such as", "e.g."),
            ("including but not limited to", "including"),
            ("Provide a detailed", "Provide"),
        ]
        
        for old, new in replacements:
            prompt = prompt.replace(old, new)
        
        return prompt
    
    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Rough token estimation."""
        # Approximate: 1 token ≈ 4 characters
        return len(text) // 4
    
    @staticmethod
    def add_examples(prompt: str, examples: List[Dict[str, Any]], 
                    max_examples: int = 3) -> str:
        """Add few-shot examples to prompt."""
        if not examples:
            return prompt
        
        example_text = "\n\n**Examples:**\n"
        for i, example in enumerate(examples[:max_examples], 1):
            example_text += f"\nExample {i}:\n"
            example_text += f"Input: {example.get('input', 'N/A')}\n"
            example_text += f"Output: {example.get('output', 'N/A')}\n"
        
        return prompt + example_text
    
    @staticmethod
    def add_constraints(prompt: str, constraints: Dict[str, Any]) -> str:
        """Add constraints to prompt."""
        if not constraints:
            return prompt
        
        constraint_text = "\n\n**Constraints:**\n"
        for key, value in constraints.items():
            constraint_text += f"- {key}: {value}\n"
        
        return prompt + constraint_text
    
    @staticmethod
    def format_for_json_output(prompt: str) -> str:
        """Ensure prompt requests JSON output."""
        json_instruction = """

**Output Format:**
Respond ONLY with valid JSON. Do not include any explanatory text outside the JSON structure.
Ensure all JSON is properly formatted with correct quotes and escaping.
"""
        return prompt + json_instruction


# Example usage
if __name__ == "__main__":
    # Get a prompt template
    prompt = PromptTemplates.get_prompt(
        "DATA_QUALITY_ANALYSIS",
        dataset_summary="1000 rows, 20 columns, 5% missing values"
    )
    print("Data Quality Prompt:")
    print(prompt[:500] + "...")
    
    # Optimize for model
    optimized = PromptOptimizer.optimize_for_model(
        prompt, 
        model="gpt-3.5-turbo",
        max_tokens=1000
    )
    print("\nOptimized Prompt:")
    print(optimized[:500] + "...")
    
    # List all available prompts
    print("\nAvailable Prompts:")
    for prompt_name in PromptTemplates.list_prompts():
        print(f"  - {prompt_name}")
