"""
AutoML Platform - Data Preprocessing Example
=============================================
Comprehensive example demonstrating data preprocessing, feature engineering,
quality assessment and their impact on model performance.

This example shows:
1. Loading datasets (public and custom)
2. Data preprocessing with DataPreprocessor
3. Automatic feature engineering with AutoFeatureEngineer
4. Data quality assessment with DataQualityAssessment
5. Training models with preprocessed features
6. Comparing model performance with and without feature engineering
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from sklearn.datasets import load_iris, load_wine, load_boston, fetch_california_housing
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Import AutoML components
from automl_platform.data_prep import DataPreprocessor, validate_data, create_lag_features
from automl_platform.feature_engineering import AutoFeatureEngineer, create_time_series_features
from automl_platform.data_quality_agent import (
    DataQualityAssessment, 
    DataRobotStyleQualityMonitor,
    IntelligentDataQualityAgent
)
from automl_platform.orchestrator import AutoMLOrchestrator
from automl_platform.config import AutoMLConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_sample_data(dataset_name: str = 'iris'):
    """
    Load sample dataset for demonstration.
    
    Args:
        dataset_name: Name of dataset ('iris', 'wine', 'california', 'custom')
    
    Returns:
        X, y: Features and target
    """
    logger.info(f"Loading dataset: {dataset_name}")
    
    if dataset_name == 'iris':
        data = load_iris()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = pd.Series(data.target, name='target')
        
    elif dataset_name == 'wine':
        data = load_wine()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = pd.Series(data.target, name='target')
        
    elif dataset_name == 'california':
        data = fetch_california_housing()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = pd.Series(data.target, name='target')
        
    elif dataset_name == 'custom':
        # Create synthetic dataset with various data types
        np.random.seed(42)
        n_samples = 1000
        
        X = pd.DataFrame({
            # Numeric features
            'age': np.random.randint(18, 80, n_samples),
            'income': np.random.lognormal(10, 1, n_samples),
            'credit_score': np.random.normal(700, 100, n_samples),
            'purchase_amount': np.random.exponential(100, n_samples),
            
            # Categorical features
            'gender': np.random.choice(['M', 'F'], n_samples),
            'region': np.random.choice(['North', 'South', 'East', 'West'], n_samples),
            'product_category': np.random.choice(['Electronics', 'Clothing', 'Food', 'Books', 'Sports'], n_samples),
            
            # Date features
            'purchase_date': pd.date_range('2023-01-01', periods=n_samples, freq='H'),
            
            # Text feature
            'customer_comment': ['Good product ' * np.random.randint(1, 5) for _ in range(n_samples)],
            
            # Features with missing values
            'optional_feature_1': np.where(np.random.random(n_samples) > 0.7, np.nan, np.random.randn(n_samples)),
            'optional_feature_2': np.where(np.random.random(n_samples) > 0.8, np.nan, 
                                          np.random.choice(['A', 'B', 'C'], n_samples)),
            
            # High cardinality feature
            'customer_id': [f'CUST_{i:05d}' for i in range(n_samples)],
            
            # Binary feature
            'is_premium': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
            
            # Feature with outliers
            'transaction_amount': np.concatenate([
                np.random.normal(100, 20, n_samples-50),
                np.random.normal(1000, 50, 50)  # Outliers
            ])[:n_samples]
        })
        
        # Target variable (binary classification)
        y = pd.Series(
            np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
            name='churned'
        )
        
        # Inject some data quality issues
        X.loc[100:110, 'age'] = -1  # Invalid ages
        X.loc[200:210, 'credit_score'] = 99999  # Invalid credit scores
        
        # Add duplicates
        X = pd.concat([X, X.iloc[:5]], ignore_index=True)
        y = pd.concat([y, y.iloc[:5]], ignore_index=True)
        
    else:
        # Load from CSV file if exists
        csv_path = Path(f'examples/data/{dataset_name}.csv')
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            X = df.drop('target', axis=1)
            y = df['target']
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
    
    logger.info(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
    return X, y


def demonstrate_data_preprocessing(X, y):
    """
    Demonstrate comprehensive data preprocessing.
    
    Args:
        X: Features DataFrame
        y: Target Series
    
    Returns:
        X_processed: Processed features
        preprocessor: Fitted preprocessor
    """
    logger.info("\n" + "="*60)
    logger.info("STEP 1: DATA PREPROCESSING")
    logger.info("="*60)
    
    # Configure preprocessor
    config = {
        'handle_outliers': True,
        'outlier_method': 'iqr',
        'outlier_threshold': 1.5,
        'imputation_method': 'median',
        'scaling_method': 'robust',
        'encoding_method': 'onehot',
        'high_cardinality_threshold': 20,
        'rare_category_threshold': 0.01,
        'enable_quality_checks': True,
        'enable_drift_detection': False,
        'max_missing_ratio': 0.5
    }
    
    preprocessor = DataPreprocessor(config)
    
    # Detect feature types
    logger.info("\nDetecting feature types...")
    feature_types = preprocessor.detect_feature_types(X)
    
    for ftype, features in feature_types.items():
        if features:
            logger.info(f"  {ftype}: {len(features)} features")
            logger.info(f"    → {features[:3]}{'...' if len(features) > 3 else ''}")
    
    # Check data quality
    logger.info("\nChecking data quality...")
    quality_report = preprocessor.check_data_quality(X)
    
    logger.info(f"  Quality Score: {quality_report['quality_score']:.1f}/100")
    
    if quality_report['issues']:
        logger.warning(f"  Found {len(quality_report['issues'])} issues:")
        for issue in quality_report['issues'][:5]:
            logger.warning(f"    - {issue}")
    
    if quality_report['warnings']:
        logger.info(f"  Found {len(quality_report['warnings'])} warnings:")
        for warning in quality_report['warnings'][:5]:
            logger.info(f"    - {warning}")
    
    # Handle datetime features
    if preprocessor.datetime_features:
        logger.info("\nCreating datetime features...")
        X = preprocessor.create_datetime_features(X)
        logger.info(f"  Created {len([c for c in X.columns if any(dt in c for dt in preprocessor.datetime_features)])} datetime features")
    
    # Handle rare categories
    if preprocessor.categorical_features:
        logger.info("\nHandling rare categories...")
        X = preprocessor.handle_rare_categories(X)
    
    # Handle outliers
    if preprocessor.numeric_features and config['handle_outliers']:
        logger.info("\nHandling outliers...")
        X = preprocessor.handle_outliers(X, method='clip')
    
    # Create preprocessing pipeline
    logger.info("\nCreating preprocessing pipeline...")
    pipeline = preprocessor.create_pipeline(X)
    
    # Fit and transform
    X_processed = preprocessor.fit_transform(X, y)
    
    logger.info(f"\nPreprocessing complete:")
    logger.info(f"  Original shape: {X.shape}")
    logger.info(f"  Processed shape: {X_processed.shape}")
    logger.info(f"  Features expanded from {X.shape[1]} to {X_processed.shape[1]}")
    
    return X_processed, preprocessor


def demonstrate_feature_engineering(X, y):
    """
    Demonstrate automatic feature engineering.
    
    Args:
        X: Features DataFrame
        y: Target Series
    
    Returns:
        X_engineered: Features with engineered features
        feature_engineer: Fitted feature engineer
    """
    logger.info("\n" + "="*60)
    logger.info("STEP 2: AUTOMATIC FEATURE ENGINEERING")
    logger.info("="*60)
    
    # Initialize feature engineer
    config = {
        'create_interactions': True,
        'max_interactions': 10,
        'create_polynomial': True,
        'polynomial_degree': 2,
        'create_ratios': True,
        'create_aggregates': True,
        'create_cyclical': True,
        'create_text_features': True,
        'text_vectorization': 'tfidf',
        'max_text_features': 50,
        'feature_selection': True,
        'selection_method': 'mutual_info',
        'n_features_to_select': 50
    }
    
    feature_engineer = AutoFeatureEngineer(config)
    
    # Analyze features
    logger.info("\nAnalyzing features for engineering opportunities...")
    suggestions = feature_engineer.suggest_features(X, y)
    
    logger.info(f"  Generated {len(suggestions)} feature suggestions:")
    for i, suggestion in enumerate(suggestions[:5], 1):
        logger.info(f"    {i}. {suggestion['name']}: {suggestion['description']}")
        logger.info(f"       Importance: {suggestion['importance']}")
    
    # Create interaction features
    logger.info("\nCreating interaction features...")
    X_interactions = feature_engineer.create_interactions(X, y)
    n_interactions = X_interactions.shape[1] - X.shape[1]
    logger.info(f"  Created {n_interactions} interaction features")
    
    # Create polynomial features
    logger.info("\nCreating polynomial features...")
    X_poly = feature_engineer.create_polynomial_features(X_interactions)
    n_poly = X_poly.shape[1] - X_interactions.shape[1]
    logger.info(f"  Created {n_poly} polynomial features")
    
    # Create ratio features
    logger.info("\nCreating ratio features...")
    X_ratios = feature_engineer.create_ratio_features(X_poly)
    n_ratios = X_ratios.shape[1] - X_poly.shape[1]
    logger.info(f"  Created {n_ratios} ratio features")
    
    # Create aggregate features (if applicable)
    if 'customer_id' in X.columns or 'group' in X.columns:
        logger.info("\nCreating aggregate features...")
        group_col = 'customer_id' if 'customer_id' in X.columns else 'group'
        X_agg = feature_engineer.create_aggregate_features(X_ratios, group_col)
        n_agg = X_agg.shape[1] - X_ratios.shape[1]
        logger.info(f"  Created {n_agg} aggregate features")
    else:
        X_agg = X_ratios
    
    # Fit and transform all features
    logger.info("\nApplying complete feature engineering pipeline...")
    X_engineered = feature_engineer.fit_transform(X, y)
    
    # Get feature importance
    feature_importance = feature_engineer.get_feature_importance(X_engineered, y)
    
    if feature_importance:
        logger.info("\nTop 10 most important engineered features:")
        for i, (feature, importance) in enumerate(feature_importance[:10], 1):
            logger.info(f"    {i}. {feature}: {importance:.4f}")
    
    logger.info(f"\nFeature engineering complete:")
    logger.info(f"  Original features: {X.shape[1]}")
    logger.info(f"  Engineered features: {X_engineered.shape[1]}")
    logger.info(f"  Total features created: {X_engineered.shape[1] - X.shape[1]}")
    
    return X_engineered, feature_engineer


def demonstrate_quality_assessment(X, y):
    """
    Demonstrate comprehensive data quality assessment.
    
    Args:
        X: Features DataFrame
        y: Target Series
    
    Returns:
        assessment: DataQualityAssessment object
    """
    logger.info("\n" + "="*60)
    logger.info("STEP 3: DATA QUALITY ASSESSMENT")
    logger.info("="*60)
    
    # Initialize quality monitor
    monitor = DataRobotStyleQualityMonitor()
    
    # Perform assessment
    logger.info("\nPerforming comprehensive quality assessment...")
    assessment = monitor.assess_quality(X, target_column=y.name if hasattr(y, 'name') else 'target')
    
    # Display results
    logger.info(f"\n📊 QUALITY ASSESSMENT RESULTS")
    logger.info(f"  Overall Quality Score: {assessment.quality_score:.1f}/100")
    logger.info(f"  Drift Risk: {assessment.drift_risk}")
    logger.info(f"  Target Leakage Risk: {'Yes' if assessment.target_leakage_risk else 'No'}")
    
    # Display alerts
    if assessment.alerts:
        logger.warning(f"\n⚠️  CRITICAL ALERTS ({len(assessment.alerts)}):")
        for alert in assessment.alerts[:5]:
            logger.warning(f"    - {alert['message']}")
            logger.warning(f"      Action: {alert['action']}")
    
    # Display warnings
    if assessment.warnings:
        logger.info(f"\n⚡ WARNINGS ({len(assessment.warnings)}):")
        for warning in assessment.warnings[:5]:
            logger.info(f"    - {warning['message']}")
    
    # Display statistics
    logger.info(f"\n📈 DATASET STATISTICS:")
    for key, value in assessment.statistics.items():
        logger.info(f"    {key.replace('_', ' ').title()}: {value}")
    
    # Display recommendations
    logger.info(f"\n💡 RECOMMENDATIONS ({len(assessment.recommendations)}):")
    for i, rec in enumerate(assessment.recommendations[:3], 1):
        logger.info(f"\n  {i}. {rec['title']} (Priority: {rec['priority']})")
        logger.info(f"     {rec['description']}")
        logger.info(f"     Actions:")
        for action in rec['actions'][:3]:
            logger.info(f"       - {action}")
    
    # Generate quality report
    agent = IntelligentDataQualityAgent()
    report = agent.get_quality_report(assessment)
    
    # Save report to file
    report_path = Path('examples/data_quality_report.md')
    report_path.parent.mkdir(exist_ok=True)
    with open(report_path, 'w') as f:
        f.write(report)
    logger.info(f"\n📄 Full quality report saved to: {report_path}")
    
    return assessment


def train_and_compare_models(X_original, X_preprocessed, X_engineered, y):
    """
    Train models and compare performance with different feature sets.
    
    Args:
        X_original: Original features
        X_preprocessed: Preprocessed features
        X_engineered: Engineered features
        y: Target variable
    
    Returns:
        comparison_results: Dictionary with comparison results
    """
    logger.info("\n" + "="*60)
    logger.info("STEP 4: MODEL TRAINING AND COMPARISON")
    logger.info("="*60)
    
    # Configure AutoML
    config = AutoMLConfig(
        task='auto',
        algorithms=['RandomForest', 'XGBoost', 'LightGBM'],
        cv_folds=3,
        scoring='auto',
        n_jobs=-1,
        random_state=42,
        hpo_n_iter=10,
        handle_imbalance=True,
        enable_cache=False
    )
    
    results = {}
    
    # Split data
    X_train_orig, X_test_orig, y_train, y_test = train_test_split(
        X_original, y, test_size=0.2, random_state=42, stratify=y if y.nunique() < 20 else None
    )
    
    X_train_prep, X_test_prep = train_test_split(
        X_preprocessed, test_size=0.2, random_state=42
    )[0:2]
    
    X_train_eng, X_test_eng = train_test_split(
        X_engineered, test_size=0.2, random_state=42
    )[0:2]
    
    # Train with original features
    logger.info("\n1️⃣  Training with ORIGINAL features...")
    logger.info(f"   Shape: {X_train_orig.shape}")
    
    orchestrator_orig = AutoMLOrchestrator(config)
    try:
        orchestrator_orig.fit(X_train_orig, y_train)
        
        # Get best model performance
        if orchestrator_orig.leaderboard:
            best_orig = orchestrator_orig.leaderboard[0]
            results['original'] = {
                'best_model': best_orig['model'],
                'cv_score': best_orig['cv_score'],
                'n_features': X_train_orig.shape[1],
                'training_time': best_orig['training_time']
            }
            logger.info(f"   Best model: {best_orig['model']}")
            logger.info(f"   CV Score: {best_orig['cv_score']:.4f}")
    except Exception as e:
        logger.warning(f"   Failed to train with original features: {e}")
        results['original'] = {'cv_score': 0, 'error': str(e)}
    
    # Train with preprocessed features
    logger.info("\n2️⃣  Training with PREPROCESSED features...")
    logger.info(f"   Shape: {X_train_prep.shape}")
    
    orchestrator_prep = AutoMLOrchestrator(config)
    try:
        # Convert to DataFrame if numpy array
        if isinstance(X_train_prep, np.ndarray):
            X_train_prep = pd.DataFrame(X_train_prep)
            X_test_prep = pd.DataFrame(X_test_prep)
        
        orchestrator_prep.fit(X_train_prep, y_train)
        
        if orchestrator_prep.leaderboard:
            best_prep = orchestrator_prep.leaderboard[0]
            results['preprocessed'] = {
                'best_model': best_prep['model'],
                'cv_score': best_prep['cv_score'],
                'n_features': X_train_prep.shape[1],
                'training_time': best_prep['training_time']
            }
            logger.info(f"   Best model: {best_prep['model']}")
            logger.info(f"   CV Score: {best_prep['cv_score']:.4f}")
    except Exception as e:
        logger.warning(f"   Failed to train with preprocessed features: {e}")
        results['preprocessed'] = {'cv_score': 0, 'error': str(e)}
    
    # Train with engineered features
    logger.info("\n3️⃣  Training with ENGINEERED features...")
    logger.info(f"   Shape: {X_train_eng.shape}")
    
    orchestrator_eng = AutoMLOrchestrator(config)
    try:
        orchestrator_eng.fit(X_train_eng, y_train)
        
        if orchestrator_eng.leaderboard:
            best_eng = orchestrator_eng.leaderboard[0]
            results['engineered'] = {
                'best_model': best_eng['model'],
                'cv_score': best_eng['cv_score'],
                'n_features': X_train_eng.shape[1],
                'training_time': best_eng['training_time']
            }
            logger.info(f"   Best model: {best_eng['model']}")
            logger.info(f"   CV Score: {best_eng['cv_score']:.4f}")
    except Exception as e:
        logger.warning(f"   Failed to train with engineered features: {e}")
        results['engineered'] = {'cv_score': 0, 'error': str(e)}
    
    # Display comparison
    logger.info("\n" + "="*60)
    logger.info("📊 PERFORMANCE COMPARISON")
    logger.info("="*60)
    
    comparison_df = pd.DataFrame(results).T
    logger.info("\n" + comparison_df.to_string())
    
    # Calculate improvements
    if 'original' in results and 'cv_score' in results['original']:
        base_score = results['original'].get('cv_score', 0)
        
        if base_score > 0:
            if 'preprocessed' in results and 'cv_score' in results['preprocessed']:
                prep_improvement = ((results['preprocessed']['cv_score'] - base_score) / base_score) * 100
                logger.info(f"\n✅ Preprocessing improvement: {prep_improvement:+.2f}%")
            
            if 'engineered' in results and 'cv_score' in results['engineered']:
                eng_improvement = ((results['engineered']['cv_score'] - base_score) / base_score) * 100
                logger.info(f"✅ Feature engineering improvement: {eng_improvement:+.2f}%")
    
    return results


def demonstrate_advanced_preprocessing():
    """
    Demonstrate advanced preprocessing scenarios.
    """
    logger.info("\n" + "="*60)
    logger.info("ADVANCED PREPROCESSING EXAMPLES")
    logger.info("="*60)
    
    # Example 1: Time series preprocessing
    logger.info("\n📈 Time Series Preprocessing:")
    
    # Create time series data
    dates = pd.date_range('2023-01-01', periods=365, freq='D')
    ts_data = pd.DataFrame({
        'date': dates,
        'value': np.sin(np.arange(365) * 2 * np.pi / 30) + np.random.randn(365) * 0.1,
        'temperature': 20 + 10 * np.sin(np.arange(365) * 2 * np.pi / 365) + np.random.randn(365),
        'holiday': np.random.choice([0, 1], 365, p=[0.95, 0.05])
    })
    
    # Create lag features
    ts_features = create_lag_features(
        ts_data,
        target_col='value',
        lag_periods=[1, 7, 30],
        rolling_windows=[7, 30]
    )
    
    logger.info(f"  Created {ts_features.shape[1] - ts_data.shape[1]} time series features")
    
    # Example 2: Text preprocessing
    logger.info("\n📝 Text Feature Engineering:")
    
    text_data = pd.DataFrame({
        'review': [
            "This product is amazing! Best purchase ever.",
            "Terrible quality, would not recommend.",
            "Good value for money, satisfied with purchase.",
            "Average product, nothing special.",
            "Excellent customer service and fast delivery!"
        ],
        'rating': [5, 1, 4, 3, 5]
    })
    
    # Create text features
    config = {
        'create_text_features': True,
        'text_vectorization': 'tfidf',
        'max_text_features': 20
    }
    
    feature_engineer = AutoFeatureEngineer(config)
    text_features = feature_engineer.create_text_features(text_data)
    
    logger.info(f"  Created {text_features.shape[1] - text_data.shape[1]} text features")
    
    # Example 3: Handling imbalanced data
    logger.info("\n⚖️  Imbalanced Data Handling:")
    
    # Create imbalanced dataset
    X_imb = pd.DataFrame(np.random.randn(1000, 5), columns=[f'feat_{i}' for i in range(5)])
    y_imb = pd.Series(np.random.choice([0, 1], 1000, p=[0.95, 0.05]))
    
    logger.info(f"  Original class distribution: {y_imb.value_counts().to_dict()}")
    
    # Validate data
    validation_report = validate_data(X_imb)
    logger.info(f"  Validation score: {validation_report.get('quality_score', 'N/A')}")


def main():
    """
    Main function to run all demonstrations.
    """
    logger.info("="*60)
    logger.info("AUTOML PLATFORM - DATA PREPROCESSING DEMONSTRATION")
    logger.info("="*60)
    
    # Select dataset
    dataset_name = 'custom'  # Change to 'iris', 'wine', 'california', or 'custom'
    
    # Load data
    X, y = load_sample_data(dataset_name)
    
    # 1. Data Preprocessing
    X_preprocessed, preprocessor = demonstrate_data_preprocessing(X, y)
    
    # 2. Feature Engineering  
    X_engineered, feature_engineer = demonstrate_feature_engineering(X, y)
    
    # 3. Data Quality Assessment
    assessment = demonstrate_quality_assessment(X, y)
    
    # 4. Model Training and Comparison
    comparison_results = train_and_compare_models(X, X_preprocessed, X_engineered, y)
    
    # 5. Advanced preprocessing examples
    demonstrate_advanced_preprocessing()
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("DEMONSTRATION COMPLETE")
    logger.info("="*60)
    logger.info("\nKey Takeaways:")
    logger.info("1. Data preprocessing significantly improves model performance")
    logger.info("2. Feature engineering can create powerful predictive features")
    logger.info("3. Data quality assessment helps identify and fix issues early")
    logger.info("4. Comparing different feature sets helps optimize the pipeline")
    logger.info("5. The AutoML platform automates these complex processes")
    
    logger.info("\n✅ All demonstrations completed successfully!")
    
    return {
        'preprocessor': preprocessor,
        'feature_engineer': feature_engineer,
        'assessment': assessment,
        'comparison': comparison_results
    }


if __name__ == "__main__":
    # Run the demonstration
    results = main()
    
    # Optional: Save results
    import json
    results_path = Path('examples/preprocessing_results.json')
    
    # Convert results to serializable format
    serializable_results = {
        'comparison': results['comparison'],
        'quality_score': results['assessment'].quality_score,
        'drift_risk': results['assessment'].drift_risk
    }
    
    with open(results_path, 'w') as f:
        json.dump(serializable_results, f, indent=2, default=str)
    
    logger.info(f"\n📊 Results saved to: {results_path}")
