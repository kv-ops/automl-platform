#!/usr/bin/env python3
"""
Main entry point for AutoML Platform CLI.
Provides command-line interface for training and prediction.
"""

import argparse
import logging
import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Optional, List
import warnings

# Import platform modules
from automl_platform.config import AutoMLConfig
from automl_platform.orchestrator import AutoMLOrchestrator
from automl_platform.inference import load_pipeline, predict, predict_proba, save_predictions
from automl_platform.data_prep import validate_data
from automl_platform.metrics import calculate_metrics

# Setup logging
def setup_logging(verbose: int = 1, log_file: Optional[str] = None):
    """Configure logging for the application."""
    log_level = {
        0: logging.WARNING,
        1: logging.INFO,
        2: logging.DEBUG
    }.get(verbose, logging.INFO)
    
    format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=log_level,
        format=format_str,
        handlers=handlers
    )
    
    # Suppress some warnings
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)

logger = logging.getLogger(__name__)


def train(args):
    """Train AutoML model."""
    logger.info("="*80)
    logger.info("AUTOML PLATFORM - TRAINING MODE")
    logger.info("="*80)
    
    # Load configuration
    if args.config:
        logger.info(f"Loading configuration from {args.config}")
        config = AutoMLConfig.from_yaml(args.config)
    else:
        logger.info("Using default configuration")
        config = AutoMLConfig()
    
    # Override config with command line arguments
    if args.cv_folds:
        config.cv_folds = args.cv_folds
    if args.algorithms:
        config.algorithms = args.algorithms.split(',')
    if args.exclude:
        config.exclude_algorithms = args.exclude.split(',')
    if args.hpo_method:
        config.hpo_method = args.hpo_method
    if args.hpo_iter:
        config.hpo_n_iter = args.hpo_iter
    if args.scoring:
        config.scoring = args.scoring
    if args.ensemble:
        config.ensemble_method = args.ensemble
    
    # Validate configuration
    try:
        config.validate()
    except AssertionError as e:
        logger.error(f"Configuration validation failed: {e}")
        sys.exit(1)
    
    # Load data
    logger.info(f"Loading data from {args.data}")
    try:
        if args.data.endswith('.csv'):
            df = pd.read_csv(args.data)
        elif args.data.endswith('.parquet'):
            df = pd.read_parquet(args.data)
        elif args.data.endswith('.json'):
            df = pd.read_json(args.data)
        else:
            df = pd.read_csv(args.data)  # Try CSV as default
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        sys.exit(1)
    
    logger.info(f"Data loaded: {len(df)} rows, {len(df.columns)} columns")
    
    # Validate data
    validation = validate_data(df)
    if not validation['valid']:
        logger.warning(f"Data quality issues detected: {validation['issues']}")
        if not args.force:
            response = input("Continue anyway? (y/n): ")
            if response.lower() != 'y':
                logger.info("Training cancelled")
                sys.exit(0)
    
    # Check target column
    if args.target not in df.columns:
        logger.error(f"Target column '{args.target}' not found in data")
        logger.info(f"Available columns: {', '.join(df.columns)}")
        sys.exit(1)
    
    # Split features and target
    X = df.drop(columns=[args.target])
    y = df[args.target]
    
    logger.info(f"Features shape: {X.shape}")
    logger.info(f"Target distribution: {y.value_counts().to_dict() if y.nunique() < 20 else f'{y.describe().to_dict()}'}")
    
    # Create output directory
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    config.output_dir = str(output_path)
    
    # Save configuration
    config_path = output_path / "config.yaml"
    config.to_yaml(str(config_path))
    logger.info(f"Configuration saved to {config_path}")
    
    # Create and run orchestrator
    logger.info("Initializing AutoML orchestrator...")
    orchestrator = AutoMLOrchestrator(config)
    
    logger.info("Starting AutoML training...")
    logger.info(f"Task: {args.task}")
    logger.info(f"CV Folds: {config.cv_folds}")
    logger.info(f"HPO Method: {config.hpo_method}")
    logger.info(f"Algorithms: {config.algorithms[:5]}..." if len(config.algorithms) > 5 else config.algorithms)
    
    # Train models
    try:
        orchestrator.fit(X, y, task=args.task)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        if args.debug:
            raise
        sys.exit(1)
    
    # Print leaderboard
    leaderboard = orchestrator.get_leaderboard(top_n=args.top_k)
    
    print("\n" + "="*80)
    print("LEADERBOARD - Top {} Models".format(min(args.top_k, len(leaderboard))))
    print("="*80)
    print(leaderboard.to_string())
    print("="*80)
    
    # Save outputs
    logger.info(f"Saving results to {output_path}")
    
    # Save pipeline
    pipeline_path = output_path / "pipeline.joblib"
    orchestrator.save_pipeline(str(pipeline_path))
    logger.info(f"Pipeline saved to {pipeline_path}")
    
    # Save leaderboard
    leaderboard_path = output_path / "leaderboard.csv"
    full_leaderboard = orchestrator.get_leaderboard()
    full_leaderboard.to_csv(leaderboard_path, index=False)
    logger.info(f"Leaderboard saved to {leaderboard_path}")
    
    # Save feature importance
    if orchestrator.feature_importance:
        importance_path = output_path / "feature_importance.json"
        with open(importance_path, 'w') as f:
            json.dump(orchestrator.feature_importance, f, indent=2)
        logger.info(f"Feature importance saved to {importance_path}")
    
    # Save training predictions
    if config.save_predictions:
        logger.info("Generating training predictions...")
        train_predictions = orchestrator.predict(X)
        
        if orchestrator.task == 'classification' and hasattr(orchestrator.best_pipeline, 'predict_proba'):
            train_probabilities = orchestrator.predict_proba(X)
        else:
            train_probabilities = None
        
        predictions_path = output_path / "train_predictions.csv"
        save_predictions(train_predictions, predictions_path, probabilities=train_probabilities)
        logger.info(f"Training predictions saved to {predictions_path}")
        
        # Calculate and save metrics
        metrics = calculate_metrics(y.values, train_predictions, train_probabilities, orchestrator.task)
        metrics_path = output_path / "train_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Training metrics saved to {metrics_path}")
    
    # Generate report if requested
    if config.generate_report and args.report:
        logger.info("Generating HTML report...")
        # This would generate a comprehensive HTML report
        # Placeholder for now
        report_path = output_path / "report.html"
        with open(report_path, 'w') as f:
            f.write("<h1>AutoML Report</h1><p>Report generation not fully implemented yet.</p>")
        logger.info(f"Report saved to {report_path}")
    
    logger.info("="*80)
    logger.info("Training completed successfully!")
    logger.info(f"Best model: {leaderboard.iloc[0]['model'] if len(leaderboard) > 0 else 'None'}")
    logger.info(f"Best CV score: {leaderboard.iloc[0]['cv_score'] if len(leaderboard) > 0 else 'N/A':.4f}")
    logger.info("="*80)


def predict_cmd(args):
    """Make predictions using saved model."""
    logger.info("="*80)
    logger.info("AUTOML PLATFORM - PREDICTION MODE")
    logger.info("="*80)
    
    # Load pipeline
    logger.info(f"Loading pipeline from {args.model}")
    try:
        pipeline, metadata = load_pipeline(args.model)
    except Exception as e:
        logger.error(f"Failed to load pipeline: {e}")
        sys.exit(1)
    
    logger.info(f"Pipeline loaded: task={metadata.get('task', 'unknown')}, "
                f"model={metadata.get('best_model', 'unknown')}")
    
    # Load data
    logger.info(f"Loading data from {args.data}")
    try:
        if args.data.endswith('.csv'):
            df = pd.read_csv(args.data)
        elif args.data.endswith('.parquet'):
            df = pd.read_parquet(args.data)
        elif args.data.endswith('.json'):
            df = pd.read_json(args.data)
        else:
            df = pd.read_csv(args.data)  # Try CSV as default
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        sys.exit(1)
    
    logger.info(f"Data loaded: {len(df)} rows, {len(df.columns)} columns")
    
    # Make predictions
    logger.info("Generating predictions...")
    try:
        if args.batch_size:
            from automl_platform.inference import predict_batch
            predictions = predict_batch(pipeline, df, batch_size=args.batch_size)
        else:
            predictions = predict(pipeline, df)
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        if args.debug:
            raise
        sys.exit(1)
    
    # Get probabilities if available
    probabilities = None
    if args.proba and hasattr(pipeline, 'predict_proba'):
        logger.info("Generating probability predictions...")
        try:
            probabilities = predict_proba(pipeline, df)
        except Exception as e:
            logger.warning(f"Could not generate probabilities: {e}")
    
    # Save or display predictions
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        save_predictions(predictions, output_path, probabilities=probabilities)
        logger.info(f"Predictions saved to {output_path}")
        
        # Print summary
        if metadata.get('task') == 'classification':
            unique, counts = np.unique(predictions, return_counts=True)
            logger.info("Prediction distribution:")
            for val, count in zip(unique, counts):
                logger.info(f"  Class {val}: {count} ({count/len(predictions)*100:.1f}%)")
        else:
            logger.info(f"Prediction statistics:")
            logger.info(f"  Mean: {np.mean(predictions):.4f}")
            logger.info(f"  Std: {np.std(predictions):.4f}")
            logger.info(f"  Min: {np.min(predictions):.4f}")
            logger.info(f"  Max: {np.max(predictions):.4f}")
    else:
        # Display to console (first 20 predictions)
        print("\nPredictions:")
        print("-" * 40)
        for i, pred in enumerate(predictions[:20]):
            if probabilities is not None and len(probabilities.shape) > 1:
                prob_str = f" (prob: {probabilities[i].max():.3f})"
            else:
                prob_str = ""
            print(f"Sample {i+1}: {pred}{prob_str}")
        
        if len(predictions) > 20:
            print(f"... ({len(predictions) - 20} more predictions)")
    
    logger.info("="*80)
    logger.info("Prediction completed successfully!")
    logger.info("="*80)


def api(args):
    """Start API server."""
    logger.info("="*80)
    logger.info("AUTOML PLATFORM - API MODE")
    logger.info("="*80)
    
    try:
        import uvicorn
        from automl_platform.api.app import app
    except ImportError:
        logger.error("API dependencies not installed. Run: pip install automl-platform[api]")
        sys.exit(1)
    
    logger.info(f"Starting API server on {args.host}:{args.port}")
    logger.info(f"Workers: {args.workers}")
    logger.info(f"Reload: {args.reload}")
    
    uvicorn.run(
        "automl_platform.api.app:app",
        host=args.host,
        port=args.port,
        workers=args.workers if not args.reload else 1,
        reload=args.reload,
        log_level="info" if args.verbose else "warning"
    )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='AutoML Platform - Production-ready AutoML with no data leakage',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with default settings
  %(prog)s train --data data.csv --target target_column
  
  # Train with custom configuration
  %(prog)s train --data data.csv --target target_column --config config.yaml
  
  # Train with specific algorithms
  %(prog)s train --data data.csv --target target --algorithms RandomForest,XGBoost,LightGBM
  
  # Make predictions
  %(prog)s predict --model model.joblib --data test.csv --output predictions.csv
  
  # Start API server
  %(prog)s api --host 0.0.0.0 --port 8000
        """
    )
    
    parser.add_argument('--version', action='version', version='AutoML Platform 3.0.0')
    parser.add_argument('--verbose', '-v', action='count', default=1,
                       help='Increase verbosity (can be repeated: -v, -vv)')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Quiet mode (minimal output)')
    parser.add_argument('--log-file', help='Log to file')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode with full stack traces')
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train AutoML model')
    train_parser.add_argument('--data', required=True, help='Path to training data')
    train_parser.add_argument('--target', required=True, help='Target column name')
    train_parser.add_argument('--task', default='auto',
                            choices=['auto', 'classification', 'regression', 'timeseries'],
                            help='Task type (default: auto-detect)')
    train_parser.add_argument('--config', help='Path to configuration YAML file')
    train_parser.add_argument('--cv-folds', type=int, help='Number of CV folds')
    train_parser.add_argument('--algorithms', help='Comma-separated list of algorithms to test')
    train_parser.add_argument('--exclude', help='Comma-separated list of algorithms to exclude')
    train_parser.add_argument('--hpo-method', choices=['grid', 'random', 'optuna', 'none'],
                            help='Hyperparameter optimization method')
    train_parser.add_argument('--hpo-iter', type=int, help='Number of HPO iterations')
    train_parser.add_argument('--scoring', help='Scoring metric')
    train_parser.add_argument('--ensemble', choices=['voting', 'stacking', 'none'],
                            help='Ensemble method')
    train_parser.add_argument('--output', default='./automl_output',
                            help='Output directory (default: ./automl_output)')
    train_parser.add_argument('--top-k', type=int, default=10,
                            help='Show top K models in leaderboard')
    train_parser.add_argument('--report', action='store_true',
                            help='Generate HTML report')
    train_parser.add_argument('--force', action='store_true',
                            help='Force training even with data quality issues')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Make predictions')
    predict_parser.add_argument('--model', required=True, help='Path to saved model')
    predict_parser.add_argument('--data', required=True, help='Path to data for prediction')
    predict_parser.add_argument('--output', help='Output file path (if not specified, prints to console)')
    predict_parser.add_argument('--proba', action='store_true',
                              help='Include probability predictions')
    predict_parser.add_argument('--batch-size', type=int,
                              help='Batch size for large datasets')
    
    # API command
    api_parser = subparsers.add_parser('api', help='Start API server')
    api_parser.add_argument('--host', default='0.0.0.0', help='API host')
    api_parser.add_argument('--port', type=int, default=8000, help='API port')
    api_parser.add_argument('--workers', type=int, default=1, help='Number of workers')
    api_parser.add_argument('--reload', action='store_true', help='Auto-reload on code changes')
    
    args = parser.parse_args()
    
    # Setup logging
    if args.quiet:
        args.verbose = 0
    setup_logging(args.verbose, args.log_file)
    
    # Execute command
    if args.command == 'train':
        train(args)
    elif args.command == 'predict':
        predict_cmd(args)
    elif args.command == 'api':
        api(args)
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == '__main__':
    main()
