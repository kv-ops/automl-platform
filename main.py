#!/usr/bin/env python3
"""
Main entry point for AutoML Platform CLI.
Provides command-line interface for training and prediction with template support and expert mode.
Version: 3.2.1
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
from tabulate import tabulate

# Import platform modules
from automl_platform.config import AutoMLConfig, load_config
from automl_platform.orchestrator import AutoMLOrchestrator
from automl_platform.inference import load_pipeline, predict, predict_proba, save_predictions, predict_batch
from automl_platform.data_prep import validate_data
from automl_platform.metrics import calculate_metrics
from automl_platform.template_loader import TemplateLoader  # Moved from templates/

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
    """Train AutoML model with optional template support and expert mode."""
    logger.info("="*80)
    logger.info("AUTOML PLATFORM - TRAINING MODE - v3.2.1")
    if args.expert:
        logger.info("🎓 EXPERT MODE ENABLED - All advanced options available")
    else:
        logger.info("🚀 SIMPLIFIED MODE - Using optimized defaults")
    logger.info("="*80)
    
    # Initialize template loader
    template_loader = TemplateLoader()
    
    # Load configuration
    if args.template:
        # Load from template
        logger.info(f"Loading template: {args.template}")
        try:
            config = template_loader.apply_template(args.template)
            logger.info(f"Template '{args.template}' loaded successfully")
            
            # Show template info
            template_info = template_loader.get_template_info(args.template)
            logger.info(f"Template description: {template_info['description']}")
            logger.info(f"Template task: {template_info['task']}")
            logger.info(f"Template algorithms: {', '.join(template_info['algorithms'][:5])}")
            
            # Validate template (with error handling)
            try:
                validation = template_loader.validate_template(args.template)
            except AttributeError:
                # If validate_template method doesn't exist, create default validation
                validation = {"valid": True, "errors": [], "warnings": []}
            
            if not validation.get('valid', True):
                logger.warning(f"Template validation issues: {validation.get('errors', [])}")
                if not args.force:
                    response = input("Continue anyway? (y/n): ")
                    if response.lower() != 'y':
                        logger.info("Training cancelled")
                        sys.exit(0)
            
        except ValueError as e:
            logger.error(f"Failed to load template: {e}")
            
            # List available templates
            logger.info("\nAvailable templates:")
            templates = template_loader.list_templates()
            for t in templates:
                logger.info(f"  - {t['name']}: {t['description']}")
            sys.exit(1)
            
    elif args.config:
        logger.info(f"Loading configuration from {args.config}")
        config = load_config(args.config, expert_mode=args.expert)
        
        # Apply template on top of config if specified
        if args.template_override:
            logger.info(f"Applying template override: {args.template_override}")
            try:
                config = template_loader.apply_template(args.template_override, config)
            except ValueError as e:
                logger.error(f"Failed to apply template override: {e}")
                sys.exit(1)
    else:
        logger.info("Using default configuration")
        config = AutoMLConfig(expert_mode=args.expert)
    
    # Set expert mode from command line
    config.expert_mode = args.expert
    
    # Handle expert mode vs simplified mode parameters
    if args.expert:
        # In expert mode, allow all command-line overrides
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
        if args.n_workers:
            config.worker.max_workers = args.n_workers
        if args.gpu:
            config.worker.enable_gpu_queue = True
            config.worker.gpu_workers = args.gpu_workers if args.gpu_workers else 1
    else:
        # In simplified mode, use optimized defaults
        logger.info("Using simplified configuration with optimized defaults:")
        
        # Apply simplified settings
        simplified_hpo = config.get_simplified_hpo_config()
        config.hpo_method = simplified_hpo['method']
        config.hpo_n_iter = simplified_hpo['n_iter']
        config.hpo_time_budget = simplified_hpo['time_budget']
        config.early_stopping_rounds = simplified_hpo['early_stopping_rounds']
        
        # Use simplified algorithm list
        config.algorithms = config.get_simplified_algorithms(task=args.task)
        
        # Set simplified defaults
        config.cv_folds = 3  # Faster validation
        config.ensemble_method = "voting"  # Simpler ensemble
        config.worker.max_workers = 2  # Limited parallelism
        
        # Show simplified configuration
        logger.info(f"  • Algorithms: {', '.join(config.algorithms)}")
        logger.info(f"  • HPO iterations: {config.hpo_n_iter}")
        logger.info(f"  • Time budget: {config.hpo_time_budget}s")
        logger.info(f"  • CV folds: {config.cv_folds}")
        logger.info(f"  • Workers: {config.worker.max_workers}")
        
        # Override with basic parameters if provided
        if args.scoring:
            config.scoring = args.scoring
    
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
    
    # Save configuration (including template and expert mode info)
    config_path = output_path / "config.yaml"
    config.to_yaml(str(config_path))
    logger.info(f"Configuration saved to {config_path}")
    
    # Save mode information
    mode_info = {
        "expert_mode": config.expert_mode,
        "simplified_algorithms": config.get_simplified_algorithms(task=args.task) if not config.expert_mode else None,
        "simplified_hpo": config.get_simplified_hpo_config() if not config.expert_mode else None,
        "version": "3.2.1"
    }
    mode_info_path = output_path / "mode_info.json"
    with open(mode_info_path, 'w') as f:
        json.dump(mode_info, f, indent=2)
    logger.info(f"Mode information saved to {mode_info_path}")
    
    # Save template info if used
    if args.template:
        template_info_path = output_path / "template_info.json"
        template_info = template_loader.get_template_info(args.template)
        with open(template_info_path, 'w') as f:
            json.dump(template_info, f, indent=2)
        logger.info(f"Template information saved to {template_info_path}")
    
    # Create and run orchestrator
    logger.info("Initializing AutoML orchestrator...")
    orchestrator = AutoMLOrchestrator(config)
    
    # Display training configuration
    logger.info("="*80)
    logger.info("TRAINING CONFIGURATION")
    logger.info("="*80)
    logger.info(f"Mode: {'Expert' if config.expert_mode else 'Simplified'}")
    logger.info(f"Task: {args.task}")
    if args.template:
        logger.info(f"Template: {args.template}")
    logger.info(f"CV Folds: {config.cv_folds}")
    if config.expert_mode:
        logger.info(f"HPO Method: {config.hpo_method}")
        logger.info(f"HPO Iterations: {config.hpo_n_iter}")
    logger.info(f"Scoring Metric: {config.scoring}")
    if config.expert_mode:
        logger.info(f"Ensemble Method: {config.ensemble_method}")
        logger.info(f"Workers: {config.worker.max_workers}")
        if config.worker.enable_gpu_queue:
            logger.info(f"GPU Workers: {config.worker.gpu_workers}")
    logger.info(f"Algorithms ({len(config.algorithms)}): {', '.join(config.algorithms[:5])}...")
    logger.info("="*80)
    
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
    
    # Save metadata
    metadata = {
        "task": orchestrator.task,
        "best_model": leaderboard.iloc[0]['model'] if len(leaderboard) > 0 else "Unknown",
        "expert_mode": config.expert_mode,
        "template_used": args.template if args.template else None,
        "version": "3.2.1"
    }
    metadata_path = output_path / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
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
            f.write(f"<h1>AutoML Report v3.2.1</h1><p>Report generation not fully implemented yet.</p>")
        logger.info(f"Report saved to {report_path}")
    
    logger.info("="*80)
    logger.info("Training completed successfully!")
    if leaderboard is not None and len(leaderboard) > 0:
        logger.info(f"Best model: {leaderboard.iloc[0]['model']}")
        logger.info(f"Best CV score: {leaderboard.iloc[0]['cv_score']:.4f}")
    if not config.expert_mode:
        logger.info("\n💡 TIP: Use --expert flag to access advanced configuration options")
    logger.info("="*80)


def list_templates(args):
    """List available templates."""
    logger.info("="*80)
    logger.info("AUTOML PLATFORM - AVAILABLE TEMPLATES - v3.2.1")
    logger.info("="*80)
    
    template_loader = TemplateLoader()
    
    # Get templates with optional filtering
    templates = template_loader.list_templates(
        task=args.task,
        tags=args.tags.split(',') if args.tags else None
    )
    
    if not templates:
        logger.info("No templates found matching the criteria.")
        return
    
    # Prepare table data
    table_data = []
    for t in templates:
        table_data.append([
            t['name'],
            t['task'],
            t['description'][:50] + '...' if len(t['description']) > 50 else t['description'],
            ', '.join(t['tags'][:3]),
            ', '.join(t['algorithms'][:3]) + ('...' if len(t['algorithms']) > 3 else ''),
            f"{t.get('estimated_time', 30)} min"
        ])
    
    # Print table
    headers = ["Name", "Task", "Description", "Tags", "Algorithms", "Est. Time"]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    print(f"\nTotal templates: {len(templates)}")
    print("\nTo use a template: python main.py train --template <name> --data <file> --target <column>")
    print("To view template details: python main.py template-info <name>")


def template_info(args):
    """Show detailed information about a template."""
    logger.info("="*80)
    logger.info(f"TEMPLATE INFORMATION: {args.name} - v3.2.1")
    logger.info("="*80)
    
    template_loader = TemplateLoader()
    
    try:
        info = template_loader.get_template_info(args.name)
    except ValueError as e:
        logger.error(e)
        logger.info("\nUse 'python main.py list-templates' to see available templates")
        sys.exit(1)
    
    # Display template information
    print(f"\nTemplate: {info['name']}")
    print(f"Version: {info['version']}")
    print(f"Author: {info['author']}")
    print("-" * 60)
    print(f"Description: {info['description']}")
    print(f"Task Type: {info['task']}")
    print(f"Tags: {', '.join(info['tags'])}")
    print("-" * 60)
    
    print("\nConfiguration:")
    print(f"  Algorithms ({len(info['algorithms'])}): {', '.join(info['algorithms'][:5])}")
    if len(info['algorithms']) > 5:
        print(f"    ... and {len(info['algorithms']) - 5} more")
    
    if info['excluded_algorithms']:
        print(f"  Excluded: {', '.join(info['excluded_algorithms'])}")
    
    print(f"  HPO Method: {info['hpo_method']}")
    print(f"  CV Folds: {info['cv_folds']}")
    print(f"  Ensemble: {info['ensemble_method']}")
    
    if info['metrics']:
        print(f"  Metrics: {', '.join(info['metrics'])}")
    
    print("\nFeatures:")
    for feature, enabled in info['features'].items():
        status = "✓" if enabled else "✗"
        print(f"  [{status}] {feature.replace('_', ' ').title()}")
    
    if info.get('has_business_rules'):
        print("\nBusiness Rules:")
        if 'probability_threshold' in info:
            print(f"  Probability Threshold: {info['probability_threshold']}")
        if 'cost_sensitive' in info:
            print(f"  Cost-Sensitive Learning: Enabled")
    
    # Validate template (with error handling)
    try:
        validation = template_loader.validate_template(args.name)
    except AttributeError:
        # If validate_template method doesn't exist, create default validation
        validation = {"valid": True, "errors": [], "warnings": []}
    
    if not validation.get('valid', True):
        print("\n⚠ Template Validation Issues:")
        for error in validation.get('errors', []):
            print(f"  ✗ {error}")
    if validation.get('warnings'):
        print("\nWarnings:")
        for warning in validation['warnings']:
            print(f"  ⚠ {warning}")
    
    if args.export:
        # Export template to file
        export_path = Path(args.export)
        export_format = export_path.suffix[1:] if export_path.suffix else 'yaml'
        
        try:
            content = template_loader.export_template(args.name, format=export_format)
            with open(export_path, 'w') as f:
                f.write(content)
            print(f"\nTemplate exported to: {export_path}")
        except Exception as e:
            logger.error(f"Failed to export template: {e}")


def create_template(args):
    """Create a custom template from existing configuration."""
    logger.info("="*80)
    logger.info("CREATE CUSTOM TEMPLATE - v3.2.1")
    logger.info("="*80)
    
    template_loader = TemplateLoader()
    
    # Load base configuration
    if args.from_config:
        logger.info(f"Loading configuration from {args.from_config}")
        config = AutoMLConfig.from_yaml(args.from_config)
        template_config = config.to_dict()
    elif args.from_template:
        logger.info(f"Loading from template {args.from_template}")
        try:
            template_config = template_loader.load_template(args.from_template)
        except ValueError as e:
            logger.error(e)
            sys.exit(1)
    else:
        # Create minimal template
        template_config = {
            "task": "auto",
            "algorithms": ["XGBoost", "RandomForest", "LogisticRegression"],
            "cv": {"n_folds": 5},
            "hpo": {"method": "optuna", "n_iter": 20}
        }
    
    # Apply modifications if provided
    if args.set:
        for setting in args.set:
            key, value = setting.split('=', 1)
            keys = key.split('.')
            
            # Navigate to nested key and set value
            current = template_config
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
            
            # Try to parse value as JSON, otherwise use as string
            try:
                current[keys[-1]] = json.loads(value)
            except json.JSONDecodeError:
                current[keys[-1]] = value
    
    # Create the template
    try:
        template = template_loader.create_custom_template(
            name=args.name,
            config=template_config,
            description=args.description or f"Custom template created from {'config' if args.from_config else 'template'}",
            tags=args.tags.split(',') if args.tags else [],
            save=not args.no_save
        )
        
        logger.info(f"Template '{args.name}' created successfully")
        
        if not args.no_save:
            logger.info(f"Template saved to: automl_platform/templates/use_cases/{args.name}.yaml")
        
        # Show template info
        info = template_loader.get_template_info(args.name)
        print(f"\nCreated template: {info['name']}")
        print(f"Description: {info['description']}")
        print(f"Task: {info['task']}")
        print(f"Algorithms: {', '.join(info['algorithms'][:5])}")
        
    except Exception as e:
        logger.error(f"Failed to create template: {e}")
        sys.exit(1)


def predict_cmd(args):
    """Make predictions using saved model."""
    logger.info("="*80)
    logger.info("AUTOML PLATFORM - PREDICTION MODE - v3.2.1")
    if args.expert:
        logger.info("🎓 EXPERT MODE ENABLED")
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
    
    # Check if model was trained with a template
    if 'template_used' in metadata:
        logger.info(f"Model was trained using template: {metadata['template_used']}")
    
    # Check if model was trained in expert mode
    if 'expert_mode' in metadata:
        logger.info(f"Model was trained in {'expert' if metadata['expert_mode'] else 'simplified'} mode")
    
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
        if args.expert and args.batch_size:
            predictions = predict_batch(pipeline, df, batch_size=args.batch_size)
            logger.info(f"Using batch prediction with size: {args.batch_size}")
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
    logger.info("AUTOML PLATFORM - API MODE - v3.2.1")
    if args.expert:
        logger.info("🎓 EXPERT MODE ENABLED - All API endpoints available")
    else:
        logger.info("🚀 SIMPLIFIED MODE - Core API endpoints only")
    logger.info("="*80)
    
    try:
        import uvicorn
        from automl_platform.api.api import app
        
        # Set expert mode in environment for API
        import os
        os.environ["AUTOML_EXPERT_MODE"] = "true" if args.expert else "false"
        
    except ImportError:
        logger.error("API dependencies not installed. Run: pip install automl-platform[api]")
        sys.exit(1)
    
    logger.info(f"Starting API server on {args.host}:{args.port}")
    if args.expert:
        logger.info(f"Workers: {args.workers}")
        logger.info(f"Reload: {args.reload}")
    
    uvicorn.run(
        "automl_platform.api.api:app",
        host=args.host,
        port=args.port,
        workers=args.workers if not args.reload else 1,
        reload=args.reload,
        log_level="info" if args.verbose else "warning"
    )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='AutoML Platform v3.2.1 - Production-ready AutoML with template support',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with simplified mode (default)
  %(prog)s train --data data.csv --target churn
  
  # Train with expert mode (all options available)
  %(prog)s train --expert --data data.csv --target churn --algorithms XGBoost,LightGBM --hpo-iter 100
  
  # Train with a template
  %(prog)s train --template customer_churn --data data.csv --target churn
  
  # List available templates
  %(prog)s list-templates
  
  # Get template information
  %(prog)s template-info customer_churn
  
  # Create custom template
  %(prog)s create-template my_template --from-config config.yaml --description "My custom template"
  
  # Make predictions
  %(prog)s predict --model model.joblib --data test.csv --output predictions.csv
  
  # Start API server
  %(prog)s api --host 0.0.0.0 --port 8000
  
  # Start API server in expert mode
  %(prog)s api --expert --host 0.0.0.0 --port 8000 --workers 4
        """
    )
    
    parser.add_argument('--version', action='version', version='AutoML Platform 3.2.1')
    parser.add_argument('--verbose', '-v', action='count', default=1,
                       help='Increase verbosity (can be repeated: -v, -vv)')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Quiet mode (minimal output)')
    parser.add_argument('--log-file', help='Log to file')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode with full stack traces')
    parser.add_argument('--expert', action='store_true',
                       help='Enable expert mode to access all advanced options')
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train AutoML model')
    train_parser.add_argument('--data', required=True, help='Path to training data')
    train_parser.add_argument('--target', required=True, help='Target column name')
    train_parser.add_argument('--task', default='auto',
                            choices=['auto', 'classification', 'regression', 'timeseries'],
                            help='Task type (default: auto-detect)')
    train_parser.add_argument('--template', help='Use a predefined template')
    train_parser.add_argument('--template-override', help='Apply template on top of existing config')
    train_parser.add_argument('--config', help='Path to configuration YAML file')
    
    # Basic options (always visible)
    train_parser.add_argument('--scoring', help='Scoring metric')
    train_parser.add_argument('--output', default='./automl_output',
                            help='Output directory (default: ./automl_output)')
    train_parser.add_argument('--top-k', type=int, default=10,
                            help='Show top K models in leaderboard')
    train_parser.add_argument('--report', action='store_true',
                            help='Generate HTML report')
    train_parser.add_argument('--force', action='store_true',
                            help='Force training even with data quality issues')
    
    # Expert options (only meaningful in expert mode)
    expert_group = train_parser.add_argument_group('expert options', 
                                                   'Advanced options (requires --expert flag)')
    expert_group.add_argument('--cv-folds', type=int, help='Number of CV folds')
    expert_group.add_argument('--algorithms', help='Comma-separated list of algorithms to test')
    expert_group.add_argument('--exclude', help='Comma-separated list of algorithms to exclude')
    expert_group.add_argument('--hpo-method', choices=['grid', 'random', 'optuna', 'none'],
                            help='Hyperparameter optimization method')
    expert_group.add_argument('--hpo-iter', type=int, help='Number of HPO iterations')
    expert_group.add_argument('--ensemble', choices=['voting', 'stacking', 'none'],
                            help='Ensemble method')
    expert_group.add_argument('--n-workers', type=int, help='Number of parallel workers')
    expert_group.add_argument('--gpu', action='store_true', help='Enable GPU acceleration')
    expert_group.add_argument('--gpu-workers', type=int, help='Number of GPU workers')
    
    # Template commands
    template_list_parser = subparsers.add_parser('list-templates', 
                                                 help='List available templates')
    template_list_parser.add_argument('--task', help='Filter by task type')
    template_list_parser.add_argument('--tags', help='Filter by tags (comma-separated)')
    
    template_info_parser = subparsers.add_parser('template-info',
                                                 help='Show template details')
    template_info_parser.add_argument('name', help='Template name')
    template_info_parser.add_argument('--export', help='Export template to file')
    
    create_template_parser = subparsers.add_parser('create-template',
                                                   help='Create custom template')
    create_template_parser.add_argument('name', help='Template name')
    create_template_parser.add_argument('--from-config', help='Create from config file')
    create_template_parser.add_argument('--from-template', help='Create from existing template')
    create_template_parser.add_argument('--description', help='Template description')
    create_template_parser.add_argument('--tags', help='Template tags (comma-separated)')
    create_template_parser.add_argument('--set', action='append',
                                       help='Set config values (e.g., --set hpo.n_iter=50)')
    create_template_parser.add_argument('--no-save', action='store_true',
                                       help='Do not save template to file')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Make predictions')
    predict_parser.add_argument('--model', required=True, help='Path to saved model')
    predict_parser.add_argument('--data', required=True, help='Path to data for prediction')
    predict_parser.add_argument('--output', help='Output file path (if not specified, prints to console)')
    predict_parser.add_argument('--proba', action='store_true',
                              help='Include probability predictions')
    
    # Expert option for prediction
    predict_expert_group = predict_parser.add_argument_group('expert options')
    predict_expert_group.add_argument('--batch-size', type=int,
                              help='Batch size for large datasets (expert mode only)')
    
    # API command
    api_parser = subparsers.add_parser('api', help='Start API server')
    api_parser.add_argument('--host', default='0.0.0.0', help='API host')
    api_parser.add_argument('--port', type=int, default=8000, help='API port')
    
    # Expert options for API
    api_expert_group = api_parser.add_argument_group('expert options')
    api_expert_group.add_argument('--workers', type=int, default=1, help='Number of workers')
    api_expert_group.add_argument('--reload', action='store_true', help='Auto-reload on code changes')
    
    args = parser.parse_args()
    
    # Setup logging
    if args.quiet:
        args.verbose = 0
    setup_logging(args.verbose, args.log_file)
    
    # Show expert mode warning for advanced options used without --expert flag
    if hasattr(args, 'expert') and not args.expert:
        expert_options_used = []
        
        if args.command == 'train':
            if args.cv_folds:
                expert_options_used.append('--cv-folds')
            if args.algorithms:
                expert_options_used.append('--algorithms')
            if args.exclude:
                expert_options_used.append('--exclude')
            if args.hpo_method:
                expert_options_used.append('--hpo-method')
            if args.hpo_iter:
                expert_options_used.append('--hpo-iter')
            if args.ensemble:
                expert_options_used.append('--ensemble')
            if hasattr(args, 'n_workers') and args.n_workers:
                expert_options_used.append('--n-workers')
            if hasattr(args, 'gpu') and args.gpu:
                expert_options_used.append('--gpu')
        elif args.command == 'predict':
            if hasattr(args, 'batch_size') and args.batch_size:
                expert_options_used.append('--batch-size')
        elif args.command == 'api':
            if hasattr(args, 'workers') and args.workers != 1:
                expert_options_used.append('--workers')
            if hasattr(args, 'reload') and args.reload:
                expert_options_used.append('--reload')
        
        if expert_options_used:
            logger.warning("="*80)
            logger.warning("⚠️  EXPERT OPTIONS IGNORED")
            logger.warning(f"The following options require --expert flag: {', '.join(expert_options_used)}")
            logger.warning("Add --expert to enable these advanced options")
            logger.warning("="*80)
    
    # Execute command
    if args.command == 'train':
        train(args)
    elif args.command == 'list-templates':
        list_templates(args)
    elif args.command == 'template-info':
        template_info(args)
    elif args.command == 'create-template':
        create_template(args)
    elif args.command == 'predict':
        predict_cmd(args)
    elif args.command == 'api':
        api(args)
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == '__main__':
    main()
