"""
Test Suite for AutoML Platform
===============================

Comprehensive test coverage for all AutoML platform modules including:
- A/B testing and model comparison
- API endpoints and services
- Data preparation and quality assessment
- Model training and ensemble methods
- Model export and deployment
- Incremental and streaming ML
- Monitoring and retraining
- Job scheduling and orchestration
"""

import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Test module imports for availability checking
MODULES_STATUS = {
    # Core modules
    'ab_testing': False,
    'api': False,
    'billing': False,
    'streaming': False,
    'config': False,
    'orchestrator': False,
    'scheduler': False,
    
    # Data modules
    'data_prep': False,
    'data_quality_agent': False,
    'metrics': False,
    
    # Model modules
    'model_selection': False,
    'ensemble': False,
    'tabnet_sklearn': False,
    'incremental_learning': False,
    
    # MLOps modules
    'mlflow_registry': False,
    'monitoring': False,
    'retraining_service': False,
    'export_service': False,
    
    # LLM modules
    'prompts': False,
}

# Check module availability
def check_module_availability():
    """Check which AutoML platform modules are available."""
    # Core modules
    try:
        from automl_platform import ab_testing
        MODULES_STATUS['ab_testing'] = True
    except ImportError:
        pass
    
    try:
        from automl_platform.api import api, billing, streaming
        MODULES_STATUS['api'] = True
        MODULES_STATUS['billing'] = True
        MODULES_STATUS['streaming'] = True
    except ImportError:
        pass
    
    try:
        from automl_platform import config
        MODULES_STATUS['config'] = True
    except ImportError:
        pass
    
    try:
        from automl_platform import orchestrator
        MODULES_STATUS['orchestrator'] = True
    except ImportError:
        pass
    
    try:
        from automl_platform import scheduler
        MODULES_STATUS['scheduler'] = True
    except ImportError:
        pass
    
    # Data modules
    try:
        from automl_platform import data_prep
        MODULES_STATUS['data_prep'] = True
    except ImportError:
        pass
    
    try:
        from automl_platform import data_quality_agent
        MODULES_STATUS['data_quality_agent'] = True
    except ImportError:
        pass
    
    try:
        from automl_platform import metrics
        MODULES_STATUS['metrics'] = True
    except ImportError:
        pass
    
    # Model modules
    try:
        from automl_platform import model_selection
        MODULES_STATUS['model_selection'] = True
    except ImportError:
        pass
    
    try:
        from automl_platform import ensemble
        MODULES_STATUS['ensemble'] = True
    except ImportError:
        pass
    
    try:
        from automl_platform import tabnet_sklearn
        MODULES_STATUS['tabnet_sklearn'] = True
    except ImportError:
        pass
    
    try:
        from automl_platform import incremental_learning
        MODULES_STATUS['incremental_learning'] = True
    except ImportError:
        pass
    
    # MLOps modules
    try:
        from automl_platform import mlflow_registry
        MODULES_STATUS['mlflow_registry'] = True
    except ImportError:
        pass
    
    try:
        from automl_platform import monitoring
        MODULES_STATUS['monitoring'] = True
    except ImportError:
        pass
    
    try:
        from automl_platform import retraining_service
        MODULES_STATUS['retraining_service'] = True
    except ImportError:
        pass
    
    try:
        from automl_platform import export_service
        MODULES_STATUS['export_service'] = True
    except ImportError:
        pass
    
    # LLM modules
    try:
        from automl_platform import prompts
        MODULES_STATUS['prompts'] = True
    except ImportError:
        pass
    
    return MODULES_STATUS

# Test discovery helpers
def get_test_modules():
    """Get list of all test modules."""
    test_modules = [
        'test_ab_testing',
        'test_api',
        'test_billing',
        'test_data_prep',
        'test_data_quality_agent',
        'test_ensemble',
        'test_export_service',
        'test_incremental_learning',
        'test_metrics',
        'test_mlflow_registry',
        'test_model_selection',
        'test_monitoring',
        'test_orchestrator',
        'test_prompts',
        'test_retraining_service',
        'test_scheduler',
        'test_streaming',
        'test_tabnet_sklearn',
    ]
    return test_modules

def run_all_tests(verbose=False, coverage=False):
    """
    Run all available tests.
    
    Args:
        verbose: Enable verbose output
        coverage: Enable coverage reporting
    
    Returns:
        Test results summary
    """
    import pytest
    
    args = [__file__.replace('__init__.py', '')]
    
    if verbose:
        args.append('-v')
    
    if coverage:
        args.extend(['--cov=automl_platform', '--cov-report=term-missing'])
    
    # Check module availability first
    check_module_availability()
    
    # Print module status
    print("\nAutoML Platform Module Status:")
    print("-" * 40)
    for module, available in MODULES_STATUS.items():
        status = "✓" if available else "✗"
        print(f"{status} {module:20} {'Available' if available else 'Not Available'}")
    print("-" * 40)
    
    # Run tests
    return pytest.main(args)

def run_specific_tests(test_module, verbose=False):
    """
    Run tests for a specific module.
    
    Args:
        test_module: Name of the test module (e.g., 'test_api')
        verbose: Enable verbose output
    
    Returns:
        Test results
    """
    import pytest
    
    test_file = Path(__file__).parent / f"{test_module}.py"
    
    if not test_file.exists():
        raise FileNotFoundError(f"Test file {test_module}.py not found")
    
    args = [str(test_file)]
    
    if verbose:
        args.append('-v')
    
    return pytest.main(args)

# Test fixtures and utilities
def create_test_data(task='classification', n_samples=100, n_features=10):
    """
    Create synthetic test data.
    
    Args:
        task: 'classification' or 'regression'
        n_samples: Number of samples
        n_features: Number of features
    
    Returns:
        X, y arrays
    """
    from sklearn.datasets import make_classification, make_regression
    
    if task == 'classification':
        return make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_features - 2,
            n_redundant=2,
            random_state=42
        )
    else:
        return make_regression(
            n_samples=n_samples,
            n_features=n_features,
            noise=0.1,
            random_state=42
        )

# Export key items
__all__ = [
    'MODULES_STATUS',
    'check_module_availability',
    'get_test_modules',
    'run_all_tests',
    'run_specific_tests',
    'create_test_data',
]

# Run module check on import
check_module_availability()
