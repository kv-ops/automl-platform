"""
Test Suite for AutoML Platform
===============================

Comprehensive test coverage for all AutoML platform modules including:
- Core infrastructure services (health monitoring, service registry, configuration)
- A/B testing and model comparison
- API endpoints and services
- Authentication and authorization (including SSO and RGPD endpoints)
- Billing and subscription management
- Data preparation and quality assessment
- Model training and ensemble methods
- Model export and deployment
- Incremental and streaming ML
- Monitoring and retraining
- Job scheduling and orchestration
- Storage and feature store
- Prompts and LLM integration
- UI components (Streamlit dashboard)
"""

import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Test module imports for availability checking
MODULES_STATUS = {
    # Core infrastructure modules
    'core': False,
    'health_monitor': False,
    'service_registry': False,
    'config_manager': False,
    
    # Core modules
    'ab_testing': False,
    'api': False,
    'auth': False,
    'auth_endpoints': False,  # Added for SSO and RGPD endpoints
    'billing': False,
    'streaming': False,
    'config': False,
    'orchestrator': False,
    'scheduler': False,
    'storage': False,
    
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
    'mlops_service': False,
    'mlops_endpoints': False,
    'monitoring': False,
    'retraining_service': False,
    'export_service': False,
    
    # LLM modules
    'prompts': False,
    
    # UI modules
    'ui_streamlit': False,
    
    # Security and compliance modules
    'sso_service': False,
    'audit_service': False,
    'rgpd_compliance_service': False,
}

# Check module availability
def check_module_availability():
    """Check which AutoML platform modules are available."""
    # Core infrastructure modules
    try:
        from automl_platform import core
        MODULES_STATUS['core'] = True
    except ImportError:
        pass
    
    try:
        from automl_platform.core import health_monitor
        MODULES_STATUS['health_monitor'] = True
    except ImportError:
        pass
    
    try:
        from automl_platform.core import service_registry
        MODULES_STATUS['service_registry'] = True
    except ImportError:
        pass
    
    try:
        from automl_platform.core import config_manager
        MODULES_STATUS['config_manager'] = True
    except ImportError:
        pass
    
    # Core modules
    try:
        from automl_platform import ab_testing
        MODULES_STATUS['ab_testing'] = True
    except ImportError:
        pass
    
    try:
        from automl_platform.api import api
        MODULES_STATUS['api'] = True
    except ImportError:
        pass
    
    try:
        from automl_platform import auth
        MODULES_STATUS['auth'] = True
    except ImportError:
        pass
    
    try:
        from automl_platform.api import auth_endpoints
        MODULES_STATUS['auth_endpoints'] = True
    except ImportError:
        pass
    
    try:
        from automl_platform.api import billing
        MODULES_STATUS['billing'] = True
    except ImportError:
        pass
    
    try:
        from automl_platform.api import streaming
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
    
    try:
        from automl_platform import storage
        MODULES_STATUS['storage'] = True
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
        from automl_platform import mlops_service
        MODULES_STATUS['mlops_service'] = True
    except ImportError:
        pass
    
    try:
        from automl_platform.api import mlops_endpoints
        MODULES_STATUS['mlops_endpoints'] = True
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
    
    # UI modules
    try:
        from automl_platform.ui import streamlit_app
        MODULES_STATUS['ui_streamlit'] = True
    except ImportError:
        pass
    
    # Security and compliance modules
    try:
        from automl_platform import sso_service
        MODULES_STATUS['sso_service'] = True
    except ImportError:
        pass
    
    try:
        from automl_platform import audit_service
        MODULES_STATUS['audit_service'] = True
    except ImportError:
        pass
    
    try:
        from automl_platform import rgpd_compliance_service
        MODULES_STATUS['rgpd_compliance_service'] = True
    except ImportError:
        pass
    
    return MODULES_STATUS

# Test discovery helpers
def get_test_modules():
    """Get list of all test modules that exist in the tests directory."""
    # List of all test files that actually exist (in alphabetical order)
    test_modules = [
        'test_ab_testing',           # A/B testing framework
        'test_api',                  # API endpoints
        'test_auth',                 # Authentication and authorization
        'test_auth_endpoints',       # SSO and RGPD API endpoints
        'test_billing',              # Billing and subscription management
        'test_config',               # Configuration management (legacy)
        'test_config_manager',       # Core configuration manager
        'test_data_prep',            # Data preparation
        'test_data_quality_agent',   # Data quality agent with LLM
        'test_ensemble',             # Ensemble methods
        'test_export_service',       # Model export service
        'test_health_monitor',       # Core health monitoring
        'test_incremental_learning', # Incremental/online learning
        'test_metrics',              # Metrics calculation
        'test_mlflow_registry',      # MLflow registry integration
        'test_mlops_endpoints',      # MLOps API endpoints
        'test_mlops_service',        # MLOps service layer
        'test_model_selection',      # Model selection and tuning
        'test_monitoring',           # Model monitoring
        'test_orchestrator',         # AutoML orchestrator
        'test_prompts',              # LLM prompts
        'test_retraining_service',   # Automated retraining
        'test_scheduler',            # Job scheduler
        'test_service_registry',     # Core service registry
        'test_storage',              # Storage backends
        'test_streaming',            # Streaming ML
        'test_tabnet_sklearn',       # TabNet implementation
        'test_ui_streamlit',         # Streamlit UI dashboard
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
        print(f"{status} {module:25} {'Available' if available else 'Not Available'}")
    print("-" * 40)
    
    # Print test files status
    print("\nTest Files Status:")
    print("-" * 40)
    test_dir = Path(__file__).parent
    for test_module in get_test_modules():
        test_file = test_dir / f"{test_module}.py"
        exists = test_file.exists()
        status = "✓" if exists else "✗"
        print(f"{status} {test_module:25} {'Found' if exists else 'Not Found'}")
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

def run_test_category(category, verbose=False):
    """
    Run tests for a specific category.
    
    Args:
        category: Category of tests to run
                 Options: 'core', 'infrastructure', 'data', 'models', 'mlops', 'api', 'ui', 'auth', 'compliance'
        verbose: Enable verbose output
    
    Returns:
        Test results
    """
    import pytest
    
    category_tests = {
        'core': [
            'test_config',
            'test_orchestrator',
            'test_scheduler',
            'test_storage',
        ],
        'infrastructure': [
            'test_health_monitor',
            'test_service_registry',
            'test_config_manager',
        ],
        'data': [
            'test_data_prep',
            'test_data_quality_agent',
            'test_metrics',
        ],
        'models': [
            'test_model_selection',
            'test_ensemble',
            'test_tabnet_sklearn',
            'test_incremental_learning',
        ],
        'mlops': [
            'test_mlflow_registry',
            'test_mlops_service',
            'test_mlops_endpoints',
            'test_monitoring',
            'test_retraining_service',
            'test_export_service',
            'test_ab_testing',
        ],
        'api': [
            'test_api',
            'test_auth_endpoints',  # Added SSO and RGPD endpoints tests
            'test_billing',
            'test_streaming',
        ],
        'auth': [
            'test_auth',
            'test_auth_endpoints',
        ],
        'compliance': [
            'test_auth_endpoints',  # Includes RGPD tests
        ],
        'ui': [
            'test_ui_streamlit',
        ],
        'llm': [
            'test_prompts',
            'test_data_quality_agent',
        ]
    }
    
    if category not in category_tests:
        raise ValueError(f"Unknown category: {category}. Options: {list(category_tests.keys())}")
    
    test_dir = Path(__file__).parent
    test_files = []
    
    for test_module in category_tests[category]:
        test_file = test_dir / f"{test_module}.py"
        if test_file.exists():
            test_files.append(str(test_file))
    
    if not test_files:
        raise FileNotFoundError(f"No test files found for category: {category}")
    
    args = test_files
    if verbose:
        args.append('-v')
    
    print(f"\nRunning {category} tests:")
    print("-" * 40)
    for test_file in test_files:
        print(f"  • {Path(test_file).name}")
    print("-" * 40)
    
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

def create_mock_user(plan_type='PRO', is_admin=False):
    """
    Create a mock user for testing.
    
    Args:
        plan_type: User's subscription plan
        is_admin: Whether user has admin role
    
    Returns:
        Mock user object
    """
    from unittest.mock import Mock
    import uuid
    
    user = Mock()
    user.id = uuid.uuid4()
    user.username = "admin" if is_admin else "testuser"
    user.email = f"{user.username}@example.com"
    user.tenant_id = uuid.uuid4()
    user.plan_type = plan_type
    user.is_active = True
    
    if is_admin:
        user.roles = [Mock(name="admin")]
    else:
        user.roles = [Mock(name="user")]
    
    return user

def create_test_token(user_id=None, tenant_id=None, roles=None, expired=False):
    """
    Create a test JWT token.
    
    Args:
        user_id: User ID to include in token
        tenant_id: Tenant ID to include in token
        roles: List of roles
        expired: Whether token should be expired
    
    Returns:
        JWT token string
    """
    import jwt
    from datetime import datetime, timedelta
    import uuid
    
    if user_id is None:
        user_id = str(uuid.uuid4())
    if tenant_id is None:
        tenant_id = str(uuid.uuid4())
    if roles is None:
        roles = ["user"]
    
    payload = {
        "sub": user_id,
        "tenant_id": tenant_id,
        "roles": roles,
        "iat": datetime.utcnow(),
        "jti": str(uuid.uuid4())
    }
    
    if expired:
        payload["exp"] = datetime.utcnow() - timedelta(hours=1)
    else:
        payload["exp"] = datetime.utcnow() + timedelta(hours=1)
    
    return jwt.encode(payload, "test_secret", algorithm="HS256")

def create_mock_config(environment='test'):
    """
    Create a mock configuration for testing.
    
    Args:
        environment: Environment name (test, development, staging, production)
    
    Returns:
        Mock AutoMLConfig object
    """
    from unittest.mock import Mock
    
    config = Mock()
    config.environment = environment
    config.storage = Mock(backend='local', max_versions_per_model=3)
    config.worker = Mock(enabled=True, max_workers=4, gpu_workers=0)
    config.billing = Mock(enabled=True, plan_type='professional')
    config.monitoring = Mock(enabled=True)
    config.streaming = Mock(enabled=False)
    config.llm = Mock(enabled=False)
    config.api = Mock(host='localhost', port=8000)
    
    return config

def create_mock_health_check(service_name, status='healthy'):
    """
    Create a mock health check result.
    
    Args:
        service_name: Name of the service
        status: Health status (healthy, degraded, unhealthy, unknown)
    
    Returns:
        Mock HealthCheck object
    """
    from unittest.mock import Mock
    from datetime import datetime
    
    check = Mock()
    check.service = service_name
    check.status = Mock(value=status)
    check.message = f"{service_name} is {status}"
    check.latency_ms = 10.5
    check.timestamp = datetime.utcnow()
    check.details = {}
    
    return check

# Export key items
__all__ = [
    'MODULES_STATUS',
    'check_module_availability',
    'get_test_modules',
    'run_all_tests',
    'run_specific_tests',
    'run_test_category',
    'create_test_data',
    'create_mock_user',
    'create_test_token',
    'create_mock_config',
    'create_mock_health_check',
]

# Run module check on import
check_module_availability()
