#!/usr/bin/env python
"""
Test d'installation pour la plateforme AutoML
Vérifie que tous les packages nécessaires sont installés correctement
"""

import sys
import warnings
warnings.filterwarnings('ignore')

def test_package(package_name, import_name=None, version_attr='__version__'):
    """Test si un package est installé et retourne sa version."""
    if import_name is None:
        import_name = package_name
    
    try:
        module = __import__(import_name)
        # Gestion des sous-modules
        components = import_name.split('.')
        for comp in components[1:]:
            module = getattr(module, comp)
        
        # Obtenir la version
        version = "OK"
        if hasattr(module, version_attr):
            version = getattr(module, version_attr)
        elif hasattr(module, 'VERSION'):
            version = module.VERSION
        
        return True, version
    except ImportError as e:
        return False, str(e)
    except Exception as e:
        return False, f"Error: {e}"

def print_status(category, packages):
    """Affiche le statut d'une catégorie de packages."""
    print(f"\n{'='*60}")
    print(f" {category}")
    print('='*60)
    
    all_ok = True
    for package_info in packages:
        if isinstance(package_info, tuple):
            display_name, import_name = package_info
        else:
            display_name = import_name = package_info
        
        is_installed, version = test_package(display_name, import_name)
        
        if is_installed:
            status = f"✅ {display_name:<20} {version}"
            print(status)
        else:
            status = f"❌ {display_name:<20} NOT INSTALLED"
            print(status)
            all_ok = False
    
    return all_ok

def main():
    print("\n" + "="*60)
    print(" AutoML Platform - Installation Test")
    print("="*60)
    print(f"\nPython Version: {sys.version}")
    print(f"Python Path: {sys.executable}")
    
    # Core packages
    core_packages = [
        'pandas',
        'numpy',
        ('scikit-learn', 'sklearn'),
        'scipy',
        'joblib',
        'yaml'
    ]
    
    # API packages
    api_packages = [
        'fastapi',
        'uvicorn',
        'pydantic',
        ('python-multipart', 'multipart'),
        'aiofiles'
    ]
    
    # ML/Boosting packages
    ml_packages = [
        'xgboost',
        'lightgbm',
        'catboost',
        'optuna',
        ('imbalanced-learn', 'imblearn')
    ]
    
    # Storage & Monitoring packages
    storage_packages = [
        'minio',
        ('prometheus-client', 'prometheus_client'),
        ('pyarrow', 'pyarrow')
    ]
    
    # Visualization packages
    viz_packages = [
        'matplotlib',
        'seaborn',
        'plotly'
    ]
    
    # Optional packages
    optional_packages = [
        ('pytorch', 'torch'),
        ('tensorflow', 'tensorflow'),
        ('pytorch-tabnet', 'pytorch_tabnet'),
        'shap',
        'lime',
        'evidently'
    ]
    
    # Test each category
    results = {}
    results['core'] = print_status("🔧 CORE PACKAGES (Required)", core_packages)
    results['api'] = print_status("🌐 API PACKAGES (Required for API)", api_packages)
    results['ml'] = print_status("🤖 ML/BOOSTING PACKAGES (Recommended)", ml_packages)
    results['storage'] = print_status("💾 STORAGE & MONITORING (Recommended)", storage_packages)
    results['viz'] = print_status("📊 VISUALIZATION (Recommended)", viz_packages)
    results['optional'] = print_status("🔌 OPTIONAL PACKAGES", optional_packages)
    
    # Summary
    print("\n" + "="*60)
    print(" SUMMARY")
    print("="*60)
    
    if results['core']:
        print("✅ Core packages: ALL INSTALLED - Ready for basic AutoML")
    else:
        print("❌ Core packages: MISSING - Install required packages")
    
    if results['api']:
        print("✅ API packages: ALL INSTALLED - Ready to run API server")
    else:
        print("⚠️  API packages: MISSING - Install for API functionality")
    
    if results['ml']:
        print("✅ ML packages: ALL INSTALLED - All algorithms available")
    else:
        print("⚠️  ML packages: PARTIAL - Some algorithms unavailable")
    
    # Installation commands
    print("\n" + "="*60)
    print(" INSTALLATION COMMANDS")
    print("="*60)
    
    if not results['core']:
        print("\n# Install core packages:")
        print("pip install pandas numpy scikit-learn scipy joblib pyyaml")
    
    if not results['api']:
        print("\n# Install API packages:")
        print("pip install fastapi uvicorn python-multipart aiofiles")
    
    if not results['ml']:
        print("\n# Install ML packages (one by one if needed):")
        print("pip install xgboost")
        print("pip install lightgbm")
        print("pip install catboost")
        print("pip install optuna")
        print("pip install imbalanced-learn")
    
    if not results['storage']:
        print("\n# Install storage/monitoring packages:")
        print("pip install minio prometheus-client pyarrow")
    
    # Quick start
    print("\n" + "="*60)
    print(" QUICK START")
    print("="*60)
    
    if results['core']:
        print("\n1. Test basic functionality:")
        print("   python -c \"from automl_platform.orchestrator import AutoMLOrchestrator; print('OK')\"")
    
    if results['api']:
        print("\n2. Start API server:")
        print("   python app.py")
        print("   # Or:")
        print("   uvicorn app:app --reload --host 0.0.0.0 --port 8000")
    
    print("\n3. Access API documentation:")
    print("   http://localhost:8000/docs")
    
    # Test imports from automl_platform
    print("\n" + "="*60)
    print(" TESTING AUTOML_PLATFORM MODULES")
    print("="*60)
    
    modules_to_test = [
        'config',
        'data_prep',
        'model_selection',
        'metrics',
        'orchestrator',
        'ensemble',
        'feature_engineering',
        'storage',
        'monitoring'
    ]
    
    for module in modules_to_test:
        try:
            exec(f"from automl_platform.{module} import *")
            print(f"✅ automl_platform.{module}")
        except ImportError as e:
            print(f"❌ automl_platform.{module} - {e}")
        except Exception as e:
            print(f"⚠️  automl_platform.{module} - {e}")
    
    print("\n" + "="*60)
    print(" TEST COMPLETE")
    print("="*60)
    
    if results['core'] and results['api']:
        print("\n🎉 Your AutoML platform is ready to use!")
        print("\nNext steps:")
        print("1. Place all the module files in automl_platform/")
        print("2. Create config.yaml in the root directory")
        print("3. Run: python app.py")
    else:
        print("\n⚠️  Please install missing packages before running the platform")

if __name__ == "__main__":
    main()
