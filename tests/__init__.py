"""
Module de tests pour l'API Gateway
"""

import sys
import os
import unittest
from typing import List, Optional

# Ajouter le répertoire parent au path pour l'import des modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_test_modules() -> List[str]:
    """
    Retourne la liste de tous les modules de test disponibles
    
    Returns:
        List[str]: Liste des noms de modules de test
    """
    return [
        'test_api_gateway',
        'test_audit_service',
        'test_auth_middleware',
        'test_cache_manager',
        'test_circuit_breaker',
        'test_config_manager',
        'test_database_service',
        'test_health_monitor',
        'test_load_balancer',
        'test_monitoring_service',
        'test_rate_limiter',
        'test_request_validator',
        'test_response_transformer',
        'test_rgpd_compliance_service',
        'test_route_manager',
        'test_security_headers',
        'test_service_discovery',
        'test_service_registry',
        'test_sso_service',
        'test_websocket_handler'
    ]


def run_all_tests(verbosity: int = 2) -> unittest.TestResult:
    """
    Exécute tous les tests disponibles
    
    Args:
        verbosity: Niveau de détail des résultats (0, 1 ou 2)
        
    Returns:
        unittest.TestResult: Résultats des tests
    """
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Charger tous les modules de test
    for module_name in get_test_modules():
        try:
            module = __import__(f'tests.{module_name}', fromlist=[module_name])
            suite.addTests(loader.loadTestsFromModule(module))
            print(f"✓ Module {module_name} chargé")
        except ImportError as e:
            print(f"✗ Impossible de charger {module_name}: {e}")
    
    # Exécuter les tests
    runner = unittest.TextTestRunner(verbosity=verbosity)
    return runner.run(suite)


def run_test_module(module_name: str, verbosity: int = 2) -> Optional[unittest.TestResult]:
    """
    Exécute les tests d'un module spécifique
    
    Args:
        module_name: Nom du module de test à exécuter
        verbosity: Niveau de détail des résultats
        
    Returns:
        Optional[unittest.TestResult]: Résultats des tests ou None si le module n'existe pas
    """
    if module_name not in get_test_modules():
        print(f"Module de test '{module_name}' non trouvé")
        print(f"Modules disponibles: {', '.join(get_test_modules())}")
        return None
    
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    try:
        module = __import__(f'tests.{module_name}', fromlist=[module_name])
        suite.addTests(loader.loadTestsFromModule(module))
        print(f"Exécution des tests du module {module_name}...")
        
        runner = unittest.TextTestRunner(verbosity=verbosity)
        return runner.run(suite)
    except ImportError as e:
        print(f"Erreur lors du chargement du module {module_name}: {e}")
        return None


def run_test_category(category: str, verbosity: int = 2) -> Optional[unittest.TestResult]:
    """
    Exécute les tests d'une catégorie spécifique
    
    Args:
        category: Catégorie de tests à exécuter
        verbosity: Niveau de détail des résultats
        
    Returns:
        Optional[unittest.TestResult]: Résultats des tests
    """
    categories = {
        'core': [
            'test_api_gateway',
            'test_database_service',
            'test_route_manager',
            'test_websocket_handler'
        ],
        'core_health': [
            'test_health_monitor'
        ],
        'core_registry': [
            'test_service_registry',
            'test_service_discovery'
        ],
        'core_config': [
            'test_config_manager'
        ],
        'middleware': [
            'test_auth_middleware',
            'test_rate_limiter',
            'test_security_headers',
            'test_request_validator',
            'test_response_transformer'
        ],
        'services': [
            'test_monitoring_service',
            'test_cache_manager',
            'test_load_balancer',
            'test_circuit_breaker'
        ],
        'sso': [
            'test_sso_service'
        ],
        'compliance': [
            'test_audit_service',
            'test_rgpd_compliance_service'
        ],
        'reliability': [
            'test_circuit_breaker',
            'test_health_monitor',
            'test_load_balancer'
        ],
        'security': [
            'test_auth_middleware',
            'test_security_headers',
            'test_sso_service',
            'test_audit_service'
        ],
        'performance': [
            'test_cache_manager',
            'test_rate_limiter',
            'test_load_balancer'
        ]
    }
    
    if category not in categories:
        print(f"Catégorie '{category}' non trouvée")
        print(f"Catégories disponibles: {', '.join(categories.keys())}")
        return None
    
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    modules_to_run = categories[category]
    print(f"Exécution des tests de la catégorie '{category}'...")
    print(f"Modules: {', '.join(modules_to_run)}")
    
    for module_name in modules_to_run:
        try:
            module = __import__(f'tests.{module_name}', fromlist=[module_name])
            suite.addTests(loader.loadTestsFromModule(module))
            print(f"✓ Module {module_name} chargé")
        except ImportError as e:
            print(f"✗ Impossible de charger {module_name}: {e}")
    
    runner = unittest.TextTestRunner(verbosity=verbosity)
    return runner.run(suite)


def get_test_statistics() -> dict:
    """
    Obtient des statistiques sur les tests disponibles
    
    Returns:
        dict: Statistiques des tests
    """
    stats = {
        'total_modules': len(get_test_modules()),
        'modules': {},
        'categories': {
            'core': 4,
            'core_health': 1,
            'core_registry': 2,
            'core_config': 1,
            'middleware': 5,
            'services': 4,
            'sso': 1,
            'compliance': 2,
            'reliability': 3,
            'security': 4,
            'performance': 3
        }
    }
    
    # Compter les tests par module
    loader = unittest.TestLoader()
    for module_name in get_test_modules():
        try:
            module = __import__(f'tests.{module_name}', fromlist=[module_name])
            suite = loader.loadTestsFromModule(module)
            test_count = suite.countTestCases()
            stats['modules'][module_name] = test_count
        except ImportError:
            stats['modules'][module_name] = 0
    
    stats['total_tests'] = sum(stats['modules'].values())
    
    return stats


def print_test_summary():
    """
    Affiche un résumé des tests disponibles
    """
    stats = get_test_statistics()
    
    print("\n" + "="*60)
    print("RÉSUMÉ DES TESTS")
    print("="*60)
    print(f"Total de modules: {stats['total_modules']}")
    print(f"Total de tests: {stats['total_tests']}")
    
    print("\n📦 MODULES DE TEST:")
    print("-"*40)
    for module, count in sorted(stats['modules'].items()):
        status = "✓" if count > 0 else "✗"
        print(f"{status} {module:<35} ({count} tests)")
    
    print("\n📁 CATÉGORIES:")
    print("-"*40)
    for category, count in sorted(stats['categories'].items()):
        print(f"  • {category:<20} ({count} modules)")
    
    print("\n💡 UTILISATION:")
    print("-"*40)
    print("  • Tous les tests:        python -m tests")
    print("  • Module spécifique:     python -m tests test_api_gateway")
    print("  • Catégorie spécifique:  python -m tests --category core")
    print("  • Afficher ce résumé:    python -m tests --summary")
    print("="*60 + "\n")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Exécuter les tests de l\'API Gateway')
    parser.add_argument('module', nargs='?', help='Module de test spécifique à exécuter')
    parser.add_argument('--category', '-c', help='Catégorie de tests à exécuter')
    parser.add_argument('--all', '-a', action='store_true', help='Exécuter tous les tests')
    parser.add_argument('--summary', '-s', action='store_true', help='Afficher le résumé des tests')
    parser.add_argument('--verbose', '-v', type=int, default=2, choices=[0, 1, 2],
                       help='Niveau de verbosité (0=minimal, 1=normal, 2=détaillé)')
    
    args = parser.parse_args()
    
    if args.summary:
        print_test_summary()
    elif args.all:
        print("Exécution de tous les tests...")
        result = run_all_tests(verbosity=args.verbose)
        sys.exit(0 if result.wasSuccessful() else 1)
    elif args.category:
        result = run_test_category(args.category, verbosity=args.verbose)
        if result:
            sys.exit(0 if result.wasSuccessful() else 1)
        else:
            sys.exit(1)
    elif args.module:
        result = run_test_module(args.module, verbosity=args.verbose)
        if result:
            sys.exit(0 if result.wasSuccessful() else 1)
        else:
            sys.exit(1)
    else:
        print_test_summary()
