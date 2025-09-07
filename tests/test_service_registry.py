"""
Tests for ServiceRegistry
=========================
Comprehensive tests for the service registry with dependency management.
"""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime
import threading

from automl_platform.core.service_registry import (
    ServiceRegistry,
    ServiceInfo,
    ServiceStatus,
    get_registry
)


class TestServiceRegistry:
    """Test suite for ServiceRegistry."""
    
    def test_singleton_pattern(self):
        """Test that ServiceRegistry implements singleton pattern correctly."""
        # Multiple calls should return the same instance
        registry1 = ServiceRegistry()
        registry2 = ServiceRegistry()
        registry3 = get_registry()
        
        assert registry1 is registry2
        assert registry2 is registry3
        assert id(registry1) == id(registry2) == id(registry3)
        
        # Clear for other tests
        registry1.clear()
    
    def test_thread_safe_singleton(self):
        """Test that singleton is thread-safe."""
        instances = []
        
        def create_instance():
            instances.append(ServiceRegistry())
        
        # Create multiple threads trying to create instances
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=create_instance)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # All instances should be the same
        first_instance = instances[0]
        for instance in instances[1:]:
            assert instance is first_instance
        
        # Clear for other tests
        first_instance.clear()
    
    def test_register_service_without_dependencies(self):
        """Test registering a service without dependencies."""
        registry = ServiceRegistry()
        registry.clear()
        
        # Create mock service
        mock_service = Mock()
        mock_service.name = "test_service"
        
        # Register service
        success = registry.register(
            name="test_service",
            service=mock_service,
            service_type="storage",
            metadata={"version": "1.0"}
        )
        
        assert success == True
        
        # Verify service is registered
        service_info = registry.get_info("test_service")
        assert service_info is not None
        assert service_info.name == "test_service"
        assert service_info.service_type == "storage"
        assert service_info.instance is mock_service
        assert service_info.dependencies == []
        assert service_info.status == ServiceStatus.HEALTHY
        assert service_info.metadata == {"version": "1.0"}
        
        registry.clear()
    
    def test_register_service_with_dependencies(self):
        """Test registering a service with dependencies."""
        registry = ServiceRegistry()
        registry.clear()
        
        # Register dependency first
        dependency_service = Mock()
        registry.register("dependency", dependency_service, "storage")
        
        # Register service with dependency
        main_service = Mock()
        success = registry.register(
            name="main_service",
            service=main_service,
            service_type="worker",
            dependencies=["dependency"]
        )
        
        assert success == True
        
        # Verify service and dependencies
        service_info = registry.get_info("main_service")
        assert service_info.dependencies == ["dependency"]
        assert service_info.status == ServiceStatus.HEALTHY
        
        registry.clear()
    
    def test_register_service_with_missing_dependency(self):
        """Test that registration fails when dependency is missing."""
        registry = ServiceRegistry()
        registry.clear()
        
        # Try to register service with non-existent dependency
        service = Mock()
        success = registry.register(
            name="service",
            service=service,
            dependencies=["non_existent"]
        )
        
        assert success == False
        assert registry.get("service") is None
        
        registry.clear()
    
    def test_register_existing_service_updates(self):
        """Test that re-registering a service updates it."""
        registry = ServiceRegistry()
        registry.clear()
        
        # Register service
        service1 = Mock()
        registry.register("service", service1, "type1")
        
        # Re-register with different instance
        service2 = Mock()
        success = registry.register("service", service2, "type2")
        
        assert success == True
        
        # Verify update
        service_info = registry.get_info("service")
        assert service_info.instance is service2
        assert service_info.service_type == "type2"
        
        registry.clear()
    
    def test_unregister_service(self):
        """Test unregistering a service."""
        registry = ServiceRegistry()
        registry.clear()
        
        # Register service
        service = Mock()
        registry.register("service", service)
        
        # Unregister
        success = registry.unregister("service")
        assert success == True
        
        # Verify service is gone
        assert registry.get("service") is None
        assert registry.get_info("service") is None
        
        registry.clear()
    
    def test_unregister_nonexistent_service(self):
        """Test unregistering a non-existent service."""
        registry = ServiceRegistry()
        registry.clear()
        
        success = registry.unregister("non_existent")
        assert success == False
        
        registry.clear()
    
    def test_unregister_service_with_dependents(self):
        """Test that unregistering fails when other services depend on it."""
        registry = ServiceRegistry()
        registry.clear()
        
        # Register services with dependencies
        service1 = Mock()
        service2 = Mock()
        
        registry.register("service1", service1)
        registry.register("service2", service2, dependencies=["service1"])
        
        # Try to unregister service1 (should fail)
        success = registry.unregister("service1")
        assert success == False
        
        # service1 should still be registered
        assert registry.get("service1") is not None
        
        registry.clear()
    
    def test_get_service(self):
        """Test getting a service instance."""
        registry = ServiceRegistry()
        registry.clear()
        
        service = Mock()
        service.test_attr = "test_value"
        
        registry.register("service", service)
        
        # Get service
        retrieved = registry.get("service")
        assert retrieved is service
        assert retrieved.test_attr == "test_value"
        
        # Get non-existent service
        assert registry.get("non_existent") is None
        
        registry.clear()
    
    def test_list_services(self):
        """Test listing services."""
        registry = ServiceRegistry()
        registry.clear()
        
        # Register multiple services
        registry.register("service1", Mock(), "type1")
        registry.register("service2", Mock(), "type1")
        registry.register("service3", Mock(), "type2")
        
        # List all services
        all_services = registry.list_services()
        assert set(all_services) == {"service1", "service2", "service3"}
        
        # List by type
        type1_services = registry.list_services(service_type="type1")
        assert set(type1_services) == {"service1", "service2"}
        
        type2_services = registry.list_services(service_type="type2")
        assert type2_services == ["service3"]
        
        registry.clear()
    
    def test_update_status(self):
        """Test updating service status."""
        registry = ServiceRegistry()
        registry.clear()
        
        # Register service
        service = Mock()
        registry.register("service", service)
        
        # Initial status should be HEALTHY
        assert registry.get_service_status("service") == ServiceStatus.HEALTHY
        
        # Update status
        success = registry.update_status("service", ServiceStatus.DEGRADED)
        assert success == True
        assert registry.get_service_status("service") == ServiceStatus.DEGRADED
        
        # Update non-existent service
        success = registry.update_status("non_existent", ServiceStatus.HEALTHY)
        assert success == False
        
        registry.clear()
    
    def test_status_propagation_to_dependents(self):
        """Test that unhealthy status propagates to dependent services."""
        registry = ServiceRegistry()
        registry.clear()
        
        # Create dependency chain: service3 -> service2 -> service1
        registry.register("service1", Mock())
        registry.register("service2", Mock(), dependencies=["service1"])
        registry.register("service3", Mock(), dependencies=["service2"])
        
        # All should be healthy initially
        assert registry.get_service_status("service1") == ServiceStatus.HEALTHY
        assert registry.get_service_status("service2") == ServiceStatus.HEALTHY
        assert registry.get_service_status("service3") == ServiceStatus.HEALTHY
        
        # Mark service1 as unhealthy
        registry.update_status("service1", ServiceStatus.UNHEALTHY)
        
        # service2 and service3 should become degraded
        assert registry.get_service_status("service2") == ServiceStatus.DEGRADED
        assert registry.get_service_status("service3") == ServiceStatus.HEALTHY  # Not directly dependent
        
        # Mark service2 as unhealthy
        registry.update_status("service2", ServiceStatus.UNHEALTHY)
        
        # service3 should become degraded
        assert registry.get_service_status("service3") == ServiceStatus.DEGRADED
        
        registry.clear()
    
    def test_get_dependency_graph(self):
        """Test getting the dependency graph."""
        registry = ServiceRegistry()
        registry.clear()
        
        # Create services with dependencies
        registry.register("service1", Mock())
        registry.register("service2", Mock(), dependencies=["service1"])
        registry.register("service3", Mock(), dependencies=["service1", "service2"])
        
        graph = registry.get_dependency_graph()
        
        assert graph == {
            "service1": [],
            "service2": ["service1"],
            "service3": ["service1", "service2"]
        }
        
        registry.clear()
    
    def test_check_circular_dependencies(self):
        """Test detection of circular dependencies."""
        registry = ServiceRegistry()
        registry.clear()
        
        # Create services without circular dependencies
        registry.register("service1", Mock())
        registry.register("service2", Mock(), dependencies=["service1"])
        
        cycles = registry.check_circular_dependencies()
        assert cycles == []
        
        # Manually create circular dependency (bypassing normal checks)
        registry._dependencies["service1"] = {"service2"}
        
        cycles = registry.check_circular_dependencies()
        assert len(cycles) > 0
        
        registry.clear()
    
    def test_startup_order(self):
        """Test calculating service startup order."""
        registry = ServiceRegistry()
        registry.clear()
        
        # Create dependency chain
        registry.register("database", Mock())
        registry.register("cache", Mock())
        registry.register("api", Mock(), dependencies=["database", "cache"])
        registry.register("worker", Mock(), dependencies=["database"])
        
        startup_order = registry.startup_order()
        
        # Database and cache should come before api
        # Database should come before worker
        db_index = startup_order.index("database")
        cache_index = startup_order.index("cache")
        api_index = startup_order.index("api")
        worker_index = startup_order.index("worker")
        
        assert db_index < api_index
        assert cache_index < api_index
        assert db_index < worker_index
        
        registry.clear()
    
    def test_shutdown_order(self):
        """Test calculating service shutdown order."""
        registry = ServiceRegistry()
        registry.clear()
        
        # Create dependency chain
        registry.register("database", Mock())
        registry.register("cache", Mock())
        registry.register("api", Mock(), dependencies=["database", "cache"])
        registry.register("worker", Mock(), dependencies=["database"])
        
        shutdown_order = registry.shutdown_order()
        
        # API should shut down before database and cache
        # Worker should shut down before database
        db_index = shutdown_order.index("database")
        cache_index = shutdown_order.index("cache")
        api_index = shutdown_order.index("api")
        worker_index = shutdown_order.index("worker")
        
        assert api_index < db_index
        assert api_index < cache_index
        assert worker_index < db_index
        
        registry.clear()
    
    def test_get_service_tree(self):
        """Test getting hierarchical service tree."""
        registry = ServiceRegistry()
        registry.clear()
        
        # Create services
        registry.register("database", Mock(), "storage")
        registry.register("api", Mock(), "web", dependencies=["database"])
        
        tree = registry.get_service_tree()
        
        # Database should be root with api as dependent
        assert "database" in tree
        assert tree["database"]["name"] == "database"
        assert tree["database"]["type"] == "storage"
        assert len(tree["database"]["dependents"]) == 1
        assert tree["database"]["dependents"][0]["name"] == "api"
        
        registry.clear()
    
    def test_get_statistics(self):
        """Test getting registry statistics."""
        registry = ServiceRegistry()
        registry.clear()
        
        # Register services
        registry.register("service1", Mock(), "storage")
        registry.register("service2", Mock(), "storage")
        registry.register("service3", Mock(), "worker")
        
        # Update some statuses
        registry.update_status("service2", ServiceStatus.DEGRADED)
        
        stats = registry.get_statistics()
        
        assert stats["total_services"] == 3
        assert stats["by_type"]["storage"] == 2
        assert stats["by_type"]["worker"] == 1
        assert stats["by_status"]["healthy"] == 2
        assert stats["by_status"]["degraded"] == 1
        assert stats["circular_dependencies"] == False
        
        registry.clear()
    
    def test_clear_registry(self):
        """Test clearing all services from registry."""
        registry = ServiceRegistry()
        
        # Register some services
        registry.register("service1", Mock())
        registry.register("service2", Mock())
        
        # Clear registry
        registry.clear()
        
        # Verify everything is cleared
        assert registry.list_services() == []
        assert registry.get_dependency_graph() == {}
        
    def test_service_info_dataclass(self):
        """Test ServiceInfo dataclass."""
        service = Mock()
        
        info = ServiceInfo(
            name="test",
            service_type="storage",
            instance=service,
            dependencies=["dep1", "dep2"],
            status=ServiceStatus.HEALTHY,
            registered_at=datetime.utcnow(),
            metadata={"key": "value"}
        )
        
        assert info.name == "test"
        assert info.service_type == "storage"
        assert info.instance is service
        assert info.dependencies == ["dep1", "dep2"]
        assert info.status == ServiceStatus.HEALTHY
        assert info.metadata == {"key": "value"}
        assert info.last_health_check is None
