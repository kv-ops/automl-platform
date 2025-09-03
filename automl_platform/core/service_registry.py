"""
Service Registry for AutoML Platform
====================================
Central registry for all platform services with dependency management.
"""

import logging
from typing import Any, Dict, Optional, List, Set
from dataclasses import dataclass
from datetime import datetime
import threading
from enum import Enum

logger = logging.getLogger(__name__)


class ServiceStatus(Enum):
    """Service health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    STARTING = "starting"
    STOPPING = "stopping"
    STOPPED = "stopped"


@dataclass
class ServiceInfo:
    """Service registration information."""
    name: str
    service_type: str
    instance: Any
    dependencies: List[str]
    status: ServiceStatus
    registered_at: datetime
    last_health_check: Optional[datetime] = None
    metadata: Dict[str, Any] = None


class ServiceRegistry:
    """
    Central service registry with singleton pattern.
    Manages all platform services and their dependencies.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._services: Dict[str, ServiceInfo] = {}
        self._dependencies: Dict[str, Set[str]] = {}
        self._initialized = True
        logger.info("ServiceRegistry initialized")
    
    def register(self, 
                 name: str, 
                 service: Any,
                 service_type: str = "unknown",
                 dependencies: List[str] = None,
                 metadata: Dict[str, Any] = None) -> bool:
        """
        Register a service with the registry.
        
        Args:
            name: Service name (unique identifier)
            service: Service instance
            service_type: Type of service (storage, billing, scheduler, etc.)
            dependencies: List of service names this service depends on
            metadata: Additional metadata about the service
        
        Returns:
            True if registration successful
        """
        if name in self._services:
            logger.warning(f"Service {name} already registered, updating...")
        
        # Check dependencies exist
        deps = dependencies or []
        for dep in deps:
            if dep not in self._services:
                logger.error(f"Cannot register {name}: dependency {dep} not found")
                return False
        
        # Register service
        service_info = ServiceInfo(
            name=name,
            service_type=service_type,
            instance=service,
            dependencies=deps,
            status=ServiceStatus.STARTING,
            registered_at=datetime.utcnow(),
            metadata=metadata or {}
        )
        
        self._services[name] = service_info
        
        # Update dependency graph
        self._dependencies[name] = set(deps)
        
        # Update status to healthy if all dependencies are healthy
        if self._check_dependencies_health(name):
            service_info.status = ServiceStatus.HEALTHY
        
        logger.info(f"Service registered: {name} (type: {service_type})")
        return True
    
    def unregister(self, name: str) -> bool:
        """
        Unregister a service.
        
        Args:
            name: Service name to unregister
        
        Returns:
            True if unregistration successful
        """
        if name not in self._services:
            logger.warning(f"Service {name} not found")
            return False
        
        # Check if other services depend on this
        dependent_services = self._get_dependent_services(name)
        if dependent_services:
            logger.error(f"Cannot unregister {name}: services {dependent_services} depend on it")
            return False
        
        # Mark as stopping
        self._services[name].status = ServiceStatus.STOPPING
        
        # Remove from registry
        del self._services[name]
        del self._dependencies[name]
        
        logger.info(f"Service unregistered: {name}")
        return True
    
    def get(self, name: str) -> Optional[Any]:
        """
        Get a service instance by name.
        
        Args:
            name: Service name
        
        Returns:
            Service instance or None if not found
        """
        service_info = self._services.get(name)
        return service_info.instance if service_info else None
    
    def get_info(self, name: str) -> Optional[ServiceInfo]:
        """
        Get service information.
        
        Args:
            name: Service name
        
        Returns:
            ServiceInfo or None if not found
        """
        return self._services.get(name)
    
    def list_services(self, service_type: str = None) -> List[str]:
        """
        List all registered services.
        
        Args:
            service_type: Filter by service type (optional)
        
        Returns:
            List of service names
        """
        if service_type:
            return [
                name for name, info in self._services.items()
                if info.service_type == service_type
            ]
        return list(self._services.keys())
    
    def get_service_status(self, name: str) -> Optional[ServiceStatus]:
        """
        Get service status.
        
        Args:
            name: Service name
        
        Returns:
            ServiceStatus or None if not found
        """
        service_info = self._services.get(name)
        return service_info.status if service_info else None
    
    def update_status(self, name: str, status: ServiceStatus) -> bool:
        """
        Update service status.
        
        Args:
            name: Service name
            status: New status
        
        Returns:
            True if update successful
        """
        if name not in self._services:
            logger.warning(f"Service {name} not found")
            return False
        
        old_status = self._services[name].status
        self._services[name].status = status
        self._services[name].last_health_check = datetime.utcnow()
        
        logger.info(f"Service {name} status updated: {old_status.value} -> {status.value}")
        
        # Check if this affects dependent services
        if status in [ServiceStatus.UNHEALTHY, ServiceStatus.STOPPED]:
            self._propagate_unhealthy_status(name)
        
        return True
    
    def _check_dependencies_health(self, name: str) -> bool:
        """Check if all dependencies are healthy."""
        deps = self._dependencies.get(name, set())
        for dep in deps:
            dep_info = self._services.get(dep)
            if not dep_info or dep_info.status != ServiceStatus.HEALTHY:
                return False
        return True
    
    def _get_dependent_services(self, name: str) -> List[str]:
        """Get services that depend on the given service."""
        dependent = []
        for service_name, deps in self._dependencies.items():
            if name in deps:
                dependent.append(service_name)
        return dependent
    
    def _propagate_unhealthy_status(self, name: str):
        """Propagate unhealthy status to dependent services."""
        dependent = self._get_dependent_services(name)
        for dep_name in dependent:
            if self._services[dep_name].status == ServiceStatus.HEALTHY:
                self._services[dep_name].status = ServiceStatus.DEGRADED
                logger.warning(f"Service {dep_name} degraded due to {name} being unhealthy")
    
    def get_dependency_graph(self) -> Dict[str, List[str]]:
        """
        Get the dependency graph.
        
        Returns:
            Dictionary mapping service names to their dependencies
        """
        return {name: list(deps) for name, deps in self._dependencies.items()}
    
    def get_service_tree(self) -> Dict[str, Any]:
        """
        Get hierarchical service tree.
        
        Returns:
            Nested dictionary representing service hierarchy
        """
        tree = {}
        
        # Find root services (no dependencies)
        roots = [
            name for name, deps in self._dependencies.items()
            if not deps
        ]
        
        def build_subtree(service_name: str) -> Dict[str, Any]:
            info = self._services[service_name]
            dependents = self._get_dependent_services(service_name)
            
            return {
                "name": service_name,
                "type": info.service_type,
                "status": info.status.value,
                "dependents": [build_subtree(dep) for dep in dependents]
            }
        
        for root in roots:
            tree[root] = build_subtree(root)
        
        return tree
    
    def check_circular_dependencies(self) -> List[List[str]]:
        """
        Check for circular dependencies.
        
        Returns:
            List of circular dependency paths
        """
        visited = set()
        rec_stack = set()
        cycles = []
        
        def dfs(node: str, path: List[str]) -> bool:
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            
            for neighbor in self._dependencies.get(node, []):
                if neighbor not in visited:
                    if dfs(neighbor, path.copy()):
                        return True
                elif neighbor in rec_stack:
                    # Found cycle
                    cycle_start = path.index(neighbor)
                    cycles.append(path[cycle_start:] + [neighbor])
            
            rec_stack.remove(node)
            return False
        
        for node in self._services:
            if node not in visited:
                dfs(node, [])
        
        return cycles
    
    def shutdown_order(self) -> List[str]:
        """
        Get the order in which services should be shutdown.
        
        Returns:
            List of service names in shutdown order (reverse dependency order)
        """
        # Topological sort in reverse
        visited = set()
        order = []
        
        def dfs(node: str):
            visited.add(node)
            for dep in self._get_dependent_services(node):
                if dep not in visited:
                    dfs(dep)
            order.append(node)
        
        # Start with leaf services (no dependents)
        leaves = [
            name for name in self._services
            if not self._get_dependent_services(name)
        ]
        
        for leaf in leaves:
            if leaf not in visited:
                dfs(leaf)
        
        # Add any remaining services
        for service in self._services:
            if service not in visited:
                order.append(service)
        
        return order
    
    def startup_order(self) -> List[str]:
        """
        Get the order in which services should be started.
        
        Returns:
            List of service names in startup order (dependency order)
        """
        return list(reversed(self.shutdown_order()))
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get registry statistics.
        
        Returns:
            Dictionary with registry statistics
        """
        total = len(self._services)
        by_type = {}
        by_status = {}
        
        for info in self._services.values():
            # Count by type
            if info.service_type not in by_type:
                by_type[info.service_type] = 0
            by_type[info.service_type] += 1
            
            # Count by status
            status_name = info.status.value
            if status_name not in by_status:
                by_status[status_name] = 0
            by_status[status_name] += 1
        
        return {
            "total_services": total,
            "by_type": by_type,
            "by_status": by_status,
            "circular_dependencies": len(self.check_circular_dependencies()) > 0,
            "registry_initialized_at": self._services.get("_registry", {}).get("registered_at", "N/A")
        }
    
    def clear(self):
        """Clear all services from registry (use with caution)."""
        logger.warning("Clearing all services from registry")
        self._services.clear()
        self._dependencies.clear()


# Global instance getter
def get_registry() -> ServiceRegistry:
    """Get the global ServiceRegistry instance."""
    return ServiceRegistry()
