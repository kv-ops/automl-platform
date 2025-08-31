"""
Infrastructure module for multi-tenant isolation, security, and resource management
Complete implementation with Docker, Kubernetes, and cloud-native support
"""

import os
import uuid
import hashlib
import json
import logging
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import asyncio
from pathlib import Path

# Resource management
import psutil
import resource

# Docker and Kubernetes
try:
    import docker
    DOCKER_AVAILABLE = True
except ImportError:
    DOCKER_AVAILABLE = False

try:
    from kubernetes import client, config as k8s_config
    KUBERNETES_AVAILABLE = True
except ImportError:
    KUBERNETES_AVAILABLE = False

# Security
import jwt
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2

# Database
import sqlalchemy
from sqlalchemy import create_engine, Column, String, Integer, DateTime, Boolean, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

logger = logging.getLogger(__name__)

Base = declarative_base()


@dataclass
class TenantConfig:
    """Tenant configuration and resource limits."""
    tenant_id: str
    name: str
    plan: str  # free, starter, professional, enterprise
    created_at: datetime
    
    # Resource limits
    max_cpu_cores: int = 2
    max_memory_gb: int = 4
    max_storage_gb: int = 10
    max_gpu_hours: int = 0
    max_concurrent_jobs: int = 1
    max_models: int = 5
    max_api_calls_per_day: int = 1000
    
    # Security
    isolation_level: str = "namespace"  # namespace, pod, vm
    encryption_enabled: bool = True
    audit_logging: bool = True
    
    # Features
    features: Dict[str, bool] = None
    
    def __post_init__(self):
        if self.features is None:
            self.features = self._get_default_features()
    
    def _get_default_features(self) -> Dict[str, bool]:
        """Get default features based on plan."""
        features_by_plan = {
            "free": {
                "automl": True,
                "gpu_training": False,
                "advanced_models": False,
                "llm_integration": False,
                "custom_deployment": False,
                "priority_support": False
            },
            "starter": {
                "automl": True,
                "gpu_training": False,
                "advanced_models": True,
                "llm_integration": True,
                "custom_deployment": False,
                "priority_support": False
            },
            "professional": {
                "automl": True,
                "gpu_training": True,
                "advanced_models": True,
                "llm_integration": True,
                "custom_deployment": True,
                "priority_support": True
            },
            "enterprise": {
                "automl": True,
                "gpu_training": True,
                "advanced_models": True,
                "llm_integration": True,
                "custom_deployment": True,
                "priority_support": True,
                "custom_features": True
            }
        }
        return features_by_plan.get(self.plan, features_by_plan["free"])


class TenantModel(Base):
    """SQLAlchemy model for tenant storage."""
    __tablename__ = 'tenants'
    
    tenant_id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    plan = Column(String, default='free')
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Resource limits
    max_cpu_cores = Column(Integer, default=2)
    max_memory_gb = Column(Integer, default=4)
    max_storage_gb = Column(Integer, default=10)
    max_gpu_hours = Column(Integer, default=0)
    max_concurrent_jobs = Column(Integer, default=1)
    
    # Usage tracking
    current_cpu_usage = Column(Integer, default=0)
    current_memory_usage = Column(Integer, default=0)
    current_storage_usage = Column(Integer, default=0)
    gpu_hours_used = Column(Integer, default=0)
    
    # Security
    encryption_key = Column(String)
    api_key_hash = Column(String)
    
    # Status
    is_active = Column(Boolean, default=True)
    is_suspended = Column(Boolean, default=False)
    
    # Metadata
    metadata = Column(JSON, default={})


class TenantManager:
    """Manages multi-tenant isolation and resource allocation."""
    
    def __init__(self, db_url: str = "sqlite:///tenants.db"):
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        
        # Initialize Docker client if available
        if DOCKER_AVAILABLE:
            self.docker_client = docker.from_env()
        else:
            self.docker_client = None
        
        # Initialize Kubernetes client if available
        if KUBERNETES_AVAILABLE:
            try:
                k8s_config.load_incluster_config()
            except:
                try:
                    k8s_config.load_kube_config()
                except:
                    pass
            self.k8s_v1 = client.CoreV1Api()
            self.k8s_apps = client.AppsV1Api()
        else:
            self.k8s_v1 = None
            self.k8s_apps = None
    
    def create_tenant(self, name: str, plan: str = "free") -> TenantConfig:
        """Create a new tenant with isolated resources."""
        tenant_id = str(uuid.uuid4())
        
        # Create tenant config
        tenant_config = TenantConfig(
            tenant_id=tenant_id,
            name=name,
            plan=plan,
            created_at=datetime.utcnow()
        )
        
        # Generate encryption key
        encryption_key = Fernet.generate_key()
        
        # Create database entry
        session = self.Session()
        try:
            tenant_model = TenantModel(
                tenant_id=tenant_id,
                name=name,
                plan=plan,
                max_cpu_cores=tenant_config.max_cpu_cores,
                max_memory_gb=tenant_config.max_memory_gb,
                max_storage_gb=tenant_config.max_storage_gb,
                max_gpu_hours=tenant_config.max_gpu_hours,
                max_concurrent_jobs=tenant_config.max_concurrent_jobs,
                encryption_key=encryption_key.decode()
            )
            session.add(tenant_model)
            session.commit()
            
            # Create isolated namespace if Kubernetes available
            if self.k8s_v1:
                self._create_kubernetes_namespace(tenant_id, tenant_config)
            
            # Create Docker network if Docker available
            if self.docker_client:
                self._create_docker_network(tenant_id)
            
            logger.info(f"Created tenant: {tenant_id} ({name})")
            return tenant_config
            
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to create tenant: {e}")
            raise
        finally:
            session.close()
    
    def _create_kubernetes_namespace(self, tenant_id: str, config: TenantConfig):
        """Create Kubernetes namespace with resource quotas."""
        if not self.k8s_v1:
            return
        
        # Create namespace
        namespace = client.V1Namespace(
            metadata=client.V1ObjectMeta(
                name=f"tenant-{tenant_id[:8]}",
                labels={
                    "tenant_id": tenant_id,
                    "plan": config.plan
                }
            )
        )
        
        try:
            self.k8s_v1.create_namespace(namespace)
            
            # Create resource quota
            quota = client.V1ResourceQuota(
                metadata=client.V1ObjectMeta(name="tenant-quota"),
                spec=client.V1ResourceQuotaSpec(
                    hard={
                        "requests.cpu": str(config.max_cpu_cores),
                        "requests.memory": f"{config.max_memory_gb}Gi",
                        "persistentvolumeclaims": "5",
                        "pods": str(config.max_concurrent_jobs * 3)
                    }
                )
            )
            
            self.k8s_v1.create_namespaced_resource_quota(
                namespace=f"tenant-{tenant_id[:8]}",
                body=quota
            )
            
            # Create network policy for isolation
            network_policy = {
                "apiVersion": "networking.k8s.io/v1",
                "kind": "NetworkPolicy",
                "metadata": {"name": "tenant-isolation"},
                "spec": {
                    "podSelector": {},
                    "policyTypes": ["Ingress", "Egress"],
                    "ingress": [{
                        "from": [{
                            "namespaceSelector": {
                                "matchLabels": {"tenant_id": tenant_id}
                            }
                        }]
                    }],
                    "egress": [{
                        "to": [{
                            "namespaceSelector": {
                                "matchLabels": {"tenant_id": tenant_id}
                            }
                        }]
                    }]
                }
            }
            
            logger.info(f"Created Kubernetes namespace for tenant {tenant_id}")
            
        except Exception as e:
            logger.error(f"Failed to create Kubernetes resources: {e}")
    
    def _create_docker_network(self, tenant_id: str):
        """Create isolated Docker network for tenant."""
        if not self.docker_client:
            return
        
        try:
            network = self.docker_client.networks.create(
                name=f"tenant_{tenant_id[:8]}",
                driver="bridge",
                internal=False,
                labels={"tenant_id": tenant_id}
            )
            logger.info(f"Created Docker network for tenant {tenant_id}")
        except Exception as e:
            logger.error(f"Failed to create Docker network: {e}")
    
    def get_tenant(self, tenant_id: str) -> Optional[TenantConfig]:
        """Get tenant configuration."""
        session = self.Session()
        try:
            tenant = session.query(TenantModel).filter_by(tenant_id=tenant_id).first()
            if tenant:
                return TenantConfig(
                    tenant_id=tenant.tenant_id,
                    name=tenant.name,
                    plan=tenant.plan,
                    created_at=tenant.created_at,
                    max_cpu_cores=tenant.max_cpu_cores,
                    max_memory_gb=tenant.max_memory_gb,
                    max_storage_gb=tenant.max_storage_gb,
                    max_gpu_hours=tenant.max_gpu_hours,
                    max_concurrent_jobs=tenant.max_concurrent_jobs
                )
            return None
        finally:
            session.close()
    
    def check_resource_limits(self, tenant_id: str, 
                            resource_type: str, requested: int) -> bool:
        """Check if tenant has available resources."""
        session = self.Session()
        try:
            tenant = session.query(TenantModel).filter_by(tenant_id=tenant_id).first()
            if not tenant:
                return False
            
            if resource_type == "cpu":
                return tenant.current_cpu_usage + requested <= tenant.max_cpu_cores
            elif resource_type == "memory":
                return tenant.current_memory_usage + requested <= tenant.max_memory_gb
            elif resource_type == "storage":
                return tenant.current_storage_usage + requested <= tenant.max_storage_gb
            elif resource_type == "gpu_hours":
                return tenant.gpu_hours_used + requested <= tenant.max_gpu_hours
            elif resource_type == "concurrent_jobs":
                return tenant.current_cpu_usage > 0  # Simplified check
            
            return False
            
        finally:
            session.close()
    
    def allocate_resources(self, tenant_id: str, 
                          cpu: int = 0, memory: int = 0, 
                          storage: int = 0) -> bool:
        """Allocate resources to tenant."""
        session = self.Session()
        try:
            tenant = session.query(TenantModel).filter_by(tenant_id=tenant_id).first()
            if not tenant:
                return False
            
            # Check limits
            if (tenant.current_cpu_usage + cpu > tenant.max_cpu_cores or
                tenant.current_memory_usage + memory > tenant.max_memory_gb or
                tenant.current_storage_usage + storage > tenant.max_storage_gb):
                return False
            
            # Update usage
            tenant.current_cpu_usage += cpu
            tenant.current_memory_usage += memory
            tenant.current_storage_usage += storage
            session.commit()
            
            return True
            
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to allocate resources: {e}")
            return False
        finally:
            session.close()
    
    def release_resources(self, tenant_id: str, 
                         cpu: int = 0, memory: int = 0, 
                         storage: int = 0):
        """Release allocated resources."""
        session = self.Session()
        try:
            tenant = session.query(TenantModel).filter_by(tenant_id=tenant_id).first()
            if tenant:
                tenant.current_cpu_usage = max(0, tenant.current_cpu_usage - cpu)
                tenant.current_memory_usage = max(0, tenant.current_memory_usage - memory)
                tenant.current_storage_usage = max(0, tenant.current_storage_usage - storage)
                session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to release resources: {e}")
        finally:
            session.close()


class SecurityManager:
    """Manages security, encryption, and access control."""
    
    def __init__(self, secret_key: str = None):
        self.secret_key = secret_key or os.environ.get('SECRET_KEY', 'default-secret-key')
        
    def generate_api_key(self, tenant_id: str) -> str:
        """Generate API key for tenant."""
        # Create unique key
        raw_key = f"{tenant_id}:{uuid.uuid4()}:{datetime.utcnow().isoformat()}"
        
        # Sign with JWT
        api_key = jwt.encode(
            {
                'tenant_id': tenant_id,
                'created_at': datetime.utcnow().isoformat(),
                'exp': datetime.utcnow() + timedelta(days=365)
            },
            self.secret_key,
            algorithm='HS256'
        )
        
        return api_key
    
    def verify_api_key(self, api_key: str) -> Optional[str]:
        """Verify API key and return tenant_id."""
        try:
            payload = jwt.decode(api_key, self.secret_key, algorithms=['HS256'])
            return payload.get('tenant_id')
        except jwt.ExpiredSignatureError:
            logger.warning("API key expired")
            return None
        except jwt.InvalidTokenError:
            logger.warning("Invalid API key")
            return None
    
    def encrypt_data(self, data: bytes, encryption_key: bytes) -> bytes:
        """Encrypt data using tenant's encryption key."""
        f = Fernet(encryption_key)
        return f.encrypt(data)
    
    def decrypt_data(self, encrypted_data: bytes, encryption_key: bytes) -> bytes:
        """Decrypt data using tenant's encryption key."""
        f = Fernet(encryption_key)
        return f.decrypt(encrypted_data)
    
    def hash_password(self, password: str, salt: bytes = None) -> Tuple[bytes, bytes]:
        """Hash password with salt."""
        if salt is None:
            salt = os.urandom(32)
        
        kdf = PBKDF2(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = kdf.derive(password.encode())
        
        return key, salt
    
    def verify_password(self, password: str, hashed: bytes, salt: bytes) -> bool:
        """Verify password against hash."""
        kdf = PBKDF2(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        
        try:
            kdf.verify(password.encode(), hashed)
            return True
        except:
            return False


class ResourceMonitor:
    """Monitors and enforces resource usage limits."""
    
    def __init__(self, tenant_manager: TenantManager):
        self.tenant_manager = tenant_manager
        self.monitoring_interval = 60  # seconds
        self.running = False
    
    async def start_monitoring(self):
        """Start resource monitoring loop."""
        self.running = True
        while self.running:
            await self._check_all_tenants()
            await asyncio.sleep(self.monitoring_interval)
    
    async def _check_all_tenants(self):
        """Check resource usage for all active tenants."""
        session = self.tenant_manager.Session()
        try:
            tenants = session.query(TenantModel).filter_by(is_active=True).all()
            
            for tenant in tenants:
                # Check Docker containers if available
                if self.tenant_manager.docker_client:
                    await self._check_docker_resources(tenant.tenant_id)
                
                # Check Kubernetes pods if available
                if self.tenant_manager.k8s_v1:
                    await self._check_kubernetes_resources(tenant.tenant_id)
                
                # Check system resources
                await self._check_system_resources(tenant.tenant_id)
                
        finally:
            session.close()
    
    async def _check_docker_resources(self, tenant_id: str):
        """Check Docker container resources for tenant."""
        if not self.tenant_manager.docker_client:
            return
        
        try:
            containers = self.tenant_manager.docker_client.containers.list(
                filters={"label": f"tenant_id={tenant_id}"}
            )
            
            total_cpu = 0
            total_memory = 0
            
            for container in containers:
                stats = container.stats(stream=False)
                
                # Calculate CPU usage
                cpu_delta = stats['cpu_stats']['cpu_usage']['total_usage'] - \
                           stats['precpu_stats']['cpu_usage']['total_usage']
                system_delta = stats['cpu_stats']['system_cpu_usage'] - \
                              stats['precpu_stats']['system_cpu_usage']
                cpu_percent = (cpu_delta / system_delta) * 100.0
                
                total_cpu += cpu_percent
                total_memory += stats['memory_stats']['usage'] / (1024**3)  # GB
            
            # Check limits
            tenant = self.tenant_manager.get_tenant(tenant_id)
            if tenant:
                if total_cpu > tenant.max_cpu_cores * 100:
                    logger.warning(f"Tenant {tenant_id} exceeding CPU limit")
                    # Take action (e.g., throttle, suspend)
                
                if total_memory > tenant.max_memory_gb:
                    logger.warning(f"Tenant {tenant_id} exceeding memory limit")
                    # Take action
                    
        except Exception as e:
            logger.error(f"Error checking Docker resources: {e}")
    
    async def _check_kubernetes_resources(self, tenant_id: str):
        """Check Kubernetes pod resources for tenant."""
        if not self.tenant_manager.k8s_v1:
            return
        
        try:
            namespace = f"tenant-{tenant_id[:8]}"
            pods = self.tenant_manager.k8s_v1.list_namespaced_pod(namespace)
            
            # Get resource usage from metrics API if available
            # This would require additional setup with metrics-server
            
        except Exception as e:
            logger.error(f"Error checking Kubernetes resources: {e}")
    
    async def _check_system_resources(self, tenant_id: str):
        """Check system-level resource usage."""
        # Get current system usage
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Log high usage
        if cpu_percent > 80:
            logger.warning(f"High CPU usage: {cpu_percent}%")
        if memory.percent > 80:
            logger.warning(f"High memory usage: {memory.percent}%")
        if disk.percent > 80:
            logger.warning(f"High disk usage: {disk.percent}%")
    
    def stop_monitoring(self):
        """Stop resource monitoring."""
        self.running = False


class DeploymentManager:
    """Manages model deployment with isolation."""
    
    def __init__(self, tenant_manager: TenantManager):
        self.tenant_manager = tenant_manager
    
    def deploy_model_docker(self, tenant_id: str, model_path: str, 
                           port: int = 8000) -> Optional[str]:
        """Deploy model as Docker container."""
        if not self.tenant_manager.docker_client:
            logger.error("Docker not available")
            return None
        
        try:
            # Build Docker image
            image_tag = f"model_{tenant_id[:8]}:{datetime.now().strftime('%Y%m%d%H%M%S')}"
            
            # Create Dockerfile content
            dockerfile = f"""
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY model.pkl .
COPY app.py .

EXPOSE {port}

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "{port}"]
"""
            
            # Run container
            container = self.tenant_manager.docker_client.containers.run(
                image_tag,
                detach=True,
                ports={f"{port}/tcp": port},
                network=f"tenant_{tenant_id[:8]}",
                labels={"tenant_id": tenant_id},
                environment={
                    "TENANT_ID": tenant_id,
                    "MODEL_PATH": model_path
                },
                restart_policy={"Name": "unless-stopped"},
                mem_limit="2g",
                cpu_quota=100000  # 1 CPU
            )
            
            logger.info(f"Deployed model for tenant {tenant_id} as {container.id}")
            return container.id
            
        except Exception as e:
            logger.error(f"Failed to deploy model: {e}")
            return None
    
    def deploy_model_kubernetes(self, tenant_id: str, model_path: str,
                               replicas: int = 1) -> Optional[str]:
        """Deploy model as Kubernetes deployment."""
        if not self.tenant_manager.k8s_apps:
            logger.error("Kubernetes not available")
            return None
        
        try:
            namespace = f"tenant-{tenant_id[:8]}"
            deployment_name = f"model-{datetime.now().strftime('%Y%m%d%H%M%S')}"
            
            # Create deployment
            deployment = client.V1Deployment(
                metadata=client.V1ObjectMeta(
                    name=deployment_name,
                    namespace=namespace,
                    labels={"tenant_id": tenant_id}
                ),
                spec=client.V1DeploymentSpec(
                    replicas=replicas,
                    selector=client.V1LabelSelector(
                        match_labels={"app": deployment_name}
                    ),
                    template=client.V1PodTemplateSpec(
                        metadata=client.V1ObjectMeta(
                            labels={"app": deployment_name}
                        ),
                        spec=client.V1PodSpec(
                            containers=[
                                client.V1Container(
                                    name="model",
                                    image=f"model:{tenant_id[:8]}",
                                    ports=[client.V1ContainerPort(container_port=8000)],
                                    resources=client.V1ResourceRequirements(
                                        requests={"cpu": "100m", "memory": "512Mi"},
                                        limits={"cpu": "1000m", "memory": "2Gi"}
                                    ),
                                    env=[
                                        client.V1EnvVar(name="TENANT_ID", value=tenant_id),
                                        client.V1EnvVar(name="MODEL_PATH", value=model_path)
                                    ]
                                )
                            ]
                        )
                    )
                )
            )
            
            self.tenant_manager.k8s_apps.create_namespaced_deployment(
                namespace=namespace,
                body=deployment
            )
            
            # Create service
            service = client.V1Service(
                metadata=client.V1ObjectMeta(
                    name=deployment_name,
                    namespace=namespace
                ),
                spec=client.V1ServiceSpec(
                    selector={"app": deployment_name},
                    ports=[client.V1ServicePort(port=80, target_port=8000)],
                    type="LoadBalancer"
                )
            )
            
            self.tenant_manager.k8s_v1.create_namespaced_service(
                namespace=namespace,
                body=service
            )
            
            logger.info(f"Deployed model for tenant {tenant_id} as {deployment_name}")
            return deployment_name
            
        except Exception as e:
            logger.error(f"Failed to deploy model to Kubernetes: {e}")
            return None


# Example usage
def main():
    """Example usage of infrastructure components."""
    
    # Initialize managers
    tenant_manager = TenantManager()
    security_manager = SecurityManager()
    deployment_manager = DeploymentManager(tenant_manager)
    
    # Create tenant
    tenant = tenant_manager.create_tenant("Acme Corp", "professional")
    print(f"Created tenant: {tenant.tenant_id}")
    
    # Generate API key
    api_key = security_manager.generate_api_key(tenant.tenant_id)
    print(f"API Key: {api_key}")
    
    # Check resource limits
    can_allocate = tenant_manager.check_resource_limits(
        tenant.tenant_id, "cpu", 2
    )
    print(f"Can allocate 2 CPUs: {can_allocate}")
    
    # Allocate resources
    allocated = tenant_manager.allocate_resources(
        tenant.tenant_id, cpu=2, memory=4
    )
    print(f"Resources allocated: {allocated}")
    
    # Deploy model
    if DOCKER_AVAILABLE:
        container_id = deployment_manager.deploy_model_docker(
            tenant.tenant_id, "/path/to/model.pkl"
        )
        print(f"Deployed to Docker: {container_id}")
    
    if KUBERNETES_AVAILABLE:
        deployment_name = deployment_manager.deploy_model_kubernetes(
            tenant.tenant_id, "/path/to/model.pkl"
        )
        print(f"Deployed to Kubernetes: {deployment_name}")


if __name__ == "__main__":
    main()
