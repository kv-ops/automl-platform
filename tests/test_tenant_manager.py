import uuid
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from automl_platform.api.infrastructure import TenantManager
from automl_platform.auth import QuotaService
from automl_platform.plans import PlanType


class DummyRedis:
    def __init__(self):
        self.storage = {}
        self.expirations = {}

    def get(self, key):
        value = self.storage.get(key)
        if value is None:
            return None
        return str(value).encode()

    def incrby(self, key, amount):
        self.storage[key] = self.storage.get(key, 0) + amount
        return self.storage[key]

    def ttl(self, key):
        return self.expirations.get(key, -1)

    def expire(self, key, seconds):
        self.expirations[key] = seconds


@pytest.fixture()
def tenant_manager(tmp_path: Path) -> TenantManager:
    db_path = tmp_path / "tenants.db"
    return TenantManager(db_url=f"sqlite:///{db_path}")


def test_create_and_retrieve_tenant(tenant_manager: TenantManager):
    tenant = tenant_manager.create_tenant("Acme Corp", plan=PlanType.PROFESSIONAL.value)

    # Identifiers should be exposed as strings while remaining valid UUIDs.
    assert isinstance(tenant.id, str)
    assert tenant.id == tenant.tenant_id
    assert uuid.UUID(tenant.id)  # Does not raise

    # Plan information should surface through plan_type while keeping legacy alias.
    assert tenant.plan_type == PlanType.PROFESSIONAL.value
    assert tenant.plan == tenant.plan_type

    # Fetch tenant back using the string identifier.
    retrieved = tenant_manager.get_tenant(tenant.id)
    assert retrieved is not None
    assert retrieved.id == tenant.id
    assert retrieved.plan_type == tenant.plan_type

    # Fetch tenant using a UUID instance to ensure transparent conversion.
    retrieved_with_uuid = tenant_manager.get_tenant(uuid.UUID(tenant.id))
    assert retrieved_with_uuid is not None
    assert retrieved_with_uuid.id == tenant.id


def test_quota_service_with_tenant_config(tenant_manager: TenantManager):
    tenant = tenant_manager.create_tenant("Quota Corp", plan=PlanType.FREE.value)

    quota_service = QuotaService(MagicMock(), DummyRedis())

    assert quota_service.check_quota(tenant, "api_calls", 500)

    quota_service.consume_quota(tenant, "api_calls", 999)
    assert quota_service.check_quota(tenant, "api_calls", 2) is False

    usage_key = f"usage:{tenant.id}:api_calls"
    assert usage_key in quota_service.redis.storage
    assert quota_service.redis.storage[usage_key] == 999
