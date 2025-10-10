import sys
import types

import pytest

# Provide lightweight stubs for heavy modules imported by auth_middleware.
if "automl_platform.auth" not in sys.modules:
    auth_stub = types.ModuleType("automl_platform.auth")

    class _StubService:
        def __init__(self, *args, **kwargs):
            pass

    auth_stub.TokenService = _StubService
    auth_stub.RBACService = _StubService
    auth_stub.QuotaService = _StubService
    auth_stub.AuditService = _StubService
    auth_stub.get_db = lambda *args, **kwargs: None

    class _StubUser:
        id = None
        tenant_id = None

    auth_stub.User = _StubUser
    sys.modules["automl_platform.auth"] = auth_stub

if "automl_platform.sso_service" not in sys.modules:
    sso_stub = types.ModuleType("automl_platform.sso_service")

    class _StubSSOService:
        def __init__(self, *args, **kwargs):
            pass

    sso_stub.SSOService = _StubSSOService
    sys.modules["automl_platform.sso_service"] = sso_stub

if "automl_platform.audit_service" not in sys.modules:
    audit_stub = types.ModuleType("automl_platform.audit_service")

    class _StubAuditService:
        def __init__(self, *args, **kwargs):
            pass

    class _StubEnum:
        pass

    audit_stub.AuditService = _StubAuditService
    audit_stub.AuditEventType = _StubEnum
    audit_stub.AuditSeverity = _StubEnum
    sys.modules["automl_platform.audit_service"] = audit_stub

if "automl_platform.rgpd_compliance_service" not in sys.modules:
    rgpd_stub = types.ModuleType("automl_platform.rgpd_compliance_service")

    class _StubRGPDComplianceService:
        def __init__(self, *args, **kwargs):
            pass

    class _StubConsentType:
        pass

    class _StubGDPRRequestType:
        pass

    rgpd_stub.RGPDComplianceService = _StubRGPDComplianceService
    rgpd_stub.ConsentType = _StubConsentType
    rgpd_stub.GDPRRequestType = _StubGDPRRequestType
    sys.modules["automl_platform.rgpd_compliance_service"] = rgpd_stub

import automl_platform.auth_middleware as auth_middleware_module
from automl_platform.auth_middleware import UnifiedAuthMiddleware
from automl_platform.config import AutoMLConfig, APIConfig


class DummyApp:
    async def __call__(self, scope, receive, send):
        raise RuntimeError("Dummy app should not be invoked during initialization")


def _setup_service_stubs(monkeypatch):
    calls = {}

    class DummyRedisClient:
        pass

    def fake_from_url(url):
        calls["url"] = url
        return DummyRedisClient()

    monkeypatch.setattr(auth_middleware_module.redis, "from_url", fake_from_url)
    monkeypatch.setattr(auth_middleware_module, "TokenService", lambda: types.SimpleNamespace())

    class DummySSOService:
        def __init__(self, redis_client):
            self.redis_client = redis_client

    monkeypatch.setattr(auth_middleware_module, "SSOService", DummySSOService)

    class DummyAuditService:
        def __init__(self, *args, **kwargs):
            self.redis_client = kwargs.get("redis_client")

    monkeypatch.setattr(auth_middleware_module, "AuditService", DummyAuditService)

    class DummyRGPDComplianceService:
        def __init__(self, *args, **kwargs):
            self.redis_client = kwargs.get("redis_client")
            self.audit_service = kwargs.get("audit_service")

    monkeypatch.setattr(auth_middleware_module, "RGPDComplianceService", DummyRGPDComplianceService)

    return calls


def test_unified_auth_middleware_uses_api_config_redis_url(monkeypatch):
    calls = _setup_service_stubs(monkeypatch)
    config = AutoMLConfig(
        api=APIConfig(
            redis_url="redis://config-host:6380",
            sso_realm_url="https://sso.example.com/realms/demo",
        )
    )

    middleware = UnifiedAuthMiddleware(DummyApp(), config=config)

    assert calls["url"] == "redis://config-host:6380"
    assert middleware.redis_client is not None


def test_unified_auth_middleware_falls_back_to_env(monkeypatch):
    calls = _setup_service_stubs(monkeypatch)
    monkeypatch.setenv("REDIS_URL", "redis://env-host:6390")
    config = AutoMLConfig(api=APIConfig(redis_url=None))

    UnifiedAuthMiddleware(DummyApp(), config=config)

    assert calls["url"] == "redis://env-host:6390"


def test_unified_auth_middleware_requires_redis_url(monkeypatch):
    _setup_service_stubs(monkeypatch)
    monkeypatch.delenv("REDIS_URL", raising=False)

    config = AutoMLConfig(api=APIConfig(redis_url=None))

    with pytest.raises(RuntimeError):
        UnifiedAuthMiddleware(DummyApp(), config=config)


def test_api_config_validates_redis_url_scheme(monkeypatch):
    monkeypatch.setenv("REDIS_URL", "redis://valid-host:6379")

    # Valid scheme should not raise
    APIConfig(redis_url="redis://cache")
    APIConfig(redis_url="rediss://secure-cache")

    with pytest.raises(ValueError):
        APIConfig(redis_url="http://not-redis")


def test_automl_config_roundtrip_includes_redis_url(tmp_path, monkeypatch):
    monkeypatch.delenv("REDIS_URL", raising=False)
    config = AutoMLConfig(api=APIConfig(redis_url="rediss://cache"))

    serialized = config.to_dict()
    assert serialized["api"]["redis_url"] == "rediss://cache"

    config_path = tmp_path / "config.yaml"
    config_path.write_text("api:\n  redis_url: rediss://cache\n")

    reloaded = AutoMLConfig.from_yaml(str(config_path))

    assert reloaded.api.redis_url == "rediss://cache"
