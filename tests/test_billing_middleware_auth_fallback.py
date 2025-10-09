"""Behavioral tests for the billing middleware auth dependency handling."""

import pytest
from fastapi import FastAPI

from automl_platform.api import billing_middleware


class _DummyUsageTracker:
    def track_api_call(self, *args, **kwargs):  # pragma: no cover - simple stub
        return True


class _DummyBillingManager:
    def __init__(self):
        self.usage_tracker = _DummyUsageTracker()


def test_billing_middleware_requires_auth_by_default(monkeypatch):
    """Without auth and without explicit opt-in we should fail fast."""

    monkeypatch.setattr(billing_middleware, "get_current_user", None)
    monkeypatch.delenv("AUTOML_BILLING_ALLOW_NO_AUTH", raising=False)

    app = FastAPI()

    with pytest.raises(RuntimeError) as exc:
        billing_middleware.BillingMiddleware(app, _DummyBillingManager())

    assert "requires the auth module" in str(exc.value)


def test_billing_middleware_can_opt_in_to_header_fallback(monkeypatch):
    """Operators can explicitly opt-in to the header-based fallback when needed."""

    monkeypatch.setattr(billing_middleware, "get_current_user", None)
    monkeypatch.setenv("AUTOML_BILLING_ALLOW_NO_AUTH", "1")

    app = FastAPI()

    middleware = billing_middleware.BillingMiddleware(app, _DummyBillingManager())

    assert isinstance(middleware, billing_middleware.BillingMiddleware)
