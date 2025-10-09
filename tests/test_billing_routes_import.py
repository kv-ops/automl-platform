"""Tests ensuring the billing API routes can be imported successfully."""

import importlib
import importlib.metadata as importlib_metadata
import sys
import types

import pydantic.networks as pydantic_networks


# The authentication module depends on ``email_validator`` for FastAPI models. Provide a
# lightweight stub so the import can succeed in minimal environments where the optional
# dependency is not installed.
if "email_validator" not in sys.modules:  # pragma: no cover - setup code
    email_validator = types.ModuleType("email_validator")

    class EmailNotValidError(Exception):
        """Fallback error raised by the stub validator."""

    def validate_email(email, *_, **__):  # type: ignore[override]
        return {"email": email, "domain": email.split("@")[-1] if "@" in email else ""}

    email_validator.EmailNotValidError = EmailNotValidError
    email_validator.validate_email = validate_email
    sys.modules["email_validator"] = email_validator

    _real_version = importlib_metadata.version

    def _import_email_validator_stub():
        pydantic_networks.email_validator = email_validator
        return email_validator

    pydantic_networks.import_email_validator = _import_email_validator_stub

    def _version_stub(name: str) -> str:
        if name == "email-validator":
            return "2.0.0"
        return _real_version(name)

    importlib_metadata.version = _version_stub


def test_billing_router_importable():
    """Importing the billing router should not raise and exposes the expected prefix."""
    module = importlib.import_module("automl_platform.api.billing_routes")

    assert hasattr(module, "billing_router"), "billing_router should be defined"
    router = module.billing_router

    assert router.prefix == "/api/billing"
    # Ensure at least one route is registered so the router isn't empty
    assert router.routes, "billing router should register endpoints"


def test_billing_router_dependencies_resolve():
    """The critical auth dependencies should be pulled from the top-level auth module."""
    module = importlib.import_module("automl_platform.api.billing_routes")

    assert hasattr(module, "get_current_user"), "get_current_user should be imported"
    assert hasattr(module, "User"), "User model should be available"
    assert hasattr(module, "require_permission"), "require_permission should be available"

    # Confirm these attributes come from the shared auth module
    auth_module = importlib.import_module("automl_platform.auth")
    assert module.get_current_user is auth_module.get_current_user
    assert module.require_permission is auth_module.require_permission
    assert module.User is auth_module.User
