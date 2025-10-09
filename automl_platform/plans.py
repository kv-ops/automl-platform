"""Shared subscription plan metadata and helpers.

This module centralizes the definition of supported pricing plans and the
associated hierarchy utilities that are consumed across auth, billing, and
other services.  Consolidating this logic prevents individual modules from
falling out of sync when new plans are introduced.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional, Union


class PlanType(str, Enum):
    """Supported subscription plan tiers."""

    FREE = "free"
    TRIAL = "trial"
    STARTER = "starter"
    PRO = "pro"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"
    CUSTOM = "custom"


# Plan precedence used for upgrade validation and access checks.
PLAN_HIERARCHY = {
    PlanType.FREE: 0,
    PlanType.TRIAL: 1,
    PlanType.STARTER: 2,
    PlanType.PRO: 3,
    PlanType.PROFESSIONAL: 4,
    PlanType.ENTERPRISE: 5,
    PlanType.CUSTOM: 6,
}


PlanValue = Union[str, PlanType, None]


def normalize_plan_type(plan: PlanValue, *, default: Optional[PlanType] = PlanType.FREE) -> PlanType:
    """Normalize a plan value coming from requests, storage, or enums.

    Args:
        plan: Raw plan representation. Accepts :class:`PlanType`, strings (case
            insensitive), or ``None``.
        default: Fallback plan to use when the provided value cannot be
            resolved. ``None`` can be passed to surface invalid plans to the
            caller.

    Returns:
        The resolved :class:`PlanType` instance.

    Raises:
        ValueError: If the plan value cannot be resolved and ``default`` is
            ``None``.
    """

    if isinstance(plan, PlanType):
        return plan

    if isinstance(plan, str):
        normalized = plan.strip().lower()
        for candidate in PlanType:
            if candidate.value == normalized:
                return candidate

    if default is None:
        raise ValueError(f"Unknown plan value: {plan!r}")

    return default


def plan_level(plan: PlanValue, *, default: int = 0) -> int:
    """Return the numeric level for a plan for comparisons.

    Args:
        plan: Plan representation to evaluate.
        default: Level to use when the plan cannot be resolved.
    """

    try:
        resolved = normalize_plan_type(plan, default=None)
    except ValueError:
        return default

    return PLAN_HIERARCHY.get(resolved, default)


def is_plan_at_least(plan: PlanValue, minimum: PlanValue) -> bool:
    """Check whether ``plan`` meets or exceeds ``minimum`` in the hierarchy."""

    return plan_level(plan) >= plan_level(minimum)
