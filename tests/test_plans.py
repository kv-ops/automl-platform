"""Tests for shared plan utilities."""

import pytest

from automl_platform.plans import PlanType, normalize_plan_type, plan_level, is_plan_at_least


class TestNormalizePlanType:
    """Unit tests for plan normalization helper."""

    def test_accepts_enum(self):
        assert normalize_plan_type(PlanType.STARTER) is PlanType.STARTER

    def test_accepts_string_case_insensitive(self):
        assert normalize_plan_type("Professional") is PlanType.PROFESSIONAL
        assert normalize_plan_type(" pro ") is PlanType.PRO

    def test_unknown_plan_with_default(self):
        assert normalize_plan_type("unknown", default=PlanType.FREE) is PlanType.FREE

    def test_unknown_plan_raises_without_default(self):
        with pytest.raises(ValueError):
            normalize_plan_type("unknown", default=None)


class TestPlanLevel:
    """Tests for plan hierarchy helpers."""

    def test_level_ordering(self):
        assert plan_level(PlanType.FREE) < plan_level(PlanType.TRIAL)
        assert plan_level(PlanType.TRIAL) < plan_level(PlanType.STARTER)
        assert plan_level(PlanType.STARTER) < plan_level(PlanType.PRO)
        assert plan_level(PlanType.PRO) < plan_level(PlanType.PROFESSIONAL)
        assert plan_level(PlanType.PROFESSIONAL) < plan_level(PlanType.ENTERPRISE)
        assert plan_level(PlanType.ENTERPRISE) < plan_level(PlanType.CUSTOM)

    def test_default_for_invalid_plan(self):
        assert plan_level("unknown") == 0
        assert plan_level("unknown", default=-1) == -1

    def test_is_plan_at_least(self):
        assert is_plan_at_least(PlanType.PROFESSIONAL, PlanType.PRO)
        assert not is_plan_at_least(PlanType.STARTER, PlanType.PRO)
        assert is_plan_at_least("enterprise", PlanType.PROFESSIONAL)
