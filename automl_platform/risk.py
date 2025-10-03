"""Common risk level utilities shared across AutoML components."""

from __future__ import annotations

from enum import Enum
from typing import Any, Callable, Optional, Tuple


class RiskLevel(str, Enum):
    """Normalized risk levels used throughout data quality workflows."""

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value

    @classmethod
    def normalize(cls, value: Any, *, field_name: str) -> "RiskLevel":
        """Normalize input into a valid ``RiskLevel`` with backward compatibility."""

        if isinstance(value, cls):
            return value

        if isinstance(value, bool):
            return cls.HIGH if value else cls.NONE

        if value is None:
            return cls.NONE

        if isinstance(value, str):
            normalized = value.strip().lower()
            for level in cls:
                if normalized == level.value:
                    return level

        raise ValueError(
            f"{field_name} must be one of {[level.value for level in cls]}, got {value!r}"
        )

    @classmethod
    def _coerce(cls, value: Any) -> Optional["RiskLevel"]:
        """Best-effort conversion for comparison helpers."""

        if isinstance(value, cls):
            return value

        if isinstance(value, str):
            try:
                return cls.normalize(value, field_name="risk_level")
            except ValueError:
                return None

        return None

    @classmethod
    def _order(cls) -> Tuple["RiskLevel", ...]:
        return (cls.NONE, cls.LOW, cls.MEDIUM, cls.HIGH)

    def _compare(self, other: Any, op: Callable[[int, int], bool]) -> Any:
        other_level = self._coerce(other)
        if other_level is None:
            return NotImplemented

        order = self._order()
        return op(order.index(self), order.index(other_level))

    def __lt__(self, other: Any) -> Any:
        return self._compare(other, lambda a, b: a < b)

    def __le__(self, other: Any) -> Any:
        return self._compare(other, lambda a, b: a <= b)

    def __gt__(self, other: Any) -> Any:
        return self._compare(other, lambda a, b: a > b)

    def __ge__(self, other: Any) -> Any:
        return self._compare(other, lambda a, b: a >= b)

    @property
    def is_critical(self) -> bool:
        """Return ``True`` when the risk level should be treated as critical."""

        return self in (self.MEDIUM, self.HIGH)

    @property
    def color(self) -> str:
        """Provide a semantic color for UI/logging contexts."""

        return {
            self.NONE: "green",
            self.LOW: "yellow",
            self.MEDIUM: "orange",
            self.HIGH: "red",
        }[self]


__all__ = ["RiskLevel"]

