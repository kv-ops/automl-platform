"""Common risk level utilities shared across AutoML components."""

from __future__ import annotations

from enum import Enum
import logging
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
    def _convert(cls, value: Any) -> Optional["RiskLevel"]:
        """Convert raw values into a ``RiskLevel`` when possible."""

        if isinstance(value, cls):
            return value

        if isinstance(value, bool):
            return cls.HIGH if value else cls.NONE

        if value is None:
            return None

        if isinstance(value, str):
            normalized = value.strip().lower()
            for level in cls:
                if normalized == level.value:
                    return level

        return None

    @classmethod
    def from_string(
        cls, value: Any, *, default: Optional["RiskLevel"] = None
    ) -> "RiskLevel":
        """Best-effort conversion to ``RiskLevel``.

        Parameters
        ----------
        value:
            The input value to normalize. Strings are matched case-insensitively.
            ``True``/``False`` values are treated as ``HIGH``/``NONE`` for backward
            compatibility with historical boolean leakage indicators.
        default:
            The level returned when the value cannot be mapped. When omitted the
            method returns ``RiskLevel.NONE`` for unknown inputs. The fallback is
            logged at ``DEBUG`` level for observability without polluting normal
            application logs.
        """

        level = cls._convert(value)
        if level is not None:
            return level

        if default is not None:
            logging.getLogger(__name__).debug(
                "RiskLevel.from_string falling back to default %s for value %r", default, value
            )
            return default

        logging.getLogger(__name__).debug(
            "RiskLevel.from_string falling back to RiskLevel.NONE for value %r", value
        )
        return cls.NONE

    @classmethod
    def normalize(cls, value: Any, *, field_name: str) -> "RiskLevel":
        """Normalize input into a valid ``RiskLevel`` with strict validation."""

        level = cls._convert(value)
        if level is None:
            raise ValueError(
                f"{field_name} must be one of {[risk.value for risk in cls]}, got {value!r}"
            )
        return level

    @classmethod
    def _coerce(cls, value: Any) -> Optional["RiskLevel"]:
        """Best-effort conversion for comparison helpers."""

        return cls._convert(value)

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

