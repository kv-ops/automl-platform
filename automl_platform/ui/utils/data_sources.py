"""Helpers for normalising and migrating data source identifiers."""

from __future__ import annotations

import re
from typing import Dict, Optional

_ZERO_WIDTH_CHARACTERS = {
    "\u200b",  # zero-width space
    "\u200c",  # zero-width non-joiner
    "\u200d",  # zero-width joiner
    "\ufeff",  # zero-width no-break space / BOM
}

_NON_BREAKING_SPACES = {"\u202f", "\xa0", "\u2060"}


def normalize_data_source_label(label: str) -> str:
    """Normalise data source labels to make emoji/spacing variants comparable."""
    sanitized = label
    for character in _NON_BREAKING_SPACES:
        sanitized = sanitized.replace(character, " ")
    for character in _ZERO_WIDTH_CHARACTERS:
        sanitized = sanitized.replace(character, "")
    sanitized = sanitized.replace("\ufe0f", "")  # emoji variation selector
    sanitized = sanitized.strip()
    sanitized = re.sub(r"^([^\w\s])(?=\w)", r"\1 ", sanitized)
    sanitized = re.sub(r"\s+", " ", sanitized)
    return sanitized.casefold()


def coerce_data_source_identifier(
    raw_value: Optional[str],
    fallback: str,
    label_by_identifier: Dict[str, str],
) -> str:
    """Coerce legacy label-based values to the new identifier format."""
    if isinstance(raw_value, str):
        if raw_value in label_by_identifier:
            return raw_value

        normalized = normalize_data_source_label(raw_value)
        normalized_compact = normalized.replace(" ", "")
        for identifier, label in label_by_identifier.items():
            normalized_label = normalize_data_source_label(label)
            if normalized_label == normalized:
                return identifier
            if normalized_label.replace(" ", "") == normalized_compact:
                return identifier

    return fallback


__all__ = ["normalize_data_source_label", "coerce_data_source_identifier"]
