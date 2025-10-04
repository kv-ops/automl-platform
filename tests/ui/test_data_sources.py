"""Tests for UI data source utilities."""

from automl_platform.ui.utils.data_sources import (
    coerce_data_source_identifier,
    normalize_data_source_label,
)


def test_normalize_data_source_label_handles_emoji_variants():
    base = normalize_data_source_label("🤝 CRM")
    assert normalize_data_source_label("🤝\u202fCRM") == base
    assert normalize_data_source_label("🤝\u2060CRM") == base
    assert normalize_data_source_label("🤝CRM") == base


def test_coerce_data_source_identifier_maps_legacy_labels():
    label_by_identifier = {
        "file": "📁 Fichier local",
        "crm": "🤝 CRM",
    }
    fallback = "file"

    assert coerce_data_source_identifier("crm", fallback, label_by_identifier) == "crm"
    assert coerce_data_source_identifier("🤝 CRM", fallback, label_by_identifier) == "crm"
    assert coerce_data_source_identifier("🤝\u202fCRM", fallback, label_by_identifier) == "crm"
    assert coerce_data_source_identifier("🤝CRM", fallback, label_by_identifier) == "crm"
    assert coerce_data_source_identifier("🤝\u2060CRM", fallback, label_by_identifier) == "crm"
    assert coerce_data_source_identifier(None, fallback, label_by_identifier) == "file"
    assert coerce_data_source_identifier("inconnue", fallback, label_by_identifier) == "file"
