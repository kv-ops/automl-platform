"""Tests for worker configuration interactions with the global config loader."""

import logging
import yaml

from automl_platform.config import load_config


def test_load_config_exposes_database_and_security(tmp_path):
    """load_config should hydrate nested database and security dataclasses."""
    config_path = tmp_path / "config.yaml"
    payload = {
        'database': {'url': 'sqlite:///:memory:'},
        'security': {'secret_key': 'super-secret-key'},
    }

    with config_path.open('w') as handle:
        yaml.safe_dump(payload, handle)

    config = load_config(filepath=str(config_path))

    assert config.database.url == 'sqlite:///:memory:'
    assert config.security.secret_key == 'super-secret-key'


def test_load_config_backfills_legacy_flat_keys(tmp_path, caplog):
    """Legacy flat keys should still populate nested configuration objects."""
    config_path = tmp_path / "legacy.yaml"
    payload = {
        'database_url': 'sqlite:///legacy.db',
        'secret_key': 'legacy-secret',
    }

    with config_path.open('w') as handle:
        yaml.safe_dump(payload, handle)

    with caplog.at_level(logging.WARNING):
        config = load_config(filepath=str(config_path))

    assert config.database.url == 'sqlite:///legacy.db'
    assert config.security.secret_key == 'legacy-secret'
    assert any("legacy 'database_url'" in record.getMessage() for record in caplog.records)
    assert any("legacy 'secret_key'" in record.getMessage() for record in caplog.records)


def test_nested_sections_take_precedence_over_flat_keys(tmp_path, caplog):
    """When both formats are provided, nested configuration should win with a warning."""
    config_path = tmp_path / "mixed.yaml"
    payload = {
        'database': {'url': 'postgresql://nested/db'},
        'security': {'secret_key': 'nested-secret'},
        'database_url': 'sqlite:///legacy.db',
        'secret_key': 'legacy-secret',
    }

    with config_path.open('w') as handle:
        yaml.safe_dump(payload, handle)

    with caplog.at_level(logging.WARNING):
        config = load_config(filepath=str(config_path))

    assert config.database.url == 'postgresql://nested/db'
    assert config.security.secret_key == 'nested-secret'
    assert config.database_url == 'postgresql://nested/db'
    assert config.secret_key == 'nested-secret'
    assert any("ignoring the flat key" in record.getMessage() for record in caplog.records)
