# Release Notes

## Nested configuration migration (Unreleased)

- **Change**: The `database` and `security` settings are now expected to live in
  dedicated nested sections within `config.yaml`. Flat keys such as
  `database_url` and `secret_key` are deprecated and will be removed in a future
  release.
- **Action required**: Update your configuration files to move the legacy flat
  keys into the nested blocks:

  ```yaml
  database:
    url: postgresql://user:password@host/db

  security:
    secret_key: your-secret
  ```

- **Backward compatibility**: For the time being the loader automatically maps
  the flat keys to the nested dataclasses and emits warnings so operators can
  track down outdated files. When both formats are present, the nested values
  win to avoid silent regressions.
