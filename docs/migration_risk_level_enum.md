# Migration Guide: RiskLevel Enum

## Breaking Changes
- `drift_risk` and `target_leakage_risk` now return the `RiskLevel` enum instead of `bool`/`str` values.

## Migration
```python
from automl_platform.risk import RiskLevel

# Old code
if assessment.drift_risk == "high":
    ...

# New code (recommended)
if assessment.drift_risk >= RiskLevel.MEDIUM:
    ...

# Backwards compatible
if assessment.drift_risk.value == "high":
    ...

# Normalizing legacy payloads
payload = {"drift_risk": "HIGH", "target_leakage_risk": True}

drift = RiskLevel.from_string(payload["drift_risk"])       # -> RiskLevel.HIGH
leakage = RiskLevel.from_string(payload["target_leakage_risk"])  # -> RiskLevel.HIGH
```

## Additional Helpers
- Use `RiskLevel.from_string(...)` to normalize legacy payloads or user input.
- Call `assessment.to_dict()` to obtain JSON-serializable payloads where risk levels are exported as lowercase strings.
- Boolean or legacy string inputs can be normalized with `RiskLevel.from_string(..., default=RiskLevel.NONE)`.
