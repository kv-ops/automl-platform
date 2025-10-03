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

# Invalid values default to ``RiskLevel.NONE`` (with a DEBUG log)
payload = {"drift_risk": "SUPER_HIGH"}
normalized = RiskLevel.from_string(payload["drift_risk"])  # -> RiskLevel.NONE
```

## Bulk migration helper

```python
import json
from pathlib import Path

from automl_platform.data_quality_agent import DataQualityAssessment

legacy_payload = json.loads(Path("legacy_assessment.json").read_text())
assessment = DataQualityAssessment(**legacy_payload)

# Export normalized risks for downstream systems
Path("normalized_assessment.json").write_text(json.dumps(assessment.to_dict(), indent=2))
```

## Additional Helpers
- Use `RiskLevel.from_string(...)` to normalize legacy payloads or user input.
- Call `assessment.to_dict()` to obtain JSON-serializable payloads where risk levels are exported as lowercase strings.
- Boolean or legacy string inputs can be normalized with `RiskLevel.from_string(..., default=RiskLevel.NONE)`.
- Invalid strings (e.g. ``"super_high"``) and unsupported types fall back to `RiskLevel.NONE` unless a custom default is supplied. The fallback is logged at `DEBUG` level so noisy payloads can be traced without impacting production logs.
- A quick ``python -m timeit "from automl_platform.risk import RiskLevel; RiskLevel.from_string('HIGH')"`` benchmark processes ~160k conversions/sec on a single core, so no additional caching is required for typical workloads.
