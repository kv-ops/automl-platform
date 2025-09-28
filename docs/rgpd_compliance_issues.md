# RGPD Compliance Service Inconsistencies

During the investigation of the RGPD compliance stack, several mismatches between the implementation and its expected behaviour (as exercised by the test-suite) were identified:

## 1. Redis namespace exposure
- **Issue (fixed):** `core.rgpd_compliance_service` did not expose a `redis` namespace when the optional dependency was absent. The tests patch `core.rgpd_compliance_service.redis.Redis`, which previously raised an `AttributeError`.
- **Resolution:** The module now re-exports a stub namespace that always exposes a `Redis` attribute so that dependency injection works even without the real package. 【F:automl_platform/rgpd_compliance_service.py†L8-L36】【F:core/rgpd_compliance_service.py†L11-L134】

## 2. Request metadata handling
- **Symptom:** Multiple tests fail with `TypeError: RGPDRequest.__init__() got an unexpected keyword argument 'metadata_json'` when `RGPDComplianceService.create_request` is executed. 【3863d8†L6-L63】
- **Root cause:** The `RGPDRequest` dataclass only defines a `metadata` field, while the service instantiates it using the non-existent `metadata_json` argument. 【F:automl_platform/rgpd_compliance_service.py†L108-L145】【F:automl_platform/rgpd_compliance_service.py†L512-L552】

## 3. Consent metadata handling
- **Symptom:** Similar `TypeError` exceptions are raised for `ConsentRecord` during consent creation. 【3863d8†L6-L63】
- **Root cause:** `ConsentRecord` does not define a `metadata_json` field, yet the service populates the dataclass with this argument. 【F:automl_platform/rgpd_compliance_service.py†L154-L187】【F:automl_platform/rgpd_compliance_service.py†L698-L736】

## 4. Audit log keyword mismatches
- **Symptom:** Assertions such as `self.mock_audit.log_event.assert_called_with(..., metadata=ANY)` fail because no call is registered with the expected keyword. 【3863d8†L63-L126】
- **Root cause:** The service calls `log_event` with the keyword `metadata_json`, but the stubbed audit service and the tests both expect the keyword `metadata`. 【F:automl_platform/rgpd_compliance_service.py†L509-L548】【F:automl_platform/rgpd_compliance_service.py†L756-L844】【F:automl_platform/rgpd_compliance_service.py†L358-L400】

## 5. Redis client initialisation
- **Observation:** Even when Redis is patched in tests, `RGPDComplianceService` only creates a client if a `redis_url` is present in the configuration. This prevents the patched client from being used, so caching code paths relying on Redis never execute (hence expectations such as `mock_redis.setex.assert_called_once` fail). 【F:automl_platform/rgpd_compliance_service.py†L403-L476】【3863d8†L6-L63】

These inconsistencies explain the majority of the failing unit tests in `tests/test_rgpd_compliance_service.py` and should be addressed to bring the RGPD compliance service in line with the expected contract.
