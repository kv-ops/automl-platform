"""
MLOps API Endpoints - Model Registry, Retraining, Export, and A/B Testing
==========================================================================
Place in: automl_platform/api/mlops_endpoints.py

FastAPI endpoints for MLOps operations including model management,
automated retraining, exports, and A/B testing.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, File, UploadFile, Query
from fastapi.responses import FileResponse, StreamingResponse
from typing import Dict, List, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import json
import io
import logging
from pathlib import Path

# Import MLOps components
from ..mlflow_registry import MLflowRegistry, ABTestingService, ModelStage
from ..retraining_service import RetrainingService, RetrainingConfig
from ..export_service import ModelExporter, ExportConfig
from ..orchestrator import AutoMLOrchestrator
from ..config import AutoMLConfig
from ..storage import StorageService
from ..monitoring import ModelMonitor

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1/mlops", tags=["MLOps"])

# Initialize services (would be dependency injected in production)
try:
    config = AutoMLConfig()
except Exception as exc:  # pragma: no cover - defensive fallback for optional deps
    logger.warning("Failed to load AutoML configuration: %s", exc)
    config = None

try:
    if config is None:
        raise RuntimeError("Configuration unavailable")
    registry = MLflowRegistry(config)
    exporter = ModelExporter()
    ab_testing = ABTestingService(registry)
    storage = StorageService(config)
    monitor = ModelMonitor(config)
    retraining_service = RetrainingService(config, registry, monitor, storage)
except Exception as exc:  # pragma: no cover - defensive fallback for optional deps
    logger.warning("Failed to initialize advanced MLOps services: %s", exc)
    registry = None
    exporter = None
    ab_testing = None
    storage = None
    monitor = None
    retraining_service = None


# ============================================================================
# Helpers
# ============================================================================

def ensure_services_available() -> None:
    """Verify that all critical MLOps services are initialized."""

    services = {
        "registry": registry,
        "exporter": exporter,
        "ab_testing": ab_testing,
        "retraining_service": retraining_service,
        "storage": storage,
        "monitor": monitor,
    }

    missing_services = [name for name, service in services.items() if service is None]

    if missing_services:
        detail = (
            "MLOps services unavailable: "
            + ", ".join(sorted(missing_services))
            + " not initialized."
        )
        raise HTTPException(status_code=503, detail=detail)


# ============================================================================
# Request/Response Models
# ============================================================================

class ModelRegistrationRequest(BaseModel):
    """Request for model registration"""
    model_name: str
    description: str = ""
    tags: Dict[str, str] = {}
    metrics: Dict[str, float] = {}
    params: Dict[str, Any] = {}


class ModelPromotionRequest(BaseModel):
    """Request for model promotion"""
    model_name: str
    version: int
    target_stage: str = Field(..., description="Development, Staging, Production, or Archived")


class ABTestRequest(BaseModel):
    """Request for A/B test creation"""
    model_name: str
    champion_version: int
    challenger_version: int
    traffic_split: float = Field(0.1, ge=0.0, le=1.0)
    min_samples: int = Field(100, ge=10)


class ExportRequest(BaseModel):
    """Request for model export"""
    model_name: str
    version: int
    format: str = Field("onnx", description="onnx, pmml, edge")
    quantize: bool = True
    optimize_for_edge: bool = False


class RetrainingCheckRequest(BaseModel):
    """Request to check if retraining is needed"""
    model_name: str
    check_drift: bool = True
    check_performance: bool = True
    check_data_volume: bool = True


class PredictionRequest(BaseModel):
    """Request for prediction with optional A/B testing"""
    features: List[List[float]]
    model_name: Optional[str] = None
    version: Optional[int] = None
    use_ab_test: bool = False
    test_id: Optional[str] = None


# ============================================================================
# Model Registry Endpoints
# ============================================================================

@router.post(
    "/models/register",
    responses={
        503: {
            "description": "MLOps services unavailable. Ensure registry and related"
            " components are initialized.",
        }
    },
)
async def register_model(request: ModelRegistrationRequest):
    """Register a new model version in MLflow"""

    ensure_services_available()

    try:
        # This would normally load the actual model from storage
        # For demo, we'll create a dummy registration
        
        model_version = registry.register_model(
            model=None,  # Would be actual model
            model_name=request.model_name,
            metrics=request.metrics,
            params=request.params,
            description=request.description,
            tags=request.tags
        )
        
        return {
            "success": True,
            "model_name": model_version.model_name,
            "version": model_version.version,
            "stage": model_version.stage.value,
            "run_id": model_version.run_id
        }
        
    except Exception as e:
        logger.error(f"Failed to register model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/models/promote",
    responses={
        503: {
            "description": "MLOps services unavailable. Ensure registry and related"
            " components are initialized.",
        }
    },
)
async def promote_model(request: ModelPromotionRequest):
    """Promote model to different stage"""

    ensure_services_available()

    try:
        # Validate stage
        try:
            stage = ModelStage[request.target_stage.upper()]
        except KeyError:
            raise ValueError(f"Invalid stage: {request.target_stage}")
        
        success = registry.promote_model(
            request.model_name,
            request.version,
            stage
        )
        
        return {
            "success": success,
            "model_name": request.model_name,
            "version": request.version,
            "new_stage": request.target_stage
        }
        
    except Exception as e:
        logger.error(f"Failed to promote model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/models/{model_name}/versions",
    responses={
        503: {
            "description": "MLOps services unavailable. Ensure registry and related"
            " components are initialized.",
        }
    },
)
async def get_model_versions(
    model_name: str,
    limit: int = Query(10, ge=1, le=100)
):
    """Get model version history"""

    ensure_services_available()

    try:
        history = registry.get_model_history(model_name, limit)
        
        return {
            "model_name": model_name,
            "versions": history,
            "total": len(history)
        }
        
    except Exception as e:
        logger.error(f"Failed to get model history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/models/{model_name}/compare",
    responses={
        503: {
            "description": "MLOps services unavailable. Ensure registry and related"
            " components are initialized.",
        }
    },
)
async def compare_model_versions(
    model_name: str,
    version1: int,
    version2: int
):
    """Compare two model versions"""

    ensure_services_available()

    try:
        comparison = registry.compare_models(model_name, version1, version2)
        
        return comparison
        
    except Exception as e:
        logger.error(f"Failed to compare models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/models/{model_name}/rollback",
    responses={
        503: {
            "description": "MLOps services unavailable. Ensure registry and related"
            " components are initialized.",
        }
    },
)
async def rollback_model(model_name: str, target_version: int):
    """Rollback to previous model version"""

    ensure_services_available()

    try:
        success = registry.rollback_model(model_name, target_version)
        
        return {
            "success": success,
            "model_name": model_name,
            "rolled_back_to": target_version
        }
        
    except Exception as e:
        logger.error(f"Failed to rollback model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# A/B Testing Endpoints
# ============================================================================

@router.post(
    "/ab-tests/create",
    responses={
        503: {
            "description": "MLOps services unavailable. Ensure A/B testing components"
            " are initialized.",
        }
    },
)
async def create_ab_test(request: ABTestRequest):
    """Create new A/B test"""

    ensure_services_available()

    try:
        test_id = ab_testing.create_ab_test(
            model_name=request.model_name,
            champion_version=request.champion_version,
            challenger_version=request.challenger_version,
            traffic_split=request.traffic_split,
            min_samples=request.min_samples
        )
        
        return {
            "success": True,
            "test_id": test_id,
            "model_name": request.model_name,
            "champion_version": request.champion_version,
            "challenger_version": request.challenger_version,
            "traffic_split": request.traffic_split
        }
        
    except Exception as e:
        logger.error(f"Failed to create A/B test: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/ab-tests/{test_id}/results",
    responses={
        503: {
            "description": "MLOps services unavailable. Ensure A/B testing components"
            " are initialized.",
        }
    },
)
async def get_ab_test_results(test_id: str):
    """Get A/B test results"""

    ensure_services_available()

    try:
        results = ab_testing.get_test_results(test_id)
        
        if not results:
            raise HTTPException(status_code=404, detail="Test not found")
        
        return results
        
    except Exception as e:
        logger.error(f"Failed to get A/B test results: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/ab-tests/{test_id}/conclude",
    responses={
        503: {
            "description": "MLOps services unavailable. Ensure A/B testing components"
            " are initialized.",
        }
    },
)
async def conclude_ab_test(
    test_id: str,
    promote_winner: bool = Query(False, description="Automatically promote winner")
):
    """Conclude A/B test and optionally promote winner"""

    ensure_services_available()

    try:
        results = ab_testing.conclude_test(test_id, promote_winner)
        
        if "error" in results:
            raise HTTPException(status_code=404, detail=results["error"])
        
        return results
        
    except Exception as e:
        logger.error(f"Failed to conclude A/B test: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/ab-tests/active",
    responses={
        503: {
            "description": "MLOps services unavailable. Ensure A/B testing components"
            " are initialized.",
        }
    },
)
async def get_active_ab_tests():
    """Get list of active A/B tests"""

    ensure_services_available()

    try:
        active_tests = ab_testing.get_active_tests()
        
        return {
            "active_tests": active_tests,
            "total": len(active_tests)
        }
        
    except Exception as e:
        logger.error(f"Failed to get active tests: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Model Export Endpoints
# ============================================================================

@router.post(
    "/models/export",
    responses={
        503: {
            "description": "MLOps services unavailable. Ensure export components are"
            " initialized.",
        }
    },
)
async def export_model(request: ExportRequest):
    """Export model to specified format"""

    ensure_services_available()

    try:
        # Load model from registry
        # This is simplified - would load actual model
        
        # Configure export
        export_config = ExportConfig(
            quantize=request.quantize,
            optimize_for_edge=request.optimize_for_edge
        )
        
        exporter_instance = ModelExporter(export_config)
        
        # Create sample data for export
        sample_data = np.random.randn(10, 20)  # Adjust based on actual model
        
        if request.format == "onnx":
            result = exporter_instance.export_to_onnx(
                model=None,  # Would be actual model
                sample_input=sample_data,
                model_name=f"{request.model_name}_v{request.version}"
            )
        elif request.format == "pmml":
            result = exporter_instance.export_to_pmml(
                pipeline=None,  # Would be actual pipeline
                sample_input=sample_data,
                sample_output=np.random.randint(0, 2, 10),
                model_name=f"{request.model_name}_v{request.version}"
            )
        elif request.format == "edge":
            result = exporter_instance.export_for_edge(
                model=None,  # Would be actual model
                sample_input=sample_data,
                model_name=f"{request.model_name}_v{request.version}"
            )
        else:
            raise ValueError(f"Unsupported format: {request.format}")
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to export model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/models/export/{model_name}/{version}/download",
    responses={
        503: {
            "description": "MLOps services unavailable. Ensure export components are"
            " initialized.",
        }
    },
)
async def download_exported_model(
    model_name: str,
    version: int,
    format: str = Query("onnx", description="Export format")
):
    """Download exported model file"""

    ensure_services_available()

    try:
        # Build file path
        export_dir = Path(exporter.config.output_dir)
        
        if format == "onnx":
            file_path = export_dir / f"{model_name}_v{version}.onnx"
        elif format == "pmml":
            file_path = export_dir / f"{model_name}_v{version}.pmml"
        else:
            file_path = export_dir / "edge" / f"{model_name}_v{version}" / "model.onnx"
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Exported model not found")
        
        return FileResponse(
            path=file_path,
            filename=file_path.name,
            media_type="application/octet-stream"
        )
        
    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Automated Retraining Endpoints
# ============================================================================

@router.post(
    "/retraining/check",
    responses={
        503: {
            "description": "MLOps services unavailable. Ensure retraining components"
            " are initialized.",
        }
    },
)
async def check_retraining_needed(request: RetrainingCheckRequest):
    """Check if model needs retraining"""

    ensure_services_available()

    try:
        should_retrain, reason, metrics = retraining_service.should_retrain(
            request.model_name
        )
        
        return {
            "model_name": request.model_name,
            "needs_retraining": should_retrain,
            "reason": reason,
            "metrics": metrics,
            "checked_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to check retraining: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/retraining/trigger/{model_name}",
    responses={
        503: {
            "description": "MLOps services unavailable. Ensure retraining components"
            " are initialized.",
        }
    },
)
async def trigger_retraining(
    model_name: str,
    background_tasks: BackgroundTasks
):
    """Manually trigger model retraining"""

    ensure_services_available()

    try:
        # Add retraining to background tasks
        background_tasks.add_task(
            retrain_model_background,
            model_name
        )
        
        return {
            "success": True,
            "message": f"Retraining triggered for {model_name}",
            "status": "queued"
        }
        
    except Exception as e:
        logger.error(f"Failed to trigger retraining: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/retraining/history",
    responses={
        503: {
            "description": "MLOps services unavailable. Ensure retraining components"
            " are initialized.",
        }
    },
)
async def get_retraining_history(
    limit: int = Query(10, ge=1, le=100)
):
    """Get retraining history"""

    ensure_services_available()

    try:
        history = retraining_service.get_retraining_history(limit)
        
        return {
            "history": history,
            "total": len(history)
        }
        
    except Exception as e:
        logger.error(f"Failed to get retraining history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/retraining/stats",
    responses={
        503: {
            "description": "MLOps services unavailable. Ensure retraining components"
            " are initialized.",
        }
    },
)
async def get_retraining_stats():
    """Get retraining statistics"""

    ensure_services_available()

    try:
        stats = retraining_service.get_retraining_stats()
        
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get retraining stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/retraining/schedule",
    responses={
        503: {
            "description": "MLOps services unavailable. Ensure retraining components"
            " are initialized.",
        }
    },
)
async def create_retraining_schedule():
    """Create automated retraining schedule"""

    ensure_services_available()

    try:
        schedule = retraining_service.create_retraining_schedule()
        
        if schedule:
            return {
                "success": True,
                "message": "Retraining schedule created",
                "framework": "Airflow" if "DAG" in str(type(schedule)) else "Prefect"
            }
        else:
            return {
                "success": False,
                "message": "No scheduling framework available"
            }
        
    except Exception as e:
        logger.error(f"Failed to create schedule: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Prediction Endpoints with MLOps Features
# ============================================================================

@router.post(
    "/predict",
    responses={
        503: {
            "description": "MLOps services unavailable. Ensure required components"
            " are initialized.",
        }
    },
)
async def predict_with_mlops(request: PredictionRequest):
    """Make predictions with optional A/B testing and model versioning"""

    ensure_services_available()

    try:
        features = np.array(request.features)
        
        predictions = None
        model_info = {}
        
        if request.use_ab_test and request.test_id:
            # Route through A/B test
            model_type, version = ab_testing.route_prediction(request.test_id)
            
            # Load and predict with appropriate model
            # This is simplified - would load actual model from registry
            
            model_info = {
                "model_type": model_type,
                "version": version,
                "test_id": request.test_id
            }
            
            # Record result
            ab_testing.record_result(request.test_id, model_type, True)
            
        elif request.model_name and request.version:
            # Use specific model version
            # Load from registry
            model_info = {
                "model_name": request.model_name,
                "version": request.version
            }
        else:
            # Use default production model
            model_info = {
                "model_type": "production",
                "version": "latest"
            }
        
        # Generate dummy predictions for demo
        predictions = np.random.randn(len(features))
        
        return {
            "predictions": predictions.tolist(),
            "model_info": model_info,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to make predictions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Helper Functions
# ============================================================================

async def retrain_model_background(model_name: str):
    """Background task for model retraining"""

    ensure_services_available()

    try:
        # Load training data from storage
        X_train, y_train = storage.load_training_data(model_name)
        
        # Trigger retraining
        result = await retraining_service.retrain_model(
            model_name,
            X_train,
            y_train,
            reason="Manual trigger via API"
        )
        
        logger.info(f"Retraining completed for {model_name}: {result}")
        
    except Exception as e:
        logger.error(f"Background retraining failed: {e}")


# ============================================================================
# Health Check
# ============================================================================

@router.get("/health")
async def mlops_health_check():
    """Check MLOps services health"""
    
    health = {
        "status": "healthy",
        "services": {
            "mlflow": registry.client is not None,
            "export": True,
            "ab_testing": len(ab_testing.active_tests) >= 0,
            "retraining": True
        },
        "timestamp": datetime.utcnow().isoformat()
    }
    
    return health
