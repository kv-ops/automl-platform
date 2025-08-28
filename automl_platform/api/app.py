"""FastAPI application for AutoML platform."""

from fastapi import FastAPI, HTTPException, File, UploadFile, BackgroundTasks, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import uuid
import io
import json
from pathlib import Path
import asyncio
import aiofiles

from ..orchestrator import AutoMLOrchestrator
from ..config import AutoMLConfig
from ..inference import load_pipeline, predict, predict_proba, save_predictions
from ..data_prep import validate_data
from ..metrics import calculate_metrics

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="AutoML Platform API",
    description="Production-ready AutoML with no data leakage",
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state management
class AppState:
    def __init__(self):
        self.active_jobs = {}
        self.models_cache = {}
        self.max_models = 10  # Maximum models to keep in cache
        
state = AppState()

# Pydantic models for request/response
class TrainRequest(BaseModel):
    target: str = Field(..., description="Target column name")
    task: str = Field("auto", description="Task type: auto, classification, regression")
    cv_folds: int = Field(5, ge=2, le=20, description="Number of CV folds")
    algorithms: List[str] = Field(["all"], description="List of algorithms to test")
    hpo_method: str = Field("optuna", description="HPO method: optuna, random, grid, none")
    hpo_n_iter: int = Field(20, ge=1, le=100, description="Number of HPO iterations")
    config: Optional[Dict[str, Any]] = Field(None, description="Additional configuration")

class PredictionRequest(BaseModel):
    model_id: str = Field(..., description="Model ID")
    features: Dict[str, Any] = Field(..., description="Feature values")
    explain: bool = Field(False, description="Include explanation")

class BatchPredictionRequest(BaseModel):
    model_id: str = Field(..., description="Model ID")
    data: List[Dict[str, Any]] = Field(..., description="List of feature dictionaries")
    include_probabilities: bool = Field(False, description="Include probability predictions")

class JobStatus(BaseModel):
    job_id: str
    status: str  # pending, running, completed, failed
    progress: float
    message: str
    created_at: datetime
    updated_at: datetime
    result: Optional[Dict[str, Any]] = None

class ModelInfo(BaseModel):
    model_id: str
    created_at: str
    task: str
    best_model: str
    cv_score: float
    n_features: int
    n_samples_trained: int

# Background training task
async def train_background(job_id: str, df: pd.DataFrame, train_request: TrainRequest):
    """Async background training task."""
    try:
        # Update job status
        state.active_jobs[job_id]["status"] = "running"
        state.active_jobs[job_id]["message"] = "Preparing data..."
        state.active_jobs[job_id]["progress"] = 0.1
        
        # Split features and target
        if train_request.target not in df.columns:
            raise ValueError(f"Target column '{train_request.target}' not found")
        
        X = df.drop(columns=[train_request.target])
        y = df[train_request.target]
        
        # Create configuration
        config_dict = train_request.config or {}
        config_dict.update({
            "cv_folds": train_request.cv_folds,
            "algorithms": train_request.algorithms,
            "hpo_method": train_request.hpo_method,
            "hpo_n_iter": train_request.hpo_n_iter
        })
        config = AutoMLConfig(**config_dict)
        
        # Update progress
        state.active_jobs[job_id]["message"] = "Training models..."
        state.active_jobs[job_id]["progress"] = 0.2
        
        # Train
        orchestrator = AutoMLOrchestrator(config)
        await asyncio.get_event_loop().run_in_executor(
            None, orchestrator.fit, X, y, train_request.task
        )
        
        # Update progress
        state.active_jobs[job_id]["message"] = "Saving model..."
        state.active_jobs[job_id]["progress"] = 0.9
        
        # Save model
        model_id = str(uuid.uuid4())
        model_dir = Path("./api_models")
        model_dir.mkdir(exist_ok=True)
        model_path = model_dir / f"{model_id}.joblib"
        
        orchestrator.save_pipeline(str(model_path))
        
        # Cache model (limit cache size)
        if len(state.models_cache) >= state.max_models:
            # Remove oldest model from cache
            oldest_id = min(state.models_cache, 
                          key=lambda k: state.models_cache[k]["created_at"])
            del state.models_cache[oldest_id]
        
        state.models_cache[model_id] = {
            "orchestrator": orchestrator,
            "path": str(model_path),
            "created_at": datetime.now().isoformat(),
            "task": orchestrator.task,
            "best_model": orchestrator.leaderboard[0]["model"] if orchestrator.leaderboard else None,
            "cv_score": orchestrator.leaderboard[0]["cv_score"] if orchestrator.leaderboard else None,
            "n_features": X.shape[1],
            "n_samples": len(X)
        }
        
        # Prepare result
        leaderboard_dict = orchestrator.get_leaderboard(top_n=10).to_dict('records')
        
        # Update job as completed
        state.active_jobs[job_id]["status"] = "completed"
        state.active_jobs[job_id]["message"] = "Training completed successfully"
        state.active_jobs[job_id]["progress"] = 1.0
        state.active_jobs[job_id]["result"] = {
            "model_id": model_id,
            "task": orchestrator.task,
            "best_model": state.models_cache[model_id]["best_model"],
            "cv_score": state.models_cache[model_id]["cv_score"],
            "leaderboard": leaderboard_dict,
            "n_models_tested": len(orchestrator.leaderboard)
        }
        
    except Exception as e:
        logger.error(f"Training failed for job {job_id}: {str(e)}")
        state.active_jobs[job_id]["status"] = "failed"
        state.active_jobs[job_id]["message"] = str(e)
        state.active_jobs[job_id]["progress"] = 0.0
    
    finally:
        state.active_jobs[job_id]["updated_at"] = datetime.now()

# API Endpoints
@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with API documentation."""
    html_content = """
    <html>
        <head>
            <title>AutoML Platform API</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                h1 { color: #333; }
                .endpoint { background: #f4f4f4; padding: 10px; margin: 10px 0; border-radius: 5px; }
                .method { font-weight: bold; color: #007bff; }
                code { background: #e9ecef; padding: 2px 5px; border-radius: 3px; }
            </style>
        </head>
        <body>
            <h1>🚀 AutoML Platform API v3.0</h1>
            <p>Production-ready AutoML with no data leakage</p>
            
            <h2>Available Endpoints:</h2>
            
            <div class="endpoint">
                <span class="method">POST</span> <code>/train</code> - Train new AutoML model
            </div>
            
            <div class="endpoint">
                <span class="method">POST</span> <code>/predict</code> - Single prediction
            </div>
            
            <div class="endpoint">
                <span class="method">POST</span> <code>/predict_batch</code> - Batch predictions
            </div>
            
            <div class="endpoint">
                <span class="method">GET</span> <code>/models</code> - List trained models
            </div>
            
            <div class="endpoint">
                <span class="method">GET</span> <code>/model/{model_id}</code> - Get model details
            </div>
            
            <div class="endpoint">
                <span class="method">GET</span> <code>/job/{job_id}</code> - Check job status
            </div>
            
            <div class="endpoint">
                <span class="method">GET</span> <code>/health</code> - Health check
            </div>
            
            <div class="endpoint">
                <span class="method">GET</span> <code>/docs</code> - Interactive API documentation
            </div>
            
            <h2>Quick Start:</h2>
            <ol>
                <li>Upload your data using POST /train</li>
                <li>Check training status with GET /job/{job_id}</li>
                <li>Make predictions with POST /predict</li>
            </ol>
        </body>
    </html>
    """
    return html_content

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_cached": len(state.models_cache),
        "active_jobs": len(state.active_jobs),
        "version": "3.0.0"
    }

@app.post("/train", status_code=status.HTTP_202_ACCEPTED)
async def train_model(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="CSV file with training data"),
    config: str = None
):
    """Train a new AutoML model (async)."""
    try:
        # Parse configuration
        if config:
            train_request = TrainRequest(**json.loads(config))
        else:
            return HTTPException(
                status_code=400,
                detail="Configuration required. Pass as JSON string in 'config' parameter"
            )
        
        # Load data
        content = await file.read()
        df = pd.read_csv(io.StringIO(content.decode('utf-8')))
        
        logger.info(f"Loaded data: {df.shape}")
        
        # Validate data
        validation = validate_data(df)
        if not validation["valid"]:
            logger.warning(f"Data issues: {validation['issues']}")
        
        # Create job
        job_id = str(uuid.uuid4())
        state.active_jobs[job_id] = {
            "job_id": job_id,
            "status": "pending",
            "progress": 0.0,
            "message": "Job created",
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
            "result": None
        }
        
        # Start background training
        background_tasks.add_task(train_background, job_id, df, train_request)
        
        return {
            "job_id": job_id,
            "message": "Training started",
            "status_url": f"/job/{job_id}"
        }
        
    except Exception as e:
        logger.error(f"Training error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
async def predict_single(request: PredictionRequest):
    """Make a single prediction."""
    try:
        # Get model
        if request.model_id not in state.models_cache:
            # Try to load from disk
            model_path = Path(f"./api_models/{request.model_id}.joblib")
            if not model_path.exists():
                raise HTTPException(status_code=404, detail=f"Model {request.model_id} not found")
            
            # Load and cache
            from ..inference import load_pipeline
            pipeline, metadata = load_pipeline(model_path)
            orchestrator = AutoMLOrchestrator(AutoMLConfig())
            orchestrator.best_pipeline = pipeline
            orchestrator.task = metadata.get("task", "classification")
            
            state.models_cache[request.model_id] = {
                "orchestrator": orchestrator,
                "path": str(model_path),
                "created_at": datetime.now().isoformat(),
                "task": orchestrator.task
            }
        
        orchestrator = state.models_cache[request.model_id]["orchestrator"]
        
        # Create dataframe
        df = pd.DataFrame([request.features])
        
        # Predict
        prediction = orchestrator.predict(df)[0]
        
        response = {
            "model_id": request.model_id,
            "prediction": prediction.item() if hasattr(prediction, 'item') else prediction,
            "timestamp": datetime.now().isoformat()
        }
        
        # Add probabilities if classification
        if orchestrator.task == "classification" and hasattr(orchestrator.best_pipeline, 'predict_proba'):
            try:
                proba = orchestrator.predict_proba(df)[0]
                response["probabilities"] = proba.tolist()
                response["confidence"] = float(max(proba))
            except:
                pass
        
        # Add explanation if requested
        if request.explain:
            try:
                explanation = orchestrator.explain_predictions(df, indices=[0])
                response["explanation"] = explanation
            except Exception as e:
                logger.warning(f"Could not generate explanation: {e}")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_batch")
async def predict_batch(request: BatchPredictionRequest):
    """Make batch predictions."""
    try:
        # Get model
        if request.model_id not in state.models_cache:
            raise HTTPException(status_code=404, detail=f"Model {request.model_id} not found")
        
        orchestrator = state.models_cache[request.model_id]["orchestrator"]
        
        # Create dataframe
        df = pd.DataFrame(request.data)
        
        # Predict
        predictions = orchestrator.predict(df)
        
        results = []
        for i, pred in enumerate(predictions):
            result = {
                "index": i,
                "prediction": pred.item() if hasattr(pred, 'item') else pred
            }
            
            # Add probabilities if requested
            if request.include_probabilities and orchestrator.task == "classification":
                try:
                    proba = orchestrator.predict_proba(df.iloc[[i]])[0]
                    result["probabilities"] = proba.tolist()
                    result["confidence"] = float(max(proba))
                except:
                    pass
            
            results.append(result)
        
        return {
            "model_id": request.model_id,
            "predictions": results,
            "n_predictions": len(results),
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models", response_model=List[ModelInfo])
async def list_models():
    """List all available models."""
    models = []
    for model_id, info in state.models_cache.items():
        models.append(ModelInfo(
            model_id=model_id,
            created_at=info["created_at"],
            task=info.get("task", "unknown"),
            best_model=info.get("best_model", "unknown"),
            cv_score=info.get("cv_score", 0.0),
            n_features=info.get("n_features", 0),
            n_samples_trained=info.get("n_samples", 0)
        ))
    return models

@app.get("/model/{model_id}")
async def get_model_details(model_id: str):
    """Get detailed model information."""
    if model_id not in state.models_cache:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
    
    info = state.models_cache[model_id]
    orchestrator = info["orchestrator"]
    
    return {
        "model_id": model_id,
        "created_at": info["created_at"],
        "task": info.get("task", "unknown"),
        "best_model": info.get("best_model", "unknown"),
        "cv_score": info.get("cv_score", 0.0),
        "n_features": info.get("n_features", 0),
        "n_samples_trained": info.get("n_samples", 0),
        "leaderboard": orchestrator.get_leaderboard(top_n=5).to_dict('records') if orchestrator.leaderboard else [],
        "feature_importance": orchestrator.feature_importance if hasattr(orchestrator, 'feature_importance') else {}
    }

@app.get("/job/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    """Get job status."""
    if job_id not in state.active_jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    return JobStatus(**state.active_jobs[job_id])

@app.delete("/model/{model_id}")
async def delete_model(model_id: str):
    """Delete a model from cache and disk."""
    if model_id not in state.models_cache:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
    
    # Remove from cache
    info = state.models_cache[model_id]
    del state.models_cache[model_id]
    
    # Delete file if exists
    model_path = Path(info["path"])
    if model_path.exists():
        model_path.unlink()
        # Also delete metadata
        meta_path = model_path.with_suffix('.meta.json')
        if meta_path.exists():
            meta_path.unlink()
    
    return {"message": f"Model {model_id} deleted successfully"}

@app.get("/jobs")
async def list_jobs():
    """List all jobs."""
    jobs = []
    for job_id, job_info in state.active_jobs.items():
        jobs.append({
            "job_id": job_id,
            "status": job_info["status"],
            "progress": job_info["progress"],
            "created_at": job_info["created_at"].isoformat(),
            "message": job_info["message"]
        })
    return {"jobs": jobs, "total": len(jobs)}

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize application on startup."""
    logger.info("AutoML Platform API starting up...")
    
    # Create directories
    Path("./api_models").mkdir(exist_ok=True)
    Path("./api_temp").mkdir(exist_ok=True)
    
    logger.info("API ready to serve requests")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("AutoML Platform API shutting down...")
    
    # Wait for active jobs to complete (max 30 seconds)
    active_jobs = [j for j in state.active_jobs.values() if j["status"] == "running"]
    if active_jobs:
        logger.info(f"Waiting for {len(active_jobs)} active jobs to complete...")
        await asyncio.sleep(min(30, len(active_jobs) * 5))
    
    logger.info("Shutdown complete")

# Main entry point for running directly
def main():
    """Run the API server."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    main()
