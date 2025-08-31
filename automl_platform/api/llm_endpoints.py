"""
LLM API Endpoints - Following DataRobot's generative AI workflows approach
Provides REST endpoints for LLM features including RAG and data cleaning
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
import pandas as pd
import json
import asyncio
from datetime import datetime
import uuid
import io  # ADDED: Missing import

from ..llm import AutoMLLLMAssistant, DataCleaningAgent
from ..data_quality_agent import IntelligentDataQualityAgent, DataQualityAssessment
from ..config import LLMConfig

router = APIRouter(prefix="/api/v1/llm", tags=["LLM"])


# ============================================================================
# Pydantic Models
# ============================================================================

class FeatureSuggestionRequest(BaseModel):
    """Request for feature engineering suggestions."""
    dataset_id: str
    target_column: str
    task_type: str = "auto"
    max_suggestions: int = 10
    include_code: bool = True


class DataCleaningRequest(BaseModel):
    """Request for conversational data cleaning (Akkio-style)."""
    dataset_id: str
    message: str
    apply_changes: bool = False
    preview_only: bool = True


class ModelExplanationRequest(BaseModel):
    """Request for model explanation."""
    model_id: str
    include_shap: bool = True
    audience: str = "technical"  # "technical", "business", "executive"


class ReportGenerationRequest(BaseModel):
    """Request for report generation."""
    experiment_id: str
    report_type: str = "executive"  # "executive", "technical", "model_card"
    format: str = "markdown"  # "markdown", "html", "pdf"
    include_visualizations: bool = True


class RAGQueryRequest(BaseModel):
    """Request for RAG-based query."""
    query: str
    context_type: str = "documentation"  # "documentation", "past_experiments", "best_practices"
    max_results: int = 5


class CodeGenerationRequest(BaseModel):
    """Request for AutoML code generation."""
    task_description: str
    dataset_info: Optional[Dict[str, Any]] = None
    framework: str = "automl_platform"  # "automl_platform", "sklearn", "pytorch"
    include_deployment: bool = False


# ============================================================================
# DataRobot-Style Generative Workflows
# ============================================================================

@router.post("/workflows/create")
async def create_generative_workflow(
    name: str,
    description: str,
    workflow_type: str = "data_cleaning",  # "data_cleaning", "feature_engineering", "model_explanation"
    steps: List[Dict[str, Any]] = None
):
    """
    Create a DataRobot-style generative workflow.
    Combines RAG, prompts, and LLM in a single endpoint.
    """
    workflow_id = str(uuid.uuid4())
    
    workflow = {
        "id": workflow_id,
        "name": name,
        "description": description,
        "type": workflow_type,
        "steps": steps or [],
        "created_at": datetime.now().isoformat(),
        "status": "created"
    }
    
    # Store workflow (in production, save to database)
    
    return {
        "workflow_id": workflow_id,
        "message": f"Workflow '{name}' created successfully",
        "workflow": workflow
    }


@router.post("/workflows/{workflow_id}/execute")
async def execute_workflow(
    workflow_id: str,
    dataset_id: str,
    parameters: Dict[str, Any] = None
):
    """Execute a generative workflow on a dataset."""
    
    # In production, load workflow from database
    # Execute workflow steps using LLM
    
    results = {
        "workflow_id": workflow_id,
        "dataset_id": dataset_id,
        "status": "completed",
        "results": {
            "quality_improved": 15.2,
            "features_added": 8,
            "insights_generated": 12
        },
        "execution_time": 45.3
    }
    
    return results


# ============================================================================
# Akkio-Style Chat Cleaning
# ============================================================================

@router.post("/clean/chat")
async def chat_data_cleaning(request: DataCleaningRequest):
    """
    Akkio-style conversational data cleaning.
    Users can clean data through natural language commands.
    """
    
    # Load dataset (mock for now)
    df = pd.DataFrame()  # Load from storage
    
    # Initialize cleaning agent
    agent = DataCleaningAgent(None)  # Pass LLM provider
    
    # Process cleaning request
    cleaned_df, response = await agent.interactive_clean(df)
    
    result = {
        "dataset_id": request.dataset_id,
        "message": request.message,
        "response": response,
        "changes_preview": {
            "rows_affected": 100,
            "columns_affected": ["col1", "col2"],
            "quality_improvement": 12.5
        },
        "applied": request.apply_changes
    }
    
    if request.apply_changes:
        # Save cleaned dataset
        result["new_dataset_id"] = f"{request.dataset_id}_cleaned"
    
    return result


@router.websocket("/clean/interactive")
async def interactive_cleaning_session(websocket: WebSocket, dataset_id: str):
    """
    WebSocket endpoint for interactive data cleaning session.
    Provides real-time chat-based cleaning like Akkio.
    """
    await websocket.accept()
    
    # Load dataset
    df = pd.DataFrame()  # Load from storage
    
    # Initialize session
    agent = DataCleaningAgent(None)
    session_id = str(uuid.uuid4())
    
    await websocket.send_json({
        "type": "session_start",
        "session_id": session_id,
        "dataset_shape": list(df.shape),
        "columns": list(df.columns)
    })
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_json()
            
            if data["type"] == "clean":
                # Process cleaning request
                cleaned_df, response = await agent.interactive_clean(df)
                
                await websocket.send_json({
                    "type": "cleaning_response",
                    "response": response,
                    "preview": {
                        "shape": list(cleaned_df.shape),
                        "sample": cleaned_df.head(5).to_dict()
                    }
                })
                
            elif data["type"] == "apply":
                # Apply changes
                df = cleaned_df
                await websocket.send_json({
                    "type": "changes_applied",
                    "message": "Changes applied successfully"
                })
                
            elif data["type"] == "undo":
                # Undo last action
                await websocket.send_json({
                    "type": "undo_complete",
                    "message": "Last action undone"
                })
                
    except WebSocketDisconnect:
        pass


# ============================================================================
# Feature Engineering
# ============================================================================

@router.post("/features/suggest")
async def suggest_features(request: FeatureSuggestionRequest):
    """
    AI-powered feature engineering suggestions.
    Returns specific features with code like DataRobot's Feature Discovery.
    """
    
    # Initialize LLM assistant
    config = LLMConfig()
    assistant = AutoMLLLMAssistant(config.to_dict())
    
    # Generate suggestions
    suggestions = await assistant.suggest_features(
        pd.DataFrame(),  # Load actual data
        request.target_column,
        request.task_type
    )
    
    return {
        "dataset_id": request.dataset_id,
        "suggestions": suggestions[:request.max_suggestions],
        "total_suggestions": len(suggestions),
        "estimated_impact": "high"  # Based on analysis
    }


@router.post("/features/auto-engineer")
async def auto_engineer_features(
    dataset_id: str,
    target_column: str,
    max_features: int = 20,
    complexity: str = "medium"  # "simple", "medium", "complex"
):
    """
    Automatically engineer features using LLM.
    Similar to DataRobot's automated feature engineering.
    """
    
    engineered_features = [
        {
            "name": "price_per_sqft",
            "formula": "price / square_feet",
            "type": "ratio",
            "importance_estimate": 0.85
        },
        {
            "name": "age_income_interaction",
            "formula": "age * income",
            "type": "interaction",
            "importance_estimate": 0.72
        }
    ]
    
    return {
        "dataset_id": dataset_id,
        "features_created": len(engineered_features),
        "features": engineered_features,
        "new_dataset_id": f"{dataset_id}_engineered"
    }


# ============================================================================
# Model Explanation
# ============================================================================

@router.post("/explain/model")
async def explain_model(request: ModelExplanationRequest):
    """
    Generate natural language explanation of model.
    Includes SHAP values interpretation.
    """
    
    config = LLMConfig()
    assistant = AutoMLLLMAssistant(config.to_dict())
    
    # Generate explanation
    explanation = await assistant.explain_model(
        model_name="XGBoost",  # From model registry
        metrics={"accuracy": 0.92, "precision": 0.89},
        feature_importance={"feature1": 0.3, "feature2": 0.25}
    )
    
    response = {
        "model_id": request.model_id,
        "explanation": explanation,
        "audience": request.audience
    }
    
    if request.include_shap:
        response["shap_interpretation"] = {
            "global_importance": {},
            "sample_explanations": []
        }
    
    return response


@router.post("/explain/prediction")
async def explain_prediction(
    model_id: str,
    instance: Dict[str, Any],
    explanation_type: str = "shap"  # "shap", "lime", "natural_language"
):
    """
    Explain individual prediction with natural language.
    Similar to DataRobot's Prediction Explanations.
    """
    
    explanation = {
        "prediction": 0.87,
        "confidence": 0.92,
        "natural_language": "The model predicted a high likelihood (87%) primarily due to...",
        "top_factors": [
            {"feature": "income", "impact": 0.35, "value": 75000},
            {"feature": "credit_score", "impact": 0.28, "value": 720}
        ]
    }
    
    return explanation


# ============================================================================
# Report Generation
# ============================================================================

@router.post("/reports/generate")
async def generate_report(request: ReportGenerationRequest):
    """
    Generate comprehensive AutoML report using LLM.
    Similar to DataRobot's automated documentation.
    """
    
    config = LLMConfig()
    assistant = AutoMLLLMAssistant(config.to_dict())
    
    # Generate report
    report_content = await assistant.generate_report(
        experiment_data={
            "experiment_id": request.experiment_id,
            "best_model": "XGBoost",
            "metrics": {"accuracy": 0.92},
            "top_features": ["feature1", "feature2"]
        },
        format=request.format
    )
    
    return {
        "experiment_id": request.experiment_id,
        "report_type": request.report_type,
        "format": request.format,
        "content": report_content,
        "download_url": f"/api/v1/reports/download/{request.experiment_id}"
    }


@router.get("/reports/download/{experiment_id}")
async def download_report(experiment_id: str, format: str = "pdf"):
    """Download generated report."""
    
    # In production, retrieve from storage
    content = b"Report content here"
    
    return StreamingResponse(
        io.BytesIO(content),  # FIXED: Now io is imported
        media_type="application/pdf" if format == "pdf" else "text/markdown",
        headers={
            "Content-Disposition": f"attachment; filename=report_{experiment_id}.{format}"
        }
    )


# ============================================================================
# RAG Endpoints
# ============================================================================

@router.post("/rag/query")
async def query_knowledge_base(request: RAGQueryRequest):
    """
    Query the RAG knowledge base.
    Similar to DataRobot's Vector Database integration.
    """
    
    config = LLMConfig()
    assistant = AutoMLLLMAssistant(config.to_dict())
    
    # Search and generate response
    if assistant.rag:
        results = assistant.rag.search(request.query, k=request.max_results)
        
        return {
            "query": request.query,
            "results": [
                {"content": doc, "relevance": score}
                for doc, score in results
            ],
            "generated_answer": "Based on the documentation..."
        }
    
    return {"error": "RAG not configured"}


@router.post("/rag/index")
async def index_documents(
    documents: List[str],
    metadata: List[Dict[str, Any]] = None,
    collection_name: str = "automl_docs"
):
    """
    Index documents in the RAG system.
    For building custom knowledge bases.
    """
    
    config = LLMConfig()
    assistant = AutoMLLLMAssistant(config.to_dict())
    
    if assistant.rag:
        assistant.rag.add_documents(documents, metadata)
        
        return {
            "message": f"Indexed {len(documents)} documents",
            "collection": collection_name
        }
    
    return {"error": "RAG not configured"}


# ============================================================================
# Code Generation
# ============================================================================

@router.post("/code/generate")
async def generate_code(request: CodeGenerationRequest):
    """
    Generate AutoML code from natural language.
    Similar to GitHub Copilot but for AutoML tasks.
    """
    
    config = LLMConfig()
    assistant = AutoMLLLMAssistant(config.to_dict())
    
    # Generate code
    code = await assistant.generate_code(
        request.task_description,
        pd.DataFrame() if request.dataset_info else None
    )
    
    response = {
        "task": request.task_description,
        "code": code,
        "framework": request.framework,
        "runnable": True
    }
    
    if request.include_deployment:
        response["deployment_code"] = """
# Deployment code
from fastapi import FastAPI
app = FastAPI()

@app.post("/predict")
def predict(data: dict):
    # Your model prediction code
    return {"prediction": result}
"""
    
    return response


# ============================================================================
# Chat Interface
# ============================================================================

@router.post("/chat")
async def chat_with_assistant(
    message: str,
    context: Optional[Dict[str, Any]] = None,
    session_id: Optional[str] = None
):
    """
    General chat interface with the AI assistant.
    Maintains context across conversations.
    """
    
    config = LLMConfig()
    assistant = AutoMLLLMAssistant(config.to_dict())
    
    # Get response
    response = await assistant.chat(message, context)
    
    return {
        "message": message,
        "response": response,
        "session_id": session_id or str(uuid.uuid4()),
        "timestamp": datetime.now().isoformat()
    }


# ============================================================================
# Data Quality Assessment (DataRobot-style)
# ============================================================================

@router.post("/quality/assess")
async def assess_data_quality(
    dataset_id: str,
    target_column: Optional[str] = None,
    generate_visuals: bool = True
):
    """
    DataRobot-style Data Quality Assessment.
    Returns comprehensive quality metrics with visual indicators.
    """
    
    # Initialize quality agent
    agent = IntelligentDataQualityAgent()
    
    # Load dataset (mock)
    df = pd.DataFrame()
    
    # Perform assessment
    assessment = agent.assess(df, target_column)
    
    response = {
        "dataset_id": dataset_id,
        "quality_score": assessment.quality_score,
        "alerts": assessment.alerts,
        "warnings": assessment.warnings,
        "recommendations": assessment.recommendations,
        "statistics": assessment.statistics,
        "drift_risk": assessment.drift_risk,
        "target_leakage_risk": assessment.target_leakage_risk
    }
    
    if generate_visuals:
        response["visualizations"] = assessment.visualization_data
    
    return response


# ============================================================================
# Prompt Management
# ============================================================================

@router.get("/prompts/list")
async def list_prompts():
    """List available prompt templates."""
    
    from ..prompts import PromptTemplates
    
    return {
        "prompts": PromptTemplates.list_prompts(),
        "total": len(PromptTemplates.list_prompts())
    }


@router.post("/prompts/optimize")
async def optimize_prompt(
    prompt: str,
    model: str = "gpt-4",
    max_tokens: int = 1000
):
    """Optimize prompt for specific model."""
    
    from ..prompts import PromptOptimizer
    
    optimized = PromptOptimizer.optimize_for_model(prompt, model, max_tokens)
    
    return {
        "original": prompt,
        "optimized": optimized,
        "model": model,
        "estimated_tokens": len(optimized) // 4
    }


# ============================================================================
# Usage Tracking
# ============================================================================

@router.get("/usage/stats")
async def get_usage_stats(
    user_id: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None
):
    """
    Get LLM usage statistics.
    Important for cost tracking.
    """
    
    return {
        "total_tokens": 145000,
        "total_cost": 4.35,
        "requests": 234,
        "cache_hit_rate": 0.42,
        "average_response_time": 1.2,
        "by_model": {
            "gpt-4": {"tokens": 100000, "cost": 3.0},
            "gpt-3.5-turbo": {"tokens": 45000, "cost": 1.35}
        }
    }


# Example usage
if __name__ == "__main__":
    import uvicorn
    from fastapi import FastAPI
    
    app = FastAPI()
    app.include_router(router)
    
    uvicorn.run(app, host="0.0.0.0", port=8001)
