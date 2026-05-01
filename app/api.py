"""FastAPI application for Risk Assessment predictions."""

import sys
import logging
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config_loader import load_config
from src.decisioning.engine import DecisionEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Risk Assessment API",
    description="Loan default risk prediction and decision system",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state for models
class ModelState:
    """Store loaded models and engine."""
    model = None
    preprocessor = None
    decision_engine = None
    config = None

state = ModelState()


@app.on_event("startup")
async def startup_event():
    """Load models on startup."""
    try:
        logger.info("Loading models and configuration...")
        
        # Get project root directory
        project_root = Path(__file__).parent.parent
        
        # Load config
        config_path = project_root / "configs" / "config.yaml"
        state.config = load_config(str(config_path))
        logger.info(f"Config loaded from {config_path}")
        
        # Load preprocessor
        preprocessor_path = project_root / "models" / "preprocessor.pkl"
        logger.info(f"Looking for preprocessor at: {preprocessor_path}")
        if preprocessor_path.exists():
            state.preprocessor = joblib.load(preprocessor_path)
            logger.info(f"✓ Preprocessor loaded from {preprocessor_path}")
        else:
            logger.warning(f"✗ Preprocessor not found at {preprocessor_path}")
        
        # Load model
        model_path = project_root / "models" / "baseline_model.pkl"
        logger.info(f"Looking for model at: {model_path}")
        if model_path.exists():
            state.model = joblib.load(model_path)
            logger.info(f"✓ Model loaded from {model_path}")
        else:
            logger.warning(f"✗ Model not found at {model_path}")
            logger.warning("Available .pkl files in models/:")
            models_dir = project_root / "models"
            if models_dir.exists():
                pkl_files = list(models_dir.glob("*.pkl"))
                if pkl_files:
                    for f in pkl_files:
                        logger.warning(f"  - {f.name}")
                else:
                    logger.warning("  (no .pkl files found)")
        
        # Initialize decision engine
        reports_dir = project_root / "reports"
        state.decision_engine = DecisionEngine(output_dir=str(reports_dir))
        logger.info("Decision engine initialized")
        
        logger.info("Startup complete")
    except Exception as e:
        logger.error(f"Error during startup: {e}", exc_info=True)
        raise


class PredictionRequest(BaseModel):
    """Schema for prediction request."""
    features: Dict[str, Any] = Field(..., description="Feature dictionary for a single applicant")
    
    class Config:
        example = {
            "features": {
                "Age": 20,
                "Income": 63000,
                "CreditScore": 420,
                "LoanAmount": 12800,
                "YearsExperience": 3,
                "Gender": "Male",
                "Education": "Bachelor",
                "City": "New York",
                "EmploymentType": "Unemployed",
            }
        }


class PredictionResponse(BaseModel):
    """Schema for prediction response."""
    risk_score: float = Field(..., description="Risk score (0-1, probability of default)")
    decision: str = Field(..., description="Decision: APPROVE, REVIEW, or REJECT")
    explanation: str = Field(..., description="Explanation of the decision")


class BatchPredictionRequest(BaseModel):
    """Schema for batch predictions."""
    applicants: List[Dict[str, Any]] = Field(..., description="List of feature dictionaries")


class BatchPredictionResponse(BaseModel):
    """Schema for batch prediction response."""
    predictions: List[PredictionResponse] = Field(..., description="List of predictions")
    statistics: Dict[str, Any] = Field(..., description="Batch statistics")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": state.model is not None,
        "message": "Risk Assessment API is running"
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predict risk score and decision for a single applicant.
    
    Returns:
        - risk_score: Probability of default (0-1)
        - decision: APPROVE, REVIEW, or REJECT
        - explanation: Reasoning for the decision
    """
    if state.model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please run 'python train.py' first."
        )
    
    try:
        # Convert features to DataFrame
        df = pd.DataFrame([request.features])
        
        # Preprocess
        if state.preprocessor:
            df_processed = state.preprocessor.transform(df)
        else:
            df_processed = df
        
        # Predict
        risk_score = float(state.model.predict_proba(df_processed)[:, 1][0])
        
        # Apply decision policy
        approve_threshold = state.config.get("approve_threshold", 0.30)
        reject_threshold = state.config.get("reject_threshold", 0.70)
        
        if risk_score < approve_threshold:
            decision = "APPROVE"
            explanation = f"Risk score {risk_score:.2%} is below approval threshold ({approve_threshold:.0%})"
        elif risk_score > reject_threshold:
            decision = "REJECT"
            explanation = f"Risk score {risk_score:.2%} exceeds rejection threshold ({reject_threshold:.0%})"
        else:
            decision = "REVIEW"
            explanation = f"Risk score {risk_score:.2%} requires manual review (between {approve_threshold:.0%}-{reject_threshold:.0%})"
        
        return PredictionResponse(
            risk_score=risk_score,
            decision=decision,
            explanation=explanation
        )
    
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")


@app.post("/predict_batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """
    Predict risk scores and decisions for multiple applicants.
    
    Returns batch predictions and summary statistics.
    """
    if state.model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please run 'python train.py' first."
        )
    
    try:
        predictions = []
        
        for features in request.applicants:
            pred_request = PredictionRequest(features=features)
            pred_response = await predict(pred_request)
            predictions.append(pred_response)
        
        # Calculate statistics
        risk_scores = [p.risk_score for p in predictions]
        decision_counts = {}
        for p in predictions:
            decision_counts[p.decision] = decision_counts.get(p.decision, 0) + 1
        
        statistics = {
            "total_applicants": len(predictions),
            "mean_risk_score": float(np.mean(risk_scores)),
            "median_risk_score": float(np.median(risk_scores)),
            "min_risk_score": float(np.min(risk_scores)),
            "max_risk_score": float(np.max(risk_scores)),
            "decision_distribution": decision_counts,
            "approval_rate": decision_counts.get("APPROVE", 0) / len(predictions)
        }
        
        return BatchPredictionResponse(
            predictions=predictions,
            statistics=statistics
        )
    
    except Exception as e:
        logger.error(f"Batch prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Batch prediction failed: {str(e)}")


@app.get("/config")
async def get_config():
    """Get current system configuration."""
    if state.config is None:
        raise HTTPException(status_code=503, detail="Configuration not loaded")
    
    return {
        "approve_threshold": state.config.get("approve_threshold", 0.30),
        "reject_threshold": state.config.get("reject_threshold", 0.70),
        "model_type": state.config.get("model_type", "Logistic Regression"),
        "feature_count": state.config.get("feature_count", 0)
    }


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "Risk Assessment API",
        "version": "1.0.0",
        "docs": "/docs",
        "redoc": "/redoc"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
