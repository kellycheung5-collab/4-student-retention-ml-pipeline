import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, status
from pydantic import BaseModel, ConfigDict, Field

# 1. Define global model reference
pipeline_artifact = None

# 2. Resolve MODEL_PATH relative to project root
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "early_warning_pipeline.joblib"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager to load ML model on startup."""
    global pipeline_artifact
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. Run src/train_pipeline.py first."
        )
    pipeline_artifact = joblib.load(MODEL_PATH)
    print(f"Successfully loaded model pipeline from {MODEL_PATH}")
    yield
    pipeline_artifact = None


app = FastAPI(
    title="Student Early Warning System API",
    description="Inference API for predicting student dropout risk and enabling early interventions.",
    version="1.0.0",
    lifespan=lifespan,
)


class StudentFeatures(BaseModel):
    """Input payload schema containing all dataset features required by the pipeline."""
    model_config = ConfigDict(populate_by_name=True)

    marital_status: int = Field(..., alias="Marital status")
    application_mode: int = Field(..., alias="Application mode")
    application_order: int = Field(..., alias="Application order")
    course: int = Field(..., alias="Course")
    daytime_evening_attendance: int = Field(..., alias="Daytime/evening attendance")
    previous_qualification: int = Field(..., alias="Previous qualification")
    previous_qualification_grade: float = Field(..., alias="Previous qualification (grade)")
    nacionality: int = Field(..., alias="Nacionality")
    mother_qualification: int = Field(..., alias="Mother's qualification")
    father_qualification: int = Field(..., alias="Father's qualification")
    mother_occupation: int = Field(..., alias="Mother's occupation")
    father_occupation: int = Field(..., alias="Father's occupation")
    admission_grade: float = Field(..., alias="Admission grade")
    displaced: int = Field(..., alias="Displaced")
    educational_special_needs: int = Field(..., alias="Educational special needs")
    debtor: int = Field(..., alias="Debtor")
    tuition_fees_up_to_date: int = Field(..., alias="Tuition fees up to date")
    gender: int = Field(..., alias="Gender")
    scholarship_holder: int = Field(..., alias="Scholarship holder")
    age_at_enrollment: int = Field(..., alias="Age at enrollment")
    international: int = Field(..., alias="International")
    
    # Curricular Units 1st Semester
    curricular_units_1st_sem_credited: int = Field(..., alias="Curricular units 1st sem (credited)")
    curricular_units_1st_sem_enrolled: int = Field(..., alias="Curricular units 1st sem (enrolled)")
    curricular_units_1st_sem_evaluations: int = Field(..., alias="Curricular units 1st sem (evaluations)")
    curricular_units_1st_sem_approved: int = Field(..., alias="Curricular units 1st sem (approved)")
    curricular_units_1st_sem_grade: float = Field(..., alias="Curricular units 1st sem (grade)")
    curricular_units_1st_sem_without_evaluations: int = Field(..., alias="Curricular units 1st sem (without evaluations)")
    
    # Macroeconomic Metrics
    unemployment_rate: float = Field(..., alias="Unemployment rate")
    inflation_rate: float = Field(..., alias="Inflation rate")
    gdp: float = Field(..., alias="GDP")


class PredictionResponse(BaseModel):
    """Output prediction payload containing raw probability and binary decision."""
    
    dropout_probability: float = Field(..., description="Continuous risk probability score (0.0 to 1.0)")
    is_high_risk: bool = Field(..., description="Flag indicating if risk probability exceeds the active threshold")
    decision_threshold: float = Field(..., description="Applied decision threshold for classification")
    risk_level: str = Field(..., description="Categorized risk label (Low, Moderate, High)")


@app.get("/health")
def health_check():
    """Verify service health and model loading status."""
    if pipeline_artifact is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model artifact is not loaded.",
        )
    return {"status": "healthy", "model_loaded": True}


@app.post("/predict", response_model=PredictionResponse)
def predict_dropout_risk(
    payload: StudentFeatures,
    threshold: float = Query(
        0.35,
        ge=0.0,
        le=1.0,
        description="Probability threshold for flagging high-risk students (Default: 0.35 for high recall)",
    ),
):
    """Predict dropout risk probability for an individual student."""
    if pipeline_artifact is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model pipeline is not loaded.",
        )

    # Convert incoming Pydantic payload to DataFrame (preserving exact feature key names)
    input_data = pd.DataFrame([payload.model_dump(by_alias=True)])

    try:
        # Extract dropout probability score (Class 1)
        proba = float(pipeline_artifact.predict_proba(input_data)[0, 1])
        is_high_risk = proba >= threshold

        # Categorize operational risk level
        if proba >= 0.60:
            risk_level = "High"
        elif proba >= threshold:
            risk_level = "Moderate"
        else:
            risk_level = "Low"

        return PredictionResponse(
            dropout_probability=round(proba, 4),
            is_high_risk=is_high_risk,
            decision_threshold=threshold,
            risk_level=risk_level,
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}",
        )