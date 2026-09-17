import os
from typing import Dict, Any, List, Optional
from contextlib import asynccontextmanager
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, status
from pydantic import BaseModel, Field

# Global reference for the loaded model pipeline
pipeline_artifact = None
MODEL_PATH = os.path.join("models", "early_warning_pipeline.joblib")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager to load ML model on startup."""
    global pipeline_artifact
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. Run src/train_pipeline.py first."
        )
    pipeline_artifact = joblib.load(MODEL_PATH)
    print(f"Successfully loaded model pipeline from {MODEL_PATH}")
    yield
    # Clean up resources on shutdown if needed
    pipeline_artifact = None


app = FastAPI(
    title="Student Early Warning System API",
    description="Inference API for predicting student dropout risk and enabling early interventions.",
    version="1.0.0",
    lifespan=lifespan,
)


class StudentFeatures(BaseModel):
    """Input payload schema containing all dataset features required by the pipeline."""

    marital_status: int = Field(1, alias="Marital status")
    application_mode: int = Field(1, alias="Application mode")
    application_order: int = Field(1, alias="Application order")
    course: int = Field(9254, alias="Course")
    daytime_evening_attendance: int = Field(1, alias="Daytime/evening attendance")
    previous_qualification: int = Field(1, alias="Previous qualification")
    previous_qualification_grade: float = Field(120.0, alias="Previous qualification (grade)")
    nacionality: int = Field(1, alias="Nacionality")
    mother_qualification: int = Field(13, alias="Mother's qualification")
    father_qualification: int = Field(10, alias="Father's qualification")
    mother_occupation: int = Field(6, alias="Mother's occupation")
    father_occupation: int = Field(10, alias="Father's occupation")
    admission_grade: float = Field(120.0, alias="Admission grade")
    displaced: int = Field(0, alias="Displaced")
    educational_special_needs: int = Field(0, alias="Educational special needs")
    debtor: int = Field(0, alias="Debtor")
    tuition_fees_up_to_date: int = Field(1, alias="Tuition fees up to date")
    gender: int = Field(1, alias="Gender")
    scholarship_holder: int = Field(0, alias="Scholarship holder")
    age_at_enrollment: int = Field(20, alias="Age at enrollment")
    international: int = Field(0, alias="International")
    
    # Curricular Units 1st Semester
    curricular_units_1st_sem_credited: int = Field(0, alias="Curricular units 1st sem (credited)")
    curricular_units_1st_sem_enrolled: int = Field(5, alias="Curricular units 1st sem (enrolled)")
    curricular_units_1st_sem_evaluations: int = Field(5, alias="Curricular units 1st sem (evaluations)")
    curricular_units_1st_sem_approved: int = Field(5, alias="Curricular units 1st sem (approved)")
    curricular_units_1st_sem_grade: float = Field(12.0, alias="Curricular units 1st sem (grade)")
    curricular_units_1st_sem_without_evaluations: int = Field(0, alias="Curricular units 1st sem (without evaluations)")
    
    # Macroeconomic Metrics
    unemployment_rate: float = Field(10.8, alias="Unemployment rate")
    inflation_rate: float = Field(1.4, alias="Inflation rate")
    gdp: float = Field(1.74, alias="GDP")

    class Config:
        populate_by_name = True


class PredictionResponse(BaseModel):
    """Output prediction payload containing raw probability and binary decision."""
    
    dropout_probability: float = Field(..., description="Continuous risk probability score (0.0 to 1.0)")
    is_high_risk: bool = Field(..., description="Flag indicating if risk probability exceeds the active threshold")
    decision_threshold: float = Field(..., description="Applied decision threshold for classification")
    risk_level: str = Field(..., description="Categorized risk label (Low, Moderate, High)")


@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    """Health check endpoint to confirm service availability."""
    return {
        "status": "healthy",
        "model_loaded": pipeline_artifact is not None,
    }


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