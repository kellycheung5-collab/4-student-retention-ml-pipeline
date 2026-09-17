import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException, Query, status
from pydantic import BaseModel, ConfigDict, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Base path resolution using pathlib
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "early_warning_pipeline.joblib"

# Global application state using modern Python 3.10+ union syntax
pipeline_artifact: Any | None = None
shap_explainer: shap.TreeExplainer | None = None


class StudentFeatures(BaseModel):
    marital_status: int = Field(..., alias="Marital status", ge=1, le=6)
    application_mode: int = Field(..., alias="Application mode", ge=1, le=57)
    application_order: int = Field(..., alias="Application order", ge=0, le=9)
    course: int = Field(..., alias="Course", ge=1)
    daytime_evening_attendance: int = Field(..., alias="Daytime/evening attendance", ge=0, le=1)
    previous_qualification: int = Field(..., alias="Previous qualification", ge=1)
    previous_qualification_grade: float = Field(..., alias="Previous qualification (grade)", ge=0.0, le=200.0)
    nacionality: int = Field(..., alias="Nacionality", ge=1)
    mother_qualification: int = Field(..., alias="Mother's qualification", ge=1)
    father_qualification: int = Field(..., alias="Father's qualification", ge=1)
    mother_occupation: int = Field(..., alias="Mother's occupation", ge=0)
    father_occupation: int = Field(..., alias="Father's occupation", ge=0)
    admission_grade: float = Field(..., alias="Admission grade", ge=0.0, le=200.0)
    displaced: int = Field(..., alias="Displaced", ge=0, le=1)
    educational_special_needs: int = Field(..., alias="Educational special needs", ge=0, le=1)
    debtor: int = Field(..., alias="Debtor", ge=0, le=1)
    tuition_fees_up_to_date: int = Field(..., alias="Tuition fees up to date", ge=0, le=1)
    gender: int = Field(..., alias="Gender", ge=0, le=1)
    scholarship_holder: int = Field(..., alias="Scholarship holder", ge=0, le=1)
    age_at_enrollment: int = Field(..., alias="Age at enrollment", ge=10, le=100)
    international: int = Field(..., alias="International", ge=0, le=1)
    curricular_units_1st_sem_credited: int = Field(..., alias="Curricular units 1st sem (credited)", ge=0)
    curricular_units_1st_sem_enrolled: int = Field(..., alias="Curricular units 1st sem (enrolled)", ge=0)
    curricular_units_1st_sem_evaluations: int = Field(..., alias="Curricular units 1st sem (evaluations)", ge=0)
    curricular_units_1st_sem_approved: int = Field(..., alias="Curricular units 1st sem (approved)", ge=0)
    curricular_units_1st_sem_grade: float = Field(..., alias="Curricular units 1st sem (grade)", ge=0.0, le=20.0)
    curricular_units_1st_sem_without_evaluations: int = Field(..., alias="Curricular units 1st sem (without evaluations)", ge=0)
    unemployment_rate: float = Field(..., alias="Unemployment rate")
    inflation_rate: float = Field(..., alias="Inflation rate")
    gdp: float = Field(..., alias="GDP")

    model_config = ConfigDict(
        populate_by_name=True,
        protected_namespaces=(),
    )


class RiskFactor(BaseModel):
    feature: str
    shap_value: float
    feature_value: float


class PredictionResponse(BaseModel):
    student_id: str | None = None
    dropout_probability: float
    is_high_risk: bool
    decision_threshold: float
    top_risk_factors: list[RiskFactor] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for initializing model and SHAP explainer."""
    global pipeline_artifact, shap_explainer

    if not MODEL_PATH.exists():
        logger.error("Model file not found at %s", MODEL_PATH)
        raise FileNotFoundError(f"Model artifact missing: {MODEL_PATH}")

    logger.info("Loading pipeline artifact from %s", MODEL_PATH)
    pipeline_artifact = joblib.load(MODEL_PATH)

    try:
        if hasattr(pipeline_artifact, "named_steps"):
            model_step = pipeline_artifact.steps[-1][1]
        else:
            model_step = pipeline_artifact

        logger.info("Initializing SHAP TreeExplainer...")
        shap_explainer = shap.TreeExplainer(model_step)
        logger.info("SHAP TreeExplainer initialized successfully.")
    except Exception as e:  # noqa: BLE001
        logger.warning("Failed to initialize SHAP TreeExplainer: %s", e)
        shap_explainer = None

    yield

    logger.info("Shutting down application...")
    pipeline_artifact = None
    shap_explainer = None


app = FastAPI(
    title="Student Dropout Early Warning API",
    description="Microservice providing dropout risk predictions and local SHAP feature attributions.",
    version="1.1.0",
    lifespan=lifespan,
)


@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check() -> dict[str, Any]:
    """Health check endpoint validating pipeline availability."""
    if pipeline_artifact is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ML pipeline artifact is not loaded.",
        )
    return {
        "status": "healthy",
        "pipeline_loaded": True,
        "shap_explainer_ready": shap_explainer is not None,
    }


@app.post("/predict", response_model=PredictionResponse, status_code=status.HTTP_200_OK)
async def predict_dropout_risk(
    features: StudentFeatures,
    student_id: str | None = Query(None, description="Optional unique identifier for student tracking"),
    threshold: float = Query(0.35, ge=0.0, le=1.0, description="Risk classification probability threshold"),
) -> PredictionResponse:
    """Predict student dropout risk and return top positive SHAP risk attributions."""
    if pipeline_artifact is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model pipeline is uninitialized.",
        )

    input_df = pd.DataFrame([features.model_dump(by_alias=True)])

    try:
        probabilities = pipeline_artifact.predict_proba(input_df)
        dropout_prob = float(probabilities[0][1])
        is_high_risk = dropout_prob >= threshold

        top_risk_factors: list[RiskFactor] = []

        if shap_explainer is not None:
            if hasattr(pipeline_artifact, "named_steps"):
                preprocessor = pipeline_artifact[:-1]
                transformed_data = preprocessor.transform(input_df)
            else:
                transformed_data = input_df

            shap_values = shap_explainer.shap_values(transformed_data)

            if isinstance(shap_values, list):
                values_cls1 = shap_values[1][0]
            elif len(shap_values.shape) == 3:
                values_cls1 = shap_values[0, :, 1]
            else:
                values_cls1 = shap_values[0]

            cols = input_df.columns.tolist()
            raw_vals = input_df.iloc[0].values

            impacts = []
            for col, val, s_val in zip(cols, raw_vals, values_cls1):
                impacts.append({
                    "feature": col,
                    "feature_value": float(val),
                    "shap_value": float(s_val),
                })

            sorted_impacts = sorted(impacts, key=lambda x: x["shap_value"], reverse=True)
            positive_impacts = [x for x in sorted_impacts if x["shap_value"] > 0][:3]

            top_risk_factors = [RiskFactor(**item) for item in positive_impacts]

        return PredictionResponse(
            student_id=student_id,
            dropout_probability=round(dropout_prob, 4),
            is_high_risk=is_high_risk,
            decision_threshold=threshold,
            top_risk_factors=top_risk_factors,
        )

    except Exception as e:
        logger.exception("Inference error occurred during prediction.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference execution failed: {e!s}",
        )