from contextlib import asynccontextmanager
import logging

from pathlib import Path
import uuid
from typing import Any, Optional
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request, status
from pydantic import BaseModel, ConfigDict, Field

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api_logger")

BASE_DIR = Path(__file__).resolve().parent.parent
MLMODEL_DIR = BASE_DIR / "models" / "MLmodel"
JOBLIB_PATH = BASE_DIR / "models" / "early_warning_pipeline.joblib"

model: Optional[Any] = None


def load_model_artifact():
    """Load model from MLflow directory or fallback to standalone joblib file."""
    global model
    try:
        if MLMODEL_DIR.exists():
            import mlflow.pyfunc

            model = mlflow.pyfunc.load_model(str(BASE_DIR / "models"))
            logger.info(f"Loaded MLflow model successfully from {MLMODEL_DIR}")
        elif JOBLIB_PATH.exists():
            model = joblib.load(JOBLIB_PATH)
            logger.info(f"Loaded joblib model successfully from {JOBLIB_PATH}")
        else:
            logger.warning(
                f"Model artifact not found at {MLMODEL_DIR} or {JOBLIB_PATH}"
            )
            model = None
    except Exception as e:
        logger.error(f"Failed to load model artifact: {str(e)}")
        model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown tasks."""
    load_model_artifact()
    yield


app = FastAPI(
    title="Student Early Warning System API",
    description="API for predicting student dropout risk using ML models.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def add_correlation_id_header(request: Request, call_next):
    """Middleware to inject/propagate X-Correlation-ID across HTTP requests."""
    correlation_id = request.headers.get("X-Correlation-ID") or str(
        uuid.uuid4()
    )
    response = await call_next(request)
    response.headers["X-Correlation-ID"] = correlation_id
    return response


class StudentFeatures(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    Marital_status: int = Field(..., alias="Marital status")
    Application_mode: int = Field(..., alias="Application mode")
    Application_order: int = Field(..., alias="Application order")
    Course: int
    Daytime_evening_attendance: int = Field(
        ..., alias="Daytime/evening attendance"
    )
    Previous_qualification: int = Field(..., alias="Previous qualification")
    Previous_qualification_grade: float = Field(
        ..., alias="Previous qualification (grade)"
    )
    Nacionality: int
    Mothers_qualification: int = Field(..., alias="Mother's qualification")
    Fathers_qualification: int = Field(..., alias="Father's qualification")
    Mothers_occupation: int = Field(..., alias="Mother's occupation")
    Fathers_occupation: int = Field(..., alias="Father's occupation")
    Admission_grade: float = Field(..., alias="Admission grade")
    Displaced: int
    Educational_special_needs: int = Field(
        ..., alias="Educational special needs"
    )
    Debtor: int
    Tuition_fees_up_to_date: int = Field(..., alias="Tuition fees up to date")
    Gender: int
    Scholarship_holder: int = Field(..., alias="Scholarship holder")
    Age_at_enrollment: int = Field(..., alias="Age at enrollment")
    International: int
    Curricular_units_1st_sem_credited: int = Field(
        ..., alias="Curricular units 1st sem (credited)"
    )
    Curricular_units_1st_sem_enrolled: int = Field(
        ..., alias="Curricular units 1st sem (enrolled)"
    )
    Curricular_units_1st_sem_evaluations: int = Field(
        ..., alias="Curricular units 1st sem (evaluations)"
    )
    Curricular_units_1st_sem_approved: int = Field(
        ..., alias="Curricular units 1st sem (approved)"
    )
    Curricular_units_1st_sem_grade: float = Field(
        ..., alias="Curricular units 1st sem (grade)"
    )
    Curricular_units_1st_sem_without_evaluations: int = Field(
        ..., alias="Curricular units 1st sem (without evaluations)"
    )
    Unemployment_rate: float = Field(..., alias="Unemployment rate")
    Inflation_rate: float = Field(..., alias="Inflation rate")
    GDP: float


@app.get("/health")
def health_check():
    """Health check endpoint checking service status and model readiness."""
    is_ready = model is not None
    return {
        "status": "healthy",
        "shap_explainer_ready": is_ready,
        "model_loaded": is_ready,
    }


@app.post("/predict")
async def predict(
    payload: StudentFeatures,
    threshold: float = Query(0.5, ge=0.0, le=1.0),
):
    """Predict student dropout risk probability."""
    global model
    if model is None:
        load_model_artifact()
        if model is None:
            logger.warning("Attempted prediction but model is uninitialized.")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model artifact is not available.",
            )

    try:
        data_dict = payload.model_dump(by_alias=True)
        input_df = pd.DataFrame([data_dict])

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_df)
            dropout_prob = float(probabilities[0][1])
        elif hasattr(model, "predict"):
            dropout_prob = float(model.predict(input_df)[0])
        else:
            raise ValueError("Model object lacks prediction methods.")

        is_high_risk = dropout_prob >= threshold

        return {
            "dropout_probability": round(dropout_prob, 4),
            "is_high_risk": is_high_risk,
            "decision_threshold": threshold,
            "top_risk_factors": [],
        }
    except Exception as e:
        logger.error(f"Prediction failure: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction error: {str(e)}",
        )