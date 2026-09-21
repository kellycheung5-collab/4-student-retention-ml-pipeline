import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException, Query, Request, status
from pydantic import BaseModel, ConfigDict, Field

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api_logger")

BASE_DIR = Path(__file__).resolve().parent.parent
MLMODEL_DIR = BASE_DIR / "models" / "MLmodel"
JOBLIB_PATH = BASE_DIR / "models" / "early_warning_pipeline.joblib"

model: Any | None = None
explainer: Any | None = None


def load_model_artifact():
    """Load model from MLflow directory or fallback to standalone joblib file, and initialize SHAP explainer."""
    global model, explainer
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

        # Initialize SHAP explainer if model is successfully loaded
        if model is not None:
            try:
                # Retrieve underlying estimator if wrapped in a Pipeline
                estimator = (
                    model.steps[-1][1] if hasattr(model, "steps") else model
                )
                explainer = shap.TreeExplainer(estimator)
                logger.info("SHAP TreeExplainer initialized successfully.")
            except Exception as exp_err:  # noqa: BLE001
                logger.warning(
                    f"Failed to initialize SHAP TreeExplainer: {exp_err!s}"
                )
                explainer = None
    except Exception as e:  # noqa: BLE001
        logger.error(f"Failed to load model artifact: {e!s}")
        model = None
        explainer = None


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
        "shap_explainer_ready": explainer is not None,
        "model_loaded": is_ready,
    }


@app.post("/predict")
async def predict(
    payload: StudentFeatures,
    threshold: float = Query(0.5, ge=0.0, le=1.0),
):
    """Predict student dropout risk probability."""
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

        # Extract top SHAP risk factors driving dropout risk
        top_risk_factors = []
        if explainer is not None:
            try:
                # Preprocess input if pipeline step transformation is needed
                features_transformed = (
                    model.steps[0][1].transform(input_df)
                    if hasattr(model, "steps") and len(model.steps) > 1
                    else input_df
                )

                shap_values = explainer(features_transformed)
                raw_vals = shap_values.values[0]

                # Extract SHAP array for positive class (dropout risk)
                if len(raw_vals.shape) == 2:
                    raw_vals = raw_vals[:, 1]

                # Rank factors by highest positive contribution to dropout risk
                sorted_indices = np.argsort(raw_vals)[::-1]
                feature_names = input_df.columns

                for idx in sorted_indices[:5]:  # Top 5 contributing factors
                    val = float(raw_vals[idx])
                    if val > 0:
                        feat_name = str(feature_names[idx])
                        top_risk_factors.append(
                            {
                                "feature": feat_name,
                                "shap_value": round(val, 4),
                                "feature_value": float(data_dict[feat_name]),
                            }
                        )
            except Exception as exp_err:  # noqa: BLE001
                logger.warning(
                    f"Failed to generate SHAP values for request: {exp_err!s}"
                )

        return {
            "dropout_probability": round(dropout_prob, 4),
            "is_high_risk": is_high_risk,
            "decision_threshold": threshold,
            "top_risk_factors": top_risk_factors,
        }
    except Exception as e:  # noqa: BLE001
        logger.error(f"Prediction failure: {e!s}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction error: {e!s}",
        )