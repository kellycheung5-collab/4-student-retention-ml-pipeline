from unittest.mock import patch

import pytest
from httpx import AsyncClient


@pytest.mark.anyio
async def test_health_check(client: AsyncClient):
    """Verify that the health check endpoint returns 200 OK and model readiness status."""
    response = await client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "healthy"
    assert "shap_explainer_ready" in data
    assert "model_loaded" in data


@pytest.mark.anyio
async def test_predict_success_low_risk(client: AsyncClient):
    """Test valid prediction payload returning low risk prediction."""
    payload = {
        "Marital status": 1,
        "Application mode": 1,
        "Application order": 1,
        "Course": 9254,
        "Daytime/evening attendance": 1,
        "Previous qualification": 1,
        "Previous qualification (grade)": 120.0,
        "Nacionality": 1,
        "Mother's qualification": 13,
        "Father's qualification": 10,
        "Mother's occupation": 6,
        "Father's occupation": 10,
        "Admission grade": 120.0,
        "Displaced": 0,
        "Educational special needs": 0,
        "Debtor": 0,
        "Tuition fees up to date": 1,
        "Gender": 1,
        "Scholarship holder": 0,
        "Age at enrollment": 20,
        "International": 0,
        "Curricular units 1st sem (credited)": 0,
        "Curricular units 1st sem (enrolled)": 5,
        "Curricular units 1st sem (evaluations)": 5,
        "Curricular units 1st sem (approved)": 5,
        "Curricular units 1st sem (grade)": 12.0,
        "Curricular units 1st sem (without evaluations)": 0,
        "Unemployment rate": 10.8,
        "Inflation rate": 1.4,
        "GDP": 1.74,
    }

    response = await client.post("/predict?threshold=0.35", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert "dropout_probability" in data
    assert "is_high_risk" in data
    assert data["decision_threshold"] == 0.35
    assert "top_risk_factors" in data


@pytest.mark.anyio
async def test_predict_validation_error_invalid_types(client: AsyncClient):
    """Verify 422 Unprocessable Entity when field types are invalid."""
    invalid_payload = {
        "Marital status": "not_an_int",
        "Admission grade": "invalid_float",
    }

    response = await client.post("/predict", json=invalid_payload)
    assert response.status_code == 422


@pytest.mark.anyio
async def test_health_check_returns_tracing_headers(client: AsyncClient):
    """Verify that auto-generated correlation tracing headers are present on API responses."""
    response = await client.get("/health")
    assert response.status_code == 200
    assert "X-Correlation-ID" in response.headers


@pytest.mark.anyio
async def test_custom_correlation_id_propagation(client: AsyncClient):
    """Verify that an incoming X-Correlation-ID is preserved across the request lifecycle."""
    custom_id = "test-correlation-id-12345"
    response = await client.get("/health", headers={"X-Correlation-ID": custom_id})

    assert response.status_code == 200
    assert response.headers["X-Correlation-ID"] == custom_id


@pytest.mark.anyio
async def test_predict_model_uninitialized_returns_503(client: AsyncClient):
    """Cover 503 HTTP Exception when model artifact fails to load."""
    with patch("api.main.model", None), patch("api.main.load_model_artifact"):
        response = await client.post(
            "/predict",
            json={
                "Marital status": 1,
                "Application mode": 1,
                "Application order": 1,
                "Course": 9254,
                "Daytime/evening attendance": 1,
                "Previous qualification": 1,
                "Previous qualification (grade)": 120.0,
                "Nacionality": 1,
                "Mother's qualification": 13,
                "Father's qualification": 10,
                "Mother's occupation": 6,
                "Father's occupation": 10,
                "Admission grade": 120.0,
                "Displaced": 0,
                "Educational special needs": 0,
                "Debtor": 0,
                "Tuition fees up to date": 1,
                "Gender": 1,
                "Scholarship holder": 0,
                "Age at enrollment": 20,
                "International": 0,
                "Curricular units 1st sem (credited)": 0,
                "Curricular units 1st sem (enrolled)": 5,
                "Curricular units 1st sem (evaluations)": 5,
                "Curricular units 1st sem (approved)": 5,
                "Curricular units 1st sem (grade)": 12.0,
                "Curricular units 1st sem (without evaluations)": 0,
                "Unemployment rate": 10.8,
                "Inflation rate": 1.4,
                "GDP": 1.74,
            },
        )
        assert response.status_code == 503
        assert response.json()["detail"] == "Model artifact is not available."


@pytest.mark.anyio
async def test_predict_internal_error_returns_500(client: AsyncClient):
    """Cover 500 HTTP Exception when model prediction throws an exception."""
    with patch("api.main.model") as mock_model:
        mock_model.predict_proba.side_effect = Exception("Inference error")
        response = await client.post(
            "/predict",
            json={
                "Marital status": 1,
                "Application mode": 1,
                "Application order": 1,
                "Course": 9254,
                "Daytime/evening attendance": 1,
                "Previous qualification": 1,
                "Previous qualification (grade)": 120.0,
                "Nacionality": 1,
                "Mother's qualification": 13,
                "Father's qualification": 10,
                "Mother's occupation": 6,
                "Father's occupation": 10,
                "Admission grade": 120.0,
                "Displaced": 0,
                "Educational special needs": 0,
                "Debtor": 0,
                "Tuition fees up to date": 1,
                "Gender": 1,
                "Scholarship holder": 0,
                "Age at enrollment": 20,
                "International": 0,
                "Curricular units 1st sem (credited)": 0,
                "Curricular units 1st sem (enrolled)": 5,
                "Curricular units 1st sem (evaluations)": 5,
                "Curricular units 1st sem (approved)": 5,
                "Curricular units 1st sem (grade)": 12.0,
                "Curricular units 1st sem (without evaluations)": 0,
                "Unemployment rate": 10.8,
                "Inflation rate": 1.4,
                "GDP": 1.74,
            },
        )
        assert response.status_code == 500
        assert "Prediction error" in response.json()["detail"]