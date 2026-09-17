import pytest
from asgi_lifespan import LifespanManager
from httpx import ASGITransport, AsyncClient

from api.main import app


@pytest.fixture(scope="module")
async def client():
    """Asynchronous test client with lifespan execution, bypassing Starlette TestClient deprecations."""
    async with LifespanManager(app) as manager:
        transport = ASGITransport(app=manager.app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac


@pytest.mark.anyio
async def test_health_check(client: AsyncClient):
    """Verify that the health check endpoint returns 200 OK and SHAP readiness."""
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "shap_explainer_ready" in data


@pytest.mark.anyio
async def test_predict_success_low_risk(client: AsyncClient):
    """Test valid prediction payload returning low risk and SHAP risk factors."""
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
    
    # Validate SHAP risk factors payload structure
    assert "top_risk_factors" in data
    assert isinstance(data["top_risk_factors"], list)
    if len(data["top_risk_factors"]) > 0:
        factor = data["top_risk_factors"][0]
        assert "feature" in factor
        assert "shap_value" in factor
        assert "feature_value" in factor


@pytest.mark.anyio
async def test_predict_validation_error_invalid_types(client: AsyncClient):
    """Verify 422 Unprocessable Entity when field types are invalid."""
    invalid_payload = {
        "Marital status": "not_an_int",
        "Admission grade": "invalid_float",
    }

    response = await client.post("/predict", json=invalid_payload)
    assert response.status_code == 422