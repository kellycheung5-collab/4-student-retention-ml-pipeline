import subprocess
from collections.abc import AsyncGenerator
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from api.main import app

BASE_DIR = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def anyio_backend():
    return "asyncio"


@pytest.fixture(scope="session", autouse=True)
def ensure_model_artifact_exists():
    """Ensure raw data preprocessing and training pipeline execute before API tests run."""
    joblib_path = BASE_DIR / "models" / "early_warning_pipeline.joblib"
    mlmodel_dir = BASE_DIR / "models" / "MLmodel"

    if not (joblib_path.exists() or mlmodel_dir.exists()):
        subprocess.run(["python", "src/load_data.py"], check=True, cwd=BASE_DIR)
        subprocess.run(
            ["python", "src/train_pipeline.py"], check=True, cwd=BASE_DIR
        )


@pytest.fixture
async def client() -> AsyncGenerator[AsyncClient, None]:
    """Provide an AsyncClient instance configured with the FastAPI app transport."""
    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport, base_url="http://testserver"
    ) as ac:
        yield ac