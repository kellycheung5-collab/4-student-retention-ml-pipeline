4-Student Retention ML Pipeline
End-to-end MLOps pipeline predicting student dropout risk using the UCI Predict Students' Dropout and Academic Success dataset. Features an scikit-learn training pipeline, MLflow experiment tracking, SHAP explainability with a subgroup fairness audit, automated GitHub Actions CI/CD, and a containerized FastAPI REST API deployed to Azure Container Apps.   


Project Structure

4-student-retention-ml-pipeline/
├── .github/
│   └── workflows/
│       └── ci.yml             # Continuous Integration (linting, tests, coverage)
├── api/
│   ├── __init__.py
│   ├── main.py                # FastAPI app with /predict & /health endpoints
│   └── schemas.py             # Pydantic input/output validation schemas
├── data/
│   ├── raw/                   # Raw UCI dataset (data.csv)
│   └── processed/             # Filtered early-warning split datasets
├── models/
│   └── early_warning_pipeline.joblib  # Exported inference pipeline artifact
├── notebooks/
│   └── data_exploration.ipynb # Exploratory data analysis & initial SHAP experiments
├── src/
│   ├── __init__.py
│   ├── load_data.py           # Ingestion & data leakage boundary definition
│   └── train_pipeline.py      # scikit-learn Pipeline & MLflow tracking
├── tests/
│   ├── conftest.py            # Test fixtures
│   └── test_api.py            # Pytest suite for API endpoints and validation
├── Dockerfile                 # Multi-stage production container configuration
├── LICENSE                    # Repository license
├── README.md                  # Project documentation
├── ml-requirements.txt        # Production dependencies
├── pyproject.toml             # Ruff and Pytest configurations
└── requirements-dev.txt       # Development and testing dependencies

Key Features & ArchitectureData Leakage Safeguards: Strict temporal splitting ensuring 2nd-semester variables are excluded to maintain an authentic early-warning window (1st-semester evaluation phase).MLflow Tracking: Complete tracking of model parameters, metrics (ROC-AUC, F1-Score, Precision, Recall), and serialized pipeline artifacts.Explainability & Fairness: Local and global SHAP (SHapley Additive exPlanations) values coupled with demographic subgroup audits (gender, age at enrollment, scholarship status).Automated CI Pipeline: Automated unit testing, code linting (Ruff), and coverage reporting via GitHub Actions.Cloud Infrastructure: Azure Container Registry (ACR) hosting dockerized images served on serverless Azure Container Apps.Cloud Deployment DetailsLive API Base URL: [https://student-dropout-api.ashyhill-c0196008.westus2.azurecontainerapps.io](https://student-dropout-api.ashyhill-c0196008.westus2.azurecontainerapps.io)   Swagger OpenAPI Specs: [https://student-dropout-api.ashyhill-c0196008.westus2.azurecontainerapps.io/docs](https://student-dropout-api.ashyhill-c0196008.westus2.azurecontainerapps.io/docs)   Health Check: [https://student-dropout-api.ashyhill-c0196008.westus2.azurecontainerapps.io/health](https://student-dropout-api.ashyhill-c0196008.westus2.azurecontainerapps.io/health)   Quickstart & Usage1. Local SetupClone the repository and set up your virtual environment:Bashgit clone https://github.com/your-username/4-student-retention-ml-pipeline.git
cd 4-student-retention-ml-pipeline
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\Activate.ps1
pip install -r requirements.txt
2. Run Tests & PipelineExecute the test suite and train the pipeline locally:Bashpytest
python src/train_pipeline.py
3. Launch Local REST APIStart the FastAPI development server:Bashuvicorn api.main:app --reload --port 8000
4. Sample REST API RequestSend an inference request to the live Azure endpoint using PowerShell:PowerShell$URL = "https://student-dropout-api.ashyhill-c0196008.westus2.azurecontainerapps.io/predict"

$Body = @{
    "Marital status" = 1
    "Application mode" = 17
    "Application order" = 1
    "Course" = 9254
    "Daytime/evening attendance" = 1
    "Previous qualification" = 1
    "Previous qualification (grade)" = 122.0
    "Nacionality" = 1
    "Mother's qualification" = 19
    "Father's qualification" = 37
    "Mother's occupation" = 5
    "Father's occupation" = 9
    "Admission grade" = 127.3
    "Displaced" = 1
    "Educational special needs" = 0
    "Debtor" = 0
    "Tuition fees up to date" = 1
    "Gender" = 1
    "Scholarship holder" = 0
    "Age at enrollment" = 20
    "International" = 0
    "Curricular units 1st sem (credited)" = 0
    "Curricular units 1st sem (enrolled)" = 6
    "Curricular units 1st sem (evaluations)" = 6
    "Curricular units 1st sem (approved)" = 6
    "Curricular units 1st sem (grade)" = 14.0
    "Curricular units 1st sem (without evaluations)" = 0
    "Unemployment rate" = 10.8
    "Inflation rate" = 1.4
    "GDP" = 1.74
} | ConvertTo-Json

Invoke-RestMethod -Uri $URL -Method Post -ContentType "application/json" -Body $Body