# 4-student-retention-ml-pipeline
End-to-end ML pipeline predicting student dropout risk using the UCI dataset. Features an sklearn Pipeline, MLflow experiment tracking, SHAP explainability with a subgroup fairness audit, and a FastAPI REST API deployed to Azure.

student-retention-ml-pipeline/
├── data/
│   ├── raw/                 # Cache/export of UCI dataset
│   └── processed/           # Filtered early-warning split CSVs
├── notebooks/               # Exploratory notebooks (EDA, initial SHAP)
├── src/
│   ├── __init__.py
│   ├── load_data.py         # Step 2: Ingestion & leakage boundary definition
│   ├── train_pipeline.py    # Step 3 & 4: sklearn Pipeline + MLflow tracking
│   └── evaluate_fairness.py # Step 5: SHAP + subgroup fairness audit
├── api/
│   ├── __init__.py
│   ├── main.py              # Step 6: FastAPI app with /predict & /health
│   └── schemas.py           # Step 6: Pydantic input/output schemas
├── mlruns/                  # Local MLflow runs (gitignored)
├── notes.md                 # Decision log (leakage, fairness findings, deployment choices)
├── requirements.txt         # Dependency declarations
└── README.md                # Final architecture, findings, and deployment guide