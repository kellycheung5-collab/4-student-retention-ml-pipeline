import os

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_processed_data(data_path: str):
    """Load early-warning processed dataset and separate features from target."""
    df = pd.read_csv(data_path)
    X = df.drop(columns=["Target", "target_binary"], errors="ignore")
    y = df["target_binary"]
    return X, y


def build_preprocessing_pipeline(X: pd.DataFrame) -> ColumnTransformer:
    """Define column transformer for numerical scaling and categorical encoding."""
    categorical_cols = [
        "Marital status",
        "Application mode",
        "Course",
        "Previous qualification",
        "Nacionality",
        "Mother's qualification",
        "Father's qualification",
        "Mother's occupation",
        "Father's occupation",
    ]
    
    # Filter to ensure columns exist in dataframe
    categorical_cols = [col for col in categorical_cols if col in X.columns]
    numeric_cols = [col for col in X.columns if col not in categorical_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                numeric_cols,
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_cols,
            ),
        ]
    )
    return preprocessor


def get_candidate_models():
    """Returns candidate estimators for pipeline comparison."""
    return {
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            class_weight="balanced",
            random_state=42,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=100,
            max_depth=6,
            class_weight="balanced",
            random_state=42,
        ),
    }


def train_and_evaluate_all(
    data_path: str = "data/processed/student_data_early_warning.csv",
    model_output_path: str = "models/early_warning_pipeline.joblib",
    experiment_name: str = "Student_Dropout_Early_Warning",
):
    """Trains candidate models, tracks runs in MLflow, and saves the best pipeline."""
    mlflow.set_experiment(experiment_name)

    # 1. Load Data & Stratified Split
    X, y = load_processed_data(data_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    preprocessor = build_preprocessing_pipeline(X)
    candidate_models = get_candidate_models()

    best_roc_auc = 0.0
    best_pipeline = None
    best_model_name = ""

    print("=== Starting Model Training & MLflow Comparison ===")

    # 2. Train and Evaluate Each Candidate Model
    for model_name, clf in candidate_models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", clf),
            ]
        )

        with mlflow.start_run(run_name=model_name):
            # Train pipeline on training split
            pipeline.fit(X_train, y_train)

            # Predictions
            y_pred = pipeline.predict(X_test)
            y_proba = pipeline.predict_proba(X_test)[:, 1]

            # Compute Metrics
            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
                "f1_score": f1_score(y_test, y_pred),
                "roc_auc": roc_auc_score(y_test, y_proba),
            }

            # Log Parameters & Metrics to MLflow
            mlflow.log_params(clf.get_params())
            mlflow.log_metrics(metrics)
            
            mlflow.sklearn.log_model(
                pipeline,
                artifact_path="model",
                serialization_format="cloudpickle",
            )

            print(f"\n--- {model_name} Results ---")
            for metric, val in metrics.items():
                print(f"- {metric.capitalize()}: {val:.4f}")

            # Track Best Performing Model based on ROC-AUC
            if metrics["roc_auc"] > best_roc_auc:
                best_roc_auc = metrics["roc_auc"]
                best_pipeline = pipeline
                best_model_name = model_name

    # 3. Save Top-Performing Model Artifact
    if best_pipeline is not None:
        os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
        joblib.dump(best_pipeline, model_output_path)
        print("\n==========================================")
        print(f" Best Model: {best_model_name} (ROC-AUC: {best_roc_auc:.4f})")
        print(f" Saved pipeline artifact to: {model_output_path}")
        print("==========================================")


if __name__ == "__main__":
    train_and_evaluate_all()