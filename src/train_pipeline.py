from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_PATH = BASE_DIR / "data" / "processed" / "student_data_early_warning.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "early_warning_pipeline.joblib"


def load_and_prep_features(data_path: Path = PROCESSED_PATH):
    if not data_path.exists():
        from load_data import load_and_preprocess_data

        load_and_preprocess_data()

    df = pd.read_csv(data_path)

    non_feature_cols = [
        "Target",
        "target_binary",
        "dropout_risk",
        "student_id",
    ]
    X = df.drop(columns=[col for col in non_feature_cols if col in df.columns])
    y = df["dropout_risk"]

    return X, y


def train_and_evaluate_all():
    X, y = load_and_prep_features()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_train if len(X_test) == 0 else X_test)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    print(f"Model saved successfully to {MODEL_PATH}")


if __name__ == "__main__":
    train_and_evaluate_all()