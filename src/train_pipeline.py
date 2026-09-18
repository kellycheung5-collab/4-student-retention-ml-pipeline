import sys
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from src.load_data import load_and_prep_features

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "early_warning_pipeline.joblib"


def train_and_save_pipeline():
    """Train RandomForest baseline model and persist pipeline artifact."""
    X, y = load_and_prep_features()

    X_train, X_test, y_train, _y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Validate model inference pipeline
    clf.predict(X_train if len(X_test) == 0 else X_test)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, MODEL_PATH)


if __name__ == "__main__":
    train_and_save_pipeline()