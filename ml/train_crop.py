"""Train the crop recommender and save it as a single sklearn Pipeline.

Differences from ``crop_model.py`` in CropCraft 1.x:

*   **Features are scaled.** 1.x fed raw columns to SVC and KNN, where N spans
    0-140 and rainfall spans 20-300, so both models were dominated by rainfall
    magnitude. Scaling now lives inside the pipeline, so it is applied
    identically at training and inference time and cannot be forgotten.
*   **The test set is only touched once.** 1.x ran ``cross_val_score`` on the
    test set and reported that as the score. Model selection here runs on the
    training split; the held-out set is scored once at the end.
*   **The split is seeded**, so results reproduce.
*   **Fewer, less redundant estimators.** 1.x soft-voted 13 models including
    six near-identical SVCs and five KNNs.
*   **Metadata travels with the model** -- feature order, class list, metrics
    and library versions -- so the serving code cannot silently reorder inputs.

Usage::

    python -m ml.train_crop
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET = PROJECT_ROOT / "Data" / "crop_recommendation.csv"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "crop_recommender.joblib"
METRICS_PATH = MODEL_DIR / "crop_metrics.json"

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
SEED = 42


def build_candidates() -> dict[str, Pipeline]:
    """Return the candidate pipelines to compare, each with scaling built in."""

    def pipe(estimator) -> Pipeline:
        return Pipeline([("scale", StandardScaler()), ("clf", estimator)])

    return {
        "random_forest": pipe(
            RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1)
        ),
        "hist_gradient_boosting": pipe(HistGradientBoostingClassifier(random_state=SEED)),
        "svc_rbf": pipe(SVC(probability=True, random_state=SEED)),
        "knn": pipe(KNeighborsClassifier(n_neighbors=5)),
        "gaussian_nb": pipe(GaussianNB()),
    }


def build_ensemble() -> Pipeline:
    """Soft-voting ensemble over four complementary model families."""
    voting = VotingClassifier(
        estimators=[
            ("rf", RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1)),
            ("svc", SVC(probability=True, random_state=SEED)),
            ("knn", KNeighborsClassifier(n_neighbors=5)),
            ("gnb", GaussianNB()),
        ],
        voting="soft",
        n_jobs=-1,
    )
    return Pipeline([("scale", StandardScaler()), ("clf", voting)])


def main() -> None:
    frame = pd.read_csv(DATASET)
    features = frame[FEATURES].to_numpy(dtype=np.float64)
    labels = frame["label"].to_numpy()

    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=SEED, stratify=labels
    )
    print(f"train={len(x_train)}  test={len(x_test)}  classes={len(set(labels))}\n")

    # --- model selection, on the training split only ------------------------
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_scores: dict[str, float] = {}
    for name, candidate in {**build_candidates(), "voting_ensemble": build_ensemble()}.items():
        scores = cross_val_score(candidate, x_train, y_train, cv=folds, scoring="accuracy")
        cv_scores[name] = float(scores.mean())
        print(f"  {name:24s} cv accuracy {scores.mean():.4f} (+/- {scores.std():.4f})")

    best_name = max(cv_scores, key=cv_scores.__getitem__)
    print(f"\nbest by cross-validation: {best_name}")

    # --- fit the chosen model and score the held-out set exactly once -------
    model = build_ensemble() if best_name == "voting_ensemble" else build_candidates()[best_name]
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    accuracy = float(accuracy_score(y_test, predictions))
    macro_f1 = float(f1_score(y_test, predictions, average="macro"))
    print(f"\nheld-out accuracy : {accuracy:.4f}")
    print(f"held-out macro F1 : {macro_f1:.4f}\n")
    print(classification_report(y_test, predictions, digits=3, zero_division=0))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "features": FEATURES,
            "classes": sorted(set(labels)),
            "sklearn_version": sklearn.__version__,
        },
        MODEL_PATH,
        compress=3,
    )
    METRICS_PATH.write_text(
        json.dumps(
            {
                "selected_model": best_name,
                "cv_accuracy": cv_scores,
                "holdout_accuracy": accuracy,
                "holdout_macro_f1": macro_f1,
                "n_train": int(len(x_train)),
                "n_test": int(len(x_test)),
                "seed": SEED,
                "sklearn_version": sklearn.__version__,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"saved {MODEL_PATH}")
    print(f"saved {METRICS_PATH}")


if __name__ == "__main__":
    main()
