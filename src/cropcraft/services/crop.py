"""Crop recommendation service."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from cropcraft.config import get_settings
from cropcraft.schemas import CropCandidate, CropRequest, CropResponse


class ModelUnavailableError(RuntimeError):
    """Raised when a model artifact has not been built yet."""


@lru_cache
def _load_bundle() -> dict[str, Any]:
    """Load the crop pipeline and its metadata once per process."""
    path = get_settings().crop_model_path
    if not path.exists():
        raise ModelUnavailableError(
            f"{path.name} is missing. Build it with: python -m ml.train_crop"
        )
    return joblib.load(path)


def crop_classes() -> list[str]:
    """Every crop the model can predict."""
    return list(_load_bundle()["classes"])


def metrics() -> dict[str, Any]:
    """Training metrics, or an empty dict if the report is absent."""
    path = Path(get_settings().crop_model_path).with_name("crop_metrics.json")
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def recommend(request: CropRequest, top_k: int = 3) -> CropResponse:
    """Recommend a crop for the given soil and weather readings.

    Feature order comes from the saved bundle rather than being hardcoded at
    the call site, so the columns cannot silently drift out of the order the
    scaler was fitted on.
    """
    bundle = _load_bundle()
    model, feature_names = bundle["model"], bundle["features"]

    values = request.model_dump()
    # The CSV column names differ from the form field names.
    lookup = {
        "N": values["nitrogen"],
        "P": values["phosphorous"],
        "K": values["potassium"],
        "temperature": values["temperature"],
        "humidity": values["humidity"],
        "ph": values["ph"],
        "rainfall": values["rainfall"],
    }
    row = np.array([[float(lookup[name]) for name in feature_names]], dtype=np.float64)

    probabilities = model.predict_proba(row)[0]
    classes = list(model.classes_)
    ranked = sorted(
        zip(classes, probabilities, strict=True), key=lambda pair: pair[1], reverse=True
    )

    best_crop, best_score = ranked[0]
    return CropResponse(
        crop=best_crop,
        confidence=float(best_score),
        alternatives=[
            CropCandidate(crop=crop, confidence=float(score))
            for crop, score in ranked[1 : top_k + 1]
        ],
    )
