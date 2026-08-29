r"""Pest identification service.

Upload handling here is deliberately different from CropCraft 1.x, which did::

    file_path = os.path.join('static/user uploaded', file.filename)
    file.save(file_path)

That trusted a user-controlled filename, so ``..\..\app.py`` escaped the
upload directory; it enforced no size or type limit; and it left every upload
on disk forever. This version decodes the image in memory and never writes it
anywhere.
"""

from __future__ import annotations

import io
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError

from cropcraft.config import get_settings
from cropcraft.domain.pests import get_pest
from cropcraft.schemas import PesticideOut, PestResponse

# Refuse absurd pixel counts before allocating for them.
Image.MAX_IMAGE_PIXELS = 64_000_000

CONFIDENCE_THRESHOLD = 0.45


class ModelUnavailableError(RuntimeError):
    """Raised when the classifier artifact has not been built yet."""


class InvalidImageError(ValueError):
    """Raised when an upload is not a decodable image."""


@lru_cache
def _load_model() -> Any:
    """Load the Keras classifier once per process.

    TensorFlow is imported lazily so that the crop and fertilizer endpoints,
    and the whole test suite, do not pay a multi-second import they don't use.
    """
    settings = get_settings()
    if not settings.pest_model_path.exists():
        raise ModelUnavailableError(
            f"{settings.pest_model_path.name} is missing. Build it with: "
            "python -m ml.prepare_dataset && python -m ml.train_pest"
        )
    import keras  # noqa: PLC0415 -- deliberately deferred

    return keras.models.load_model(settings.pest_model_path)


@lru_cache
def pest_labels() -> list[str]:
    """Class labels in the order the model emits them."""
    path = get_settings().pest_labels_path
    if not path.exists():
        raise ModelUnavailableError(
            f"{path.name} is missing. Build it with: python -m ml.train_pest"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def metrics() -> dict[str, Any]:
    """Training metrics, or an empty dict if the report is absent."""
    path = Path(get_settings().pest_model_path).with_name("pest_metrics.json")
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def decode_image(payload: bytes) -> np.ndarray:
    """Decode bytes into the raw 0-255 RGB array the model expects.

    The model carries its own normalisation layer, so this returns unscaled
    pixels. In 1.x the training pipeline divided by 255 and the inference path
    did not, so the deployed model saw inputs on a scale it never saw in
    training.
    """
    size = get_settings().pest_image_size
    try:
        with Image.open(io.BytesIO(payload)) as image:
            image.load()  # force decode inside the try, so truncation is caught
            rgb = image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
            return np.asarray(rgb, dtype=np.float32)[np.newaxis, ...]
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise InvalidImageError("That file could not be read as an image.") from exc


def identify(payload: bytes) -> PestResponse:
    """Identify the pest in an uploaded image."""
    array = decode_image(payload)
    probabilities = np.asarray(_load_model().predict(array, verbose=0))[0]
    labels = pest_labels()

    order = probabilities.argsort()[::-1]
    top, second = order[0], order[1] if len(order) > 1 else order[0]
    pest = get_pest(labels[top])
    if pest is None:  # pragma: no cover -- only if labels and data disagree
        raise ModelUnavailableError(f"No reference data for predicted label {labels[top]!r}")

    runner_up = get_pest(labels[second])
    confidence = float(probabilities[top])

    return PestResponse(
        pest=pest.slug,
        name=pest.name,
        confidence=confidence,
        confident=confidence >= CONFIDENCE_THRESHOLD,
        beneficial=pest.beneficial,
        description=pest.description,
        damage=pest.damage,
        cultural_controls=list(pest.cultural_controls),
        pesticides=[PesticideOut(name=p.name, dose=p.dose, image=p.image) for p in pest.pesticides],
        runner_up=runner_up.name if runner_up else None,
        runner_up_confidence=float(probabilities[second]),
    )
