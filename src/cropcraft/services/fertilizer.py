"""Fertilizer recommendation service."""

from __future__ import annotations

from functools import lru_cache

import pandas as pd

from cropcraft.config import get_settings
from cropcraft.domain.fertilizer_advice import NUTRIENT_ADVICE
from cropcraft.schemas import FertilizerRequest, FertilizerResponse, NutrientResult

# The crop classifier and Crop_NPK.csv disagree on two spellings. 1.x hit an
# IndexError here and rendered the exception text to the user; mapping the
# names keeps the crop -> fertilizer hand-off working.
CROP_ALIASES = {"pigeonpeas": "pigeonpea"}

NUTRIENTS = (
    ("Nitrogen", "N", "nitrogen"),
    ("Phosphorus", "P", "phosphorous"),
    ("Potassium", "K", "potassium"),
)


class UnknownCropError(ValueError):
    """Raised when a crop has no entry in the NPK reference table."""


@lru_cache
def _npk_table() -> pd.DataFrame:
    """Load the per-crop NPK requirements, indexed by lowercased crop name."""
    frame = pd.read_csv(get_settings().crop_npk_path)
    frame["Crop"] = frame["Crop"].str.strip().str.lower()
    return frame.set_index("Crop")


def supported_crops() -> list[str]:
    """Crops with a published NPK requirement, for the form dropdown.

    Driven from the CSV rather than hardcoded in the template, so the two
    cannot drift apart the way they did in 1.x.
    """
    return sorted(_npk_table().index)


def normalise_crop(name: str) -> str:
    """Map a crop name onto its NPK-table spelling."""
    key = name.strip().lower()
    return CROP_ALIASES.get(key, key)


def recommend(request: FertilizerRequest) -> FertilizerResponse:
    """Compare measured NPK against the crop's requirement and advise."""
    crop = normalise_crop(request.cropname)
    table = _npk_table()
    if crop not in table.index:
        raise UnknownCropError(f"No NPK reference data for crop {request.cropname!r}")

    row = table.loc[crop]
    measurements = request.model_dump()

    results: list[NutrientResult] = []
    for label, symbol, field in NUTRIENTS:
        measured = int(measurements[field])
        desired = int(row[symbol])
        if measured == desired:
            status, suffix = "ok", "No"
        elif measured > desired:
            status, suffix = "high", "High"
        else:
            status, suffix = "low", "low"

        advice = NUTRIENT_ADVICE[f"{symbol}{suffix}"]
        results.append(
            NutrientResult(
                nutrient=label,
                symbol=symbol,
                measured=measured,
                desired=desired,
                difference=abs(desired - measured),
                status=status,
                headline=advice.headline,
                tips=list(advice.tips),
            )
        )

    return FertilizerResponse(crop=crop, nutrients=results)
