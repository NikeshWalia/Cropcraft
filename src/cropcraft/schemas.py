"""Request and response models.

CropCraft 1.x read form fields with bare ``int(request.form['nitrogen'])``
calls, so any non-numeric input raised ``ValueError`` and returned a 500 with a
stack trace. Validation lives here instead, and FastAPI rejects bad input with
a 422 before any handler runs.

The bounds match the ranges present in ``Data/crop_recommendation.csv``, which
is what the model was actually trained on.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, Field

Nitrogen = Annotated[int, Field(ge=0, le=200, description="Soil nitrogen (kg/ha)")]
Phosphorus = Annotated[int, Field(ge=0, le=200, description="Soil phosphorus (kg/ha)")]
Potassium = Annotated[int, Field(ge=0, le=250, description="Soil potassium (kg/ha)")]


class CropRequest(BaseModel):
    """Soil and weather readings for a crop recommendation."""

    nitrogen: Nitrogen
    phosphorous: Phosphorus
    potassium: Potassium
    temperature: Annotated[float, Field(ge=-10, le=60, description="Mean temperature (C)")]
    humidity: Annotated[float, Field(ge=0, le=100, description="Relative humidity (%)")]
    ph: Annotated[float, Field(ge=0, le=14, description="Soil pH")]
    rainfall: Annotated[float, Field(ge=0, le=1000, description="Rainfall (mm)")]


class CropCandidate(BaseModel):
    """One crop and the model's confidence in it."""

    crop: str
    confidence: float = Field(ge=0, le=1)


class CropResponse(BaseModel):
    """Recommended crop plus the runners-up."""

    crop: str
    confidence: float = Field(ge=0, le=1)
    alternatives: list[CropCandidate]


class FertilizerRequest(BaseModel):
    """Measured NPK for a chosen crop."""

    cropname: str = Field(min_length=1, max_length=64)
    nitrogen: Nitrogen
    phosphorous: Phosphorus
    potassium: Potassium


class NutrientResult(BaseModel):
    """How one nutrient compares to the crop's requirement."""

    nutrient: Literal["Nitrogen", "Phosphorus", "Potassium"]
    symbol: Literal["N", "P", "K"]
    measured: int
    desired: int
    difference: int
    status: Literal["high", "low", "ok"]
    headline: str
    tips: list[str]


class FertilizerResponse(BaseModel):
    """Per-nutrient advice for the chosen crop."""

    crop: str
    nutrients: list[NutrientResult]


class PesticideOut(BaseModel):
    """A product recommendation."""

    name: str
    dose: str
    image: str


class PestResponse(BaseModel):
    """Classifier output for an uploaded image."""

    pest: str
    name: str
    confidence: float = Field(ge=0, le=1)
    confident: bool
    beneficial: bool
    description: str
    damage: str
    cultural_controls: list[str]
    pesticides: list[PesticideOut]
    runner_up: str | None = None
    runner_up_confidence: float = 0.0
