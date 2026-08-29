r"""Application configuration.

Every filesystem path is derived from the repository root at import time, so the
app runs from any checkout location. The 1.x code hardcoded ``E:\extra\crop``
and could only ever run on one machine.
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parents[1]


class Settings(BaseSettings):
    """Runtime settings, overridable via ``CROPCRAFT_*`` environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="CROPCRAFT_",
        env_file=".env",
        extra="ignore",
    )

    project_root: Path = PROJECT_ROOT
    debug: bool = False

    # Artifacts produced by the scripts in ml/.
    crop_model_path: Path = PROJECT_ROOT / "models" / "crop_recommender.joblib"
    pest_model_path: Path = PROJECT_ROOT / "models" / "pest_classifier.keras"
    pest_labels_path: Path = PROJECT_ROOT / "models" / "pest_labels.json"

    # Reference data.
    crop_npk_path: Path = PROJECT_ROOT / "Data" / "Crop_NPK.csv"

    # Uploads are decoded in memory and never written to disk. The 1.x code
    # wrote user-controlled filenames straight into ``static/user uploaded``
    # with no size or type limit.
    max_upload_bytes: int = 8 * 1024 * 1024
    allowed_image_types: frozenset[str] = frozenset({"image/jpeg", "image/png", "image/webp"})

    pest_image_size: int = 224


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return Settings()
