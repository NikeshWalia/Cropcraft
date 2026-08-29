r"""Application configuration.

Every filesystem path is derived from the repository root at import time, so the
app runs from any checkout location. The 1.x code hardcoded ``E:\extra\crop``
and could only ever run on one machine.
"""

import os
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PACKAGE_ROOT = Path(__file__).resolve().parent


def _resolve_project_root() -> Path:
    """Locate the directory holding templates/, static/, Data/ and models/.

    Walking up from ``__file__`` only works while the package sits in ``src/``.
    Once it is installed into ``site-packages`` -- which is what happens inside
    the container image -- that walk lands in the virtualenv and every asset
    path is wrong. So: an explicit override wins, then the source layout, then
    the working directory, which is where a deployment keeps its assets.
    """
    override = os.environ.get("CROPCRAFT_PROJECT_ROOT")
    if override:
        return Path(override).resolve()

    source_checkout = PACKAGE_ROOT.parents[1]
    if (source_checkout / "templates").is_dir():
        return source_checkout

    return Path.cwd()


PROJECT_ROOT = _resolve_project_root()


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
