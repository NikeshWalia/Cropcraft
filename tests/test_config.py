"""Settings resolve from the repository root and respond to environment overrides.

CropCraft 1.x hardcoded ``E:\extra\crop`` in four places, so it could only run
on one machine.
"""

import pathlib

from cropcraft.config import PROJECT_ROOT, Settings, get_settings


def test_paths_are_derived_from_the_repository_root():
    settings = get_settings()
    for path in (settings.crop_model_path, settings.pest_model_path, settings.crop_npk_path):
        assert path.is_absolute()
        assert PROJECT_ROOT in path.parents, f"{path} escapes the project root"


def test_project_root_is_the_checkout():
    assert (PROJECT_ROOT / "pyproject.toml").exists()
    assert pathlib.Path(__file__).resolve().parents[1] == PROJECT_ROOT


def test_environment_overrides_apply(monkeypatch):
    """Documented in .env.example, so it needs to actually work."""
    monkeypatch.setenv("CROPCRAFT_MAX_UPLOAD_BYTES", "12345")
    monkeypatch.setenv("CROPCRAFT_PEST_IMAGE_SIZE", "99")
    monkeypatch.setenv("CROPCRAFT_DEBUG", "true")

    settings = Settings()
    assert settings.max_upload_bytes == 12345
    assert settings.pest_image_size == 99
    assert settings.debug is True


def test_defaults_are_sane():
    settings = Settings()
    assert settings.debug is False, "debug must never default on"
    assert settings.max_upload_bytes == 8 * 1024 * 1024
    assert settings.pest_image_size == 224
    assert "image/jpeg" in settings.allowed_image_types


def test_env_example_documents_every_overridable_setting():
    """Keeps .env.example from drifting as settings are added."""
    text = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")
    for field in Settings.model_fields:
        if field == "project_root":
            continue
        assert f"CROPCRAFT_{field.upper()}" in text, f"{field} is undocumented"


def test_missing_models_degrade_gracefully(monkeypatch, tmp_path):
    """A checkout that has not run the training scripts must not crash.

    1.x called load_model at import time against a hardcoded path, so a missing
    artifact was an immediate traceback on startup.
    """
    from fastapi.testclient import TestClient

    monkeypatch.setenv("CROPCRAFT_CROP_MODEL_PATH", str(tmp_path / "absent.joblib"))
    monkeypatch.setenv("CROPCRAFT_PEST_MODEL_PATH", str(tmp_path / "absent.keras"))
    get_settings.cache_clear()

    try:
        from cropcraft.main import create_app

        with TestClient(create_app()) as client:
            health = client.get("/healthz").json()
            assert health["status"] == "degraded"
            assert "train_crop" in health["detail"]

            # Pages still render, and predictions fail with a clear 503.
            assert client.get("/").status_code == 200
            response = client.post(
                "/api/crop",
                json={
                    "nitrogen": 90,
                    "phosphorous": 42,
                    "potassium": 43,
                    "temperature": 21,
                    "humidity": 82,
                    "ph": 6.5,
                    "rainfall": 203,
                },
            )
            assert response.status_code == 503
            assert "train_crop" in response.json()["detail"]
    finally:
        get_settings.cache_clear()


def test_project_root_honours_an_explicit_override(tmp_path, monkeypatch):
    """Deployments that do not match the source layout can point at their assets."""
    from cropcraft.config import _resolve_project_root

    (tmp_path / "templates").mkdir()
    monkeypatch.setenv("CROPCRAFT_PROJECT_ROOT", str(tmp_path))
    assert _resolve_project_root() == tmp_path.resolve()


def test_project_root_falls_back_to_cwd_when_installed(tmp_path, monkeypatch):
    """Once the package is installed into site-packages, walking up from
    __file__ lands inside the virtualenv, so assets must come from the working
    directory instead. This is exactly the container layout."""
    import cropcraft.config as config

    monkeypatch.delenv("CROPCRAFT_PROJECT_ROOT", raising=False)
    # Pretend the package lives somewhere with no sibling templates/ directory.
    fake_site_packages = tmp_path / "venv" / "lib" / "site-packages" / "cropcraft"
    fake_site_packages.mkdir(parents=True)
    monkeypatch.setattr(config, "PACKAGE_ROOT", fake_site_packages)

    app_dir = tmp_path / "app"
    (app_dir / "templates").mkdir(parents=True)
    monkeypatch.chdir(app_dir)

    assert config._resolve_project_root() == app_dir
