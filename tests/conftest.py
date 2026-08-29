"""Shared test fixtures."""

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from cropcraft.main import create_app


@pytest.fixture(scope="session")
def client() -> TestClient:
    """A test client against the real application."""
    with TestClient(create_app()) as test_client:
        yield test_client


@pytest.fixture
def jpeg_bytes() -> bytes:
    """A small valid JPEG."""
    buffer = io.BytesIO()
    Image.new("RGB", (240, 240), (90, 130, 70)).save(buffer, format="JPEG")
    return buffer.getvalue()


@pytest.fixture
def valid_soil() -> dict[str, float]:
    """Soil readings inside the training ranges."""
    return {
        "nitrogen": 90,
        "phosphorous": 42,
        "potassium": 43,
        "temperature": 20.9,
        "humidity": 82.0,
        "ph": 6.5,
        "rainfall": 202.9,
    }
