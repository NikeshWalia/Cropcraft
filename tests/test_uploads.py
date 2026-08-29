r"""Upload handling.

CropCraft 1.x did::

    file_path = os.path.join('static/user uploaded', file.filename)
    file.save(file_path)

with no filename sanitisation, no type check and no size limit. These tests
cover the replacement.
"""

import io
import pathlib

import pytest
from PIL import Image

from cropcraft.config import get_settings


def test_rejects_non_image_content_type(client):
    response = client.post(
        "/api/pesticide",
        files={"image": ("payload.txt", b"not an image", "text/plain")},
    )
    assert response.status_code == 415


def test_rejects_oversized_upload(client):
    limit = get_settings().max_upload_bytes
    oversized = b"\xff\xd8\xff" + b"\x00" * (limit + 1024)
    response = client.post(
        "/api/pesticide",
        files={"image": ("huge.jpg", oversized, "image/jpeg")},
    )
    assert response.status_code == 413


def test_rejects_bytes_that_are_not_decodable(client):
    """A correct content-type header does not make the payload an image."""
    response = client.post(
        "/api/pesticide",
        files={"image": ("fake.jpg", b"\x00" * 4096, "image/jpeg")},
    )
    assert response.status_code in {400, 503}


def test_rejects_empty_upload(client):
    response = client.post("/api/pesticide", files={"image": ("empty.jpg", b"", "image/jpeg")})
    assert response.status_code == 400


@pytest.mark.parametrize(
    "filename",
    [
        "../../app.py",
        r"..\..\pyproject.toml",
        "....//....//etc/passwd",
        "pest.jpg\x00.py",
    ],
)
def test_traversal_filenames_write_nothing(client, jpeg_bytes, filename):
    """The filename is never used for anything, so traversal has no target.

    Watches the directories a '../' escape from the 1.x upload path would
    actually land in.
    """
    root = get_settings().project_root
    watched = [root, root / "src" / "cropcraft", root / "static", root / "templates"]

    def snapshot() -> dict[pathlib.Path, float]:
        return {
            item: item.stat().st_mtime
            for directory in watched
            if directory.is_dir()
            for item in directory.iterdir()
            if item.is_file()
        }

    before = snapshot()
    client.post("/api/pesticide", files={"image": (filename, jpeg_bytes, "image/jpeg")})
    assert snapshot() == before, "an upload created or modified a file on disk"


def test_upload_is_never_persisted(client, jpeg_bytes):
    """1.x kept every upload forever in 'static/user uploaded'.

    Uploads are decoded in memory now, so no upload directory should exist at
    all -- neither the old 1.x location nor a new one.
    """
    root = get_settings().project_root
    client.post("/api/pesticide", files={"image": ("pest.jpg", jpeg_bytes, "image/jpeg")})

    for candidate in (root / "static" / "user uploaded", root / "uploads"):
        assert not candidate.exists(), f"{candidate} was created"


def test_decoder_returns_unscaled_pixels(jpeg_bytes):
    """Regression test for the 1.x train/serve normalisation mismatch.

    Training rescaled by 1/255 and inference did not. The model now carries a
    normalisation layer, so this decoder must hand it raw 0-255 values.
    """
    from cropcraft.services.pest import decode_image

    array = decode_image(jpeg_bytes)
    size = get_settings().pest_image_size
    assert array.shape == (1, size, size, 3)
    assert array.max() > 1.5, "pixels look pre-scaled; the model expects 0-255"


def test_decoder_rejects_a_truncated_image():
    from cropcraft.services.pest import InvalidImageError, decode_image

    buffer = io.BytesIO()
    Image.new("RGB", (64, 64)).save(buffer, format="PNG")
    with pytest.raises(InvalidImageError):
        decode_image(buffer.getvalue()[:20])
