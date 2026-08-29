"""The HTML pages render and are wired to each other."""

import pytest


@pytest.mark.parametrize("path", ["/", "/crop", "/fertilizer", "/pesticide"])
def test_page_renders(client, path):
    response = client.get(path)
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_healthz(client):
    assert client.get("/healthz").json() == {"status": "ok"}


def test_no_jquery_or_bootstrap3(client):
    """1.x shipped jQuery 1.11.2 and Bootstrap 3.3.5, both long EOL."""
    body = client.get("/").text
    assert "jquery" not in body.lower()
    assert "bootstrap.min.css" in body


def test_no_external_assets(client):
    """1.x pulled Google Fonts over plain http, blocked as mixed content on https.

    Every asset is now vendored under /static, so the page has no third-party
    origins at all.
    """
    import re

    body = client.get("/").text
    external = [
        url
        for url in re.findall(r'(?:href|src)="(https?://[^"]+)"', body)
        if not url.startswith(str(client.base_url))
    ]
    assert external == []


def test_fertilizer_dropdown_is_data_driven(client):
    """The crop list comes from Crop_NPK.csv, not a hardcoded template list."""
    from cropcraft.services.fertilizer import supported_crops

    body = client.get("/fertilizer").text
    for crop in supported_crops():
        assert f'value="{crop}"' in body
