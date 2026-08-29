"""Static asset paths resolve over HTTP, not just on disk."""

import json
import pathlib

import pytest

from cropcraft.domain.pests import PESTS

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_no_static_path_contains_a_space():
    """A space in an asset path is emitted unencoded by url_for.

    'static/img/pesticide/stem borer' produced src=".../stem borer/cartop.jpg",
    which is not a valid URL.
    """
    spaced = [p for p in (PROJECT_ROOT / "static").rglob("*") if " " in p.name]
    assert spaced == []


def test_every_pesticide_image_exists_on_disk():
    missing = [
        product.image
        for pest in PESTS.values()
        for product in pest.pesticides
        if not (PROJECT_ROOT / product.image).exists()
    ]
    assert missing == []


def test_every_pesticide_image_is_served(client):
    for pest in PESTS.values():
        for product in pest.pesticides:
            url = "/" + product.image.replace("\\", "/")
            assert client.get(url).status_code == 200, url


def test_every_predictable_crop_has_an_image(client):
    """crop_result.html renders static/img/crop/<crop>.jpg."""
    from cropcraft.services.crop import crop_classes

    for crop in crop_classes():
        assert client.get(f"/static/img/crop/{crop}.jpg").status_code == 200


@pytest.mark.parametrize(
    "asset",
    ["vendor/bootstrap.min.css", "vendor/htmx.min.js", "css/cropcraft.css", "js/cropcraft.js"],
)
def test_vendored_assets_are_served(client, asset):
    assert client.get(f"/static/{asset}").status_code == 200


def test_product_manifest_matches_the_domain_data():
    raw = json.loads(
        (PROJECT_ROOT / "src/cropcraft/domain/_pest_products.json").read_text(encoding="utf-8")
    )
    # earthworm is deliberately stripped of pesticides
    assert sum(len(v) for k, v in raw.items() if k != "earthworm") == sum(
        len(p.pesticides) for p in PESTS.values()
    )
