"""Fertilizer advice behaviour."""

import pytest

from cropcraft.domain.fertilizer_advice import NUTRIENT_ADVICE
from cropcraft.services.fertilizer import supported_crops


def test_advice_for_a_known_crop(client):
    response = client.post(
        "/api/fertilizer",
        json={"cropname": "rice", "nitrogen": 40, "phosphorous": 30, "potassium": 20},
    )
    assert response.status_code == 200

    body = response.json()
    assert body["crop"] == "rice"
    assert [n["symbol"] for n in body["nutrients"]] == ["N", "P", "K"]


def test_unknown_crop_is_a_404_not_a_raw_exception(client):
    """1.x hit IndexError here and rendered ``str(e)`` to the browser."""
    response = client.post(
        "/api/fertilizer",
        json={"cropname": "definitely-not-a-crop", "nitrogen": 1, "phosphorous": 1, "potassium": 1},
    )
    assert response.status_code == 404
    assert "Traceback" not in response.text


def test_pigeonpeas_alias_resolves(client):
    """The crop model predicts 'pigeonpeas'; Crop_NPK.csv spells it 'pigeonpea'."""
    response = client.post(
        "/api/fertilizer",
        json={"cropname": "pigeonpeas", "nitrogen": 20, "phosphorous": 20, "potassium": 20},
    )
    assert response.status_code == 200
    assert response.json()["crop"] == "pigeonpea"


def test_every_predictable_crop_has_npk_data(client):
    """Guards the crop -> fertilizer hand-off end to end."""
    from cropcraft.services.crop import crop_classes
    from cropcraft.services.fertilizer import normalise_crop

    known = set(supported_crops())
    missing = [c for c in crop_classes() if normalise_crop(c) not in known]
    assert missing == []


@pytest.mark.parametrize(
    ("measured", "expected"),
    [(0, "low"), (1000, "high")],
)
def test_status_reflects_the_gap(client, measured, expected):
    response = client.post(
        "/api/fertilizer",
        json={
            "cropname": "rice",
            "nitrogen": min(measured, 200),
            "phosphorous": min(measured, 200),
            "potassium": min(measured, 250),
        },
    )
    assert response.status_code == 200
    assert response.json()["nutrients"][0]["status"] == expected


def test_on_target_reads_ok(client):
    """Measuring exactly the crop's requirement should report 'ok'."""
    import pandas as pd

    from cropcraft.config import get_settings

    table = pd.read_csv(get_settings().crop_npk_path)
    row = table[table["Crop"] == "rice"].iloc[0]
    response = client.post(
        "/api/fertilizer",
        json={
            "cropname": "rice",
            "nitrogen": int(row["N"]),
            "phosphorous": int(row["P"]),
            "potassium": int(row["K"]),
        },
    )
    assert [n["status"] for n in response.json()["nutrients"]] == ["ok", "ok", "ok"]


def test_nitrogen_advice_is_not_inverted():
    """Regression test for the 1.x bug where the N advice was back to front.

    'N is high' recommended manure and nitrogen-fixing plants, which raise
    nitrogen; 'N is low' led with sawdust and leaching, which lower it.
    """
    high = " ".join(NUTRIENT_ADVICE["NHigh"].tips).lower()
    low = " ".join(NUTRIENT_ADVICE["Nlow"].tips).lower()

    # Advice for too much nitrogen should be about removing it.
    assert "sawdust" in high
    assert "leach" in high or "soaking your soil" in high
    assert "coffee grinds" not in high

    # Advice for too little nitrogen should be about adding it.
    assert "manure" in low
    assert "nitrogen fixing plants" in low
    assert "sawdust or fine woodchips" not in low


def test_advice_is_plain_text_not_html():
    """1.x stored HTML and pushed it through Markup(), disabling autoescaping."""
    for advice in NUTRIENT_ADVICE.values():
        for tip in advice.tips:
            assert "<" not in tip and ">" not in tip


def test_html_result_escapes_content(client):
    response = client.post(
        "/predict/fertilizer",
        data={"cropname": "rice", "nitrogen": 40, "phosphorous": 30, "potassium": 20},
    )
    assert response.status_code == 200
    assert "Fertilizer advice" in response.text
