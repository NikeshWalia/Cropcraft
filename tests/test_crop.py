"""Crop recommendation behaviour."""

import pytest


def test_api_recommends_a_known_crop(client, valid_soil):
    response = client.post("/api/crop", json=valid_soil)
    assert response.status_code == 200

    body = response.json()
    from cropcraft.services.crop import crop_classes

    assert body["crop"] in crop_classes()
    assert 0 <= body["confidence"] <= 1
    assert len(body["alternatives"]) == 3


def test_rice_conditions_recommend_rice(client, valid_soil):
    """A row taken straight from the training distribution should classify as rice."""
    assert client.post("/api/crop", json=valid_soil).json()["crop"] == "rice"


def test_html_form_renders_result(client, valid_soil):
    response = client.post("/predict/crop", data=valid_soil)
    assert response.status_code == 200
    assert "Recommended crop" in response.text


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("ph", 99),  # outside 0-14
        ("humidity", -5),  # negative
        ("nitrogen", 5000),  # far outside training range
        ("temperature", 500),
    ],
)
def test_out_of_range_values_are_rejected(client, valid_soil, field, value):
    """1.x passed these straight to the model and extrapolated silently."""
    payload = {**valid_soil, field: value}
    assert client.post("/api/crop", json=payload).status_code == 422


def test_non_numeric_input_is_a_422_not_a_500(client, valid_soil):
    """1.x raised ValueError inside the handler and returned a stack trace."""
    payload = {**valid_soil, "nitrogen": "not-a-number"}
    assert client.post("/api/crop", json=payload).status_code == 422


def test_html_form_shows_a_friendly_error(client, valid_soil):
    response = client.post("/predict/crop", data={**valid_soil, "ph": 99})
    assert response.status_code == 422
    assert "Check your inputs" in response.text


def test_feature_order_comes_from_the_saved_bundle():
    """Guards against the scaler being fed columns in the wrong order."""
    import joblib

    from cropcraft.config import get_settings

    bundle = joblib.load(get_settings().crop_model_path)
    assert bundle["features"] == ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]


def test_pipeline_includes_a_scaler():
    """1.x fed unscaled features to SVC and KNN."""
    import joblib

    from cropcraft.config import get_settings

    model = joblib.load(get_settings().crop_model_path)["model"]
    assert "scale" in model.named_steps
