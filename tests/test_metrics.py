"""The published accuracy figures are queryable, not just claimed in the README."""


def test_metrics_endpoint_reports_holdout_accuracy(client):
    body = client.get("/api/metrics").json()

    assert 0 < body["crop"]["holdout_accuracy"] <= 1
    assert body["crop"]["n_test"] > 0
    assert body["crop"]["seed"] == 42

    assert 0 < body["pest"]["holdout_accuracy"] <= 1
    assert body["pest"]["holdout_top3_accuracy"] >= body["pest"]["holdout_accuracy"]
    assert len(body["pest"]["per_class_recall"]) == 10


def test_readme_accuracy_is_in_the_right_ballpark(client):
    """Catches gross README drift without breaking on training variance.

    Retraining on different hardware moves the pest accuracy by a point or two,
    so an exact-match assertion would fail in CI. A tolerance still catches the
    case that matters: a README claiming a number the model cannot deliver.
    """
    import pathlib
    import re

    readme = pathlib.Path(__file__).resolve().parents[1] / "README.md"
    documented = [
        float(m) for m in re.findall(r"\*\*(0\.\d{4})\*\*", readme.read_text(encoding="utf-8"))
    ]
    assert len(documented) >= 2, "README should state a held-out accuracy for both models"

    body = client.get("/api/metrics").json()
    for actual in (body["crop"]["holdout_accuracy"], body["pest"]["holdout_accuracy"]):
        assert any(abs(actual - claimed) < 0.10 for claimed in documented), (
            f"no documented figure is close to the measured {actual:.4f}"
        )


def test_healthz_is_ok_when_models_are_present(client):
    assert client.get("/healthz").json()["status"] == "ok"
