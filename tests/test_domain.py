"""Reference data consistency."""

import json

from cropcraft.config import get_settings
from cropcraft.domain.pests import PESTS, get_pest


def test_every_model_label_has_reference_data():
    """1.x resolved labels by concatenating '<label>.html' -- a missing template
    was a 500. The label list and the data must agree."""
    labels_path = get_settings().pest_labels_path
    if not labels_path.exists():
        import pytest

        pytest.skip("pest model not built")

    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    missing = [label for label in labels if get_pest(label) is None]
    assert missing == []


def test_lookup_normalises_spacing():
    """The 1.x label was 'stem borer'; the dataset directory is 'stem_borer'."""
    assert get_pest("stem borer") is get_pest("stem_borer")
    assert get_pest("Stem Borer") is get_pest("stem_borer")


def test_unknown_label_returns_none():
    assert get_pest("nonexistent") is None


def test_earthworm_is_marked_beneficial_with_no_pesticides():
    """1.x recommended malathion for earthworms, which improve soil health."""
    earthworm = PESTS["earthworm"]
    assert earthworm.beneficial
    assert earthworm.pesticides == ()


def test_other_pests_are_not_marked_beneficial():
    for slug, pest in PESTS.items():
        if slug != "earthworm":
            assert not pest.beneficial


def test_every_pest_has_guidance():
    for pest in PESTS.values():
        assert pest.description and pest.damage
        assert pest.cultural_controls
