import pytest

from meditation.anchors import ANCHORS, CONTROLS, Anchor


def test_all_anchors_valid():
    for anchor in [*ANCHORS, *CONTROLS]:
        assert anchor.self_referential or anchor.tracked_words or anchor.is_control


def test_non_control_non_self_referential_needs_tracked_words():
    with pytest.raises(ValueError):
        Anchor(slug="bad", description="", user="do something")


def test_slugs_unique():
    slugs = [a.slug for a in [*ANCHORS, *CONTROLS]]
    assert len(slugs) == len(set(slugs))
