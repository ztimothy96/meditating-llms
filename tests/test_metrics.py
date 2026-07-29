import numpy as np
import pytest
import torch

from meditation.metrics import (
    band_reachability,
    score_drift,
    self_reachability,
    token_ids_for_words,
)


class _FakeTokenizer:
    """Maps each of a handful of known words to a single token id; everything
    else tokenizes to 2+ ids (so token_ids_for_words skips it)."""

    _KNOWN = {"om": 5, " om": 5, "the": 7, " the": 7}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        if text in self._KNOWN:
            return [self._KNOWN[text]]
        return [1, 2]


def test_token_ids_for_words_single_token_only():
    ids = token_ids_for_words(("om",), _FakeTokenizer())
    assert ids == [5]


def test_token_ids_for_words_skips_multi_token():
    ids = token_ids_for_words(("nonexistentword",), _FakeTokenizer())
    assert ids == []


def _logits_with_rank1_at(vocab: int, positions_and_ids: dict[int, int]) -> torch.Tensor:
    """[n_positions, vocab] logits where position p's top-1 is
    positions_and_ids[p] (if given), else uniform-ish descending by id."""
    n_positions = max(positions_and_ids) + 1
    logits = torch.arange(vocab).float().unsqueeze(0).repeat(n_positions, 1)
    for p, tid in positions_and_ids.items():
        logits[p, tid] = vocab + 1  # force rank 1
    return logits


def test_band_reachability_rank1_hit():
    logits = _logits_with_rank1_at(20, {0: 5, 1: 5})
    lens_logits = {3: logits, 4: logits}
    rank = band_reachability(lens_logits, target_ids=[5], band=[3, 4])
    assert (rank[:2] == 1).all()


def test_band_reachability_takes_min_across_band():
    # layer 3: target is buried; layer 4: target is rank 1.
    buried = torch.arange(20).float().unsqueeze(0)
    buried[0, 5] = -100  # push target to the bottom at layer 3
    top = _logits_with_rank1_at(20, {0: 5})[:1]
    rank = band_reachability({3: buried, 4: top}, target_ids=[5], band=[3, 4])
    assert rank[0] == 1


def test_band_reachability_empty_targets_raises():
    with pytest.raises(ValueError):
        band_reachability({3: torch.zeros(1, 10)}, target_ids=[], band=[3])


def test_self_reachability_tracks_generated_token():
    vocab = 10
    own_ids = [3, 7]
    logits = _logits_with_rank1_at(vocab, {0: 3, 1: 7})
    rank = self_reachability({2: logits}, own_ids, band=[2])
    assert (rank == 1).all()


def test_score_drift_flags_sustained_loss_of_reachability():
    rank = np.array([1, 1, 1] + [50] * 40)
    report = score_drift(rank, k=5, window=32, threshold=0.5)
    assert report.is_drifting
    assert report.window_fraction_unreachable == 1.0


def test_score_drift_stable_run():
    rank = np.array([1, 2, 3, 1, 2] * 10)
    report = score_drift(rank, k=5, window=32, threshold=0.5)
    assert not report.is_drifting
