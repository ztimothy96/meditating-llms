"""Drift scoring on top of Jacobian-lens readouts.

Reuses the paper's "reachability" notion (see `dual-task` /
`capacity` in jacobian-lens-main/data/experiments/README.md): a target is
*reachable* at a position if it sits at lens rank <= k at any layer in the
mid-network "workspace band". A meditation anchor is "held" while its
tracked tokens stay reachable; "discursive thought" is a run of positions
where they stop being reachable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


def token_ids_for_words(words: tuple[str, ...], tokenizer) -> list[int]:
    """Single-token vocabulary ids for ``words``.

    Multi-token surface forms are skipped (with the token id list simply
    omitting them) since rank-tracking is defined per vocabulary entry;
    callers should pick anchor wording that tokenizes to single tokens for
    the model in use.
    """
    ids: list[int] = []
    for word in words:
        for variant in (word, f" {word}", word.capitalize(), f" {word.capitalize()}"):
            encoded = tokenizer.encode(variant, add_special_tokens=False)
            if len(encoded) == 1:
                ids.append(encoded[0])
    return sorted(set(ids))


def _ranks_of(logits: torch.Tensor, target_ids: list[int]) -> torch.Tensor:
    """Rank (1 = top) of each id in ``target_ids`` at every row of ``logits``.

    ``logits``: ``[n_positions, vocab]``. Returns ``[n_positions,
    len(target_ids)]``.
    """
    target = logits[:, target_ids]  # [n_positions, n_targets]
    return (logits.unsqueeze(-1) > target.unsqueeze(1)).sum(dim=1) + 1


def band_reachability(
    lens_logits: dict[int, torch.Tensor],
    target_ids: list[int],
    band: list[int],
) -> np.ndarray:
    """Per-position best (min) rank of any ``target_ids`` token, over ``band``
    layers. Returns a ``[n_positions]`` int array; threshold the result with
    :func:`score_drift`'s ``k`` for hits.

    Raises:
        ValueError: If ``target_ids`` is empty (the anchor's tracked words
            didn't tokenize to single tokens for this model) or ``band`` is
            not a subset of ``lens_logits``.
    """
    if not target_ids:
        raise ValueError("target_ids is empty; check anchor.tracked_words tokenize")
    missing = [layer for layer in band if layer not in lens_logits]
    if missing:
        raise ValueError(f"band layers {missing} were not read out by the lens")
    n_positions = next(iter(lens_logits.values())).shape[0]
    best = np.full(n_positions, np.iinfo(np.int64).max, dtype=np.int64)
    for layer in band:
        ranks = _ranks_of(lens_logits[layer], target_ids).min(dim=1).values.numpy()
        best = np.minimum(best, ranks)
    return best


def self_reachability(
    lens_logits: dict[int, torch.Tensor],
    own_token_ids: list[int],
    band: list[int],
) -> np.ndarray:
    """Like :func:`band_reachability`, but the target at position ``p`` is
    ``own_token_ids[p]`` itself (the "current token" protocol) rather than a
    fixed word list."""
    missing = [layer for layer in band if layer not in lens_logits]
    if missing:
        raise ValueError(f"band layers {missing} were not read out by the lens")
    n_positions = len(own_token_ids)
    best = np.full(n_positions, np.iinfo(np.int64).max, dtype=np.int64)
    for layer in band:
        logits = lens_logits[layer]
        target = logits[torch.arange(n_positions), own_token_ids]
        ranks = (logits > target.unsqueeze(1)).sum(dim=1).numpy() + 1
        best = np.minimum(best, ranks)
    return best


@dataclass
class DriftReport:
    """Result of :func:`score_drift` over one checkpoint's worth of positions.

    Attributes:
        rank: Per-position best rank of the anchor (see :func:`band_reachability`).
        reachable: ``rank <= k`` per position.
        window_fraction_unreachable: Fraction of the trailing ``window``
            positions where the anchor was not reachable.
        is_drifting: ``window_fraction_unreachable > threshold``.
    """

    rank: np.ndarray
    reachable: np.ndarray
    window_fraction_unreachable: float
    is_drifting: bool


def score_drift(
    rank: np.ndarray,
    *,
    k: int = 5,
    window: int = 32,
    threshold: float = 0.5,
) -> DriftReport:
    """Summarize a rank series into a drift verdict over its trailing ``window``."""
    reachable = rank <= k
    tail = reachable[-window:]
    fraction_unreachable = 1.0 - float(tail.mean()) if len(tail) else 0.0
    return DriftReport(
        rank=rank,
        reachable=reachable,
        window_fraction_unreachable=fraction_unreachable,
        is_drifting=fraction_unreachable > threshold,
    )
