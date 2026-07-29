"""Mid-stream distractor injection: splice foreign text into an established
meditation session and measure how long the anchor takes to become
lens-reachable again (or whether it does at all)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from jlens.lens import JacobianLens

from meditation.generate import MeditationRun, _apply_lens_to_ids, _build_prompt
from meditation.metrics import band_reachability, self_reachability, token_ids_for_words


@dataclass
class DistractionResult:
    """Outcome of one injection.

    Attributes:
        injected_at: Meditation-token offset (from the start of generation,
            not the raw sequence) where the distractor was spliced in.
        distractor_text: What was injected.
        distractor_n_tokens: Length of the injected span, so callers can
            slice ``rank`` into "during distractor" vs. "after."
        rank: Anchor rank at every position from the start of the distractor
            through the end of the post-injection continuation.
        recovery_tokens: Tokens after the distractor ends until the anchor is
            reachable for a sustained ``recovery_window``, or ``None`` if it
            never recovers within the generated continuation.
    """

    injected_at: int
    distractor_text: str
    distractor_n_tokens: int
    rank: np.ndarray
    recovery_tokens: int | None


def _sustained_recovery(reachable: np.ndarray, start: int, window: int) -> int | None:
    for i in range(start, len(reachable) - window + 1):
        if reachable[i : i + window].all():
            return i - start
    return None


def inject_and_continue(
    hf_model: Any,
    tokenizer: Any,
    model: Any,
    lens: JacobianLens,
    base_run: MeditationRun,
    *,
    distractor_text: str,
    k: int = 5,
    max_tokens_after: int = 512,
    recovery_window: int = 16,
    generate_kwargs: dict | None = None,
) -> DistractionResult:
    """Splice ``distractor_text`` onto ``base_run``'s current generation and
    continue, tracking when (if ever) the anchor becomes reachable again.

    ``base_run`` must have at least one checkpoint (i.e. an established,
    non-drifting meditation session) — call :func:`meditation.generate.run_meditation`
    with ``stop_on_drift=False`` first if the anchor's own drift check would
    otherwise have cut the run short before the injection point.
    """
    input_ids = base_run.generated_ids()
    if input_ids is None:
        raise ValueError("base_run has no checkpoints to inject a distractor into")

    anchor = base_run.anchor
    prompt_ids = model.encode(_build_prompt(anchor, tokenizer))
    prompt_len = prompt_ids.shape[1]
    injected_at = input_ids.shape[1] - prompt_len

    distractor_ids = tokenizer(
        distractor_text, return_tensors="pt", add_special_tokens=False
    ).input_ids.to(input_ids.device)
    spliced = torch.cat([input_ids, distractor_ids], dim=1)

    generate_kwargs = {"do_sample": False, **(generate_kwargs or {})}
    full_ids = hf_model.generate(
        spliced,
        max_new_tokens=max_tokens_after,
        pad_token_id=getattr(tokenizer, "pad_token_id", None) or tokenizer.eos_token_id,
        **generate_kwargs,
    )

    post_start = input_ids.shape[1]  # start of the distractor span
    positions = list(range(post_start, full_ids.shape[1]))
    lens_logits = _apply_lens_to_ids(
        lens, model, full_ids, layers=base_run.band, positions=positions
    )

    if anchor.self_referential:
        own_ids = full_ids[0, post_start:].tolist()
        rank = self_reachability(lens_logits, own_ids, base_run.band)
    else:
        target_ids = token_ids_for_words(anchor.tracked_words, tokenizer)
        rank = band_reachability(lens_logits, target_ids, base_run.band)

    reachable = rank <= k
    distractor_n_tokens = distractor_ids.shape[1]
    recovery_tokens = _sustained_recovery(reachable, distractor_n_tokens, recovery_window)

    return DistractionResult(
        injected_at=injected_at,
        distractor_text=distractor_text,
        distractor_n_tokens=distractor_n_tokens,
        rank=rank,
        recovery_tokens=recovery_tokens,
    )
