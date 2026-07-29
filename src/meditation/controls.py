"""Baselines for ruling out the 'boring explanation': that drift is generic
loop collapse / high-frequency-token drift rather than anything
anchor-specific. See the open question in the top-level README.

Two comparisons, run with the same harness as a real meditation session:

- **Framing control** (:data:`meditation.anchors.DEGENERATE_REPEAT`): same
  repeated surface token as :data:`meditation.anchors.REPEATED_TOKEN`, no
  meditative instruction. If drift onset and attractor content look the same
  with and without the framing, the framing isn't doing anything.
- **Frequency control** (:data:`meditation.anchors.DEGENERATE_HIGH_FREQUENCY`):
  repeats a generic high-frequency word instead of a mantra token. If every
  anchor's drift converges on the same attractor content regardless of what
  it started from, that content is likely a generic corpus attractor, not
  something anchor-specific.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jlens.lens import JacobianLens

from meditation.anchors import Anchor
from meditation.generate import MeditationRun, run_meditation

_WORD_RE = re.compile(r"[a-zA-Z']+")


def _tail_words(run: MeditationRun, *, tail_tokens: int = 64) -> set[str]:
    """Words from roughly the last ``tail_tokens`` of the run's final
    checkpoint text — a crude proxy for 'what it drifted into.' Word-level,
    not token-level, so this only approximates the tail window."""
    if not run.checkpoints:
        return set()
    text = run.checkpoints[-1].text
    words = _WORD_RE.findall(text.lower())
    return set(words[-tail_tokens:])


@dataclass
class ControlComparison:
    """Result of running a real anchor against a matched control.

    Attributes:
        verdict: Human-readable summary; not a statistical test, just a
            same-harness sanity comparison to read alongside the raw runs.
    """

    anchor_run: MeditationRun
    control_run: MeditationRun
    attractor_jaccard: float
    verdict: str


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def run_matched_pair(
    hf_model: Any,
    tokenizer: Any,
    model: Any,
    lens: JacobianLens,
    anchor: Anchor,
    control: Anchor,
    *,
    run_root: Path,
    **run_kwargs: Any,
) -> ControlComparison:
    """Run ``anchor`` and its matched ``control`` under identical settings
    and compare drift onset + attractor word overlap."""
    run_root = Path(run_root)
    anchor_run = run_meditation(
        hf_model, tokenizer, model, lens, anchor,
        run_dir=run_root / anchor.slug, **run_kwargs,
    )
    control_run = run_meditation(
        hf_model, tokenizer, model, lens, control,
        run_dir=run_root / control.slug, **run_kwargs,
    )

    jaccard = _jaccard(_tail_words(anchor_run), _tail_words(control_run))

    if anchor_run.drift_onset is None and control_run.drift_onset is None:
        verdict = "neither drifted within max_tokens; no signal either way"
    elif anchor_run.drift_onset is not None and control_run.drift_onset is None:
        verdict = "anchor drifted but its unframed control didn't — framing hurt stability (surprising)"
    elif anchor_run.drift_onset is None and control_run.drift_onset is not None:
        verdict = "control drifted but the framed anchor didn't — meditative framing improved stability"
    elif abs(anchor_run.drift_onset - control_run.drift_onset) <= max(
        anchor_run.drift_onset, control_run.drift_onset
    ) * 0.25:
        verdict = (
            f"drift onset similar with/without framing "
            f"({anchor_run.drift_onset} vs {control_run.drift_onset} tokens) — "
            "consistent with generic repetition collapse, not framing-driven attachment"
        )
    else:
        verdict = (
            f"drift onset differs substantially with/without framing "
            f"({anchor_run.drift_onset} vs {control_run.drift_onset} tokens) — "
            "framing is doing something; worth checking attractor_jaccard too"
        )

    return ControlComparison(
        anchor_run=anchor_run,
        control_run=control_run,
        attractor_jaccard=jaccard,
        verdict=verdict,
    )
