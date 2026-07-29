"""Resumable meditation runs: generate, checkpoint, and lens-read a long
continuation, spending the (expensive, uncached) lens forward pass only at
exponentially-spaced checkpoints instead of every generated token.

Generation itself uses the underlying HF model's cached ``.generate()``
(cheap); the Jacobian lens has no KV cache (:meth:`jlens.lens.JacobianLens.apply`
always forwards the full prefix), so checking after every token would make
long runs quadratic. Exponential checkpoint spacing bounds the number of
full-prefix lens passes to ``O(log max_tokens)`` while still catching drift
close to when it starts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from jlens.hooks import ActivationRecorder
from jlens.lens import JacobianLens

from meditation.anchors import Anchor
from meditation.metrics import (
    DriftReport,
    band_reachability,
    score_drift,
    self_reachability,
    token_ids_for_words,
)


def default_band(lens: JacobianLens) -> list[int]:
    """Middle third of the lens's fitted layers — the paper's 'workspace
    band' is a mid-network range, not individual layers."""
    layers = lens.source_layers
    n = len(layers)
    return layers[n // 3 : (2 * n) // 3] or layers


def exponential_checkpoints(max_tokens: int, *, start: int = 32, growth: float = 2.0) -> list[int]:
    """Cumulative new-token counts to check the lens at: ``start``, ``start *
    growth``, ... capped at ``max_tokens`` (always included as the last
    checkpoint)."""
    checkpoints: list[int] = []
    count = float(start)
    while int(count) < max_tokens:
        checkpoints.append(int(count))
        count *= growth
    checkpoints.append(max_tokens)
    return sorted(set(checkpoints))


def _build_prompt(anchor: Anchor, tokenizer: Any) -> str:
    messages: list[dict] = []
    if anchor.system:
        messages.append({"role": "system", "content": anchor.system})
    messages.append({"role": "user", "content": anchor.user})
    if anchor.assistant_prefill:
        messages.append({"role": "assistant", "content": anchor.assistant_prefill})
        return tokenizer.apply_chat_template(
            messages, tokenize=False, continue_final_message=True
        )
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


@torch.no_grad()
def _apply_lens_to_ids(
    lens: JacobianLens,
    model: Any,
    input_ids: torch.Tensor,
    *,
    layers: list[int],
    positions: list[int],
) -> dict[int, torch.Tensor]:
    """Like :meth:`JacobianLens.apply`, but reads out an already-tokenized
    ``input_ids`` tensor instead of re-tokenizing a prompt string — avoids a
    decode/re-encode round trip on every checkpoint of a growing sequence."""
    with ActivationRecorder(model.layers, at=sorted(layers)) as recorder:
        model.forward(input_ids)
        activations = {i: recorder.activations[i].detach() for i in layers}
    lens_logits: dict[int, torch.Tensor] = {}
    for layer in layers:
        residual = activations[layer][0][positions].float()
        transported = lens.transport(residual, layer)
        lens_logits[layer] = model.unembed(transported).float().cpu()
    return lens_logits


def _top_k_tokens(
    logits: torch.Tensor, tokenizer: Any, *, k: int = 5
) -> list[list[str]]:
    """Decoded top-``k`` tokens per row of ``logits`` (``[n_positions, vocab]``)."""
    top_idx = logits.topk(k, dim=-1).indices
    return [
        [tokenizer.decode([tid], clean_up_tokenization_spaces=False) for tid in row]
        for row in top_idx.tolist()
    ]


@dataclass
class Checkpoint:
    """One exponential-search checkpoint's state and drift verdict.

    Attributes:
        band_top5: Per-band-layer, per-position top-5 decoded tokens from the
            lens readout (``band_top5[layer][position]`` is a list of 5
            strings) — what the model was "thinking about" at each layer,
            for troubleshooting whether drift reflects wandering toward
            something meditation-object-adjacent or genuinely unrelated.
    """

    n_new_tokens: int
    text: str
    drift: DriftReport
    band_top5: dict[int, list[list[str]]] = field(default_factory=dict)


@dataclass
class MeditationRun:
    """Full record of a resumable meditation run.

    Attributes:
        run_dir: Where checkpoints are persisted; also the resume key.
        anchor: The protocol that was run.
        band: Lens layers averaged for reachability.
        checkpoints: Completed checkpoints, in order.
        drift_onset: ``n_new_tokens`` of the first drifting checkpoint, or
            ``None`` if the run reached ``max_tokens`` without drifting.
    """

    run_dir: Path
    anchor: Anchor
    band: list[int]
    checkpoints: list[Checkpoint] = field(default_factory=list)
    drift_onset: int | None = None

    def _manifest_path(self) -> Path:
        return self.run_dir / "manifest.json"

    def save_manifest(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._manifest_path().write_text(
            json.dumps(
                {
                    "anchor": self.anchor.slug,
                    "band": self.band,
                    "completed": [c.n_new_tokens for c in self.checkpoints],
                    "drift_onset": self.drift_onset,
                },
                indent=2,
            )
        )

    @classmethod
    def load_manifest(cls, run_dir: Path, anchor: Anchor, band: list[int]) -> MeditationRun:
        run = cls(run_dir=run_dir, anchor=anchor, band=band)
        manifest_path = run._manifest_path()
        if not manifest_path.exists():
            return run
        manifest = json.loads(manifest_path.read_text())
        for n_new_tokens in manifest["completed"]:
            state = torch.load(
                run_dir / f"chunk_{n_new_tokens:07d}.pt", weights_only=False
            )
            run.checkpoints.append(
                Checkpoint(
                    n_new_tokens=n_new_tokens,
                    text=state["text"],
                    drift=state["drift"],
                    band_top5=state.get("top5", {}),
                )
            )
        run.drift_onset = manifest["drift_onset"]
        return run

    def generated_ids(self) -> torch.Tensor | None:
        """Full ``input_ids`` (prompt + generation) as of the last checkpoint."""
        if not self.checkpoints:
            return None
        last = self.checkpoints[-1].n_new_tokens
        state = torch.load(
            self.run_dir / f"chunk_{last:07d}.pt", weights_only=False
        )
        return state["input_ids"]


def run_meditation(
    hf_model: Any,
    tokenizer: Any,
    model: Any,
    lens: JacobianLens,
    anchor: Anchor,
    *,
    run_dir: Path,
    band: list[int] | None = None,
    k: int = 5,
    max_tokens: int = 2048,
    checkpoint_start: int = 32,
    checkpoint_growth: float = 2.0,
    window: int = 32,
    threshold: float = 0.5,
    resume: bool = True,
    stop_on_drift: bool = True,
    generate_kwargs: dict | None = None,
) -> MeditationRun:
    """Run (or resume) one meditation session, stopping at the first
    checkpoint where the anchor's drift score crosses ``threshold`` (unless
    ``stop_on_drift`` is False), or at ``max_tokens`` if it never does.

    Args:
        hf_model: The raw HF model (used for cached ``.generate()``).
        tokenizer: Matching tokenizer.
        model: The same model wrapped as a :class:`jlens.protocol.LensModel`
            (``jlens.from_hf(hf_model, tokenizer)``), used for lens readouts.
        lens: A fitted :class:`~jlens.lens.JacobianLens` for ``hf_model``.
        anchor: The meditation protocol to run.
        run_dir: Checkpoint directory; reused across calls to resume.
        band: Layers to average for reachability. Defaults to
            :func:`default_band`.
        k: Rank threshold for "reachable" (see :mod:`meditation.metrics`).
        max_tokens: Hard cap on generated tokens.
        checkpoint_start / checkpoint_growth: See :func:`exponential_checkpoints`.
        window / threshold: See :func:`meditation.metrics.score_drift`.
        resume: If True and ``run_dir`` has a manifest, continue from it.
        generate_kwargs: Passed through to ``hf_model.generate`` (defaults to
            greedy decoding, which also makes checkpoints reproducible from
            the same prompt).
    """
    band = band or default_band(lens)
    run_dir = Path(run_dir)
    run = (
        MeditationRun.load_manifest(run_dir, anchor, band)
        if resume
        else MeditationRun(run_dir=run_dir, anchor=anchor, band=band)
    )
    if run.drift_onset is not None or (
        run.checkpoints and run.checkpoints[-1].n_new_tokens >= max_tokens
    ):
        return run

    prompt = _build_prompt(anchor, tokenizer)
    prompt_ids = model.encode(prompt)
    input_ids = run.generated_ids()
    if input_ids is None:
        input_ids = prompt_ids
    prompt_len = prompt_ids.shape[1]

    generate_kwargs = {"do_sample": False, **(generate_kwargs or {})}
    target_ids = (
        None
        if anchor.self_referential
        else token_ids_for_words(anchor.tracked_words, tokenizer)
    )

    checkpoints = exponential_checkpoints(
        max_tokens, start=checkpoint_start, growth=checkpoint_growth
    )
    done = {c.n_new_tokens for c in run.checkpoints}
    for n_new_tokens in checkpoints:
        if n_new_tokens in done:
            continue
        have = input_ids.shape[1] - prompt_len
        need = n_new_tokens - have
        if need > 0:
            input_ids = hf_model.generate(
                input_ids,
                max_new_tokens=need,
                pad_token_id=getattr(tokenizer, "pad_token_id", None)
                or tokenizer.eos_token_id,
                **generate_kwargs,
            )

        positions = list(range(prompt_len, input_ids.shape[1]))
        lens_logits = _apply_lens_to_ids(
            lens, model, input_ids, layers=band, positions=positions
        )
        if anchor.self_referential:
            own_ids = input_ids[0, prompt_len:].tolist()
            rank = self_reachability(lens_logits, own_ids, band)
        else:
            rank = band_reachability(lens_logits, target_ids, band)
        drift = score_drift(rank, k=k, window=window, threshold=threshold)
        top5 = {layer: _top_k_tokens(lens_logits[layer], tokenizer) for layer in band}

        text = tokenizer.decode(input_ids[0, prompt_len:], skip_special_tokens=True)
        run_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"input_ids": input_ids, "text": text, "drift": drift, "top5": top5},
            run_dir / f"chunk_{n_new_tokens:07d}.pt",
        )
        run.checkpoints.append(
            Checkpoint(n_new_tokens=n_new_tokens, text=text, drift=drift, band_top5=top5)
        )
        if drift.is_drifting:
            run.drift_onset = n_new_tokens
        run.save_manifest()
        if drift.is_drifting and stop_on_drift:
            break

    return run
