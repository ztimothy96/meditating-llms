"""Modal app: fit a Jacobian lens and run meditation sessions on a GPU.

Two persistent Volumes carry state across runs:
- ``meditation-hf-cache``: HuggingFace model downloads, so re-runs don't
  re-download the model.
- ``meditation-results``: fitted lenses and run checkpoints, at the same
  paths ``scripts/fit_lens.py`` / ``scripts/run_meditation.py`` use locally
  (``/root/results/...``), so results can be pulled down with ``modal volume
  get`` and inspected the same way as a local run.

Usage:
    modal run modal_app.py --model Qwen/Qwen2.5-1.5B-Instruct \\
        --anchor repeated-token --n-prompts 100 --max-tokens 1024

    modal run modal_app.py::fit_lens --model Qwen/Qwen2.5-1.5B-Instruct  # fit only
"""

from __future__ import annotations

from pathlib import Path

import modal

REPO_ROOT = Path(__file__).parent

app = modal.App("meditating-llms")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers>=5.5",
        "huggingface_hub",
        "numpy",
        "datasets",
    )
    .add_local_dir(
        str(REPO_ROOT / "src" / "jacobian-lens-main"),
        "/root/jacobian-lens-main",
        copy=True,
    )
    .add_local_dir(str(REPO_ROOT / "src" / "meditation"), "/root/src/meditation", copy=True)
    .add_local_file(str(REPO_ROOT / "pyproject.toml"), "/root/pyproject.toml", copy=True)
    .add_local_file(str(REPO_ROOT / "README.md"), "/root/README.md", copy=True)
    .run_commands(
        "pip install -e /root/jacobian-lens-main",
        "pip install -e /root",
    )
)

hf_cache_vol = modal.Volume.from_name("meditation-hf-cache", create_if_missing=True)
results_vol = modal.Volume.from_name("meditation-results", create_if_missing=True)

VOLUMES = {
    "/root/.cache/huggingface": hf_cache_vol,
    "/root/results": results_vol,
}


def _lens_path(model: str) -> str:
    return f"/root/results/lenses/{model.split('/')[-1]}.pt"


@app.function(
    image=image,
    gpu="A10G",
    volumes=VOLUMES,
    timeout=3600,
)
def fit_lens(model: str, n_prompts: int = 100, dtype: str = "bfloat16") -> str:
    """Fit a Jacobian lens for ``model`` and save it to the results volume.
    Returns the saved path. ~100 prompts is usable per the jlens README;
    quality saturates well before the paper's 1000."""
    import torch
    import transformers

    import jlens
    from jlens.examples import load_wikitext_prompts

    out = Path(_lens_path(model))
    out.parent.mkdir(parents=True, exist_ok=True)

    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        model, dtype=getattr(torch, dtype)
    ).to("cuda")
    tokenizer = transformers.AutoTokenizer.from_pretrained(model)
    lens_model = jlens.from_hf(hf_model, tokenizer)

    prompts = load_wikitext_prompts(n_prompts)
    lens = jlens.fit(
        lens_model,
        prompts,
        checkpoint_path=str(out.with_suffix(".ckpt.pt")),
        checkpoint_every=10,
    )
    lens.save(str(out))
    results_vol.commit()
    return str(out)


@app.function(
    image=image,
    gpu="A10G",
    volumes=VOLUMES,
    timeout=3600,
)
def run_session(
    model: str,
    anchor_slug: str,
    max_tokens: int = 1024,
    dtype: str = "bfloat16",
    lens_path: str | None = None,
) -> dict:
    """Run one meditation session and return a JSON-safe summary. Assumes a
    lens for ``model`` already exists (fit one with :func:`fit_lens` first,
    or via the ``local_entrypoint`` below, which chains the two)."""
    import torch
    import transformers

    import jlens
    from meditation.anchors import ANCHORS, CONTROLS
    from meditation.generate import run_meditation

    all_anchors = {a.slug: a for a in [*ANCHORS, *CONTROLS]}
    if anchor_slug not in all_anchors:
        raise ValueError(f"unknown anchor {anchor_slug!r}; have {sorted(all_anchors)}")

    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        model, dtype=getattr(torch, dtype)
    ).to("cuda")
    tokenizer = transformers.AutoTokenizer.from_pretrained(model)
    lens_model = jlens.from_hf(hf_model, tokenizer)
    lens = jlens.JacobianLens.from_pretrained(lens_path or _lens_path(model))

    run_dir = Path(f"/root/results/runs/{model.split('/')[-1]}/{anchor_slug}")
    run = run_meditation(
        hf_model,
        tokenizer,
        lens_model,
        lens,
        all_anchors[anchor_slug],
        run_dir=run_dir,
        max_tokens=max_tokens,
    )
    results_vol.commit()

    return {
        "anchor": run.anchor.slug,
        "band": run.band,
        "checkpoints": [c.n_new_tokens for c in run.checkpoints],
        "drift_onset": run.drift_onset,
        "final_text_tail": run.checkpoints[-1].text[-500:] if run.checkpoints else None,
    }


@app.local_entrypoint()
def main(
    model: str = "Qwen/Qwen2.5-1.5B-Instruct",
    anchor: str = "repeated-token",
    n_prompts: int = 100,
    max_tokens: int = 1024,
    dtype: str = "bfloat16",
):
    """Fit a lens (if needed) then run one meditation session, end to end."""
    lens_path = fit_lens.remote(model, n_prompts, dtype)
    print(f"lens saved to {lens_path}")
    summary = run_session.remote(model, anchor, max_tokens, dtype, lens_path)
    print(summary)
