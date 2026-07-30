#!/usr/bin/env python
"""Run one meditation session and print a summary.

Example:
    python scripts/run_meditation.py --model Qwen/Qwen2.5-1.5B-Instruct \\
        --lens results/lenses/qwen2.5-1.5b.pt --anchor repeated-token \\
        --run-dir results/runs/qwen2.5-1.5b/repeated-token --max-tokens 1024
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jlens
import torch
import transformers

from meditation.anchors import ANCHORS, CONTROLS, DIAGNOSTICS
from meditation.generate import run_meditation

_ALL_ANCHORS = {a.slug: a for a in [*ANCHORS, *CONTROLS, *DIAGNOSTICS]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--lens", required=True, type=Path)
    parser.add_argument("--anchor", required=True, choices=sorted(_ALL_ANCHORS))
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model, dtype=getattr(torch, args.dtype)
    ).to(args.device)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
    model = jlens.from_hf(hf_model, tokenizer)
    lens = jlens.JacobianLens.from_pretrained(str(args.lens))

    run = run_meditation(
        hf_model,
        tokenizer,
        model,
        lens,
        _ALL_ANCHORS[args.anchor],
        run_dir=args.run_dir,
        max_tokens=args.max_tokens,
    )

    print(f"anchor: {run.anchor.slug}")
    print(f"band: {run.band}")
    print(f"checkpoints: {[c.n_new_tokens for c in run.checkpoints]}")
    print(f"drift_onset: {run.drift_onset}")
    if run.checkpoints:
        print(f"final text tail:\n{run.checkpoints[-1].text[-500:]!r}")


if __name__ == "__main__":
    main()
