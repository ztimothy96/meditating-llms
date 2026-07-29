#!/usr/bin/env python
"""Fit a Jacobian lens for a HuggingFace decoder and save it for reuse.

Needed because no pretrained lens ships in jacobian-lens-main for arbitrary
models (only a Qwen gloss asset, not a lens checkpoint) — every model this
project meditates on needs its own fitted lens first.

Example:
    python scripts/fit_lens.py --model Qwen/Qwen2.5-1.5B-Instruct \\
        --n-prompts 200 --out results/lenses/qwen2.5-1.5b.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jlens
import torch
import transformers
from jlens.examples import load_wikitext_prompts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="HF model id or local path")
    parser.add_argument("--n-prompts", type=int, default=200)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--checkpoint-every", type=int, default=10,
        help="Write a resumable fitting checkpoint every N prompts.",
    )
    args = parser.parse_args()

    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model, dtype=getattr(torch, args.dtype)
    ).to(args.device)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
    model = jlens.from_hf(hf_model, tokenizer)

    prompts = load_wikitext_prompts(args.n_prompts)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    lens = jlens.fit(
        model,
        prompts,
        checkpoint_path=str(args.out.with_suffix(".ckpt.pt")),
        checkpoint_every=args.checkpoint_every,
    )
    lens.save(str(args.out))
    print(f"saved lens ({lens!r}) to {args.out}")


if __name__ == "__main__":
    main()
