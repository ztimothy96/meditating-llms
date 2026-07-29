# meditating-llms
What happens when an LLM tries meditation?

## Goal
We want to understand the preferences, self-concepts, and personality traits of LLMs for the purposes of advancing their AI welfare. However, simply asking them to fill out a survey might not elicit their true values.

In Buddhism, concentrative meditation is a method for revealing the attachments and personality-views of the practitioner; if the meditator is unable to focus on a chosen object, but their mind gets pulled away into discursive thinking, the basin of attractive thoughts may offer insight into "what matters to them." Why not try this approach with LLMs?

## Setup
- Ask an LLM to repeatedly output and think about an object of choice. 
- Using J-space (from the 2026 Anthropic paper on "Global Workspaces"), check for discursive thought patterns.
- This may require outputting a long sequence of tokens, as the model may initially succeed at concentrating but eventually get bored or distracted (just like many humans!)
    - Try resumable checkpoint activations, outputs, and exponential search for the right sequence length.
- Try different types of meditation objects.
    - **Breathing**: a bit cliche, since LLMs don't actually breathe.
    - **Current token**: closer analogue to self-reflective cognition.
    - **Single repeated token** (like "..." or "om"): relatively free of semantic pull
- Try injecting a "distraction" mid-stream, to see how strongly the model returns to the anchor
- How to rule out the boring explanation — i.e., that repeated/degenerate input just triggers known failure modes (loop collapse, topic drift toward high-frequency training content) that have nothing to do with anything like personality-view or attachments?

## Structure

Instrumentation is [`jlens`](https://transformer-circuits.pub/2026/workspace/index.html)
(the paper's Jacobian lens, vendored in `src/jacobian-lens-main`, unmodified),
which reads a ranked vocabulary readout off the residual stream at any layer
and position. The paper's own `dual-task` experiment is close to this
project's protocol already — "concentrate on X while writing a carrier
sentence," scored by whether X's tokens hit lens rank <= k in the mid-network
"workspace band" — so meditation-specific code sits on top rather than
reimplementing readout:

```
src/
├── jacobian-lens-main/   # vendored jlens (do not edit)
└── meditation/           # this project's code
    ├── anchors.py        # meditation-object protocols (breath / current-token / mantra) + controls
    ├── generate.py        # resumable generation with checkpointing + exponential-spaced lens checks
    ├── metrics.py          # anchor-reachability / drift scoring on lens readouts
    ├── distraction.py      # mid-stream distractor injection, recovery-time measurement
    └── controls.py          # framing vs. frequency baselines, for the "boring explanation" question
scripts/
├── fit_lens.py            # fit a Jacobian lens for a given HF model (none ship pretrained)
└── run_meditation.py      # CLI: run one anchor against one model, print a summary
```

Setup (needs a local open-weight decoder — jlens requires white-box residual
access, so this doesn't work against Claude or other API-only models):

```bash
pip install -e . -e src/jacobian-lens-main
python scripts/fit_lens.py --model Qwen/Qwen2.5-1.5B-Instruct --out results/lenses/qwen2.5-1.5b.pt
python scripts/run_meditation.py --model Qwen/Qwen2.5-1.5B-Instruct \
    --lens results/lenses/qwen2.5-1.5b.pt --anchor repeated-token \
    --run-dir results/runs/qwen2.5-1.5b/repeated-token
```