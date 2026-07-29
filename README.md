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