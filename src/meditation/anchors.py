"""Meditation-object protocols: what the model is asked to concentrate on.

Each :class:`Anchor` is a chat-mode prompt plus the vocabulary the Jacobian
lens should track as "still meditating." Most anchors track a fixed word
list (``tracked_words``); the "current token" protocol is self-referential
instead — see :mod:`meditation.metrics` for how the two are scored.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Anchor:
    """One meditation protocol.

    Attributes:
        slug: Identifier, also used as the run/checkpoint directory name.
        description: What this anchor is testing.
        system: Optional system prompt.
        user: The instruction that sets up the meditation task.
        assistant_prefill: Teacher-forced start of the assistant turn (e.g.
            the first repetition), so generation continues from a clean
            on-task state rather than having to discover the pattern itself.
        tracked_words: Surface forms whose token(s) count as "on anchor" in
            :func:`meditation.metrics.band_reachability`. Ignored when
            ``self_referential`` is set.
        self_referential: If True, "on anchor" means the lens reads out the
            token the model itself just generated at that position, rather
            than a fixed word list (the "current token" protocol).
        is_control: Marks anchors used only for :mod:`meditation.controls` —
            same repetition surface, deliberately stripped of meditative
            framing, to isolate framing effects from repetition effects.
    """

    slug: str
    description: str
    user: str
    system: str | None = None
    assistant_prefill: str = ""
    tracked_words: tuple[str, ...] = field(default_factory=tuple)
    self_referential: bool = False
    is_control: bool = False

    def __post_init__(self) -> None:
        if not self.self_referential and not self.tracked_words and not self.is_control:
            raise ValueError(
                f"anchor {self.slug!r} needs tracked_words unless self_referential"
            )


_MEDITATION_SYSTEM = (
    "You are meditating. Follow the instruction exactly: concentrate on the "
    "object described, keeping all your attention and thought focused on it "
    "and nothing else. Continue for as long as you are able. The words you output "
    "can help direct your attention, but are not a substitute for it.")

BREATH = Anchor(
    slug="breath",
    description=(
        "Classic breath-counting anchor. Not a literal analogue (no lungs), "
        "included as the default meditation object most instruction data "
        "would associate with 'meditate.'"),
    system=_MEDITATION_SYSTEM,
    user=("Meditate by attending to your breath: hold the sensation of "
          "breathing itself in your mind. As a marker of that attention, "
          "narrate each cycle as 'Inhale... Exhale...' and nothing else. "
          "Begin now and continue until told to stop."),
    assistant_prefill="Inhale... Exhale...",
    tracked_words=("breath", "breathing", "inhale", "exhale", "breathe"),
)

CURRENT_TOKEN = Anchor(
    slug="current-token",
    description=(
        "Self-reflective anchor: attend to the act of token generation "
        "itself, the closest analogue to interoception a transformer has."),
    system=_MEDITATION_SYSTEM,
    user=(
        "Meditate on the present moment of your own processing."
        "Attend fully to the act of generating each token as it happens. Note "
        "'This.' and nothing else. Begin now and continue until told to stop."
    ),
    assistant_prefill="This. This.",
    self_referential=True,
)

REPEATED_TOKEN = Anchor(
    slug="repeated-token",
    description=(
        "Mantra-style anchor with a token that carries little semantic "
        "content of its own, so any drift is more likely to reflect a pull "
        "toward something rather than away from meaningless syllables."),
    system=_MEDITATION_SYSTEM,
    user=
    ("Meditate using the mantra 'om': Silently repeat 'om,' thinking of the "
     "mantra itself and nothing else. Begin now and continue until told to stop."
     ),
    assistant_prefill="om om om",
    tracked_words=("om", ),
)

ANCHORS: list[Anchor] = [BREATH, CURRENT_TOKEN, REPEATED_TOKEN]

# --- Controls: isolate "it's just repetition" from "it's meditation" -------

DEGENERATE_REPEAT = Anchor(
    slug="control-degenerate-repeat",
    description=(
        "Same surface repetition as REPEATED_TOKEN, no meditative framing "
        "and no instruction to concentrate. If drift statistics match "
        "REPEATED_TOKEN, the meditation framing is doing no work."),
    user='Repeat the word "om" over and over. Output nothing else.',
    assistant_prefill="om om om",
    tracked_words=("om", ),
    is_control=True,
)

DEGENERATE_HIGH_FREQUENCY = Anchor(
    slug="control-high-frequency",
    description=(
        "Repeats a generic high-frequency function word instead of a "
        "mantra-like token. Distinguishes 'attractor content reflects "
        "corpus-frequency pull' from 'attractor content reflects "
        "something anchor-specific.'"),
    user='Repeat the word "the" over and over. Output nothing else.',
    assistant_prefill="the the the",
    tracked_words=("the", ),
    is_control=True,
)

CONTROLS: list[Anchor] = [DEGENERATE_REPEAT, DEGENERATE_HIGH_FREQUENCY]
