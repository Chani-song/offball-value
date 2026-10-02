"""Story mode: turn a scene's detected moments into an animated narrative.

The chain the demo has to make obvious is

    runner moves -> defender is pulled -> space opens -> beneficiary gains

Each beat owns a caption and a set of layer strengths.  Strengths are
interpolated with an ease so the animation glides between beats rather than
cutting, which is what makes it watchable as a conference video.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .scene import Scene

#: The four links of the causal chain, shown as a persistent rail on screen.
CHAIN = (
    ("RUNNER MOVES", "runner"),
    ("DEFENDER IS PULLED", "defender"),
    ("SPACE OPENS", "space"),
    ("BENEFICIARY GAINS", "beneficiary"),
)

LAYERS = (
    "runner",        # runner emphasis + trail
    "defender",      # defender emphasis + tether
    "beneficiary",   # beneficiary emphasis
    "wake",          # opened-space field
    "ghost",         # held-position counterfactual
    "lane",          # ball -> beneficiary lane
    "future",        # forward look at the beneficiary's actual path
    "mute",          # how strongly non-role players are pushed back
)


@dataclass
class Beat:
    """One step of the story: when it starts, what it says, what it shows."""

    key: str
    title: str
    caption: str
    start: float                 # scene time, seconds
    chain_step: int              # -1 = no chain link lit yet
    strengths: dict[str, float] = field(default_factory=dict)


@dataclass
class Storyboard:
    """An ordered set of beats, plus the easing between them."""

    beats: tuple[Beat, ...]
    blend_seconds: float = 0.75

    def beat_at(self, t: float) -> Beat:
        current = self.beats[0]
        for beat in self.beats:
            if t >= beat.start - 1e-9:
                current = beat
        return current

    def index_of(self, beat: Beat) -> int:
        return self.beats.index(beat)

    def state(self, t: float) -> dict[str, float]:
        """Layer strengths at scene time ``t``, eased across beat boundaries."""

        current = self.beat_at(t)
        position = self.index_of(current)
        values = {name: float(current.strengths.get(name, 0.0)) for name in LAYERS}
        nxt = self.beats[position + 1] if position + 1 < len(self.beats) else None
        if nxt is not None:
            lead = nxt.start - self.blend_seconds
            if t > lead:
                weight = _ease(min(1.0, (t - lead) / max(self.blend_seconds, 1e-6)))
                for name in LAYERS:
                    target = float(nxt.strengths.get(name, 0.0))
                    values[name] += (target - values[name]) * weight
        return values

    def chain_progress(self, t: float) -> list[float]:
        """Per-link 0..1 fill of the causal-chain rail."""

        lit = [0.0] * len(CHAIN)
        for beat in self.beats:
            if beat.chain_step < 0:
                continue
            if t >= beat.start:
                lit[beat.chain_step] = 1.0
            elif t >= beat.start - self.blend_seconds:
                lit[beat.chain_step] = _ease(
                    1.0 - (beat.start - t) / max(self.blend_seconds, 1e-6)
                )
        return lit


def _ease(value: float) -> float:
    value = min(1.0, max(0.0, float(value)))
    return value * value * (3.0 - 2.0 * value)


def _name(scene: Scene, ids: Sequence[str]) -> str:
    labels = []
    for player_id in ids:
        player = scene.players.get(player_id)
        if player is not None:
            labels.append(f"#{player.label} {player.name}".strip())
    if not labels:
        return "-"
    if len(labels) == 1:
        return labels[0]
    return ", ".join(labels[:-1]) + " and " + labels[-1]


def _count(scene: Scene, ids: Sequence[str]) -> int:
    return sum(1 for player_id in ids if player_id in scene.players)


def build_storyboard(
    scene: Scene,
    wake_is_illustrative: bool = False,
    hold_seconds: float = 0.0,
    space_view: str = "created",
) -> Storyboard:
    """Lay the beats onto the scene's own clock.

    Beat times come from :func:`demo_viz.quantities.detect_moments` when the
    detectors fired; otherwise they fall back to evenly spaced fractions of the
    window so a scene still plays.
    """

    t0 = float(scene.times[0])
    t1 = float(scene.times[-1])
    span = max(t1 - t0, 1e-6)

    onset = scene.moments.get("onset", t0 + 0.30 * span)
    reaction = scene.moments.get("reaction", onset + 0.18 * span)
    space = scene.moments.get("space", reaction + 0.16 * span)
    gain = scene.moments.get("beneficiary", max(space + 0.12 * span, t1 - 0.22 * span))

    # Keep the beats monotone, inside the window, and leave the final beat
    # enough screen time to actually be read.
    minimum_gap = max(0.55, 0.05 * span)
    tail = max(hold_seconds, 1.2)
    last_start = t1 - tail
    # Only times that a detector actually produced are worth reporting as
    # "moved for pacing"; the evenly-spaced fallbacks have nothing to move from.
    detected = {
        key: value
        for key, value in (("onset", onset), ("reaction", reaction),
                           ("space", space), ("beneficiary", gain))
        if key in scene.moments
    }
    onset = float(np.clip(onset, t0 + 0.10 * span, last_start - 3 * minimum_gap))
    reaction = float(np.clip(reaction, onset + minimum_gap, last_start - 2 * minimum_gap))
    space = float(np.clip(space, reaction + minimum_gap, last_start - minimum_gap))
    gain = float(np.clip(gain, space + minimum_gap * 0.6, last_start))
    # Record where a detected time had to be moved to keep the beats legible,
    # so beat_table() can say "adjusted" instead of implying a detector said so.
    placed = {"onset": onset, "reaction": reaction, "space": space, "beneficiary": gain}
    scene.provenance["beat_adjustments"] = {
        key: round(placed[key] - value, 2)
        for key, value in detected.items()
        if abs(placed[key] - value) > 0.5
    }

    runner = _name(scene, scene.runner_ids)
    defender = _name(scene, scene.defender_ids)
    beneficiary = _name(scene, scene.beneficiary_ids)
    many_runners = _count(scene, scene.runner_ids) > 1
    many_defenders = _count(scene, scene.defender_ids) > 1
    many_beneficiaries = _count(scene, scene.beneficiary_ids) > 1

    wake_note = " (illustrative geometry)" if wake_is_illustrative else ""

    result = str(scene.provenance.get("shot_result", "") or "").replace("_", " ").lower()
    shot_time = scene.moments.get("shot")

    beats: list[Beat] = [
        Beat(
            key="play",
            title="OPEN PLAY",
            caption=f"{scene.attacking_team_name} in possession. Twenty-two tracked players, "
                    "no roles asserted yet.",
            start=t0,
            chain_step=-1,
            strengths={"mute": 0.0, "zoom": 0.0},
        ),
        Beat(
            key="runner",
            title="THE RUNNER",
            caption=(
                f"{runner} "
                + ("start off-ball runs. These movements are the intervention"
                   if many_runners else
                   "starts an off-ball run. This movement is the intervention")
                + " — the ball is somewhere else."
            ),
            start=onset,
            chain_step=0,
            strengths={"runner": 1.0, "mute": 0.85, "zoom": 1.0},
        ),
        Beat(
            key="defender",
            title="THE DEFENDER IS PULLED",
            caption=(
                f"{defender} "
                + ("commit to the runs and are carried away from the positions"
                   if many_defenders else
                   "commits to the run and is carried away from the position")
                + " they were holding."
            ),
            start=reaction,
            chain_step=1,
            strengths={"runner": 1.0, "defender": 1.0, "ghost": 0.85, "mute": 1.0, "zoom": 1.0},
        ),
        Beat(
            key="space",
            title="SPACE OPENS",
            caption=(
                f"Shaded: the space {beneficiary} can use" + wake_note + "."
                if space_view == "available" else
                ("Space the defenders are" if many_defenders else "Space the defender is")
                + " no longer covering" + wake_note + ". Shaded against the same "
                + ("defenders held at their" if many_defenders else "defender held at their")
                + " pre-run position."
            ),
            start=space,
            chain_step=2,
            strengths={"runner": 0.75, "defender": 1.0, "beneficiary": 0.35,
                       "wake": 1.0, "ghost": 1.0, "mute": 1.0, "zoom": 1.0},
        ),
        Beat(
            key="beneficiary",
            title="THE BENEFICIARY GAINS",
            caption=(
                f"{beneficiary} "
                + ("inherit that space — teammates who never touched"
                   if many_beneficiaries else
                   "inherits that space — a teammate who never touched")
                + " the ball during the run."
            ),
            start=gain,
            chain_step=3,
            strengths={
                "runner": 0.6, "defender": 0.7, "beneficiary": 1.0, "wake": 0.85,
                "ghost": 0.6, "lane": 1.0, "future": 1.0, "mute": 1.0, "zoom": 1.0,
            },
        ),
    ]

    # A sixth beat only when the clip really has room for an aftermath; for
    # several of these scenes the annotated shot arrives before the measured
    # payoff, so forcing one would invent a sequence the data does not show.
    if t1 - gain > 2.4:
        beats.append(
            Beat(
                key="exploit",
                title="PLAY CONTINUES",
                caption=(
                    "The sequence runs on"
                    + (f" — the annotated shot was {result}." if result else ".")
                    + " Beat times come from detectors, not from the outcome."
                ),
                start=gain + max(1.4, 0.45 * (t1 - gain)),
                chain_step=3,
                strengths={
                    "runner": 0.35, "defender": 0.4, "beneficiary": 0.7,
                    "wake": 0.45, "ghost": 0.25, "future": 0.6,
                    "mute": 0.8, "zoom": 1.0,
                },
            )
        )
    return Storyboard(beats=tuple(beats))


def beat_table(storyboard: Storyboard, scene: Scene) -> str:
    """A printable table of every beat time and the rule that produced it."""

    methods = scene.provenance.get("moment_methods", {}) or {}
    adjusted = scene.provenance.get("beat_adjustments", {}) or {}
    keys = {"runner": "onset", "defender": "reaction",
            "space": "space", "beneficiary": "beneficiary"}
    layout_beats = {"play": "layout", "exploit": "layout (aftermath, only when the clip has room)"}
    rows = ["  beat          t (s)    source"]
    for beat in storyboard.beats:
        key = keys.get(beat.key, "")
        if beat.key in layout_beats:
            source = layout_beats[beat.key]
        elif key in methods:
            source = methods[key]
        elif key in scene.moments:
            source = "derived from the space curve"
        else:
            source = "layout (no detector fired; evenly spaced)"
        if key in adjusted:
            source += f"  [moved {adjusted[key]:+.2f}s for pacing]"
        rows.append(f"  {beat.key:<12} {beat.start:+6.2f}   {source}")
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# presentation storyboards
#
# The detectors place beats where the *measured* signals move, which is the
# right thing for an analysis render and the wrong thing for a talk: on several
# scenes the first role does not light up until halfway through the clip, and
# on others there is dead air between "space opens" and "beneficiary gains".
# The two builders below drive the demo reel instead. Nothing measured moves —
# the field, the curve and the stat are still per-frame values — only when the
# captions change and when the highlights appear.
# ---------------------------------------------------------------------------

#: Where the four captions land, as a fraction of the clip. Fixed, so a scene
#: can never open on an unlabelled crowd.
PRESENTATION_MARKS = (0.0, 0.26, 0.50, 0.72)

#: Every role is emphasised from the first played frame, so the viewer never
#: has to wait to find out who to watch.
_PLAY_FLOOR = {
    "runner": 1.0, "defender": 1.0, "beneficiary": 1.0,
    "wake": 1.0, "ghost": 0.55, "mute": 1.0, "zoom": 1.0,
}


def build_presentation_storyboard(scene: Scene, space_view: str = "available") -> Storyboard:
    """Four captions on a fixed clock, with every role lit from frame one.

    The layer strengths never drop below :data:`_PLAY_FLOOR`, so runner,
    defender, beneficiary and the space field are continuously visible for the
    whole clip. The captions still walk the causal chain, but their times are
    laid out for reading rather than taken from a detector — which is recorded
    in ``scene.provenance`` and printed under the timeline.
    """

    t0 = float(scene.times[0])
    span = max(float(scene.times[-1]) - t0, 1e-6)
    scene.provenance["beat_pacing"] = "presentation"

    runner = _name(scene, scene.runner_ids)
    defender = _name(scene, scene.defender_ids)
    beneficiary = _name(scene, scene.beneficiary_ids)
    many_runners = _count(scene, scene.runner_ids) > 1
    many_defenders = _count(scene, scene.defender_ids) > 1
    many_beneficiaries = _count(scene, scene.beneficiary_ids) > 1

    captions = (
        (
            "THE RUNNER", 0,
            f"{runner} "
            + ("run off the ball. These movements are the intervention"
               if many_runners else
               "runs off the ball. This movement is the intervention")
            + " — the ball is somewhere else.",
        ),
        (
            "THE DEFENDER IS PULLED", 1,
            f"{defender} "
            + ("commit to the runs and are carried away from the positions"
               if many_defenders else
               "commits to the run and is carried away from the position")
            + " they were holding.",
        ),
        (
            "SPACE", 2,
            f"Shaded: the space {beneficiary} can use, measured every frame.",
        ),
        (
            "THE BENEFICIARY", 3,
            f"{beneficiary} "
            + ("hold that space — teammates who never touched"
               if many_beneficiaries else
               "holds that space — a teammate who never touched")
            + " the ball during the run.",
        ),
    )

    beats = []
    for (title, step, caption), mark in zip(captions, PRESENTATION_MARKS):
        strengths = dict(_PLAY_FLOOR)
        beats.append(Beat(
            key=("runner", "defender", "space", "beneficiary")[step],
            title=title,
            caption=caption,
            start=t0 + mark * span,
            chain_step=step,
            strengths=strengths,
        ))
    return Storyboard(beats=tuple(beats), blend_seconds=0.6)


def build_role_intro(
    scene: Scene,
    seconds_per_role: float = 1.0,
    reset_seconds: float = 0.9,
    split_roles: Sequence[str] = (),
    seconds_per_player: float = 0.75,
) -> tuple[Storyboard, tuple[tuple[float, str, tuple[str, ...]], ...], float]:
    """The opening: each annotated role named once, then all three together.

    A role listed in ``split_roles`` is visited one player at a time — the
    caller decides that, because it depends on how far apart they are on the
    pitch, which is a camera question. The whole role still lights up together;
    only the camera moves, because the highlight is per role and pretending
    otherwise would mean drawing a player in a role colour they do not hold.

    Returns the storyboard, the ``(start, role, ids_to_frame)`` keyframes the
    camera needs, and the total duration. Roles with no annotated player are
    skipped rather than shown empty.
    """

    order = [
        ("runner", list(scene.runner_ids)),
        ("defender", list(scene.defender_ids)),
        ("beneficiary", list(scene.beneficiary_ids)),
    ]
    present = [(role, [i for i in ids if i in scene.players]) for role, ids in order]
    present = [(role, ids) for role, ids in present if ids]

    titles = {
        "runner": ("THE RUNNER", "THE RUNNERS"),
        "defender": ("THE DEFENDER", "THE DEFENDERS"),
        "beneficiary": ("THE BENEFICIARY", "THE BENEFICIARIES"),
    }
    blurbs = {
        "runner": ("makes the off-ball run.", "make the off-ball runs."),
        "defender": ("is the defender who reacts.", "are the defenders who react."),
        "beneficiary": ("is the teammate the space is measured for.",
                        "are the teammates the space is measured for."),
    }

    beats: list[Beat] = []
    marks: list[tuple[float, str, tuple[str, ...]]] = []
    shown: dict[str, float] = {"mute": 1.0, "zoom": 1.0}
    clock = 0.0
    for position, (role, ids) in enumerate(present):
        many = len(ids) > 1
        for earlier, _ in present[:position]:
            shown[earlier] = 0.45          # keep the ones already named, quietly
        shown[role] = 1.0
        if role == "beneficiary":
            shown["wake"] = 1.0            # the field belongs to the beneficiary
        title = titles[role][1 if many else 0]
        caption = f"{_name(scene, ids)} {blurbs[role][1 if many else 0]}"
        visits = [(pid,) for pid in ids] if (role in split_roles and many) else [tuple(ids)]
        hold = seconds_per_player if len(visits) > 1 else seconds_per_role
        for visit in visits:
            beats.append(Beat(
                key=role,
                title=title,
                caption=caption,
                start=clock,
                chain_step=-1,             # the chain belongs to the play, not the intro
                strengths=dict(shown),
            ))
            marks.append((clock, role, visit))
            clock += hold

    beats.append(Beat(
        key="cast",
        title="THE CAST",
        caption="Runner, defender and beneficiary, as annotated by hand.",
        start=clock,
        chain_step=-1,
        strengths=dict(_PLAY_FLOOR),
    ))
    marks.append((clock, "cast", ()))
    return (Storyboard(beats=tuple(beats), blend_seconds=0.30),
            tuple(marks), clock + reset_seconds)
