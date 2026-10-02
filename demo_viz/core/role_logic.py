"""Who reacts to whom, and who gains — computed, not guessed at.

Two rankings power the guided parts of the app:

**Candidate defenders for a runner** are ranked by a geometric response score
built on ``offball_value.dynamic_marking``: how much of the defender's motion
points at the moving goal-side point in front of the runner, and how close the
defender stays. That is a transparent heuristic, not a learned model of
defensive intent, and the app labels it that way.

**Candidate beneficiaries** are ranked by the quantity the demo actually cares
about: the change in each teammate's goal-weighted residual space when the
chosen defender is replaced by its no-response baseline. That is the
repository's own formula, evaluated through :class:`~demo_viz.core.influence.InfluenceCache`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..quantities import (
    RunOnsetConfig,
    _kinematic_onsets,
    marking_series,
    peak_acceleration_time,
)
from ..scene import Scene
from .influence import InfluenceCache


@dataclass(frozen=True)
class DefenderCandidate:
    player_id: str
    shirt: str
    name: str
    score: float                 # 0..1, higher = more like a reaction
    mean_distance_m: float
    pursuit: float               # mean alignment toward the runner's goal-side point
    displacement_m: float        # how far the defender travels after the run starts

    def caption(self) -> str:
        return f"{self.mean_distance_m:.0f} m · pursuit {self.pursuit:.2f}"


@dataclass(frozen=True)
class BeneficiaryCandidate:
    player_id: str
    shirt: str
    name: str
    gain: float                  # mean space gained vs the no-response defender
    space_now: float             # residual space at the current frame
    gain_now: float

    def caption(self) -> str:
        return f"{self.gain:+.1f} avg · {self.gain_now:+.1f} now"


# ---------------------------------------------------------------------------
# when the run starts
# ---------------------------------------------------------------------------
def runner_onset_index(scene: Scene, runner_id: str) -> tuple[int, str]:
    """Frame index where the chosen runner's run begins, and how it was found."""

    onsets = _kinematic_onsets(scene, (runner_id,), RunOnsetConfig())
    if onsets:
        return scene.index_at(onsets[0]), "run onset detector"
    cue = peak_acceleration_time(scene, (runner_id,))
    if cue is not None:
        return scene.index_at(cue), "peak acceleration"
    return 0, "clip start"


def defender_reaction_index(
    scene: Scene,
    runner_id: str,
    defender_id: str,
    from_index: int = 0,
    minimum_speed_mps: float = 1.5,
    minimum_alignment: float = 0.30,
    sustain_seconds: float = 0.40,
) -> tuple[int | None, str]:
    """When the chosen defender commits to the chosen runner.

    The repository's kinematic onset detector is tried first; if the defender's
    motion is too smooth to trip it, the same geometric pursuit rule the video
    renderer uses is applied instead, and the caller is told which fired.
    """

    onsets = [t for t in _kinematic_onsets(scene, (defender_id,), RunOnsetConfig())
              if scene.index_at(t) >= from_index]
    if onsets:
        return scene.index_at(onsets[0]), "run onset detector"

    try:
        series = marking_series(scene, runner_id, defender_id)
    except KeyError:
        return None, ""
    defender = scene.players[defender_id]
    window = max(2, int(round(0.4 * scene.fps)))
    sustain = max(1, int(round(sustain_seconds * scene.fps)))
    run = 0
    for index in range(int(from_index), scene.n_frames):
        speed = float(np.hypot(*defender.velocity(index, scene.fps, window)))
        alignment = series["pursuit_alignment"][index]
        ok = speed >= minimum_speed_mps and np.isfinite(alignment) and alignment >= minimum_alignment
        run = run + 1 if ok else 0
        if run >= sustain:
            return index - sustain + 1, "pursuit rule"
    return None, ""


# ---------------------------------------------------------------------------
# defenders
# ---------------------------------------------------------------------------
def rank_defenders(
    scene: Scene,
    runner_id: str,
    from_index: int = 0,
    minimum_speed_mps: float = 1.0,
    limit: int | None = None,
) -> list[DefenderCandidate]:
    """Opponents ordered by how much they behave like a reaction to the runner."""

    if runner_id not in scene.players:
        return []
    window = max(2, int(round(0.4 * scene.fps)))
    from_index = int(np.clip(from_index, 0, max(scene.n_frames - 2, 0)))
    rows: list[DefenderCandidate] = []
    for player in scene.players_on("defend"):
        if player.is_goalkeeper:
            continue
        try:
            series = marking_series(scene, runner_id, player.player_id)
        except KeyError:
            continue
        distance = series["actor_defender_distance_m"][from_index:]
        alignment = series["pursuit_alignment"][from_index:]
        speed = np.array([
            float(np.hypot(*player.velocity(i, scene.fps, window)))
            for i in range(from_index, scene.n_frames)
        ])
        moving = speed >= minimum_speed_mps
        pursuit = float(np.nanmean(alignment[moving])) if moving.any() else 0.0
        if not np.isfinite(pursuit):
            pursuit = 0.0
        mean_distance = float(np.nanmean(distance)) if len(distance) else np.inf
        if not np.isfinite(mean_distance):
            continue
        track = player.xy[from_index:]
        finite = track[np.all(np.isfinite(track), axis=1)]
        displacement = (
            float(np.linalg.norm(finite[-1] - finite[0])) if len(finite) > 1 else 0.0
        )
        proximity = float(np.clip(1.0 - mean_distance / 25.0, 0.0, 1.0))
        score = float(np.clip(0.55 * max(pursuit, 0.0) + 0.45 * proximity, 0.0, 1.0))
        rows.append(
            DefenderCandidate(
                player_id=player.player_id,
                shirt=str(player.label),
                name=player.name,
                score=score,
                mean_distance_m=mean_distance,
                pursuit=pursuit,
                displacement_m=displacement,
            )
        )
    rows.sort(key=lambda row: row.score, reverse=True)
    return rows[:limit] if limit else rows


# ---------------------------------------------------------------------------
# beneficiaries
# ---------------------------------------------------------------------------
def rank_beneficiaries(
    cache: InfluenceCache,
    runner_id: str,
    defender_ids: Sequence[str],
    freeze_index: int,
    slot: int,
    baseline: str = "hold",
    limit: int | None = None,
) -> list[BeneficiaryCandidate]:
    """Teammates ordered by the space they gain from the defenders' movement."""

    scene = cache.scene
    if not defender_ids:
        return []
    swap = tuple((defender_id, int(freeze_index), baseline) for defender_id in defender_ids)
    after = cache.indices >= int(freeze_index)
    if not after.any():
        after = np.ones(len(cache.indices), dtype=bool)

    rows: list[BeneficiaryCandidate] = []
    for player in scene.players_on("attack"):
        if player.is_goalkeeper or player.player_id == runner_id:
            continue
        factual = cache.residual_series(player.player_id)
        counter = cache.residual_series(player.player_id, swap)
        gain = float(np.mean(factual[after] - counter[after]))
        rows.append(
            BeneficiaryCandidate(
                player_id=player.player_id,
                shirt=str(player.label),
                name=player.name,
                gain=gain,
                space_now=float(factual[slot]),
                gain_now=float(factual[slot] - counter[slot]),
            )
        )
    rows.sort(key=lambda row: row.gain, reverse=True)
    return rows[:limit] if limit else rows


def auto_triplet(
    scene: Scene,
    cache: InfluenceCache,
    runner_id: str | None = None,
    slot: int | None = None,
    baseline: str = "hold",
    defender_pool: int = 3,
) -> dict[str, object]:
    """The app's own suggestion for a triplet.

    With a ``runner_id`` it answers "given this runner, who reacts and who
    gains". Without one it searches every attacker and returns the triplet with
    the largest space gain, which is the quantity the demo is actually about --
    a longest-run heuristic picks whoever sprinted furthest, which is often
    nobody interesting.

    This is the app's own geometry, not a research pipeline prediction, and the
    UI labels it ``suggested``.
    """

    runners = [runner_id] if runner_id else [
        p.player_id for p in scene.players_on("attack") if not p.is_goalkeeper
    ]
    best: dict[str, object] = {}
    best_gain = -np.inf
    for candidate_runner in runners:
        freeze_index, method = runner_onset_index(scene, candidate_runner)
        use_slot = cache.slot_for(freeze_index) if slot is None else slot
        defenders = rank_defenders(
            scene, candidate_runner, from_index=freeze_index, limit=defender_pool
        )
        if not defenders:
            continue
        for defender in defenders:
            gainers = rank_beneficiaries(
                cache, candidate_runner, (defender.player_id,), freeze_index,
                use_slot, baseline, limit=1,
            )
            if not gainers:
                continue
            gain = gainers[0].gain
            if gain > best_gain:
                best_gain = gain
                best = {
                    "runner": candidate_runner,
                    "defender": defender.player_id,
                    "beneficiary": gainers[0].player_id,
                    "onset_index": freeze_index,
                    "onset_method": method,
                    "gain": round(float(gain), 3),
                }
    return best
