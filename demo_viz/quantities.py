"""Optional quantities, computed only from methods this repository implements.

Nothing in this module invents a number.  Each output carries the name of the
repository function that produced it, and anything the repository cannot
currently compute (calibrated xT, pass/dribble probability, a learned defensive
best response) is left as ``None`` so the renderer simply omits that layer.

What *is* available and used here:

``offball_value.run_onset.detect_kinematic_run_onsets``
    acceleration / direction-change / check-run onset frames for one track.
``offball_value.fernandez_influence.fernandez_influence_surface``
    Fernandez & Bornn (2018) player influence.
``offball_value.goal_weighted_influence.target_residual_influence``
    goal-weighted space an attacker retains after defensive coverage.  The
    repository documents this as a transparent v0.1 operationalization, not a
    calibrated threat model, and it is labelled that way on screen.

Two counterfactual devices are offered, and both re-evaluate the *same*
repository formula with only the reacting defenders' positions changed:

``hold``   the defender is frozen at their pre-onset position.
``drift``  the defender keeps the velocity they had just before the run onset.

Neither is a learned or optimised defensive best response, and the renderer
says so in frame.  ``hold`` is the default because it is the question the demo
poses out loud ("what if the defender had stayed?") and it is what the ghost
marker depicts.  ``drift`` exists to test how much the answer depends on that
choice; over a long window a held defender does not merely fail to react, it
drops out of the phase entirely.  ``demo_viz/OVERNIGHT_REPORT.md`` reports both
for all five annotated 'strong' scenes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from .config import ensure_repo_on_path
from .scene import Scene

ensure_repo_on_path()

from offball_value.bundesliga import (  # noqa: E402
    BundesligaFrame,
    BundesligaObjectState,
)
from offball_value.goal_weighted_influence import (  # noqa: E402
    GoalWeightedInfluenceConfig,
    influence_pitch_grid,
    target_residual_influence,
)
from offball_value.pass_dynamics import VelocityEstimate  # noqa: E402
from offball_value.run_onset import (  # noqa: E402
    RunOnsetConfig,
    detect_kinematic_run_onsets,
)

PROVENANCE = {
    "influence": "offball_value.fernandez_influence.fernandez_influence_surface "
    "(Fernandez & Bornn 2018)",
    "residual": "offball_value.goal_weighted_influence.target_residual_influence (repo v0.1)",
    "onset": "offball_value.run_onset.detect_kinematic_run_onsets (repo v0.3)",
    "counterfactual": "same residual formula with the reacting defenders replaced by "
    "a no-response baseline (explanatory device, not a learned best response)",
}


# ---------------------------------------------------------------------------
# frame adaptation
# ---------------------------------------------------------------------------
def _frame_at(
    scene: Scene,
    index: int,
    replace_xy: Mapping[str, np.ndarray] | None = None,
) -> BundesligaFrame:
    """Build the repository frame object the science functions expect.

    ``replace_xy`` maps a player id to a full ``(T, 2)`` baseline track, used to
    substitute the counterfactual defenders without touching anyone else.
    """

    players: dict[str, BundesligaObjectState] = {}
    for player in scene.players.values():
        xy = player.xy[index]
        if replace_xy is not None and player.player_id in replace_xy:
            xy = np.asarray(replace_xy[player.player_id][index], dtype=float)
        if not np.all(np.isfinite(xy)):
            continue
        players[player.player_id] = BundesligaObjectState(
            object_id=player.player_id,
            team_id=player.team_id,
            x=float(xy[0]),
            y=float(xy[1]),
        )
    ball = scene.ball_xy[index]
    ball_state = None
    if np.all(np.isfinite(ball)):
        ball_state = BundesligaObjectState(
            object_id="BALL", team_id="BALL", x=float(ball[0]), y=float(ball[1])
        )
    return BundesligaFrame(
        match_id=str(scene.provenance.get("match_id", "")),
        frame_id=int(scene.frame_ids[index]) if scene.frame_ids is not None else index,
        period=int(scene.provenance.get("period", 1)),
        game_section="",
        timestamp=None,
        players=players,
        ball=ball_state,
    )


def _velocities(
    scene: Scene,
    index: int,
    baseline: Mapping[str, np.ndarray] | None = None,
    baseline_velocity: Mapping[str, tuple[float, float]] | None = None,
) -> dict[str, VelocityEstimate]:
    window = max(2, int(round(0.4 * scene.fps)))
    out: dict[str, VelocityEstimate] = {}
    for player in scene.players.values():
        if baseline is not None and player.player_id in baseline:
            vx, vy = (baseline_velocity or {}).get(player.player_id, (0.0, 0.0))
            out[player.player_id] = VelocityEstimate(
                vx=float(vx), vy=float(vy), speed=float(np.hypot(vx, vy)),
                sample_count=window, window_seconds=window / scene.fps,
            )
            continue
        vx, vy = player.velocity(index, scene.fps, window)
        out[player.player_id] = VelocityEstimate(
            vx=vx,
            vy=vy,
            speed=float(np.hypot(vx, vy)),
            sample_count=window,
            window_seconds=window / scene.fps,
        )
    return out


# ---------------------------------------------------------------------------
# story moments from the repository onset detector
# ---------------------------------------------------------------------------
def _kinematic_onsets(
    scene: Scene, player_ids: Sequence[str], config: RunOnsetConfig
) -> list[float]:
    """All repository-detected kinematic onset times for the given players."""

    if scene.frame_ids is None:
        return []
    frames_all = np.asarray(scene.frame_ids, dtype=int)
    times: list[float] = []
    for player_id in player_ids:
        player = scene.players.get(player_id)
        if player is None:
            continue
        mask = np.all(np.isfinite(player.xy), axis=1)
        if mask.sum() < 3:
            continue
        frames = frames_all[mask]
        order = np.argsort(frames)
        try:
            onsets = detect_kinematic_run_onsets(
                frames[order], player.xy[mask][order, 0], player.xy[mask][order, 1], config
            )
        except ValueError:
            continue
        for onset in onsets:
            where = np.flatnonzero(frames_all == onset.frame_id)
            if len(where):
                times.append(float(scene.times[where[0]]))
    return sorted(times)


def peak_acceleration_time(
    scene: Scene, player_ids: Sequence[str], search_fraction: float = 0.65
) -> float | None:
    """Fallback run-start cue: the runner's strongest sustained acceleration.

    Used only when the repository's onset detector reports nothing inside the
    clip window.  This is a *presentation* cue for where to start the story, not
    a detected run onset, and the renderer labels it accordingly.
    """

    window = max(2, int(round(0.4 * scene.fps)))
    best_time, best_value = None, 0.0
    for player_id in player_ids:
        player = scene.players.get(player_id)
        if player is None or scene.n_frames < 3:
            continue
        speed = np.array(
            [float(np.hypot(*player.velocity(i, scene.fps, window))) for i in range(scene.n_frames)]
        )
        smooth = np.convolve(speed, np.ones(window) / window, mode="same")
        accel = np.gradient(smooth) * scene.fps
        # Ignore the padded convolution edges.
        edge = window
        if scene.n_frames <= 2 * edge + 1:
            continue
        # Only look in the early part of the window: a cue found in the last
        # second leaves the rest of the story no time to play.
        limit = max(edge + 1, int(scene.n_frames * float(search_fraction)))
        interior = accel[edge:limit]
        if not len(interior):
            continue
        index = int(np.argmax(interior)) + edge
        if accel[index] > best_value:
            best_value, best_time = float(accel[index]), float(scene.times[index])
    return best_time


def detect_moments(
    scene: Scene, config: RunOnsetConfig = RunOnsetConfig()
) -> tuple[dict[str, float], dict[str, str]]:
    """Return story-beat times (seconds on the scene clock) and how each was found.

    ``onset``     first kinematic run onset of any annotated runner
                  (``offball_value.run_onset``), else a labelled acceleration cue.
    ``reaction``  first kinematic onset of an annotated reacting defender at or
                  after the runner onset, else the geometric pursuit rule below.
    """

    moments: dict[str, float] = {}
    methods: dict[str, str] = {}

    runner_onsets = _kinematic_onsets(scene, scene.runner_ids, config)
    if runner_onsets:
        moments["onset"] = runner_onsets[0]
        methods["onset"] = "run_onset.detect_kinematic_run_onsets"
    else:
        fallback = peak_acceleration_time(scene, scene.runner_ids)
        if fallback is not None:
            moments["onset"] = fallback
            methods["onset"] = "demo_viz peak-acceleration cue (no detector onset in window)"

    onset = moments.get("onset")
    defender_onsets = [t for t in _kinematic_onsets(scene, scene.defender_ids, config)
                       if onset is None or t >= onset - 1e-6]
    if defender_onsets:
        moments["reaction"] = defender_onsets[0]
        methods["reaction"] = "run_onset.detect_kinematic_run_onsets"
    else:
        reaction, method = detect_defender_response(scene, not_before=onset)
        if reaction is not None:
            moments["reaction"] = reaction
            methods["reaction"] = "demo_viz " + method + " on dynamic_marking goal-side target"
    return moments, methods


# ---------------------------------------------------------------------------
# defender response timing
# ---------------------------------------------------------------------------
def marking_series(scene: Scene, runner_id: str, defender_id: str) -> dict[str, np.ndarray]:
    """Per-frame goal-side marking geometry, from ``offball_value.dynamic_marking``.

    ``weighted_error_m`` is the repository's dynamic-marking cost: how far the
    defender is from the moving goal-side point in front of the runner, with a
    penalty for being on the wrong side of the runner.  A falling curve means
    the defender is committing to the runner.
    """

    from offball_value.dynamic_marking import (
        DynamicMarkingConfig,
        marking_sample,
        moving_goal_side_target,
    )

    config = DynamicMarkingConfig()
    runner = scene.players[runner_id]
    defender = scene.players[defender_id]
    goal_xy = (-scene.attacking_direction * scene.pitch_length / 2.0, 0.0)
    lookahead = config.lookahead_seconds
    window = max(2, int(round(0.4 * scene.fps)))

    weighted = np.full(scene.n_frames, np.nan)
    distance = np.full(scene.n_frames, np.nan)
    alignment = np.full(scene.n_frames, np.nan)
    for index in range(scene.n_frames):
        actor = runner.xy[index]
        marker = defender.xy[index]
        if not (np.all(np.isfinite(actor)) and np.all(np.isfinite(marker))):
            continue
        rvx, rvy = runner.velocity(index, scene.fps, window)
        anticipated = (actor[0] + rvx * lookahead, actor[1] + rvy * lookahead)
        sample = marking_sample(
            scene.times[index],
            (float(actor[0]), float(actor[1])),
            anticipated,
            (float(marker[0]), float(marker[1])),
            goal_xy,
            config,
        )
        weighted[index] = sample.weighted_error_m
        distance[index] = sample.actor_defender_distance_m
        target = moving_goal_side_target(anticipated, goal_xy, config.goal_side_offset_m)
        dvx, dvy = defender.velocity(index, scene.fps, window)
        speed = float(np.hypot(dvx, dvy))
        to_target = np.array([target[0] - marker[0], target[1] - marker[1]])
        norm = float(np.linalg.norm(to_target))
        if speed > 1e-6 and norm > 1e-6:
            alignment[index] = float((dvx * to_target[0] + dvy * to_target[1]) / (speed * norm))
    return {
        "weighted_error_m": weighted,
        "actor_defender_distance_m": distance,
        "pursuit_alignment": alignment,
    }


def detect_defender_response(
    scene: Scene,
    not_before: float | None = None,
    minimum_speed_mps: float = 1.5,
    minimum_alignment: float = 0.30,
    sustain_seconds: float = 0.40,
) -> tuple[float | None, str]:
    """First time the reacting defender commits to tracking the runner.

    The repository's kinematic onset detector is tried first (it is the same
    detector used for runners).  When a defender's motion is too smooth to trip
    those thresholds, this geometric rule is used instead: the defender is
    moving faster than ``minimum_speed_mps`` *toward* the moving goal-side point
    in front of the runner for ``sustain_seconds``.  The goal-side target comes
    from ``offball_value.dynamic_marking.moving_goal_side_target``; the timing
    rule itself is a demo_viz presentation heuristic and is labelled as such.
    """

    if not scene.runner_ids or not scene.defender_ids:
        return None, "unavailable"
    sustain = max(1, int(round(sustain_seconds * scene.fps)))
    best: float | None = None
    for defender_id in scene.defender_ids:
        series = marking_series(scene, scene.runner_ids[0], defender_id)
        defender = scene.players[defender_id]
        window = max(2, int(round(0.4 * scene.fps)))
        speed = np.array(
            [float(np.hypot(*defender.velocity(i, scene.fps, window))) for i in range(scene.n_frames)]
        )
        ok = (speed >= minimum_speed_mps) & (series["pursuit_alignment"] >= minimum_alignment)
        if not_before is not None:
            ok &= scene.times >= (not_before - 1e-6)
        run = 0
        for index, flag in enumerate(ok):
            run = run + 1 if flag else 0
            if run >= sustain:
                candidate = float(scene.times[index - sustain + 1])
                if best is None or candidate < best:
                    best = candidate
                break
    return best, "geometric_pursuit_rule"


# ---------------------------------------------------------------------------
# space / residual value
# ---------------------------------------------------------------------------
@dataclass
class SurfaceStack:
    """Sampled goal-weighted residual space surfaces over the scene window."""

    indices: np.ndarray                  # (K,) frame indices into the scene
    times: np.ndarray                    # (K,)
    xgrid: np.ndarray
    ygrid: np.ndarray
    factual: np.ndarray                  # (K, ny, nx) beneficiary residual space
    counterfactual: np.ndarray | None    # same, defenders held pre-onset
    factual_value: np.ndarray | None = None          # (K,) integrated, m^2-weighted
    counterfactual_value: np.ndarray | None = None   # (K,)
    freeze_index: int | None = None
    freeze_positions: dict[str, tuple[float, float]] = field(default_factory=dict)
    baseline_tracks: dict[str, np.ndarray] = field(default_factory=dict)
    baseline_mode: str = "hold"
    provenance: dict[str, str] = field(default_factory=lambda: dict(PROVENANCE))

    @property
    def delta(self) -> np.ndarray | None:
        if self.counterfactual is None:
            return None
        return self.factual - self.counterfactual

    def nearest(self, index: int) -> int:
        return int(np.argmin(np.abs(self.indices - index)))

    def at(self, index: int) -> np.ndarray:
        return self.factual[self.nearest(index)]

    def delta_at(self, index: int) -> np.ndarray | None:
        delta = self.delta
        return None if delta is None else delta[self.nearest(index)]


def _residual(
    scene: Scene,
    index: int,
    target_ids: Sequence[str],
    xgrid: np.ndarray,
    ygrid: np.ndarray,
    config: GoalWeightedInfluenceConfig,
    baseline: Mapping[str, np.ndarray] | None = None,
    baseline_velocity: Mapping[str, tuple[float, float]] | None = None,
) -> tuple[np.ndarray, float]:
    frame = _frame_at(scene, index, baseline)
    velocities = _velocities(scene, index, baseline, baseline_velocity)
    keepers = tuple(p.player_id for p in scene.players.values() if p.is_goalkeeper)
    total = np.zeros((len(ygrid), len(xgrid)))
    value = 0.0
    for target_id in target_ids:
        if target_id not in frame.players:
            continue
        try:
            result = target_residual_influence(
                frame,
                velocities,
                target_id,
                scene.attacking_team_id,
                scene.attacking_direction,
                goalkeeper_ids=keepers,
                config=config,
                xgrid=xgrid,
                ygrid=ygrid,
            )
        except (KeyError, ValueError):
            continue
        total = np.maximum(total, result.residual_surface)
        value += result.residual_value
    return total, value


def baseline_tracks(
    scene: Scene,
    freeze_index: int,
    mode: str = "hold",
    velocity_window_s: float = 0.6,
    defender_ids: Sequence[str] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, tuple[float, float]]]:
    """No-response tracks for the reacting defenders.

    ``hold``  the defender stays exactly where they were at ``freeze_index``.
    ``drift`` the defender keeps the velocity they had just before that frame,
              clipped to the pitch.  Over a multi-second window this is the more
              honest 'did not react to the runner' baseline: a held defender
              simply drops out of a moving phase of play.

    Before ``freeze_index`` the baseline is the observed track, so the two
    curves start identical and separate only once the run has begun.

    ``defender_ids`` overrides the scene's annotated defenders, which is what
    the interactive app uses when the user picks a different one.
    """

    tracks: dict[str, np.ndarray] = {}
    velocities: dict[str, tuple[float, float]] = {}
    half_l, half_w = scene.pitch_length / 2.0, scene.pitch_width / 2.0
    window = max(2, int(round(velocity_window_s * scene.fps)))
    for player_id in (scene.defender_ids if defender_ids is None else defender_ids):
        player = scene.players.get(player_id)
        if player is None:
            continue
        anchor = player.xy[freeze_index]
        if not np.all(np.isfinite(anchor)):
            continue
        track = player.xy.copy()
        if mode == "hold":
            vx = vy = 0.0
        else:
            start = max(0, freeze_index - window)
            delta = player.xy[freeze_index] - player.xy[start]
            seconds = max((freeze_index - start) / scene.fps, 1e-6)
            vx, vy = float(delta[0] / seconds), float(delta[1] / seconds)
        steps = np.arange(scene.n_frames) - freeze_index
        future = steps > 0
        track[future, 0] = anchor[0] + vx * steps[future] / scene.fps
        track[future, 1] = anchor[1] + vy * steps[future] / scene.fps
        track[:, 0] = np.clip(track[:, 0], -half_l, half_l)
        track[:, 1] = np.clip(track[:, 1], -half_w, half_w)
        tracks[player_id] = track
        velocities[player_id] = (vx, vy)
    return tracks, velocities


def compute_surfaces(
    scene: Scene,
    every: int = 5,
    grid_resolution_m: float = 1.0,
    counterfactual: bool = True,
    freeze_time: float | None = None,
    freeze_at: str = "onset",
    baseline_mode: str = "hold",
) -> SurfaceStack:
    """Sample the beneficiary's residual goal-weighted space over the window.

    ``every`` subsamples the scene's frames; ``freeze_time`` is the scene time at
    which the reacting defenders stop responding in the counterfactual (default:
    the detected runner onset).
    """

    config = GoalWeightedInfluenceConfig(grid_resolution_m=grid_resolution_m)
    xgrid, ygrid = influence_pitch_grid(config)
    indices = np.arange(0, scene.n_frames, max(1, int(every)))
    if indices[-1] != scene.n_frames - 1:
        indices = np.append(indices, scene.n_frames - 1)

    targets = tuple(scene.beneficiary_ids) or tuple(
        p.player_id for p in scene.players_on("attack") if not p.is_goalkeeper
    )

    freeze_index = None
    tracks: dict[str, np.ndarray] = {}
    velocities: dict[str, tuple[float, float]] = {}
    if counterfactual and scene.defender_ids:
        t_freeze = freeze_time
        if t_freeze is None:
            if freeze_at == "start":
                t_freeze = float(scene.times[0])
            else:
                t_freeze = scene.moments.get(
                    freeze_at, scene.moments.get("onset", float(scene.times[0]))
                )
        freeze_index = scene.index_at(float(t_freeze))
        tracks, velocities = baseline_tracks(scene, freeze_index, baseline_mode)

    factual = np.zeros((len(indices), len(ygrid), len(xgrid)))
    counter = np.zeros_like(factual) if tracks else None
    factual_value = np.zeros(len(indices))
    counter_value = np.zeros(len(indices)) if counter is not None else None
    for slot, index in enumerate(indices):
        factual[slot], factual_value[slot] = _residual(
            scene, int(index), targets, xgrid, ygrid, config
        )
        if counter is not None:
            counter[slot], counter_value[slot] = _residual(
                scene, int(index), targets, xgrid, ygrid, config,
                baseline=tracks, baseline_velocity=velocities,
            )

    return SurfaceStack(
        indices=indices,
        times=scene.times[indices],
        xgrid=xgrid,
        ygrid=ygrid,
        factual=factual,
        counterfactual=counter,
        factual_value=factual_value,
        counterfactual_value=counter_value,
        freeze_index=freeze_index,
        freeze_positions={
            pid: (float(track[freeze_index][0]), float(track[freeze_index][1]))
            for pid, track in tracks.items()
        } if freeze_index is not None else {},
        baseline_tracks=tracks,
        baseline_mode=baseline_mode,
    )


def residual_series(
    scene: Scene,
    every: int = 5,
    grid_resolution_m: float = 1.5,
    player_ids: Sequence[str] | None = None,
) -> dict[str, np.ndarray]:
    """Scalar residual goal-weighted space per sampled frame, per player."""

    config = GoalWeightedInfluenceConfig(grid_resolution_m=grid_resolution_m)
    xgrid, ygrid = influence_pitch_grid(config)
    indices = np.arange(0, scene.n_frames, max(1, int(every)))
    if indices[-1] != scene.n_frames - 1:
        indices = np.append(indices, scene.n_frames - 1)
    targets = tuple(player_ids) if player_ids else (
        tuple(scene.beneficiary_ids) + tuple(scene.runner_ids)
    )
    out: dict[str, np.ndarray] = {"_t": scene.times[indices], "_index": indices.astype(float)}
    for target_id in targets:
        values = np.zeros(len(indices))
        for slot, index in enumerate(indices):
            _, values[slot] = _residual(
                scene, int(index), (target_id,), xgrid, ygrid, config
            )
        out[target_id] = values
    return out


def attach_quantities(
    scene: Scene,
    surfaces: bool = True,
    every: int = 5,
    grid_resolution_m: float = 1.0,
    series_resolution_m: float = 1.5,
    freeze_at: str = "onset",
    baseline_mode: str = "hold",
) -> Scene:
    """Fill in every optional quantity the repository can actually produce."""

    moments, methods = detect_moments(scene)
    scene.moments.update(moments)
    scene.provenance['moment_methods'] = methods
    series = residual_series(scene, every=every, grid_resolution_m=series_resolution_m)
    scene.series.update(series)
    if surfaces:
        scene.surfaces = compute_surfaces(
            scene, every=every, grid_resolution_m=grid_resolution_m,
            freeze_at=freeze_at, baseline_mode=baseline_mode,
        )
        scene.provenance["counterfactual"] = {
            "freeze_at": freeze_at, "baseline": baseline_mode,
        }
        stack = scene.surfaces
        delta = stack.delta
        if delta is not None and stack.counterfactual_value is not None:
            gained = np.clip(delta, 0.0, None).sum(axis=(1, 2))
            scene.series["_delta_gained"] = gained
            scene.series["_surface_t"] = stack.times
            scene.series["_factual_value"] = stack.factual_value
            scene.series["_counterfactual_value"] = stack.counterfactual_value
            difference = stack.factual_value - stack.counterfactual_value
            scene.series["_value_gain"] = difference
            # The story's 'space opens' beat is where the factual-vs-held gap
            # grows fastest, i.e. where the defender's movement starts paying off.
            if len(difference) > 3:
                rate = np.gradient(difference)
                peak = int(np.argmax(rate))
            else:
                peak = int(np.argmax(difference))
            scene.moments.setdefault("space", float(stack.times[peak]))
            scene.scores["value_gain_peak"] = float(np.max(difference))
    beneficiary_curves = [
        series[pid] for pid in scene.beneficiary_ids if pid in series
    ]
    if beneficiary_curves:
        total = np.sum(beneficiary_curves, axis=0)
        scene.scores["beneficiary_residual_peak"] = float(np.max(total))
        scene.scores["beneficiary_residual_start"] = float(total[0])

    # The reveal beat should land where the beneficiary gains the most *because
    # the defender moved*, not simply where they hold the most space.
    gain = scene.series.get("_value_gain")
    gain_times = scene.series.get("_surface_t")
    if gain is not None and gain_times is not None and len(gain) > 2:
        times = np.asarray(gain_times, dtype=float)
        usable = times <= times[-1] - 1.2
        reaction = scene.moments.get("reaction")
        if reaction is not None:
            usable &= times >= reaction - 1e-6
        if usable.sum() < 2:
            usable = times <= times[-1] - 1.2
        if usable.sum() >= 2:
            index = int(np.argmax(np.where(usable, gain, -np.inf)))
            scene.moments.setdefault("beneficiary", float(times[index]))
    scene.provenance["quantities"] = dict(PROVENANCE)
    return scene
