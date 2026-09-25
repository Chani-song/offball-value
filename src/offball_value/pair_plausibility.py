"""Is this (runner, defender) pair one the run actually implicates?

The candidate list is the top three defenders by goal-side marking distance
at onset, always three. Rank 1 is the reacting defender in every labelled
scene; ranks 2 and 3 are filler often enough that a reviewer's first
question is "why him?". A pair that makes no sense should not get a local
game built, let alone reach the dilemma solver.

What decides responsibility is where the RUN goes, not where the runner
stands at onset. A defender behind the runner can still be the one who cuts
the run off; a defender marking someone else leaves him if the run passes
right by; the man pressing the ball turns if the run goes in behind him.
So the features here are all distances from the defender's onset position
to the runner's track over the horizon -- the run is the move being
responded to, so reading it forward is fair. The defender's own movement is
the response and is never used.

A candidate is kept if he is the pipeline's first choice, or if he stood
within `near_m` of the runner at onset AND one of the two run-path tests
passes:

  rank1        first by goal-side marking distance (58/58 in review)
  neighbour    the run passes no more than `neighbour_delta_m` further from
               him than from the closest candidate -- he is standing next
               to the man who is marking, not somewhere else
  run_arrives  the run comes to him: he can reach the runner's position at
               some moment with `catch_margin_m` to spare, at a reacting
               6 m/s after 0.3 s, and the run finishes within `catch_end_m`

One candidate is refused whatever the run path says: the defender nearest
the ball, when the run does not go past him toward the goal he defends. His
job is the man on the ball, and a run that stays in front of him never makes
him choose -- it is someone else's to track. He is kept when the run does go
in behind, because then he has to turn and the ball is no longer his
problem alone. Rank 1 is exempt, as everywhere else. Reviewed candidates
bear this out: a nearest-to-ball defender with the run in behind him was
called responsible 4 times out of 4, while one with the run in front of him
split 2 right, 2 unsure, 2 wrong.

`near_m` is the one thing measured at onset rather than over the run, and
it is there because of what the review showed: every candidate the reviewer
rejected while the gate kept him stood a median 13.9 m from the runner at
onset, against 6.1 m for the ones he accepted. A defender that far out is
the covering line. The run may well reach him, but only after the man who
was actually responsible has been beaten, and that is a later problem than
the one being modelled.

Scored against 188 candidate verdicts over 58 scenes (2026-09-21): every
candidate the reviewer called responsible is kept, and 2 of the survivors
are ones he rejected. At 15 m the near test costs no accepted candidate;
tightening below that trades one accepted for one rejected, which is the
wrong way round -- a spurious candidate is a scene the reviewer skips, a
missing one is a dilemma solved against the wrong man. The ball-nearest
rule was added on 2026-09-22 after the highest-scoring stage-3 dilemma
turned out to be a defender 4.7 m from the ball, kept only because the
runner happened to move across the spot he had been standing on.

On the v7 build: 1,931 -> 993 candidates (51%), 1.69 per scene instead of
3.33, 22 of the 23 expert-labelled defenders kept. The one lost is a
covering defender 16.5 m from the run who could not have reached it;
keeping him means keeping nearly everything.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class PairGateConfig:
    neighbour_delta_m: float = 3.0
    catch_margin_m: float = 3.0
    catch_end_m: float = 11.0
    near_m: float = 15.0
    drop_ball_nearest_in_front: bool = True
    defender_speed_mps: float = 6.0
    reaction_s: float = 0.3
    keep_rank1: bool = True


def onset_frame(frames: Sequence[dict], onset_frame_id: int) -> dict:
    return min(frames, key=lambda f: abs(int(f["frame_id"]) - int(onset_frame_id)))


def run_track(frames: Sequence[dict], runner_id: str, onset_frame_id: int,
              horizon_s: float) -> list[tuple[float, float, float]]:
    """(t, x, y) of the runner from onset to onset + horizon, t in seconds."""
    t0 = float(onset_frame(frames, onset_frame_id)["relative_time_s"])
    out = []
    for f in frames:
        t = float(f["relative_time_s"]) - t0
        if t < -1e-9 or t > horizon_s + 1e-9:
            continue
        for p in f["players"]:
            if str(p[0]) == str(runner_id):
                out.append((t, float(p[2]), float(p[3])))
                break
    return out


def pair_features(track: Sequence[tuple[float, float, float]],
                  defender_xy: tuple[float, float],
                  cfg: PairGateConfig) -> dict:
    dx, dy = defender_xy
    d = [(t, math.hypot(x - dx, y - dy)) for t, x, y in track]
    return {
        "near_m": d[0][1],
        "path_min_m": min(dd for _, dd in d),
        "path_end_m": d[-1][1],
        "catch_m": max(cfg.defender_speed_mps * max(t - cfg.reaction_s, 0.0) - dd
                       for t, dd in d),
    }


def goal_side_of_runner(track: Sequence[tuple[float, float, float]],
                        defender_xy: tuple[float, float],
                        goal_xy: tuple[float, float]) -> float:
    """Metres the defender stands goal-side of the runner, at onset.

    Positive means he is between the run and the goal it threatens.
    """
    rx, ry = track[0][1], track[0][2]
    span = math.hypot(goal_xy[0] - rx, goal_xy[1] - ry)
    if span <= 1e-9:
        return 0.0
    ux, uy = (goal_xy[0] - rx) / span, (goal_xy[1] - ry) / span
    return (defender_xy[0] - rx) * ux + (defender_xy[1] - ry) * uy


def nearest_to_ball(positions: dict[str, tuple[float, float]],
                    ball_xy: tuple[float, float]) -> str | None:
    """Whichever defender is closest to the ball, over ALL of them."""
    if not positions:
        return None
    return min(positions,
               key=lambda pid: math.hypot(positions[pid][0] - ball_xy[0],
                                          positions[pid][1] - ball_xy[1]))


def gate_pairs(track: Sequence[tuple[float, float, float]],
               candidates: Sequence[tuple[str, tuple[float, float], int]],
               cfg: PairGateConfig,
               *,
               goal_xy: tuple[float, float] | None = None,
               ball_nearest_id: str | None = None) -> list[dict]:
    """candidates: (defender_id, onset xy, pipeline rank starting at 0).

    `ball_nearest_id` is the defender closest to the ball among ALL of them,
    not just the candidates, so the caller computes it. Given it and
    `goal_xy`, the ball-nearest rule applies; without them it is skipped and
    the gate behaves as it did before that rule existed.
    """
    rows = []
    for did, xy, rank in candidates:
        row = {"defender_id": did, "rank": rank, **pair_features(track, xy, cfg)}
        row["goal_side_m"] = (goal_side_of_runner(track, xy, goal_xy)
                              if goal_xy is not None else float("nan"))
        row["is_ball_nearest"] = did == ball_nearest_id
        rows.append(row)
    if not rows:
        return rows
    best = min(r["path_min_m"] for r in rows)
    for r in rows:
        r["delta_m"] = r["path_min_m"] - best
        if cfg.keep_rank1 and r["rank"] == 0:
            reason = "rank1"
        elif (cfg.drop_ball_nearest_in_front and r["is_ball_nearest"]
              and goal_xy is not None and r["goal_side_m"] < 0.0):
            reason = ""
        elif r["near_m"] > cfg.near_m:
            reason = ""
        elif r["delta_m"] <= cfg.neighbour_delta_m:
            reason = "neighbour"
        elif (r["catch_m"] >= cfg.catch_margin_m
              and r["path_end_m"] <= cfg.catch_end_m):
            reason = "run_arrives"
        else:
            reason = ""
        r["kept"] = bool(reason)
        r["reason"] = reason
    return rows


def gate_game(game: dict, cfg: PairGateConfig = PairGateConfig()) -> list[dict]:
    """Apply the gate to one local_game_payoff_audits.json game payload."""
    frames = game["background_frames"]
    track = run_track(frames, game["runner_id"], game["onset_frame_id"],
                      float(game["horizon_seconds"]))
    if len(track) < 3:
        return []
    at = onset_frame(frames, game["onset_frame_id"])
    pos = {str(p[0]): (float(p[2]), float(p[3])) for p in at["players"]}
    attacking = str(game["attacking_team_id"])
    defenders = {str(p[0]): (float(p[2]), float(p[3]))
                 for p in at["players"] if str(p[1]) != attacking}
    ball = (float(at["ball"][0]), float(at["ball"][1]))
    goal = (float(game["goal_xy"][0]), float(game["goal_xy"][1]))
    candidates = [
        (str(d["defender_id"]), pos[str(d["defender_id"])], rank)
        for rank, d in enumerate(game.get("candidate_defenders") or [])
        if str(d["defender_id"]) in pos
    ]
    return gate_pairs(track, candidates, cfg, goal_xy=goal,
                      ball_nearest_id=nearest_to_ball(defenders, ball))

# --- Whose move is it? -------------------------------------------------------
#
# Stage 1 flags a player whose movement changes, and everything downstream
# takes him as the author of whatever bind follows. That fails when the
# author is the man on the ball. A carrier who dribbles at a defender pulls
# him off whoever is nearby, and that nearby player -- adjusting his position
# a few metres sideways -- is what stage 1 saw. The solver then finds a real
# dilemma (ball or man), but the story "his off-ball run created it" is the
# wrong way round: the run is the passenger, the dribble is the driver.
#
# Reviewed on 2026-09-22 from the top-ranked stage-3 dilemma: the carrier
# covered 10.3 m in two seconds while the flagged runner covered 7.1 m, only
# 1.8 m of it toward goal, and the defender closed on the ball (5.8 -> 3.5 m)
# while opening on the runner (9.7 -> 14.4 m). That pattern -- carrier moves
# more, runner makes almost no ground toward goal -- is 18% of the 588
# scenes. It is refused here.
#
# "Covers more ground" needs a margin. Without one the rule dropped two of the
# eighteen expert-labelled scenes: one where the runner went 10.9 m sideways
# and the carrier 12.9 m (1.18x), one where they went 6.9 m and 7.1 m (1.03x,
# a tie). Neither is a carrier driving the play; both are two players moving
# together. The reviewed case was 1.45x. The carrier must cover at least
# `min_carrier_ratio` times the runner's ground -- 1.3, the middle of the
# band from 1.2 (lowest that keeps every labelled scene) to 1.4 (highest that
# still catches the reviewed one), so it is not set to the edge of either.
#
# Deliberately narrow. A run that barely advances is not wrong on its own: a
# run that drags a defender wide, or comes short to receive, creates space
# without gaining ground. What marks the scene as not-a-run-dilemma is the
# two together: the runner goes nowhere AND someone else is doing the moving.
# Reads the runner's and the carrier's movement, which are the attack's
# action; never the defender's, which is the response.


@dataclass(frozen=True)
class ProtagonistConfig:
    horizon_s: float = 2.0
    min_goalward_m: float = 2.0
    min_carrier_ratio: float = 1.3


def _displacement(frames, pid: str, onset_frame_id: int, horizon_s: float,
                  goal_xy: tuple[float, float]) -> tuple[float, float] | None:
    """(straight-line distance, metres gained toward goal) over the horizon."""
    track = run_track(frames, pid, onset_frame_id, horizon_s)
    if len(track) < 2:
        return None
    (_, x0, y0), (_, x1, y1) = track[0], track[-1]
    span = math.hypot(goal_xy[0] - x0, goal_xy[1] - y0)
    if span <= 1e-9:
        return math.hypot(x1 - x0, y1 - y0), 0.0
    ux, uy = (goal_xy[0] - x0) / span, (goal_xy[1] - y0) / span
    return math.hypot(x1 - x0, y1 - y0), (x1 - x0) * ux + (y1 - y0) * uy


def runner_is_protagonist(game: dict,
                          cfg: ProtagonistConfig = ProtagonistConfig()) -> dict:
    """Is the flagged runner the one whose move makes this scene?

    False only when the carrier covers at least `min_carrier_ratio` times the
    runner's ground AND the runner gains less than `min_goalward_m` toward goal. Missing tracks give
    True: the rule refuses on evidence, never on its absence.
    """
    frames = game["background_frames"]
    onset = game["onset_frame_id"]
    goal = (float(game["goal_xy"][0]), float(game["goal_xy"][1]))
    runner = _displacement(frames, str(game["runner_id"]), onset, cfg.horizon_s, goal)
    carrier = _displacement(frames, str(game["carrier_id"]), onset, cfg.horizon_s, goal)
    out = {
        "runner_disp_m": runner[0] if runner else float("nan"),
        "runner_goalward_m": runner[1] if runner else float("nan"),
        "carrier_disp_m": carrier[0] if carrier else float("nan"),
        "carrier_goalward_m": carrier[1] if carrier else float("nan"),
    }
    if runner is None or carrier is None or str(game["runner_id"]) == str(game["carrier_id"]):
        out["protagonist"] = True
        return out
    out["protagonist"] = not (carrier[0] >= cfg.min_carrier_ratio * runner[0]
                              and runner[1] < cfg.min_goalward_m)
    return out
