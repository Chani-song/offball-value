"""R3: beneficiary by VACATED COVERAGE and contested exploitation.

The reviewer's formulation (2026-09-08), in his words:

  "When a defender moves in reaction to an off-ball runner, that defender's
   influence area changes. His influence over part of the area he was
   covering shrinks, and the beneficiary is whoever can best exploit it."

and, on what "best exploit" means:

  "The nearest player does not have to be the beneficiary. Even if X1 is the
   closest, it is no use if his own marker can cover that area as well; if
   X2 is heading into that area faster than his marker, or has no marker at
   all, X2 exploits it better."

So the rule is two stages, both computed on the coverage field:

1. VACATED REGION. Take the defender's coverage before the run and after a
   purely geometric reaction to it, and keep the cells where the DEFENDING
   TEAM's coverage actually drops. Using the team maximum (not this
   defender alone) is what encodes "responsibility" — space a team-mate
   still holds is not vacated at all.

2. EXPLOITER. Score every attacker on that region by CONTESTED coverage:
   his own coverage of the region minus the best defender's coverage of it.
   An unmarked attacker wins on the second term; a marked attacker can still
   win by moving into the region faster than his marker, because coverage is
   velocity-oriented. Nearest-by-distance is used only to shortlist.

COVERAGE MODEL. Deliberately NOT the full Fernández influence: the reviewer
asked for "the area covered based on position and velocity, leaving out
the ball position". In Fernández the ball enters only through the radius
transform (4 -> 10 m with distance to ball), which also gives the ball
CARRIER the smallest zone of anyone on the pitch — structurally unable to
express "the carrier is driving at me". Here the radius is a fixed
constant, so the geometry is purely position+velocity.

STATUS: developed after the round-2 blind test was scored. Round 1 is its
development set; any round-2 number is a single declared post-hoc look.
Not wired into the pipeline.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np

from .fernandez_influence import FernandezInfluenceEllipse

# Fixed coverage radius (m). Fernández scales this 4 -> 10 m by distance to
# the ball; the reviewer explicitly wants the ball left out, so we pin it at
# the paper's upper end — the "far from the ball, full reach" geometry.
COVERAGE_RADIUS_M = 10.0
MAXIMUM_SPEED_MPS = 13.0
# Region grid: 1 m cells over a window around the defender's start.
REGION_HALF_WIDTH_M = 25.0
REGION_STEP_M = 1.0
# How many nearest attackers to score on the vacated region.
SHORTLIST = 5


def coverage_ellipse(
    player_xy: tuple[float, float],
    velocity_xy: tuple[float, float],
    radius_m: float = COVERAGE_RADIUS_M,
    maximum_speed_mps: float = MAXIMUM_SPEED_MPS,
) -> FernandezInfluenceEllipse:
    """Fernández ellipse geometry with a ball-independent fixed radius."""
    speed = math.hypot(*velocity_xy)
    ratio = min(speed / maximum_speed_mps, 1.0) ** 2
    major = (radius_m + radius_m * ratio) / 2.0
    minor = (radius_m - radius_m * ratio) / 2.0
    angle = math.degrees(math.atan2(velocity_xy[1], velocity_xy[0]))
    return FernandezInfluenceEllipse(
        center_x=player_xy[0] + velocity_xy[0] * 0.5,
        center_y=player_xy[1] + velocity_xy[1] * 0.5,
        major_radius_m=max(major, 1e-6),
        minor_radius_m=max(minor, 1e-6),
        angle_degrees=angle,
        base_radius_m=radius_m,
        distance_to_ball_m=float("nan"),
        speed_mps=speed,
        speed_ratio=ratio,
        contour_sigma=2.0,
    )


def coverage_field(
    player_xy: tuple[float, float],
    velocity_xy: tuple[float, float],
    xs: np.ndarray,
    ys: np.ndarray,
    radius_m: float = COVERAGE_RADIUS_M,
) -> np.ndarray:
    """Coverage in [0, 1] on a grid: 1 at the ellipse centre, decaying out."""
    ellipse = coverage_ellipse(player_xy, velocity_xy, radius_m)
    angle = math.radians(ellipse.angle_degrees)
    dx = xs[None, :] - ellipse.center_x
    dy = ys[:, None] - ellipse.center_y
    along = dx * math.cos(angle) + dy * math.sin(angle)
    lateral = -dx * math.sin(angle) + dy * math.cos(angle)
    normalized = np.sqrt(
        (along / ellipse.major_radius_m) ** 2
        + (lateral / ellipse.minor_radius_m) ** 2
    )
    # Two-sigma contour at normalized radius 1, matching the paper's display.
    return np.exp(-0.5 * (2.0 * normalized) ** 2)


def _state_xy(row: Mapping[str, object]) -> tuple[float, float]:
    return (float(row["x"]), float(row["y"]))


def _state_v(row: Mapping[str, object]) -> tuple[float, float]:
    return (float(row["vx"]), float(row["vy"]))


def vacated_region_trail(
    state: Mapping[str, Mapping[str, object]],
    attacking_team: str,
    defender_id: str,
    samples: Sequence[tuple[tuple[float, float], tuple[float, float]]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cumulative vacated region along the defender's whole reaction path.

    The reviewer's image: "like Hansel and Gretel dropping crumbs" — the
    space a reacting defender opens is not one snapshot but everything he
    leaves behind as he goes, widening with time. So the coverage loss is
    taken at every sampled instant of the reaction and accumulated by running
    maximum: a cell counts as vacated once it has been abandoned at any
    point, and the region grows along the trail.
    """
    xs, ys = _region_grid(state[defender_id])
    others = np.zeros((len(ys), len(xs)))
    for pid, row in state.items():
        if str(row["team"]) == attacking_team or pid == defender_id:
            continue
        np.maximum(
            others, coverage_field(_state_xy(row), _state_v(row), xs, ys), out=others
        )
    before = np.maximum(
        others,
        coverage_field(
            _state_xy(state[defender_id]), _state_v(state[defender_id]), xs, ys
        ),
    )
    trail = np.zeros_like(before)
    for after_xy, after_v in samples:
        after = np.maximum(others, coverage_field(after_xy, after_v, xs, ys))
        np.maximum(trail, np.clip(before - after, 0.0, None), out=trail)
    return xs, ys, trail


def _region_grid(
    defender_row: Mapping[str, object],
) -> tuple[np.ndarray, np.ndarray]:
    start = _state_xy(defender_row)
    return (
        np.arange(
            start[0] - REGION_HALF_WIDTH_M,
            start[0] + REGION_HALF_WIDTH_M + 1e-9,
            REGION_STEP_M,
        ),
        np.arange(
            start[1] - REGION_HALF_WIDTH_M,
            start[1] + REGION_HALF_WIDTH_M + 1e-9,
            REGION_STEP_M,
        ),
    )


def vacated_region(
    state: Mapping[str, Mapping[str, object]],
    attacking_team: str,
    defender_id: str,
    defender_after_xy: tuple[float, float],
    defender_after_v: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(xs, ys, loss) where loss is the DEFENDING TEAM's coverage drop.

    Team maximum before minus team maximum after, clipped at zero: space a
    team-mate still holds never counts as vacated.
    """
    start = _state_xy(state[defender_id])
    xs = np.arange(
        start[0] - REGION_HALF_WIDTH_M,
        start[0] + REGION_HALF_WIDTH_M + 1e-9,
        REGION_STEP_M,
    )
    ys = np.arange(
        start[1] - REGION_HALF_WIDTH_M,
        start[1] + REGION_HALF_WIDTH_M + 1e-9,
        REGION_STEP_M,
    )
    others = [
        pid
        for pid, row in state.items()
        if str(row["team"]) != attacking_team and pid != defender_id
    ]
    team_others = np.zeros((len(ys), len(xs)))
    for pid in others:
        np.maximum(
            team_others,
            coverage_field(_state_xy(state[pid]), _state_v(state[pid]), xs, ys),
            out=team_others,
        )
    before = np.maximum(
        team_others,
        coverage_field(
            start, _state_v(state[defender_id]), xs, ys
        ),
    )
    after = np.maximum(
        team_others,
        coverage_field(defender_after_xy, defender_after_v, xs, ys),
    )
    return xs, ys, np.clip(before - after, 0.0, None)


def _time_to_region(
    player_xy: tuple[float, float],
    velocity_xy: tuple[float, float],
    xs: np.ndarray,
    ys: np.ndarray,
    loss: np.ndarray,
    max_speed_mps: float = 8.0,
    acceleration_mps2: float = 4.0,
) -> float:
    """Loss-weighted mean time for one player to reach the vacated region.

    Straight-line travel from rest-or-motion: the component of current
    velocity toward each cell counts, so a player already running into the
    space arrives sooner than a stationary player the same distance away.
    That asymmetry is the whole point — occupancy is not exploitation.
    """
    px, py = player_xy
    dx = xs[None, :] - px
    dy = ys[:, None] - py
    distance = np.hypot(dx, dy)
    safe = np.maximum(distance, 1e-9)
    toward = np.clip((dx * velocity_xy[0] + dy * velocity_xy[1]) / safe, 0.0, max_speed_mps)
    # Accelerate from the useful component of current speed up to top speed.
    to_top = np.maximum(max_speed_mps - toward, 0.0) / acceleration_mps2
    reach_while_accelerating = toward * to_top + 0.5 * acceleration_mps2 * to_top**2
    times = np.where(
        distance <= reach_while_accelerating,
        (np.sqrt(np.maximum(toward**2 + 2 * acceleration_mps2 * distance, 0.0)) - toward)
        / acceleration_mps2,
        to_top + (distance - reach_while_accelerating) / max_speed_mps,
    )
    # Earliest arrival into the region, not the average over it. A player
    # already standing inside is close to EVERY cell and so wins on the
    # mean, but occupying space you already stood in is not exploitation;
    # reaching any part of it first is. (Round-1 diagnosis: the mean kept
    # electing the defender's former mark over the man driving in.)
    if loss.sum() <= 1e-12:
        return float("inf")
    return float(times[loss > 0.05 * float(loss.max())].min())


def rule_r5(
    state: Mapping[str, Mapping[str, object]],
    attacking_team: str,
    runner_id: str,
    carrier_id: str,
    defender_id: str,
    samples: Sequence[tuple[tuple[float, float], tuple[float, float]]],
    attacker_at: "callable",
    shortlist: int = SHORTLIST,
) -> tuple[str, dict[str, float]]:
    """Beneficiary = who occupies the space AS it opens, over time.

    R4 scored attackers frozen at the onset instant, which cannot express
    the reviewer's 61802 reading: "as time passes, Iyoha moves away from
    that area and Klarer gets closer". A player 19 m away whose run is
    pointed into the opening space beats one standing beside it and
    drifting out — but only if BOTH sides are advanced through time.

    So at every sampled instant we recompute (a) the space this defender
    has vacated by then and (b) each attacker's coverage from where he
    actually is at that instant, and accumulate the contested product.

    ``attacker_at(player_id, fraction)`` returns that player's (xy,
    velocity), where FRACTION IS (index + 1) / len(samples) — a position in
    the window, NOT a time in seconds. Recovering the sample index from it
    therefore needs the -1; omitting that reads every attacker one step
    ahead of the defender he is compared against, which is precisely the
    bug found on 2026-09-09 (neutral on round 1, worth one round-2 scene).
    Callers that already have absolute times should pass a lookup that
    ignores the fraction convention entirely, as
    ``coupled_beneficiary.rule_r9`` does.
    """
    del carrier_id
    xs, ys = _region_grid(state[defender_id])
    others = np.zeros((len(ys), len(xs)))
    for pid, row in state.items():
        if str(row["team"]) == attacking_team or pid == defender_id:
            continue
        np.maximum(
            others, coverage_field(_state_xy(row), _state_v(row), xs, ys), out=others
        )
    before = np.maximum(
        others,
        coverage_field(
            _state_xy(state[defender_id]), _state_v(state[defender_id]), xs, ys
        ),
    )
    attackers = [
        pid
        for pid, row in state.items()
        if str(row["team"]) == attacking_team and pid != runner_id
    ]
    if not attackers:
        return "", {"_vacated_area": 0.0}

    totals: dict[str, float] = {pid: 0.0 for pid in attackers}
    vacated_total = 0.0
    for index, (after_xy, after_v) in enumerate(samples):
        loss = np.clip(
            before - np.maximum(others, coverage_field(after_xy, after_v, xs, ys)),
            0.0,
            None,
        )
        weight = float(loss.sum())
        if weight <= 1e-12:
            continue
        vacated_total += weight
        time_s = float(index + 1) / max(len(samples), 1)
        for pid in attackers:
            xy, vel = attacker_at(pid, time_s)
            mine = coverage_field(xy, vel, xs, ys)
            totals[pid] += float((loss * mine).sum())

    if vacated_total <= 1e-12:
        start = _state_xy(state[defender_id])
        nearest = min((math.dist(_state_xy(state[p]), start), p) for p in attackers)
        return nearest[1], {"_vacated_area": 0.0}

    detail = {pid: totals[pid] / vacated_total for pid in attackers}
    ranked = sorted(detail.items(), key=lambda kv: (-kv[1], kv[0]))
    best = ranked[0][0]
    detail["_vacated_area"] = vacated_total
    return best, detail


def rule_r4(
    state: Mapping[str, Mapping[str, object]],
    attacking_team: str,
    runner_id: str,
    carrier_id: str,
    defender_id: str,
    samples: Sequence[tuple[tuple[float, float], tuple[float, float]]],
    shortlist: int = SHORTLIST,
) -> tuple[str, dict[str, float]]:
    """Beneficiary = best contested COVERAGE of the cumulative vacated trail.

    Two corrections the reviewer made to the arrival-time version:

    1. Arrival time is the wrong exploitation measure. A player can be near
       the vacated space yet be carrying his momentum straight out of it,
       and a player can be further away yet be oriented into it. Coverage
       (position AND velocity, ball ignored) already encodes that; arrival
       time throws the direction away.
    2. The region must accumulate along the defender's path, not be read at
       one instant — see ``vacated_region_trail``.

    Score = sum over the trail of (my coverage - best defender coverage),
    weighted by how much each cell was vacated. Positive means the space is
    genuinely his to use.
    """
    del carrier_id  # the carrier competes on the same footing as anyone else
    xs, ys, loss = vacated_region_trail(
        state, attacking_team, defender_id, samples
    )
    total = float(loss.sum())
    attackers = [
        pid
        for pid, row in state.items()
        if str(row["team"]) == attacking_team and pid != runner_id
    ]
    if total <= 1e-12 or not attackers:
        start = _state_xy(state[defender_id])
        nearest = min(
            ((math.dist(_state_xy(state[p]), start), p) for p in attackers),
            default=(0.0, ""),
        )
        return nearest[1], {"_vacated_area": 0.0}

    weights = loss / total
    centroid = (
        float((weights.sum(axis=0) * xs).sum()),
        float((weights.sum(axis=1) * ys).sum()),
    )
    attackers.sort(
        key=lambda pid: (math.dist(_state_xy(state[pid]), centroid), pid)
    )
    candidates = attackers[:shortlist]

    # Defenders at the END of the reaction: that is the shape the attack has
    # to beat once the space has opened.
    final_xy, final_v = samples[-1]
    defence = np.zeros_like(loss)
    for pid, row in state.items():
        if str(row["team"]) == attacking_team:
            continue
        xy = final_xy if pid == defender_id else _state_xy(row)
        vel = final_v if pid == defender_id else _state_v(row)
        np.maximum(defence, coverage_field(xy, vel, xs, ys), out=defence)

    best_id, best_score, detail = "", -math.inf, {}
    for pid in candidates:
        mine = coverage_field(_state_xy(state[pid]), _state_v(state[pid]), xs, ys)
        score = float((loss * (mine - defence)).sum() / total)
        detail[pid] = score
        if score > best_score + 1e-12 or (
            abs(score - best_score) <= 1e-12 and pid < best_id
        ):
            best_id, best_score = pid, score
    detail["_vacated_area"] = total
    return best_id, detail


def rule_r3(
    state: Mapping[str, Mapping[str, object]],
    attacking_team: str,
    runner_id: str,
    carrier_id: str,
    defender_id: str,
    defender_after_xy: tuple[float, float],
    defender_after_v: tuple[float, float],
    shortlist: int = SHORTLIST,
) -> tuple[str, dict[str, float]]:
    """Beneficiary = best contested exploiter of the vacated region."""
    del carrier_id  # the carrier competes on the same footing as anyone else
    xs, ys, loss = vacated_region(
        state, attacking_team, defender_id, defender_after_xy, defender_after_v
    )
    total_loss = float(loss.sum())
    if total_loss <= 1e-12:
        # Nothing was vacated; fall back to the nearest non-runner attacker.
        start = _state_xy(state[defender_id])
        nearest = min(
            (
                (math.dist(_state_xy(row), start), pid)
                for pid, row in state.items()
                if str(row["team"]) == attacking_team and pid != runner_id
            ),
            default=(0.0, ""),
        )
        return nearest[1], {"vacated": 0.0}

    # Loss-weighted centroid, used only to shortlist candidates by distance.
    weights = loss / total_loss
    centroid = (
        float((weights.sum(axis=0) * xs).sum()),
        float((weights.sum(axis=1) * ys).sum()),
    )
    attackers = [
        pid
        for pid, row in state.items()
        if str(row["team"]) == attacking_team and pid != runner_id
    ]
    attackers.sort(
        key=lambda pid: (math.dist(_state_xy(state[pid]), centroid), pid)
    )
    candidates = attackers[:shortlist]

    defenders = [
        pid
        for pid, row in state.items()
        if str(row["team"]) != attacking_team
    ]
    # Exploitation is a RACE INTO the region, not occupancy of it. Standing
    # in the space the defender just left is not a gain — that player was
    # already there and merely stays. The gain goes to whoever can enter it
    # ahead of the defender who would have to follow him. Round-1 diagnosis:
    # occupancy-scoring always picked the defender's former mark (Kownacki
    # 7.9 m, standing) over the carrier driving in (Karbownik 11.9 m).
    best_id, best_score, detail = "", -math.inf, {}
    for pid in candidates:
        attacker_time = _time_to_region(
            _state_xy(state[pid]), _state_v(state[pid]), xs, ys, loss
        )
        marker_time = math.inf
        for qid in defenders:
            after_xy = (
                defender_after_xy if qid == defender_id else _state_xy(state[qid])
            )
            after_v = (
                defender_after_v if qid == defender_id else _state_v(state[qid])
            )
            marker_time = min(
                marker_time, _time_to_region(after_xy, after_v, xs, ys, loss)
            )
        # Positive = he gets there first and the space is genuinely his.
        score = marker_time - attacker_time
        detail[pid] = score
        if score > best_score + 1e-12 or (
            abs(score - best_score) <= 1e-12 and pid < best_id
        ):
            best_id, best_score = pid, score
    detail["_vacated_area"] = total_loss
    return best_id, detail
