"""The counterfactual-assignment beneficiary rules (R0 / R1).

Canonical module for the rules locked in docs/preregistration_round2_v1_0
(hash 55fd27adf6623edd) and raced in the round-2 blind test
(docs/blind_session_round2_final_results.md). The pre-registered adoption
verdict was INCONCLUSIVE (R1 topped the race at 14.5/22 but led the best
Q-variant by only 0.5); R1 is shipped in the pipeline as a PROVISIONAL
selection layer on parsimony / cross-population-stability / cost grounds,
never as a claimed statistical winner. Constants and their train-fit
provenance are documented in scripts/predict_assignment_rules.py and the
lock docs; do not retune them outside a new pre-registered round.

The rule, in the reviewer's own football terms: the beneficiary of an
off-ball run is the player the reacting defender WOULD HAVE ENGAGED had
the runner not existed — approximated by the nearest non-runner attacker,
with a carrier-override when the carrier is advancing at the defender and
a guarded covered-exclusion when the nearest man is already another
defender's charge. Known open refinements (v3.1, qualitative): the
carrier condition is really trajectory-into-influence-zone, exclusion is
really responsibility assignment, vacated-zone exploiters are picked by
velocity direction, threat breaks ties — see
docs/counterfactual_assignment_criterion_v3.md.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

# Locked constants — fitted on the 24 round-1 picks (train), see
# scripts/predict_assignment_rules.py for windows and sensitivity.
V_ADV_MPS = 1.75
D_ADV_M = 15.0
R_COV_M = 3.0
D_NEAR_M = 5.0
VEL_TARGET_S = 0.2

PROVISIONAL_LABEL = "assignment_rule_v1 (provisional; preregistration inconclusive)"


def _frames(scene: Mapping[str, object]) -> Sequence[Mapping[str, object]]:
    frames = scene.get("background_frames") or scene.get("frames")
    if not frames:
        raise ValueError("scene has neither 'background_frames' nor 'frames'")
    return frames  # type: ignore[return-value]


def attacking_team_id(scene: Mapping[str, object]) -> str:
    team = scene.get("attacking_team_id") or scene.get("team_id")
    if not team:
        raise ValueError("scene has neither 'attacking_team_id' nor 'team_id'")
    return str(team)


def onset_state(
    scene: Mapping[str, object], vel_target_s: float = VEL_TARGET_S
) -> dict[str, dict[str, object]]:
    """Positions at the onset frame plus finite-difference velocities.

    Velocity = (p(f+) - p(f-)) / (t+ - t-), where f+ / f- are the frames
    whose relative_time_s is nearest to +/- ``vel_target_s`` (ties broken
    toward the smaller ``|relative_time_s|``). Players missing from either
    flank frame get v = (0, 0).
    """
    frames = _frames(scene)

    def nearest(target: float) -> Mapping[str, object]:
        return min(
            frames,
            key=lambda fr: (
                abs(float(fr["relative_time_s"]) - target),
                abs(float(fr["relative_time_s"])),
            ),
        )

    f0 = nearest(0.0)
    fp = nearest(vel_target_s)
    fm = nearest(-vel_target_s)
    dt = float(fp["relative_time_s"]) - float(fm["relative_time_s"])

    def pmap(fr: Mapping[str, object]) -> dict[str, tuple[str, float, float, str]]:
        return {
            str(p[0]): (str(p[1]), float(p[2]), float(p[3]), str(p[4]))
            for p in fr["players"]  # type: ignore[index]
        }

    p0, pp, pm = pmap(f0), pmap(fp), pmap(fm)
    state: dict[str, dict[str, object]] = {}
    for pid, (team, x, y, name) in p0.items():
        if pid in pp and pid in pm and dt > 0:
            vx = (pp[pid][1] - pm[pid][1]) / dt
            vy = (pp[pid][2] - pm[pid][2]) / dt
        else:
            vx = vy = 0.0
        state[pid] = {"team": team, "x": x, "y": y, "name": name, "vx": vx, "vy": vy}
    return state


def _dist(a: Mapping[str, object], b: Mapping[str, object]) -> float:
    return math.hypot(float(a["x"]) - float(b["x"]), float(a["y"]) - float(b["y"]))


def ranked_candidates(
    state: Mapping[str, Mapping[str, object]],
    attacking_team: str,
    runner_id: str,
    defender_id: str,
) -> list[tuple[float, str]]:
    """Non-runner attackers sorted by (distance to defender, option id)."""
    defender = state[defender_id]
    cands = [
        (_dist(player, defender), pid)
        for pid, player in state.items()
        if str(player["team"]) == attacking_team and pid != runner_id
    ]
    cands.sort(key=lambda item: (round(item[0], 9), item[1]))
    return cands


def rule_r0(
    state: Mapping[str, Mapping[str, object]],
    attacking_team: str,
    runner_id: str,
    carrier_id: str,
    defender_id: str,
) -> str:
    """Nearest non-runner attacker to the defender at the onset frame."""
    del carrier_id
    return ranked_candidates(state, attacking_team, runner_id, defender_id)[0][1]


def carrier_advancing(
    state: Mapping[str, Mapping[str, object]],
    carrier_id: str,
    defender_id: str,
    v_adv: float = V_ADV_MPS,
    d_adv: float = D_ADV_M,
) -> bool:
    """True iff the carrier-override condition fires for this defender."""
    carrier = state.get(carrier_id)
    defender = state[defender_id]
    if carrier is None:
        return False
    d_cd = _dist(carrier, defender)
    if d_cd > d_adv or d_cd <= 1e-9:
        return False
    ux = (float(defender["x"]) - float(carrier["x"])) / d_cd
    uy = (float(defender["y"]) - float(carrier["y"])) / d_cd
    approach = float(carrier["vx"]) * ux + float(carrier["vy"]) * uy
    return approach >= v_adv


def rule_r1(
    state: Mapping[str, Mapping[str, object]],
    attacking_team: str,
    runner_id: str,
    carrier_id: str,
    defender_id: str,
) -> tuple[str, str]:
    """R1 prediction plus the branch that produced it.

    Branches: 'override' (carrier-override fired), 'exclusion'
    (covered-exclusion skipped the nearest candidate), 'nearest'
    (fell through to R0's pick).
    """
    if carrier_advancing(state, carrier_id, defender_id):
        return carrier_id, "override"
    cands = ranked_candidates(state, attacking_team, runner_id, defender_id)
    d0, top = cands[0]
    if top != carrier_id and d0 >= D_NEAR_M and len(cands) > 1:
        top_player = state[top]
        for qid in sorted(state):
            q = state[qid]
            if str(q["team"]) == attacking_team or qid == defender_id:
                continue
            dq = _dist(q, top_player)
            if dq <= R_COV_M and dq < d0:
                return cands[1][1], "exclusion"
    return top, "nearest"


def predict_scene(
    scene: Mapping[str, object], defender_ids: Sequence[str]
) -> list[dict[str, object]]:
    """Per-defender R0 / R1 / always-carrier predictions for one scene."""
    state = onset_state(scene)
    attacking_team = attacking_team_id(scene)
    runner_id = str(scene["runner_id"])
    carrier_id = str(scene["carrier_id"])

    def name_of(pid: str) -> str:
        player = state.get(pid)
        return str(player["name"]) if player is not None else ""

    rows: list[dict[str, object]] = []
    for defender_id in defender_ids:
        r0 = rule_r0(state, attacking_team, runner_id, carrier_id, defender_id)
        r1, branch = rule_r1(
            state, attacking_team, runner_id, carrier_id, defender_id
        )
        rows.append(
            {
                "match_id": str(scene["match_id"]),
                "onset_frame_id": int(scene["onset_frame_id"]),  # type: ignore[arg-type]
                "runner_id": runner_id,
                "carrier_id": carrier_id,
                "defender_id": defender_id,
                "defender_name": name_of(defender_id),
                "always_carrier_pred_id": carrier_id,
                "always_carrier_pred_name": name_of(carrier_id),
                "r0_pred_id": r0,
                "r0_pred_name": name_of(r0),
                "r1_pred_id": r1,
                "r1_pred_name": name_of(r1),
                "r1_branch": branch,
            }
        )
    return rows
