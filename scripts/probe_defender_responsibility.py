"""A-2: which defender must react to the runner — as the mirror of R9.

R9 works by asking what a defender gives up by LEAVING: the coverage of his
exclusive zone, the ground no team-mate holds. The defender-selection
question is the same quantity turned around — what he gives up by STAYING:

    responsibility_D(t) = max(0,  D's coverage of the runner at t
                                  - the best team-mate's coverage of him)

with D frozen at his onset position. At t=0 this is high if D is the man
marking the runner. As the runner goes, it decays — unless a team-mate picks
him up, in which case D never had to move.

    score_D = responsibility_D(0) - mean_t responsibility_D(t)

So the rule reuses parts that are already validated — the same ball-free
coverage ellipse, the same "exclusive means responsible" subtraction that
was worth +2/+1 in the R9 ablation, the same observed trajectories that beat
every counterfactual substitute — and introduces no new constant.

Two variants are scored: the geometry alone, and geometry weighted by how
dangerous the runner actually becomes (his own Q curve), mirroring how R6
needed occupation x Q rather than occupation alone.

Baselines: onset distance to the runner (today's best single feature, AUC
0.831 -> 0.772), and the two cross_cost channels the pipeline uses now.

Round 1 first. AUC is the headline because it cannot be flattered by a
threshold, and today's stage-1 result showed thresholds are exactly what
fails to transfer between these two populations.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from offball_value.assignment_rule import attacking_team_id, onset_state  # noqa: E402
from offball_value.coupled_beneficiary import (  # noqa: E402
    SAMPLE_STEP_SECONDS,
    value_at_times,
    value_near,
)
from offball_value.vacated_space import (  # noqa: E402
    _state_v,
    _state_xy,
    coverage_field,
)

_spec = importlib.util.spec_from_file_location(
    "score_coupled_beneficiary", ROOT / "scripts" / "score_coupled_beneficiary.py"
)
_coupled = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_coupled)

REVIEWS = ROOT / "examples/research_audit/human_reviews/shot_context"
ROUND_FILES = {
    "1": [
        "derived_beneficiary_blind_labels_v1.csv",
        "derived_beneficiary_represent_round1fix.csv",
    ],
    "2": ["derived_beneficiary_round2_blind_labels.csv"],
}


def load_reacts(round_key: str) -> dict[tuple[str, str, str], int]:
    labels: dict[tuple[str, str, str], int] = {}
    for name in ROUND_FILES[round_key]:
        path = REVIEWS / name
        if not path.exists():
            continue
        for row in csv.DictReader(path.open(encoding="utf-8-sig")):
            verdict = (row.get("defender_reacts") or "").strip().lower()
            if verdict in ("yes", "no"):
                labels[
                    (row["match_id"], row["onset_frame_id"], row["defender_id"])
                ] = int(verdict == "yes")
    return labels


def coverage_at(player_xy, velocity_xy, point_xy) -> float:
    """The ball-free coverage ellipse at a single point.

    Delegates to ``coverage_field`` on a one-cell grid rather than repeating
    its algebra, so this rule and R9 can never drift apart in how coverage
    is defined.
    """
    return float(
        coverage_field(
            player_xy,
            velocity_xy,
            np.array([float(point_xy[0])]),
            np.array([float(point_xy[1])]),
        )[0, 0]
    )


def auc(scores, labels) -> float:
    positives = [s for s, y in zip(scores, labels) if y == 1]
    negatives = [s for s, y in zip(scores, labels) if y == 0]
    if not positives or not negatives:
        return float("nan")
    wins = sum(
        1.0 if p > n else 0.5 if p == n else 0.0 for p in positives for n in negatives
    )
    return wins / (len(positives) * len(negatives))


def features(scene, defender_id):
    state = onset_state(scene)
    if defender_id not in state:
        return None
    defender = next(
        (
            row
            for row in scene["candidate_defenders"]
            if str(row["defender_id"]) == defender_id
        ),
        None,
    )
    if defender is None:
        return None
    response = next(
        (
            row
            for row in defender["responses"]
            if str(row["response_id"]) == str(defender.get("direct_best_response_id"))
        ),
        None,
    )
    if response is None:
        return None
    _, attacker_at = _coupled.absolute_lookups(scene, defender)
    if attacker_at is None:
        return None

    team = attacking_team_id(scene)
    runner = str(scene["runner_id"])
    if runner not in state:
        return None
    horizon = float(scene["horizon_seconds"])

    frozen_xy = _state_xy(state[defender_id])
    frozen_v = _state_v(state[defender_id])
    mates = [
        (_state_xy(row), _state_v(row))
        for player_id, row in state.items()
        if str(row["team"]) != team and player_id != defender_id
    ]

    runner_cell = response["cells"].get(runner) or {}
    runner_curve = value_at_times(runner_cell.get("candidate_grid") or [])

    def responsibility(point_xy: tuple[float, float]) -> float:
        mine = coverage_at(frozen_xy, frozen_v, point_xy)
        theirs = max(
            (coverage_at(xy, velocity, point_xy) for xy, velocity in mates),
            default=0.0,
        )
        return max(mine - theirs, 0.0)

    start = responsibility(_state_xy(state[runner]))
    kept, weighted, times = [], [], []
    time_s = SAMPLE_STEP_SECONDS
    while time_s <= horizon + 1e-9:
        runner_xy, _ = attacker_at(runner, time_s)
        value = responsibility(runner_xy)
        kept.append(value)
        weighted.append(value * value_near(runner_curve, time_s))
        times.append(time_s)
        time_s = round(time_s + SAMPLE_STEP_SECONDS, 6)
    if not kept:
        return None

    lost = start - float(np.mean(kept))
    start_weighted = start * (value_near(runner_curve, 0.0) or 0.0)
    lost_weighted = start_weighted - float(np.mean(weighted))

    cross = defender.get("cross_cost") or {}
    runner_xy0 = _state_xy(state[runner])
    distance = math.dist(frozen_xy, runner_xy0)

    # Football-side features: marking is an ASSIGNMENT problem, so what
    # matters may be less "how close is he" than "is the runner the man he
    # is closest to", "is anyone behind him", and "is he already busy".
    goal = scene.get("goal_xy") or [52.5, 0.0]
    goal_xy = (float(goal[0]), float(goal[1]))
    other_attackers = [
        _state_xy(row)
        for player_id, row in state.items()
        if str(row["team"]) == team and player_id not in (runner, str(scene["carrier_id"]))
    ]
    nearest_other = min(
        (math.dist(frozen_xy, xy) for xy in other_attackers), default=99.0
    )
    mates_to_runner = sorted(math.dist(xy, runner_xy0) for xy, _ in mates)
    closer_mates = sum(1 for d in mates_to_runner if d < distance)
    # Goal-side: is the defender between the runner and the goal he defends?
    to_goal = math.dist(runner_xy0, goal_xy)
    goal_side = to_goal - math.dist(frozen_xy, goal_xy)
    mean_kept = float(np.mean(kept))
    return {
        "평균 책임 (전 구간)": mean_kept,
        "최대 책임": float(np.max(kept)),
        "책임 증가 (평균 - 온셋)": mean_kept - start,
        "평균 책임 × 러너 위협": float(np.mean(weighted)),
        "책임 상실 (기하)": lost,
        "책임 상실 × 러너 위협": lost_weighted,
        "온셋 책임 (t=0)": start,
        "러너까지 거리 (음수)": -distance,
        "cross_cost runner": float(cross.get("runner") or 0.0),
        "cross_cost strength": float(cross.get("strength") or 0.0),
        # --- 축구 쪽 특징 ---
        "러너가 그의 최근접 공격수인가": nearest_other - distance,
        "그보다 러너에 가까운 동료 수 (음수)": -closer_mates,
        "골 사이드 여유 (m)": goal_side,
        "두번째 수비수와의 거리차": (mates_to_runner[0] - distance) if mates_to_runner else 0.0,
        "거리 × 러너가 최근접": (nearest_other - distance) - distance,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audit_dirs", nargs="+")
    parser.add_argument("--round", required=True, choices=["1", "2"])
    options = parser.parse_args()

    labels = load_reacts(options.round)
    scenes = {}
    for directory in options.audit_dirs:
        for scene in json.loads(
            (Path(directory) / "local_game_payoff_audits.json").read_text()
        ):
            scenes[(str(scene["match_id"]), str(scene["onset_frame_id"]))] = scene

    columns: dict[str, list[float]] = {}
    truth: list[int] = []
    for (match_id, frame_id, defender_id), reacts in sorted(labels.items()):
        scene = scenes.get((match_id, frame_id))
        if scene is None:
            continue
        row = features(scene, defender_id)
        if row is None:
            continue
        truth.append(reacts)
        for key, value in row.items():
            columns.setdefault(key, []).append(value)

    positives = sum(truth)
    print(
        f"라운드 {options.round}  n = {len(truth)}  "
        f"(반응함 {positives} / 반응 안함 {len(truth) - positives})\n"
    )
    print(f"{'특징':<24} {'AUC':>6}")
    for name, values in sorted(
        columns.items(), key=lambda item: -auc(item[1], truth)
    ):
        print(f"{name:<24} {auc(values, truth):>6.3f}")


if __name__ == "__main__":
    main()
