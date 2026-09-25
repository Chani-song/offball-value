"""Which screen should pick the candidate reacting defenders?

Two questions the reviewer asked on 2026-09-09.

(1) The pipeline ranks defenders by `runner_response_cost_m` — the mean gap
    a defender leaves while chasing the runner's goal side — and nobody has
    ever checked it against plain onset distance, which today scored AUC
    0.831 / 0.772 on the reacts labels. If distance is as good, the screen
    simplifies.

(2) Can the screen use the influence area instead — the same ball-free
    coverage geometry that carries R9 — rather than a bespoke marking-cost
    controller?

Two different jobs are measured separately, because they are different
questions and a screen is judged on the first:

  RECALL   over ALL defending outfielders: where does each criterion rank
           the defenders the reviewer said actually react? A screen keeping
           three must put them in the top three.
  RANKING  among the three the pipeline selected: AUC against the reacts
           label, i.e. how well the criterion sorts real from spurious.

Caveat stated up front: labels exist only for defenders the pipeline chose,
so recall measures "would this criterion have kept the reactors we know
about", not "did anything get missed before labelling". A criterion cannot
be credited for finding reactors nobody was ever asked about.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from offball_value.assignment_rule import attacking_team_id, onset_state  # noqa: E402
from offball_value.coupled_beneficiary import SAMPLE_STEP_SECONDS  # noqa: E402
from offball_value.local_game_structure import (  # noqa: E402
    LocalGameStructureConfig,
    _mean_marking_cost,
    simulate_goal_side_response_trace,
)
from offball_value.vacated_space import _state_v, _state_xy, coverage_field  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "score_coupled_beneficiary", ROOT / "scripts" / "score_coupled_beneficiary.py"
)
_coupled = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_coupled)

REVIEWS = ROOT / "examples/research_audit/human_reviews/shot_context"
LABEL_FILES = [
    "derived_beneficiary_blind_labels_v1.csv",
    "derived_beneficiary_represent_round1fix.csv",
    "derived_beneficiary_round2_blind_labels.csv",
]
CONFIG = LocalGameStructureConfig()


def load_reacts():
    labels = {}
    for name in LABEL_FILES:
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


def auc(scores, labels):
    positives = [s for s, y in zip(scores, labels) if y == 1]
    negatives = [s for s, y in zip(scores, labels) if y == 0]
    if not positives or not negatives:
        return float("nan")
    return sum(
        1.0 if p > n else 0.5 if p == n else 0.0 for p in positives for n in negatives
    ) / (len(positives) * len(negatives))


def point_coverage(player_xy, velocity_xy, point_xy):
    return float(
        coverage_field(
            player_xy,
            velocity_xy,
            np.array([float(point_xy[0])]),
            np.array([float(point_xy[1])]),
        )[0, 0]
    )


def scene_scores(scene):
    """Every defending outfielder scored by each candidate screen."""
    state = onset_state(scene)
    runner = str(scene["runner_id"])
    if runner not in state:
        return None
    team = attacking_team_id(scene)
    horizon = float(scene["horizon_seconds"])
    goal = scene.get("goal_xy") or [52.5, 0.0]
    goal_xy = (float(goal[0]), float(goal[1]))

    frames = sorted(
        (
            float(frame["relative_time_s"]),
            {str(p[0]): (float(p[2]), float(p[3])) for p in frame["players"]},
        )
        for frame in scene["background_frames"]
    )

    def observed_path(player_id):
        return tuple(
            (t, xy[player_id][0], xy[player_id][1])
            for t, xy in frames
            if player_id in xy and -1e-9 <= t <= horizon + 1e-9
        )

    runner_path = observed_path(runner)
    if len(runner_path) < 3:
        return None

    defenders = {
        player_id: row
        for player_id, row in state.items()
        if str(row["team"]) != team
    }
    if len(defenders) < 3:
        return None

    times = []
    time_s = SAMPLE_STEP_SECONDS
    while time_s <= horizon + 1e-9:
        times.append(time_s)
        time_s = round(time_s + SAMPLE_STEP_SECONDS, 6)

    def runner_at(target):
        previous = current = runner_path[0]
        for row in runner_path:
            current = row
            if row[0] >= target:
                break
            previous = row
        if current[0] == previous[0]:
            return (current[1], current[2])
        fraction = (target - previous[0]) / (current[0] - previous[0])
        return (
            previous[1] + fraction * (current[1] - previous[1]),
            previous[2] + fraction * (current[2] - previous[2]),
        )

    runner_points = [runner_at(t) for t in times]
    runner_start = _state_xy(state[runner])
    runner_to_goal = math.dist(runner_start, goal_xy)

    # Coverage of the runner's whole path by every defender, held at onset.
    coverage = {}
    for player_id, row in defenders.items():
        xy, velocity = _state_xy(row), _state_v(row)
        coverage[player_id] = [
            point_coverage(xy, velocity, point) for point in runner_points
        ]

    rows = {}
    for player_id, row in defenders.items():
        xy, velocity = _state_xy(row), _state_v(row)
        others = [coverage[q] for q in defenders if q != player_id]
        responsibility = [
            max(mine - max((o[i] for o in others), default=0.0), 0.0)
            for i, mine in enumerate(coverage[player_id])
        ]
        try:
            response, _ = simulate_goal_side_response_trace(
                xy, velocity, runner_path, goal_xy, horizon, CONFIG
            )
            marking = _mean_marking_cost(runner_path, response, goal_xy, CONFIG)
        except Exception:
            marking = float("nan")
        rows[player_id] = {
            # Lower is better for the first two, so negate to make every
            # criterion "higher = more likely to be the reacting defender".
            "마킹 비용 (현행 파이프라인)": -marking,
            "온셋 거리": -math.dist(xy, runner_start),
            "영향영역 평균 책임": float(np.mean(responsibility)),
            "영향영역 최대 책임": float(np.max(responsibility)),
            "영향영역 러너 커버 평균": float(np.mean(coverage[player_id])),
            "골사이드 여유 + 거리": (
                runner_to_goal - math.dist(xy, goal_xy)
            ) - math.dist(xy, runner_start),
        }
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audit_dirs", nargs="+")
    options = parser.parse_args()

    labels = load_reacts()
    scenes = {}
    for directory in options.audit_dirs:
        for scene in json.loads(
            (Path(directory) / "local_game_payoff_audits.json").read_text()
        ):
            scenes[(str(scene["match_id"]), str(scene["onset_frame_id"]))] = scene

    names = None
    recall = {}
    selected_scores, truth = {}, []
    for key, scene in sorted(scenes.items()):
        rows = scene_scores(scene)
        if rows is None:
            continue
        if names is None:
            names = list(next(iter(rows.values())))
            recall = {name: Counter() for name in names}
        selected = [str(d["defender_id"]) for d in scene["candidate_defenders"]]
        for defender_id in selected:
            label = labels.get((key[0], key[1], defender_id))
            if label is None or defender_id not in rows:
                continue
            truth.append(label)
            for name in names:
                selected_scores.setdefault(name, []).append(rows[defender_id][name])
            if label == 1:
                for name in names:
                    order = sorted(rows, key=lambda p: -rows[p][name])
                    recall[name][order.index(defender_id) + 1] += 1

    total = sum(recall[names[0]].values())
    print(f"라벨된 '반응함' 수비수 {total}명 / 라벨된 수비수-게임 {len(truth)}개")
    print(f"장면 {len(scenes)}개, 수비수 전원 대상으로 순위\n")

    print(f"{'기준':<26} {'상위3 재현율':>11} {'상위5':>8} {'중앙순위':>8}   {'AUC(선택된 3명 내)':>18}")
    for name in names:
        counts = recall[name]
        top3 = sum(v for k, v in counts.items() if k <= 3) / total
        top5 = sum(v for k, v in counts.items() if k <= 5) / total
        ordered = sorted(k for k, v in counts.items() for _ in range(v))
        median = ordered[len(ordered) // 2]
        print(
            f"{name:<26} {top3:>11.0%} {top5:>8.0%} {median:>8} "
            f"  {auc(selected_scores[name], truth):>18.3f}"
        )

    print("\n각 기준이 매긴 '반응함' 수비수들의 순위 분포")
    for name in names:
        counts = recall[name]
        print(f"  {name:<26} " + "  ".join(f"{k}등:{counts[k]}" for k in sorted(counts)))


if __name__ == "__main__":
    main()
