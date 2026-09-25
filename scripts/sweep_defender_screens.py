"""How many candidate reacting defenders to keep, and by which criterion.

The reviewer settled the design question on 2026-09-09:

  내가 보기에 반응 수비수를 하나 뽑는 건 말이 안 되는 것 같아. 오프더볼
  러너에 대해 반응 수비수가 2명인 것도 자연스럽고, 3명까지도 가능할 수도
  있고. 그래서 반응 수비수 후보자들을 뽑고 걔네를 돌려가면서 오프더볼
  러너와 페어해서 테스트하는 게 맞는 것 같아.

The labels agree: 7 of 17 round-1 scenes have two or three reacting
defenders, and subtracting team-mate coverage — which encodes "somebody else
has him, so this one is free" — HURTS at this stage (0.777 vs 0.797) while
it helps at the beneficiary stage. Several defenders can be stretched by the
same run.

So the screen is no longer a classifier. Its only job is retention: keep
every defender who might be in a pair, at the smallest k that does it. This
sweeps k from 1 to 8 for each criterion, and for unions of two criteria.

CIRCULARITY WARNING, applies to one column only: the pipeline chose its
three candidates by marking cost and only those three were ever labelled, so
marking cost retains 100% by construction. Its number is not evidence. Every
other criterion is being asked a fair question — "would you have kept the
reactors we know about" — and the honest comparison is between them.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from offball_value.assignment_rule import attacking_team_id, onset_state  # noqa: E402
from offball_value.local_game_structure import (  # noqa: E402
    LocalGameStructureConfig,
    _mean_marking_cost,
    simulate_goal_side_response_trace,
)
from offball_value.vacated_space import _state_v, _state_xy, coverage_field  # noqa: E402

REVIEWS = ROOT / "examples/research_audit/human_reviews/shot_context"
LABEL_FILES = [
    "derived_beneficiary_blind_labels_v1.csv",
    "derived_beneficiary_represent_round1fix.csv",
    "derived_beneficiary_round2_blind_labels.csv",
]
CONFIG = LocalGameStructureConfig()
STEP_S = 0.25


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


def point_coverage(player_xy, velocity_xy, points):
    """Coverage of a list of points by one player held at his onset state."""
    return [
        float(
            coverage_field(
                player_xy,
                velocity_xy,
                np.array([float(p[0])]),
                np.array([float(p[1])]),
            )[0, 0]
        )
        for p in points
    ]


def scene_criteria(scene):
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
    runner_path = tuple(
        (t, xy[runner][0], xy[runner][1])
        for t, xy in frames
        if runner in xy and -1e-9 <= t <= horizon + 1e-9
    )
    if len(runner_path) < 3:
        return None

    times, time_s = [], STEP_S
    while time_s <= horizon + 1e-9:
        times.append(time_s)
        time_s = round(time_s + STEP_S, 6)

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

    points = [runner_at(t) for t in times]
    runner_start = _state_xy(state[runner])
    runner_to_goal = math.dist(runner_start, goal_xy)
    defenders = {p: r for p, r in state.items() if str(r["team"]) != team}
    if len(defenders) < 4:
        return None

    coverage = {
        player_id: point_coverage(_state_xy(row), _state_v(row), points)
        for player_id, row in defenders.items()
    }
    late = np.linspace(0.2, 1.0, len(times))

    rows = {}
    for player_id, row in defenders.items():
        xy, velocity = _state_xy(row), _state_v(row)
        mine = np.array(coverage[player_id])
        others = np.array(
            [coverage[q] for q in defenders if q != player_id]
        ).max(axis=0)
        try:
            response, _ = simulate_goal_side_response_trace(
                xy, velocity, runner_path, goal_xy, horizon, CONFIG
            )
            marking = -_mean_marking_cost(runner_path, response, goal_xy, CONFIG)
        except Exception:
            marking = -999.0
        rows[player_id] = {
            "마킹 비용 (현행)": marking,
            "온셋 거리": -math.dist(xy, runner_start),
            "골사이드 거리": (runner_to_goal - math.dist(xy, goal_xy))
            - math.dist(xy, runner_start),
            "영향 커버 평균": float(mine.mean()),
            "영향 커버 최대": float(mine.max()),
            "영향 커버 온셋만": float(
                point_coverage(xy, velocity, [runner_start])[0]
            ),
            "영향 커버 후반가중": float((mine * late).sum() / late.sum()),
            "영향 배타책임 평균": float(np.clip(mine - others, 0, None).mean()),
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

    names, per_scene = None, []
    for key, scene in sorted(scenes.items()):
        rows = scene_criteria(scene)
        if rows is None:
            continue
        if names is None:
            names = list(next(iter(rows.values())))
        reactors = [
            str(d["defender_id"])
            for d in scene["candidate_defenders"]
            if labels.get((key[0], key[1], str(d["defender_id"]))) == 1
            and str(d["defender_id"]) in rows
        ]
        if reactors:
            per_scene.append((rows, reactors))

    total = sum(len(r) for _, r in per_scene)
    print(f"장면 {len(per_scene)}개, 라벨된 '반응함' 수비수 {total}명")
    print("(마킹 비용 열은 순환논리 — 그 기준으로 뽑힌 3명만 라벨됐음)\n")

    ks = [1, 2, 3, 4, 5, 6, 8]
    print(f"{'기준':<20} " + "".join(f"{'k=' + str(k):>7}" for k in ks))
    for name in names:
        kept = []
        for k in ks:
            hits = 0
            for rows, reactors in per_scene:
                order = sorted(rows, key=lambda p: -rows[p][name])[:k]
                hits += sum(1 for r in reactors if r in order)
            kept.append(hits / total)
        flag = "  <- 순환" if name.startswith("마킹") else ""
        print(f"{name:<20} " + "".join(f"{v:>7.0%}" for v in kept) + flag)

    print("\n두 기준의 합집합 (각각 top-k/2, 총 k명 이하)")
    pairs = [
        ("마킹 비용 (현행)", "영향 커버 평균"),
        ("온셋 거리", "영향 커버 평균"),
        ("영향 커버 평균", "골사이드 거리"),
        ("마킹 비용 (현행)", "온셋 거리"),
    ]
    print(f"{'합집합':<34} " + "".join(f"{'k=' + str(k):>7}" for k in (2, 4, 6, 8)))
    for left, right in pairs:
        kept = []
        for k in (2, 4, 6, 8):
            half = max(k // 2, 1)
            hits = 0
            for rows, reactors in per_scene:
                keep = set(sorted(rows, key=lambda p: -rows[p][left])[:half]) | set(
                    sorted(rows, key=lambda p: -rows[p][right])[:half]
                )
                hits += sum(1 for r in reactors if r in keep)
            kept.append(hits / total)
        label = f"{left.split(' ')[0]} + {right.split(' ')[0]}"
        print(f"{label:<34} " + "".join(f"{v:>7.0%}" for v in kept))


if __name__ == "__main__":
    main()
