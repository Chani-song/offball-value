"""The reviewer's pipeline, end to end, over every (runner, defender) pair.

His design, settled 2026-09-09: do not try to name THE reacting defender.
Keep a set of candidates, pair each with the runner, pick that pair's
beneficiary with R9, and let the trade-off say which pairs are real
dilemmas.

This runs exactly that and reports three things.

1. Does the pairing earn its keep? If all candidates in a scene produce the
   same beneficiary and the same trade-off, the extra pairs are wasted work
   and one defender would have done.
2. How well does each pair-level quantity match the reviewer's judgments,
   both "does this defender react" and "does he react AND does someone
   benefit"?
3. Which pairs does the model call the strongest dilemmas, for eyeballing.

Combinations of a stage-1 quantity (marking cost, which won the defender
question at AUC 0.861) with a stage-3 quantity (the trade-off, which won the
dilemma question and is at chance on the defender question) are included,
because today's results say the two stages want different machinery and the
product is the obvious way to ask both at once. They are exploratory: this
is a label set opened today and many things have been tried on it.
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
from offball_value.coupled_beneficiary import rule_r9, value_at_times  # noqa: E402
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
QC_FILES = [
    "settled_possession_onset_v0_1_reviews.csv",
    "shot_context_onset_v0_2_reviews.csv",
]


def load_scene_qc():
    out = {}
    for name in QC_FILES:
        path = REVIEWS / name
        if not path.exists():
            continue
        for row in csv.DictReader(path.open(encoding="utf-8-sig")):
            out[(row["match_id"], row["onset_frame_id"])] = (
                row.get("interaction_review") or ""
            ).strip()
    return out
LABEL_FILES = [
    "derived_beneficiary_blind_labels_v1.csv",
    "derived_beneficiary_represent_round1fix.csv",
    "derived_beneficiary_round2_blind_labels.csv",
]
CONFIG = LocalGameStructureConfig()


def load_labels():
    out = {}
    for name in LABEL_FILES:
        path = REVIEWS / name
        if not path.exists():
            continue
        for row in csv.DictReader(path.open(encoding="utf-8-sig")):
            verdict = (row.get("defender_reacts") or "").strip().lower()
            if verdict not in ("yes", "no"):
                continue
            named = (row.get("beneficiary") or "").strip() not in ("", "none")
            out[(row["match_id"], row["onset_frame_id"], row["defender_id"])] = {
                "reacts": int(verdict == "yes"),
                "dilemma": int(verdict == "yes" and named),
                "beneficiary": (row.get("beneficiary") or "").strip(),
            }
    return out


def auc(scores, labels):
    positives = [s for s, y in zip(scores, labels) if y == 1]
    negatives = [s for s, y in zip(scores, labels) if y == 0]
    if not positives or not negatives:
        return float("nan")
    return sum(
        1.0 if p > n else 0.5 if p == n else 0.0 for p in positives for n in negatives
    ) / (len(positives) * len(negatives))


def path_coverage(xy, velocity, points):
    """Mean coverage of the runner's whole path — the screen criterion that
    retained best (92% at k=3, 100% at k=5), so the pipeline can in principle
    run on one geometry from end to end."""
    return float(
        np.mean(
            [
                coverage_field(
                    xy, velocity, np.array([float(p[0])]), np.array([float(p[1])])
                )[0, 0]
                for p in points
            ]
        )
    )


def evaluate_pair(scene, defender, state, runner_path, goal_xy, runner_points):
    defender_id = str(defender["defender_id"])
    if defender_id not in state:
        return None
    direct = next(
        (
            row
            for row in defender["responses"]
            if str(row["response_id"]) == str(defender.get("direct_best_response_id"))
        ),
        None,
    )
    if direct is None:
        return None
    defender_at, attacker_at = _coupled.absolute_lookups(scene, defender)
    if defender_at is None:
        return None

    runner = str(scene["runner_id"])
    horizon = float(scene["horizon_seconds"])
    curves = {
        option_id: value_at_times(cell.get("candidate_grid") or [])
        for option_id, cell in direct["cells"].items()
        if cell.get("legal") is not False
        and cell.get("q") is not None
        and option_id != runner
    }
    if not curves:
        return None
    beneficiary, _ = rule_r9(
        state,
        attacking_team_id(scene),
        runner,
        str(scene["carrier_id"]),
        defender_id,
        defender_at,
        attacker_at,
        curves,
        horizon,
    )
    if not beneficiary:
        return None

    points = []
    for response in defender["responses"]:
        cells = response.get("cells") or {}
        a, b = cells.get(runner) or {}, cells.get(beneficiary) or {}
        if a.get("q") is None or b.get("q") is None:
            continue
        if a.get("legal") is False or b.get("legal") is False:
            continue
        points.append((float(a["q"]), float(b["q"])))
    if len(points) < 3:
        return None

    floor_x = min(x for x, _ in points)
    floor_y = min(y for _, y in points)
    commit_x = min(y for x, y in points if x <= floor_x + 1e-12)
    commit_y = min(x for x, y in points if y <= floor_y + 1e-12)
    knee = min(max(x, y) for x, y in points)

    xy, velocity = _state_xy(state[defender_id]), _state_v(state[defender_id])
    try:
        response, _ = simulate_goal_side_response_trace(
            xy, velocity, runner_path, goal_xy, horizon, CONFIG
        )
        marking = -_mean_marking_cost(runner_path, response, goal_xy, CONFIG)
    except Exception:
        marking = -999.0
    runner_start = (float(state[runner]["x"]), float(state[runner]["y"]))
    cover = float(
        coverage_field(
            xy, velocity, np.array([runner_start[0]]), np.array([runner_start[1]])
        )[0, 0]
    )

    path_cover = path_coverage(xy, velocity, runner_points)
    both = min(commit_x, commit_y)
    return {
        "beneficiary": beneficiary,
        "영향 커버 평균": path_cover,
        "영향커버 × 두커밋": path_cover * both,
        "영향커버 × 무릎값": path_cover * knee,
        "두 커밋의 최솟값": both,
        "무릎값": knee,
        "X 따라갔을 때 Y": commit_x,
        "Y 막았을 때 X": commit_y,
        "마킹 비용": marking,
        "러너 온셋 커버": cover,
        "마킹 × 두커밋": (marking + 20.0) * both,
        "커버 × 두커밋": cover * both,
        "커버 × 무릎값": cover * knee,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audit_dirs", nargs="+")
    parser.add_argument("--top", type=int, default=15)
    parser.add_argument("--scene-level", action="store_true",
                        help="also score scenes against the QC interaction_review")
    options = parser.parse_args()

    labels = load_labels()
    scenes = {}
    for directory in options.audit_dirs:
        for scene in json.loads(
            (Path(directory) / "local_game_payoff_audits.json").read_text()
        ):
            scenes[(str(scene["match_id"]), str(scene["onset_frame_id"]))] = scene

    columns, reacts, dilemma, table = {}, [], [], []
    scene_best = {}
    distinct = Counter()
    for key, scene in sorted(scenes.items()):
        state = onset_state(scene)
        runner = str(scene["runner_id"])
        if runner not in state:
            continue
        horizon = float(scene["horizon_seconds"])
        goal = scene.get("goal_xy") or [52.5, 0.0]
        goal_xy = (float(goal[0]), float(goal[1]))
        frames = sorted(
            (
                float(f["relative_time_s"]),
                {str(p[0]): (float(p[2]), float(p[3])) for p in f["players"]},
            )
            for f in scene["background_frames"]
        )
        runner_path = tuple(
            (t, xy[runner][0], xy[runner][1])
            for t, xy in frames
            if runner in xy and -1e-9 <= t <= horizon + 1e-9
        )
        if len(runner_path) < 3:
            continue

        step, runner_points, t = 0.25, [], 0.25
        while t <= horizon + 1e-9:
            previous = current = runner_path[0]
            for rowp in runner_path:
                current = rowp
                if rowp[0] >= t:
                    break
                previous = rowp
            if current[0] == previous[0]:
                runner_points.append((current[1], current[2]))
            else:
                f = (t - previous[0]) / (current[0] - previous[0])
                runner_points.append(
                    (
                        previous[1] + f * (current[1] - previous[1]),
                        previous[2] + f * (current[2] - previous[2]),
                    )
                )
            t = round(t + step, 6)

        names = {}
        for p in scene["background_frames"][0]["players"]:
            names[str(p[0])] = str(p[4])

        scene_beneficiaries = []
        scene_rows = []
        for defender in scene["candidate_defenders"]:
            row = evaluate_pair(
                scene, defender, state, runner_path, goal_xy, runner_points
            )
            if row is None:
                continue
            scene_beneficiaries.append(row["beneficiary"])
            scene_rows.append(row)
            label = labels.get((key[0], key[1], str(defender["defender_id"])))
            if label is not None:
                reacts.append(label["reacts"])
                dilemma.append(label["dilemma"])
                for name, value in row.items():
                    if name != "beneficiary":
                        columns.setdefault(name, []).append(value)
                table.append(
                    (
                        row["두 커밋의 최솟값"],
                        key[1],
                        str(defender["defender_name"]),
                        names.get(row["beneficiary"], row["beneficiary"]),
                        names.get(label["beneficiary"], label["beneficiary"] or "없음"),
                        label["dilemma"],
                    )
                )
        if scene_beneficiaries:
            distinct[len(set(scene_beneficiaries))] += 1
        if scene_rows:
            scene_best[key] = {
                name: max(r[name] for r in scene_rows)
                for name in scene_rows[0]
                if name != "beneficiary"
            }

    print(f"평가된 쌍 {len(reacts)}개\n")
    print("1) 페어링이 값어치가 있나 — 한 장면의 후보들이 서로 다른 수혜자를 내는가")
    total_scenes = sum(distinct.values())
    for k in sorted(distinct):
        print(f"   서로 다른 수혜자 {k}명: {distinct[k]}장면 ({distinct[k]/total_scenes:.0%})")
    same = distinct.get(1, 0)
    print(
        f"   => 후보 전원이 같은 수혜자를 낸 장면은 {same}/{total_scenes} "
        f"({same/total_scenes:.0%}). 나머지는 페어마다 답이 달라진다.\n"
    )

    print("2) 쌍 수준 측정치")
    print(f"{'측정치':<20} {'AUC reacts':>11} {'AUC dilemma':>12}")
    for name in sorted(columns, key=lambda n: -auc(columns[n], dilemma)):
        print(
            f"{name:<20} {auc(columns[name], reacts):>11.3f} "
            f"{auc(columns[name], dilemma):>12.3f}"
        )

    print(f"\n3) 모델이 가장 강한 딜레마로 꼽은 쌍 {options.top}개")
    print(f"{'장면':<8} {'수비수':<20} {'모델 수혜자':<20} {'형님 수혜자':<20} {'점수':>8}  라벨")
    for score, frame, defender, model, human, is_dilemma in sorted(table, reverse=True)[
        : options.top
    ]:
        mark = "딜레마" if is_dilemma else "-"
        print(
            f"{frame:<8} {defender[:19]:<20} {model[:19]:<20} {human[:19]:<20} "
            f"{score:>8.4f}  {mark}"
        )
    if options.scene_level:
        scene_report(scene_best)


def scene_report(scene_best):
    qc = load_scene_qc()
    positive, negative = [], []
    for key, values in scene_best.items():
        verdict = qc.get(key)
        if verdict in ("clear", "possible"):
            positive.append(values)
        elif verdict in ("unclear", "none"):
            negative.append(values)
    if not positive or not negative:
        print("\n4) 장면 수준: 양성/음성 둘 다 있어야 채점 가능 "
              f"(양성 {len(positive)} / 음성 {len(negative)})")
        return
    print(f"\n4) 장면 수준 — 후보 수비수 중 최댓값 "
          f"(clear+possible {len(positive)} vs unclear+none {len(negative)})")
    names = list(positive[0])
    scored = []
    for name in names:
        values = [r[name] for r in positive] + [r[name] for r in negative]
        truth = [1] * len(positive) + [0] * len(negative)
        scored.append((auc(values, truth), name))
    for value, name in sorted(scored, reverse=True):
        print(f"   {name:<20} {value:>6.3f}")


if __name__ == "__main__":
    main()
