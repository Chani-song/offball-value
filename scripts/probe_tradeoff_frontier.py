"""The runner/beneficiary trade-off across every path the defender could take.

The reviewer's correction (2026-09-09), which fixes the mirror-of-R9 rule I
had proposed an hour earlier:

  동료 수비수가 이어받는다고 하면, 원래 수비수는 오프더볼 러너의 반응
  수비수가 될 수 없다고 하는 건 아닌 것 같아. ... 수비수 A를 반응 수비수
  로도 판단해보고, 수비수 B를 반응 수비수로도 판단해서 두 (오프더볼러너,
  수비수) 조합을 고려하는 게 맞지 않을까?

He is right. My score gated a defender OUT when a team-mate could take over,
which decides by hand-made geometry what the model should be measuring. If
the run carries X from A's responsibility into B's, then (X, A) and (X, B)
are both legitimate pairs and each deserves its own trade-off.

And the trade-off he wants is not two endpoints but a curve:

  A가 X를 따라갔다면 수혜자 Y가 얼마나 위협적이게 되는지.
  A가 X를 따라가지 않았다면 X는 얼마나 위협적이게 되는지.
  그 사이에서 수비수 A가 어중간한 여러 동선들에 대해 X, Y의 위협도는
  어떻게 변하는지.

That data already exists: every defender carries 30-40 priced
`target_agnostic_feasible` responses, each with a Q for every option. So for
a pair (X, A) we can read off the whole scatter of (Q_X, Q_Y) over the
paths A could actually run, take its Pareto frontier, and measure the shape.

Measures computed per pair, with Y chosen by R9:

  commit_to_X    Q_Y on the response that minimises Q_X — what the runner's
                 marker concedes by going with him
  commit_to_Y    Q_X on the response that minimises Q_Y — the mirror
  knee           min over responses of max(Q_X, Q_Y) — his best split
  gap            knee - max(floor_X, floor_Y) — how much the PAIR beats the
                 best he can do against either alone
  slope          how much Q_Y rises per unit Q_X suppressed along the
                 frontier. A slope near zero means he covers X for free;
                 near one means every metre he gives X he pays to Y. This is
                 the dilemma in one number.

No new constants. Round 1 first.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from offball_value.assignment_rule import attacking_team_id, onset_state  # noqa: E402
from offball_value.coupled_beneficiary import rule_r9, value_at_times  # noqa: E402

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
QC_FILES = [
    "settled_possession_onset_v0_1_reviews.csv",
    "shot_context_onset_v0_2_reviews.csv",
]


def load_defender_labels():
    labels = {}
    for name in LABEL_FILES:
        path = REVIEWS / name
        if not path.exists():
            continue
        for row in csv.DictReader(path.open(encoding="utf-8-sig")):
            verdict = (row.get("defender_reacts") or "").strip().lower()
            if verdict not in ("yes", "no"):
                continue
            named = (row.get("beneficiary") or "").strip() not in ("", "none")
            labels[(row["match_id"], row["onset_frame_id"], row["defender_id"])] = {
                "reacts": int(verdict == "yes"),
                "dilemma": int(verdict == "yes" and named),
            }
    return labels


def load_scene_labels():
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


def pareto(points):
    """Responses where neither threat can be lowered without raising the other.

    Sweep by ascending runner threat and keep a response only if it beats
    every cheaper one on the beneficiary. An earlier version popped points
    already on the frontier, which collapsed it to the single minimum-y
    response — every slope came out 0 and every frontier had one point.
    """
    frontier = []
    best_y = float("inf")
    for x, y in sorted(points):
        if y < best_y:
            frontier.append((x, y))
            best_y = y
    return frontier


def frontier_measures(scene, defender_id):
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
        float(scene["horizon_seconds"]),
    )
    if not beneficiary:
        return None

    points = []
    for response in defender["responses"]:
        cells = response.get("cells") or {}
        runner_cell = cells.get(runner) or {}
        other_cell = cells.get(beneficiary) or {}
        if runner_cell.get("q") is None or other_cell.get("q") is None:
            continue
        if runner_cell.get("legal") is False or other_cell.get("legal") is False:
            continue
        points.append((float(runner_cell["q"]), float(other_cell["q"])))
    if len(points) < 3:
        return None

    floor_x = min(x for x, _ in points)
    floor_y = min(y for _, y in points)
    commit_to_x = min(y for x, y in points if x <= floor_x + 1e-12)
    commit_to_y = min(x for x, y in points if y <= floor_y + 1e-12)
    knee = min(max(x, y) for x, y in points)
    front = pareto(points)

    # Slope across the frontier: how much the beneficiary gains per unit of
    # runner threat the defender manages to remove.
    if len(front) >= 2:
        span_x = front[-1][0] - front[0][0]
        span_y = front[0][1] - front[-1][1]
        slope = (span_y / span_x) if span_x > 1e-9 else 0.0
    else:
        slope = 0.0

    return {
        "기울기 (Y 증가 / X 감소)": slope,
        "X를 따라갔을 때 Y 위협": commit_to_x,
        "Y를 막았을 때 X 위협": commit_to_y,
        "무릎값 min max(X,Y)": knee,
        "간극 knee - 단독바닥": knee - max(floor_x, floor_y),
        "두 커밋의 최솟값": min(commit_to_x, commit_to_y),
        "프론티어 점 개수": float(len(front)),
        "응답 개수": float(len(points)),
    }


def auc(scores, labels):
    positives = [s for s, y in zip(scores, labels) if y == 1]
    negatives = [s for s, y in zip(scores, labels) if y == 0]
    if not positives or not negatives:
        return float("nan")
    return sum(
        1.0 if p > n else 0.5 if p == n else 0.0 for p in positives for n in negatives
    ) / (len(positives) * len(negatives))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audit_dirs", nargs="+")
    options = parser.parse_args()

    defender_labels = load_defender_labels()
    scene_labels = load_scene_labels()
    scenes = {}
    for directory in options.audit_dirs:
        for scene in json.loads(
            (Path(directory) / "local_game_payoff_audits.json").read_text()
        ):
            scenes[(str(scene["match_id"]), str(scene["onset_frame_id"]))] = scene

    per_defender, truth_reacts, truth_dilemma = {}, [], []
    per_scene = {}
    for key, scene in sorted(scenes.items()):
        rows = {}
        for defender in scene["candidate_defenders"]:
            defender_id = str(defender["defender_id"])
            measures = frontier_measures(scene, defender_id)
            if measures is None:
                continue
            rows[defender_id] = measures
            label = defender_labels.get((key[0], key[1], defender_id))
            if label is not None:
                truth_reacts.append(label["reacts"])
                truth_dilemma.append(label["dilemma"])
                for name, value in measures.items():
                    per_defender.setdefault(name, []).append(value)
        verdict = scene_labels.get(key)
        if rows and verdict in ("clear", "possible", "unclear", "none"):
            per_scene[key] = (
                verdict,
                {
                    name: max(row[name] for row in rows.values())
                    for name in next(iter(rows.values()))
                },
            )

    print(f"수비수-게임 (라벨 있음) {len(truth_reacts)}개")
    print(f"{'측정치':<26} {'AUC reacts':>11} {'AUC dilemma':>12}")
    for name in sorted(
        per_defender, key=lambda n: -auc(per_defender[n], truth_dilemma)
    ):
        print(
            f"{name:<26} {auc(per_defender[name], truth_reacts):>11.3f} "
            f"{auc(per_defender[name], truth_dilemma):>12.3f}"
        )

    groups = {}
    for verdict, values in per_scene.values():
        groups.setdefault(verdict, []).append(values)
    counts = {k: len(v) for k, v in groups.items()}
    print(f"\n장면 수준 (수비수 중 최댓값)  {counts}")
    positive = groups.get("clear", []) + groups.get("possible", [])
    negative = groups.get("none", []) + groups.get("unclear", [])
    if positive and negative:
        print(f"{'측정치':<26} {'AUC (clear+possible vs unclear+none)':>38}")
        names = list(positive[0])
        for name in sorted(
            names,
            key=lambda n: -auc(
                [r[n] for r in positive] + [r[n] for r in negative],
                [1] * len(positive) + [0] * len(negative),
            ),
        ):
            scores = [r[name] for r in positive] + [r[name] for r in negative]
            labels = [1] * len(positive) + [0] * len(negative)
            print(f"{name:<26} {auc(scores, labels):>38.3f}")
    else:
        print("  (음성 장면이 이 산출물에 없음)")


if __name__ == "__main__":
    main()
