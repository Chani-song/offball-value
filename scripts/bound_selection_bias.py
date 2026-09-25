"""How much of the dilemma AUC could be selection bias?

Yesterday I tried roughly twenty measures on the 141 QC-labelled scenes and
reported the best at 0.657. Trying twenty things and keeping the winner
inflates that number even if every measure is worthless, and a train/test
split made afterwards cannot undo it — I have already seen every scene.

What CAN be done is bound the inflation. Two checks:

PERMUTATION. Shuffle the labels and re-run the same search over the same
measures, many times. The distribution of the best-of-N AUC under shuffled
labels is exactly the null the real 0.657 has to beat. If shuffled searches
routinely reach 0.63, then 0.657 means almost nothing; if they top out at
0.58, the margin is real.

PER-MATCH STABILITY. Score the adopted measure inside each match separately.
A signal that only exists in one match is a property of that match. This is
not validation — the scenes are the same ones — but a measure that holds at
0.6-0.7 in every match is harder to explain as an artefact than one that is
0.9 in one match and 0.5 elsewhere.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

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


def auc(scores, labels):
    positives = [s for s, y in zip(scores, labels) if y == 1]
    negatives = [s for s, y in zip(scores, labels) if y == 0]
    if not positives or not negatives:
        return float("nan")
    return sum(
        1.0 if p > n else 0.5 if p == n else 0.0 for p in positives for n in negatives
    ) / (len(positives) * len(negatives))


def pair_measures(scene, defender):
    state = onset_state(scene)
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
    knee = min(max(x, y) for x, y in points)
    commit_x = min(y for x, y in points if x <= floor_x + 1e-12)
    commit_y = min(x for x, y in points if y <= floor_y + 1e-12)
    cross = defender.get("cross_cost") or {}
    # The family actually searched yesterday, reconstructed so the
    # permutation faces the same number of chances the real search had.
    return {
        "무릎값": knee,
        "두 커밋의 최솟값": min(commit_x, commit_y),
        "X 따라갔을 때 Y": commit_x,
        "Y 막았을 때 X": commit_y,
        "간극": knee - max(floor_x, floor_y),
        "간극 상대값": (knee - max(floor_x, floor_y)) / knee if knee > 1e-9 else 0.0,
        "러너 바닥": floor_x,
        "수혜자 바닥": floor_y,
        "쌍 최선 both": knee,
        "cross_cost strength": float(cross.get("strength") or 0.0),
        "cross_cost runner": float(cross.get("runner") or 0.0),
        "minimax worst q": float(defender.get("exact_minimax_worst_q") or 0.0),
        "무릎값 x strength": knee * float(cross.get("strength") or 0.0),
        "무릎값 x runner": knee * float(cross.get("runner") or 0.0),
        "커밋 x strength": min(commit_x, commit_y)
        * float(cross.get("strength") or 0.0),
        "무릎값 제곱": knee**2,
        "커밋합": commit_x + commit_y,
        "커밋차 절대값": abs(commit_x - commit_y),
        "바닥합": floor_x + floor_y,
        "무릎값 - 러너바닥": knee - floor_x,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audit_dirs", nargs="+")
    parser.add_argument("--permutations", type=int, default=400)
    options = parser.parse_args()

    qc = load_scene_qc()
    scenes = {}
    for directory in options.audit_dirs:
        for scene in json.loads(
            (Path(directory) / "local_game_payoff_audits.json").read_text()
        ):
            scenes[(str(scene["match_id"]), str(scene["onset_frame_id"]))] = scene

    per_scene, verdicts, matches = [], [], []
    for key, scene in sorted(scenes.items()):
        verdict = qc.get(key)
        if verdict not in ("clear", "possible", "unclear", "none"):
            continue
        rows = [
            row
            for row in (
                pair_measures(scene, defender)
                for defender in scene["candidate_defenders"]
            )
            if row is not None
        ]
        if not rows:
            continue
        per_scene.append({n: max(r[n] for r in rows) for n in rows[0]})
        verdicts.append(verdict)
        matches.append(key[0])

    names = list(per_scene[0])
    truth = np.array(
        [1 if v in ("clear", "possible") else 0 for v in verdicts], dtype=int
    )
    columns = {n: np.array([r[n] for r in per_scene]) for n in names}
    print(f"장면 {len(per_scene)}개 (양성 {int(truth.sum())} / 음성 {int((1-truth).sum())})")
    print(f"측정치 {len(names)}개, 순열 {options.permutations}회\n")

    real = {n: auc(columns[n], truth) for n in names}
    best_name = max(real, key=lambda n: real[n])
    print(f"{'측정치':<24} {'AUC':>7}")
    for n in sorted(real, key=lambda n: -real[n])[:8]:
        mark = "  <-- 최고" if n == best_name else ""
        print(f"{n:<24} {real[n]:>7.3f}{mark}")

    rng = np.random.default_rng(0)
    best_null, knee_null = [], []
    for _ in range(options.permutations):
        shuffled = truth.copy()
        rng.shuffle(shuffled)
        aucs = [auc(columns[n], shuffled) for n in names]
        best_null.append(max(aucs))
        knee_null.append(auc(columns["무릎값"], shuffled))

    best_null = np.array(best_null)
    knee_null = np.array(knee_null)
    print(f"\n순열 귀무분포 — 라벨을 섞고 같은 탐색을 반복")
    print(
        f"  {len(names)}개 중 최고 AUC: 중앙 {np.median(best_null):.3f}  "
        f"95분위 {np.quantile(best_null, 0.95):.3f}  최대 {best_null.max():.3f}"
    )
    print(
        f"  무릎값 단독 AUC:        중앙 {np.median(knee_null):.3f}  "
        f"95분위 {np.quantile(knee_null, 0.95):.3f}"
    )
    print(
        f"\n  실제 최고 {real[best_name]:.3f} 를 섞은 라벨이 넘은 비율: "
        f"{(best_null >= real[best_name]).mean():.1%}   <- 탐색 보정 p"
    )
    print(
        f"  실제 무릎값 {real['무릎값']:.3f} 를 섞은 라벨이 넘은 비율: "
        f"{(knee_null >= real['무릎값']).mean():.1%}   <- 단일 측정치 p"
    )

    print("\n경기별 안정성 (무릎값)")
    groups = defaultdict(list)
    for row, verdict, match in zip(per_scene, verdicts, matches):
        groups[match].append((row["무릎값"], 1 if verdict in ("clear", "possible") else 0))
    values = []
    for match, rows in sorted(groups.items()):
        scores = [s for s, _ in rows]
        labels = [y for _, y in rows]
        value = auc(scores, labels)
        if value == value:
            values.append(value)
        print(
            f"  {match[-6:]}  n={len(rows):>3}  "
            f"양성 {sum(labels):>2}  AUC {value:.3f}"
        )
    if values:
        print(
            f"  경기별 중앙 {statistics.median(values):.3f}  "
            f"최소 {min(values):.3f}  최대 {max(values):.3f}"
        )


if __name__ == "__main__":
    main()
