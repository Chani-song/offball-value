"""Score the dilemma measures against the reviewer's triangle-level labels.

The re-labelling of 2026-09-12 asked a different question from the old QC:
not "does this scene contain a dilemma" but "is THIS defender, chased by THIS
runner, with THIS team-mate gaining, genuinely torn". 102 triangles over 51
scenes.

Two tests, and they are not equally strong.

WITHIN-SCENE is the clean one. Each scene carries two candidate defenders
and the reviewer judged them separately, so asking whether a measure ranks
the torn one above the other uses only information that did not exist when
these measures were designed. Scene danger, which dominated every pooled
comparison so far, cancels exactly: both defenders sit in the same scene.

POOLED AUC is the weaker one. The measures were developed on the old
scene-level labels of these same scenes, and the new labels correlate with
the old, so a good pooled number is partly a memory of that development.
Reported for continuity, not as validation.

Scenes the reviewer had already settled as `none` were excluded from the
re-labelling at his request, so this set is the previously-uncertain
remainder - harder than the full population by construction.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

_spec = importlib.util.spec_from_file_location(
    "measure_commit_tradeoff", ROOT / "scripts" / "measure_commit_tradeoff.py"
)
_commit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_commit)

LABELS = ROOT / "examples/research_audit/human_reviews/dilemma/dilemma_relabel_2026_09_12.csv"


def auc(scores, labels):
    positives = [s for s, y in zip(scores, labels) if y == 1]
    negatives = [s for s, y in zip(scores, labels) if y == 0]
    if not positives or not negatives:
        return float("nan")
    return sum(
        1.0 if p > n else 0.5 if p == n else 0.0 for p in positives for n in negatives
    ) / (len(positives) * len(negatives))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audit_dirs", nargs="+")
    parser.add_argument("--labels", type=Path, default=LABELS)
    options = parser.parse_args()

    verdicts = {}
    shown_beneficiary = {}
    for row in csv.DictReader(options.labels.open(encoding="utf-8")):
        key = (row["match_id"], row["onset_frame_id"], row["defender_id"])
        verdicts[key] = row["dilemma_verdict"].strip()
        shown_beneficiary[key] = row["beneficiary_id"].strip()

    scenes = {}
    for directory in options.audit_dirs:
        for scene in json.loads(
            (Path(directory) / "local_game_payoff_audits.json").read_text()
        ):
            scenes[(str(scene["match_id"]), str(scene["onset_frame_id"]))] = scene

    rows, mismatched = [], 0
    for (match_id, frame_id, defender_id), verdict in verdicts.items():
        scene = scenes.get((match_id, frame_id))
        if scene is None:
            continue
        defender = next(
            (
                d
                for d in scene["candidate_defenders"]
                if str(d["defender_id"]) == defender_id
            ),
            None,
        )
        if defender is None:
            continue
        measures = _commit.defender_row(scene, defender)
        if measures is None:
            continue
        rows.append(((match_id, frame_id), defender_id, verdict, measures))

    names = [n for n in rows[0][3] if n not in ("응답 수",)]
    print(f"채점된 삼각 {len(rows)} / 라벨 {len(verdicts)}")
    from collections import Counter

    print("판정 분포:", dict(Counter(v for _, _, v, _ in rows)), "\n")

    # ---------- 1. within-scene ----------
    per_scene = defaultdict(list)
    for key, defender_id, verdict, measures in rows:
        per_scene[key].append((verdict, measures))
    usable = [
        pair
        for pair in per_scene.values()
        if len(pair) == 2
        and sum(1 for v, _ in pair if v == "clear") == 1
    ]
    print(
        f"① 같은 장면 안에서 찢어지는 쪽을 골라내나 "
        f"(clear가 정확히 한 명인 장면 {len(usable)}개)"
    )
    print("   무작위 기대 50%\n")
    print(f"{'측정치':<26} {'적중':>6} {'비율':>7}")
    scored = []
    for name in names:
        hits = sum(
            1
            for pair in usable
            if max(pair, key=lambda item: item[1][name])[0] == "clear"
        )
        scored.append((hits / len(usable), hits, name))
    for rate, hits, name in sorted(scored, reverse=True):
        print(f"{name:<26} {hits:>4}/{len(usable)} {rate:>7.0%}")

    # ---------- 2. pooled ----------
    print("\n② 전체 AUC (개발에 쓰인 장면들이라 검증 아님)")
    contrasts = (
        (("clear",), ("none",), "clear vs none"),
        (("clear",), ("possible", "unclear", "none"), "clear vs 나머지"),
        (("clear", "possible"), ("unclear", "none"), "clear+possible vs 나머지"),
    )
    for positives, negatives, label in contrasts:
        subset = [
            (measures, 1 if verdict in positives else 0)
            for _, _, verdict, measures in rows
            if verdict in positives + negatives
        ]
        truth = [t for _, t in subset]
        print(f"\n   [{label}]  n={len(subset)}  양성 {sum(truth)}")
        ranked = sorted(
            ((auc([m[n] for m, _ in subset], truth), n) for n in names), reverse=True
        )
        for value, name in ranked[:6]:
            print(f"     {name:<24} {value:>6.3f}")


if __name__ == "__main__":
    main()
