"""Should the dilemma count one beneficiary, or several?

The adopted measure is the defender's own minimax against a PAIR:

    knee_1 = min over A's paths of  max(Q_X, Q_Y)

with Y the single beneficiary R9 picks. But a real defender is not choosing
between two men; he is trying to hold a situation. If two team-mates both
stand to gain when he commits to the runner, his position is worse than the
pair suggests, and the pair measure understates it.

So this sweeps how many beneficiaries enter the max:

    knee_k = min over A's paths of  max(Q_X, Q_Y1, ..., Q_Yk)

with Y1..Yk the top k by R9 score, and k = all reproducing the pipeline's
own `exact_minimax_worst_q` (the minimax over every priced option).

knee_k rises with k by construction — more options to cover can only hurt
the defender — so the question is not which is largest but which separates
the reviewer's dilemma judgments best. If k=1 wins, the pair framing is
right; if k=2 or 3 wins, the dilemma is genuinely many-sided; if k=all wins,
picking a beneficiary at all was unnecessary for this stage.

Round-1-and-2 development set, scene level against the QC verdicts.
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
QC_FILES = [
    "settled_possession_onset_v0_1_reviews.csv",
    "shot_context_onset_v0_2_reviews.csv",
]
LABEL_FILES = [
    "derived_beneficiary_blind_labels_v1.csv",
    "derived_beneficiary_represent_round1fix.csv",
    "derived_beneficiary_round2_blind_labels.csv",
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


def load_pair_labels():
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
            out[(row["match_id"], row["onset_frame_id"], row["defender_id"])] = int(
                verdict == "yes" and named
            )
    return out


def auc(scores, labels):
    positives = [s for s, y in zip(scores, labels) if y == 1]
    negatives = [s for s, y in zip(scores, labels) if y == 0]
    if not positives or not negatives:
        return float("nan")
    return sum(
        1.0 if p > n else 0.5 if p == n else 0.0 for p in positives for n in negatives
    ) / (len(positives) * len(negatives))


def knees(scene, defender):
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
    _, detail = rule_r9(
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
    ranked = [
        option_id
        for option_id in sorted(
            (k for k in detail if not k.startswith("_")),
            key=lambda k: -detail[k],
        )
    ]
    if not ranked:
        return None

    # Q for every option on every response the audit stored.
    rows = []
    for response in defender["responses"]:
        cells = response.get("cells") or {}
        runner_cell = cells.get(runner) or {}
        if runner_cell.get("q") is None or runner_cell.get("legal") is False:
            continue
        values = {runner: float(runner_cell["q"])}
        for option_id, cell in cells.items():
            if cell.get("q") is None or cell.get("legal") is False:
                continue
            values[option_id] = float(cell["q"])
        rows.append(values)
    if len(rows) < 3:
        return None

    out = {}
    for k in (1, 2, 3):
        chosen = [o for o in ranked[:k] if any(o in r for r in rows)]
        if len(chosen) < k:
            out[f"무릎값 k={k}"] = float("nan")
            continue
        out[f"무릎값 k={k}"] = min(
            max([values[runner]] + [values[o] for o in chosen if o in values])
            for values in rows
        )
    out["무릎값 k=전체"] = min(max(values.values()) for values in rows)
    out["파이프라인 exact_minimax_worst_q"] = float(
        defender.get("exact_minimax_worst_q") or 0.0
    )
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audit_dirs", nargs="+")
    options = parser.parse_args()

    qc = load_scene_qc()
    pair_labels = load_pair_labels()
    scenes = {}
    for directory in options.audit_dirs:
        for scene in json.loads(
            (Path(directory) / "local_game_payoff_audits.json").read_text()
        ):
            scenes[(str(scene["match_id"]), str(scene["onset_frame_id"]))] = scene

    pair_cols, pair_truth, scene_best = {}, [], {}
    for key, scene in sorted(scenes.items()):
        rows = []
        for defender in scene["candidate_defenders"]:
            row = knees(scene, defender)
            if row is None or any(v != v for v in row.values()):
                continue
            rows.append(row)
            label = pair_labels.get((key[0], key[1], str(defender["defender_id"])))
            if label is not None:
                pair_truth.append(label)
                for name, value in row.items():
                    pair_cols.setdefault(name, []).append(value)
        if rows:
            scene_best[key] = {n: max(r[n] for r in rows) for n in rows[0]}

    print(f"쌍 {len(pair_truth)}개\n")
    print(f"{'측정치':<32} {'AUC 쌍 딜레마':>13}")
    for name in sorted(pair_cols, key=lambda n: -auc(pair_cols[n], pair_truth)):
        print(f"{name:<32} {auc(pair_cols[name], pair_truth):>13.3f}")

    positive = [v for k, v in scene_best.items() if qc.get(k) in ("clear", "possible")]
    wide = [v for k, v in scene_best.items() if qc.get(k) in ("unclear", "none")]
    tight = [v for k, v in scene_best.items() if qc.get(k) == "none"]
    if positive and wide:
        print(
            f"\n장면 수준   양성 {len(positive)} / 음성 {len(wide)} "
            f"(none만 {len(tight)})"
        )
        print(f"{'측정치':<32} {'vs unclear+none':>16} {'vs none만':>12}")
        scored = []
        for name in list(positive[0]):
            a = auc(
                [r[name] for r in positive] + [r[name] for r in wide],
                [1] * len(positive) + [0] * len(wide),
            )
            b = auc(
                [r[name] for r in positive] + [r[name] for r in tight],
                [1] * len(positive) + [0] * len(tight),
            )
            scored.append((a, b, name))
        for a, b, name in sorted(scored, reverse=True):
            print(f"{name:<32} {a:>16.3f} {b:>12.3f}")


if __name__ == "__main__":
    main()
