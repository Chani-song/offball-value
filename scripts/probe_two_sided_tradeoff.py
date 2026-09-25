"""The two-sided trade-off, recomputed with R9's beneficiary.

The reviewer's design (2026-09-09): the dilemma judgment should not be a
separate classifier bolted onto scene features. It should FALL OUT of the
pipeline — pick the defender, pick the beneficiary, then ask whether the
defender can suppress the runner and the beneficiary at the same time.

The pipeline already has a `cross_cost` field, but it was computed against
the OLD derived-option selection, which round 2 showed to be worse than R9
(15/23 vs 20/23). So the trade-off has never been measured against the
beneficiary we now trust.

For a defender D with runner R and R9's beneficiary B, using D's own priced
response set:

    both   = min over responses of  max(Q_R, Q_B)     what the attack gets
                                                       if D plays his best
                                                       answer to the PAIR
    floorR = min over responses of  Q_R                best he can do
    floorB = min over responses of  Q_B                against one alone

    dilemma = both - max(floorR, floorB)

Positive means D provably cannot hold both: whichever he takes, the other is
worth more than his single-target floor. That is the project's own
definition of a two-sided allocation dilemma, expressed in the model's own
quantities and nothing else.

Scored here against the `defender_reacts` labels — not because the trade-off
IS the reaction question, but because a defender who faces no trade-off at
all should not have needed to react, so the two should agree if both are
sound. Round 1 first.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
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
ROUND_FILES = {
    "1": [
        "derived_beneficiary_blind_labels_v1.csv",
        "derived_beneficiary_represent_round1fix.csv",
    ],
    "2": ["derived_beneficiary_round2_blind_labels.csv"],
}


def load_labels(round_key, target):
    """target 'reacts': does he respond to the runner at all.
    target 'dilemma': does he respond AND does someone benefit — the
    two-sided situation the trade-off is actually a measure of. A defender
    can react with nobody gaining (34387/Oberdorf, high confidence), so the
    two targets are not the same question and should not share a scorer.
    """
    labels = {}
    for name in ROUND_FILES[round_key]:
        path = REVIEWS / name
        if not path.exists():
            continue
        for row in csv.DictReader(path.open(encoding="utf-8-sig")):
            verdict = (row.get("defender_reacts") or "").strip().lower()
            if verdict not in ("yes", "no"):
                continue
            named = (row.get("beneficiary") or "").strip() not in ("", "none")
            value = int(verdict == "yes") if target == "reacts" else int(
                verdict == "yes" and named
            )
            labels[(row["match_id"], row["onset_frame_id"], row["defender_id"])] = value
    return labels


def auc(scores, labels):
    positives = [s for s, y in zip(scores, labels) if y == 1]
    negatives = [s for s, y in zip(scores, labels) if y == 0]
    if not positives or not negatives:
        return float("nan")
    return sum(
        1.0 if p > n else 0.5 if p == n else 0.0 for p in positives for n in negatives
    ) / (len(positives) * len(negatives))


def tradeoff(scene, defender_id):
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

    # Every priced response is a candidate answer; a response only counts if
    # it prices BOTH options, otherwise the min/max are over different sets.
    pairs = []
    for response in defender["responses"]:
        cells = response.get("cells") or {}
        runner_cell = cells.get(runner) or {}
        other_cell = cells.get(beneficiary) or {}
        if runner_cell.get("q") is None or other_cell.get("q") is None:
            continue
        if runner_cell.get("legal") is False or other_cell.get("legal") is False:
            continue
        pairs.append((float(runner_cell["q"]), float(other_cell["q"])))
    if len(pairs) < 2:
        return None

    both = min(max(a, b) for a, b in pairs)
    floor_runner = min(a for a, _ in pairs)
    floor_other = min(b for _, b in pairs)
    single = max(floor_runner, floor_other)
    cross = defender.get("cross_cost") or {}
    return {
        "R9 딜레마 (both - single)": both - single,
        "R9 딜레마 (상대값)": (both - single) / both if both > 1e-9 else 0.0,
        "쌍에 대한 최선 방어 both": both,
        "러너 단독 바닥": floor_runner,
        "수혜자 단독 바닥": floor_other,
        "기존 cross_cost strength": float(cross.get("strength") or 0.0),
        "기존 cross_cost runner": float(cross.get("runner") or 0.0),
        "러너까지 거리 (음수)": -math.dist(
            (float(state[defender_id]["x"]), float(state[defender_id]["y"])),
            (float(state[runner]["x"]), float(state[runner]["y"])),
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audit_dirs", nargs="+")
    parser.add_argument("--round", required=True, choices=["1", "2"])
    parser.add_argument("--target", default="reacts", choices=["reacts", "dilemma"])
    options = parser.parse_args()

    labels = load_labels(options.round, options.target)
    scenes = {}
    for directory in options.audit_dirs:
        for scene in json.loads(
            (Path(directory) / "local_game_payoff_audits.json").read_text()
        ):
            scenes[(str(scene["match_id"]), str(scene["onset_frame_id"]))] = scene

    columns, truth = {}, []
    positive_dilemma = 0
    for (match_id, frame_id, defender_id), reacts in sorted(labels.items()):
        scene = scenes.get((match_id, frame_id))
        if scene is None:
            continue
        row = tradeoff(scene, defender_id)
        if row is None:
            continue
        truth.append(reacts)
        positive_dilemma += row["R9 딜레마 (both - single)"] > 1e-6
        for key, value in row.items():
            columns.setdefault(key, []).append(value)

    positives = sum(truth)
    print(
        f"라운드 {options.round}  정답지={options.target}  n = {len(truth)}  "
        f"(양성 {positives} / 음성 {len(truth) - positives})"
    )
    print(f"딜레마 > 0 인 수비수-게임: {positive_dilemma} / {len(truth)}\n")
    print(f"{'특징':<26} {'AUC':>6}")
    for name, values in sorted(columns.items(), key=lambda item: -auc(item[1], truth)):
        print(f"{name:<26} {auc(values, truth):>6.3f}")


if __name__ == "__main__":
    main()
