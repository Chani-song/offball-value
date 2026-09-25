"""Stage 1: does this defender actually have to react to the runner?

The pipeline's three stages are the reviewer's: 수비수 선정 → 수혜자 선정 →
딜레마 판단. Almost all the work so far has been stage 2, scored on the 23-24
rows that carry a named beneficiary. But the same label files carry a
`defender_reacts` column on EVERY row — 51 usable yes/no in round 1 and 89 in
round 2 — and no rule has ever been scored against it.

That matters twice over. It is the stage that gates everything downstream,
and the auto-selection is known to over-generate: the reviewer answered "no"
on 65 of 96 round-2 rows.

The hypothesis under test comes free from R9. If a defender genuinely must
follow the runner, doing so should COST him real coverage; if a team-mate
already has that ground, or the runner is going nowhere near his
responsibility, following costs nothing. So the vacated area — the same
Σ_t |loss(t)| that R9 already computes as a normaliser — should separate yes
from no. Several incumbent model fields are scored alongside it as baselines.

Scores are reported as AUC (threshold-free, so it cannot be flattered by a
tuned cut-off) plus accuracy at the best round-1 threshold.

Usage:
    python scripts/score_defender_reacts.py <audit_dir> [...] --round 1
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
    _region_grid,
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
    "1": ["derived_beneficiary_blind_labels_v1.csv", "derived_beneficiary_represent_round1fix.csv"],
    "2": ["derived_beneficiary_round2_blind_labels.csv"],
}


def load_reacts(round_key: str) -> dict[tuple[str, str, str], int]:
    """(match, frame, defender) -> 1 if the reviewer said the defender reacts.

    'unsure' rows are dropped: they are not a third class, they are the
    reviewer declining to answer, and scoring them either way would invent
    a judgement he did not make.
    """
    labels: dict[tuple[str, str, str], int] = {}
    for name in ROUND_FILES[round_key]:
        path = REVIEWS / name
        if not path.exists():
            continue
        for row in csv.DictReader(path.open(encoding="utf-8-sig")):
            verdict = (row.get("defender_reacts") or "").strip().lower()
            if verdict not in ("yes", "no"):
                continue
            labels[
                (row["match_id"], row["onset_frame_id"], row["defender_id"])
            ] = int(verdict == "yes")
    return labels


def auc(scores: list[float], labels: list[int]) -> float:
    """Mann-Whitney AUC with ties counted as half."""
    positives = [s for s, y in zip(scores, labels) if y == 1]
    negatives = [s for s, y in zip(scores, labels) if y == 0]
    if not positives or not negatives:
        return float("nan")
    wins = sum(
        1.0 if p > n else 0.5 if p == n else 0.0 for p in positives for n in negatives
    )
    return wins / (len(positives) * len(negatives))


def apply_threshold(score: float, cut: float, direction: str) -> bool:
    return score >= cut if direction == "ge" else score < cut


def best_threshold(
    scores: list[float], labels: list[int]
) -> tuple[float, float, str]:
    """(accuracy, cut, direction) of the best single split — round 1 only.

    Direction is stored explicitly rather than folded into the sign of the
    cut: several features here are negated distances whose best cut-off is
    itself negative, and multiplying the two together loses the comparison
    direction entirely.
    """
    best = (0.0, 0.0, "ge")
    for cut in sorted(set(scores)):
        for direction in ("ge", "lt"):
            hits = sum(
                1
                for score, label in zip(scores, labels)
                if apply_threshold(score, cut, direction) == bool(label)
            )
            if hits / len(scores) > best[0]:
                best = (hits / len(scores), cut, direction)
    return best


def features(scene: dict, defender_id: str) -> dict[str, float] | None:
    """Everything scored for one defender-game."""
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
    defender_at, attacker_at = _coupled.absolute_lookups(scene, defender)
    if defender_at is None:
        return None

    team = attacking_team_id(scene)
    runner = str(scene["runner_id"])
    horizon = float(scene["horizon_seconds"])

    xs, ys = _region_grid(state[defender_id])
    others = np.zeros((len(ys), len(xs)))
    for player_id, row in state.items():
        if str(row["team"]) == team or player_id == defender_id:
            continue
        np.maximum(
            others, coverage_field(_state_xy(row), _state_v(row), xs, ys), out=others
        )
    own = coverage_field(
        _state_xy(state[defender_id]), _state_v(state[defender_id]), xs, ys
    )
    before = np.maximum(others, own)

    curves = {
        option_id: value_at_times(cell.get("candidate_grid") or [])
        for option_id, cell in response["cells"].items()
        if cell.get("legal") is not False
        and cell.get("q") is not None
        and option_id != runner
    }

    vacated = 0.0
    peak_loss = 0.0
    coupled_totals = {option_id: 0.0 for option_id in curves}
    time_s = SAMPLE_STEP_SECONDS
    while time_s <= horizon + 1e-9:
        after_xy, after_v = defender_at(time_s)
        loss = np.clip(
            before - np.maximum(others, coverage_field(after_xy, after_v, xs, ys)),
            0.0,
            None,
        )
        weight = float(loss.sum())
        if weight > 1e-12:
            vacated += weight
            peak_loss = max(peak_loss, weight)
            for option_id, curve in curves.items():
                value = value_near(curve, time_s)
                if value <= 0.0:
                    continue
                xy, velocity = attacker_at(option_id, time_s)
                coupled_totals[option_id] += (
                    float((loss * coverage_field(xy, velocity, xs, ys)).sum()) * value
                )
        time_s = round(time_s + SAMPLE_STEP_SECONDS, 6)

    own_area = float(own.sum())
    top_coupled = (
        max(coupled_totals.values()) / vacated if vacated > 1e-12 and coupled_totals else 0.0
    )
    cross = defender.get("cross_cost") or {}
    runner_row = state.get(runner)
    distance = (
        math.dist(_state_xy(state[defender_id]), _state_xy(runner_row))
        if runner_row is not None
        else float("nan")
    )
    return {
        # R9's own quantity: total coverage given up along the reaction.
        "비운 면적": vacated,
        # Normalised — a big defender zone trivially loses more absolute area.
        "비운 면적 / 본인 영역": vacated / own_area if own_area > 1e-9 else 0.0,
        "순간 최대 손실": peak_loss,
        # The best beneficiary's coupled score: is anyone positioned to use it?
        "최고 수혜자 점수": top_coupled,
        # Baselines the pipeline already computes.
        "러너까지 거리 (음수)": -distance,
        # Incumbent baselines the pipeline already computes for each defender.
        "cross_cost runner": float(cross.get("runner") or 0.0),
        "cross_cost strength": float(cross.get("strength") or 0.0),
        "minimax worst q": float(defender.get("exact_minimax_worst_q") or 0.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audit_dirs", nargs="+")
    parser.add_argument("--round", required=True, choices=["1", "2"])
    parser.add_argument("--threshold-json", default=None,
                        help="write (round 1) or apply (round 2) the chosen cut-offs")
    options = parser.parse_args()

    labels = load_reacts(options.round)
    scenes: dict[tuple[str, str], dict] = {}
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
    print(f"라운드 {options.round}  n = {len(truth)}  "
          f"(반응함 {positives} / 반응 안함 {len(truth) - positives})")
    majority = max(positives, len(truth) - positives) / len(truth)
    print(f"다수결 기준선 {majority:.1%}  "
          f"(항상 '반응함' {positives / len(truth):.1%} / "
          f"항상 '반응 안함' {1 - positives / len(truth):.1%})\n")

    stored = {}
    if options.threshold_json and options.round == "2" and Path(options.threshold_json).exists():
        stored = json.loads(Path(options.threshold_json).read_text())

    print(f"{'특징':<24} {'AUC':>6}   {'정확도':>7}   {'규칙':>22}")
    results = {}
    for name, values in columns.items():
        area = auc(values, truth)
        if options.round == "1":
            accuracy, cut, direction = best_threshold(values, truth)
            results[name] = {"cut": cut, "direction": direction}
        else:
            saved = stored.get(name)
            if saved is None:
                accuracy, cut, direction = float("nan"), float("nan"), "?"
            else:
                cut, direction = saved["cut"], saved["direction"]
                accuracy = sum(
                    1
                    for score, label in zip(values, truth)
                    if apply_threshold(score, cut, direction) == bool(label)
                ) / len(truth)
        rule = f"{'>=' if direction == 'ge' else '<'} {cut:.3f} 이면 반응"
        print(f"{name:<24} {area:>6.3f}   {accuracy:>7.1%}   {rule:>22}")

    if options.threshold_json and options.round == "1":
        Path(options.threshold_json).write_text(json.dumps(results, ensure_ascii=False, indent=2))
        print(f"\n임계값 저장: {options.threshold_json}  (라운드 2에서 그대로 적용)")


if __name__ == "__main__":
    main()
