"""Score the geometry-only dilemma against the reviewer's scene verdicts.

The value-based score turned out to be mostly a danger detector: a control
with no two-sided structure (median Q in the scene) reaches 0.624 against
the structural measure's 0.639. Goal danger and dilemma structure are
entangled in Q, and dividing or subtracting it out does not separate them.

This asks the structural question with no Q anywhere:

    min over A's feasible paths of max over time of
        max(1 - A's coverage of the runner, 1 - A's coverage of the beneficiary)

and reports the same control alongside, so the comparison is like for like.
If the geometric version clears the danger baseline it is doing something
the value version was not.

The defender paths are the ones the audit already stores; the beneficiary is
R9's pick, so stages 1 and 2 are untouched and only stage 3 changes.
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

from offball_value.assignment_rule import attacking_team_id, onset_state  # noqa: E402
from offball_value.coupled_beneficiary import rule_r9, value_at_times  # noqa: E402
from offball_value.geometric_dilemma import geometric_dilemma  # noqa: E402

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
STEP_S = 0.25


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


def tracker(scene, player_id, horizon):
    frames = sorted(
        (
            float(f["relative_time_s"]),
            {str(p[0]): (float(p[2]), float(p[3])) for p in f["players"]},
        )
        for f in scene["background_frames"]
    )

    def at(time_s: float):
        previous = current = frames[0]
        for frame in frames:
            current = frame
            if frame[0] >= time_s:
                break
            previous = frame
        a = previous[1].get(player_id)
        b = current[1].get(player_id)
        if a is None or b is None:
            return b or a or (0.0, 0.0)
        if current[0] == previous[0]:
            return b
        fraction = (time_s - previous[0]) / (current[0] - previous[0])
        return (a[0] + fraction * (b[0] - a[0]), a[1] + fraction * (b[1] - a[1]))

    return at


def defender_row(scene, defender, times, horizon):
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
        horizon,
    )
    if not beneficiary:
        return None

    paths = [
        row["path_txy"]
        for row in defender["responses"]
        if row.get("path_txy")
    ]
    result = geometric_dilemma(
        paths,
        tracker(scene, runner, horizon),
        tracker(scene, beneficiary, horizon),
        times,
    )
    if not result:
        return None

    # The danger control, identical to the one used on the value measure.
    values = [
        float(cell["q"])
        for response in defender["responses"]
        for cell in (response.get("cells") or {}).values()
        if cell.get("q") is not None and cell.get("legal") is not False
    ]
    result["장면 중앙 Q (대조군)"] = float(np.median(values)) if values else 0.0
    # And the value-based knee, so all three sit in one table.
    points = []
    for response in defender["responses"]:
        cells = response.get("cells") or {}
        a, b = cells.get(runner) or {}, cells.get(beneficiary) or {}
        if a.get("q") is None or b.get("q") is None:
            continue
        if a.get("legal") is False or b.get("legal") is False:
            continue
        points.append((float(a["q"]), float(b["q"])))
    if len(points) >= 3:
        floor_x = min(x for x, _ in points)
        floor_y = min(y for _, y in points)
        knee = min(max(x, y) for x, y in points)
        result["가치 무릎값"] = knee
        result["가치 간극"] = knee - max(floor_x, floor_y)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audit_dirs", nargs="+")
    options = parser.parse_args()

    qc = load_scene_qc()
    scenes = {}
    for directory in options.audit_dirs:
        for scene in json.loads(
            (Path(directory) / "local_game_payoff_audits.json").read_text()
        ):
            scenes[(str(scene["match_id"]), str(scene["onset_frame_id"]))] = scene

    per_scene, truth, matches = [], [], []
    for key, scene in sorted(scenes.items()):
        verdict = qc.get(key)
        if verdict not in ("clear", "possible", "unclear", "none"):
            continue
        horizon = float(scene["horizon_seconds"])
        times, time_s = [], STEP_S
        while time_s <= horizon + 1e-9:
            times.append(time_s)
            time_s = round(time_s + STEP_S, 6)
        rows = [
            row
            for row in (
                defender_row(scene, defender, times, horizon)
                for defender in scene["candidate_defenders"]
            )
            if row
        ]
        rows = [r for r in rows if "가치 무릎값" in r]
        if not rows:
            continue
        per_scene.append({n: max(r[n] for r in rows) for n in rows[0]})
        truth.append(1 if verdict in ("clear", "possible") else 0)
        matches.append(key[0])

    print(f"장면 {len(per_scene)}개 (양성 {sum(truth)} / 음성 {len(truth)-sum(truth)})\n")
    print(f"{'측정치':<26} {'AUC':>7}")
    columns = {n: [r[n] for r in per_scene] for n in per_scene[0] if n != "paths"}
    for name in sorted(columns, key=lambda n: -auc(columns[n], truth)):
        tag = ""
        if name.startswith("geometric"):
            tag = "  <-- 기하 (Q 없음)"
        elif "대조군" in name:
            tag = "  <-- 기준선"
        print(f"{name:<26} {auc(columns[name], truth):>7.3f}{tag}")

    print("\n경기별 (geometric_knee)")
    groups = defaultdict(list)
    for row, label, match in zip(per_scene, truth, matches):
        groups[match].append((row["geometric_knee"], label))
    for match, rows in sorted(groups.items()):
        value = auc([s for s, _ in rows], [y for _, y in rows])
        print(f"  {match[-6:]}  n={len(rows):>3}  AUC {value:.3f}")


if __name__ == "__main__":
    main()
