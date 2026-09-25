"""Scenes the model calls strong dilemmas and the reviewer did not.

The dilemma measure is A's own minimax over his feasible paths:
min over paths of max(Q_runner, Q_beneficiary). The scenes where it fires
hardest but the QC verdict was `none` or `unclear` are the ones worth
looking at — either the model is reading something that is not there, or
the verdict was a close call the reviewer would revisit. Half of them were
`unclear`, which is him saying he could not tell.

Renders through the existing miss-review viewer, so the candidate table
carries the dilemma decomposition instead of beneficiary occupancy: for each
candidate defender, what the beneficiary gets if he chases the runner, what
the runner gets if he covers the beneficiary, and the minimax between them.

Usage:
    python scripts/export_dilemma_false_positives.py <audit_dir> [...] \
        --out fp.json --limit 8
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from offball_value.assignment_rule import attacking_team_id, onset_state  # noqa: E402
from offball_value.coupled_beneficiary import (  # noqa: E402
    SAMPLE_STEP_SECONDS,
    rule_r9,
    value_at_times,
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


def defender_row(scene, defender):
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
    return {
        "defender_id": defender_id,
        "defender_name": str(defender["defender_name"]),
        "beneficiary": beneficiary,
        "knee": min(max(x, y) for x, y in points),
        "chase_runner_gives": min(y for x, y in points if x <= floor_x + 1e-12),
        "cover_other_gives": min(x for x, y in points if y <= floor_y + 1e-12),
        "responses": len(points),
        "defender_at": defender_at,
    }


def vacated_cells(scene, state, team, defender_id, defender_at, horizon):
    xs, ys = _region_grid(state[defender_id])
    others = np.zeros((len(ys), len(xs)))
    for player_id, row in state.items():
        if str(row["team"]) == team or player_id == defender_id:
            continue
        np.maximum(
            others, coverage_field(_state_xy(row), _state_v(row), xs, ys), out=others
        )
    before = np.maximum(
        others,
        coverage_field(
            _state_xy(state[defender_id]), _state_v(state[defender_id]), xs, ys
        ),
    )
    total = np.zeros_like(before)
    time_s = SAMPLE_STEP_SECONDS
    while time_s <= horizon + 1e-9:
        after_xy, after_v = defender_at(time_s)
        total += np.clip(
            before - np.maximum(others, coverage_field(after_xy, after_v, xs, ys)),
            0.0,
            None,
        )
        time_s = round(time_s + SAMPLE_STEP_SECONDS, 6)
    peak = float(total.max())
    if peak <= 1e-12:
        return []
    return [
        {"x": round(float(x), 2), "y": round(float(y), 2), "v": round(float(total[r, c]) / peak, 3)}
        for r, y in enumerate(ys)
        for c, x in enumerate(xs)
        if total[r, c] / peak > 0.06
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audit_dirs", nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument(
        "--frame", action="append", default=[],
        help="Restrict to these onset frames; overrides the ranked selection.",
    )
    parser.add_argument(
        "--any-verdict", action="store_true",
        help="Keep clear/possible scenes too, so false NEGATIVES can be reviewed.",
    )
    parser.add_argument(
        "--lane-limit-m", type=float, default=None,
        help="Only score defenders within this distance of the carrier->runner lane.",
    )
    options = parser.parse_args()

    qc = load_scene_qc()
    scenes = {}
    for directory in options.audit_dirs:
        for scene in json.loads(
            (Path(directory) / "local_game_payoff_audits.json").read_text()
        ):
            scenes[(str(scene["match_id"]), str(scene["onset_frame_id"]))] = scene

    ranked = []
    for key, scene in sorted(scenes.items()):
        verdict = qc.get(key)
        allowed = ("none", "unclear", "clear", "possible") if options.any_verdict else ("none", "unclear")
        if verdict not in allowed:
            continue
        if options.frame and key[1] not in options.frame:
            continue
        state_now = onset_state(scene)
        carrier_id, runner_id = str(scene["carrier_id"]), str(scene["runner_id"])
        lane_ok = {}
        if options.lane_limit_m is not None and carrier_id in state_now and runner_id in state_now:
            start = np.array([state_now[carrier_id]["x"], state_now[carrier_id]["y"]], dtype=float)
            end = np.array([state_now[runner_id]["x"], state_now[runner_id]["y"]], dtype=float)
            lane_vector = end - start
            length = float(np.hypot(*lane_vector))
            for defender in scene["candidate_defenders"]:
                pid = str(defender["defender_id"])
                if pid not in state_now or length < 1e-6:
                    lane_ok[pid] = False
                    continue
                point = np.array([state_now[pid]["x"], state_now[pid]["y"]], dtype=float)
                along = float(np.dot(point - start, lane_vector) / length**2)
                foot = start + np.clip(along, 0.0, 1.0) * lane_vector
                lane_ok[pid] = float(np.hypot(*(point - foot))) <= options.lane_limit_m
        rows = [
            row
            for row in (
                defender_row(scene, defender)
                for defender in scene["candidate_defenders"]
                if lane_ok.get(str(defender["defender_id"]), True)
            )
            if row is not None
        ]
        if not rows:
            continue
        top = max(rows, key=lambda r: r["knee"])
        ranked.append((top["knee"], key, scene, rows, top, verdict))
    ranked.sort(key=lambda item: -item[0])

    payload = []
    for _, key, scene, rows, top, verdict in ranked[: options.limit]:
        state = onset_state(scene)
        team = attacking_team_id(scene)
        horizon = float(scene["horizon_seconds"])
        names = {str(p[0]): str(p[4]) for p in scene["background_frames"][0]["players"]}
        candidates = []
        for row in sorted(rows, key=lambda r: -r["knee"]):
            candidates.append(
                {
                    "name": (
                        f"{row['defender_name']} → {names.get(row['beneficiary'], row['beneficiary'])}"
                    ),
                    "occ": f"{row['chase_runner_gives']:.3f}",
                    "P": f"{row['cover_other_gives']:.3f}",
                    "G": f"{row['responses']}",
                    "score": f"{row['knee']:.4f}",
                    "is_answer": False,
                    "is_pick": row["defender_id"] == top["defender_id"],
                    "is_carrier": row["beneficiary"] == str(scene["carrier_id"]),
                }
            )
        payload.append(
            {
                "scene": f"{scene['match_label']} · {key[1]} · QC={verdict}",
                "direction": int(scene.get("attacking_direction") or 1),
                "attacking_team_id": team,
                "runner": str(scene["runner_name"]),
                "runner_id": str(scene["runner_id"]),
                "carrier": str(scene["carrier_name"]),
                "carrier_id": str(scene["carrier_id"]),
                "defender": top["defender_name"],
                "defender_id": top["defender_id"],
                "answer": f"형님 판정: {verdict}",
                "answer_id": "",
                "pick": names.get(top["beneficiary"], top["beneficiary"]),
                "pick_id": top["beneficiary"],
                "candidates": candidates,
                "cells": vacated_cells(
                    scene, state, team, top["defender_id"], top["defender_at"], horizon
                ),
                "defender_path": [
                    [round(float(p[0]), 3), round(float(p[1]), 2), round(float(p[2]), 2)]
                    for p in next(
                        row["path_txy"]
                        for row in next(
                            d
                            for d in scene["candidate_defenders"]
                            if str(d["defender_id"]) == top["defender_id"]
                        )["responses"]
                        if row.get("kind") == "target_conditioned_baseline"
                    )
                ],
                "frames": [
                    {
                        "t": round(float(f["relative_time_s"]), 3),
                        "players": f["players"],
                        "ball": f.get("ball"),
                    }
                    for f in sorted(
                        scene["background_frames"],
                        key=lambda f: float(f["relative_time_s"]),
                    )
                ],
            }
        )

    options.out.write_text(json.dumps(payload), encoding="utf-8")
    print(f"거짓 양성 {len(payload)}개 -> {options.out}")
    for entry in payload:
        print(
            f"  {entry['scene'].split('·')[-2].strip()} "
            f"{entry['defender']} → {entry['pick']}  ({entry['candidates'][0]['score']})"
        )


if __name__ == "__main__":
    main()
