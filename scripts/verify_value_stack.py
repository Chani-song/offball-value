#!/usr/bin/env python3
"""Gate the 588-scene run: does the adopted stack actually price scenes?

A wrong adapter does not crash. It returns a constant, or silently mirrors the
pitch, and the whole array run then produces 588 confidently wrong audits. So
this builds a handful of scenes under both stacks and checks the things that
would be broken if the coordinate conversion, the defender roles or the
family index were wrong:

  - every cell is finite and inside [0, 1];
  - Q is not constant across options -- a stuck model ranks nothing;
  - threat rises toward the goal the attack is actually attacking;
  - the two stacks disagree somewhere, because if they agree everywhere the
    switch is not doing anything.

Exit status is the gate: the array job is submitted with
--dependency=afterok, so a failure here means the 588 never start.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from offball_value.local_game_payoff import (
    LocalGamePayoffConfig,
    LocalGameStructureConfig,
    build_local_game_payoff_audit,
)
from offball_value.ssac_stack import threat_at_point


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--scenes-json", type=Path,
                   default=Path("data/processed/run_onset_v0_5/dir25/chunk0.json"))
    p.add_argument("--scene-indices", default="1,2,3")
    p.add_argument("--ssac-pass-model",
                   default="andrew/models/experimental_pass.json")
    return p.parse_args()


def payoff_config(stack: str, model_path: str) -> LocalGamePayoffConfig:
    # Field names, not the CLI's flag names: build_ssac.sbatch passes the same
    # settings through render_local_game_payoff_audit_v0_1.py, which renames them.
    return LocalGamePayoffConfig(
        release_step_seconds=0.4,
        influence_grid_resolution_m=3.0,
        response_search_mode="target_agnostic",
        local_game_selection_mode="pairwise_cross_cost",
        target_agnostic_raw_response_limit=160,
        response_screen_keep_per_option=8,
        response_screen_keep_minimax=16,
        response_screen_keep_low_effort=8,
        local_non_runner_limit=3,
        through_ball_lead_distances_m=(2.0, 4.0, 6.0),
        vacated_union_count=2,
        record_candidate_grid=True,
        delivery_model="hybrid",
        value_stack=stack,
        ssac_pass_model_path=model_path,
    )


def cells_of(audit: dict) -> list[dict]:
    out = []
    for game in audit.get("candidate_defenders") or []:
        for response in game.get("responses") or []:
            out.extend((response.get("cells") or {}).values())
    return out


def main() -> int:
    args = parse_args()
    scenes = json.loads(args.scenes_json.read_text())
    if isinstance(scenes, dict):
        scenes = scenes.get("scenes", [])
    indices = [int(v) for v in str(args.scene_indices).split(",") if v.strip()]

    failures: list[str] = []
    structural = LocalGameStructureConfig(candidate_defender_count=3)

    # --- threat orientation, independent of any scene -----------------------
    for direction in (1, -1):
        near = threat_at_point((direction * 45.0, 0.0), (0.0, 0.0),
                               (0.0, 30.0), direction)
        far = threat_at_point((-direction * 45.0, 0.0), (0.0, 0.0),
                              (0.0, 30.0), direction)
        print(f"[방향 {direction:+d}] 공격 골대 앞 {near:.4f} · 반대편 {far:.4f}")
        if not near > far + 0.2:
            failures.append(f"threat does not rise toward goal for {direction}")

    # --- per-scene, both stacks --------------------------------------------
    for index in indices:
        if index > len(scenes):
            failures.append(f"scene {index} missing from {args.scenes_json}")
            continue
        scene = scenes[index - 1]
        summary = {}
        for stack in ("native", "ssac"):
            try:
                audit = build_local_game_payoff_audit(
                    scene, payoff_config(stack, args.ssac_pass_model), structural)
            except Exception as error:  # noqa: BLE001 - the gate reports, not raises
                failures.append(f"scene {index} {stack}: {type(error).__name__} {error}")
                continue
            qs = np.array([float(c["q"]) for c in cells_of(audit)
                           if c.get("q") is not None and c.get("legal") is not False])
            if qs.size == 0:
                failures.append(f"scene {index} {stack}: no legal cells")
                continue
            if not np.isfinite(qs).all():
                failures.append(f"scene {index} {stack}: non-finite q")
            if qs.min() < -1e-9 or qs.max() > 1.0 + 1e-9:
                failures.append(f"scene {index} {stack}: q outside [0,1] "
                                f"({qs.min():.4f}..{qs.max():.4f})")
            if np.ptp(qs) < 1e-6:
                failures.append(f"scene {index} {stack}: q is constant, ranks nothing")
            summary[stack] = qs
            print(f"[장면 {index}] {stack:6} 셀 {qs.size:5}  "
                  f"q min {qs.min():.4f} 중앙 {np.median(qs):.4f} max {qs.max():.4f}")
        if len(summary) == 2:
            a, b = summary["native"], summary["ssac"]
            if a.size == b.size and np.allclose(a, b):
                failures.append(f"scene {index}: stacks identical, switch is inert")

    print()
    if failures:
        print("검증 실패:")
        for line in failures:
            print(f"  - {line}")
        return 1
    print("검증 통과 — 588 빌드로 진행합니다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
