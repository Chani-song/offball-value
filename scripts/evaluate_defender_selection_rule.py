#!/usr/bin/env python3
"""Evaluate the two-stage automatic dilemma-defender selection rule.

Rule under test (candidate for the population stage):

1. Geometry screen: rank outfield defenders by runner-cover marking cost and
   keep the top K (cheap, high recall required).
2. Value stage: run the payoff game for the screened defenders and pick the
   one with the maximum two-sided cross-cost strength; break an all-zero tie
   (every screened game runner-only) by the geometry rank.

Ground truth is the human-confirmed `confirmed_defender_ids` in the scene
manifest.  The value data comes from an existing multi-defender audit payload
so the evaluation is catalogue-consistent within itself.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audit-json",
        type=Path,
        default=Path(
            "data/processed/target_agnostic_local_game_v0_2_all_scenes_d3/"
            "local_game_payoff_audits.json"
        ),
        help="Multi-defender audit payload (defenders in geometry-rank order).",
    )
    parser.add_argument(
        "--scenes-json",
        type=Path,
        default=Path(
            "examples/research_audit/manifests/confirmed_core_scenes.json"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scenes = {
        int(scene["onset_frame_id"]): scene
        for scene in json.loads(args.scenes_json.read_text(encoding="utf-8"))
    }
    audits = json.loads(args.audit_json.read_text(encoding="utf-8"))

    hits = 0
    total = 0
    print(
        f"{'장면':<28}{'자동 선택':<20}{'사람 지정':<28}{'일치':>5}"
    )
    for audit in audits:
        frame_id = int(audit["onset_frame_id"])
        scene = scenes.get(frame_id, {})
        if scene.get("cohort") == "excluded":
            continue
        confirmed = [str(x) for x in scene.get("confirmed_defender_ids", [])]
        if not confirmed:
            continue
        defenders = audit["candidate_defenders"]  # geometry-rank order
        strengths = [
            float(defender["cross_cost"]["strength"]) for defender in defenders
        ]
        if max(strengths) > 0.0:
            pick = defenders[strengths.index(max(strengths))]
        else:
            pick = defenders[0]  # runner-only everywhere: geometry rank
        picked_id = str(pick["defender_id"])
        # A scene with two confirmed dilemmas counts as a hit when the
        # automatic pick is either of them.
        hit = picked_id in confirmed
        hits += int(hit)
        total += 1
        confirmed_names = " + ".join(
            next(
                (
                    str(d["defender_name"])
                    for d in defenders
                    if str(d["defender_id"]) == c
                ),
                c,
            )
            for c in confirmed
        )
        label = f"{audit['runner_name']}·{frame_id}"
        print(
            f"{label:<28}{pick['defender_name']:<20}{confirmed_names:<28}"
            f"{'O' if hit else 'X':>5}"
        )
        if not hit:
            detail = ", ".join(
                f"{d['defender_name']} {s:.3f}"
                for d, s in zip(defenders, strengths)
            )
            print(f"    strengths: {detail}")
    print(f"\n일치율: {hits}/{total}")


if __name__ == "__main__":
    main()
