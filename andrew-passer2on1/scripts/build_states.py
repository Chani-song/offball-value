#!/usr/bin/env python3
"""Add each scene's background defenders to the 2v1 start states.

Takes the states the original solver already ran on
(`data/processed/stage3/carrier_beneficiary_states_v2.json`, 167 states:
pair gate kept, runner is the protagonist, beneficiary is the carrier,
carrier speed limit 7.2 m/s) and adds, to each, where every other defender
actually was at t = 0, 1, 2, 3 s. The strategic part of every record --
carrier, runner, controlled defender, limits -- is copied untouched, so any
difference in the solved game is the background and nothing else.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from passer2on1.tracks import background_tracks

ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--states", type=Path,
                   default=ROOT / "data/processed/stage3/carrier_beneficiary_states_v2.json")
    p.add_argument("--build", type=Path,
                   default=Path("/work/hdd/bbmr/kseo1/offball-out/v7_r9_ssac"))
    p.add_argument("--output", type=Path,
                   default=ROOT / "data/processed/stage3/passer2on1_states.json")
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--step-seconds", type=float, default=1.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    source = json.loads(args.states.read_text())
    records, cache = [], {}
    counts, dropped = [], 0
    for rec in source["states"]:
        prov = rec["provenance"]
        if prov["scene_dir"] not in cache:
            cache[prov["scene_dir"]] = json.loads(
                (args.build / prov["scene_dir"] / "local_game_payoff_audits.json").read_text())[0]
        payload = cache[prov["scene_dir"]]
        strategic = {prov["carrier_id"], prov["runner_id"], prov["defender_id"]}
        bg = background_tracks(payload, strategic, args.steps, args.step_seconds)
        out = dict(rec)
        out["background"] = {
            "ids": bg["ids"], "names": bg["names"],
            "positions": bg["positions"].round(4).tolist(),
            "velocities": bg["velocities"].round(4).tolist(),
            "times_requested": bg["times_requested"], "times_used": bg["times_used"],
            "dropped": bg["dropped"],
        }
        records.append(out)
        counts.append(len(bg["ids"]))
        dropped += len(bg["dropped"])
    meta = {k: v for k, v in source.items() if k != "states"}
    meta["background"] = {
        "what": "every defender except the controlled one, keeper included, at each decision "
                "instant; linear interpolation between tracking frames; no decisions",
        "steps": args.steps, "step_seconds": args.step_seconds,
        "source_states": str(args.states),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"states": records, **meta}, indent=1))
    print(f"{len(records)} states → {args.output}")
    print(f"   background defenders per state: min {min(counts)} · max {max(counts)} · "
          f"dropped for missing tracking {dropped}")


if __name__ == "__main__":
    main()
