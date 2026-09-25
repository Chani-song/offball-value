#!/usr/bin/env python3
"""One table from a solved stage-3 study: every number the pages read.

Replaces the chain of one-off steps that built the first results table, so a
second study is summarised the same way as the first. Reads the study's
manifest for the turn length, rows.jsonl for value and certificate, the
policy files for per-step splitting, and the states file for which triple
each state is.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from defensive_positioning.models import GameConfig

from offball_value.stage3_read import merged, modal_path, root_choices, split_by_step


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--solved", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--split-below", type=float, default=0.8)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    manifest = json.loads((args.solved / "manifest.json").read_text())
    steps = int(manifest["config"]["steps"])
    step_s = float(manifest["config"]["step_seconds"])
    config = GameConfig(**{k: v for k, v in manifest["config"].items()
                           if k in ("steps", "step_seconds", "physics_step")})
    starting = {s["index"]: s for s in
                json.loads((args.solved / "starting_states.json").read_text())["states"]}
    prov = {i: s.get("provenance", {}) for i, s in starting.items()}
    rows = []
    for line in (args.solved / "rows.jsonl").read_text().splitlines():
        r = json.loads(line)
        i = int(r["index"])
        p = prov.get(i, {})
        state = json.loads((args.solved / "states" / f"state_{i:03d}.json").read_text())
        agg = merged(root_choices(state))
        per = split_by_step(args.solved / "policies" / f"state_{i:03d}.npz", steps,
                            args.split_below)
        passer = (starting[i].get("passer") or {}).get("positions")
        path = modal_path(state, args.solved / "policies" / f"state_{i:03d}.npz", config,
                          passer_track=passer, physics=manifest.get("physics"))
        rec = {
            "index": i, "stratum": r["stratum"], "value": r["value"],
            "gap": r["certificate"]["gap"], "mixed_def": r["mixed_states"]["defender"],
            "step_seconds": step_s, "steps": steps,
            "match": str(p.get("match_label", ""))[:20], "match_id": p.get("match_id"),
            "onset": p.get("onset_frame_id"), "scene_dir": p.get("scene_dir"),
            "runner": p.get("runner_name"), "runner_id": p.get("runner_id"),
            "defender": p.get("defender_name"), "defender_id": p.get("defender_id"),
            "carrier": p.get("carrier_name"), "carrier_id": p.get("carrier_id"),
            # 3v1 names a beneficiary apart from the carrier; in 2v1 they are one man
            "beneficiary": p.get("beneficiary_name") or p.get("carrier_name"),
            "beneficiary_id": p.get("beneficiary_id") or p.get("carrier_id"),
            "root_dist": json.dumps([round(float(v), 3) for v in r["root_defender"]]),
            "fsplit0": 1.0 - agg[0][1] if agg else 0.0,
            "root_top": agg[0][0] if agg else "",
            "root_second": agg[1][0] if len(agg) > 1 else "",
            # along the most likely line of play (both sides take their heaviest
            # action at every decision): the most the defender is torn at any
            # of his decisions, and each decision in football terms
            "kind": path["kind"],
            "path_max_split": path["max_split"],
            "path_split_at": (max(path["points"], key=lambda q: q["split"])["t"]
                              if path["points"] else float("nan")),
            "path_points": json.dumps([{"t": q["t"], "merged": q["merged"],
                                        "split": round(q["split"], 4),
                                        "chosen": q["chosen_name"]} for q in path["points"]],
                                      ensure_ascii=False),
            "path_end": path["end"]["event"],
            "path_end_to": path["end"]["to"] or "",
        }
        for k, d in enumerate(per):
            rec[f"split_t{k}"] = d["split"]
            rec[f"mix_t{k}"] = d["loose"]
            rec[f"top_t{k}"] = d["top"]
        rows.append(rec)
    frame = pd.DataFrame(rows).sort_values("index")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(f"{len(frame)}장면 · 턴 {steps} × {step_s}s = {steps * step_s:.1f}s · "
          f"최대 gap {frame['gap'].max():.1e} → {args.output}")
    print("   갈리는 비율 (최선 방향 < {:.0%}): ".format(args.split_below) + " · ".join(
        f"{k * step_s:.1f}s {frame[f'split_t{k}'].mean():.1%}" for k in range(steps)))
    print(f"   시작 순간 서로 다른 대상 사이에서 갈림: {int((frame['fsplit0'] > 0.2).sum())}장면")
    print(f"   경기 흐름 중 어느 순간이든 갈림(가장 가능성 높은 흐름): {int((frame['path_max_split'] > 0.2).sum())}장면 · "
          + "가장 크게 갈린 시점 " + str(frame.loc[frame['path_max_split'] > 0.2, 'path_split_at']
                                           .value_counts().sort_index().to_dict()))


if __name__ == "__main__":
    main()
