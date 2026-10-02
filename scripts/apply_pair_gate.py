#!/usr/bin/env python3
"""Apply the pair plausibility gate to a finished build and list the triples.

Post-filter: the local games are already built for every candidate, so this
only decides which of them are worth a reviewer's and the solver's time.
Writes one row per candidate with the gate features, the verdict and the
reason, so the review page and the hand-off both read from one file. The
same gate goes into candidate selection for the next build, where it will
save the compute for the pairs it drops.
"""

from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import os
from pathlib import Path

import pandas as pd

from offball_value.pair_plausibility import (
    PairGateConfig, ProtagonistConfig, gate_game, runner_is_protagonist,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--build", type=Path,
                   default=Path(os.environ.get("OFFBALL_OUT_ROOT", Path(__file__).resolve().parents[1] / "out/runs")) / "v7_r9_ssac")
    p.add_argument("--output", type=Path,
                   default=Path("data/processed/pair_gate_v7.csv"))
    p.add_argument("--delta", type=float, default=3.0)
    p.add_argument("--catch-margin", type=float, default=3.0)
    p.add_argument("--catch-end", type=float, default=11.0)
    p.add_argument("--near", type=float, default=15.0)
    p.add_argument("--protagonist-horizon", type=float, default=2.0)
    p.add_argument("--protagonist-min-goalward", type=float, default=2.0)
    p.add_argument("--protagonist-carrier-ratio", type=float, default=1.3)
    return p.parse_args()


def load_labels() -> set:
    spec = importlib.util.spec_from_file_location(
        "sv", Path(__file__).with_name("score_vacated_space_rule.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return set(m.load_labels())


def main() -> None:
    args = parse_args()
    cfg = PairGateConfig(neighbour_delta_m=args.delta,
                         catch_margin_m=args.catch_margin,
                         catch_end_m=args.catch_end,
                         near_m=args.near)
    pcfg = ProtagonistConfig(horizon_s=args.protagonist_horizon,
                             min_goalward_m=args.protagonist_min_goalward,
                             min_carrier_ratio=args.protagonist_carrier_ratio)
    labels = load_labels()
    rows = []
    for path in sorted(args.build.glob("scene_*/local_game_payoff_audits.json")):
        try:
            payload = json.load(open(path))
        except Exception:
            continue
        if not payload:
            continue
        g = payload[0]
        names = {str(p[0]): p[4] for p in g["background_frames"][0]["players"]}
        who = runner_is_protagonist(g, pcfg)
        by_id = {str(d["defender_id"]): d for d in g.get("candidate_defenders") or []}
        for r in gate_game(g, cfg):
            d = by_id[r["defender_id"]]
            ben = d.get("derived_option_id")
            rows.append({
                "match_id": str(g["match_id"]),
                "match_label": str(g.get("match_label", "")).split(" · ")[0],
                "onset_frame_id": int(g["onset_frame_id"]),
                "scene_dir": path.parent.name,
                "runner_id": str(g["runner_id"]),
                "runner_name": g.get("runner_name"),
                "carrier_name": g.get("carrier_name"),
                "defender_id": r["defender_id"],
                "defender_name": d.get("defender_name"),
                "rank": r["rank"] + 1,
                "near_m": round(r["near_m"], 2),
                "path_min_m": round(r["path_min_m"], 2),
                "path_end_m": round(r["path_end_m"], 2),
                "catch_m": round(r["catch_m"], 2),
                "delta_m": round(r["delta_m"], 2),
                "goal_side_m": round(r["goal_side_m"], 2),
                "is_ball_nearest": bool(r["is_ball_nearest"]),
                "kept": r["kept"],
                "reason": r["reason"],
                "beneficiary_id": ben,
                "beneficiary_name": names.get(str(ben), ""),
                "rule_branch": d.get("derived_rule_branch"),
                "labelled": (str(g["match_id"]), str(g["onset_frame_id"]),
                             r["defender_id"]) in labels,
                "runner_disp_m": round(who["runner_disp_m"], 2),
                "runner_goalward_m": round(who["runner_goalward_m"], 2),
                "carrier_disp_m": round(who["carrier_disp_m"], 2),
                "protagonist": bool(who["protagonist"]),
                # the gate decides the defender, protagonist decides the scene
                "kept_final": bool(r["kept"]) and bool(who["protagonist"]),
            })
    frame = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)

    kept = frame[frame["kept"]]
    per = kept.groupby(["match_id", "onset_frame_id", "runner_id"]).size()
    dist = collections.Counter(per.values)
    print(f"candidates {len(frame)} → kept {len(kept)} ({len(kept)/len(frame):.0%}) · "
          f"scenes {per.size} · per scene {len(kept)/per.size:.2f}")
    print("defenders per scene: " + " · ".join(f"{k}: {dist[k]} ({dist[k]/per.size:.0%})"
                                     for k in sorted(dist)))
    print("reasons: " + " · ".join(f"{k} {v}" for k, v in
                              kept["reason"].value_counts().items()))
    print(f"labels kept {int(kept['labelled'].sum())}/{int(frame['labelled'].sum())}")
    fin = frame[frame["kept_final"]]
    sc = frame.drop_duplicates(["match_id", "onset_frame_id", "runner_id"])
    perf = fin.groupby(["match_id", "onset_frame_id", "runner_id"]).size()
    print(f"\n+ protagonist rule (C): scenes {int(sc['protagonist'].sum())}/{len(sc)} · "
          f"pairs {len(kept)} → {len(fin)} · per scene {len(fin)/max(perf.size,1):.2f} · "
          f"labels {int(fin['labelled'].sum())}/{int(frame['labelled'].sum())}")
    print(f"→ {args.output}")


if __name__ == "__main__":
    main()
