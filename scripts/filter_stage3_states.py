#!/usr/bin/env python3
"""Which start states are fit for a reduced game at all?

Two scene-level filters, applied to a stage-3 states file (either study):

  run quality   the run's mean speed over the 0.8 s after onset reaches
                `--min-run-speed` (3.8 m/s, decided 2026-09-24). Calibrated
                on 58 reviewed runs: it drops 9 of the 12 the reviewer called
                "not a run" and keeps 25 of the 33 he called one.
  isolation     nobody else stands inside the triangle of the players the
                game is about, at any of the decision instants. The game
                reduces eleven against eleven to two or three against one; a
                player INSIDE that triangle is in the middle of the duel and
                would respond to it, which a reduced game cannot represent.
                2v1: carrier, runner, defender. 3v1: beneficiary, defender,
                runner -- the passer is not a vertex (his decision is not the
                one being studied) and, being in the game, is not "someone
                else" either. Proposed by the reviewer 2026-09-24.

Inside, or within `--buffer-m` of it. The triangle is usually thin -- the
defender stands roughly between the two attackers -- so a man a metre from
the defender can fall outside it. The default 0 is "inside" only. On
2026-09-25 the reviewer set 2 m, having seen a scene (Tanaka / Ananou) where
one defender stayed 1.8-2.5 m outside the triangle throughout and another
ended 3 cm outside it: to him both were in the duel. 2 m is his choice, not a
measured value, and is recorded as such.

Positions are the real tracking, linearly interpolated to each instant.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--states", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--build", type=Path,
                   default=Path(os.environ.get("OFFBALL_OUT_ROOT", Path(__file__).resolve().parents[1] / "out/runs")) / "v7_r9_ssac")
    p.add_argument("--onsets", default=str(ROOT / "data/processed/run_onset_v0_5/dir25/*/audit_selection.csv"))
    p.add_argument("--min-run-speed", type=float, default=3.8)
    p.add_argument("--buffer-m", type=float, default=0.0,
                   help="also drop a scene when someone is within this distance of the triangle")
    p.add_argument("--instants", default="0,0.6,1.2",
                   help="decision instants (s after onset) at which the triangle is checked")
    return p.parse_args()


def inside(p, a, b, c) -> bool:
    """p inside or on triangle abc (degenerate triangles contain nothing)."""
    def cross(u, v, w):
        return (v[0] - u[0]) * (w[1] - u[1]) - (v[1] - u[1]) * (w[0] - u[0])
    area = cross(a, b, c)
    if abs(area) < 1e-9:
        return False
    s = [cross(a, b, p), cross(b, c, p), cross(c, a, p)]
    return all(x >= 0 for x in s) if area > 0 else all(x <= 0 for x in s)


def distance_to_triangle(p, a, b, c) -> float:
    """0 inside, else the distance to the nearest edge (works for thin or
    degenerate triangles too, which contain nothing)."""
    if inside(p, a, b, c):
        return 0.0
    def seg(u, v):
        uv = v - u
        t = float(np.clip(np.dot(p - u, uv) / max(float(np.dot(uv, uv)), 1e-12), 0.0, 1.0))
        return float(np.hypot(*(p - (u + t * uv))))
    return min(seg(a, b), seg(b, c), seg(c, a))


def positions_at(frames, t0, t):
    """Every player's position at onset + t, interpolated between frames."""
    times = np.array([float(f["relative_time_s"]) - t0 for f in frames])
    t = float(np.clip(t, times.min(), times.max()))
    hi = int(np.searchsorted(times, t))
    lo = max(0, hi - 1)
    hi = min(hi, len(frames) - 1)
    w = 0.0 if times[hi] == times[lo] else (t - times[lo]) / (times[hi] - times[lo])
    a = {str(p[0]): np.array(p[2:4], float) for p in frames[lo]["players"]}
    b = {str(p[0]): np.array(p[2:4], float) for p in frames[hi]["players"]}
    return {k: a[k] + (b[k] - a[k]) * w for k in a if k in b}


def main() -> None:
    args = parse_args()
    instants = [float(x) for x in args.instants.split(",")]
    source = json.loads(args.states.read_text())
    onsets = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(args.onsets))], ignore_index=True)
    speed = {(str(r.match_id), int(r.frame_id), str(r.player_id)): float(r.post_mean_speed_mps)
             for r in onsets.itertuples()}
    kept, dropped, cache = [], [], {}
    for rec in source["states"]:
        prov = rec["provenance"]
        three = bool(prov.get("beneficiary_id")) and prov.get("beneficiary_id") != prov.get("carrier_id")
        key = (str(prov["match_id"]), int(prov["onset_frame_id"]), str(prov["runner_id"]))
        v = speed.get(key)
        if v is None:
            dropped.append({"index": rec["index"], "reason": "no onset speed"})
            continue
        if v < args.min_run_speed:
            dropped.append({"index": rec["index"], "reason": f"run speed {v:.2f} < {args.min_run_speed}"})
            continue
        sd = prov["scene_dir"]
        if sd not in cache:
            cache[sd] = json.loads((args.build / sd / "local_game_payoff_audits.json").read_text())[0]
        g = cache[sd]
        frames = g["background_frames"]
        t0 = float(min(frames, key=lambda f: abs(f["frame_id"] - g["onset_frame_id"]))["relative_time_s"])
        if three:
            verts = (str(prov["beneficiary_id"]), str(prov["defender_id"]), str(prov["runner_id"]))
            in_game = set(verts) | {str(prov["carrier_id"])}
        else:
            verts = (str(prov["carrier_id"]), str(prov["runner_id"]), str(prov["defender_id"]))
            in_game = set(verts)
        names = {str(p[0]): p[4] for p in frames[0]["players"]}
        intruder = None
        for t in instants:
            pos = positions_at(frames, t0, t)
            if not all(x in pos for x in verts):
                continue
            a, b, c = (pos[x] for x in verts)
            for pid, xy in pos.items():
                if pid in in_game:
                    continue
                dist = distance_to_triangle(xy, a, b, c)
                if dist <= args.buffer_m + 1e-12 and (dist > 0.0 or inside(xy, a, b, c)):
                    intruder = (t, pid, dist)
                    break
            if intruder:
                break
        if intruder:
            where = ("inside the triangle" if intruder[2] == 0.0
                     else f"{intruder[2]:.2f} m from the triangle")
            dropped.append({"index": rec["index"],
                            "reason": f"{names.get(intruder[1], intruder[1])} {where} at {intruder[0]:g} s"})
            continue
        kept.append(rec)
    out = {k: v for k, v in source.items() if k != "states"}
    out["states"] = kept
    out["filters"] = {"min_run_speed": args.min_run_speed, "triangle_instants": instants,
                      "triangle_buffer_m": args.buffer_m,
                      "source": str(args.states), "kept": len(kept), "dropped": dropped}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=1))
    runs = sum(1 for d in dropped if d["reason"].startswith("run speed"))
    tri = sum(1 for d in dropped if "triangle" in d["reason"])  # inside or within the buffer
    print(f"{len(source['states'])} → {len(kept)}  (dropped for run speed {runs} · triangle {tri} · "
          f"other {len(dropped) - runs - tri})  → {args.output}")


if __name__ == "__main__":
    main()
