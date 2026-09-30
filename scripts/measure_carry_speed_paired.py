#!/usr/bin/env python3
"""Does taking the ball slow a player down? Paired, same player, same seconds.

The first attempt compared ball carriers against all off-ball team-mates and
found carriers FASTER (99th percentile 7.72 against 6.46 m/s). That comparison
is confounded: the off-ball group includes everyone jogging back on the far
side, so it measures involvement, not the cost of carrying.

This pairs instead. At each moment a player takes control, compare his own
speed over the second BEFORE with his own speed over the second AFTER. Same
player, adjacent seconds, so fitness and role cancel and what is left is much
closer to the thing the arrival model needs: how much slower a player is with
the ball at his feet than without it.

Still not clean -- a player often takes the ball while decelerating to receive,
which biases the "after" downward -- so the gain at each horizon is reported
separately rather than collapsed into one number.

Usage:
    python scripts/measure_carry_speed_paired.py --output out/delivery_analysis/carry_speed_paired.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from offball_value.bundesliga import (
    FPS,
    find_bundesliga_files,
    list_bundesliga_match_ids,
    load_bundesliga_frame_clock,
    load_bundesliga_frames,
    load_bundesliga_match_metadata,
)

EXCLUDED_MATCHES = {"DFL-MAT-J03WN1"}
CONTROL_M = 1.5
STRIDE = 5
HORIZONS = (0.4, 1.0, 1.6)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", type=Path, default=Path("data/raw/bundesliga-integrated"))
    p.add_argument("--output", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dt = STRIDE / FPS
    pairs = {h: [] for h in HORIZONS}
    peak_with, peak_without = [], []

    for match_id in [m for m in list_bundesliga_match_ids(args.raw_dir)
                     if m not in EXCLUDED_MATCHES]:
        files = find_bundesliga_files(args.raw_dir, match_id)
        meta = load_bundesliga_match_metadata(files["matchinfo"])
        clock = load_bundesliga_frame_clock(files["positions"])
        keepers = {k for t in (meta.home_team_id, meta.away_team_id)
                   if (k := meta.goalkeeper_id(t)) is not None}
        targets = []
        for _s, (start, _t) in clock.section_start.items():
            targets.extend(range(int(start), int(start) + 75_000, STRIDE))
        frames = load_bundesliga_frames(files["positions"], targets)
        order = sorted(frames)
        index = {f: i for i, f in enumerate(order)}

        def owner(fid):
            frame = frames.get(fid)
            if frame is None or frame.ball is None or not frame.players:
                return None
            ball = (float(frame.ball.x), float(frame.ball.y))
            pid, st = min(frame.players.items(),
                          key=lambda kv: math.hypot(kv[1].x - ball[0], kv[1].y - ball[1]))
            if math.hypot(st.x - ball[0], st.y - ball[1]) > CONTROL_M or pid in keepers:
                return None
            return pid

        def speed(pid, fid):
            a, b = frames.get(fid - STRIDE), frames.get(fid)
            if a is None or b is None:
                return None
            p, q = a.players.get(pid), b.players.get(pid)
            if p is None or q is None:
                return None
            v = math.hypot(q.x - p.x, q.y - p.y) / dt
            return v if v <= 12.0 else None

        for i, fid in enumerate(order):
            if i < 12 or i + 12 >= len(order):
                continue
            here, before = owner(fid), owner(fid - STRIDE)
            if here is None or here == before:
                continue                      # only the instant control is taken
            for horizon in HORIZONS:
                span = int(round(horizon * FPS / STRIDE))
                pre = [speed(here, order[index[fid] - k]) for k in range(1, span + 1)]
                post = [speed(here, order[index[fid] + k]) for k in range(1, span + 1)]
                pre = [v for v in pre if v is not None]
                post = [v for v in post if v is not None]
                if len(pre) < span or len(post) < span:
                    continue
                if not all(owner(order[index[fid] + k]) == here for k in range(1, span + 1)):
                    continue              # must keep the ball for the whole window
                pairs[horizon].append((max(pre), max(post)))
        del frames
        print(f"  [{match_id}] total so far {len(pairs[1.0]):,}", flush=True)

    print(f"\n{'='*76}")
    print("Same player's top speed just before vs just after getting the ball")
    print("="*76)
    print(f"  {'window':<10}{'n':>9}{'before':>9}{'after':>9}{'ratio':>9}{'median aft/bef':>16}")
    out = {}
    for horizon in HORIZONS:
        v = np.array(pairs[horizon])
        if len(v) < 50:
            continue
        pre, post = v[:, 0], v[:, 1]
        ratio = post / np.maximum(pre, 1e-6)
        out[horizon] = {"n": int(len(v)), "pre_p99": float(np.percentile(pre, 99)),
                        "post_p99": float(np.percentile(post, 99)),
                        "ratio_median": float(np.median(ratio)),
                        "ratio_p99": float(np.percentile(post, 99) / np.percentile(pre, 99))}
        print(f"  ±{horizon:<8.1f}{len(v):>9,}{np.percentile(pre,99):>9.2f}"
              f"{np.percentile(post,99):>9.2f}"
              f"{np.percentile(post,99)/np.percentile(pre,99):>9.3f}"
              f"{np.median(ratio):>16.3f}")
    print("\n  'ratio' compares the 99th percentiles; the last column is the median of the per-pair ratios.")
    print("  Close to 1.0 means that getting the ball does not slow the player down.")
    print("  Caveat: players slowing down to receive are mixed in, which biases 'after' low.")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(out, ensure_ascii=False, indent=1))
        print(f"\nsaved {args.output}")


if __name__ == "__main__":
    main()
