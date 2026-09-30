#!/usr/bin/env python3
"""How hard real players speed up, brake and turn -- the agile physics' three limits, from our tracking.

The meeting solver's agile physics (agile_motion.AGILE, 2026-09-25) caps a body's change of velocity
by an ellipse in his own frame: speeding up 4.5, braking 6.0, turning 6.0 m/s^2, plus a plant-and-cut
braking at 9.0. None of these has a data source (the module says so). This measures the same three
quantities on every outfield player of the seven DFL matches, so the solver can be run with limits
real players reach (2026-09-28, the user's request; the 99th percentile is the user's pick of the two
offered, 99 or 95).

For each outfield player and half (goalkeepers, referees and the ball left out), on each run of
consecutive frames: velocity and acceleration from a Savitzky-Golay fit (quadratic, 11 frames =
0.44 s -- the smoothing is mine, chosen near the 0.4 s window the state builders already read
velocities over). Where he moves at 1 m/s or more (below that his heading is noise; my cut) the
acceleration is split along his velocity (+ speeding up, - braking) and across it (turning).

Tracking glitches (a position that jumps between two frames) put 200+ m/s^2 into the first run's top
band, so a run is also cut wherever one frame's step implies more than 12.4 m/s -- the fastest a
human has been timed (Bolt, Berlin 2009) -- and each side is fitted on its own.

A second measure matches the solver's turn: over 0.6 s (one decision step) how much his velocity
changes along and across his heading at the start, divided by 0.6. With a constant cap a the solver
can change velocity by at most a x 0.6 in a turn, so this is the effective cap a turn needs.

Reads the raw position XML line by line (one <Frame> per line), which is far faster than a tree parse.

Usage:
    python scripts/measure_accelerations.py --output data/processed/physics_limits/report.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter

FPS = 25.0
FRAMESET = re.compile(r'<FrameSet GameSection="(\w+)"[^>]*TeamId="([^"]+)" PersonId="([^"]+)"')
FRAME = re.compile(r'<Frame N="(\d+)"[^>]*? X="(-?[\d.]+)" Y="(-?[\d.]+)"')
PLAYER = re.compile(r'<Player PersonId="([^"]+)"[^>]*PlayingPosition="(\w*)"')
PERCENTILES = (50, 90, 95, 99, 99.9)
GLITCH_SPEED = 12.4        # m/s, the human sprint record: a faster frame-to-frame step is a tracking jump
TURN_S = 0.6               # the solver's decision step
SPEED_BANDS = ((1, 3), (3, 5), (5, 7), (7, 99))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", type=Path, default=Path("data/raw/bundesliga-integrated"))
    p.add_argument("--window-frames", type=int, default=11)
    p.add_argument("--min-speed", type=float, default=1.0)
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def tracks(positions: Path, keepers: set[str]):
    """(person, half, frames, x, y) for every outfield player's FrameSet."""
    who = None
    n, xs, ys = [], [], []
    with positions.open(encoding="utf-8") as fh:
        for line in fh:
            if "<Frame " in line:
                if who is not None:
                    m = FRAME.search(line)
                    n.append(int(m.group(1))); xs.append(float(m.group(2))); ys.append(float(m.group(3)))
                continue
            m = FRAMESET.search(line)
            if m:
                half, team, person = m.groups()
                who = (person, half) if team.startswith("DFL-CLU-") and person not in keepers else None
                n, xs, ys = [], [], []
            elif "</FrameSet>" in line and who is not None:
                yield who[0], who[1], np.array(n), np.array(xs), np.array(ys)
                who = None


def components(n, x, y, window: int, min_speed: float):
    """(speed, along, across) per frame, and (along, across) of the 0.6 s velocity change, on every
    clean run of consecutive frames long enough to fit."""
    out, turns = [], []
    step = np.hypot(np.diff(x), np.diff(y)) * FPS
    breaks = np.flatnonzero((np.diff(n) != 1) | (step > GLITCH_SPEED)) + 1
    lag = round(TURN_S * FPS)
    for seg in np.split(np.arange(len(n)), breaks):
        if len(seg) < window:
            continue
        d = 1.0 / FPS
        vx, vy = (savgol_filter(c[seg], window, 2, deriv=1, delta=d) for c in (x, y))
        ax, ay = (savgol_filter(c[seg], window, 2, deriv=2, delta=d) for c in (x, y))
        speed = np.hypot(vx, vy)
        ok = speed >= min_speed
        ux, uy = vx[ok] / speed[ok], vy[ok] / speed[ok]
        out.append(np.column_stack([speed[ok], ax[ok] * ux + ay[ok] * uy, np.abs(ay[ok] * ux - ax[ok] * uy)]))
        if len(seg) > lag:
            dvx, dvy = vx[lag:] - vx[:-lag], vy[lag:] - vy[:-lag]
            s0 = speed[:-lag]
            k = s0 >= min_speed
            hx, hy = vx[:-lag][k] / s0[k], vy[:-lag][k] / s0[k]
            turns.append(np.column_stack([s0[k], (dvx[k] * hx + dvy[k] * hy) / TURN_S,
                                          np.abs(dvy[k] * hx - dvx[k] * hy) / TURN_S]))
    empty = np.zeros((0, 3))
    return (np.vstack(out) if out else empty), (np.vstack(turns) if turns else empty)


def summary(values: np.ndarray) -> dict:
    return {"n": int(values.size), **{f"p{q:g}": round(float(np.percentile(values, q)), 2) for q in PERCENTILES}}


def main() -> None:
    args = parse_args()
    rows, turn_rows, per_match = [], [], {}
    for positions in sorted(args.raw_dir.glob("DFL_04_03_positions_raw_observed_*.xml")):
        match = positions.stem.split("_")[-1]
        info = next(args.raw_dir.glob(f"DFL_02_01_matchinformation_*_{match}.xml")).read_text(encoding="utf-8")
        keepers = {pid for pid, pos in PLAYER.findall(info) if pos == "TW"}
        both = [components(n, x, y, args.window_frames, args.min_speed)
                for _, _, n, x, y in tracks(positions, keepers)]
        comp = np.vstack([c for c, _ in both])
        turn_rows.append(np.vstack([t for _, t in both]))
        per_match[match] = {"frames": int(len(comp)), "keepers_left_out": len(keepers),
                            "speed_up_p99": round(float(np.percentile(comp[comp[:, 1] > 0, 1], 99)), 2),
                            "braking_p99": round(float(np.percentile(-comp[comp[:, 1] < 0, 1], 99)), 2),
                            "turning_p99": round(float(np.percentile(comp[:, 2], 99)), 2)}
        print(match, per_match[match], flush=True)
        rows.append(comp)
    allc = np.vstack(rows)
    speed, along, across = allc.T
    t_speed, t_along, t_across = np.vstack(turn_rows).T
    report = {
        "what": "outfield players' acceleration split along/across their velocity, m/s^2",
        "method": {"smoothing": f"Savitzky-Golay, quadratic, {args.window_frames} frames at {FPS:g} fps",
                   "min_speed_m_s": args.min_speed, "glitch_speed_m_s": GLITCH_SPEED,
                   "matches": sorted(per_match)},
        "speed_up": summary(along[along > 0]),
        "braking": summary(-along[along < 0]),
        "turning": summary(across),
        "over_one_turn": {"what": f"velocity change over {TURN_S} s along/across the starting heading, / {TURN_S}",
                          "speed_up": summary(t_along[t_along > 0]), "braking": summary(-t_along[t_along < 0]),
                          "turning": summary(t_across)},
        "by_speed": {f"{lo}-{hi} m/s": {"speed_up_p99": round(float(np.percentile(along[(speed >= lo) & (speed < hi) & (along > 0)], 99)), 2),
                                        "braking_p99": round(float(np.percentile(-along[(speed >= lo) & (speed < hi) & (along < 0)], 99)), 2),
                                        "turning_p99": round(float(np.percentile(across[(speed >= lo) & (speed < hi)], 99)), 2)}
                     for lo, hi in SPEED_BANDS},
        "per_match": per_match,
        "agile_now": {"speed_up": 4.5, "braking": 6.0, "turning": 6.0, "plant_braking": 9.0},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1))
    print(json.dumps({k: report[k] for k in ("speed_up", "braking", "turning", "over_one_turn", "by_speed")},
                     indent=1))


if __name__ == "__main__":
    main()
