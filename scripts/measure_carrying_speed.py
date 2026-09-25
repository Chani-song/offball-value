#!/usr/bin/env python3
"""How fast does a player actually move while carrying the ball?

A point-wise carry value needs the carrier's arrival time at a cell, and the
arrival model in the pipeline caps everyone at player_max_speed_mps = 9.0 --
a free sprint. Carrying is slower, and by how much is a number nobody in this
repo has ever set on evidence. Rather than invent one, measure it: take every
sampled instant where a player is in foot control of the ball, and compare his
speed with the speed of the same side's players who are not.

Reported as a ratio so it can be used as a multiplier on the existing arrival
model rather than as a new absolute constant, and split by whether a defender
is closing, because a pressed carrier slows for reasons the cap should not be
made to absorb.

Usage:
    python scripts/measure_carrying_speed.py --output out/delivery_analysis/carry_speed.json
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", type=Path, default=Path("data/raw/bundesliga-integrated"))
    p.add_argument("--output", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dt = STRIDE / FPS
    carrier_speed, teammate_speed, carrier_pressed, carrier_free = [], [], [], []
    carrier_top = []

    for match_id in [m for m in list_bundesliga_match_ids(args.raw_dir)
                     if m not in EXCLUDED_MATCHES]:
        files = find_bundesliga_files(args.raw_dir, match_id)
        meta = load_bundesliga_match_metadata(files["matchinfo"])
        clock = load_bundesliga_frame_clock(files["positions"])
        keepers = {k for t in (meta.home_team_id, meta.away_team_id)
                   if (k := meta.goalkeeper_id(t)) is not None}
        targets = []
        for _sec, (start, _t) in clock.section_start.items():
            targets.extend(range(int(start), int(start) + 75_000, STRIDE))
        frames = load_bundesliga_frames(files["positions"], targets)
        order = sorted(frames)

        for i, fid in enumerate(order):
            if i == 0 or order[i - 1] != fid - STRIDE:
                continue
            frame, prev = frames[fid], frames[order[i - 1]]
            if frame.ball is None or not frame.players:
                continue
            ball = (float(frame.ball.x), float(frame.ball.y))
            pid, state = min(frame.players.items(),
                             key=lambda kv: math.hypot(kv[1].x - ball[0], kv[1].y - ball[1]))
            if math.hypot(state.x - ball[0], state.y - ball[1]) > CONTROL_M:
                continue
            if pid in keepers:
                continue

            def speed_of(player_id):
                now = frame.players.get(player_id)
                was = prev.players.get(player_id)
                if now is None or was is None:
                    return None
                return math.hypot(now.x - was.x, now.y - was.y) / dt

            s = speed_of(pid)
            if s is None or s > 12.0:
                continue
            carrier_speed.append(s)

            nearest = min(
                (math.hypot(p.x - state.x, p.y - state.y)
                 for oid, p in frame.players.items()
                 if p.team_id != state.team_id and oid not in keepers),
                default=99.0,
            )
            (carrier_pressed if nearest <= 3.0 else carrier_free).append(s)

            for oid, other in frame.players.items():
                if oid == pid or oid in keepers or other.team_id != state.team_id:
                    continue
                v = speed_of(oid)
                if v is not None and v <= 12.0:
                    teammate_speed.append(v)
        del frames
        print(f"  [{match_id}] 누적 캐리 표본 {len(carrier_speed):,}", flush=True)

    c = np.array(carrier_speed)
    m = np.array(teammate_speed)
    print(f"\n{'='*72}")
    print("공을 가진 선수 vs 같은 팀의 공 없는 선수 (0.2초 간격 표본)")
    print("="*72)
    print(f"  {'':<22}{'n':>10}{'중앙':>8}{'평균':>8}{'90%':>8}{'99%':>8}{'최대':>8}")
    for name, v in (("공 가진 선수", c), ("공 없는 동료", m)):
        print(f"  {name:<20}{v.size:>10,}{np.median(v):>8.2f}{v.mean():>8.2f}"
              f"{np.percentile(v,90):>8.2f}{np.percentile(v,99):>8.2f}{v.max():>8.2f}")
    print(f"\n  상위 속도 비율 (99%): {np.percentile(c,99)/np.percentile(m,99):.3f}")
    print(f"  중앙 속도 비율:       {np.median(c)/max(np.median(m),1e-9):.3f}")
    print(f"\n  현재 arrival model 의 최대 속도 가정: 9.00 m/s")
    print(f"  공 가진 선수가 실제로 낸 속도의 99% 분위: {np.percentile(c,99):.2f} m/s")

    p_ = np.array(carrier_pressed)
    f_ = np.array(carrier_free)
    print(f"\n  압박 여부별 (가장 가까운 수비수 3m 기준)")
    print(f"    압박받음 n={p_.size:,} · 중앙 {np.median(p_):.2f} · 99% {np.percentile(p_,99):.2f}")
    print(f"    자유    n={f_.size:,} · 중앙 {np.median(f_):.2f} · 99% {np.percentile(f_,99):.2f}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({
            "carrier": {"n": int(c.size), "median": float(np.median(c)),
                        "p90": float(np.percentile(c, 90)), "p99": float(np.percentile(c, 99))},
            "teammate": {"n": int(m.size), "median": float(np.median(m)),
                         "p90": float(np.percentile(m, 90)), "p99": float(np.percentile(m, 99))},
            "ratio_p99": float(np.percentile(c, 99) / np.percentile(m, 99)),
            "pressed_p99": float(np.percentile(p_, 99)) if p_.size else None,
            "free_p99": float(np.percentile(f_, 99)) if f_.size else None,
        }, ensure_ascii=False, indent=1))
        print(f"\nsaved {args.output}")


if __name__ == "__main__":
    main()
