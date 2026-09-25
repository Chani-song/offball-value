#!/usr/bin/env python3
"""Turn our (runner, defender, beneficiary) triples into stage-3 start states.

The imported solver plays one carrier and one receiver against one defender.
Where R9 names the ball carrier as the beneficiary, that IS our dilemma --
the defender is caught between going to the ball and tracking the run, and
the alternative to the run is the carrier keeping it. Those triples map onto
the solver with nothing changed:

    carrier   <- our ball carrier
    receiver  <- our runner
    defender  <- the gated reacting defender

Where the beneficiary is a third attacker the solver has no body for him, and
those triples wait for the two-receiver question to come back.

Coordinates: ours are centre-origin, x in [-52.5, 52.5] and y in [-34, 34];
the solver wants corner origin, x in [0, 105] and y in [0, 68]. Velocities are
a difference of positions so the shift does not touch them. attack_direction
carries over unchanged -- the solver reads it the same way we do, +1 meaning
the attack runs toward increasing x.

Velocity is estimated over `velocity_window_s` before the onset rather than
taken from a single frame difference, which at 25 fps is mostly noise.

Speed and acceleration limits are the solver's PlayerState defaults, 7.2 m/s
and 3.8 m/s^2, for all three roles including the carrier.

That is a deliberate departure from the imported code's own scenario helper,
which gives the carrier 5.6 m/s and 3.1 m/s^2. Those numbers appear only in
`scenarios.py`'s `carrier()` and in `research/state_pilot.py`, carry no
comment, source or validation, and never bound anything there: the pilot
samples carrier speed from 0 to 3.0 m/s, so a synthetic carrier could not
reach 5.6 in the first place. Its own docstring calls the range "a stated
plausible region for this drill, not an empirical distribution".

Against real Bundesliga tracking the limit does bind. Carrier speed at onset
is a median 3.8 m/s but 6.5 at the 90th percentile and 7.8 at the maximum,
and 23% of our carriers exceed 5.6. Keeping 5.6 would have dropped 55 of 238
triples, every one of them because the man on the ball was moving fast --
which removes the counter-attacks as a class, not a random 23%. 7.2 is the
code's own default for the other two roles, so this widens the carrier to
match his team-mates rather than inventing a number.

Recorded in the output under "limits" so a reader of the study knows.
A state is dropped, with the reason recorded, when a body is off the pitch or
its estimated speed exceeds the limit its role is given: the solver's
constructor rejects both, and silently clipping them would move a player to
make a scene fit.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

FIELD_LENGTH, FIELD_WIDTH = 105.0, 68.0
# The solver's own per-role limits (scenarios.py gives the carrier 5.6/3.1 and
# PlayerState defaults the other two to 7.2/3.8).
LIMITS = {"carrier": (7.2, 3.8), "receiver": (7.2, 3.8), "defender": (7.2, 3.8)}
UPSTREAM_CARRIER_LIMITS = (5.6, 3.1)   # scenarios.py's carrier(), not used here
EDGE_M = 0.05   # keep bodies off the exact touchline, which the check excludes


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--gate", type=Path,
                   default=Path("data/processed/pair_gate_v7.csv"))
    p.add_argument("--build", type=Path,
                   default=Path("/work/hdd/bbmr/kseo1/offball-out/v7_r9_ssac"))
    p.add_argument("--output", type=Path,
                   default=Path("data/processed/stage3/carrier_beneficiary_states.json"))
    p.add_argument("--velocity-window-s", type=float, default=0.4)
    # Agile physics (2026-09-25): runner and defender 9.0, the carrier on the
    # ball keeps 7.2. Acceleration here only fills PlayerState's field -- with
    # --physics agile the solver moves bodies by agile_motion.AGILE instead.
    p.add_argument("--max-speed", type=float, default=None,
                   help="receiver (runner) and defender top speed; default the 7.2 above")
    p.add_argument("--carrier-max-speed", type=float, default=None)
    p.add_argument("--max-acceleration", type=float, default=None)
    p.add_argument("--note", default=None, help="why the limits differ, stored in the output")
    return p.parse_args()


def corner(xy: tuple[float, float]) -> tuple[float, float]:
    return (float(xy[0]) + FIELD_LENGTH / 2.0, float(xy[1]) + FIELD_WIDTH / 2.0)


def velocity(frames, pid: str, onset_t: float, window_s: float) -> tuple[float, float]:
    """Mean velocity over the window ending at onset, from observed positions."""
    track = []
    for f in frames:
        t = float(f["relative_time_s"])
        if t < onset_t - window_s - 1e-9 or t > onset_t + 1e-9:
            continue
        for p in f["players"]:
            if str(p[0]) == pid:
                track.append((t, float(p[2]), float(p[3])))
                break
    if len(track) < 2:
        return (0.0, 0.0)
    track.sort()
    span = track[-1][0] - track[0][0]
    if span <= 1e-9:
        return (0.0, 0.0)
    return ((track[-1][1] - track[0][1]) / span,
            (track[-1][2] - track[0][2]) / span)


def body(position, velocity_xy, role: str) -> dict:
    speed, accel = LIMITS[role]
    return {"position": list(position), "velocity": list(velocity_xy),
            "maximum_speed": speed, "maximum_acceleration": accel}


def main() -> None:
    args = parse_args()
    for role in LIMITS:
        speed, accel = LIMITS[role]
        if role == "carrier" and args.carrier_max_speed is not None:
            speed = args.carrier_max_speed
        if role != "carrier" and args.max_speed is not None:
            speed = args.max_speed
        if args.max_acceleration is not None:
            accel = args.max_acceleration
        LIMITS[role] = (speed, accel)
    gate = pd.read_csv(args.gate)
    # kept_final = the defender passed the pair gate AND the runner is the
    # author of the scene (runner_is_protagonist), not a passenger of a dribble
    want = gate[gate["kept_final"] & (gate["beneficiary_name"] == gate["carrier_name"])]
    by_scene: dict[str, list] = {}
    for r in want.itertuples():
        by_scene.setdefault(r.scene_dir, []).append(r)

    states, dropped = [], []
    for scene_dir in sorted(by_scene):
        path = args.build / scene_dir / "local_game_payoff_audits.json"
        payload = json.loads(path.read_text())
        if not payload:
            continue
        g = payload[0]
        frames = g["background_frames"]
        onset = g["onset_frame_id"]
        at = min(frames, key=lambda f: abs(f["frame_id"] - onset))
        onset_t = float(at["relative_time_s"])
        pos = {str(p[0]): (float(p[2]), float(p[3])) for p in at["players"]}
        names = {str(p[0]): p[4] for p in at["players"]}
        carrier_id, runner_id = str(g["carrier_id"]), str(g["runner_id"])
        direction = int(g["attacking_direction"])

        for r in by_scene[scene_dir]:
            defender_id = str(r.defender_id)
            roles = {"carrier": carrier_id, "receiver": runner_id,
                     "defender": defender_id}
            if any(pid not in pos for pid in roles.values()):
                dropped.append({"scene_dir": scene_dir, "defender_id": defender_id,
                                "reason": "player missing at onset"})
                continue
            scenario, bad = {}, None
            for role, pid in roles.items():
                x, y = corner(pos[pid])
                vx, vy = velocity(frames, pid, onset_t, args.velocity_window_s)
                limit = LIMITS[role][0]
                if not (EDGE_M <= x <= FIELD_LENGTH - EDGE_M
                        and EDGE_M <= y <= FIELD_WIDTH - EDGE_M):
                    bad = f"{role} off the pitch ({x:.1f}, {y:.1f})"
                    break
                if math.hypot(vx, vy) > limit:
                    bad = (f"{role} speed {math.hypot(vx, vy):.1f} over "
                           f"its {limit} m/s limit")
                    break
                scenario[role] = body((x, y), (vx, vy), role)
            if bad:
                dropped.append({"scene_dir": scene_dir,
                                "defender_id": defender_id, "reason": bad})
                continue
            scenario.update(pitch_length=FIELD_LENGTH, pitch_width=FIELD_WIDTH,
                            attack_direction=direction,
                            name=f"{r.match_id}:{int(r.onset_frame_id)}:{defender_id}")
            states.append({
                "index": len(states),
                # The solver groups its report by stratum; ours is the branch
                # of the pair gate that kept this defender.
                "stratum": str(r.reason),
                "scenario": scenario,
                "provenance": {
                    "match_id": r.match_id, "match_label": r.match_label,
                    "onset_frame_id": int(r.onset_frame_id),
                    "scene_dir": scene_dir,
                    "runner_id": runner_id, "runner_name": names.get(runner_id),
                    "defender_id": defender_id, "defender_name": r.defender_name,
                    "carrier_id": carrier_id, "carrier_name": names.get(carrier_id),
                    "gate_reason": r.reason, "gate_rank": int(r.rank),
                },
            })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "states": states,
        # exact_study reads only "states"; this rides along so the departure
        # from the imported defaults is visible wherever the file is read.
        "limits": {
            "applied": {role: {"maximum_speed": v[0],
                               "maximum_acceleration": v[1]}
                        for role, v in LIMITS.items()},
            "upstream_carrier": {"maximum_speed": UPSTREAM_CARRIER_LIMITS[0],
                                 "maximum_acceleration": UPSTREAM_CARRIER_LIMITS[1],
                                 "source": "defensive_positioning/scenarios.py carrier()"},
            "why": ("The carrier is given the code's own 7.2/3.8 default rather "
                    "than scenarios.py's 5.6/3.1. That value is uncommented and "
                    "never bound upstream, where carrier speed is sampled from "
                    "0-3.0 m/s. On real tracking 23% of carriers exceed 5.6, and "
                    "holding to it would drop the counter-attacks as a class."),
            "decided": "2026-09-22, with 서규혁",
            "override": args.note,
        },
    }, indent=1))
    print(f"대상 {len(want)} → 상태 {len(states)} · 제외 {len(dropped)}")
    if dropped:
        reasons: dict[str, int] = {}
        for d in dropped:
            key = d["reason"].split("(")[0].strip()
            reasons[key] = reasons.get(key, 0) + 1
        for k, v in sorted(reasons.items(), key=lambda kv: -kv[1]):
            print(f"   제외 사유: {k} — {v}건")
    strata: dict[str, int] = {}
    for s in states:
        strata[s["stratum"]] = strata.get(s["stratum"], 0) + 1
    print("   게이트 가지별:", strata)
    print(f"→ {args.output}")


if __name__ == "__main__":
    main()
