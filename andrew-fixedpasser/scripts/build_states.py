#!/usr/bin/env python3
"""Start states for the 3v1 game: every gated triple whose beneficiary is NOT the carrier.

Source: `data/processed/pair_gate_v7.csv`, rows with `kept_final` (pair gate
kept the defender AND the runner is the protagonist) and a beneficiary other
than the ball carrier -- 718 triples. `--min-run-speed` additionally keeps only
runs whose mean speed over the 0.8 s after onset reaches the threshold, the
run-quality cut calibrated on the 58 reviewed runs.

Per triple:
  scenario   carrier slot = runner, receiver slot = beneficiary, defender slot
             = controlled defender; positions at onset, velocity the mean over
             the 0.4 s before onset (as build_stage3_states.py), limits
             7.2 m/s and 3.8 m/s^2 for all three.
  passer     the ball carrier's real position and velocity at t = 0, 1, 2, 3 s
  background every other defender, keeper included, at the same instants

A triple is dropped, with the reason kept, when a strategic body is off the
pitch or faster than its limit at onset -- the solver's own constructor refuses
both, and moving a player to make a scene fit would change the scene.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import math
from pathlib import Path

import pandas as pd

from fixedpasser.tracks import background_tracks, player_track

ROOT = Path(__file__).resolve().parents[2]
FIELD_LENGTH, FIELD_WIDTH = 105.0, 68.0
LIMIT = (7.2, 3.8)
EDGE_M = 0.05


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--gate", type=Path, default=ROOT / "data/processed/pair_gate_v7.csv")
    p.add_argument("--build", type=Path,
                   default=Path(os.environ.get("OFFBALL_OUT_ROOT", Path(__file__).resolve().parents[2] / "out/runs")) / "v7_r9_ssac")
    p.add_argument("--onsets", default=str(ROOT / "data/processed/run_onset_v0_5/dir25/*/audit_selection.csv"))
    p.add_argument("--output", type=Path,
                   default=ROOT / "data/processed/stage3/fixedpasser_states.json")
    p.add_argument("--min-run-speed", type=float, default=None,
                   help="keep runs whose post_mean_speed_mps is at least this (m/s)")
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--step-seconds", type=float, default=1.0)
    p.add_argument("--velocity-window-s", type=float, default=0.4)
    p.add_argument("--max-speed", type=float, default=LIMIT[0],
                   help="top speed of runner, beneficiary and defender (agile physics: 9.0)")
    p.add_argument("--max-acceleration", type=float, default=LIMIT[1],
                   help="PlayerState's field only; --physics agile moves bodies by agile_motion")
    return p.parse_args()


def corner(x, y):
    return (float(x) + FIELD_LENGTH / 2.0, float(y) + FIELD_WIDTH / 2.0)


def onset_velocity(frames, pid, onset_t, window_s):
    """Mean velocity over the window ending at onset (build_stage3_states.velocity)."""
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
    return ((track[-1][1] - track[0][1]) / span, (track[-1][2] - track[0][2]) / span)


def main() -> None:
    args = parse_args()
    limit = (args.max_speed, args.max_acceleration)
    gate = pd.read_csv(args.gate)
    want = gate[gate["kept_final"] & (gate["beneficiary_name"] != gate["carrier_name"])].copy()
    n_source = len(want)
    if args.min_run_speed is not None:
        onsets = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(args.onsets))], ignore_index=True)
        onsets = onsets.rename(columns={"frame_id": "onset_frame_id", "player_id": "runner_id"})
        want = want.merge(onsets[["match_id", "onset_frame_id", "runner_id", "post_mean_speed_mps"]],
                          on=["match_id", "onset_frame_id", "runner_id"], how="left")
        if want["post_mean_speed_mps"].isna().any():
            raise SystemExit("some triples have no onset speed; cannot apply --min-run-speed")
        want = want[want["post_mean_speed_mps"] >= args.min_run_speed]
    cache, states, dropped = {}, [], []
    for r in want.itertuples():
        if r.scene_dir not in cache:
            cache[r.scene_dir] = json.loads(
                (args.build / r.scene_dir / "local_game_payoff_audits.json").read_text())[0]
        payload = cache[r.scene_dir]
        frames = payload["background_frames"]
        at = min(frames, key=lambda f: abs(f["frame_id"] - payload["onset_frame_id"]))
        onset_t = float(at["relative_time_s"])
        pos = {str(p[0]): (p[2], p[3]) for p in at["players"]}
        names = {str(p[0]): p[4] for p in at["players"]}
        carrier = str(payload["carrier_id"])
        roles = {"carrier": str(r.runner_id), "receiver": str(r.beneficiary_id),
                 "defender": str(r.defender_id)}
        key = {"scene_dir": r.scene_dir, "defender_id": str(r.defender_id)}
        if any(pid not in pos for pid in (*roles.values(), carrier)):
            dropped.append({**key, "reason": "player missing at onset"})
            continue
        scenario, bad = {}, None
        for slot, pid in roles.items():
            x, y = corner(*pos[pid])
            vx, vy = onset_velocity(frames, pid, onset_t, args.velocity_window_s)
            if not (EDGE_M <= x <= FIELD_LENGTH - EDGE_M and EDGE_M <= y <= FIELD_WIDTH - EDGE_M):
                bad = f"{slot} off the pitch"
                break
            if math.hypot(vx, vy) > limit[0]:
                bad = f"{slot} faster than {limit[0]} m/s"
                break
            scenario[slot] = {"position": [x, y], "velocity": [vx, vy],
                              "maximum_speed": limit[0], "maximum_acceleration": limit[1]}
        if bad:
            dropped.append({**key, "reason": bad})
            continue
        try:
            passer = player_track(payload, carrier, args.steps, args.step_seconds)
        except ValueError as exc:
            dropped.append({**key, "reason": f"passer track: {exc}"})
            continue
        bg = background_tracks(payload, {*roles.values(), carrier}, args.steps, args.step_seconds)
        scenario.update(pitch_length=FIELD_LENGTH, pitch_width=FIELD_WIDTH,
                        attack_direction=int(payload["attacking_direction"]),
                        name=f"{r.match_id}:{int(r.onset_frame_id)}:{r.defender_id}")
        states.append({
            "index": len(states), "stratum": str(r.reason), "scenario": scenario,
            "passer": {"positions": passer["positions"].round(4).tolist(),
                       "velocities": passer["velocities"].round(4).tolist(),
                       "times_requested": passer["times_requested"],
                       "times_used": passer["times_used"]},
            "background": {"ids": bg["ids"], "names": bg["names"],
                           "positions": bg["positions"].round(4).tolist(),
                           "velocities": bg["velocities"].round(4).tolist(),
                           "times_requested": bg["times_requested"],
                           "times_used": bg["times_used"], "dropped": bg["dropped"]},
            "provenance": {
                "match_id": r.match_id, "match_label": r.match_label,
                "onset_frame_id": int(r.onset_frame_id), "scene_dir": r.scene_dir,
                "runner_id": str(r.runner_id), "runner_name": names.get(str(r.runner_id)),
                "beneficiary_id": str(r.beneficiary_id), "beneficiary_name": names.get(str(r.beneficiary_id)),
                "defender_id": str(r.defender_id), "defender_name": r.defender_name,
                "carrier_id": carrier, "carrier_name": names.get(carrier),
                "gate_reason": r.reason, "gate_rank": int(r.rank),
                "post_mean_speed_mps": (float(r.post_mean_speed_mps)
                                        if "post_mean_speed_mps" in want.columns else None),
            },
        })
    out = {"states": states,
           "source": {"gate": str(args.gate), "triples": n_source,
                      "min_run_speed": args.min_run_speed, "after_speed_filter": len(want)},
           "background": {"steps": args.steps, "step_seconds": args.step_seconds,
                          "what": "every defender except the controlled one, keeper included"},
           "limits": {"all_strategic": {"maximum_speed": limit[0], "maximum_acceleration": limit[1]}},
           "dropped": dropped}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=1))
    reasons: dict[str, int] = {}
    for d in dropped:
        reasons[d["reason"]] = reasons.get(d["reason"], 0) + 1
    print(f"source {n_source}" + (f" → speed ≥ {args.min_run_speed} {len(want)}" if args.min_run_speed else "")
          + f" → states {len(states)} · dropped {len(dropped)} {reasons}")
    print(f"→ {args.output}")


if __name__ == "__main__":
    main()
