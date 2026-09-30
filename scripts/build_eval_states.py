#!/usr/bin/env python3
"""Solver states for the evaluation scenes (2026-09-29): each pipeline scene's real 0 / 0.6 / 1.2 s moments.

The abstract's evaluation solves a fresh game at every 0.6 s real moment of a scene and compares what the
players did with the game's optimum. The scenes here are the pipeline's own windows (the rated S46-S75): the
pipeline chose the runner, defender, beneficiary, ball carrier and the start (the run onset), so nothing about
the window is picked by hand. Their states are rebuilt from the raw tracking with scripts/build_showcase_states.py's
own functions, at the onset and 0.6 s / 1.2 s after it.

Checks and rules:
  rebuild   each scene's onset state is first rebuilt the imported way and compared field for field with the
            pipeline's states file (stage3 *_final2m.json); any difference stops the script
  holder    who has the ball at an instant: the attacker nearest the ball, unless the ball moves faster than
            TOP_SPEED -- the model's top player speed (the states files' 9.0 m/s limit): no player can run
            with a ball that fast, so it is in flight. Our rule, not the pipeline's.
  2v1       a moment is a 2v1 game only while the pipeline's ball carrier has the ball at its start
  3v1       the scripted passer at each of the game's four instants is the holder then (a ball in flight or
            already with the runner or the beneficiary: no pass can be made, release_steps False); a moment
            whose start has the ball with the runner or the beneficiary is over, and one with no instant
            left to pass is skipped
  skipped   every moment that fails a rule (or the builder's own checks: off the pitch, over the speed limit)
            is listed with the reason, not built

Usage (from the meeting tree, like build_showcase_states.py):
    PYTHONPATH=src:scripts:andrew-passer2on1:andrew-fixedpasser \\
      .venv/bin/python scripts/build_eval_states.py --codes S46,S48,... --output <dir>
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path

import numpy as np

from build_showcase_states import (DATA, REF_2V1, REF_3V1, STEPS, STEP_S, Match, compare, record_2v1,
                                   record_3v1)

FPS = 25
STEP_FRAMES = round(STEP_S * FPS)
HALF_FRAMES = 4                    # ball speed over +-0.16 s, the window build_showcase_states.ball_track uses
TOP_SPEED = 9.0                    # m/s: the states files' strategic top speed
RATED = DATA / "data/processed/rating_v1/merged.csv"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--codes", required=True, help="comma-separated rated scene codes (pipeline scenes)")
    p.add_argument("--moments", default="0,1,2", help="which 0.6 s moments after the onset to build")
    p.add_argument("--raw-dir", type=Path, default=DATA / "data/raw/bundesliga-integrated")
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def holder(mt: Match, frame: int, attackers: set[str]):
    """(player id or None for a ball in flight, ball speed m/s) at a frame."""
    fr, before, after = (mt.frames.get(frame + d) for d in (0, -HALF_FRAMES, HALF_FRAMES))
    if fr is None or fr.ball is None:
        return None, math.nan
    speed = (math.hypot(after.ball.x - before.ball.x, after.ball.y - before.ball.y) / (2 * HALF_FRAMES / FPS)
             if before is not None and after is not None and before.ball and after.ball else math.nan)
    if not speed <= TOP_SPEED:
        return None, speed
    near = min((math.hypot(p.x - fr.ball.x, p.y - fr.ball.y), pid) for pid, p in fr.players.items() if pid in attackers)
    return near[1], speed


def main() -> None:
    args = parse_args()
    codes = args.codes.split(",")
    moments = [int(k) for k in args.moments.split(",")]
    rated = {r["code"]: r for r in csv.DictReader(RATED.open(encoding="utf-8"))}
    ref = {"2대1": json.loads(REF_2V1.read_text()), "3대1": json.loads(REF_3V1.read_text())}
    by_index = {k: {s["index"]: s for s in v["states"]} for k, v in ref.items()}
    lim2 = {slot: (v["maximum_speed"], v["maximum_acceleration"]) for slot, v in ref["2대1"]["limits"]["applied"].items()}
    a3 = ref["3대1"]["limits"]["all_strategic"]
    lim3 = {slot: (a3["maximum_speed"], a3["maximum_acceleration"]) for slot in ("carrier", "receiver", "defender")}

    scenes = []
    for code in codes:
        r = rated[code]
        if r["source"] != "solver":
            raise SystemExit(f"{code}: not a pipeline scene")
        kind, pv = r["detail"], by_index[r["detail"]][int(float(r["solver_index"]))]["provenance"]
        scenes.append((code, kind, int(float(r["solver_index"])), pv))
    starts: dict[str, list[int]] = {}
    for _, _, _, pv in scenes:
        onset = int(pv["onset_frame_id"])
        starts.setdefault(pv["match_id"], []).extend(onset + STEP_FRAMES * k for k in range(STEPS + 3))
    matches = {m: Match(args.raw_dir, m, s) for m, s in sorted(starts.items())}

    out, skipped = {"2대1": [], "3대1": []}, []
    for code, kind, index, pv in scenes:
        mt = matches[pv["match_id"]]
        onset = int(pv["onset_frame_id"])
        team = mt.frames[onset].players[pv["runner_id"]].team_id
        ids = {"runner": pv["runner_id"], "defender": pv["defender_id"], "carrier": pv["carrier_id"],
               "beneficiary": pv.get("beneficiary_id") or pv["carrier_id"]}
        build, lim = (record_2v1, lim2) if kind == "2대1" else (record_3v1, lim3)
        # 1  the onset rebuilt the imported way must be the pipeline's state
        compare(f"{code} ({kind} index {index})", build(mt.payload(onset, team), ids, lim, index, pv),
                by_index[kind][index])
        attackers = {pid for pid, p in mt.frames[onset].players.items() if p.team_id == team}
        # 2  the moments
        for k in moments:
            start = onset + STEP_FRAMES * k
            label = code if k == 0 else f"{code}@{k * STEP_S:.1f}"
            holders = [holder(mt, start + STEP_FRAMES * i, attackers) for i in range(STEPS + 1)]
            h0 = holders[0][0]
            prov = {**pv, "code": label, "source": "pipeline", "scene_dir": f"eval/{label}", "onset_frame_id": start,
                    "pipeline_onset_frame_id": onset, "start_t": round(k * STEP_S, 2), "rated_code": code,
                    "holders": [[h, None if math.isnan(s) else round(s, 2)] for h, s in holders],
                    "beneficiary_id": ids["beneficiary"]}
            if kind == "2대1" and h0 != ids["carrier"]:
                skipped.append((label, f"the carrier has not got the ball at the start (holder {h0}, "
                                       f"ball {holders[0][1]:.1f} m/s)"))
                continue
            per_instant = None
            if kind == "3대1":
                if h0 in (ids["runner"], ids["beneficiary"]):
                    skipped.append((label, "the ball is already with the runner or the beneficiary"))
                    continue
                per_instant = [h if h is not None and h not in (ids["runner"], ids["beneficiary"]) else "-"
                               for h, _ in holders]
                # once the ball reaches a receiver the passer has nothing left to play
                reached = next((i for i, (h, _) in enumerate(holders) if h in (ids["runner"], ids["beneficiary"])), None)
                if reached is not None:
                    per_instant = [x if i < reached else "-" for i, x in enumerate(per_instant)]
                if all(x == "-" for x in per_instant):
                    skipped.append((label, "no instant with a passer on the ball"))
                    continue
            try:
                payload = mt.payload(start, team)
                rec = (build(payload, ids, lim, 0, prov) if kind == "2대1"
                       else build(payload, ids, lim, 0, prov, per_instant))
            except SystemExit as exc:          # the builder's own checks: off the pitch, over the speed limit
                skipped.append((label, str(exc)))
                continue
            out[kind].append(rec)

    args.output.mkdir(parents=True, exist_ok=True)
    for kind, name in (("2대1", "passer2on1"), ("3대1", "fixedpasser")):
        for i, rec in enumerate(out[kind]):
            rec["index"] = i
        meta = {k: v for k, v in ref[kind].items() if k not in ("states", "filters", "dropped", "source")}
        body = {"states": out[kind], **meta,
                "source": {"built_by": "scripts/build_eval_states.py", "codes": codes, "moments": moments,
                           "holder_rule": f"nearest attacker unless the ball moves faster than {TOP_SPEED} m/s",
                           "checked_against": f"{REF_2V1.name} / {REF_3V1.name}: every onset rebuilt equal"}}
        path = args.output / f"states_{name}.json"
        path.write_text(json.dumps(body, indent=1))
        print(f"{kind}: {len(out[kind])} states -> {path}")
    for label, why in skipped:
        print(f"  skipped {label}: {why}")


if __name__ == "__main__":
    main()
