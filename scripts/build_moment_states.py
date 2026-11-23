#!/usr/bin/env python3
"""Solver states for every state of a stage-3 file: each scene's real 0 / 0.6 / 1.2 s moments.

scripts/build_eval_states.py does this for the rated pipeline scenes (S46-S75), which it looks up by code in
the rating sheet. This script takes the scenes from stage-3 states files instead (2026-10-24: the v0.6 states
whose possession ends in a shot, data/processed/stage3_v06_shotposs), and otherwise runs the same steps with
the same functions:

  rebuild   each state's onset is rebuilt from the raw tracking and compared field for field with the
            stage-3 state it comes from; any difference stops the script
  moments   build_eval_states.py's rules, unchanged: the holder at each instant, a 2v1 moment only while the
            ball carrier has the ball at its start, a 3v1 moment's scripted passer per instant, and the
            builders' own checks (off the pitch, over the speed limit); every skipped moment is listed

Codes are "<game>_<stage-3 index>" (2v1_007), and "<code>@0.6" / "@1.2" for the later moments
("<code>@-0.6" / "@-1.2" for moments before the onset, --moments=-2,-1).

Usage (as build_eval_states.py):
    PYTHONPATH=src:scripts:andrew-passer2on1:andrew-fixedpasser \\
      .venv/bin/python scripts/build_moment_states.py \\
        --states-2v1 data/processed/stage3_v06_shotposs/passer2on1_states.json \\
        --states-3v1 data/processed/stage3_v06_shotposs/fixedpasser_states.json --output <dir>
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from build_eval_states import GAME_2V1, GAME_3V1, STEP_FRAMES, TOP_SPEED, holder
from build_showcase_states import DATA, STEPS, STEP_S, Match, compare, record_2v1, record_3v1

SHORT = {GAME_2V1: "2v1", GAME_3V1: "3v1"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--states-2v1", type=Path, required=True)
    p.add_argument("--states-3v1", type=Path, required=True)
    p.add_argument("--moments", default="0,1,2",
                   help="which 0.6 s moments after the onset to build (negative: before it, as --moments=-2,-1)")
    p.add_argument("--raw-dir", type=Path, default=DATA / "data/raw/bundesliga-integrated")
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    moments = [int(k) for k in args.moments.split(",")]
    ref = {GAME_2V1: json.loads(args.states_2v1.read_text()), GAME_3V1: json.loads(args.states_3v1.read_text())}
    lim2 = {slot: (v["maximum_speed"], v["maximum_acceleration"]) for slot, v in ref[GAME_2V1]["limits"]["applied"].items()}
    a3 = ref[GAME_3V1]["limits"]["all_strategic"]
    lim3 = {slot: (a3["maximum_speed"], a3["maximum_acceleration"]) for slot in ("carrier", "receiver", "defender")}

    scenes = [(f"{SHORT[kind]}_{int(s['index']):03d}", kind, int(s["index"]), s)
              for kind in (GAME_2V1, GAME_3V1) for s in ref[kind]["states"]]
    starts: dict[str, list[int]] = {}
    for _, _, _, s in scenes:
        pv = s["provenance"]
        onset = int(pv["onset_frame_id"])
        # every moment's start and the instants its game reads (negative moments lie before the onset);
        # for the default 0,1,2 this is the range(STEPS + 3) it always was
        starts.setdefault(pv["match_id"], []).extend(
            onset + STEP_FRAMES * k for k in range(min(0, min(moments)), max(moments) + STEPS + 1))
    matches = {m: Match(args.raw_dir, m, st) for m, st in sorted(starts.items())}

    out, skipped = {GAME_2V1: [], GAME_3V1: []}, []
    for code, kind, index, state in scenes:
        pv = state["provenance"]
        mt = matches[pv["match_id"]]
        onset = int(pv["onset_frame_id"])
        team = mt.frames[onset].players[pv["runner_id"]].team_id
        ids = {"runner": pv["runner_id"], "defender": pv["defender_id"], "carrier": pv["carrier_id"],
               "beneficiary": pv.get("beneficiary_id") or pv["carrier_id"]}
        build, lim = (record_2v1, lim2) if kind == GAME_2V1 else (record_3v1, lim3)
        # 1  the onset rebuilt the imported way must be the stage-3 state
        compare(f"{code} ({kind} index {index})", build(mt.payload(onset, team), ids, lim, index, pv), state)
        attackers = {pid for pid, p in mt.frames[onset].players.items() if p.team_id == team}
        # 2  the moments, as build_eval_states.py
        for k in moments:
            start = onset + STEP_FRAMES * k
            label = code if k == 0 else f"{code}@{k * STEP_S:.1f}"
            holders = [holder(mt, start + STEP_FRAMES * i, attackers) for i in range(STEPS + 1)]
            h0 = holders[0][0]
            prov = {**pv, "code": label, "source": "pipeline", "scene_dir": f"moments/{label}", "onset_frame_id": start,
                    "pipeline_onset_frame_id": onset, "start_t": round(k * STEP_S, 2), "stage3_code": code,
                    "stage3_index": index, "stage3_scene_dir": pv["scene_dir"],
                    "holders": [[h, None if math.isnan(s) else round(s, 2)] for h, s in holders],
                    "beneficiary_id": ids["beneficiary"]}
            if kind == GAME_2V1 and h0 != ids["carrier"]:
                skipped.append((label, f"the carrier has not got the ball at the start (holder {h0}, "
                                       f"ball {holders[0][1]:.1f} m/s)"))
                continue
            per_instant = None
            if kind == GAME_3V1:
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
                rec = (build(payload, ids, lim, 0, prov) if kind == GAME_2V1
                       else build(payload, ids, lim, 0, prov, per_instant))
            except SystemExit as exc:          # the builder's own checks: off the pitch, over the speed limit
                skipped.append((label, str(exc)))
                continue
            out[kind].append(rec)

    args.output.mkdir(parents=True, exist_ok=True)
    for kind, name in ((GAME_2V1, "passer2on1"), (GAME_3V1, "fixedpasser")):
        for i, rec in enumerate(out[kind]):
            rec["index"] = i
        meta = {k: v for k, v in ref[kind].items() if k not in ("states", "filters", "dropped", "source", "selection")}
        body = {"states": out[kind], **meta,
                "source": {"built_by": "scripts/build_moment_states.py", "moments": moments,
                           "stage3_states": str(args.states_2v1 if kind == GAME_2V1 else args.states_3v1),
                           "holder_rule": f"nearest attacker unless the ball moves faster than {TOP_SPEED} m/s",
                           "checked_against": "the stage-3 states: every onset rebuilt equal"}}
        path = args.output / f"states_{SHORT[kind]}.json"
        path.write_text(json.dumps(body, indent=1))
        print(f"{kind}: {len(out[kind])} states -> {path}")
    for label, why in skipped:
        print(f"  skipped {label}: {why}")


if __name__ == "__main__":
    main()
