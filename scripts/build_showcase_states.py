#!/usr/bin/env python3
"""Solver start states for hand-picked scenes, read straight from the tracking.

The stage-3 builders read each scene from a pipeline payload
(`local_game_payoff_audits.json`): scripts/build_stage3_states.py and
andrew-passer2on1/scripts/build_states.py for 2v1,
andrew-fixedpasser/scripts/build_states.py for 3v1. The scenes the reviewer
picked for the abstract (2026-09-28, most of them Chani's) have no payload, so
this script builds one from the raw tracking in the payload's shape -- every
2nd frame from -2.00 s to +2.96 s around the start, centre-origin positions,
players as [id, team, x, y, name] -- and then calls those builders' own
functions unchanged:

  velocity     build_stage3_states.velocity: mean over the 0.4 s before the start
  background   passer2on1.tracks / fixedpasser.tracks.background_tracks
  passer       fixedpasser.tracks.player_track (3v1)
  limits       copied from the meeting study's states files: 2v1 carrier 7.2,
               runner and defender 9.0; 3v1 all three 9.0; acceleration 4.5

Before writing anything it rebuilds two states the meeting study already has --
2v1 index 51 (the rated S53) and 3v1 index 654 (the rated S58) -- from the raw
tracking and compares every field with the states files. Any difference stops it.

A 3v1 scene may name who has the ball at each decision instant (--passer
CODE=ID,ID,ID,ID, `-` for a ball on its way between two players). A `-`
instant gets release_steps False (fixedpasser.game: no pass can be made there)
and the passer track there is the ball's own position, which nothing prices.
S36 (the reviewer, 2026-09-28): Appelkamp at t0, the ball rolling to Iyoha at
+0.6 and +1.2, Iyoha crossing at +1.8.

Reported, not applied: the isolation filter of scripts/filter_stage3_states.py
(nobody within 2 m of the triangle at the four instants). The scenes were
picked by hand, so the report only says which the filter would have dropped.
Its other filter, run speed after a detected onset, does not apply: these
starts are "release - 1.8 s", not run onsets.

Usage (from this repository; data are read from OFFBALL_DATA_ROOT):
    PYTHONPATH=src:scripts:andrew-passer2on1:andrew-fixedpasser \\
      .venv-delta/bin/python scripts/build_showcase_states.py \\
      --starts <solver_starts.csv> --passer S36=<id>,-,-,<id>
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path

import numpy as np

from offball_value.bundesliga import (FPS, find_bundesliga_files, infer_attacking_direction,
                                      load_bundesliga_frames, load_bundesliga_match_metadata)
from build_stage3_states import corner, velocity
from filter_stage3_states import distance_to_triangle, inside, positions_at
import fixedpasser.tracks as fp_tracks
import passer2on1.tracks as p2_tracks

DATA = Path(os.environ.get("OFFBALL_DATA_ROOT", "/scratch/bbmr/kseo1/offball-value"))
STAGE3 = DATA / "data/processed/stage3"
REF_2V1 = STAGE3 / "passer2on1_states_agile06_final2m.json"
REF_3V1 = STAGE3 / "fixedpasser_states_agile06_final2m.json"
STEPS, STEP_S, VELOCITY_WINDOW_S = 3, 0.6, 0.4
FIRST, LAST = -50, 74            # frames around the start: -2.00 s .. +2.96 s, the payload clip
FIELD_LENGTH, FIELD_WIDTH = 105.0, 68.0
EDGE_M = 0.05
BUFFER_M = 2.0                   # filter_stage3_states, the reviewer's 2 m (2026-09-25)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--starts", type=Path, required=True, help="solver_starts CSV from the picker page")
    p.add_argument("--passer", action="append", default=[],
                   help="CODE=ID,ID,ID,ID: who has the ball at each instant, - for a ball in flight")
    p.add_argument("--raw-dir", type=Path, default=DATA / "data/raw/bundesliga-integrated")
    p.add_argument("--output", type=Path, default=DATA / "data/processed/showcase_v1")
    return p.parse_args()


class Match:
    def __init__(self, raw_dir: Path, match_id: str, starts: list[int]):
        files = find_bundesliga_files(raw_dir, match_id)
        self.id = match_id
        self.meta = load_bundesliga_match_metadata(files["matchinfo"])
        wanted = {f for s in starts for f in range(s + FIRST, s + LAST + 1)}
        self.frames = load_bundesliga_frames(files["positions"], sorted(wanted))

    def payload(self, start: int, attacking_team: str) -> dict:
        """The pipeline payload's shape, for the fields the builders read."""
        name = lambda pid: self.meta.players[pid].short_name if pid in self.meta.players else ""
        frames = []
        for f in range(start + FIRST, start + LAST + 1, 2):
            fr = self.frames.get(f)
            if fr is None:
                continue
            frames.append({"frame_id": f, "relative_time_s": round((f - start) / FPS, 2),
                           "ball": None if fr.ball is None else [fr.ball.x, fr.ball.y],
                           "players": [[pid, p.team_id, p.x, p.y, name(pid)]
                                       for pid, p in fr.players.items()]})
        direction = int(infer_attacking_direction(self.frames[start], attacking_team, self.meta))
        return {"onset_frame_id": start, "attacking_team_id": attacking_team,
                "attacking_direction": direction, "background_frames": frames}

    def label(self) -> str:
        return f"{self.meta.home_team_name} vs {self.meta.away_team_name}"


def at_start(payload):
    frames = payload["background_frames"]
    at = min(frames, key=lambda f: abs(f["frame_id"] - payload["onset_frame_id"]))
    return frames, at, float(at["relative_time_s"])


def bodies(payload, roles: dict, limits: dict) -> dict:
    """Position at the start and velocity over the 0.4 s before, as both builders do."""
    frames, at, t0 = at_start(payload)
    pos = {str(p[0]): (float(p[2]), float(p[3])) for p in at["players"]}
    out = {}
    for slot, pid in roles.items():
        if pid not in pos:
            raise SystemExit(f"{slot} {pid} missing at the start")
        x, y = corner(pos[pid])
        vx, vy = velocity(frames, pid, t0, VELOCITY_WINDOW_S)
        speed, accel = limits[slot]
        if not (EDGE_M <= x <= FIELD_LENGTH - EDGE_M and EDGE_M <= y <= FIELD_WIDTH - EDGE_M):
            raise SystemExit(f"{slot} {pid} off the pitch ({x:.1f}, {y:.1f})")
        if math.hypot(vx, vy) > speed:
            raise SystemExit(f"{slot} {pid} at {math.hypot(vx, vy):.2f} m/s, over its {speed} m/s limit")
        out[slot] = {"position": [x, y], "velocity": [vx, vy],
                     "maximum_speed": speed, "maximum_acceleration": accel}
    return out


def rounded(track: dict, keys=("positions", "velocities")) -> dict:
    return {k: (np.asarray(v).round(4).tolist() if k in keys else v) for k, v in track.items()}


def ball_track(payload) -> dict:
    """The ball at each instant, read off the frames like fp_tracks.player_track reads a player."""
    frames, _, t0 = at_start(payload)
    times = np.array([float(f["relative_time_s"]) - t0 for f in frames if f["ball"] is not None])
    xy = np.array([(f["ball"][0] + FIELD_LENGTH / 2.0, f["ball"][1] + FIELD_WIDTH / 2.0)
                   for f in frames if f["ball"] is not None])
    at = lambda t: np.array([np.interp(t, times, xy[:, 0]), np.interp(t, times, xy[:, 1])])
    wanted = [k * STEP_S for k in range(STEPS + 1)]
    used = [float(np.clip(t, times.min(), times.max())) for t in wanted]
    half = 0.16
    return {"positions": np.array([at(t) for t in used]),
            "velocities": np.array([(at(t + half) - at(t - half)) / (2 * half) for t in used]),
            "times_requested": wanted, "times_used": used}


def record_2v1(payload, ids: dict, limits: dict, index: int, prov: dict) -> dict:
    roles = {"carrier": ids["carrier"], "receiver": ids["runner"], "defender": ids["defender"]}
    scenario = bodies(payload, roles, limits)
    scenario.update(pitch_length=FIELD_LENGTH, pitch_width=FIELD_WIDTH,
                    attack_direction=int(payload["attacking_direction"]),
                    name=f"{prov['match_id']}:{payload['onset_frame_id']}:{ids['defender']}")
    bg = p2_tracks.background_tracks(payload, set(roles.values()), STEPS, STEP_S)
    return {"index": index, "stratum": "showcase", "scenario": scenario, "provenance": prov,
            "background": rounded(bg)}


def record_3v1(payload, ids: dict, limits: dict, index: int, prov: dict, per_instant=None) -> dict:
    roles = {"carrier": ids["runner"], "receiver": ids["beneficiary"], "defender": ids["defender"]}
    scenario = bodies(payload, roles, limits)
    scenario.update(pitch_length=FIELD_LENGTH, pitch_width=FIELD_WIDTH,
                    attack_direction=int(payload["attacking_direction"]),
                    name=f"{prov['match_id']}:{payload['onset_frame_id']}:{ids['defender']}")
    rec = {"index": index, "stratum": "showcase", "scenario": scenario}
    if per_instant is None:
        passer = fp_tracks.player_track(payload, ids["carrier"], STEPS, STEP_S)
        passers = {ids["carrier"]}
    else:
        tracks = {pid: fp_tracks.player_track(payload, pid, STEPS, STEP_S)
                  for pid in set(per_instant) - {"-"}}
        ball = ball_track(payload)
        rows = [(tracks[pid] if pid != "-" else ball) for pid in per_instant]
        passer = {"positions": np.array([r["positions"][k] for k, r in enumerate(rows)]),
                  "velocities": np.array([r["velocities"][k] for k, r in enumerate(rows)]),
                  "times_requested": ball["times_requested"], "times_used": ball["times_used"],
                  "by_instant": [pid if pid != "-" else "ball" for pid in per_instant]}
        rec["release_steps"] = [pid != "-" for pid in per_instant]
        passers = set(per_instant) - {"-"}
    bg = fp_tracks.background_tracks(payload, {*roles.values(), *passers}, STEPS, STEP_S)
    rec.update(passer=rounded(passer), background=rounded(bg), provenance=prov)
    return rec


def isolation(payload, verts, in_game) -> tuple[float, str]:
    """Nearest outsider to the triangle over the four instants (filter_stage3_states)."""
    frames, _, t0 = at_start(payload)
    names = {str(p[0]): p[4] for p in frames[0]["players"]}
    best = (math.inf, "")
    for k in range(STEPS + 1):
        pos = positions_at(frames, t0, k * STEP_S)
        a, b, c = (pos[x] for x in verts)
        for pid, xy in pos.items():
            if pid in in_game:
                continue
            d = distance_to_triangle(xy, a, b, c)
            if d < best[0] or (d == 0.0 and inside(xy, a, b, c)):
                best = (d, f"{names.get(pid, pid)} at {k * STEP_S:.1f} s")
    return best


def compare(name: str, ours, theirs, tol=1e-9) -> None:
    """Every field equal (numbers within tol); background rows matched by id."""
    def walk(a, b, path):
        if isinstance(b, dict):
            for k in b:
                if k in ("provenance", "stratum", "index"):
                    continue
                if k not in a:
                    raise SystemExit(f"{name}: {path}.{k} missing in the rebuild")
                walk(a[k], b[k], f"{path}.{k}")
        elif isinstance(b, (list, tuple)) and b and isinstance(b[0], (list, tuple, int, float)):
            x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
            if x.shape != y.shape or np.nanmax(np.abs(x - y), initial=0.0) > tol:
                raise SystemExit(f"{name}: {path} differs (max {np.nanmax(np.abs(x - y)) if x.shape == y.shape else x.shape})")
        elif isinstance(b, float):
            if abs(float(a) - b) > tol:
                raise SystemExit(f"{name}: {path} {a} != {b}")
        elif a != b:
            raise SystemExit(f"{name}: {path} {a!r} != {b!r}")

    def by_id(rec):
        bg = rec["background"]
        order = np.argsort(bg["ids"])
        return {**bg, "ids": [bg["ids"][i] for i in order], "names": [bg["names"][i] for i in order],
                "positions": np.asarray(bg["positions"])[:, order].tolist(),
                "velocities": np.asarray(bg["velocities"])[:, order].tolist()}
    walk({**ours, "background": by_id(ours)}, {**theirs, "background": by_id(theirs)}, name)


def main() -> None:
    args = parse_args()
    ref2, ref3 = json.loads(REF_2V1.read_text()), json.loads(REF_3V1.read_text())
    lim2 = {slot: (v["maximum_speed"], v["maximum_acceleration"]) for slot, v in ref2["limits"]["applied"].items()}
    a3 = ref3["limits"]["all_strategic"]
    lim3 = {slot: (a3["maximum_speed"], a3["maximum_acceleration"]) for slot in ("carrier", "receiver", "defender")}
    check2 = next(s for s in ref2["states"] if s["index"] == 51)
    check3 = next(s for s in ref3["states"] if s["index"] == 654)
    picks = [r for r in csv.DictReader(args.starts.open(encoding="utf-8-sig")) if r["start_t"]]
    per_instant = {}
    for spec in args.passer:
        code, ids = spec.split("=", 1)
        per_instant[code] = ids.split(",")
        if len(per_instant[code]) != STEPS + 1:
            raise SystemExit(f"--passer {code}: {STEPS + 1} entries, got {len(per_instant[code])}")

    starts: dict[str, list[int]] = {}
    for p in check2, check3:
        starts.setdefault(p["provenance"]["match_id"], []).append(int(p["provenance"]["onset_frame_id"]))
    for r in picks:
        starts.setdefault(r["match_id"], []).append(int(r["start_frame"]))
    matches = {}
    for m, s in sorted(starts.items()):
        matches[m] = Match(args.raw_dir, m, s)
        print(f"  {m}: 프레임 {len(matches[m].frames)}", flush=True)

    # 1  the two rebuilds must equal the meeting study's states, field for field
    for ref, lim, build in ((check2, lim2, record_2v1), (check3, lim3, record_3v1)):
        pv = ref["provenance"]
        mt = matches[pv["match_id"]]
        onset = int(pv["onset_frame_id"])
        team = mt.frames[onset].players[pv["runner_id"]].team_id
        ids = {"runner": pv["runner_id"], "defender": pv["defender_id"], "carrier": pv["carrier_id"],
               "beneficiary": pv.get("beneficiary_id") or pv["carrier_id"]}
        ours = build(mt.payload(onset, team), ids, lim, ref["index"], pv)
        compare(f"{'2v1' if build is record_2v1 else '3v1'} index {ref['index']}", ours, ref)
        print(f"  재구성 일치: {'2대1' if build is record_2v1 else '3대1'} index {ref['index']} "
              f"({pv['runner_name']} / {pv['defender_name']})", flush=True)

    # 2  the picked scenes
    out = {"2대1": [], "3대1": []}
    report = []
    for r in picks:
        mt = matches[r["match_id"]]
        start = int(r["start_frame"])
        team = mt.frames[start].players[r["runner_id"]].team_id
        payload = mt.payload(start, team)
        ids = {"runner": r["runner_id"], "defender": r["defender_id"],
               "beneficiary": r["beneficiary_id"], "carrier": r["carrier_id"]}
        prov = {"match_id": r["match_id"], "match_label": mt.label(), "onset_frame_id": start,
                "scene_dir": f"showcase/{r['code']}", "code": r["code"], "source": r["source"],
                "start_t": float(r["start_t"]), "zero_kind": r["zero_kind"], "zero_frame": int(r["zero_frame"]),
                "runner_id": r["runner_id"], "runner_name": r["runner_name"],
                "defender_id": r["defender_id"], "defender_name": r["defender_name"],
                "beneficiary_id": r["beneficiary_id"], "beneficiary_name": r["beneficiary_name"],
                "carrier_id": r["carrier_id"], "carrier_name": r["carrier_name"], "note": r["note"]}
        game = r["game"]
        if game == "2대1":
            if r["carrier_id"] != r["beneficiary_id"]:
                raise SystemExit(f"{r['code']}: 2v1 needs the carrier to be the beneficiary")
            if r["code"] in per_instant:
                raise SystemExit(f"{r['code']}: --passer is for 3v1 scenes")
            rec = record_2v1(payload, ids, lim2, len(out[game]), prov)
            verts = (ids["carrier"], ids["runner"], ids["defender"])
            in_game = set(verts)
        else:
            rec = record_3v1(payload, ids, lim3, len(out[game]), prov, per_instant.get(r["code"]))
            verts = (ids["beneficiary"], ids["defender"], ids["runner"])
            passers = set(per_instant.get(r["code"], [ids["carrier"]])) - {"-"}
            in_game = set(verts) | passers
        out[game].append(rec)
        d, who = isolation(payload, verts, in_game)
        name = lambda pid: mt.meta.players[pid].short_name if pid in mt.meta.players else pid
        holders = ("→".join("공" if x == "-" else name(x) for x in per_instant[r["code"]])
                   if r["code"] in per_instant else name(ids["carrier"]))
        report.append((r["code"], game, rec, d, who, holders))

    for game, name, ref in (("2대1", "passer2on1", ref2), ("3대1", "fixedpasser", ref3)):
        meta = {k: v for k, v in ref.items() if k not in ("states", "filters", "dropped", "source")}
        meta["background"] = {**meta["background"], "source_states": str(args.starts)}
        body = {"states": out[game], **meta,
                "source": {"starts": str(args.starts), "passer": args.passer,
                           "built_by": "scripts/build_showcase_states.py",
                           "checked_against": {"2v1": f"{REF_2V1.name} index 51",
                                               "3v1": f"{REF_3V1.name} index 654"}}}
        path = args.output / f"states_{name}.json"
        path.write_text(json.dumps(body, indent=1))
        print(f"  {game} {len(out[game])}개 → {path}")
    print("\n장면   게임  t0      볼 소유(패서)       release_steps           삼각형 2 m 안 (필터 참고)")
    for code, game, rec, d, who, holders in report:
        pv = rec["provenance"]
        steps = rec.get("release_steps", [True] * (STEPS + 1))
        print(f"  {code}  {game}  {pv['start_t']:+.2f}  {holders:<18}  {str(steps):<24}  "
              f"{'걸림' if d <= BUFFER_M else '통과'} (가장 가까운 {d:.2f} m, {who})")


if __name__ == "__main__":
    main()
