#!/usr/bin/env python3
"""Scenes for the three-rater dilemma rating (2026-09-25 meeting).

75 scenes, every rater sees the same ones in the same order:

  chani   all 45 strong + medium shot clips from chani/chani_shot_annotation.xlsx
  solver  the 30 scenes the solver calls torn under the shipped pass model
          (agile motion, 3 x 0.6 s, triangle 2 m): 12 2v1 + 18 3v1

What a rater sees is the REAL movement only. Scenes are numbered by group --
S01-S05 Chani strong, S06-S45 Chani medium, S46-S75 solver (reviewer, 2026-09-26)
-- shuffled with a fixed seed within each group, with the same three roles marked.

  clip     chani: 8 s before the shot to 3 s after (11 s, t = 0 at the shot) --
           the window of her annotation app and its mp4s (CLIP_BEFORE /
           CLIP_AFTER in tools/shot_annotation_app/app.py, branch
           chani-fancy-viz). v2 cut 3 s before the anchor to 4 s after, which
           ended 14 of the 45 before the shot (S04 by 6.8 s); Chani asked for
           her window back (2026-09-27).
           solver: 3 s before the run onset to 4 s after (7 s, t = 0 at the
           onset); there is no shot to align to.
  anchor   solver: the run's onset. chani: the named runner's run moment from
           scripts/trace_chani_scenes.py (a detected run, or a pipeline
           candidate), the latest one at least 1 s before the shot within the
           12 s before it; with none, 4 s before the shot. This rule is my
           choice. For chani it now only picks the marked roles below; it may
           fall before the clip starts.
  roles    ONE (runner, defender, beneficiary) per scene -- the reviewer's rule,
           2026-09-25. solver: its triple (beneficiary = carrier in 2v1).
           chani: she may name several of each, so one triple is chosen (my
           rule): the runner whose run moment is the clip's anchor (the
           first-named runner when the anchor is shot - 4 s); among her drawn
           defenders, the one nearest that runner at the anchor; her first-named
           beneficiary who is not that runner. Her full lists stay in the key.
  pitch    metres, centre origin, turned so the attack always runs to the right.

The key that maps S01..S75 back to the source is written separately and is not
in the HTML. Tracking is licensed: the page is for the three raters only.

Usage:
    PYTHONPATH=src:scripts python scripts/build_rating_package.py
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import pandas as pd

from offball_value.bundesliga import (FPS, find_bundesliga_files, infer_attacking_direction,
                                      load_bundesliga_frames, load_bundesliga_match_metadata)
from diagnose_missed_onsets import read_xlsx, split_numbers

ROOT = Path(__file__).resolve().parents[1]
D3 = ROOT / "data/processed/stage3"
SOLVER = (("2대1", "passer2on1_results_agile_2m.csv", "passer2on1_states_agile06_final2m.json"),
          ("3대1", "fixedpasser_results_agile_2m.csv", "fixedpasser_states_agile06_final2m.json"))
PRE_S, POST_S, STEP = 3.0, 4.0, 2          # solver: 7 s around the onset, every 2nd frame (12.5 fps)
CHANI_PRE_S, CHANI_POST_S = 8.0, 3.0       # chani: her annotation app's CLIP_BEFORE / CLIP_AFTER


def clip(s: dict) -> tuple[int, int, int]:
    """First frame, last frame and the t = 0 frame of a scene's clip."""
    if s["source"] == "chani":
        z = s["shot_frame"]
        return z - int(CHANI_PRE_S * FPS), z + int(CHANI_POST_S * FPS), z
    z = s["anchor"]
    return z - int(PRE_S * FPS), z + int(POST_S * FPS), z


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", type=Path, default=ROOT / "data/raw/bundesliga-integrated")
    p.add_argument("--payload", type=Path, default=ROOT / "data/processed/rating_v1/scenes.json")
    p.add_argument("--key", type=Path, default=ROOT / "data/processed/rating_v1/scene_key.csv")
    p.add_argument("--seed", type=int, default=20260925)
    return p.parse_args()


def solver_scenes() -> list[dict]:
    out = []
    for game, results, states in SOLVER:
        res = pd.read_csv(D3 / results)
        prov = {s["index"]: s for s in json.loads((D3 / states).read_text())["states"]}
        for r in res[res["path_max_split"] > 0.2].itertuples():
            s = prov[int(r.index)]
            p = s["provenance"]
            beneficiary = p.get("beneficiary_id") or p["carrier_id"]
            out.append({"source": "solver", "detail": game, "match_id": p["match_id"],
                        "anchor": int(p["onset_frame_id"]), "anchor_kind": "run onset",
                        "runners": [p["runner_id"]], "defenders": [p["defender_id"]],
                        "beneficiaries": [beneficiary], "attack_direction": int(s["scenario"]["attack_direction"]),
                        "solver_index": int(r.index), "path_max_split": float(r.path_max_split)})
    return out


def chani_scenes() -> list[dict]:
    rows = read_xlsx(ROOT / "chani/chani_shot_annotation.xlsx")
    head = rows[0]
    ann = {r[head.index("clip_id")]: dict(zip(head, r)) for r in rows[1:] if any(r)}
    join = pd.read_csv(ROOT / "data/processed/chani_annotation_join_v0_4.csv").set_index("clip_id")
    trace = pd.read_csv(ROOT / "data/processed/chani_trace_strong_medium_v05.csv")
    out = []
    for clip in join.index[join["effect"].isin(["strong", "medium"])]:
        shot = int(join.loc[clip, "shot_frame_id"])
        moments = [(int(x), r.player) for r in trace[trace["clip_id"] == clip].itertuples()
                   for x in re.findall(r"@(\d+)", str(r.detail))]
        ok = sorted((f, who) for f, who in moments if shot - 12 * FPS <= f <= shot - FPS)
        a = ann[clip]
        out.append({"source": "chani", "detail": join.loc[clip, "effect"], "match_id": join.loc[clip, "match_id"],
                    "clip_id": clip, "shot_frame": shot,
                    "anchor": int(ok[-1][0] if ok else shot - 4 * FPS),
                    "anchor_kind": "run moment" if ok else "shot - 4 s",
                    "anchor_player": ok[-1][1] if ok else None,
                    "team_name": str(a["team"]).strip(),
                    "runner_shirts": split_numbers(a.get("offball_attackers", "")),
                    "defender_shirts": split_numbers(a.get("drawn_defenders", "")),
                    "beneficiary_shirts": split_numbers(a.get("space_beneficiaries", ""))})
    return out


def main() -> None:
    args = parse_args()
    scenes = chani_scenes() + solver_scenes()
    random.Random(args.seed).shuffle(scenes)
    # grouped for the raters (2026-09-26): Chani strong, Chani medium, then solver,
    # still shuffled within each group
    rank = {"strong": 0, "medium": 1}
    scenes.sort(key=lambda s: 2 if s["source"] == "solver" else rank[s["detail"]])
    for i, s in enumerate(scenes, 1):
        s["code"] = f"S{i:02d}"
    payload, keyrows = [], []
    for match_id in sorted({s["match_id"] for s in scenes}):
        files = find_bundesliga_files(args.raw_dir, match_id)
        meta = load_bundesliga_match_metadata(files["matchinfo"])
        team_of = {t.name: tid for tid, t in meta.teams.items()}
        mine = [s for s in scenes if s["match_id"] == match_id]
        wanted = set()
        for s in mine:
            first, last, _ = clip(s)
            wanted.update(range(first, last + 1, STEP))
            wanted.add(s["anchor"])            # roles are picked at the anchor, even outside the clip
        frames = load_bundesliga_frames(files["positions"], sorted(wanted))
        for s in mine:
            first, last, zero = clip(s)
            ids = [f for f in range(first, last + 1, STEP) if f in frames]
            at = frames.get(s["anchor"]) or frames[min(ids, key=lambda f: abs(f - s["anchor"]))]
            if s["source"] == "chani":
                att = team_of[s["team_name"]]
                deff = next(t for t in meta.teams if t != att)
                by = {(p.team_id, str(p.shirt_number)): pid for pid, p in meta.players.items()}
                runners = [by[(att, x)] for x in s["runner_shirts"] if (att, x) in by]
                defenders = [by[(deff, x)] for x in s["defender_shirts"] if (deff, x) in by]
                beneficiaries = [by[(att, x)] for x in s["beneficiary_shirts"] if (att, x) in by]
                s["named"] = (runners, defenders, beneficiaries)
                named_runner = [pid for pid in runners if meta.players[pid].short_name == s["anchor_player"]]
                runner = named_runner[0] if named_runner else (runners[0] if runners else None)
                def gap(pid):
                    a, b = at.players.get(pid), at.players.get(runner)
                    return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5 if a and b else float("inf")
                defender = min(defenders, key=gap) if defenders else None
                bene = next((pid for pid in beneficiaries if pid != runner), None)
                s["runners"] = [runner] if runner else []
                s["defenders"] = [defender] if defender else []
                s["beneficiaries"] = [bene] if bene else []
                s["attack_direction"] = int(infer_attacking_direction(at, att, meta))
            else:
                att = at.players[s["runners"][0]].team_id
            sign = 1 if s["attack_direction"] == 1 else -1
            roles = {}
            for key, role in (("runners", "runner"), ("defenders", "defender"), ("beneficiaries", "beneficiary")):
                for pid in s[key]:
                    roles.setdefault(pid, role)
            present = sorted({pid for f in ids for pid in frames[f].players})
            players = []
            for pid in present:
                team = next(frames[f].players[pid].team_id for f in ids if pid in frames[f].players)
                xy = [[round(sign * frames[f].players[pid].x, 2), round(sign * frames[f].players[pid].y, 2)]
                      if pid in frames[f].players else None for f in ids]
                player = meta.players.get(pid)
                players.append({"side": "att" if team == att else "def", "role": roles.get(pid),
                                "shirt": str(player.shirt_number) if player else "",
                                "name": player.short_name if player else "", "xy": xy})
            ball = [[round(sign * frames[f].ball.x, 2), round(sign * frames[f].ball.y, 2)]
                    if frames[f].ball is not None else None for f in ids]
            payload.append({"code": s["code"], "t": [round((f - zero) / FPS, 2) for f in ids],
                            "zero": "슛" if s["source"] == "chani" else "러너 출발",
                            "ball": ball, "players": players})
            name = lambda pid: meta.players[pid].short_name if pid in meta.players else pid
            keyrows.append({"code": s["code"], "source": s["source"], "detail": s["detail"], "match_id": match_id,
                            "clip_id": s.get("clip_id", ""), "clip_first_frame": ids[0], "clip_last_frame": ids[-1],
                            "anchor_frame": s["anchor"], "anchor_kind": s["anchor_kind"],
                            "shot_frame": s.get("shot_frame", ""), "solver_index": s.get("solver_index", ""),
                            "path_max_split": s.get("path_max_split", ""),
                            "runners": "; ".join(map(name, s["runners"])),
                            "defenders": "; ".join(map(name, s["defenders"])),
                            "beneficiaries": "; ".join(map(name, s["beneficiaries"])),
                            "chani_named_runners": "; ".join(map(name, s["named"][0])) if "named" in s else "",
                            "chani_named_defenders": "; ".join(map(name, s["named"][1])) if "named" in s else "",
                            "chani_named_beneficiaries": "; ".join(map(name, s["named"][2])) if "named" in s else ""})
        print(f"  {match_id}: {len(mine)}장면", flush=True)
        del frames
    payload.sort(key=lambda p: p["code"])
    args.payload.parent.mkdir(parents=True, exist_ok=True)
    args.payload.write_text(json.dumps(payload, separators=(",", ":")))
    pd.DataFrame(keyrows).sort_values("code").to_csv(args.key, index=False)
    missing = [k["code"] for k in keyrows if not k["runners"] or not k["defenders"]]
    print(f"장면 {len(payload)}개 → {args.payload} ({args.payload.stat().st_size / 1e6:.1f} MB)")
    print(f"대응표 → {args.key}" + (f" · 러너나 수비 표시가 빠진 장면: {missing}" if missing else ""))


if __name__ == "__main__":
    main()
