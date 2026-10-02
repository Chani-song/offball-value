#!/usr/bin/env python3
"""Showcase scenes: the rated scenes every rater scored 4 or 5, with at least one 5.

The reviewer's rule (2026-09-28): keep 5/5, 5/4 and 4/5. On the ratings of Kyuhyeok
and Chani that is 21 of the 75 (Chani strong 4, medium 12; solver 2v1 3, 3v1 2);
Andrew has not rated yet. Every CSV in data/processed/rating_v1/ratings/ is one
rater, so re-running after a third file arrives applies the rule to all three.

Written to data/processed/showcase_v1/, to draw these scenes later in our own format:

  scenes.csv          one row per scene: source, match, clip frames, the three
                      marked players, each rater's answers and note, Chani's
                      annotation, the shot and its result, the solver's split
  tracking/<code>.csv every player and the ball on every frame (25 fps) of the clip
                      the raters saw -- chani: shot - 8 s to shot + 3 s; solver:
                      run onset - 3 s to + 4 s (build_rating_package.clip)
  solver/<code>.json  solver scenes: the solver state and its result rows under
                      agile_2m (which picked the scene for rating) and v3
  README.md           columns, coordinates, provenance

x, y are metres from the centre spot, turned so the attack runs to the right
(the rating page's frame); x_raw, y_raw are the tracking as recorded.
Tracking is licensed: none of this leaves the team.

Usage:
    PYTHONPATH=src:scripts python scripts/export_showcase.py
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from offball_value.bundesliga import (FPS, find_bundesliga_files, infer_attacking_direction,
                                      load_bundesliga_frames, load_bundesliga_match_metadata)
from build_rating_package import D3, SOLVER, clip
from diagnose_missed_onsets import read_xlsx, split_numbers

# The game labels exactly as the team's rating and start sheets write them (Korean for "2v1" / "3v1"):
# data values the code must match, so they are kept verbatim here and used by name below.
GAME_2V1, GAME_3V1 = "2대1", "3대1"

ROOT = Path(__file__).resolve().parents[1]
RATING = ROOT / "data/processed/rating_v1"
V3 = {GAME_2V1: "passer2on1_results_v3.csv", GAME_3V1: "fixedpasser_results_v3.csv"}  # keys = SOLVER game names (Korean 2v1 / 3v1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", type=Path, default=ROOT / "data/raw/bundesliga-integrated")
    p.add_argument("--out", type=Path, default=ROOT / "data/processed/showcase_v1")
    return p.parse_args()


def read_raters() -> dict[str, dict[str, dict]]:
    """rater -> scene code -> that rater's row, from every CSV in ratings/."""
    raters = {}
    for path in sorted((RATING / "ratings").glob("*.csv")):
        rows = list(csv.DictReader(path.open(encoding="utf-8-sig")))
        raters[rows[0]["rater"]] = {r["scene"]: r for r in rows}
    return raters


def selected(raters: dict[str, dict[str, dict]]) -> list[str]:
    codes = []
    for code in next(iter(raters.values())):
        scores = [r[code]["rating"] for r in raters.values()]
        if all(s in ("4", "5") for s in scores) and "5" in scores:
            codes.append(code)
    return codes


def main() -> None:
    args = parse_args()
    raters = read_raters()
    codes = selected(raters)
    key = {r["code"]: r for r in csv.DictReader((RATING / "scene_key.csv").open(encoding="utf-8"))}
    rows = read_xlsx(ROOT / "chani/chani_shot_annotation.xlsx")
    ann = {r[rows[0].index("clip_id")]: dict(zip(rows[0], r)) for r in rows[1:] if any(r)}
    states = {game: {s["index"]: s for s in json.loads((D3 / st).read_text())["states"]} for game, _, st in SOLVER}
    agile = {game: {int(r["index"]): r for r in csv.DictReader((D3 / res).open(encoding="utf-8"))}
             for game, res, _ in SOLVER}
    v3 = {game: {(r["match_id"], int(r["onset"]), r["runner"], r["defender"]): r
                 for r in csv.DictReader((D3 / fn).open(encoding="utf-8"))} for game, fn in V3.items()}
    print(f"raters {', '.join(raters)} · 5/5·5/4·4/5 scenes {len(codes)}: {' '.join(codes)}")

    (args.out / "tracking").mkdir(parents=True, exist_ok=True)
    (args.out / "solver").mkdir(parents=True, exist_ok=True)
    scenes = []
    for code in codes:
        k = key[code]
        s = {"source": k["source"], "anchor": int(k["anchor_frame"]),
             "shot_frame": int(k["shot_frame"]) if k["shot_frame"] else None}
        first, last, zero = clip(s)
        scenes.append((code, k, s, first, last, zero))

    out_rows = []
    for match_id in sorted({k["match_id"] for _, k, *_ in scenes}):
        files = find_bundesliga_files(args.raw_dir, match_id)
        meta = load_bundesliga_match_metadata(files["matchinfo"])
        mine = [x for x in scenes if x[1]["match_id"] == match_id]
        wanted = set()
        for _, _, s, first, last, _ in mine:
            wanted.update(range(first, last + 1))
            wanted.add(s["anchor"])
        frames = load_bundesliga_frames(files["positions"], sorted(wanted))
        by_name = {}
        for pid, p in meta.players.items():
            by_name.setdefault(p.short_name, []).append(pid)

        def pid_of(name: str) -> str:
            assert len(by_name.get(name, [])) == 1, f"{match_id}: {name!r} is not one player"
            return by_name[name][0]

        for code, k, s, first, last, zero in mine:
            game = k["detail"]
            role_id = {role: pid_of(k[col]) for role, col in
                       (("runner", "runners"), ("defender", "defenders"), ("beneficiary", "beneficiaries")) if k[col]}
            if k["source"] == "chani":
                a = ann[k["clip_id"]]
                att = next(tid for tid, t in meta.teams.items() if t.name == str(a["team"]).strip())
                at = frames[s["anchor"]]
                direction = int(infer_attacking_direction(at, att, meta))
                state = None
            else:
                a = {}
                att = meta.players[role_id["runner"]].team_id
                state = states[game][int(k["solver_index"])]
                direction = int(state["scenario"]["attack_direction"])
            deff = next(t for t in meta.teams if t != att)
            sign = 1 if direction == 1 else -1
            role = {pid: r for r, pid in role_id.items()}
            ids = [f for f in range(first, last + 1) if f in frames]

            with (args.out / "tracking" / f"{code}.csv").open("w", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                w.writerow(["frame_id", "t", "period", "object_id", "side", "team_id", "shirt", "name", "role",
                            "x", "y", "z", "speed", "x_raw", "y_raw"])
                for f in ids:
                    fr = frames[f]
                    t = round((f - zero) / FPS, 2)
                    for pid, p in sorted(fr.players.items()):
                        pm = meta.players.get(pid)
                        w.writerow([f, t, fr.period, pid, "att" if p.team_id == att else "def", p.team_id,
                                    pm.shirt_number if pm else "", pm.short_name if pm else "", role.get(pid, ""),
                                    round(sign * p.x, 3), round(sign * p.y, 3), "",
                                    "" if p.speed is None else round(p.speed, 3), round(p.x, 3), round(p.y, 3)])
                    if fr.ball is not None:
                        b = fr.ball
                        w.writerow([f, t, fr.period, "ball", "ball", "", "", "", "", round(sign * b.x, 3),
                                    round(sign * b.y, 3), "" if b.z is None else round(b.z, 3),
                                    "" if b.speed is None else round(b.speed, 3), round(b.x, 3), round(b.y, 3)])

            if state is not None:
                prov = state["provenance"]
                (args.out / "solver" / f"{code}.json").write_text(json.dumps({
                    "game": game, "state": state, "result_agile_2m": agile[game][int(k["solver_index"])],
                    "result_v3": v3[game].get((match_id, int(prov["onset_frame_id"]), prov["runner_name"],
                                               prov["defender_name"]))}, ensure_ascii=False, indent=1))

            player = lambda pid: meta.players[pid]
            row = {"code": code, "source": k["source"], "detail": game, "match_id": match_id,
                   "match": f"{meta.home_team_name} vs {meta.away_team_name}", "period": frames[zero].period,
                   "zero_kind": "shot" if k["source"] == "chani" else "run onset", "zero_frame": zero,
                   "clip_first_frame": ids[0], "clip_last_frame": ids[-1], "n_frames": len(ids),
                   "attack_team": meta.teams[att].name, "defend_team": meta.teams[deff].name,
                   "attack_direction_raw": direction,
                   "anchor_frame": s["anchor"], "anchor_kind": k["anchor_kind"],
                   "shot_frame": s["shot_frame"] or "", "shot_result": a.get("shot_result", ""),
                   "shooter": a.get("shooter", "")}
            for r, pid in (("runner", role_id.get("runner")), ("defender", role_id.get("defender")),
                           ("beneficiary", role_id.get("beneficiary"))):
                row[f"{r}_id"] = pid or ""
                row[f"{r}_name"] = player(pid).short_name if pid else ""
                row[f"{r}_shirt"] = player(pid).shirt_number if pid else ""
            row.update({"chani_named_runner_shirts": " ".join(split_numbers(a.get("offball_attackers", ""))),
                        "chani_named_defender_shirts": " ".join(split_numbers(a.get("drawn_defenders", ""))),
                        "chani_named_beneficiary_shirts": " ".join(split_numbers(a.get("space_beneficiaries", ""))),
                        "chani_annotation_note": a.get("notes", "")})
            for name, rr in raters.items():
                x = rr[code]
                row[f"rating_{name}"] = x["rating"]
                row[f"offball_{name}"] = x.get("offball", "")
                row[f"note_{name}"] = x["note"]
            row["rating_mean"] = sum(int(rr[code]["rating"]) for rr in raters.values()) / len(raters)
            if state is not None:
                res3 = v3[game].get((match_id, s["anchor"], k["runners"], k["defenders"]))
                row.update({"solver_index": k["solver_index"], "split_agile_2m": round(float(k["path_max_split"]), 4),
                            "split_v3": round(float(res3["path_max_split"]), 4) if res3 else ""})
            else:
                row.update({"solver_index": "", "split_agile_2m": "", "split_v3": ""})
            out_rows.append(row)
        print(f"  {match_id}: {len(mine)} scenes", flush=True)
        del frames

    out_rows.sort(key=lambda r: r["code"])
    with (args.out / "scenes.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0]))
        w.writeheader()
        w.writerows(out_rows)
    print(f"{len(out_rows)} scenes → {args.out}")


if __name__ == "__main__":
    main()
