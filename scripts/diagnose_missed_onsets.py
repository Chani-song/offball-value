#!/usr/bin/env python3
"""Why does run-onset detection miss the runs Chani marked as strong/medium?

Coverage of Chani's labels, decomposed, says the loss is concentrated in onset
detection, not phase segmentation: for the 5 strong labels the phase gate is
5/5 but onset detection is 3/5, and for strong+medium it is 33/45 -> 19/45.
So the question is not "is this moment in our population" but "why did the
detector not fire on a run a human called obvious".

This answers it per scene instead of guessing. For every strong/medium label
whose shot sits inside a detected possession phase but whose phase carries no
onset, it takes the off-ball attacker Chani named, pulls his real trajectory
around the shot, and reruns the kinematic detector while relaxing one threshold
at a time. The output says which threshold each miss is blocked by and how far
short it fell.

Jersey numbers are mapped to DFL object ids through the match metadata and the
mapping is validated against the shooter name Chani recorded; rows that fail
that check are reported, not silently used.

Human labels are the evaluation. Nothing here feeds a model.
"""

from __future__ import annotations

import argparse
import re
import zipfile
from dataclasses import replace
from pathlib import Path
from xml.etree import ElementTree as ET

import pandas as pd

from offball_value.bundesliga import (
    FPS,
    find_bundesliga_files,
    load_bundesliga_frames,
    load_bundesliga_match_metadata,
    normalize_bundesliga_match_id,
)
from offball_value.run_onset import RunOnsetConfig, detect_kinematic_run_onsets

NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"

# One at a time, most-permissive first, so a miss is attributed to the single
# threshold that unblocks it rather than to a bundle of simultaneous changes.
SWEEP = {
    "minimum_speed_gain_mps": (1.50, 1.20, 1.00, 0.80, 0.60, 0.40),
    "minimum_direction_change_degrees": (30.0, 25.0, 20.0, 15.0, 10.0),
    "minimum_post_displacement_m": (3.0, 2.5, 2.0, 1.5, 1.0),
    "acceleration_onset_threshold_mps2": (0.75, 0.60, 0.45, 0.30),
}


def read_xlsx(path: Path, sheet_index: int = 0) -> list[list[str]]:
    archive = zipfile.ZipFile(path)
    shared: list[str] = []
    if "xl/sharedStrings.xml" in archive.namelist():
        for si in ET.fromstring(archive.read("xl/sharedStrings.xml")):
            shared.append("".join(t.text or "" for t in si.iter(NS + "t")))
    names = [n for n in archive.namelist() if re.match(r"xl/worksheets/sheet\d+\.xml$", n)]
    names.sort(key=lambda n: int(re.search(r"(\d+)", n).group(1)))
    root = ET.fromstring(archive.read(names[sheet_index]))
    rows: list[list[str]] = []
    for r in root.iter(NS + "row"):
        cells: dict[int, str] = {}
        for c in r.findall(NS + "c"):
            col = re.match(r"([A-Z]+)", c.get("r")).group(1)
            idx = 0
            for ch in col:
                idx = idx * 26 + (ord(ch) - 64)
            v = c.find(NS + "v")
            inline = c.find(NS + "is")
            if c.get("t") == "s" and v is not None:
                value = shared[int(v.text)]
            elif inline is not None:
                value = "".join(t.text or "" for t in inline.iter(NS + "t"))
            elif v is not None:
                value = v.text
            else:
                value = ""
            cells[idx - 1] = value
        if cells:
            rows.append([cells.get(i, "") for i in range(max(cells) + 1)])
    width = max((len(r) for r in rows), default=0)
    return [r + [""] * (width - len(r)) for r in rows]


def split_numbers(value: str) -> list[str]:
    return [x for x in re.split(r"[,;/\s]+", str(value).strip()) if x]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--annotations", type=Path, default=Path("chani/chani_shot_annotation.xlsx"))
    p.add_argument("--join", type=Path, default=Path("data/processed/chani_annotation_join_v0_4.csv"))
    p.add_argument("--root", type=Path, default=Path("data/processed/run_onset_v0_4"))
    p.add_argument("--raw-dir", type=Path, default=Path("data/raw/bundesliga-integrated"))
    p.add_argument("--effects", nargs="+", default=["strong", "medium"])
    p.add_argument("--pre-seconds", type=float, default=30.0)
    p.add_argument("--post-seconds", type=float, default=3.0)
    p.add_argument("--output", type=Path, default=Path("data/processed/missed_onset_diagnosis.csv"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_xlsx(args.annotations)
    header = rows[0]
    ann = {r[header.index("clip_id")]: dict(zip(header, r)) for r in rows[1:] if any(r)}

    join = pd.read_csv(args.join)
    phases = pd.read_csv(args.root / "attacking_phases.csv")
    cands = pd.read_csv(args.root / "shot_context_run_onsets.csv")
    tol = 3.0 * FPS

    targets = join[join["labelled"] & join["effect"].isin(args.effects)]
    print(f"{len(targets)} {'/'.join(args.effects)} labels\n")

    work: dict[str, list[dict]] = {}
    stats = {"phase_missing": 0, "onset_present": 0, "to_check": 0}
    for _, r in targets.iterrows():
        match_id, frame = r["match_id"], r["shot_frame_id"]
        P = phases[phases["match_id"] == match_id]
        inph = P[
            (P["possession_start_frame_id"] <= frame)
            & (P["possession_end_frame_id"] + tol >= frame)
        ]
        if inph.empty:
            stats["phase_missing"] += 1
            continue
        p0 = inph.iloc[0]
        S = cands[cands["match_id"] == match_id]
        inside = S[
            (S["frame_id"] >= p0["possession_start_frame_id"])
            & (S["frame_id"] <= p0["possession_end_frame_id"])
        ]
        if len(inside):
            stats["onset_present"] += 1
            continue
        stats["to_check"] += 1
        work.setdefault(match_id, []).append(
            {"clip_id": r["clip_id"], "shot_frame_id": int(frame), "effect": r["effect"]}
        )

    print(f"phase not captured     {stats['phase_missing']}")
    print(f"onset inside the phase {stats['onset_present']}")
    print(f"→ to diagnose          {stats['to_check']}\n")

    base = RunOnsetConfig()
    out: list[dict] = []
    for match_id, items in sorted(work.items()):
        files = find_bundesliga_files(args.raw_dir, match_id)
        meta = load_bundesliga_match_metadata(files["matchinfo"])
        by_team_shirt = {
            (p.team_id, str(p.shirt_number)): p for p in meta.players.values()
        }
        name_to_team = {t.name: tid for tid, t in meta.teams.items()}

        wanted: set[int] = set()
        for item in items:
            lo = item["shot_frame_id"] - int(args.pre_seconds * FPS)
            hi = item["shot_frame_id"] + int(args.post_seconds * FPS)
            wanted.update(range(lo, hi + 1))
        frames = load_bundesliga_frames(files["positions"], sorted(wanted))
        print(f"=== {match_id} ({len(items)} scenes, {len(frames)} frames) ===", flush=True)

        for item in items:
            rec = ann.get(item["clip_id"])
            if rec is None:
                print(f"  {item['clip_id']}: no annotation")
                continue
            team_id = name_to_team.get(str(rec["team"]).strip())
            if team_id is None:
                print(f"  {item['clip_id']}: team name '{rec['team']}' not matched")
                out.append({**item, "status": "team_unmatched"})
                continue
            # Check: is the shooter Chani recorded on that team?
            shooter = str(rec.get("shooter", "")).strip()
            squad = {p.short_name for p in meta.players.values() if p.team_id == team_id}
            shooter_ok = shooter in squad

            lo = item["shot_frame_id"] - int(args.pre_seconds * FPS)
            hi = item["shot_frame_id"] + int(args.post_seconds * FPS)
            ids = [f for f in range(lo, hi + 1) if f in frames]

            for shirt in split_numbers(rec.get("offball_attackers", "")):
                player = by_team_shirt.get((team_id, shirt))
                if player is None:
                    out.append({**item, "shirt": shirt, "status": "shirt_unmatched",
                                "shooter_ok": shooter_ok})
                    print(f"  {item['clip_id']} shirt {shirt}: not matched")
                    continue
                track = [
                    (f, frames[f].players[player.player_id].x, frames[f].players[player.player_id].y)
                    for f in ids
                    if player.player_id in frames[f].players
                ]
                if len(track) < int(2 * FPS):
                    out.append({**item, "shirt": shirt, "player_id": player.player_id,
                                "player": player.short_name, "status": "track_too_short",
                                "shooter_ok": shooter_ok})
                    continue
                fs = [t[0] for t in track]
                xs = [t[1] for t in track]
                ys = [t[2] for t in track]

                row = {**item, "shirt": shirt, "player_id": player.player_id,
                       "player": player.short_name, "shooter": shooter,
                       "shooter_ok": shooter_ok, "track_frames": len(track)}
                baseline = detect_kinematic_run_onsets(fs, xs, ys, base)
                row["baseline_onsets"] = len(baseline)
                row["status"] = "detected_by_baseline" if baseline else "missed"

                for field, values in SWEEP.items():
                    unlocked = None
                    for value in values[1:]:
                        relaxed = replace(base, **{field: value})
                        if detect_kinematic_run_onsets(fs, xs, ys, relaxed):
                            unlocked = value
                            break
                    row[f"unlock_{field}"] = unlocked
                out.append(row)
                mark = "✓detected" if baseline else "✗missed"
                unlocks = {k.replace("unlock_", ""): v for k, v in row.items()
                           if k.startswith("unlock_") and v is not None}
                print(f"  {item['clip_id']} [{item['effect']}] {player.short_name}(#{shirt}) "
                      f"{mark}  unlocked by={unlocks or 'none'}"
                      + ("" if shooter_ok else "  ⚠shooter check failed"))
        del frames

    frame = pd.DataFrame(out)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)

    print("\n" + "=" * 66)
    if "status" in frame:
        print("status:", dict(frame["status"].value_counts()))
    if "shooter_ok" in frame:
        bad = frame[frame["shooter_ok"] == False]  # noqa: E712
        print(f"shooter name check failed: {len(bad)}/{len(frame)}")
    missed = frame[frame.get("status") == "missed"] if "status" in frame else frame.iloc[:0]
    if len(missed):
        print(f"\nmissed with the baseline settings: {len(missed)}. Which threshold, lowered, catches them:")
        for field in SWEEP:
            col = f"unlock_{field}"
            if col not in missed:
                continue
            n = missed[col].notna().sum()
            if n:
                print(f"  {field:36} {n:>2} unlocked  "
                      f"(median value needed {missed[col].median():.2f}, baseline {SWEEP[field][0]})")
        none_unlocked = missed[[f"unlock_{f}" for f in SWEEP]].isna().all(axis=1).sum()
        print(f"  not unlocked by any single one: {none_unlocked}")
    print(f"\nsaved: {args.output}")


if __name__ == "__main__":
    main()
