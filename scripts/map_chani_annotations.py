#!/usr/bin/env python3
"""Map Chani's shot-anchored annotations onto tracking frames, and test the join.

Two things are unknown and both must hold before the annotations can be used to
check the pipeline:

1. CLOCK. The annotations carry (period, period_seconds) read off video clips.
   The pipeline works in tracking frame ids. period_seconds -> frame is a linear
   map through the half's first BALL frame, but the video clock may be offset
   from the tracking clock. This validates the map by matching every annotated
   shot against DFL's own ShotAtGoal events: if the map is right, each annotated
   shot lands within a fraction of a second of a real shot event, one to one.

2. COVERAGE. The candidate pool is anchored on possession END, not on shots.
   A shot-ending possession puts the two near each other but nothing guarantees
   it. This reports, for each labelled shot, whether the pool holds a run onset
   inside the same possession phase -- which is what makes the strong/medium
   cross-check possible at all. If that rate is near zero the cross-check does
   not exist and the plan has to change.

Human labels are NOT written into anything the model reads. This only produces
a join table for evaluation.

Usage:
    python scripts/map_chani_annotations.py \
        --candidates data/processed/settled_possession_run_onset_v0_2_full/shot_context_run_onsets.csv \
        --output data/processed/chani_annotation_join.csv
"""

from __future__ import annotations

import argparse
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import pandas as pd

from offball_value.bundesliga import (
    FPS,
    find_bundesliga_files,
    load_bundesliga_events,
    load_bundesliga_frame_clock,
    normalize_bundesliga_match_id,
)

NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


def read_xlsx(path: Path, sheet_index: int = 0) -> list[list[str]]:
    """Read a sheet without openpyxl (not installed in our environment)."""
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--annotations", type=Path, default=Path("chani/chani_shot_annotation.xlsx"))
    p.add_argument("--raw-dir", type=Path, default=Path("data/raw/bundesliga-integrated"))
    p.add_argument(
        "--candidates",
        type=Path,
        default=Path(
            "data/processed/settled_possession_run_onset_v0_2_full/shot_context_run_onsets.csv"
        ),
    )
    p.add_argument("--output", type=Path, default=Path("data/processed/chani_annotation_join.csv"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_xlsx(args.annotations)
    header = rows[0]
    records = [dict(zip(header, r)) for r in rows[1:] if any(r)]
    print(f"{len(records)} annotation rows")

    candidates = pd.read_csv(args.candidates)
    print(f"candidate pool {len(candidates)} rows, {candidates['match_id'].nunique()} matches")

    out = []
    for short_id in sorted({r["match_id"] for r in records}):
        match_id = normalize_bundesliga_match_id(short_id)
        files = find_bundesliga_files(args.raw_dir, match_id)
        print(f"\n=== {match_id} ===", flush=True)
        clock = load_bundesliga_frame_clock(files["positions"])
        events = load_bundesliga_events(files["events"], clock=clock)
        shots = events[events["event_type"].str.lower() == "shotatgoal"].copy()
        if shots["frame_id"].isna().any() and "time" in shots.columns:
            # A couple of shots carry no CalculatedFrame; recover them from the
            # timestamp through the same clock so the offset test sees them too.
            filled = shots["time"].map(
                lambda t: clock.frame_for_time(t) if pd.notna(t) else None
            )
            shots["frame_id"] = shots["frame_id"].fillna(filled)
        shots = shots.dropna(subset=["frame_id"])
        print(f"  {len(shots)} DFL ShotAtGoal events")

        mine = [r for r in records if r["match_id"] == short_id]
        pool = candidates[candidates["match_id"] == match_id]

        for rec in mine:
            period = int(rec["period"])
            seconds = float(rec["period_seconds"])
            section = "secondHalf" if period == 2 else "firstHalf"
            start_frame, _ = clock.section_start[section]
            frame = int(round(start_frame + seconds * FPS))

            if len(shots):
                delta = (shots["frame_id"] - frame).abs()
                j = delta.idxmin()
                nearest_shot_frame = int(shots.loc[j, "frame_id"])
                nearest_shot_s = float(delta.loc[j]) / FPS
            else:
                nearest_shot_frame, nearest_shot_s = -1, float("nan")

            in_phase = pool[
                (pool["phase_start_frame_id"] <= frame)
                & (pool["possession_end_frame_id"] >= frame)
            ]
            near = pool[(pool["frame_id"] - frame).abs() <= 10 * FPS]

            out.append(
                {
                    "clip_id": rec["clip_id"],
                    "match_id": match_id,
                    "period": period,
                    "period_seconds": seconds,
                    "match_clock": rec["match_clock"],
                    "shot_frame_id": frame,
                    "nearest_dfl_shot_frame": nearest_shot_frame,
                    "clock_offset_s": nearest_shot_s,
                    "effect": rec["effect"],
                    "labelled": bool(str(rec.get("space_beneficiaries", "")).strip()),
                    "space_beneficiaries": rec.get("space_beneficiaries", ""),
                    "drawn_defenders": rec.get("drawn_defenders", ""),
                    "offball_attackers": rec.get("offball_attackers", ""),
                    "shooter": rec.get("shooter", ""),
                    "onsets_in_phase": len(in_phase),
                    "onsets_within_10s": len(near),
                    "onset_frames_in_phase": ";".join(str(v) for v in in_phase["frame_id"].tolist()),
                }
            )

    frame = pd.DataFrame(out)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)

    print("\n" + "=" * 70)
    print("1. Clock alignment — annotated shot times vs DFL ShotAtGoal events")
    off = frame["clock_offset_s"].dropna()
    print(f"   median {off.median():.3f}s  mean {off.mean():.3f}s  "
          f"P90 {off.quantile(0.9):.3f}s  max {off.max():.3f}s")
    for thr in (0.5, 1.0, 2.0, 5.0):
        print(f"   within {thr}s: {(off <= thr).sum()}/{len(off)} ({(off <= thr).mean():.1%})")

    print("\n2. Coverage — do labelled shots fall inside a candidate-pool possession?")
    lab = frame[frame["labelled"]]
    print(f"   {len(lab)} labelled")
    print(f"   onset in the same possession: {(lab['onsets_in_phase'] > 0).sum()}/{len(lab)} "
          f"({(lab['onsets_in_phase'] > 0).mean():.1%})")
    print(f"   onset within ±10s:            {(lab['onsets_within_10s'] > 0).sum()}/{len(lab)} "
          f"({(lab['onsets_within_10s'] > 0).mean():.1%})")
    print("\n   same-possession hits by effect:")
    for eff in ("strong", "medium", "low"):
        sub = lab[lab["effect"] == eff]
        if len(sub):
            print(f"     {eff:7} {(sub['onsets_in_phase'] > 0).sum():>3}/{len(sub):<3} "
                  f"({(sub['onsets_in_phase'] > 0).mean():.1%})")
    print(f"\nsaved: {args.output}")


if __name__ == "__main__":
    main()
