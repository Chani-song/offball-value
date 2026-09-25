#!/usr/bin/env python3
"""How often does possession segmentation cut a spell the same team keeps playing?

Chani's strong scene J03WOH shot_010 has no phase at all. The segmenter ended
Fortuna's possession at frame 53400 on a stoppage_event; DFL's own events show
Fortuna passing again 27 frames (1.1 s) later and shooting at 53905. The whole
build-up and the shot fell into a 21-second hole that belongs to nobody.

If that is common, no amount of onset tuning reaches those scenes, because the
possession they live in does not exist. This measures it over all 7 matches:
for every phase boundary, whether the SAME team has an on-ball event shortly
after the cut, and how much football ends up inside the holes between phases.

Read-only.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from offball_value.bundesliga import (
    FPS,
    find_bundesliga_files,
    list_bundesliga_match_ids,
    load_bundesliga_events,
    load_bundesliga_frame_clock,
)

ON_BALL = {"pass", "cross", "shotatgoal", "otherballaction", "ballclaiming", "tacklinggame"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("data/processed/run_onset_v0_5/dir25"))
    p.add_argument("--raw-dir", type=Path, default=Path("data/raw/bundesliga-integrated"))
    p.add_argument("--resume-seconds", type=float, default=5.0)
    p.add_argument("--output", type=Path, default=Path("data/processed/possession_cut_audit.csv"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    phases = pd.read_csv(args.root / "attacking_phases.csv")
    rows = []
    holes = []
    for match_id in sorted(phases["match_id"].unique()):
        files = find_bundesliga_files(args.raw_dir, match_id)
        clock = load_bundesliga_frame_clock(files["positions"])
        ev = load_bundesliga_events(files["events"], clock).dropna(subset=["frame_id"])
        ev = ev[ev["event_type"].str.lower().isin(ON_BALL)]
        P = phases[phases["match_id"] == match_id].sort_values("possession_start_frame_id")
        print(f"{match_id}: phase {len(P)}, 이벤트 {len(ev)}", flush=True)

        for _, p0 in P.iterrows():
            end = float(p0["possession_end_frame_id"])
            team = p0["team_id"]
            win = ev[(ev["frame_id"] > end) & (ev["frame_id"] <= end + args.resume_seconds * FPS)]
            same = win[win["team_id"] == team]
            rows.append({
                "match_id": match_id, "team_id": team,
                "end_frame": int(end), "reason": p0["boundary_reason"],
                "duration_s": float(p0["duration_seconds"]),
                "same_team_events_after": len(same),
                "first_same_team_gap_s": (float(same["frame_id"].min() - end) / FPS) if len(same) else np.nan,
            })

        ends = P["possession_end_frame_id"].tolist()
        starts = P["possession_start_frame_id"].tolist()
        for a, b in zip(ends[:-1], starts[1:]):
            if b > a:
                inside = ev[(ev["frame_id"] > a) & (ev["frame_id"] < b)]
                shots = inside[inside["event_type"].str.lower() == "shotatgoal"]
                holes.append({"match_id": match_id, "gap_s": float(b - a) / FPS,
                              "events_inside": len(inside), "shots_inside": len(shots)})

    cuts = pd.DataFrame(rows)
    hole = pd.DataFrame(holes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cuts.to_csv(args.output, index=False)
    hole.to_csv(args.output.with_name("possession_holes.csv"), index=False)

    print("\n" + "=" * 72)
    print(f"A. 종료 직후 {args.resume_seconds:.0f}초 안에 같은 팀이 다시 공을 만지는가")
    print("=" * 72)
    n = len(cuts)
    resumed = (cuts["same_team_events_after"] > 0).sum()
    print(f"  phase {n}개 중 {resumed}개 ({resumed / n:.1%})가 같은 팀 이벤트로 이어짐")
    print(f"  이어질 때까지 간격(초): 중앙값 {cuts['first_same_team_gap_s'].median():.2f}")
    print("\n  종료사유별:")
    g = cuts.groupby("reason").agg(
        n=("reason", "size"),
        resumed=("same_team_events_after", lambda s: (s > 0).sum()),
        gap=("first_same_team_gap_s", "median"),
    )
    g["비율"] = (g["resumed"] / g["n"]).map("{:.1%}".format)
    print(g.to_string())

    print("\n" + "=" * 72)
    print("B. phase 사이 공백(어느 팀도 소유 안 함)에 무엇이 들어있나")
    print("=" * 72)
    print(f"  공백 {len(hole)}개, 총 {hole['gap_s'].sum() / 60:.0f}분")
    print(f"  공백 길이(초): 중앙값 {hole['gap_s'].median():.1f}  75% {hole['gap_s'].quantile(.75):.1f}"
          f"  최대 {hole['gap_s'].max():.1f}")
    print(f"  공백 안의 온볼 이벤트: 총 {int(hole['events_inside'].sum())}")
    print(f"  공백 안의 슛: 총 {int(hole['shots_inside'].sum())}  "
          f"(슛이 든 공백 {int((hole['shots_inside'] > 0).sum())}개)")


if __name__ == "__main__":
    main()
