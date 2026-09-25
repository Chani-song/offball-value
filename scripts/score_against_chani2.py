#!/usr/bin/env python3
"""Corrected scoring of the pipeline's runner against Chani's off-ball attacker.

The previous version matched a shot to any phase whose [start, end+3s] spanned
it. That window admits the OPPONENT possession beginning just after the shot,
and it fired: J03WOH shot_010 matched 53925-54370, owned by the opposing team,
instead of 52870-53400, owned by the shooting team. Every "defending-team
runner" and "onset after the shot" in that report was this mismatch, not the
pipeline -- across all five builds the pipeline produces 0 of each.

The correct rule: a shot belongs to a possession by the SHOOTING team, and the
possession ends at or just before the shot. So candidates are restricted to
phases whose team_id is Chani's shooting team, and the one whose end sits
closest to the shot wins. How far "closest" may be is not obvious, so the gap
is reported as a distribution and the score is given across several windows
instead of resting on one chosen number.

Human labels are the evaluation and never enter the model.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from offball_value.bundesliga import (
    FPS,
    find_bundesliga_files,
    load_bundesliga_match_metadata,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diagnose_missed_onsets import read_xlsx, split_numbers  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--annotations", type=Path, default=Path("chani/chani_shot_annotation.xlsx"))
    p.add_argument("--join", type=Path, default=Path("data/processed/chani_annotation_join_v0_4.csv"))
    p.add_argument("--root", type=Path, default=Path("data/processed/run_onset_v0_5/dir25"))
    p.add_argument("--raw-dir", type=Path, default=Path("data/raw/bundesliga-integrated"))
    p.add_argument("--windows", type=float, nargs="+", default=[3.0, 5.0, 10.0, 20.0, 30.0])
    p.add_argument("--output", type=Path, default=Path("data/processed/chani_runner_score.csv"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_xlsx(args.annotations)
    header = rows[0]
    ann = {r[header.index("clip_id")]: dict(zip(header, r)) for r in rows[1:] if any(r)}

    join = pd.read_csv(args.join)
    phases = pd.read_csv(args.root / "attacking_phases.csv")
    cands = pd.read_csv(args.root / "shot_context_run_onsets.csv")
    scenes = cands[cands["primary_candidate"] == True]  # noqa: E712

    meta_cache: dict[str, object] = {}

    def meta(match_id: str):
        if match_id not in meta_cache:
            files = find_bundesliga_files(args.raw_dir, match_id)
            meta_cache[match_id] = load_bundesliga_match_metadata(files["matchinfo"])
        return meta_cache[match_id]

    out = []
    for _, r in join[join["labelled"]].iterrows():
        rec = ann.get(r["clip_id"])
        if rec is None:
            continue
        m = meta(r["match_id"])
        team_by_name = {t.name: tid for tid, t in m.teams.items()}
        attack_team = team_by_name.get(str(rec["team"]).strip())
        if attack_team is None:
            out.append({"clip_id": r["clip_id"], "effect": r["effect"], "stage": "team_unmatched"})
            continue
        by_shirt = {(p.team_id, str(p.shirt_number)): p.player_id for p in m.players.values()}
        want = {by_shirt.get((attack_team, s)) for s in split_numbers(rec["offball_attackers"])}
        want.discard(None)

        # 소유팀이 슛 팀인 phase만 후보로 두고, 종료가 슛에 가장 가까운 것을 고른다.
        P = phases[(phases["match_id"] == r["match_id"]) & (phases["team_id"] == attack_team)]
        row = {"clip_id": r["clip_id"], "match_id": r["match_id"], "effect": r["effect"],
               "n_labels": len(want)}
        if P.empty:
            row["stage"] = "no_phase_for_team"
            out.append(row); continue
        gap = (P["possession_end_frame_id"] - r["shot_frame_id"]) / FPS
        idx = gap.abs().idxmin()
        p0 = P.loc[idx]
        row["gap_s"] = float(gap.loc[idx])
        row["phase_seconds"] = float(p0["duration_seconds"])
        row["phase_start"] = int(p0["possession_start_frame_id"])
        row["phase_end"] = int(p0["possession_end_frame_id"])

        ours = scenes[(scenes["match_id"] == r["match_id"])
                      & (scenes["frame_id"] >= p0["possession_start_frame_id"])
                      & (scenes["frame_id"] <= p0["possession_end_frame_id"])]
        row["our_runners"] = len(ours)
        our_ids = set(ours["player_id"].astype(str))
        row["runner_match"] = bool(our_ids & want)
        # 검증용: 우리 러너가 소유팀 선수인가
        row["all_runners_own_team"] = bool((ours["team_id"] == attack_team).all()) if len(ours) else None
        out.append(row)

    frame = pd.DataFrame(out)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)

    ok = frame[frame["gap_s"].notna()] if "gap_s" in frame else frame.iloc[:0]
    print("=" * 72)
    print("검증")
    print("=" * 72)
    bad_team = ok[ok["all_runners_own_team"] == False]  # noqa: E712
    print(f"  소유팀 아닌 러너가 섞인 국면: {len(bad_team)}건 (0이어야 정상)")
    print(f"  possession 종료 − 슛 시각 (초, 음수 = 슛이 종료 뒤):")
    print("   ", ok["gap_s"].describe().to_string().replace("\n", "\n    "))

    print("\n" + "=" * 72)
    print("① 슛이 속한 possession을 우리가 갖고 있는가 (창 크기별)")
    print("=" * 72)
    print(f"{'창':>8}", end="")
    for title, _ in (("strong", None), ("s+m", None), ("전체", None)):
        print(f"{title:>14}", end="")
    print()
    subsets = {"strong": ("strong",), "s+m": ("strong", "medium"),
               "전체": ("strong", "medium", "low")}
    for w in args.windows:
        print(f"{w:>6.0f}s", end="")
        for _, eff in subsets.items():
            f = frame[frame["effect"].isin(eff)]
            n = len(f)
            hit = int((f["gap_s"].abs() <= w).sum()) if "gap_s" in f else 0
            print(f"{hit:>8}/{n:<3}{'':>0}", end="")
        print()

    print("\n" + "=" * 72)
    print("②③ 창 안에 든 것만으로: 런을 잡았나 / 같은 선수인가")
    print("=" * 72)
    for w in args.windows:
        print(f"\n창 ±{w:.0f}초")
        for title, eff in subsets.items():
            f = frame[frame["effect"].isin(eff)]
            n = len(f)
            inw = f[f["gap_s"].abs() <= w]
            got = inw[inw["our_runners"] > 0]
            hit = inw[inw["runner_match"] == True]  # noqa: E712
            print(f"  {title:6} ① {len(inw):>2}/{n:<3}({len(inw)/n:>4.0%})"
                  f"   ② {len(got):>2}/{n:<3}({len(got)/n:>4.0%})"
                  f"   ③ {len(hit):>2}/{n:<3}({len(hit)/n:>4.0%})")
    print(f"\n저장: {args.output}")


if __name__ == "__main__":
    main()
