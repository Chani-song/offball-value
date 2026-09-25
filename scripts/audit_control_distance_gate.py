#!/usr/bin/env python3
"""Is the carrier-distance gate rejecting passes in flight?

_context_candidate requires the ball's nearest player to be within
RunOnsetConfig.control_distance_m (1.5 m) at the onset frame, and calls that
player the carrier. Four strong/medium runs Chani named were dropped by it, at
2.03, 3.01, 5.05 and 5.79 m. While a pass is in the air nobody is within 1.5 m
of the ball, so the gate would discard every off-ball run that starts during a
pass -- one of the moments the study is about.

But "far from the ball" also describes a loose ball or a tracking artefact, and
those SHOULD be rejected. The ball carries z and speed, so the two are
separable rather than a matter of opinion. This reports, for the rejected runs
and for the pool at large, how high and how fast the ball was.

The blast radius matters too: control_distance_m is an independent field on
nine configs, so changing RunOnsetConfig's copy does not touch phase
segmentation or the settled gate. It DOES change ball_carrier_id, which flows
downstream, so this also reports how often the nearest player would change.
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
    load_bundesliga_frames,
    load_bundesliga_match_metadata,
)
from offball_value.run_onset import _nearest_player

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diagnose_missed_onsets import read_xlsx, split_numbers  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--gate-csv", type=Path,
                   default=Path("data/processed/missed_onset_gate_diagnosis.csv"))
    p.add_argument("--root", type=Path, default=Path("data/processed/run_onset_v0_4"))
    p.add_argument("--raw-dir", type=Path, default=Path("data/raw/bundesliga-integrated"))
    p.add_argument("--sample-stride", type=int, default=25)
    p.add_argument("--output", type=Path,
                   default=Path("data/processed/control_distance_audit.csv"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cands = pd.read_csv(args.root / "shot_context_run_onsets.csv")
    phases = pd.read_csv(args.root / "attacking_phases.csv")

    print("=" * 68)
    print("A. 풀 전체: possession phase 안에서 '가장 가까운 선수-공' 거리 분포")
    print("=" * 68)
    rows = []
    for match_id in sorted(phases["match_id"].unique()):
        P = phases[phases["match_id"] == match_id]
        wanted: set[int] = set()
        for _, r in P.iterrows():
            wanted.update(range(int(r["possession_start_frame_id"]),
                                int(r["possession_end_frame_id"]) + 1,
                                args.sample_stride))
        files = find_bundesliga_files(args.raw_dir, match_id)
        frames = load_bundesliga_frames(files["positions"], sorted(wanted))
        for fid, fr in frames.items():
            if fr is None or fr.ball is None:
                continue
            _pid, dist = _nearest_player(fr)
            if dist is None:
                continue
            rows.append({"match_id": match_id, "frame_id": fid, "nearest_m": dist,
                         "ball_z": fr.ball.z, "ball_speed": fr.ball.speed})
        print(f"  {match_id}: 표본 {len(frames)}", flush=True)
        del frames

    d = pd.DataFrame(rows)
    print(f"\n표본 {len(d)}개 (possession phase 내부, {args.sample_stride / FPS:.1f}초 간격)")
    print(d["nearest_m"].describe().to_string())
    for thr in (1.5, 2.0, 2.5, 3.0, 4.0, 5.0):
        share = (d["nearest_m"] > thr).mean()
        print(f"  가장 가까운 선수가 {thr}m 밖: {share:.1%}")

    have_z = d["ball_z"].notna()
    print(f"\n공 높이(z) 있는 표본: {have_z.sum()}/{len(d)}")
    if have_z.any():
        far = d[have_z & (d["nearest_m"] > 1.5)]
        near = d[have_z & (d["nearest_m"] <= 1.5)]
        print(f"  1.5m 이내일 때  z 중앙값 {near['ball_z'].median():.2f}m  "
              f"속도 중앙값 {near['ball_speed'].median():.2f}")
        print(f"  1.5m 밖일 때    z 중앙값 {far['ball_z'].median():.2f}m  "
              f"속도 중앙값 {far['ball_speed'].median():.2f}")
        print(f"  1.5m 밖 중 z>0.5m (공중): {(far['ball_z'] > 0.5).mean():.1%}")
        print(f"  1.5m 밖 중 속도>8 m/s   : {(far['ball_speed'] > 8).mean():.1%}")

    d.to_csv(args.output, index=False)
    print(f"\n저장: {args.output}")

    print("\n" + "=" * 68)
    print("B. 탈락한 strong/medium 런의 온셋 순간 공 상태")
    print("=" * 68)
    if not args.gate_csv.exists():
        print("  게이트 진단 CSV 없음")
        return
    gate = pd.read_csv(args.gate_csv)
    far_rows = gate[gate.get("gates", pd.Series(dtype=str)).astype(str).str.contains("carrier_far", na=False)]
    print(f"  carrier_far로 탈락: {len(far_rows)}건")
    print(far_rows[["clip_id", "effect", "player", "gates"]].to_string(index=False)
          if len(far_rows) else "  없음")


if __name__ == "__main__":
    main()
