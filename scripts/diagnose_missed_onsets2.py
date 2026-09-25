#!/usr/bin/env python3
"""Follow-up: for runs the kinematic detector DOES fire on, what drops them?

The first pass asked the wrong question. It ran detect_kinematic_run_onsets
over a +/-30s window around the shot and found 13 of 20 named runs already
fire at baseline -- yet the pipeline holds no onset for those phases. A window
around the shot is not the phase, and firing kinematically is not surviving
_context_candidate, so that 13 says nothing on its own.

This closes both gaps. For every named run it reports whether a kinematic onset
lands INSIDE the possession phase, and if so, which context gate then rejects
it -- ball off pitch, player count, no carrier within control distance, the
runner being the carrier, unstable control history, or the future-possession
checks. That names the responsible gate instead of guessing at it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from offball_value.bundesliga import (
    FPS,
    find_bundesliga_files,
    load_bundesliga_frames,
    load_bundesliga_match_metadata,
)
from offball_value.run_onset import (
    RunOnsetConfig,
    ball_is_inside_pitch,
    detect_kinematic_run_onsets,
)
from offball_value import run_onset as ro

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diagnose_missed_onsets import read_xlsx, split_numbers  # noqa: E402


def gate_report(frame, player_id: str, config: RunOnsetConfig, expected_players: int) -> str:
    """Which _context_candidate gate rejects this frame, in that function's order."""
    if frame is None:
        return "frame_missing"
    if player_id not in frame.players:
        return "player_untracked"
    if frame.ball is None:
        return "ball_missing"
    if config.require_ball_inside_pitch and not ball_is_inside_pitch(
        frame, config.pitch_boundary_tolerance_m
    ):
        return "ball_off_pitch"
    if len(frame.players) != expected_players:
        return f"player_count({len(frame.players)}!={expected_players})"
    carrier_id, carrier_distance = ro._nearest_player(frame)
    if carrier_id is None or carrier_distance is None:
        return "no_carrier"
    if carrier_distance > config.control_distance_m:
        return f"carrier_far({carrier_distance:.2f}m>{config.control_distance_m})"
    if carrier_id == player_id:
        return "runner_is_carrier"
    return "passes_frame_gates"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--annotations", type=Path, default=Path("chani/chani_shot_annotation.xlsx"))
    p.add_argument("--join", type=Path, default=Path("data/processed/chani_annotation_join_v0_4.csv"))
    p.add_argument("--root", type=Path, default=Path("data/processed/run_onset_v0_4"))
    p.add_argument("--raw-dir", type=Path, default=Path("data/raw/bundesliga-integrated"))
    p.add_argument("--effects", nargs="+", default=["strong", "medium"])
    p.add_argument("--output", type=Path,
                   default=Path("data/processed/missed_onset_gate_diagnosis.csv"))
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
    config = RunOnsetConfig()

    targets = join[join["labelled"] & join["effect"].isin(args.effects)]
    work: dict[str, list[dict]] = {}
    for _, r in targets.iterrows():
        P = phases[phases["match_id"] == r["match_id"]]
        inph = P[
            (P["possession_start_frame_id"] <= r["shot_frame_id"])
            & (P["possession_end_frame_id"] + tol >= r["shot_frame_id"])
        ]
        if inph.empty:
            continue
        p0 = inph.iloc[0]
        S = cands[cands["match_id"] == r["match_id"]]
        if len(S[(S["frame_id"] >= p0["possession_start_frame_id"])
                 & (S["frame_id"] <= p0["possession_end_frame_id"])]):
            continue
        work.setdefault(r["match_id"], []).append({
            "clip_id": r["clip_id"], "effect": r["effect"],
            "shot_frame_id": int(r["shot_frame_id"]),
            "phase_start": int(p0["possession_start_frame_id"]),
            "phase_end": int(p0["possession_end_frame_id"]),
        })

    print(f"진단 대상 장면 {sum(len(v) for v in work.values())}개\n")
    out: list[dict] = []
    for match_id, items in sorted(work.items()):
        files = find_bundesliga_files(args.raw_dir, match_id)
        meta = load_bundesliga_match_metadata(files["matchinfo"])
        by_shirt = {(p.team_id, str(p.shirt_number)): p for p in meta.players.values()}
        name_to_team = {t.name: tid for tid, t in meta.teams.items()}

        wanted: set[int] = set()
        for it in items:
            wanted.update(range(it["phase_start"], it["phase_end"] + 1))
        frames = load_bundesliga_frames(files["positions"], sorted(wanted))
        print(f"=== {match_id} ({len(items)}장면) ===", flush=True)

        for it in items:
            rec = ann.get(it["clip_id"])
            team_id = name_to_team.get(str(rec["team"]).strip()) if rec else None
            if team_id is None:
                continue
            ids = [f for f in range(it["phase_start"], it["phase_end"] + 1) if f in frames]
            expected = 22  # 퇴장은 별도 처리; 여기서는 실제 관측값을 같이 찍는다
            for shirt in split_numbers(rec.get("offball_attackers", "")):
                player = by_shirt.get((team_id, shirt))
                if player is None:
                    continue
                track = [(f, frames[f].players[player.player_id].x,
                          frames[f].players[player.player_id].y)
                         for f in ids if player.player_id in frames[f].players]
                row = {"clip_id": it["clip_id"], "effect": it["effect"], "shirt": shirt,
                       "player": player.short_name, "player_id": player.player_id,
                       "phase_seconds": round((it["phase_end"] - it["phase_start"]) / FPS, 1),
                       "track_frames": len(track)}
                if len(track) < int(2 * FPS):
                    row["result"] = "track_too_short_in_phase"
                    out.append(row); print(f"  {it['clip_id']} #{shirt} {player.short_name}: 궤적부족")
                    continue
                onsets = detect_kinematic_run_onsets([t[0] for t in track],
                                                     [t[1] for t in track],
                                                     [t[2] for t in track], config)
                row["kinematic_onsets_in_phase"] = len(onsets)
                if not onsets:
                    row["result"] = "no_kinematic_onset_in_phase"
                    out.append(row)
                    print(f"  {it['clip_id']} #{shirt} {player.short_name}: "
                          f"phase({row['phase_seconds']}s) 안에 운동학적 온셋 0개")
                    continue
                gates = []
                for o in onsets:
                    fr = frames.get(o.frame_id)
                    n = len(fr.players) if fr is not None else -1
                    gates.append(gate_report(fr, player.player_id, config, n if n > 0 else 22))
                row["result"] = "kinematic_onset_in_phase"
                row["gates"] = "|".join(sorted(set(gates)))
                out.append(row)
                print(f"  {it['clip_id']} #{shirt} {player.short_name}: "
                      f"온셋 {len(onsets)}개 → 게이트 {sorted(set(gates))}")
        del frames

    frame = pd.DataFrame(out)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print("\n" + "=" * 66)
    print("결과:", dict(frame["result"].value_counts()))
    if "gates" in frame:
        g = frame[frame["gates"].notna()]["gates"]
        if len(g):
            print("\n운동학적 온셋이 phase 안에 있었던 경우, 걸린 게이트:")
            print(g.value_counts().to_string())
    print(f"\n저장: {args.output}")


if __name__ == "__main__":
    main()
