#!/usr/bin/env python3
"""Where do Chani's labelled scenes leave the pipeline, and on which rule?

For every off-ball attacker Chani named in a strong (or medium) shot clip,
follow the CURRENT pipeline stage by stage and stop at the first rule that
drops him:

  stage 0  settled attacking phase: is the shot inside one of the shooting
           team's phases among the 720
           (run_onset_v0_5/dir25/attacking_phases.csv, 3 s tolerance at the
           end, as the earlier diagnosis scripts)? If not, the nearest phase
           and the gap to it.
  stage 1  run onset, in three layers:
           a. kinematic detector (run_onset.detect_kinematic_run_onsets with
              the v0.5 settings -- the defaults except a 25 deg turn) on his
              track inside the phase. If nothing fires, each threshold is
              relaxed on its own (the old diagnose_missed_onsets grid) to
              show which one blocks him.
           b. context gates, in _context_candidate's order: ball on pitch,
              player count, a carrier within 1.5 m, runner not the carrier,
              not the keeper, runner not on the ball, 0.5 s of stable team
              control, 1.0 s of future same-team possession. The first gate
              that fails is reported with its value.
           c. selection: settled possession, primary candidate, and not
              already reviewed (726 candidates -> 588 scenes).
  stage 2  the pair gate and the protagonist rule (pair_gate_v7.csv): which
           defenders are kept, and the beneficiary.
  solver   the stage-3 inputs: on-pitch and speed limits, run speed 3.8 m/s,
           triangle strict and 2 m (the agile state files and their filters).

A scene that no stage-1 layer explains (a detected, context-passing onset that
is still missing) is reported as such, not guessed at.

Usage:
    PYTHONPATH=src:scripts python scripts/trace_chani_scenes.py --effects strong
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path

import pandas as pd

from offball_value import run_onset as ro
from offball_value.bundesliga import (FPS, expected_player_count_at, find_bundesliga_files,
                                      load_bundesliga_frame_clock, load_bundesliga_frames,
                                      load_bundesliga_match_metadata, load_bundesliga_sendings_off)
from offball_value.run_onset import RunOnsetConfig, detect_kinematic_run_onsets
from diagnose_missed_onsets import read_xlsx, split_numbers

ROOT = Path(__file__).resolve().parents[1]
V05 = ROOT / "data/processed/run_onset_v0_5/dir25"
D3 = ROOT / "data/processed/stage3"
TOL = int(3.0 * FPS)
NO_PHASE_WINDOW_S = 20.0
SWEEP = {
    "minimum_speed_gain_mps": (1.20, 1.00, 0.80, 0.60),
    "minimum_direction_change_degrees": (20.0, 15.0, 10.0),
    "minimum_post_displacement_m": (2.5, 2.0, 1.5, 1.0),
    "acceleration_onset_threshold_mps2": (0.60, 0.45, 0.30),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--effects", nargs="+", default=["strong"])
    p.add_argument("--raw-dir", type=Path, default=ROOT / "data/raw/bundesliga-integrated")
    p.add_argument("--output", type=Path, default=None)
    return p.parse_args()


def v05_config() -> RunOnsetConfig:
    m = json.loads((V05 / "DFL-MAT-J03WOH" / "manifest.json").read_text())["run_onset_config"]
    base = RunOnsetConfig()
    return replace(base, **{k: v for k, v in m.items() if hasattr(base, k)})


def gate(frames, fid, pid, meta, cfg, expected=None) -> str:
    """_context_candidate's checks in order; the first that fails, with its value."""
    f = frames.get(fid)
    if f is None:
        return "frame_missing"
    if pid not in f.players:
        return "player_untracked"
    if f.ball is None:
        return "ball_missing"
    if cfg.require_ball_inside_pitch and not ro.ball_is_inside_pitch(f, cfg.pitch_boundary_tolerance_m):
        return "ball_off_pitch"
    # as the pipeline: after a dismissal 21 players is the right count (adjust_for_dismissals)
    need = cfg.required_player_count
    if need is not None and expected is not None:
        need = expected(fid)
    if need is not None and len(f.players) != need:
        return f"player_count({len(f.players)} != {need})"
    cid, cd = ro._nearest_player(f)
    if cid is None or cd is None:
        return "no_carrier"
    if cd > cfg.control_distance_m:
        return f"carrier_far({cd:.2f} m > {cfg.control_distance_m})"
    if cid == pid:
        return "runner_is_carrier"
    runner = f.players[pid]
    if meta.goalkeeper_id(runner.team_id) == pid:
        return "runner_is_keeper"
    rb = math.hypot(runner.x - f.ball.x, runner.y - f.ball.y)
    if rb <= cfg.control_distance_m:
        return f"runner_on_ball({rb:.2f} m)"
    hist = int(round(cfg.controlled_history_seconds * FPS))
    stable = avail = 0
    for k in range(fid - hist, fid + 1):
        s = frames.get(k)
        if s is None or s.ball is None:
            continue
        avail += 1
        stable += ro.team_controls_ball(s, runner.team_id, cfg.control_distance_m)
    if avail < hist + 1 or stable / (hist + 1) < cfg.minimum_stable_control_fraction:
        return f"unstable_control_history({stable}/{hist + 1} < {cfg.minimum_stable_control_fraction})"
    fut = int(round(cfg.future_possession_seconds * FPS))
    known = same = avail = 0
    for k in range(fid, fid + fut + 1):
        s = frames.get(k)
        if s is None or s.ball is None:
            continue
        avail += 1
        nid, nd = ro._nearest_player(s)
        if nid is None or nd is None or nd > cfg.future_control_distance_m:
            continue
        known += 1
        same += s.players[nid].team_id == runner.team_id
    if avail < fut + 1:
        return "future_frames_missing"
    kf, sf = known / avail, (same / known if known else 0.0)
    if kf < cfg.minimum_future_known_fraction or sf < cfg.minimum_future_same_team_fraction:
        return f"future_possession(known {kf:.2f}, same team {sf:.2f})"
    return "passes_context"


def truthy(v) -> bool:
    return str(v).strip().lower() in ("true", "1", "yes")


def track(frames, pid, lo, hi):
    t = [(k, frames[k].players[pid].x, frames[k].players[pid].y)
         for k in range(lo, hi + 1) if k in frames and pid in frames[k].players]
    return [a[0] for a in t], [a[1] for a in t], [a[2] for a in t]


def stage_solver(match_id, onset, runner):
    """Where a stage-2 triple goes in the agile stage-3 inputs."""
    out = []
    for game, src in (("2v1", "passer2on1_states_agile06.json"), ("3v1", "fixedpasser_states_agile06_all.json")):
        states = json.loads((D3 / src).read_text())["states"]
        idx = [s["index"] for s in states if s["provenance"]["match_id"] == match_id
               and int(s["provenance"]["onset_frame_id"]) == onset and s["provenance"]["runner_id"] == runner]
        if not idx:
            continue
        final = json.loads((D3 / src.replace(".json", "").replace("_all", "") .__add__("_final2m.json")).read_text())
        kept = {s["index"] for s in final["states"]}
        why = {d["index"]: d["reason"] for d in final["filters"]["dropped"]}
        for i in idx:
            out.append(f"{game} state {i}: " + ("kept (passes the triangle filter up to 2 m)" if i in kept else f"dropped — {why.get(i, '?')}"))
    return out


def main() -> None:
    args = parse_args()
    cfg = v05_config()
    rows = read_xlsx(ROOT / "chani/chani_shot_annotation.xlsx")
    head = rows[0]
    ann = {r[head.index("clip_id")]: dict(zip(head, r)) for r in rows[1:] if any(r)}
    join = pd.read_csv(ROOT / "data/processed/chani_annotation_join_v0_4.csv")
    phases = pd.read_csv(V05 / "attacking_phases.csv")
    cands = pd.read_csv(V05 / "shot_context_run_onsets.csv")
    selected = pd.concat([pd.read_csv(f) for f in sorted(V05.glob("DFL-*/audit_selection.csv"))])
    gatecsv = pd.read_csv(ROOT / "data/processed/pair_gate_v7.csv")
    targets = join[join["effect"].isin(args.effects)]
    out = []
    for match_id, grp in targets.groupby("match_id"):
        files = find_bundesliga_files(args.raw_dir, match_id)
        meta = load_bundesliga_match_metadata(files["matchinfo"])
        by_shirt = {(p.team_id, str(p.shirt_number)): p for p in meta.players.values()}
        team_of = {t.name: tid for tid, t in meta.teams.items()}
        sendings = load_bundesliga_sendings_off(files["events"], load_bundesliga_frame_clock(files["positions"]))
        expected = expected_player_count_at(sendings) if sendings else None
        P = phases[phases["match_id"] == match_id]
        plan, wanted = [], set()
        for r in grp.itertuples():
            shot = int(r.shot_frame_id)
            # the SHOOTING team's phase only -- a time match alone can hit the opponent's
            shooting_team = team_of.get(str(ann[r.clip_id]["team"]).strip())
            inph = P[(P["team_id"] == shooting_team) & (P["possession_start_frame_id"] <= shot)
                     & (P["possession_end_frame_id"] + TOL >= shot)]
            if len(inph):
                lo, hi, ph = int(inph.iloc[0]["possession_start_frame_id"]), int(inph.iloc[0]["possession_end_frame_id"]), inph.iloc[0]
            else:
                lo, hi, ph = shot - int(NO_PHASE_WINDOW_S * FPS), shot, None
            wanted.update(range(lo - int(1 * FPS), hi + int(1.5 * FPS) + 1))
            plan.append((r, lo, hi, ph))
        frames = load_bundesliga_frames(files["positions"], sorted(wanted))
        print(f"\n=== {match_id} ===", flush=True)
        for r, lo, hi, ph in plan:
            rec = ann[r.clip_id]
            team = team_of.get(str(rec["team"]).strip())
            shot = int(r.shot_frame_id)
            if ph is None:
                before = P[(P["team_id"] == team) & (P["possession_end_frame_id"] < shot)]
                gap = (shot - int(before["possession_end_frame_id"].max())) / FPS if len(before) else float("nan")
                s0 = f"✗ no phase — the same team's last phase ended {gap:.1f} s before the shot (stage 1 below is tested on the {NO_PHASE_WINDOW_S:g} s window before the shot)"
            else:
                s0 = (f"✓ phase {(hi - lo) / FPS:.1f} s (starts {(shot - lo) / FPS:.1f} s before the shot, "
                      f"end reason {ph['boundary_reason']})")
            print(f"\n{r.clip_id} [{r.effect}] shooter {rec['shooter']} · off-ball {rec['offball_attackers']} · "
                  f"beneficiaries {rec['space_beneficiaries']}\n  stage 0: {s0}")
            for shirt in split_numbers(rec.get("offball_attackers", "")):
                p = by_shirt.get((team, shirt))
                if p is None:
                    print(f"  #{shirt}: shirt number not matched")
                    continue
                pid, name = p.player_id, p.short_name
                row = {"clip_id": r.clip_id, "effect": r.effect, "shirt": shirt, "player": name,
                       "stage0": "ok" if ph is not None else "no_phase"}
                C = cands[(cands["match_id"] == match_id) & (cands["player_id"] == pid)
                          & (cands["frame_id"] >= lo) & (cands["frame_id"] <= hi)]
                fs, xs, ys = track(frames, pid, lo, hi)
                kin = detect_kinematic_run_onsets(fs, xs, ys, cfg) if len(fs) > 30 else ()
                lines = []
                if len(C):
                    for c in C.itertuples():
                        sel = selected[(selected["match_id"] == match_id) & (selected["frame_id"] == c.frame_id)
                                       & (selected["player_id"] == pid)]
                        where = ("selected (in the 588)" if len(sel) else
                                 "excluded: already reviewed" if truthy(c.excluded_reviewed) else
                                 f"excluded: not settled possession ({c.settled_rejection_reasons})" if not truthy(c.settled_possession) else
                                 "excluded: not the primary candidate" if not truthy(c.primary_candidate) else "excluded: reason unknown")
                        lines.append(f"candidate found @{c.frame_id} ({(shot - c.frame_id) / FPS:.1f} s before the shot, "
                                     f"{c.onset_labels}, run speed {c.post_mean_speed_mps:.1f}) → {where}")
                        if len(sel):
                            G = gatecsv[(gatecsv["match_id"] == match_id) & (gatecsv["onset_frame_id"] == c.frame_id)
                                        & (gatecsv["runner_id"] == pid)]
                            for g in G.itertuples():
                                lines.append(f"    stage 2 defender {g.defender_name}: "
                                             + ("passed" if truthy(g.kept_final) else "dropped (protagonist rule)" if truthy(g.kept) else "dropped (pair gate)")
                                             + (f", beneficiary {g.beneficiary_name}" if truthy(g.kept_final) else ""))
                            solver = stage_solver(match_id, int(c.frame_id), pid)
                            for s in solver:
                                lines.append("    solver input: " + s)
                            if G["kept_final"].map(truthy).any() and not solver:
                                lines.append("    solver input: not in the state files (dropped at the start-state step: off the pitch or over the speed limit)")
                    row["stage1"] = "candidate"
                elif not kin:
                    unlock = {}
                    for field, values in SWEEP.items():
                        for v in values:
                            if detect_kinematic_run_onsets(fs, xs, ys, replace(cfg, **{field: v})):
                                unlock[field] = v
                                break
                    lines.append(f"✗ stage 1-a detection rule: no run detected ({len(fs)} tracked frames). "
                                 f"single relaxation that catches it: {unlock or 'none (relaxing any one of the four does not catch it)'}")
                    row["stage1"] = "not_detected"
                    row["unlock"] = json.dumps(unlock)
                else:
                    for k in kin:
                        g = gate(frames, int(k.frame_id), pid, meta, cfg, expected)
                        lines.append(f"stage 1-a detected @{k.frame_id} ({(shot - k.frame_id) / FPS:.1f} s before the shot, "
                                     f"{'/'.join(k.labels)}) → stage 1-b context gates: "
                                     + (("passed (no phase, so no candidate is built)" if ph is None
                                         else "passed — but not in the candidate list (unexplained)")
                                        if g == "passes_context" else f"✗ {g}"))
                    row["stage1"] = "context_gate"
                print(f"  #{shirt} {name}:")
                for line in lines:
                    print("    " + line)
                row["detail"] = " | ".join(lines)
                out.append(row)
        del frames
    if args.output:
        pd.DataFrame(out).to_csv(args.output, index=False)
        print(f"\n→ {args.output}")


if __name__ == "__main__":
    main()
