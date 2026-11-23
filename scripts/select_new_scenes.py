#!/usr/bin/env python3
"""Which scenes of a new onset pool are not in an old one? Write them as chunks.

A scene is the same scene when match, runner and onset frame agree within the
extraction's own dedupe window (3 s), so a relaxed gate that moves an onset by
a few frames does not count as a new scene. New scenes are written in the
chunk format the stage-2 renderer reads (a flat list of scene payloads), with
a manifest giving scene_count / chunk_size / chunk_count for a Slurm array.

--min-run-speed is the pre-filter before stage 2: the stage-3 input cut drops
every run below 3.8 m/s, and run speed is known from stage 1, so a scene below
the cut is a build whose result is thrown away. Set it at or below the lowest
run-speed cut you may use later, or the dropped scenes have to be built then.
--mark-run-speed only counts how many kept scenes reach a stricter cut; the
per-state mark is made at stage 3 (filter_stage3_states.py --mark-run-speed).

Usage:
    python scripts/merge_full_extraction.py --root data/processed/run_onset_v0_6/all3 \
        --version run_onset_v0_6_all3
    python scripts/select_new_scenes.py --new data/processed/run_onset_v0_6/all3 \
        --old data/processed/run_onset_v0_5/dir25 --output data/processed/run_onset_v0_6/all3_new \
        --min-run-speed 3.5 --mark-run-speed 3.8
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

FPS = 25
WINDOW_S = 3.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--new", type=Path, required=True)
    p.add_argument("--old", type=Path, nargs="+", required=True,
                   help="pools already built; a scene in any of them is not new")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--chunk-size", type=int, default=30)
    p.add_argument("--near-shots", type=Path, default=None,
                   help="keep only scenes within --shot-window seconds before a shot in this CSV "
                        "(data/processed/chani_annotation_join_v0_4.csv, effect strong/medium): the scenes the "
                        "relaxation was made for, before spending a build on the whole pool")
    p.add_argument("--shot-window", type=float, default=12.0,
                   help="seconds before the shot (the window trace_chani_scenes.py uses; my choice)")
    p.add_argument("--min-run-speed", type=float, default=None,
                   help="keep only runs whose post_speed_mps reaches this (the stage-3 input cut is 3.8): "
                        "runs below it never reach the solver, so building them is wasted")
    p.add_argument("--mark-run-speed", type=float, default=None,
                   help="count in the manifest how many kept scenes reach this stricter run speed "
                        "(nothing is dropped for it)")
    return p.parse_args()


def scenes(root: Path) -> list[dict]:
    manifest = json.loads((root / "manifest.json").read_text())
    out = []
    for k in range(int(manifest["chunk_count"])):
        out.extend(json.loads((root / f"chunk{k}.json").read_text()))
    return out


def select_fresh(
    new: Sequence[dict],
    old: Sequence[dict],
    *,
    window_frames: int = int(WINDOW_S * FPS),
    near_shot_frames: Mapping[str, Sequence[int]] | None = None,
    shot_window_frames: int = 0,
    min_run_speed: float | None = None,
) -> tuple[list[dict], dict[str, int]]:
    """Scenes of `new` with no `old` onset by the same runner within the window,
    then the optional near-shot and run-speed cuts. Returns the scenes and how
    many each cut dropped."""
    seen: dict[tuple[str, str], list[int]] = {}
    for s in old:
        seen.setdefault((s["match_id"], s["runner_id"]), []).append(int(s["onset_frame_id"]))
    fresh = [s for s in new
             if not any(abs(int(s["onset_frame_id"]) - f) <= window_frames
                        for f in seen.get((s["match_id"], s["runner_id"]), []))]
    dropped = {"not_near_shots": 0, "slow": 0}
    if near_shot_frames is not None:
        keep = [s for s in fresh if any(0 < f - int(s["onset_frame_id"]) <= shot_window_frames
                                        for f in near_shot_frames.get(s["match_id"], []))]
        dropped["not_near_shots"], fresh = len(fresh) - len(keep), keep
    if min_run_speed is not None:
        keep = [s for s in fresh if float(s.get("post_speed_mps", 0.0)) >= min_run_speed]
        dropped["slow"], fresh = len(fresh) - len(keep), keep
    return fresh, dropped


def main() -> None:
    args = parse_args()
    old = [s for root in args.old for s in scenes(root)]
    new = scenes(args.new)
    near = None
    if args.near_shots is not None:
        import pandas as pd
        shots = pd.read_csv(args.near_shots)
        shots = shots[shots["effect"].isin(["strong", "medium"])]
        near = {m: g["shot_frame_id"].astype(int).tolist() for m, g in shots.groupby("match_id")}
    fresh, dropped = select_fresh(new, old, near_shot_frames=near,
                                  shot_window_frames=int(args.shot_window * FPS),
                                  min_run_speed=args.min_run_speed)
    in_flight = sum(1 for s in fresh if s.get("ball_in_flight"))
    marked = (None if args.mark_run_speed is None
              else sum(1 for s in fresh if float(s.get("post_speed_mps", 0.0)) >= args.mark_run_speed))
    args.output.mkdir(parents=True, exist_ok=True)
    for k in range(0, len(fresh), args.chunk_size):
        (args.output / f"chunk{k // args.chunk_size}.json").write_text(json.dumps(fresh[k:k + args.chunk_size]))
    chunk_count = (len(fresh) + args.chunk_size - 1) // args.chunk_size
    (args.output / "manifest.json").write_text(json.dumps({
        "version": "run_onset_v0_6_new_only", "derived_from": str(args.new),
        "compared_to": str(args.old[0]) if len(args.old) == 1 else [str(root) for root in args.old],
        "same_scene_window_seconds": WINDOW_S, "min_run_speed": args.min_run_speed,
        "dropped_slow": dropped["slow"],
        "run_speed_mark": args.mark_run_speed, "scenes_meeting_mark": marked,
        "near_shots": str(args.near_shots) if args.near_shots else None, "shot_window_s": args.shot_window,
        "dropped_not_near_shots": dropped["not_near_shots"],
        "scene_count": len(fresh), "chunk_size": args.chunk_size,
        "chunk_count": chunk_count, "ball_in_flight_scenes": in_flight}, indent=2))
    print(f"new pool {len(new)} · old pool {len(old)} · new scenes {len(fresh)} (ball in flight {in_flight}"
          + (f", below run speed {args.min_run_speed} dropped {dropped['slow']}"
             if args.min_run_speed is not None else "")
          + (f", {marked} reach run speed {args.mark_run_speed}" if marked is not None else "")
          + (f", outside {args.shot_window:g} s before a Chani shot dropped {dropped['not_near_shots']}"
             if args.near_shots else "")
          + f") → {args.output} ({chunk_count} chunks)")


if __name__ == "__main__":
    main()
