#!/usr/bin/env python3
"""Tracking evidence for the showcase's "Compare players" view, per scene.

The Dilemma view lets a reader put another defender beside the one the
reviewer picked and see how the tracking reads for him. Two of the three
quantities it shows come from repository modules that the browser does not
implement, so they are computed here, once, and read there:

    marking distance   offball_value.dynamic_marking.marking_sample ->
                       weighted_error_m = |defender - goal-side target|
                       + 2.0 * (wrong-side displacement), in metres, over the
                       window from the run's onset. Lower = marking tighter.
    reaction time      demo_viz.core.role_logic.defender_reaction_index: the
                       repository's kinematic onset detector where it fires,
                       otherwise a sustained-pursuit rule, reported as seconds
                       after the run starts. It says which rule fired.

The third, space created, is the influence cache's own number and the browser
computes it exactly (tests/test_demo_viz_app.py pins the cache against
offball_value.goal_weighted_influence), so it is not duplicated here.

Nothing in this file is an extraction criterion. The showcase's triplets were
picked by hand (upstream build_showcase_states.py), so this is evidence about
a play, not a record of how it was selected.

Usage:
    python -m demo_viz.web.export_compare [--out demo_viz/web_data/compare]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):                      # python demo_viz/web/export_compare.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "demo_viz.web"

from ..annotations import select_clips
from ..core.role_logic import defender_reaction_index, runner_onset_index
from ..loader import load_scene
from ..quantities import marking_series
from ..scene import Scene
from .export_data import WEB_DATA, slug

#: A defender this far from the runner, on average, is not in the duel at all.
#: Only a presentational cut: the browser lists the nearest handful, and a
#: 60 m row would be noise in a three-line panel.
MAX_MEAN_DISTANCE_M = 40.0


def _runner_of(scene: Scene) -> str | None:
    """The annotated runner, which is the one the comparison is about."""

    for player_id in scene.runner_ids:
        if player_id in scene.players:
            return player_id
    return None


def defender_rows(scene: Scene, runner_id: str) -> list[dict]:
    """Every outfield defender's marking distance and reaction, for one runner."""

    onset_index, onset_method = runner_onset_index(scene, runner_id)
    rows = []
    for player in scene.players_on("defend"):
        if player.is_goalkeeper:
            continue
        try:
            series = marking_series(scene, runner_id, player.player_id)
        except KeyError:
            continue
        error = series["weighted_error_m"][onset_index:]
        distance = series["actor_defender_distance_m"][onset_index:]
        if not np.isfinite(error).any() or not np.isfinite(distance).any():
            continue
        mean_distance = float(np.nanmean(distance))
        if not np.isfinite(mean_distance) or mean_distance > MAX_MEAN_DISTANCE_M:
            continue
        index, rule = defender_reaction_index(scene, runner_id, player.player_id,
                                              from_index=onset_index)
        rows.append({
            "id": player.player_id,
            "marking_distance_m": round(float(np.nanmean(error)), 2),
            "marking_distance_best_m": round(float(np.nanmin(error)), 2),
            "mean_distance_m": round(mean_distance, 2),
            "reaction_s": (None if index is None
                           else round(float(scene.times[index] - scene.times[onset_index]), 2)),
            "reaction_rule": rule or "",
        })
    rows.sort(key=lambda row: row["marking_distance_m"])
    return rows


def scene_payload(scene: Scene) -> dict | None:
    runner_id = _runner_of(scene)
    if runner_id is None:
        return None
    onset_index, onset_method = runner_onset_index(scene, runner_id)
    return {
        "schema": "compare/1",
        "scene_id": scene.scene_id,
        "runner": runner_id,
        "run_onset": {"index": int(onset_index), "method": onset_method,
                      "time_s": round(float(scene.times[onset_index]), 2)},
        "definitions": {
            "marking_distance_m": {
                "source": "offball_value.dynamic_marking.marking_sample",
                "formula": "|defender - goal-side target| + 2.0 x wrong-side displacement",
                "window": "mean over the frames from the run's onset",
                "unit": "m", "better": "lower",
            },
            "reaction_s": {
                "source": "demo_viz.core.role_logic.defender_reaction_index",
                "formula": "the repository's kinematic onset detector, else speed >= 1.5 m/s "
                           "and pursuit alignment >= 0.30 sustained for 0.40 s",
                "unit": "s after the run starts", "better": "lower",
            },
        },
        "defenders": defender_rows(scene, runner_id),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--effects", nargs="*", default=["strong", "medium"])
    parser.add_argument("--out", type=Path, default=WEB_DATA / "compare")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    clips = select_clips(tuple(args.effects))
    if args.limit:
        clips = clips[: args.limit]
    args.out.mkdir(parents=True, exist_ok=True)

    total = written = 0
    for position, clip in enumerate(clips, start=1):
        try:
            scene = load_scene(clip.clip_id, quantities=False, surfaces=False)
        except Exception as error:                              # noqa: BLE001
            print(f"!! {clip.clip_id}: {error}", file=sys.stderr)
            continue
        payload = scene_payload(scene)
        if payload is None:
            print(f"[{position:2d}/{len(clips)}] {clip.clip_id:<30} no annotated runner")
            continue
        path = args.out / f"{slug(clip.clip_id)}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
        total += path.stat().st_size
        written += 1
        print(f"[{position:2d}/{len(clips)}] {clip.clip_id:<30} "
              f"{len(payload['defenders']):2d} defenders · {path.stat().st_size / 1024:.1f} KB")

    print(f"\n{written} files, {total / 1024:.0f} KB total, "
          f"{total / max(written, 1) / 1024:.1f} KB each (lazy) -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
