#!/usr/bin/env python3
"""Export browser-ready scene data for the static demo.

    python -m demo_viz.web.export_data              # strong + medium
    python -m demo_viz.web.export_data --effects strong
    python -m demo_viz.web.export_data --limit 3 --out /tmp/check

Writes one compact JSON per scene plus an ``index.json``, into
``demo_viz/web_data/``. Only the fields the browser actually needs are written:
no local paths, no usernames, no cache locations, no raw source data.

Position tracks are the numbers the IDSSE feed itself carries (two decimals, so
metres to the centimetre), which is what lets the browser reproduce the Python
residual-space values exactly rather than approximately.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    __package__ = "demo_viz.web"

from ..annotations import select_clips
from ..core.role_logic import runner_onset_index
from ..loader import load_scene
from ..scene import Scene
from ..sources.window_scene import EFFECT, find_spec, pipeline_scene_ids

WEB_DATA = Path(__file__).resolve().parent.parent / "web_data"

#: Everything the browser needs to reproduce the Python numbers. Keep this list
#: short on purpose: it is the whole public surface of the exported data.
SCENE_FIELDS = (
    "scene_id", "effect", "shape", "title", "subtitle", "notes", "match_clock",
    "fps", "n_frames", "t0", "pitch", "attacking_team", "defending_team",
    "attacking_direction", "players", "ball", "roles", "onsets",
)


def _round_track(values: np.ndarray, places: int = 2) -> list:
    """Track as a plain list, with NaN written as null."""

    out = []
    for value in np.asarray(values, dtype=float):
        out.append(None if not np.isfinite(value) else round(float(value), places))
    return out


def scene_payload(scene: Scene, clip) -> dict:
    players = []
    for player in scene.players.values():
        players.append({
            "id": player.player_id,
            "shirt": str(player.label),
            "name": player.name,
            "side": player.side,
            "gk": bool(player.is_goalkeeper),
            "x": _round_track(player.xy[:, 0]),
            "y": _round_track(player.xy[:, 1]),
        })
    players.sort(key=lambda p: (p["side"], int(p["shirt"]) if p["shirt"].isdigit() else 99))

    # The browser needs a run start for whichever runner the visitor picks, and
    # porting the whole onset detector would be a lot of JavaScript for no gain.
    onsets = {}
    for player in scene.players.values():
        if player.side != "attack" or player.is_goalkeeper:
            continue
        index, method = runner_onset_index(scene, player.player_id)
        onsets[player.player_id] = {"index": int(index), "method": method}

    return {
        "scene_id": scene.scene_id,
        "effect": (clip.effect if clip else ""),
        "shape": (clip.shape if clip else ""),
        "title": scene.title,
        "subtitle": scene.subtitle,
        "notes": scene.notes or "",
        "match_clock": (clip.match_clock if clip else ""),
        "fps": float(scene.fps),
        "n_frames": int(scene.n_frames),
        "t0": round(float(scene.times[0]), 3),
        "pitch": [scene.pitch_length, scene.pitch_width],
        "attacking_team": scene.attacking_team_name,
        "defending_team": scene.defending_team_name,
        "attacking_direction": int(scene.attacking_direction),
        "players": players,
        "ball": {"x": _round_track(scene.ball_xy[:, 0]),
                 "y": _round_track(scene.ball_xy[:, 1])},
        "roles": {
            "runner": list(scene.runner_ids),
            "defender": list(scene.defender_ids),
            "beneficiary": list(scene.beneficiary_ids),
        },
        "onsets": onsets,
    }


class _PipelineClip:
    """What scene_payload reads from an annotated clip, for a scene that has none."""

    def __init__(self, effect: str, shape: str, match_clock: str):
        self.effect, self.shape, self.match_clock = effect, shape, match_clock


def slug(scene_id: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in scene_id)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--effects", nargs="*", default=["strong", "medium"])
    parser.add_argument("--out", type=Path, default=WEB_DATA)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--indent", type=int, default=None,
                        help="pretty-print (bigger files); default is compact")
    args = parser.parse_args(argv)

    clips = select_clips(tuple(args.effects))
    if args.limit:
        clips = clips[: args.limit]
    args.out.mkdir(parents=True, exist_ok=True)

    index = []
    total = 0
    for position, clip in enumerate(clips, start=1):
        try:
            scene = load_scene(clip.clip_id, quantities=False, surfaces=False)
        except Exception as error:
            print(f"!! {clip.clip_id}: {error}", file=sys.stderr)
            continue
        payload = scene_payload(scene, clip)
        unexpected = set(payload) - set(SCENE_FIELDS)
        if unexpected:
            raise RuntimeError(f"refusing to export unexpected fields: {sorted(unexpected)}")
        name = f"{slug(clip.clip_id)}.json"
        path = args.out / name
        path.write_text(json.dumps(payload, ensure_ascii=False,
                                   separators=(",", ":"), indent=args.indent))
        size = path.stat().st_size
        total += size
        index.append({
            "scene_id": clip.clip_id,
            "effect": clip.effect,
            "shape": clip.shape,
            "file": name,
            "title": scene.title,
            "match_clock": clip.match_clock,
            "kb": round(size / 1024, 1),
        })
        print(f"[{position:2d}/{len(clips)}] {clip.clip_id:<30} {size / 1024:6.1f} KB")

    # the curated scenes the pipeline found (data/pipeline_scenes.json), after the annotated ones
    for scene_id in pipeline_scene_ids():
        spec = find_spec(scene_id)
        try:
            scene = load_scene(scene_id, quantities=False, surfaces=False)
        except Exception as error:
            print(f"!! {scene_id}: {error}", file=sys.stderr)
            continue
        clip = _PipelineClip(EFFECT, "1R-1D-1B", spec["match_clock"])
        payload = scene_payload(scene, clip)
        name = f"{slug(scene_id)}.json"
        path = args.out / name
        path.write_text(json.dumps(payload, ensure_ascii=False,
                                   separators=(",", ":"), indent=args.indent))
        size = path.stat().st_size
        total += size
        index.append({"scene_id": scene_id, "effect": clip.effect, "shape": clip.shape, "file": name,
                      "title": scene.title, "match_clock": clip.match_clock, "kb": round(size / 1024, 1)})
        print(f"[pipeline] {scene_id:<30} {size / 1024:6.1f} KB")

    order = {"strong": 0, "medium": 1, "low": 2}
    index.sort(key=lambda row: (order.get(row["effect"], 9), row["scene_id"]))
    (args.out / "index.json").write_text(
        json.dumps({"scenes": index}, ensure_ascii=False, separators=(",", ":"))
    )
    print(f"\n{len(index)} scenes · {total / 1024 / 1024:.2f} MB total "
          f"· {total / max(len(index), 1) / 1024:.0f} KB average")
    print(f"index: {args.out / 'index.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
