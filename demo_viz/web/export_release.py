#!/usr/bin/env python3
"""Precompute the solver's release quantities for the exploratory pass targets.

    python -m demo_viz.web.export_release                  # strong + medium
    python -m demo_viz.web.export_release --limit 2
    python -m demo_viz.web.export_release --stride 5

Writes ``demo_viz/web_data/release/<scene>.json``, lazy-loaded when a visitor
opens the pass-model explorer. Each record is what the research code returned
for one (frame, receiver, exploratory target):

    legal, offside, completion proxy, positional threat, release payoff,
    and the seven model features the panel shows.

WHY PRECOMPUTED
---------------
The chain is Andrew's 33-feature logistic plus Kyuhyeok's threat and offside
rule. Porting it to JavaScript would mean reimplementing segment-clamped lane
distance, lexsort tie-breaking, a 180-degree rotation and two pressure length
scales -- four places a silent error would never show on screen. The browser is
handed numbers this repository's Python produced instead.

WHAT THE TARGETS ARE, AND ARE NOT
---------------------------------
The five targets per frame are the demo's own geometry: a 90-degree sector from
the carrier, five 25 m rays, clipped to the pitch. **They are not the solver's
action space.** The solver's own 18 releases are built from the receiver
(``receiver + direction * offset``) and belong to states this demo does not
publish. Feeding our rays into the real model is an exploration of the model's
response surface; the values are real, the targets are ours.

OPTIONAL
--------
Needs Andrew's ``defensive_positioning`` and the fitted
``experimental_pass.json``, neither of which is committed. Without them this
exits cleanly and the site builds with no pass-model payload.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    __package__ = "demo_viz.web"

from ..annotations import select_clips  # noqa: E402
from ..core.candidates import CANDIDATE_ANGLES_DEG, CANDIDATE_DISTANCE_M, candidate_passes  # noqa: E402
from ..loader import load_scene  # noqa: E402
from ..scene import Scene  # noqa: E402
from ..solver_native import PANEL_FEATURES, availability, release_quantities  # noqa: E402
from .export_data import WEB_DATA, slug  # noqa: E402

RELEASE_DIR = WEB_DATA / "release"

#: Frames between samples. The fan only exists on frames with a clear carrier,
#: and the quantities move smoothly between them.
DEFAULT_STRIDE = 5

#: Rounding. The panel shows three or four figures; storing more is payload.
VALUE_PLACES = 5
FEATURE_PLACES = 3


def _defending_side(scene: Scene, index: int) -> list[list[float]]:
    """Every defending player on the pitch at this frame.

    The whole side, deliberately: the model picks its own two defender roles
    from what it is handed, and Kyuhyeok's wrapper hands it the controlled
    defender plus the background. Pre-selecting would reproduce Andrew's
    reduced 2v1 convention, which is not what the pipeline ran.
    """

    out = []
    for player in scene.players.values():
        if player.side != "defend":
            continue
        xy = player.xy[index]
        if np.all(np.isfinite(xy)):
            out.append([float(xy[0]), float(xy[1])])
    return out


def scene_payload(scene: Scene, stride: int) -> dict:
    frames = []
    feature_keys = [key for key, _, _ in PANEL_FEATURES]
    attackers = [p for p in scene.players.values()
                 if p.side == "attack" and not p.is_goalkeeper]

    for index in range(0, scene.n_frames, max(1, int(stride))):
        fan = candidate_passes(scene, index)
        if fan is None:
            continue
        defenders = _defending_side(scene, index)
        if not defenders:
            continue
        carrier_xy = fan.origin
        receivers = {}
        for receiver in attackers:
            if receiver.player_id == fan.carrier_id:
                continue                      # the carrier cannot receive his own pass
            xy = receiver.xy[index]
            if not np.all(np.isfinite(xy)):
                continue
            rows = []
            for _, target in fan.rays:
                result = release_quantities(carrier_xy, (float(xy[0]), float(xy[1])),
                                            target, defenders,
                                            int(scene.attacking_direction))
                if result is None:
                    return {}
                rows.append([
                    int(result["legal"]), int(result["offside"]),
                    round(result["completion_proxy"], VALUE_PLACES),
                    round(result["positional_threat"], VALUE_PLACES),
                    round(result["release_payoff"], VALUE_PLACES),
                    *[round(result["features"][k], FEATURE_PLACES) for k in feature_keys],
                ])
            receivers[receiver.player_id] = rows
        frames.append({"index": int(index), "carrier": fan.carrier_id,
                       "receivers": receivers})

    return {
        "scene_id": scene.scene_id,
        "quantity": "solver release",
        "method": "legal x completion_proxy x positional_threat "
                  "(release_payoffs_with_background)",
        "targets": "exploratory: demo geometry, not the solver's action space",
        "ray_angles_deg": list(CANDIDATE_ANGLES_DEG),
        "ray_distance_m": CANDIDATE_DISTANCE_M,
        "columns": ["legal", "offside", "completion_proxy", "positional_threat",
                    "release_payoff", *feature_keys],
        "feature_labels": {key: label for key, label, _ in PANEL_FEATURES},
        "stride": int(stride),
        "frames": frames,
    }


def model_card() -> dict:
    """What the artifact says about itself, for the Source panel."""

    from ..solver_native.release import model_path

    raw = json.loads(model_path().read_text())
    meta = raw.get("metadata", {})
    return {
        "schema": raw.get("schema"),
        "kind": meta.get("kind"),
        "target_semantics": meta.get("target_semantics"),
        "validation_status": meta.get("validation_status"),
        "feature_set": meta.get("feature_set"),
        "n_features": len(raw.get("features", ())),
        "training_rows": meta.get("training_rows"),
        "test_rows": meta.get("test_rows"),
        "note": "Experimental proxy. ExpectedPass refuses to load it without "
                "allow_proxy; the solver CLI passes --allow-proxy-labels.",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--effects", nargs="*", default=["strong", "medium"])
    parser.add_argument("--out", type=Path, default=RELEASE_DIR)
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    state = availability()
    if not state.ok:
        print("solver pass model unavailable; skipping release export", file=sys.stderr)
        print(f"  {state.reason}", file=sys.stderr)
        print("  the site still builds; the pass-model explorer stays disabled",
              file=sys.stderr)
        return 0                                  # not a failure: this is optional

    clips = select_clips(tuple(args.effects))
    if args.limit:
        clips = clips[: args.limit]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "model.json").write_text(json.dumps(model_card(), indent=2))

    started, total = time.time(), 0
    for position, clip in enumerate(clips, start=1):
        try:
            scene = load_scene(clip.clip_id, quantities=False, surfaces=False)
        except Exception as error:                        # pragma: no cover
            print(f"!! {clip.clip_id}: {error}", file=sys.stderr)
            continue
        payload = scene_payload(scene, args.stride)
        if not payload:
            continue
        path = args.out / f"{slug(clip.clip_id)}.json"
        path.write_text(json.dumps(payload, separators=(",", ":")))
        total += path.stat().st_size
        print(f"[{position}/{len(clips)}] {clip.clip_id}  "
              f"{len(payload['frames'])} frames  {path.stat().st_size / 1024:.0f} KB",
              flush=True)

    print(f"\n{len(clips)} scenes, {total / 1024 / 1024:.2f} MB total, "
          f"{total / max(len(clips), 1) / 1024:.0f} KB each (lazy)")
    print(f"done in {time.time() - started:.0f}s -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
