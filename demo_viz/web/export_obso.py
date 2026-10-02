#!/usr/bin/env python3
"""Precompute the OBSO threat surface for the browser demo.

    python -m demo_viz.web.export_obso                 # strong + medium
    python -m demo_viz.web.export_obso --limit 2
    python -m demo_viz.web.export_obso --stride 10     # coarser sampling

Writes one file per scene into ``demo_viz/web_data/obso/`` plus a shared
``score_grid.json``, loaded by the browser only when a visitor first selects
the OBSO threat view. The base scene JSON is untouched, so the initial page
load does not grow.

WHAT THE SURFACE IS
-------------------
``offball_value.reference_obso.evaluate_reference_obso`` on a 50 x 32 grid:

    OBSO = pitch control  x  ball transition  x  EPV score

    pitch control    Spearman-style control of the cell by the attacking team,
                     with offside applied.
    ball transition  a Gaussian in distance from the ball -- where the next
                     ball actually goes.
    EPV score        the static PAUSA grid (``data/static/EPV_grid.csv``).

The EPV term is position-only: a function of pitch coordinates alone, with no
representation of the defensive line, so a cell three metres behind the last
defender scores the same as one three metres in front of him. Exporting the
bare grid as "threat" would be misleading; the product is not, because pitch
control does move when a defender is beaten. That caveat belongs in the Source
panel, and it is why this is labelled OBSO threat and not xT.

WHAT IS EXPORTED, AND WHY ONLY PART OF IT
-----------------------------------------
Only **pitch control** is sampled and stored. The browser multiplies it by the
other two terms, which it computes exactly at every frame:

    transition  is a Gaussian on the ball -- a dozen lines of arithmetic, and
                the ball moves fast enough that interpolating it would smear
                the brightest moment of a clip.
    score       is one static grid, shipped once for all 45 scenes rather than
                baked into each of them.

Control is the expensive term (numerical integration, ~260 ms a frame) and the
slow-moving one, so it is the right thing to sample. It is also independent of
which runner, defender and beneficiary the visitor has picked -- it is a
property of the frame and the two teams -- which is what makes precomputing it
possible at all.

SAMPLING ERROR
--------------
Measured on J03WOH:shot_006_P1_1054 against an every-frame computation, as a
percentage of the scene's peak OBSO:

    stride  5 (0.20 s)   max 5.8 %   mean 0.01 %
    stride 10 (0.40 s)   max 8.4 %   mean 0.02 %

with one exception that no stride fixes. ``apply_offside=True`` makes an
attacker's contribution appear or vanish the instant he crosses the line, and
interpolating across that step reaches 73 % of peak. It is a real
discontinuity, not undersampling, so the sample list carries **both frames
either side of every offside change** and the ramp is one frame wide. The
numbers above are the error away from those steps.

Quantisation to one byte of full scale costs a further 0.2 % of peak.

``reference_obso.py`` is byte-identical to origin/kyuhyeok-dev @ 3c9965b, so it
can still be diffed against that branch.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    __package__ = "demo_viz.web"

from offball_value.reference_obso import (  # noqa: E402
    ReferenceOBSOConfig,
    evaluate_reference_obso,
    offside_attacker_ids,
    score_surface,
)

from ..annotations import select_clips  # noqa: E402
from ..loader import load_scene  # noqa: E402
from ..quantities import _frame_at, _velocities  # noqa: E402
from ..scene import Scene  # noqa: E402
from .export_data import WEB_DATA, slug  # noqa: E402

OBSO_DIR = WEB_DATA / "obso"

#: Frames between samples; the browser interpolates linearly in between.
#: 0.2 s at 25 fps, which the header's measurement puts at 5.8 % of peak worst
#: case and 0.01 % on average.
DEFAULT_STRIDE = 5

#: Pitch control is a probability, so full scale is 1.0 and one byte is
#: 0.2 % of it -- finer than the 220 alpha levels the surface is drawn with.
QUANT_LEVELS = 255


def offside_change_frames(scene: Scene, config: ReferenceOBSOConfig) -> list[int]:
    """Frames where the set of offside attackers changes.

    Cheap to evaluate (it only sorts the defenders), and worth evaluating
    every frame: control steps here, and no sampling rate smooths a step.
    """

    previous: set[str] | None = None
    changes: list[int] = []
    for index in range(scene.n_frames):
        ball = scene.ball_xy[index]
        if not np.all(np.isfinite(ball)):
            continue
        current = offside_attacker_ids(
            _frame_at(scene, index),
            scene.attacking_team_id,
            int(scene.attacking_direction),
            (float(ball[0]), float(ball[1])),
            config.offside_tolerance_m,
        )
        if previous is not None and current != previous:
            changes.append(index)
        previous = current
    return changes


def sample_indices(scene: Scene, stride: int, config: ReferenceOBSOConfig) -> list[int]:
    """Regular samples, plus both frames either side of every offside step."""

    wanted = set(range(0, scene.n_frames, max(1, int(stride))))
    wanted.add(scene.n_frames - 1)
    for frame in offside_change_frames(scene, config):
        wanted.add(max(0, frame - 1))
        wanted.add(frame)
    return sorted(wanted)


def control_series(scene: Scene, stride: int = DEFAULT_STRIDE,
                   config: ReferenceOBSOConfig | None = None):
    """``(indices, stack)`` of pitch-control surfaces, shaped (n, ny, nx)."""

    config = config or ReferenceOBSOConfig()
    goalkeepers = tuple(p.player_id for p in scene.players.values() if p.is_goalkeeper)
    indices = sample_indices(scene, stride, config)

    surfaces = []
    for index in indices:
        frame = _frame_at(scene, index)
        if frame.ball is None:
            surfaces.append(np.zeros((config.grid_cells_y, config.grid_cells_x)))
            continue
        surface = evaluate_reference_obso(
            frame,
            attacking_team_id=scene.attacking_team_id,
            attacking_direction=int(scene.attacking_direction),
            velocities=_velocities(scene, index),
            goalkeeper_ids=goalkeepers,
            apply_offside=True,
            config=config,
        )
        surfaces.append(np.asarray(surface.pitch_control, dtype=float))
    return indices, np.stack(surfaces, axis=0)


def quantise(stack: np.ndarray) -> bytes:
    """One byte per cell of a [0, 1] field."""

    clipped = np.clip(np.nan_to_num(stack, nan=0.0), 0.0, 1.0)
    return np.rint(clipped * QUANT_LEVELS).astype(np.uint8).tobytes()


def scene_payload(scene: Scene, stride: int, config: ReferenceOBSOConfig) -> dict:
    indices, stack = control_series(scene, stride, config)
    return {
        "scene_id": scene.scene_id,
        "quantity": "pitch_control",
        "method": "OBSO = pitch control x ball transition x EPV score (repo v0.1)",
        "note": "the browser multiplies this by transition and score per frame",
        "grid": {"nx": int(config.grid_cells_x), "ny": int(config.grid_cells_y),
                 "length": float(config.field_length_m),
                 "width": float(config.field_width_m)},
        "transition_sigma_m": float(config.transition_sigma_m),
        "attacking_direction": int(scene.attacking_direction),
        "indices": [int(i) for i in indices],
        "n_frames": int(scene.n_frames),
        "levels": QUANT_LEVELS,
        "data": base64.b64encode(quantise(stack)).decode("ascii"),
    }


def score_payload(config: ReferenceOBSOConfig) -> dict:
    """The static EPV grid, once, for the attack-to-the-right orientation.

    The browser mirrors it when a scene attacks the other way, exactly as
    ``score_surface`` does.
    """

    score = score_surface(1, config)
    return {
        "quantity": "epv_score",
        "source": "PAUSA EPV_grid.csv, peak-normalised (Apache-2.0)",
        "caveat": "position-only; does not encode the defensive line",
        "grid": {"nx": int(config.grid_cells_x), "ny": int(config.grid_cells_y)},
        "values": [round(float(v), 6) for v in np.asarray(score).ravel()],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--effects", nargs="*", default=["strong", "medium"])
    parser.add_argument("--out", type=Path, default=OBSO_DIR)
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    clips = select_clips(tuple(args.effects))
    if args.limit:
        clips = clips[: args.limit]
    args.out.mkdir(parents=True, exist_ok=True)
    config = ReferenceOBSOConfig()

    grid_path = args.out / "score_grid.json"
    grid_path.write_text(json.dumps(score_payload(config), separators=(",", ":")))
    print(f"score grid  {grid_path.stat().st_size / 1024:.0f} KB  (shared by every scene)")

    started = time.time()
    total = 0
    for position, clip in enumerate(clips, start=1):
        try:
            scene = load_scene(clip.clip_id, quantities=False, surfaces=False)
        except Exception as error:                        # pragma: no cover
            print(f"!! {clip.clip_id}: {error}", file=sys.stderr)
            continue
        payload = scene_payload(scene, args.stride, config)
        path = args.out / f"{slug(clip.clip_id)}.json"
        path.write_text(json.dumps(payload, separators=(",", ":")))
        size = path.stat().st_size
        total += size
        print(f"[{position}/{len(clips)}] {clip.clip_id}  "
              f"{len(payload['indices'])} samples  {size / 1024:.0f} KB", flush=True)

    print(f"\n{len(clips)} scenes, {total / 1024 / 1024:.2f} MB total, "
          f"{total / max(len(clips), 1) / 1024:.0f} KB each (lazy, one at a time)")
    print(f"done in {time.time() - started:.0f}s -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
