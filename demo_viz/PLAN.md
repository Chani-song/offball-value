# demo_viz implementation plan

Written 2026-09-21, branch `chani-fancy-viz`.

## Story to communicate

`OFF-BALL RUNNER MOVES -> DEFENDER IS PULLED -> SPACE OPENS -> BENEFICIARY GAINS`

## What the repository actually gives us (verified)

| Need | Source | Status |
| --- | --- | --- |
| 25 Hz tracking, centered metres, 105x68 | IDSSE positions XML at `~/idsse_shots/idsse-data/` | present (7 matches, ~400 MB each) |
| shirt number -> player id | `DFL_02_01_matchinformation_*.xml` via `offball_value.bundesliga` | present |
| manual runner/defender/beneficiary triplets | `shot_annotations.xlsx` (170 rows, 5 `strong`) | present |
| clip videos | `~/idsse_shots/output/ALL_SHOT_CLIPS/*.mp4` | present |
| 5 Hz normalized cross-check cache | `~/idsse_shots/output/tracking_overlay_cache/` | present |
| run-onset detection | `offball_value.run_onset.detect_kinematic_run_onsets` | implemented, pure fn |
| player influence surface | `offball_value.fernandez_influence` (Fernandez & Bornn 2018) | implemented |
| goal-weighted residual space value | `offball_value.goal_weighted_influence.target_residual_influence` | implemented |
| velocity estimates | `offball_value.pass_dynamics.estimate_frame_velocities` | implemented |
| calibrated xT / pass probability / learned counterfactual | - | NOT available -> renderer hooks only |

## Layers

1. `sources/` data discovery + fast IDSSE window extractor (line-streaming, ~1 s/match) + disk cache.
2. `scene.py` canonical `Scene` object, format-agnostic.
3. `annotations.py` Excel adapter -> role triplets; `sources/pipeline.py` adapter stub for predicted triplets.
4. `quantities.py` thin wrapper over the repo's *existing* science. Every field carries a `grounded` flag.
5. `render/` pitch + layers (trail, tether, wake, ghost, reveal) + figure composition.
6. `story.py` beat timeline; `animate.py` driver + mp4/gif/png export; `interactive.py` Plotly HTML.
7. CLIs: `render_scene.py`, `view_scene.py`, `export_preview.py`.

## Scientific boundary (enforced in code)

- Grounded: tracking, roles, run-onset frames, Fernandez influence, goal-weighted residual
  space value, and the frozen-defender counterfactual difference computed with the same repo formula.
- Illustrative only (labelled in-frame): geometric space-wake shading when surfaces are disabled,
  and the frozen ghost defender (a held-position device, not a learned defensive best response).
- Never rendered: xT, pass/dribble probability, calibrated threat. Hooks exist; they stay empty.

## Palette (checked with an OKLab + Vienot CVD script, see README)

runner `#FF3E9D` / defender `#FFA62B` / beneficiary `#26D9F2`,
attack neutral `#D9CFB8`, defend neutral `#5F7392`, pitch `#132019`.
Roles always carry >=2 redundant encodings (hue + halo ring + size + direct label).
