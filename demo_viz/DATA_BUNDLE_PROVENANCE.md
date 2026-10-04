# The local data bundle

`offball_demo_data_20260930`, installed at `local_inputs/offball_demo_data_20260930/`.

Produced by `origin/kyuhyeok-dev` @
`8a5c69d0bb1c42948c3f9d50becd2ffc41e72344` — `scripts/analyze_eval.py` and
`scripts/extract_panel_policy.py`, over the run in `jobs/solve_evaluation.sbatch`.

---

## 1. What it contains

26 scenes, **69 real moments** (each scene's 0.0 / 0.6 / 1.2 s), each solved as
its own game.

| | |
| --- | --- |
| `panels/eval-<code>[@0.6\|@1.2].json` | 69 files. Three bodies × five commands, each with `prob`, the solver's own **0.6 s `path`** (25 points), `end`, `aim`; plus `release`, `passes`, `slot_values`, `value`, `provenance` |
| `analysis/moments.json` | 69 rows: `value`, `defender_mixed`, `attack_mixed`, `no_saddle`, `saddle_gap`, `static_attack_*`, and per player `rank`, `rank_frac`, `similarity`, `regret`, `fit_m` |
| `analysis/players.csv` | the same 207 player decisions, flat |
| `analysis/summary.json` | a corpus summary over moments |
| `scenes.csv` | one row per scene: match, frame range, role ids, reviewer ratings |
| `tracking/<scene>.csv` | **licensed** — every player and the ball, 25 fps |

## 2. Licensed vs public

**Licensed, never leaves this machine:** `tracking/*.csv` and the raw
coordinates in it. Used here only for mapping, frame alignment and validation.

**Exported to the public build:** equilibrium probabilities, the solver's own
0.6 s paths and endpoints, game values, ranks, relative ranks, the equilibrium
probability of the observed option, regrets, fit distances, the dilemma flags
and the static comparison — plus provenance.

Guards, all tested in `tests/test_bundle_integration.py`:

* `local_inputs/`, `*.tar.gz` and `tracking/` are gitignored, and
  `git ls-files local_inputs` is empty;
* the built site is walked for `tracking`, `bundesliga-integrated`, `.csv`,
  `.xml`, `.tar.gz`, `.npz`, `x_raw` and `local_inputs` — any hit fails;
* an exported option's `path` must be ≤ 30 points (the solver's 0.6 s path is
  25; a clip is hundreds), so a trajectory cannot masquerade as one.

The seven scenes shown already publish their own tracking in this repository.
**The bundle's CSVs did not replace them** — the existing public scene
payloads are untouched.

## 3. Coordinates and frames

Panels are corner-origin on the solver's pitch; this repository is
centre-origin and unnormalised, so the conversion is a translation only:

```
x_scene = x_panel - 52.5      y_scene = y_panel - 34.0
```

Verified at 0.0000 m against the published scene payloads for every body of
every moment.

```
scene frame index = panel.start_frame - scenes.csv:clip_first_frame
```

S05's moments land on frames **55, 70, 85** — 15 frames, 0.6 s apart.

## 4. The Korean command names

The panels were written **before** upstream's English cleanup (`e89e78f`,
2026-09-30 18:19), so a defender's command still carries its Korean name. The
map is that commit's own rename, recovered from the diff and re-derived from
the commit by a test:

| panel | upstream English |
| --- | --- |
| `멈추기(감속)` | `slow down` |
| `옆으로` | `sideways` |
| `볼 쪽` | `toward ball` |
| `러너 쪽` | `toward runner` |
| `수혜자 쪽` | `toward beneficiary` |
| `골문 쪽` | `toward goal` |

All six appear in the bundle. The attackers' compass names (`stop`, `forward`,
`left`, `back`, `right`) were always English. The export carries the English
`label` and keeps `legacy_label` beside it.

## 5. What the bundle does **not** contain

* **`static_counterfactual.py`'s whole-window V / S / R / A.** What it has is
  `analyze_eval`'s static comparison: the defender held to his *observed*
  command, the attack's best answer there (`static_attack_appeared`), the same
  choice once he may answer (`static_attack_responsive`), and the difference
  (`static_attack_loss`). Those schema slots stay unavailable, and a test
  asserts no `V`/`S`/`R`/`A` appears in any payload.
* **A clip-level player score.** `analysis/summary.json` aggregates over
  *moments* across the corpus; that is not a per-clip score, so
  `clip_metrics` stays empty.
* **Solved moments for every scene.** S46, S48, S51 and S60 have no 0.6 / 1.2;
  S64 has no 1.2. Missing moments are left out, never carried forward.

## 6. Scene coverage

Indexed only where the curated showcase names a **published** scene, so a
policy can never appear on a scene the demo does not publish:

`S05 S13 S15 S20 S34 S36 S44` → 7 scenes × 3 moments.

**S53** is in the bundle but has no published tracking, so it is not indexed.

## 7. The second bundle (2026-10-03)

    local_inputs/offball_demo_data_batch2_20261003/

Six more curated scenes, each solved at its real 0.0 / 0.6 / 1.2 s: **S02 S03
S04 S08 S10 S35** → 6 scenes × 3 moments. Same solver, flags, scripts and file
layout as the first bundle (eval_v1's `run.py` call; `extract_panel_policy.py`
+ `analyze_eval.py`); the starts and roles are the ones picked for these scenes
on 2026-10-03 and travel with the bundle as `starts.csv`. Its `analysis/` covers
only these 18 moments, so the first bundle's corpus summary is unchanged.

`paper_story/bundle.py` reads it after the first (`EXTRA_BUNDLES`, looked for
beside the first bundle); a code the first bundle already has is never
replaced. Each scene's payload names the bundle it came from
(`provenance.bundle`, and the story payload's `evaluation_source`). Every body's
start was checked against the published scene payload at its frame: 0.0000 m
on all 18 moments.

Coverage is now 13 published scenes: `S05 S13 S15 S20 S34 S36 S44` from the
first bundle, `S02 S03 S04 S08 S10 S35` from this one.

## 8. The five pipeline scenes, published (2026-10-03)

**S46 S48 S53 S58 S66** were solved in the first bundle but could not be
played: the stage-3 pipeline found them, so they have no shot row and no
published scene (section 6). `data/pipeline_scenes.json` now lists each one --
match, half, run-onset frame (time 0) and the frame window the bundle's panels
are indexed against, copied from the bundle's `scenes.csv`, plus the three role
ids; metadata only, no trajectory. `sources/window_scene.py` builds a scene
from that window exactly as `annotation_scene.py` builds one around a shot
(same IDSSE window loader, player records, direction and carrier rules), and
`loader.load_scene` resolves those ids (`J03WOH:run_P2_1009` = S53: half and
seconds at the onset, as the annotated ids carry the shot's). `export_data`,
`export_compare` and `export_release` export them after the annotated clips;
their index rows carry `effect: "solver"`. The ingest then maps all five
("roles identical"), so all 21 curated entries are playable.

Checks: re-exporting the 45 annotated scenes, their compare and release files
reproduced the committed ones byte for byte; every solved body's start equals
the new scene payload at its frame (0.0000 m). S46 and S48 have the 0.0 s
moment only (the bundle skipped 0.6 / 1.2 s; section 5).

Coverage is now 18 published scenes with an equilibrium: twelve from the first
bundle, six from the second.

## 9. The third bundle (2026-10-04)

    local_inputs/offball_demo_data_batch3_20261004/

**S06** and **S27**, each at its real 0.0 / 0.6 / 1.2 s, same solver, flags
and layout as the others; read as the second `EXTRA_BUNDLES` entry. Starts and
roles in its `starts.csv`. Two things differ from the earlier scenes:

* **S06's game swaps the annotation's roles.** The annotation names #13
  Choupo-Moting the runner and #10 Sané the beneficiary, but Choupo's run
  happens while the long ball is flying to him, which no game here can model.
  The game is solved from his first touch as a 2v1 with **Choupo the ball
  carrier and Sané the runner** (the reviewer's choice). So the Players card
  (the curated roles) and the equilibrium's runner name different players in
  this scene.
* **S27's passer is on the ball only once per game.** Paqarada crosses first
  time from a pass that rolls to him for about 2 s; he counts as on the ball
  only within 1.5 m of it (as S36's schedule), so each game has one pass
  instant (0.0 s game: the last; 0.6 s: the third; 1.2 s: the second) and no
  pass at any game's opening decision -- which is the moment the Game
  solution panel shows.

Every solved body's start equals the published scene at its frame (0.0000 m).
Coverage is now 20 published scenes with an equilibrium; of the 21 curated
entries only S30 has none.
