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
