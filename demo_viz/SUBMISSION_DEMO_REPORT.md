# Submission demo — research inspector

What the public demo exposes, where every number comes from, and what is still
unavailable. Written for a reviewer who has not read the code.

---

## 0. Audit: what exists, and where

Taken from the current tree, `origin/kyuhyeok-dev` @ `3c9965b` (read-only) and
`~/Research/offball_demo/mit_ssac2027_ref` @ `e8b0a95` (read-only, unmodified).

### Already in this tree

| Quantity | Where | Status |
| --- | --- | --- |
| Player/ball tracks, 25 Hz | `web_data/<scene>.json` | measured |
| Velocity (central difference, 0.4 s window) | `js/influence.js` `velocityAt` | derived |
| Run onset | `core/role_logic.runner_onset_index`, exported per attacker | derived |
| Residual space (Available space) | `offball_value.goal_weighted_influence` | derived |
| Held-defender counterfactual (Space created) | `core/influence.InfluenceCache` | derived |
| OBSO = control × transition × EPV | `offball_value.reference_obso` | derived |
| EPV grid | `data/static/EPV_grid.csv` (PAUSA, Apache-2.0) | reference data |
| Candidate pass fan | `core/candidates.py`, `js/obso.js` | fixed geometry |

### On `origin/kyuhyeok-dev`, inspected but **not** brought across

| Module | Why not |
| --- | --- |
| `physics_pass.py` (pass candidate A) | needs a fitted JSON that lives only on Delta |
| `xpass.py`, `kinematic_xpass.py` | training-side; no local fitted artefact |
| `vacated_space.py` | author flags it as post-hoc on the round-2 test |
| `defender_response.py`, `attacker_maximin.py` | need Delta pipeline CSVs |
| `steering_reachable.py` | see below — measured at 9.5 s per player-frame |

### Reachability: three implementations, one choice

| Model | Cost | Used by |
| --- | --- | --- |
| `action_space.solve_endpoint_motion` | microseconds, closed form | endpoint action grids |
| `steering_reachable` | **9.5 s** per player-frame (measured, 1115 endpoints) | defender-response pipeline |
| `dynamic_reachable` | not measured | calibration scripts |

The defender-response pipeline uses the steering model. At 9.5 s per
player-frame it cannot run in a browser and cannot be precomputed for 22
players × 276 frames × 45 scenes. **The Reachable area layer therefore shows
`action_space.solve_endpoint_motion`** — accelerate-then-cruise, 2.0 s horizon,
9 m/s, 3.5 m/s² — ported to JavaScript with the same constants and checked
against Python. The Source panel names both models and says which is drawn.
This is a different model shown exactly, not the steering model approximated.

### Solver artifacts (`mit_ssac2027_ref`)

Real, and present locally: `results/exact_100/` holds 100 solved 2v1 states.

| File | Contents |
| --- | --- |
| `states/state_NNN.json` | scenario (carrier/receiver/defender position, velocity, speed and acceleration caps), `value`, `certificate` {lower, upper, gap}, `root_attack` (26), `root_defender` (5), timings |
| `rows.jsonl` | the same per state, plus `rollouts` |
| `policies/state_NNN.npz` | `attack_k` (n,n,n,26), `defender_k` (n,n,n,5), `pass_index_k`, `terminal_release`, `metadata` (schema, fingerprint, root value) |
| `manifest.json` | `steps`, `step_seconds`, `physics_step`, 5 `directions`, 18 `passes` |

Action encoding, from `markov.py`: 5 directions `[(0,0), (1,0), (0,1), (-1,0),
(0,-1)]`; an attack action `a < 25` splits as `carry, run = divmod(a, 5)`, and
`a == 25` is release. Rollout events are `move`, `release`, `retain`,
`tackled_during_previous_interval`. Coordinates are corner-origin metres on
105 × 68; the demo's are centre-origin, so the adapter shifts them.

**These are the solver's own declared 2v1 study states, not Bundesliga
scenes.** No solver artifact exists locally for any demo scene, and per
Kyuhyeok's 2026-09-25 note exactly one annotated scene reaches the solver at
all, on Delta. So every demo scene reports *Not computed for this scene*, and
the adapter is validated against the real artifacts above.

---

## 1. What was added

| | |
| --- | --- |
| **Analysis panel** | Player / Space & threat / Passing / Solver, each shown only when its selection or layer is live |
| **Player inspector** | click any player; speed, acceleration, distances, run onset, available space, reachable area |
| **Reachable area** | layer, off by default |
| **Solver solution** | layer, off by default, with an honest unavailable state |
| **Threat components** | pitch control / ball transition / EPV / OBSO at a player or a pass endpoint |
| **Candidate endpoint inspector** | click an endpoint: direction, distance, ball travel |
| **Submission manifest** | schema, validator, loader, example; absent ⇒ current explorer, unchanged |
| **Solver adapter** | reads real `mit_ssac2027` artifacts, with provenance |

Clicking a player inspects it *and* still assigns a role exactly as before, so
no existing interaction changed. Multi-runner, multi-beneficiary and
multi-defender assignment, drag and drop, the three space views, Focus, and the
Pages build all behave as they did.

## 2. Quantities exposed, and their provenance

Three categories, never blurred:

| Quantity | Category | Source |
| --- | --- | --- |
| Position, speed, acceleration, distances | **Measured** | tracking at 25 Hz; velocity by central difference over 0.4 s |
| Role | **Human** | manual annotation |
| Run start | **Derived** | `run_onset.detect_kinematic_run_onsets`, else a labelled acceleration cue |
| Available space | **Derived** | `goal_weighted_influence` residual |
| Space created | **Derived** | the same, minus a held-defender baseline |
| Pitch control, Ball transition, EPV, OBSO | **Derived** | `reference_obso`; EPV is the static PAUSA grid |
| Reachable area | **Derived** | `action_space.solve_endpoint_motion`, 2.0 s / 9 m/s / 3.5 m/s² |
| Candidate passes | **Derived** | fixed geometry: 90° sector, five 25 m rays |
| Ball travel | **Derived** | endpoint distance ÷ 15 m/s, the `ReferenceOBSOConfig` ball speed |
| Equilibrium value, certificate gap, root policies, trajectories | **Solver** | read from a real artifact only |

Two things are deliberately absent, because no validated local artifact defines
them: **pass completion probability** and any **expected value** combining
completion with threat. The candidate fan carries neither, and the Passing
section reports geometry and ball travel only.

`Ball travel` is worth one line of honesty: 15 m/s is a single constant from
the OBSO configuration, which the pitch-control term already assumes. It is not
a fitted flight model, and no arrival margin is claimed from it — receiver and
defender arrival would need `physics_pass`, whose coefficients are not local.

## 3. Reachability: which model, and why

`action_space.solve_endpoint_motion` — one constant acceleration, then cruise —
ported to JavaScript and checked against Python on 64 cases per run, feasibility
and solved motion, to 1e-9. It runs live: the 1 m action grid is 7,140
feasibility tests, a few milliseconds.

The defender-response pipeline uses the richer `steering_reachable` instead
(tangential and normal control with plant-and-cut). Measured here at **9.5 s for
one player at one frame**, producing 1,115 endpoints. That cannot run in a
browser and cannot be precomputed for 22 players × 276 frames × 45 scenes, so it
is not drawn. The Source panel names both models and says which one the layer
shows. Nothing is approximated silently.

## 4. Solver adapter format

Input is a run directory from `defensive_positioning.exact_study`, referenced as
`<run directory>#<state index>`:

    manifest.json           steps, step_seconds, directions, passes
    rows.jsonl              scenario, value, certificate, root policies, rollouts
    states/state_NNN.json   the same without rollouts (fallback)
    policies/state_NNN.npz  full conditional policy tables; metadata fingerprint

The adapter normalises corner-origin solver metres to the demo's centre-origin,
decodes the action encoding from the run's own manifest (`carry, run =
divmod(a, 5)`, `a == 25` is release) rather than assuming it, and attaches
repository, commit, artifact, state index, policy fingerprint and run timestamp
to every state.

`load_for_scene` returns `Unavailable(reason)` for a missing, malformed or
unreadable reference. There is no branch that substitutes anything else.

## 5. Which scenes have genuine solver output

**None.** All 45 report *Not computed for this scene*.

That is the true state of the pipeline, not a gap in this work: per the
2026-09-25 pipeline note, exactly one annotated scene reaches the solver at all,
and that run lives on Delta. The adapter is instead validated against real
artifacts — `mit_ssac2027@e8b0a95 results/exact_100`, 100 solved 2v1 states —
and three of them are exported to `web_data/solver/` under a
`solver_reference` kind that the tests forbid from shadowing any tracked scene.

### Integration run (§10)

The solver was run once, locally, on its own declared scenario set. No licensed
data was involved and the reference repository was not modified (verified clean
afterwards); output went to a temporary directory.

```
repo     ~/Research/offball_demo/mit_ssac2027_ref @ e8b0a95
command  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
         python -m defensive_positioning.exact_study \
           --states data/scenarios/conditioning_100.json \
           --model models/experimental_pass.json --allow-proxy-labels \
           --workers 2 --limit 1 --output <tmp>
runtime  12 s wall (one state; the published run reports a 20.3 s median)
input    data/scenarios/conditioning_100.json, state 0
output   manifest.json, rows.jsonl, states/, policies/state_000.npz
```

The freshly produced artifact parses with the adapter, and reproduces the
committed run exactly: value `0.7045360969038921` against `0.7045360969038921`,
and the policy tables are **bit-identical**. The `metadata.fingerprint` string
differs between the two runs even though the tables and the manifest config do
not, so it is not a content hash of the policy alone — worth knowing before
anyone uses it as one.

One dependency (`imageio_ffmpeg`) had to be installed into this project's venv
for the solver's clip-rendering step; `pyproject.toml` was not changed.

## 6. What remains unavailable

* **Pass completion probability.** Candidate A (`physics_pass`) needs a fitted
  JSON that exists only on Delta. The solver's current estimator is described by
  its own author as "an experimental proxy, not a validated replacement".
* **Expected value** of a pass, which needs the above.
* **Receiver / defender arrival margins**, same reason.
* **Solver output for any demo scene** — see §5.
* **Steering reachability** in the browser — see §3.
* **Optimal defender best response** as a separate overlay: the artifacts hold a
  joint equilibrium policy, not a labelled "best response" trajectory, so
  presenting one would be an interpretation rather than a reading.

## 7. Performance

| | |
| --- | --- |
| Initial load | 140 KB (shell + scene index) |
| Scene | 78 KB, lazy |
| OBSO surface | ~122 KB per scene, lazy, only when the threat view is picked |
| Solver artifact | ~4 KB per state, lazy, only when the solver layer is picked |
| Reachable area | computed live, 7,140 feasibility tests, memoised per player and frame |
| Role switch | unchanged at 0.8 ms |

No solver runs in the browser. Nothing added to the initial payload but the
shell, which grew from 103 KB to 132 KB.

## 8. Tests

230 pass. New: reachability JS↔Python parity (feasibility and motion, 1e-9),
solver artifact parsing against the real run (coordinates, action decoding,
probability normalisation, mixed vs pure, events, provenance, pass targets),
the unavailable state in four failure modes, a guard that no exported solver
file shadows a tracked scene, and manifest schema/validator/loader including
graceful absence. Browser↔Python parity for Available space and Space created
is unchanged at 8.5e-14.

## 9. Screenshots

`demo_viz/exports/submission/`

| File | What |
| --- | --- |
| `a_strong_inspector.png` | Strong scene, player inspector (1440p) |
| `b_medium_scene.png` | Medium scene, OBSO threat |
| `c_multi_runner.png` | two runners, two defenders, inspector on #10 |
| `d_obso_passes.png` | OBSO threat with the candidate fan (1440p) |
| `e_reachable.png` | Reachable area for a selected player |
| `g_solver_unavailable.png` | the honest solver state on a demo scene |

There is no solver-populated screenshot of a demo scene, because there is no
such artifact. §18's scenario F is the one thing here that genuinely does not
exist yet.
