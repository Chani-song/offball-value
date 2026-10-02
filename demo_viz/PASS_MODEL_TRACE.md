# Pass model: exact execution trace

What `experimental_pass.json` actually computes, how Kyuhyeok's pipeline calls
it, how the solver uses the result, and what the demo may safely show.

Traced from code on 2026-09-28 against:

* current tree and `origin/kyuhyeok-dev` @ `3c9965b`
* `~/Research/offball_demo/mit_ssac2027_ref` @ `e8b0a95` (read-only, unmodified)
* `local_inputs/models/experimental_pass.json` (local-only, gitignored)

Nothing here was inferred from variable names. Every claim is from the
implementation, and the numbers at the end are reproduced twice.

---

## 1. Artifact identity

```
local_inputs/models/experimental_pass.json
mit_ssac2027_ref/models/experimental_pass.json
```

**Byte-identical** (`cmp` clean, 4826 bytes). There is one model, and it is
Andrew's. Safe to proceed.

| field | value |
| --- | --- |
| `schema` | `kick_positions_target_two_defender_roles_v1` |
| `features` | 33 names (§4) |
| `mean`, `scale`, `weights` | 33 each |
| `bias` | `3.8922706118930854` |
| `metadata.kind` | `fitted_logistic` |
| `metadata.target_semantics` | `annotated_receipt_proxy` |
| `metadata.validation_status` | **`experimental_proxy`** |
| `metadata.feature_set` | `positions` |
| `metadata.fitting_method` | `fixed_receiver_marginal_likelihood` |
| training / test | 7,332 rows (776 failures) / 1,964 rows (206 failures) |
| `metadata.warning` | "Known-receiver component is not identified/validated by bag-level prediction metrics." |

There is **no top-level `kind`**, which matters in §3.

---

## 2. The call chain that produced the meeting results

From `deploy/delta/stage3_passer2on1_agile.sbatch` — the 2v1 solver page:

```
data/processed/stage3/passer2on1_states_agile06_final.json
  └─ andrew-passer2on1/scripts/run.py
       --model andrew/models/experimental_pass.json --allow-proxy-labels
       --steps 3 --step-seconds 0.6 --physics agile
     ├─ run.py:model_router          → plain path (no per-match router)
     └─ passer2on1/solve.py:solve_one
          ├─ physics_pass.py:load_pass_model(path, allow_proxy)
          │    └─ ExpectedPass.load(path, allow_proxy=True)      ← Andrew's
          ├─ passer2on1/game.py:Background2on1Game
          │    └─ passer2on1/payoff.py:release_payoffs_with_background
          │         ├─ payoff.defence()          defender + background
          │         ├─ payoff.offside_flags()
          │         ├─ model.predict(...)        ← Andrew's ExpectedPass
          │         │    └─ expected_pass.pass_features(positions_only=True)
          │         │    └─ expected_pass.probability()
          │         └─ payoff.possession_value_all()
          └─ markov.solve_markov_game / certificate / rollout
  └─ scripts/summarise_stage3.py  → passer2on1_results_agile.csv
  └─ scripts/render_solver_story.py → out/passer2on1_agile/index.html
```

`--allow-proxy-labels` is not cosmetic: without it `ExpectedPass.load` **raises**
and the run stops (§6).

A second, separate path exists and is **not** what produced the solver page:
`src/offball_value/ssac_stack.py:completion_probability`, used by
`local_game_payoff.py` for stage-2 scene ranking. It calls the same model
through the same class, from this repository's coordinates (§8).

---

## 3. Kyuhyeok vs Andrew: what is the same, what changed

**Category C — wraps Andrew's implementation and changes which defenders it is
shown.** The model code is never copied or reimplemented.

`physics_pass.py:load_pass_model` is a two-line router: if the JSON has
`kind == "physics_race_logit"` it builds Kyuhyeok's Candidate A; otherwise it
delegates to `ExpectedPass.load`. `experimental_pass.json` has no top-level
`kind`, so **it always takes Andrew's loader**.

| quantity / behaviour | Andrew | Kyuhyeok pipeline | same? | file:function |
| --- | --- | --- | --- | --- |
| model class | `ExpectedPass` | same object, imported | **same** | `expected_pass.ExpectedPass` |
| loader | `ExpectedPass.load` | reached via router | **same** | `physics_pass.load_pass_model` |
| proxy guard | raises unless `allow_proxy` | `--allow-proxy-labels` → True | **same** | `expected_pass.ExpectedPass.load` |
| feature builder | `pass_features(positions_only=True)` | unchanged, called by `predict` | **same** | `expected_pass.pass_features` |
| feature order | `POSITION_FEATURES` | unchanged | **same** | `expected_pass.FEATURES` |
| standardisation | `(x-mean)/scale` | unchanged | **same** | `ExpectedPass.probability` |
| link | `sigmoid = 0.5(1+tanh(z/2))` | unchanged | **same** | `expected_pass.sigmoid` |
| defender set handed to the model | the one controlled defender | **controlled defender + background** | **changed** | `passer2on1/payoff.defence` |
| offside | `offside_against_line` (single defender as the line) | `offside` (second-last of the set) when ≥2 defenders | **changed** | `passer2on1/payoff.offside_flags` |
| threat "room" | distance to the one defender | distance to the **nearest** of the set | **changed** | `passer2on1/payoff.positional_threat_all` |
| payoff formula | `completion × threat`, 0 if illegal | identical | **same** | `passer2on1/payoff.release_payoffs_with_background` |
| `value_function` hook | optional override | dropped; always `positional_threat_all` | **changed** | same |

Verified numerically: with an **empty background** the wrapper's pass payoffs
are `np.array_equal` to Andrew's `release_payoffs` — bit-identical, all 18 pass
choices, on real declared states 3 and 7.

The three changes are consequential, not cosmetic:

* Showing one extra defender on state 3 moved the completion proxy for
  `pass[6]` from **0.992984** to **0.907020**. Only the
  `defender_near_receiver_*` block and `same_defender` moved; the
  `defender_near_passer_*` block did not, because the controlled defender was
  still nearest the passer.
* On that same state the offside switch took **every** pass payoff to 0. The
  legality gate, not the model, did that.

---

## 4. The 33 features, exactly

`expected_pass.pass_features(..., positions_only=True)`. Corner-origin metres
on 105 × 68. When `attack_direction == -1` every position is replaced by
`[pitch_length, pitch_width] - v` — a **180° rotation, not a mirror**.

Let `c` = carrier, `r` = receiver, `t` = target, `lane = t - c`,
`length = |lane|`, `offset = t - r`, `gap = |offset|`.

| # | name | definition |
| --- | --- | --- |
| 0–1 | `carrier_x/y` | `c` |
| 2–3 | `receiver_x/y` | `r` |
| 4–5 | `target_x/y` | `t` |
| 6 | `pass_length` | `length` |
| 7 | `forward_distance` | `lane[0]`, signed, after rotation |
| 8 | `receiver_target_gap` | `gap` |
| 9 | `receiver_target_gap_squared` | `gap²` |
| 10–19 | `defender_near_passer_*` | block below, defender nearest **c** |
| 20–29 | `defender_near_receiver_*` | block below, defender nearest **r** |
| 30 | `same_defender` | 1.0 when both roles select the same opponent |
| 31 | `driven` | `family == 1` |
| 32 | `lofted` | `family == 2` |

Per defender block, with `p` the selected defender's position:

| suffix | definition |
| --- | --- |
| `_x`, `_y` | `p` |
| `_passer_distance` | `|p - c|` |
| `_receiver_distance` | `|p - r|` |
| `_target_distance` | `|p - t|` |
| `_lane_distance` | distance from `p` to the **segment** `c→t`: `f = clip((p-c)·lane / length², 0, 1)`, then `|p - (c + f·lane)|` |
| `_passer_pressure` | `exp(-passer_distance / 3)` |
| `_receiver_pressure` | `exp(-receiver_distance / 5)` |
| `_target_pressure` | `exp(-target_distance / 5)` |
| `_lane_pressure` | `exp(-lane_distance / 3)` |

Details that are easy to get wrong and are **not** inferred:

* **Receiver semantics.** Roles are nearest to the passer and nearest to the
  **receiver at kick** — never to the target or the endpoint.
* **Tie-breaking.** `_nearest_index` uses `np.lexsort` on `(y, x, distance)`
  (plus velocities when present), so exact ties are canonical, not input-order.
* **One defender fills both roles** when it is nearest to both; that is what
  `same_defender` records. The 2v1 game supplies its sole defender once and the
  two blocks are then identical — it does not add a player.
* **`ground` is the base level**; `driven`/`lofted` are one-hot, no third column.
* **No missing-value handling.** Any non-finite input raises
  `"kick-time positions, velocities and target must be finite 2D vectors"`.
* **No clipping inside the model.** `ssac_stack.completion_probability` clips
  the *output* to [0, 1]; the solver path instead *validates* that it is already
  in [0, 1] and raises otherwise.

---

## 5. The prediction

`ExpectedPass.probability`:

```
z  = (features - mean) / scale        # elementwise, 33 long
logit = z · weights + bias
p  = 0.5 * (1 + tanh(logit / 2))      # == 1 / (1 + exp(-logit))
```

That is the whole model. One standardised logistic regression, no interactions
beyond those in the feature list, no calibration layer.

---

## 6. `--allow-proxy-labels`

`ExpectedPass.load` refuses this artifact by default. The guard fires because
`target_semantics == "annotated_receipt_proxy"` **and**
`validation_status == "experimental_proxy"`:

```python
raise ValueError("experimental proxy model: explicitly opt in with --allow-proxy-labels")
```

So the flag is the repository's own statement that the number is not a
validated completion probability. `ssac_stack._pass_model` hardcodes
`allow_proxy=True`; every solver sbatch passes `--allow-proxy-labels`.

---

## 7. How the solver uses the output

`passer2on1/payoff.py:release_payoffs_with_background`, per pass choice:

```python
target     = choice.target(receiver, scenario.attack_direction)
legal      = inside_pitch(target, ...) & ~offside_flags(...)
completion = model.predict(carrier, receiver, defenders, ..., target,
                           FAMILIES.index(choice.family), ...)
threat     = possession_value_all(target, carrier, defenders, scenario)
payoff     = np.where(legal, completion * threat, 0.0)
```

**The pass-model output is one multiplicative factor in a terminal payoff.** It
is not the solver's objective and it is not used to rank passes on its own.

`threat` is `positional_threat_all` — evaluated **at the target**, with the
carrier as support:

```
room          = 1 - exp(-dist(target,  nearest defender) / 9)
support_room  = 1 - exp(-dist(carrier, nearest defender) / 9)
threat = 0.2 + 0.8 · location_core(target) · (0.55 + 0.25·room
                                              + 0.2·location_core(carrier)·support_room)
```

bounded to [0.2, 1], with `location_core` a tanh in goalward progress times a
Gaussian in centrality (`value.py:location_core`). `payoff.possession_value`'s
own docstring: *"Zero is the value of losing possession… The value scale need
not be a probability."*

The 18 per-choice payoffs then enter `markov.solve_markov_game` as the terminal
values of the release action; the equilibrium value reported on the solver page
is the root value of that game.

---

## 8. Numerical reproduction

Real declared pipeline state — `conditioning_100.json` state 0 (`open`), the
same file the solver run consumes. Ground pass, zero offset, so `target = receiver`.

```
carrier  [63.0782029956, 32.6487439341]
receiver [73.8538458894, 49.4193714256]
defender [53.7595227467, 47.8555279197]
attack_direction = 1
```

33 raw features, in model order:

```
 0 carrier_x                                63.078202996
 1 carrier_y                                32.648743934
 2 receiver_x                               73.853845889
 3 receiver_y                               49.419371426
 4 target_x                                 73.853845889
 5 target_y                                 49.419371426
 6 pass_length                              19.934102092
 7 forward_distance                         10.775642894
 8 receiver_target_gap                       0.000000000
 9 receiver_target_gap_squared               0.000000000
10 defender_near_passer_x                   53.759522747
11 defender_near_passer_y                   47.855527920
12 defender_near_passer_passer_distance     17.834911852
13 defender_near_passer_receiver_distance   20.155084447
14 defender_near_passer_target_distance     20.155084447
15 defender_near_passer_lane_distance       16.060065684
16 defender_near_passer_passer_pressure      0.002618979
17 defender_near_passer_receiver_pressure    0.017756265
18 defender_near_passer_target_pressure      0.017756265
19 defender_near_passer_lane_pressure        0.004732247
20 defender_near_receiver_x                 53.759522747
21 defender_near_receiver_y                 47.855527920
22 defender_near_receiver_passer_distance   17.834911852
23 defender_near_receiver_receiver_distance 20.155084447
24 defender_near_receiver_target_distance   20.155084447
25 defender_near_receiver_lane_distance     16.060065684
26 defender_near_receiver_passer_pressure    0.002618979
27 defender_near_receiver_receiver_pressure  0.017756265
28 defender_near_receiver_target_pressure    0.017756265
29 defender_near_receiver_lane_pressure      0.004732247
30 same_defender                             1.000000000
31 driven                                    0.000000000
32 lofted                                    0.000000000
```

| | |
| --- | --- |
| logit, via `ExpectedPass` | `6.498830792478147` |
| logit, independent numpy | `6.498830792478147` |
| completion proxy, `model.probability` | `0.998497064162670` |
| completion proxy, `model.predict` | `0.998497064162670` |
| completion proxy, independent `1/(1+e^-z)` | `0.998497064162670` |
| **max difference** | **1.11e-16** |

Agreement is at float epsilon. Note `same_defender = 1` and the two defender
blocks identical: the 2v1 game has one defender, exactly as documented.

---

## 9. Recommendation for demo integration

**Import the existing Python, do not re-implement in JavaScript.** The feature
builder has four details a port would plausibly get wrong — segment-clamped
lane distance, lexsort tie-breaking, the 180° rotation for
`attack_direction == -1`, and two different pressure length-scales (3 m and
5 m). A silent error in any of them would be invisible on screen.

The right entry point is **`ssac_stack.completion_probability`**: it already
takes this repository's centre-origin coordinates, converts with
`_to_corner_origin`, hands the model the whole defending side rather than
pre-selecting, and loads with `allow_proxy=True`. It is the one function that
bridges the two coordinate systems.

Suggested shape, mirroring how OBSO is already handled:

1. bring `ssac_stack.py` across byte-identical, as was done for
   `reference_obso.py` and `action_space.py`;
2. `defensive_positioning` is a separate repository — so either vendor nothing
   and precompute, or make the export step depend on it being present, and skip
   with a clear message when it is not;
3. **precompute** the proxy and its inputs for the five candidate endpoints per
   frame at export time, and ship the derived numbers, rather than porting 33
   features to JS;
4. the model file is local-only and gitignored; the *derived* per-endpoint
   numbers are small and safe to publish, but the model's weights must not be
   committed.

Blocker to note: this path needs Andrew's `defensive_positioning` package
importable at export time. It is present locally at `mit_ssac2027_ref/src` but
is **not** a declared dependency of this repository, and the model JSON is not
committed. Any export step must therefore be optional and clearly skipped in a
clean checkout.

---

## 10. What the demo may safely say

| label | value | why this wording |
| --- | --- | --- |
| **Completion proxy** | `0.xxx` | the artifact ships `validation_status: experimental_proxy` and `target_semantics: annotated_receipt_proxy`, and the code refuses to load it without an explicit opt-in |
| Pass length | m | feature 6 |
| Forward distance | m | feature 7 |
| Receiver–target gap | m | feature 8 |
| Passer pressure | 0–1 | features 16 / 26, `exp(-d/3)` |
| Receiver pressure | 0–1 | features 17 / 27, `exp(-d/5)` |
| Lane pressure | 0–1 | features 19 / 29, `exp(-d/3)` |
| Same defender | yes / no | feature 30 |

**Must not be said:** "pass success probability", "calibrated", "validated",
"expected value", "best pass". The solver multiplies this by an explicitly
uncalibrated threat term; neither factor is a probability of anything observed.

**Must also be said, if the five fan directions are priced:** those targets are
generated by the demo's own fixed geometry (90° sector, five 25 m rays). They
were not generated, proposed or ranked by the research pipeline. Feeding them
into the real model is exploratory inspection of the model's response surface,
not a pipeline result.

The solver's own 18 pass choices are a different set: three families × six
offsets applied to the **receiver's** position, not rays from the carrier —
`target = receiver + direction · offset`, with offsets `(0,0), (4,0), (8,0),
(-4,0), (0,4), (0,-4)` (`models.DEFAULT_PASSES`). Four lie on the goalward
axis and two are lateral, and both axes rotate with the attacking direction.
