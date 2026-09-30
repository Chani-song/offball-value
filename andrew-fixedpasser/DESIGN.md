# 3v1 fixed passer — design (for review, updated 2026-09-23)

## Rules for this folder

- `andrew/` is Andrew's original (commit e8b0a95). **Do not modify it.** Import from it only.
- This folder overrides **only the parts that change**, on top of the original.
- The shared computations (background defenders · offside · xT) must give the same values as the sibling folder `andrew-passer2on1/` (2v1); tests check this.

## Scope

Of the 896 triples that pass the gate + protagonist rule, the **718 triples (464 scenes)** where **R9 beneficiary ≠ ball carrier**.
The defender's dilemma: **follow the runner, or guard the beneficiary.** The ball carrier is not part of the dilemma;
he is the one who passes to whichever of the two is left open. Agreed with Andrew: "3v1, but the passer cannot move otherwise, only pass.
Keep the number of pass options at every moment."

## Key idea: swap only the body slots

Andrew's solver builds the state as a product of **three moving body slots**; it does not know who fills each slot.
`matrices()`, `masks()`, solving, verification and saving run on the three slots only.

| Slot | Original 2v1 | 3v1 |
|---|---|---|
| `carrier` | ball carrier | **runner** (decides) |
| `receiver` | receiver | **beneficiary** (decides) |
| `defender` | defender | **reacting defender** (decides) |
| outside the slots | — | **passer** + **all other defenders + goalkeeper** (no decisions, actual tracks) |

Subclass `FiniteGame` and rewrite **only the constructor**. Everything else is inherited.

## Used unchanged from the original

Motion model, xPass model weights, the framework for the 18 pass types to one receiver, backward solve, correctness check, saving.

## Decided (2026-09-23)

1. **3 s** — 1 s × 3 turns.
2. **The passer follows his actual path (option B)** — passer position/velocity at turn k = the actual values k s after run onset.
   He makes no decisions. Real ball carriers move a median 7.4 m in 3 s (36% move 10 m or more), so the pass
   origin matches reality better than holding him in place. Same state count and cost.
3. **Background defenders** — everyone except the reacting defender, plus the goalkeeper; actual tracks, no decisions.
4. **Passes**: every turn, in every state, the best of 18 passer→runner + 18 passer→beneficiary = 36 pass types.
   The pass comes before the simultaneous move, so keeping only the best is exact (Andrew's code comment). Record to whom and which pass.
   → meets "keep the pass options every turn". Attacking actions = runner 5 × beneficiary 5 + pass 1 = **26, unchanged**.
5. **All defenders in xPass** — as originally designed. "Nearest defender to the passer/receiver" is taken from the background + reacting defenders.
   Measured: the defender nearest the beneficiary is not the reacting defender in 71% of cases.
6. **Real offside rule** — second-last defender. Measured: about 18% are offside only when judged against the one defender.
7. **xT against all defenders** — the distance in `room` and `support_room` becomes the distance to "the nearest defender".
   The only change to Andrew's formula; coefficients unchanged.
8. **Tackles off** — otherwise a solution may appear where the defender stands next to the non-deciding passer and wins the ball for free. The reacting defender is not the presser (gate).
9. **Retention off** — retention value 0 at the end of 3 s → the attack must pass within 3 s (at the best moment).
   "The defender assumes the passer will certainly pass to one of the two."

## New code

- `FixedPasserScenario`: passer track, runner, beneficiary, reacting defender, background defender tracks, pitch, attack direction.
- `FixedPasserGame(FiniteGame)`: constructor — `build_layers` for the three slots, 36 pass types, survival 1, retention 0.
- Sample playback: the original always computes the pass target from the `receiver` slot, so it draws passes to the runner wrongly → own version.
- Runner script: same outputs as our `run_stage3.py` → the results table and page code read them unchanged.
- Input conversion: `kept_final` in `pair_gate_v7.csv` & beneficiary ≠ ball carrier.

## Size and cost

State-count bound `5^9 ≈ 1.97 M` unchanged. 26 attacking actions unchanged. Twice the pass computation + search over all defenders →
about 50-70 CPU s per scene expected → about **10-14 SU** for the 718 triples. Separate approval before the full run.

## Checks (before the full run)

1. Correctness gap ≈ 0.
2. **Symmetry**: swapping the runner and beneficiary slots leaves the game value unchanged.
3. **Reduction**: with every pass to the beneficiary blocked, the defender commits to the runner.
4. The defender does not drift toward the ball.
5. Coordinates: solver start positions = actual tracking (zero error). Direction commands are relative to the attack direction (`× attack_direction`).
6. About 10 scenes on the login node → shown on a page → approval for the full 718 run.

## Known limitations

- The passer and background defenders follow their actual paths — had the reacting defender moved differently, they would have moved differently too.
  If the real ball carrier had already passed within the 3 s, his later positions mean little.
- xT and xPass weights are Andrew's, unchanged. Andrew: "If a scene looks odd, it is likely down to the xT/xPass values."
- With retention off, the question becomes conditional on "a pass is certainly made" — intended.
