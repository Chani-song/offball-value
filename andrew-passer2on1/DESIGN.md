# 2v1: beneficiary = ball carrier — design (for review, 2026-09-23)

## Rules for this folder

- `andrew/` is Andrew's original (commit e8b0a95). **Do not modify it.** Import from it only.
- This folder overrides **only the parts that change**, on top of the original.
- The shared computations (background defenders · offside · xT) must give the same values as the sibling folder `andrew-fixedpasser/` (3v1); tests check this.

## Scope

Of the 896 triples that pass the gate + protagonist rule, the 178 where **R9 beneficiary = ball carrier** → 167 after the speed-cap filter.
The defender's dilemma: **press the ball, or follow the runner.** The benefit is the ball carrier gaining by dribbling.
Same bodies as Andrew's original game: ball carrier (moves and decides) + runner (receiver) + reacting defender.

Already solved twice with the unchanged original on 2026-09-22/23 (1.2 s `stage3_carrier`, 3 s `stage3_carrier_3s`).
This version differs only in "Changes" below — so **there is a baseline to compare the effect of the changes directly.**

## Used unchanged from the original

Motion model, tackles (reacting defender ↔ ball carrier), retention, xPass model weights, backward solve, correctness check, saving.

## Changes

1. **Game length 3 s** — 1 s × 3 turns. The state count depends only on the number of turns, so the cost is the same (measured: 33 CPU s per scene).
2. **Background defenders** — add every defender except the reacting one, plus the goalkeeper. **They make no decisions and
   follow their actual tracking paths** (position at turn k = actual position k s after run onset). They are a function of time only,
   not part of the state, so the state count is unchanged.
3. **All defenders in xPass** — the model was designed to take several defenders and was trained on all visible defenders
   in real data. "Nearest defender to the passer" and "nearest defender to the receiver" are chosen from the background + reacting
   defenders. (Now: the one reacting defender plays both roles.)
4. **Real offside rule** — the line is the **second-last** of the background + reacting defenders. (Now: the one reacting defender.
   Measured: the runner is offside only against that one defender in about 18% of cases.)
5. **xT against all defenders** — in `room` and `support_room` of Andrew's `positional_threat`,
   "distance to that defender" becomes "distance to the nearest defender". **The only change to Andrew's formula.** The other
   coefficients (9 m, 0.2, 0.8, 0.55, 0.25) are unchanged.
6. **Ball-carrier speed cap 7.2 m/s** — as decided on 2026-09-22 (the original 5.6 is a convention for synthetic scenes).

## Kept as is (review requested)

- **Tackles on.** In this game "press the ball" is itself one side of the dilemma, so a defender who gets close must be able to win the ball.
  Tackles only between reacting defender ↔ ball carrier. Background defenders do not tackle (they make no decisions).
- **Retention on.** If the defender follows the runner, the ball carrier carries the ball forward — that is the benefit in this case.

## Checks (before the full run)

1. Correctness gap ≈ 0.
2. **With background defenders removed**, results must be **identical** to the original 3 s run (`stage3_carrier_3s`) — evidence that the new code
   did not break the original.
3. Coordinates: solver start positions = actual tracking (zero error).
4. About 10 scenes on the login node → shown on a page → approval for the full 167 run.

## Known limitations

- Background defenders follow their actual paths — had the reacting defender moved differently, they would have moved differently too.
- xT and xPass weights are Andrew's, unchanged. Andrew: "If a scene looks odd, it is likely down to the xT/xPass values."
