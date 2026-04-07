# Idea sketch

## Target question

How much does an attacking player's off-ball movement improve a teammate's attacking options, even when the runner does not receive the ball?

## Minimal decomposition

For a candidate runner `i`, ball carrier `b`, and evaluation horizon `t -> t+Δ`:

- detect motion by `i`
- compute option-quality summary for attacking team at `t+Δ`
- compute a counterfactual summary with runner `i` held fixed or moved by a simple baseline trajectory
- attribute the difference to runner `i`

## Toy score

A very rough first score used in this repo is:

```text
OptionScore = w1 * forward_receiver_gain
            + w2 * proximity_to_goal_gain
            + w3 * separation_from_nearest_defender_gain
```

This is intentionally crude.

## More serious future score

A stronger version should replace the toy score with one of:

1. pass availability model
2. pitch-control or space-generation model
3. EPV / xT / action-value model
4. learned option-value model over tracking + events

## Candidate move classes

- decoy run
- support run
- overlap
- underlap
- run in behind
- pinning / occupation run

## What would count as a good first paper contribution

Not "off-ball movement matters".

A better contribution would be:

> We propose a reproducible open-data benchmark and a counterfactual framework for valuing non-receiving attackers based on how their movement changes teammates' option quality.

## Realistic proof-of-concept plan

1. Start with Metrica to validate parsing and visualization.
2. Move to SkillCorner for richer tracking-based experiments.
3. Use StatsBomb 360 for event-level sanity checks.
4. Optionally test on the Bundesliga integrated dataset as a benchmark-style replication dataset.
