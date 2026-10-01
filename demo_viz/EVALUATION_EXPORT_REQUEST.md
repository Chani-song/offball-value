# What the demo needs to show real evaluation numbers

Traced 2026-10-01 against `origin/kyuhyeok-dev` @
`8a5c69d0bb1c42948c3f9d50becd2ffc41e72344`.

**The code is done. This machine has none of the inputs.** Nothing here is a
request for new science — only for the outputs of runs that already have a
command line.

---

## 1. The canonical chain

```bash
# 1. the evaluation states: each scene's real 0.0 / 0.6 / 1.2 s moments
PYTHONPATH=src:scripts:andrew-passer2on1:andrew-fixedpasser \
  python scripts/build_eval_states.py --codes S05,S13,S34,S53,S15,S20,S36,S44 \
         --output data/processed/eval_v1

# 2. solve every moment as its own game  (jobs/solve_evaluation.sbatch)
PYTHONPATH=andrew-passer2on1 python andrew-passer2on1/scripts/run.py \
  --states data/processed/eval_v1/states_2v1_eval.json \
  --model data/processed/pass_models_sym/Asym_crossfit.json --allow-proxy-labels \
  --steps 3 --step-seconds 0.6 --commands compass --threat andrew --passes run \
  --physics-file data/processed/physics_limits/agile_p999_nodelay.json \
  --multi-pass --pass-reaction-s 0.2 --background-tackles \
  --workers 37 --output out/runs/eval_v1_2v1
#   ... and the 3v1 variant with andrew-fixedpasser and no --background-tackles

# 3. one panel per moment (options' ends and paths)
PYTHONPATH=src python scripts/extract_panel_policy.py \
  --solved out/runs/eval_v1_2v1 --code S05 --label eval \
  --output data/processed/eval_v1/panels/eval-S05.json

# 4. the evaluation numbers
PYTHONPATH=src:scripts python scripts/analyze_eval.py \
  --solved out/runs/eval_v1_2v1 out/runs/eval_v1_3v1 \
  --panels data/processed/eval_v1/panels --output <numbers>

# 5. the static counterfactuals  (jobs/static_counterfactual.sbatch)
PYTHONPATH=andrew-passer2on1:src:scripts python scripts/static_counterfactual.py 2v1 \
  --solved out/runs/eval_v1_2v1 --output data/processed/eval_v1/static_full --workers N
python scripts/summarize_static.py data/processed/eval_v1/static_full
```

Runtime, from the job files: **37 CPUs × 1.5 h** (2v1) and **32 × 2 h** (3v1).
Not a laptop job.

## 2. What is missing here

| input | needed by | status |
| --- | --- | --- |
| `data/raw/bundesliga-integrated/` | steps 1, 4, 5 and `figure_style.match_line` | **absent — licensed DFL tracking and match XML** |
| `data/processed/stage3/passer2on1_states_agile06_final2m.json` | step 1 (field-for-field check) | **absent** |
| `data/processed/stage3/fixedpasser_states_agile06_final2m.json` | step 1 | **absent** |
| `data/processed/rating_v1/merged.csv` | step 1 (the codes and their roles) | **absent** |
| `data/processed/showcase_v1/` (`scenes.csv`, `solver_starts_*.csv`, `tracking/`) | Figures 1 and 2 | **absent** |
| `out/runs/eval_v1_{2v1,3v1}/` | steps 3, 4, 5 | **absent** |
| `Asym_crossfit.json`, `agile_p999_nodelay.json` | step 2 | present in `origin/kyuhyeok-dev` |
| `defensive_positioning` | step 2 | present (`mit_ssac2027_ref/src`, 18 modules) |

Steps 1–5 cannot be started here, and per the standing rule no licensed data is
fetched. **The solver package and the fitted models are the only pieces this
machine has.**

## 3. What to send — numbers, not tracking

The join is already in place: upstream's showcase codes are **exactly** the
demo's curated ids, and seven of the eight already name a published scene.

| code | demo scene |
| --- | --- |
| S05 | `J03WOH:shot_010_P1_1759` |
| S13 | `J03WOH:shot_017_P1_2548` |
| S15 | `J03WOH:shot_004_P1_0429` |
| S20 | `J03WN1:shot_002_P1_0057` |
| S34 | `J03WR9:shot_004_P1_0794` |
| S36 | `J03WQQ:shot_016_P2_0313` |
| S44 | `J03WOH:shot_014_P1_2178` |
| S53 | *mapping unresolved — no published tracking* |

Per scene, per moment (`0.0 / 0.6 / 1.2 s`), from `analyze_eval.py`'s output
and `static_counterfactual`'s JSON:

```jsonc
{
  "code": "S05", "dt": 0.6, "kind": "2v1",
  "observed": { "<role>": { "command": 3, "label": "toward runner",
                            "nearest_m": 0.41, "second_nearest_m": 1.83,
                            "is_pass": false } },
  "rank":      { "<role>": { "rank": 2, "rank_frac": 0.75, "tie": false, "n": 5 } },
  "similarity":{ "<role>": 0.43 },     // equilibrium probability of the observed option
  "regret":    { "<role>": 0.012 },
  "policy":    { "<role>": [ { "command": 1, "label": "...", "prob": 0.57,
                               "path": [[x, y], ...], "end": [x, y],
                               "stop": false, "rests": false } ],
                 "passes": [ { "to": "runner", "prob": 0.74,
                               "ball": [x, y], "target": [x, y],
                               "where": "run +8 / side 0 / goal 0" } ] },
  "value": 0.681,
  "dilemma": { "defender_mixes": true, "saddle_gap": 0.031, "is_dilemma": true },
  "static":  { "V": 0.681, "S": 0.742, "R": 0.655, "A": 0.603 }
}
```

Keep upstream's names. The demo relabels for the public and records the raw
names in Source — `similarity` is shown as **"Equilibrium probability of
observed action"**, never as "Similarity to optimal", because it is neither a
distance nor a cosine.

**Not needed:** raw tracking, match XML, policy `.npz` (the panel's per-command
`path` is what the demo draws), or anything licensed. The payload above is the
same kind of thing as the pass-model coefficients already shared.

## 4. Once it arrives

Implement `EvaluationSource` over the files, set `OFFBALL_EVALUATION_SOURCE`,
re-export, rebuild. The schema slots and their definitions are already in
`demo_viz/paper_story/adapter.py`;
`KYUHYEOK_UPDATE_INTEGRATION.md` has the field-by-field mapping.

## 5. One semantic caveat to carry forward

`clip_metrics` stays empty. `summarize_static.py` aggregates over **moments**
for a corpus summary (mean / median / max, grouped by game kind and by the
showcase set) — that is not a per-clip player score, and the demo will not
average rank, equilibrium probability or regret itself.
