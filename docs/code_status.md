# Code status

Which code produces the paper's results, which supports it, and which is left from earlier stages of
the project. Compiled on 2026-10-01 from the import graph and the jobs' commands. The tree before
this audit is commit `33378ac`; the tree before the first cleanup (199 scripts, docs, examples, deploy
files) is tagged `pre-cleanup-2026-09-30`.

Nothing listed here was moved or deleted. The demo (`demo_viz/`) imports some of the legacy modules
and refers to the `andrew-*` folders by path, so renaming or removing them would break it. That
decision is left for after the submission.

## Paper path

| Role | Code |
| --- | --- |
| 2v1 game (ball carrier and runner against the runner's defender) | `andrew-passer2on1/passer2on1/`, run by `andrew-passer2on1/scripts/run.py`; design notes in `andrew-passer2on1/DESIGN.md` |
| 3v1 game (runner and beneficiary against the defender, scripted passer) | `andrew-fixedpasser/fixedpasser/`, run by `andrew-fixedpasser/scripts/run.py`; `andrew-fixedpasser/DESIGN.md` |
| Solver base (imported, not in this repository) | `defensive_positioning` @ `e8b0a95`, see [reproduction.md](reproduction.md#solver-base) |
| Solver jobs | `jobs/solve_figure2_panels.sbatch`, `jobs/solve_defender_grid.sbatch`, `jobs/solve_evaluation.sbatch`, `jobs/static_counterfactual.sbatch` |
| Scene and state construction | `scripts/extract_settled_possession_run_onsets.py`, `render_local_game_payoff_audit_v0_1.py`, `apply_pair_gate.py`, `build_stage3_states.py`, `filter_stage3_states.py`, `build_showcase_states.py`, `build_eval_states.py`, `build_defender_grid.py` |
| Scene selection (needs team data) | `scripts/map_chani_annotations.py`, `trace_chani_scenes.py`, `build_rating_package.py`, `merge_ratings.py`, `export_showcase.py` |
| Pass model and limits | `scripts/build_pass_dataset.py`, `fit_pass_candidates.py`, `fit_pass_sym.py`, `measure_accelerations.py` |
| Reading solver output | `scripts/extract_panel_policy.py`, `extract_defender_grid.py`, `check_defender_grid.py` |
| Evaluation | `scripts/analyze_eval.py`, `static_counterfactual.py`, `summarize_static.py` |
| Figures | `scripts/render_figure1_dilemma.py`, `render_figure2_abstract.py`, `add_caption.py`; imported by them: `figure_style.py`, `render_panel_figure.py`, `render_defender_flow_moments.py` |

The 34 `src/offball_value` modules these scripts import, directly or through each other:
`agile_motion`, `assignment_rule`, `bundesliga`, `carry_dynamics`, `causal_defender_policy`,
`coupled_beneficiary`, `defender_trajectory_search`, `delivery_calibration`, `dynamic_marking`,
`empirical_action_space`, `fernandez_influence`, `goal_weighted_influence`, `kinematic_xpass`,
`local_game_payoff`, `local_game_payoff_audit`, `local_game_structure`, `observed_passes`, `obso`,
`pair_plausibility`, `pass_dynamics`, `physics_pass`, `post_reception_value`, `r9_selection`,
`run_onset`, `run_passes`, `settled_possession_phase`, `shot_context`, `shot_context_onset`,
`shot_context_onset_audit`, `ssac_stack`, `stage3_read`, `steering_reachable`, `vacated_space`,
`xpass`; and `baseline`, `possession`, `schema`, which the package imports on load.

`agile_motion.py`, `physics_pass.py` and `run_passes.py` exist in three copies (`src/offball_value/`
and both solver packages). The three copies of each are byte-identical (checked 2026-10-01); the
solver jobs import the packages' copies.

## Supporting scripts

Not part of the solver path. Most fit or check a constant or rule that a `src/offball_value` module
cites by script name. `calibrate_hybrid_delivery.py`, `train_xpass_360_kinematic.py` (delivery and
xPass models, cited by `local_game_payoff` and `delivery_calibration`; the second needs the `fit`
extra), `train_kinematic_delivery.py` (an alternative delivery model, cited nowhere),
`measure_carrying_speed.py`, `measure_carry_speed_paired.py` (carrying speed),
`predict_assignment_rules.py`, `score_coupled_beneficiary.py` (beneficiary assignment),
`compile_run_onset_reviews.py` (run-onset review cohorts, unit-tested), `diagnose_missed_onsets.py`
(why the onset detector missed annotated runs; it also provides the spreadsheet reader for
`build_rating_package.py` and `export_showcase.py`), `explore_progression_goal_danger.py` (a prototype
cited by `local_game_payoff`).

## Legacy modules

Not imported by any pipeline script, job or paper-path module. They come from earlier formulations
(the OBSO-style proxy, the pass-window value, the defender best-response and attacker max-min
prototypes, the structural audits). Kept because the demo uses some of them (`reference_obso`,
`action_space`).

`action_space`, `action_value`, `adapters`, `animated_audit`, `attacker_maximin`,
`attacker_trajectory`, `beneficiary_selection`, `blind_derived_review`, `causal_attribution`,
`counterfactual_state`, `defender_best_response`, `defender_response`, `direct_derived_response`,
`dynamic_reachable`, `endpoint_audit`, `geometric_dilemma`, `influence_dilemma`, `lane_kinematics`,
`loaders`, `marginal_assignment`, `pass_window_search`, `pass_window_value`, `point_value`,
`reference_obso`, `scene_audit`, `scene_extractor`, `shot_context_audit`, `vacated_space_rule`.

Used only by their unit tests: `background_rollout`,
`clear_core_scene_audit`, `dynamic_response_game`, `local_game_structure_audit`,
`meeting_scene_gallery`.

## Known loose ends

- `andrew-*/scripts/run.py` docstrings mention `scripts/run_stage3.py` and `scripts/summarise_stage3.py`,
  which were removed in the first cleanup. The solver code was left unchanged.
- Panel files written before commit `e89e78f` carry the solver's Korean command names; the renderers
  accept only the English names. The mapping is `NAMES` in `scripts/render_panel_figure.py` at tag
  `pre-cleanup-2026-09-30`.
- The rating and start sheets label games "2대1" / "3대1"; `build_eval_states.py` and the readers
  match these strings verbatim (`GAME_2V1`, `GAME_3V1`).
- Generated output and restricted inputs: none are tracked. `.gitignore` covers `data/raw/`,
  `data/processed/` except the model files, `out/`, `logs/`, `andrew/`, `chani/`, `local_inputs/`.
