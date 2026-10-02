#!/usr/bin/env python3
"""Build the provisional P×G×A local-game audit for confirmed scenes."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from offball_value.local_game_payoff import (
    LocalGamePayoffConfig,
    build_local_game_payoff_audit,
)
from offball_value.local_game_payoff_audit import render_local_game_payoff_audit
from offball_value.local_game_structure import LocalGameStructureConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenes-json",
        type=Path,
        default=Path(
            "examples/research_audit/manifests/confirmed_core_scenes.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/local_game_payoff_audit_v0_1"),
    )
    parser.add_argument("--release-step-seconds", type=float, default=0.4)
    parser.add_argument("--influence-grid-resolution-m", type=float, default=3.0)
    parser.add_argument(
        "--response-search-mode",
        choices=("target_conditioned", "target_agnostic"),
        default="target_conditioned",
    )
    parser.add_argument(
        "--local-game-selection-mode",
        choices=("response_effect", "pairwise_cross_cost"),
        default="response_effect",
    )
    parser.add_argument(
        "--carry-pressure-model",
        choices=("nearest_defender", "independent_race"),
        default="independent_race",
    )
    parser.add_argument("--candidate-defender-count", type=int, default=3)
    parser.add_argument("--raw-response-limit", type=int, default=160)
    parser.add_argument("--screen-keep-per-option", type=int, default=4)
    parser.add_argument("--screen-keep-minimax", type=int, default=8)
    parser.add_argument("--screen-keep-low-effort", type=int, default=4)
    parser.add_argument(
        "--pursuit-information",
        choices=("kinematic", "clairvoyant"),
        default="kinematic",
        help=(
            "What the pursuing defender may know about where the actor is "
            "going. kinematic extrapolates the speed observed so far and "
            "re-plans each step; clairvoyant reads the actor's real future "
            "and is kept only as an upper reference."
        ),
    )
    parser.add_argument(
        "--vacated-union",
        type=int,
        default=0,
        help=(
            "Union the top-K attackers by vacated-space overlap into each "
            "defender's option catalogue. The catalogue otherwise ranks by "
            "marking release only, which puts an attacker who benefits by "
            "moving into the space the defender left dead last -- he was "
            "never marked, so there is no marking to release. 0 keeps the "
            "historical behaviour."
        ),
    )
    parser.add_argument(
        "--screen-keep-random",
        type=int,
        default=0,
        help=(
            "Extra responses drawn at random from those the screen rejected. "
            "The screen ranks responses on a cheap proxy that agrees with the "
            "real Q only 10%% of the time, so its keeps are correlated in the "
            "same wrong direction; 40 random extras cut the worst-case minimax "
            "error from 40%% to 12%%. Zero reproduces existing artifacts."
        ),
    )
    parser.add_argument(
        "--scene-index",
        type=int,
        action="append",
        help="1-based scene index; repeat to render a subset.",
    )
    parser.add_argument(
        "--include-excluded",
        action="store_true",
        help="Also render scenes the review marked cohort=excluded.",
    )
    parser.add_argument(
        "--include-causal-policy",
        action="store_true",
        help="Price the causal observation-limited policy as a reference.",
    )
    parser.add_argument(
        "--delivery-model",
        choices=("mechanistic", "xpass", "hybrid", "xpass360", "xpass360_kinematic"),
        default="mechanistic",
    )
    parser.add_argument(
        "--value-stack",
        choices=("native", "ssac", "ssac_threat", "ssac_pass",
                 "native_no_access", "ssac_ourcarry", "ssac_keepaccess"),
        default="native",
        help=(
            "'ssac' prices delivery, threat and carry retention with stage 3's "
            "own models, so the ranking and the equilibrium share one scale. "
            "It replaces all three together: swapping one piece already broke "
            "commensurability between pass and carry options."
        ),
    )
    parser.add_argument(
        "--derived-option-source",
        choices=("assignment_rule_v1", "coupled_r9", "payoff"),
        default="assignment_rule_v1",
        help="Beneficiary rule. coupled_r9 did better than R1 in the label check.",
    )
    parser.add_argument("--ssac-tackle-rate", type=float, default=2.2)
    parser.add_argument("--ssac-tackle-radius", type=float, default=1.0)
    parser.add_argument("--ssac-tackle-softness", type=float, default=0.45)
    parser.add_argument(
        "--ssac-pass-model",
        type=str,
        default="andrew/models/experimental_pass.json",
        help="Adopted logistic completion model used by --value-stack ssac.",
    )
    parser.add_argument(
        "--xpass-360-model",
        type=str,
        default="",
        help="Override the defender-aware xPass artefact used by xpass360.",
    )
    parser.add_argument(
        "--delivery-calibration",
        type=str,
        default="",
        help=(
            "Isotonic map from raw pass delivery onto observed completion "
            "(scripts/calibrate_hybrid_delivery.py). Off by default: it fixes "
            "the level but flattens the defender response."
        ),
    )
    parser.add_argument(
        "--through-ball-leads",
        type=str,
        default="2,4,6",
        help="Comma-separated lead metres; empty string disables the templates.",
    )
    parser.add_argument(
        "--cutback-leads",
        type=str,
        default="",
        help="Comma-separated cutback lead metres; empty string disables.",
    )
    parser.add_argument(
        "--record-candidate-grid",
        action="store_true",
        help="Store every release x continuation candidate per cell.",
    )
    parser.add_argument(
        "--goal-danger-grid",
        type=str,
        default="",
        help="EPV grid CSV for G; empty string keeps the geometric proxy.",
    )
    parser.add_argument(
        "--displayed-options",
        type=int,
        default=5,
        help=(
            "Scene-level option catalogue size. The first blind round proved "
            "the threat-ranked top-5 cuts the human's beneficiary (3 of 19 "
            "scenes); 10 effectively lists every attacking outfielder."
        ),
    )
    parser.add_argument(
        "--local-non-runner-limit",
        type=int,
        default=3,
        help="How many non-runner options count as the defender's local set.",
    )
    parser.add_argument(
        "--frozen-response-catalogue",
        type=str,
        default="",
        help=(
            "Audit JSON whose search-candidate defender trajectories are "
            "replayed verbatim instead of searching, so two pricing models "
            "can be compared on identical inputs."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scenes = json.loads(args.scenes_json.read_text(encoding="utf-8"))
    if args.scene_index:
        # Explicit indices refer to the full manifest order and override the
        # cohort filter, so an excluded scene can still be re-audited on demand.
        indices = [index - 1 for index in args.scene_index]
        if any(index < 0 or index >= len(scenes) for index in indices):
            raise ValueError("scene-index is outside the confirmed scene list")
        scenes = [scenes[index] for index in indices]
    elif not args.include_excluded:
        scenes = [
            scene for scene in scenes if scene.get("cohort") != "excluded"
        ]
    config = LocalGamePayoffConfig(
        release_step_seconds=float(args.release_step_seconds),
        influence_grid_resolution_m=float(args.influence_grid_resolution_m),
        response_search_mode=str(args.response_search_mode),
        local_game_selection_mode=str(args.local_game_selection_mode),
        carry_pressure_model=str(args.carry_pressure_model),
        local_non_runner_limit=int(args.local_non_runner_limit),
        goal_danger_grid_path=(str(args.goal_danger_grid) or None),
        frozen_response_catalogue_path=(
            str(args.frozen_response_catalogue) or None
        ),
        include_causal_policy_reference=bool(args.include_causal_policy),
        delivery_model=str(args.delivery_model),
        value_stack=str(args.value_stack),
        ssac_pass_model_path=str(args.ssac_pass_model),
        derived_option_source=str(args.derived_option_source),
        ssac_tackle_rate=float(args.ssac_tackle_rate),
        ssac_tackle_radius_m=float(args.ssac_tackle_radius),
        ssac_tackle_softness_m=float(args.ssac_tackle_softness),
        **(
            {"xpass_360_model_path": str(args.xpass_360_model)}
            if args.xpass_360_model
            else {}
        ),
        **(
            {"delivery_calibration_path": str(args.delivery_calibration)}
            if args.delivery_calibration
            else {}
        ),
        through_ball_lead_distances_m=tuple(
            float(value)
            for value in args.through_ball_leads.split(",")
            if value.strip()
        ),
        cutback_lead_distances_m=tuple(
            float(value)
            for value in args.cutback_leads.split(",")
            if value.strip()
        ),
        record_candidate_grid=bool(args.record_candidate_grid),
        target_agnostic_raw_response_limit=int(args.raw_response_limit),
        response_screen_keep_per_option=int(args.screen_keep_per_option),
        response_screen_keep_minimax=int(args.screen_keep_minimax),
        response_screen_keep_low_effort=int(args.screen_keep_low_effort),
        response_screen_keep_random=int(args.screen_keep_random),
        vacated_union_count=int(args.vacated_union),
    )
    structural_config = LocalGameStructureConfig(
        candidate_defender_count=int(args.candidate_defender_count),
        displayed_option_count=int(args.displayed_options),
        pursuit_information=str(args.pursuit_information),
    )
    games = []
    for index, scene in enumerate(scenes, 1):
        print(
            f"[{index}/{len(scenes)}] {scene['runner_name']} · "
            f"{scene['match_id']}:{scene['onset_frame_id']}",
            flush=True,
        )
        games.append(
            build_local_game_payoff_audit(scene, config, structural_config)
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "local_game_payoff_audits.json"
    html_path = args.output_dir / "local_game_payoff_audit.html"
    summary_path = args.output_dir / "local_game_payoff_summary.csv"
    manifest_path = args.output_dir / "manifest.json"
    json_path.write_text(
        json.dumps(games, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    html_path.write_text(
        render_local_game_payoff_audit(games),
        encoding="utf-8",
    )

    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "match_id",
            "onset_frame_id",
            "runner_name",
            "carrier_name",
            "defender_name",
            "local_option_names",
            "derived_option_name",
            "minimax_response",
            "actual_reselected_option",
            "minimax_reselected_option",
            "actual_worst_q",
            "minimax_worst_q",
            "actual_minus_minimax_q",
            "runner_cross_cost",
            "derived_cross_cost",
            "cross_cost_strength",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for game in games:
            for defender in game["candidate_defenders"]:
                option_by_id = {
                    option["option_id"]: option for option in defender["options"]
                }
                derived = option_by_id.get(defender["derived_option_id"])
                writer.writerow(
                    {
                        "match_id": game["match_id"],
                        "onset_frame_id": game["onset_frame_id"],
                        "runner_name": game["runner_name"],
                        "carrier_name": game["carrier_name"],
                        "defender_name": defender["defender_name"],
                        "local_option_names": " | ".join(
                            option_by_id[option_id]["option_name"]
                            for option_id in defender["local_option_ids"]
                        ),
                        "derived_option_name": (
                            derived["option_name"] if derived else ""
                        ),
                        "minimax_response": defender["minimax_response_label"],
                        "actual_reselected_option": option_by_id[
                            defender["actual_reselected_option_id"]
                        ]["option_name"],
                        "minimax_reselected_option": option_by_id[
                            defender["minimax_reselected_option_id"]
                        ]["option_name"],
                        "actual_worst_q": defender["actual_worst_q"],
                        "minimax_worst_q": defender["minimax_worst_q"],
                        "actual_minus_minimax_q": defender[
                            "actual_minus_minimax_q"
                        ],
                        "runner_cross_cost": defender["cross_cost"]["runner"],
                        "derived_cross_cost": defender["cross_cost"]["derived"],
                        "cross_cost_strength": defender["cross_cost"]["strength"],
                    }
                )

    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": (
                    games[0]["schema_version"]
                    if games
                    else "local-game-payoff-audit-v0.1"
                ),
                "status": (
                    games[0]["status"]
                    if games
                    else "retrospective provisional payoff audit"
                ),
                "scene_count": len(games),
                "defender_game_count": sum(
                    len(game["candidate_defenders"]) for game in games
                ),
                "value_contract": (
                    "delivery_or_path_min_retention × geometric_goal_danger × "
                    "post_success_goal_side_accessibility"
                ),
                "config": games[0]["config"] if games else {},
                "actual_in_optimizer": False,
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    print(f"json: {json_path.resolve()}")
    print(f"html: {html_path.resolve()}")
    print(f"summary: {summary_path.resolve()}")


if __name__ == "__main__":
    main()
