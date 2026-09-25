#!/usr/bin/env python3
"""Run the first end-to-end attacker max--defender min scene experiment."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import time

import pandas as pd

from offball_value.action_value import decompose_threat_by_nearest_attacker
from offball_value.attacker_maximin import (
    TerminalBranchMinimum,
    TerminalThreatResult,
    evaluate_terminal_threat,
    search_terminal_branch_responses,
    search_terminal_best_response,
)
from offball_value.bundesliga import (
    FPS,
    find_bundesliga_files,
    infer_attacking_direction,
    load_bundesliga_frames,
    load_bundesliga_match_metadata,
    normalize_bundesliga_match_id,
)
from offball_value.defender_best_response import (
    DefenderBestResponseConfig,
    build_local_counterfactual_state,
    prepare_observed_background_sequence,
)
from offball_value.defender_response import (
    DefenderResponseConfig,
    generate_defender_response_actions,
    shortlist_relevant_defenders,
)
from offball_value.empirical_action_space import causal_motion_state_from_frames
from offball_value.pass_dynamics import ArrivalModelConfig, estimate_frame_velocities
from offball_value.reference_obso import evaluate_reference_obso
from offball_value.steering_reachable import (
    SteeringEndpointAction,
    SteeringReachabilityConfig,
    generate_hybrid_endpoint_actions,
)


ATTACK_TIMES_S = (0.0, 0.4, 0.8, 1.2, 1.6, 2.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data/raw/bundesliga-integrated")
    )
    parser.add_argument("--match-id", default="J03WOH")
    parser.add_argument("--frame-id", type=int, default=68836)
    parser.add_argument("--player-id", default="DFL-OBJ-00003X")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/attacker_maximin_v0_1"),
    )
    parser.add_argument("--response-delay", type=float, default=0.2)
    parser.add_argument("--relevant-defenders", type=int, default=3)
    parser.add_argument("--obso-batch-size", type=int, default=8)
    parser.add_argument(
        "--attack-screen-limit",
        type=int,
        default=8,
        help=(
            "Number of high fixed-background attack paths receiving exact defender "
            "search in this v0.1 prototype. Use -1 to disable the compute cap."
        ),
    )
    parser.add_argument("--progress-every", type=int, default=200)
    parser.add_argument(
        "--minimum-screen-separation",
        type=float,
        default=3.0,
        help="Minimum endpoint distance between screened attacks before backfill.",
    )
    return parser.parse_args()


def _attack_path(frame, player_id: str, action: SteeringEndpointAction):
    player = frame.players[player_id]
    return ((float(player.x), float(player.y)), *action.motion.path_xy)


def _action_record(action: SteeringEndpointAction) -> dict[str, object]:
    return {
        "action_id": action.action_id,
        "endpoint_x": action.endpoint_x,
        "endpoint_y": action.endpoint_y,
        "endpoint_cell_x": action.endpoint_cell_x,
        "endpoint_cell_y": action.endpoint_cell_y,
        "maneuver_type": action.motion.maneuver_type,
        "terminal_heading_bin": action.motion.terminal_heading_bin,
        "effort_m2ps3": action.motion.effort_m2ps3,
        "path_xy": [list(point) for point in action.motion.path_xy],
    }


def _representative_actions(
    actions: tuple[SteeringEndpointAction, ...],
) -> tuple[SteeringEndpointAction, ...]:
    """Keep the least-effort witness for every endpoint cell."""

    best: dict[tuple[float, float], SteeringEndpointAction] = {}
    for action in actions:
        key = (action.endpoint_cell_x, action.endpoint_cell_y)
        incumbent = best.get(key)
        if incumbent is None or (
            action.motion.effort_m2ps3,
            action.grid_snap_distance_m,
            action.action_id,
        ) < (
            incumbent.motion.effort_m2ps3,
            incumbent.grid_snap_distance_m,
            incumbent.action_id,
        ):
            best[key] = action
    return tuple(sorted(best.values(), key=lambda action: action.action_id))


def _spatially_diverse_screen(
    ranked: list[tuple[SteeringEndpointAction, TerminalThreatResult]],
    limit: int,
    minimum_endpoint_separation_m: float,
) -> list[tuple[SteeringEndpointAction, TerminalThreatResult]]:
    """Select high-value attacks without duplicating one local peak."""

    if limit < 0:
        return ranked
    selected: list[tuple[SteeringEndpointAction, TerminalThreatResult]] = []
    deferred: list[tuple[SteeringEndpointAction, TerminalThreatResult]] = []
    for item in ranked:
        action = item[0]
        if all(
            math.hypot(
                action.endpoint_x - chosen[0].endpoint_x,
                action.endpoint_y - chosen[0].endpoint_y,
            )
            >= minimum_endpoint_separation_m
            for chosen in selected
        ):
            selected.append(item)
            if len(selected) >= limit:
                return selected
        else:
            deferred.append(item)
    for item in deferred:
        if len(selected) >= limit:
            break
        selected.append(item)
    return selected


def _decomposition_payload(
    terminal_background,
    attacker_id,
    attack_path,
    attacking_team_id,
    attacking_direction,
    goalkeeper_ids,
    defender_id=None,
    defender_path=None,
    defender_times=None,
):
    frame, velocities = build_local_counterfactual_state(
        terminal_background,
        attacker_id,
        attack_path,
        ATTACK_TIMES_S,
        defender_id=defender_id,
        defender_path_xy=defender_path,
        defender_path_times_s=defender_times,
    )
    surface = evaluate_reference_obso(
        frame,
        attacking_team_id,
        attacking_direction,
        velocities=velocities,
        goalkeeper_ids=goalkeeper_ids,
        apply_offside=True,
    )
    split = decompose_threat_by_nearest_attacker(
        surface,
        frame,
        attacking_team_id,
        attacker_id,
        attacking_direction,
    )
    return asdict(split)


def main() -> None:
    args = parse_args()
    match_id = normalize_bundesliga_match_id(args.match_id)
    files = find_bundesliga_files(args.data_dir, match_id)
    metadata = load_bundesliga_match_metadata(files["matchinfo"])
    frames = load_bundesliga_frames(
        files["positions"], range(args.frame_id - 10, args.frame_id + 51)
    )
    decision = frames[args.frame_id]
    if args.player_id not in decision.players or decision.ball is None:
        raise ValueError("Decision frame is missing the selected runner or ball")
    history = tuple(
        frames[target] for target in range(args.frame_id - 10, args.frame_id + 1)
    )
    attacking_team_id = decision.players[args.player_id].team_id
    attacking_direction = infer_attacking_direction(
        decision, attacking_team_id, metadata
    )
    goalkeeper_ids = tuple(
        goalkeeper_id
        for team_id in metadata.teams
        if (goalkeeper_id := metadata.goalkeeper_id(team_id)) is not None
    )
    reachability_config = SteeringReachabilityConfig()
    response_config = DefenderResponseConfig(
        response_delay_seconds=args.response_delay,
        relevant_defender_count=args.relevant_defenders,
    )
    motion_state = causal_motion_state_from_frames(
        history, args.player_id, args.frame_id
    )
    started = time.perf_counter()
    raw_attack_set = generate_hybrid_endpoint_actions(
        decision.players[args.player_id],
        decision.ball,
        motion_state,
        empirical_action_set=None,
        config=reachability_config,
    )
    attacks = _representative_actions(raw_attack_set.optimization_actions)
    print(
        f"attack actions: {len(raw_attack_set.optimization_actions)} paths, "
        f"{len(attacks)} endpoint representatives ({time.perf_counter()-started:.1f}s)",
        flush=True,
    )

    background = prepare_observed_background_sequence(
        frames, args.frame_id, DefenderBestResponseConfig()
    )
    terminal_background = background.states[-1]
    decision_velocities = estimate_frame_velocities(
        history,
        args.frame_id,
        ArrivalModelConfig(history_seconds=0.4),
    )
    decision_surface = evaluate_reference_obso(
        decision,
        attacking_team_id,
        attacking_direction,
        velocities=decision_velocities,
        goalkeeper_ids=goalkeeper_ids,
        apply_offside=True,
    )
    decision_split = decompose_threat_by_nearest_attacker(
        decision_surface,
        decision,
        attacking_team_id,
        args.player_id,
        attacking_direction,
    )
    decision_baseline = {
        "time_s": 0.0,
        "team_maximum": decision_surface.maximum,
        "maximum_x": decision_surface.maximum_position[0],
        "maximum_y": decision_surface.maximum_position[1],
        "decomposition": asdict(decision_split),
        "velocity_model": "causal 0.4 s history at run onset",
    }

    # All endpoint representatives receive an exact fixed-background score.
    fixed_rows: list[tuple[SteeringEndpointAction, TerminalThreatResult]] = []
    for index, action in enumerate(attacks, start=1):
        path = _attack_path(decision, args.player_id, action)
        fixed = evaluate_terminal_threat(
            terminal_background,
            args.player_id,
            path,
            ATTACK_TIMES_S,
            attacking_team_id,
            attacking_direction,
            goalkeeper_ids,
            branch_id="observed-background-reference",
            obso_batch_size=args.obso_batch_size,
        )
        fixed_rows.append((action, fixed))
        if index % 50 == 0 or index == len(attacks):
            print(f"fixed-background attacks evaluated: {index}/{len(attacks)}", flush=True)
    fixed_rows.sort(
        key=lambda item: (
            -item[1].value,
            item[0].motion.effort_m2ps3,
            item[0].action_id,
        )
    )

    # Defender action sets are candidate-independent and generated once.
    defender_sets = {}
    for defender_id, player in decision.players.items():
        if player.team_id == attacking_team_id or defender_id in goalkeeper_ids:
            continue
        print(
            f"generating defender responses: {metadata.players[defender_id].short_name}",
            flush=True,
        )
        defender_sets[defender_id] = generate_defender_response_actions(
            decision,
            history,
            defender_id,
            response_config,
            reachability_config,
        )

    attack_results: list[dict[str, object]] = []
    cap_applied = args.attack_screen_limit >= 0
    search_rows = _spatially_diverse_screen(
        fixed_rows,
        args.attack_screen_limit,
        args.minimum_screen_separation,
    )
    for attack_rank, (action, fixed) in enumerate(search_rows, start=1):
        fixed_rank = 1 + next(
            index
            for index, (candidate, _) in enumerate(fixed_rows)
            if candidate.action_id == action.action_id
        )
        path = _attack_path(decision, args.player_id, action)
        relevant = shortlist_relevant_defenders(
            decision,
            history,
            attacking_team_id,
            args.player_id,
            path,
            ATTACK_TIMES_S,
            goalkeeper_ids,
            response_config,
            reachability_config,
            defender_sets,
        )
        selected_sets = {
            defender.defender_id: defender_sets[defender.defender_id]
            for defender in relevant
        }
        # Seed the minimization with the feasible action whose terminal point
        # is closest to the attacker endpoint; this accelerates safe pruning.
        seed_action = min(
            (
                response
                for action_set in selected_sets.values()
                for response in action_set.actions
            ),
            key=lambda response: (
                math.hypot(
                    response.endpoint_x - action.endpoint_x,
                    response.endpoint_y - action.endpoint_y,
                ),
                response.base_action.motion.effort_m2ps3,
                response.action_id,
            ),
        )
        seed = evaluate_terminal_threat(
            terminal_background,
            args.player_id,
            path,
            ATTACK_TIMES_S,
            attacking_team_id,
            attacking_direction,
            goalkeeper_ids,
            branch_id=seed_action.action_id,
            defender_id=seed_action.defender_id,
            defender_path_xy=seed_action.full_path_xy,
            defender_path_times_s=seed_action.response_path_times_s,
            defender_action_id=seed_action.action_id,
            defender_effort_m2ps3=seed_action.base_action.motion.effort_m2ps3,
            obso_batch_size=args.obso_batch_size,
        )
        print(
            f"[{attack_rank}/{len(search_rows)}] defender min for attack "
            f"({action.endpoint_cell_x:.1f}, {action.endpoint_cell_y:.1f}); "
            f"fixed={fixed.value:.6f}, seed={seed.value:.6f}",
            flush=True,
        )
        search_started = time.perf_counter()
        response = search_terminal_best_response(
            terminal_background,
            args.player_id,
            path,
            ATTACK_TIMES_S,
            attacking_team_id,
            attacking_direction,
            goalkeeper_ids,
            selected_sets,
            fixed_background=fixed,
            initial_feasible_response=seed,
            obso_batch_size=args.obso_batch_size,
            progress_every=args.progress_every,
        )
        best_action = next(
            candidate
            for candidate in selected_sets[response.best.defender_id].actions
            if candidate.action_id == response.best.defender_action_id
        )
        fixed_decomposition = _decomposition_payload(
            terminal_background,
            args.player_id,
            path,
            attacking_team_id,
            attacking_direction,
            goalkeeper_ids,
        )
        best_response_decomposition = _decomposition_payload(
            terminal_background,
            args.player_id,
            path,
            attacking_team_id,
            attacking_direction,
            goalkeeper_ids,
            defender_id=response.best.defender_id,
            defender_path=best_action.full_path_xy,
            defender_times=best_action.response_path_times_s,
        )
        branch_response = search_terminal_branch_responses(
            terminal_background,
            args.player_id,
            path,
            ATTACK_TIMES_S,
            attacking_team_id,
            attacking_direction,
            goalkeeper_ids,
            selected_sets,
            initial_direct_cover=TerminalBranchMinimum(
                branch="R",
                value=best_response_decomposition["focal_maximum"],
                maximum_x=best_response_decomposition["focal_maximum_x"],
                maximum_y=best_response_decomposition["focal_maximum_y"],
                evaluated_cell_count=0,
                defender_id=response.best.defender_id,
                defender_action_id=response.best.defender_action_id,
                defender_effort_m2ps3=response.best.defender_effort_m2ps3,
            ),
            initial_other_cover=TerminalBranchMinimum(
                branch="O",
                value=best_response_decomposition["other_maximum"],
                maximum_x=best_response_decomposition["other_maximum_x"],
                maximum_y=best_response_decomposition["other_maximum_y"],
                evaluated_cell_count=0,
                defender_id=response.best.defender_id,
                defender_action_id=response.best.defender_action_id,
                defender_effort_m2ps3=response.best.defender_effort_m2ps3,
            ),
            obso_batch_size=args.obso_batch_size,
            progress_every=args.progress_every,
        )

        def branch_response_payload(selected: TerminalBranchMinimum):
            selected_action = next(
                candidate
                for candidate in selected_sets[selected.defender_id].actions
                if candidate.action_id == selected.defender_action_id
            )
            decomposition = _decomposition_payload(
                terminal_background,
                args.player_id,
                path,
                attacking_team_id,
                attacking_direction,
                goalkeeper_ids,
                defender_id=selected.defender_id,
                defender_path=selected_action.full_path_xy,
                defender_times=selected_action.response_path_times_s,
            )
            return {
                "selection": asdict(selected),
                "defender_name": metadata.players[selected.defender_id].short_name,
                "defender_path_times_s": list(selected_action.response_path_times_s),
                "defender_path_xy": [list(point) for point in selected_action.full_path_xy],
                "decomposition": decomposition,
            }

        attack_results.append(
            {
                "fixed_rank": fixed_rank,
                "attack": _action_record(action),
                "attack_path_xy": [list(point) for point in path],
                "fixed": asdict(fixed),
                "best_response": asdict(response.best),
                "best_defender_name": metadata.players[
                    response.best.defender_id
                ].short_name,
                "best_defender_path_times_s": list(
                    best_action.response_path_times_s
                ),
                "best_defender_path_xy": [
                    list(point) for point in best_action.full_path_xy
                ],
                "fixed_decomposition": fixed_decomposition,
                "best_response_decomposition": best_response_decomposition,
                "dilemma_responses": {
                    "direct_cover": branch_response_payload(
                        branch_response.direct_cover
                    ),
                    "other_cover": branch_response_payload(
                        branch_response.other_cover
                    ),
                    "global_best": {
                        "selection": {
                            "branch": "max(R,O)",
                            "value": response.best.value,
                            "maximum_x": response.best.maximum_x,
                            "maximum_y": response.best.maximum_y,
                            "evaluated_cell_count": response.best.evaluated_cell_count,
                            "defender_id": response.best.defender_id,
                            "defender_action_id": response.best.defender_action_id,
                            "defender_effort_m2ps3": response.best.defender_effort_m2ps3,
                        },
                        "defender_name": metadata.players[
                            response.best.defender_id
                        ].short_name,
                        "defender_path_times_s": list(
                            best_action.response_path_times_s
                        ),
                        "defender_path_xy": [
                            list(point) for point in best_action.full_path_xy
                        ],
                        "decomposition": best_response_decomposition,
                    },
                },
                "branch_response_evaluated_count": (
                    branch_response.evaluated_response_count
                ),
                "fixed_gain_from_onset": (
                    fixed.value - decision_baseline["team_maximum"]
                ),
                "robust_gain_from_onset": (
                    response.best.value - decision_baseline["team_maximum"]
                ),
                "relevant_defenders": [
                    {
                        **asdict(defender),
                        "defender_name": metadata.players[
                            defender.defender_id
                        ].short_name,
                    }
                    for defender in relevant
                ],
                "evaluated_response_count": response.evaluated_response_count,
                "evaluation_seconds": time.perf_counter() - search_started,
            }
        )
        print(
            f"robust={response.best.value:.6f} via "
            f"{metadata.players[response.best.defender_id].short_name}",
            flush=True,
        )

    if not attack_results:
        raise RuntimeError("No attack candidate received a defender search")
    winner = max(
        attack_results,
        key=lambda item: (
            item["best_response"]["value"],
            -item["attack"]["effort_m2ps3"],
            item["attack"]["action_id"],
        ),
    )
    winner["robust_rank"] = 1
    observed_path = [
        [
            (target - args.frame_id) / FPS,
            frames[target].players[args.player_id].x,
            frames[target].players[args.player_id].y,
        ]
        for target in range(args.frame_id, args.frame_id + 51)
        if args.player_id in frames[target].players
    ]
    output = {
        "schema_version": "attacker-maximin-v0.1",
        "status": "screened_prototype" if cap_applied else "full_outer_search",
        "match_id": match_id,
        "frame_id": args.frame_id,
        "attacker_id": args.player_id,
        "attacker_name": metadata.players[args.player_id].short_name,
        "attacking_team_id": attacking_team_id,
        "attacking_direction": attacking_direction,
        "objective": "max_attack min_one_defender max_pitch OBSO at t+2s",
        "background_model": "observed future for ball and non-intervened players",
        "decision_baseline": decision_baseline,
        "attack_path_times_s": list(ATTACK_TIMES_S),
        "raw_attack_action_count": len(raw_attack_set.optimization_actions),
        "unique_endpoint_count": len(attacks),
        "fixed_background_evaluated_count": len(fixed_rows),
        "defender_searched_attack_count": len(attack_results),
        "attack_screen_limit": args.attack_screen_limit,
        "minimum_screen_separation_m": args.minimum_screen_separation,
        "response_delay_seconds": args.response_delay,
        "relevant_defender_count": args.relevant_defenders,
        "winner": winner,
        "searched_attacks": attack_results,
        "fixed_attack_landscape": [
            {**_action_record(action), "fixed_value": result.value}
            for action, result in fixed_rows
        ],
        "observed_attacker_timed": observed_path,
        "background_frames": [
            {
                "time_s": (target - args.frame_id) / FPS,
                "players": [
                    [player_id, state.team_id, state.x, state.y]
                    for player_id, state in frames[target].players.items()
                ],
                "ball": [frames[target].ball.x, frames[target].ball.y],
            }
            for target in range(args.frame_id, args.frame_id + 51)
        ],
        "parameters": {
            "reachability": asdict(reachability_config),
            "defender_response": asdict(response_config),
            "obso_batch_size": args.obso_batch_size,
        },
    }
    output_dir = args.output_dir / match_id
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"frame_{args.frame_id}_attacker_maximin.json"
    csv_path = output_dir / f"frame_{args.frame_id}_attacker_maximin.csv"
    json_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame(
        [
            {
                "match_id": match_id,
                "frame_id": args.frame_id,
                "attacker_id": args.player_id,
                "fixed_rank": item["fixed_rank"],
                "endpoint_x": item["attack"]["endpoint_x"],
                "endpoint_y": item["attack"]["endpoint_y"],
                "fixed_value": item["fixed"]["value"],
                "robust_value": item["best_response"]["value"],
                "onset_value": decision_baseline["team_maximum"],
                "fixed_gain_from_onset": item["fixed_gain_from_onset"],
                "robust_gain_from_onset": item["robust_gain_from_onset"],
                "best_defender_id": item["best_response"]["defender_id"],
                "best_defender_name": item["best_defender_name"],
                "evaluated_response_count": item["evaluated_response_count"],
                "evaluation_seconds": item["evaluation_seconds"],
            }
            for item in attack_results
        ]
    ).to_csv(csv_path, index=False)
    print(f"result: {json_path}")
    print(f"table:  {csv_path}")


if __name__ == "__main__":
    main()
