#!/usr/bin/env python3
"""Evaluate the full direct--derived landscape of feasible defender endpoints."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import time

import numpy as np
import pandas as pd

from offball_value.bundesliga import (
    FIELD_LENGTH,
    FIELD_WIDTH,
    FPS,
    find_bundesliga_files,
    load_bundesliga_frames,
    load_bundesliga_match_metadata,
    normalize_bundesliga_match_id,
)
from offball_value.defender_best_response import (
    DefenderBestResponseConfig,
    ObservedBackgroundSequence,
    ObservedBackgroundState,
    interpolate_path_state,
    prepare_observed_background_sequence,
)
from offball_value.defender_response import (
    DefenderResponseConfig,
    generate_defender_response_actions,
)
from offball_value.direct_derived_response import (
    AdaptiveHorizonConfig,
    choose_adaptive_horizon,
    pareto_frontier_indices,
)
from offball_value.fernandez_influence import fernandez_influence_surface
from offball_value.goal_weighted_influence import (
    GoalWeightedInfluenceConfig,
    goal_weighted_space_surface,
    influence_pitch_grid,
)
from offball_value.influence_dilemma import analyze_pair_tradeoff
from offball_value.pass_dynamics import (
    ArrivalModelConfig,
    VelocityEstimate,
    estimate_frame_velocities,
)
from offball_value.steering_reachable import SteeringReachabilityConfig


DEFAULT_SCENES = (
    "DFL-MAT-J03WOH:53844",  # clear positive
    "DFL-MAT-J03WOY:24414",  # possible / borderline
    "DFL-MAT-J03WQQ:73928",  # reviewed none
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audit-csv",
        type=Path,
        default=Path(
            "data/processed/shot_context_run_onset_v0_1/audit_selection.csv"
        ),
    )
    parser.add_argument(
        "--review-csv",
        type=Path,
        default=Path(
            "/Users/kyuhyeokseo/Downloads/shot_context_onset_v0_1_reviews.csv"
        ),
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data/raw/bundesliga-integrated")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/direct_derived_response_map_v0_1"),
    )
    parser.add_argument(
        "--scene",
        action="append",
        help="MATCH:ONSET_FRAME. Defaults to one positive, borderline and negative.",
    )
    parser.add_argument("--max-horizon-seconds", type=float, default=3.2)
    parser.add_argument("--pre-onset-seconds", type=float, default=1.2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--progress-every", type=int, default=100)
    return parser.parse_args()


def _velocity(velocities, player_id):
    value = velocities.get(player_id)
    return (float(value.vx), float(value.vy)) if value is not None else (0.0, 0.0)


def _surface(frame, velocities, player_id, xgrid, ygrid, config):
    player = frame.players[player_id]
    return fernandez_influence_surface(
        xgrid,
        ygrid,
        (float(player.x), float(player.y)),
        _velocity(velocities, player_id),
        (float(frame.ball.x), float(frame.ball.y)),
        config.maximum_influence_speed_mps,
    )


def _mean_axis(values: np.ndarray, times: np.ndarray) -> np.ndarray:
    if values.shape[-1] == 1:
        return values[..., 0]
    return np.trapezoid(values, times, axis=-1) / (times[-1] - times[0])


def _nearest_controller(frame, maximum_distance_m=1.5):
    if frame.ball is None or (frame.ball.z is not None and frame.ball.z > 1.5):
        return None, math.inf
    candidates = [
        (math.hypot(player.x - frame.ball.x, player.y - frame.ball.y), player_id)
        for player_id, player in frame.players.items()
    ]
    if not candidates:
        return None, math.inf
    distance, player_id = min(candidates)
    return (player_id if distance <= maximum_distance_m else None), float(distance)


def _observed_termination(
    frame_map,
    onset_frame_id,
    maximum_seconds,
    runner_id,
    attacking_team_id,
):
    """Find an observed event that ends the off-ball diagnostic window."""

    opponent_streak = 0
    runner_streak = 0
    final = onset_frame_id + int(math.floor(maximum_seconds * FPS))
    for frame_id in range(onset_frame_id + 1, final + 1):
        frame = frame_map.get(frame_id)
        if frame is None or frame.ball is None:
            return (frame_id - onset_frame_id) / FPS, "tracking_or_ball_missing"
        if abs(frame.ball.x) > FIELD_LENGTH / 2.0 + 0.2 or abs(frame.ball.y) > FIELD_WIDTH / 2.0 + 0.2:
            return (frame_id - onset_frame_id) / FPS, "ball_outside_pitch"
        controller, _ = _nearest_controller(frame)
        if controller is not None and frame.players[controller].team_id != attacking_team_id:
            opponent_streak += 1
        else:
            opponent_streak = 0
        if controller == runner_id and (frame_id - onset_frame_id) / FPS >= 0.4:
            runner_streak += 1
        else:
            runner_streak = 0
        if opponent_streak >= 6:
            start = frame_id - opponent_streak + 1
            return (start - onset_frame_id) / FPS, "opponent_control"
        if runner_streak >= 3:
            start = frame_id - runner_streak + 1
            return (start - onset_frame_id) / FPS, "runner_reception"
    return None, None


def _unique_endpoint_actions(actions):
    """Keep one minimum-effort feasible path for every 1 m endpoint cell."""

    selected = {}
    for action in actions:
        key = (
            float(action.base_action.endpoint_cell_x),
            float(action.base_action.endpoint_cell_y),
        )
        incumbent = selected.get(key)
        action_key = (
            float(action.base_action.motion.effort_m2ps3),
            action.action_id,
        )
        if incumbent is None or action_key < (
            float(incumbent.base_action.motion.effort_m2ps3),
            incumbent.action_id,
        ):
            selected[key] = action
    return tuple(selected[key] for key in sorted(selected))


def _prepare_components(
    background,
    attacker_ids,
    defender_ids,
    direction,
    xgrid,
    ygrid,
    influence_config,
):
    components = []
    observed = np.zeros((len(attacker_ids), len(background.states)), dtype=float)
    for time_index, state in enumerate(background.states):
        frame = state.frame
        velocities = dict(state.velocities)
        defender_surfaces = {
            player_id: _surface(
                frame, velocities, player_id, xgrid, ygrid, influence_config
            )
            for player_id in defender_ids
        }
        defender_total = np.sum(list(defender_surfaces.values()), axis=0)
        intrinsic = []
        for attacker_id in attacker_ids:
            attacker = frame.players[attacker_id]
            intrinsic.append(
                _surface(
                    frame, velocities, attacker_id, xgrid, ygrid, influence_config
                )
                * goal_weighted_space_surface(
                    xgrid,
                    ygrid,
                    (float(attacker.x), float(attacker.y)),
                    direction,
                    influence_config,
                )
            )
        intrinsic = np.asarray(intrinsic, dtype=float)
        denominators = np.maximum(np.sum(intrinsic, axis=(1, 2)), 1e-12)
        uncovered = np.exp(
            -influence_config.defender_suppression_strength * defender_total
        )
        observed[:, time_index] = (
            np.sum(intrinsic * uncovered[None, :, :], axis=(1, 2)) / denominators
        )
        components.append(
            {
                "state": state,
                "xgrid": xgrid,
                "ygrid": ygrid,
                "intrinsic": intrinsic,
                "denominators": denominators,
                "defender_surfaces": defender_surfaces,
                "defender_total": defender_total,
                "suppression_strength": influence_config.defender_suppression_strength,
            }
        )
    return components, observed


def _rank_defenders(
    components,
    observed,
    attacker_ids,
    runner_id,
    candidate_defender_ids,
    times,
):
    runner_index = attacker_ids.index(runner_id)
    other_indices = [index for index in range(len(attacker_ids)) if index != runner_index]
    base_horizon = _mean_axis(observed, times)
    rows = []
    for defender_id in candidate_defender_ids:
        removed = np.zeros_like(observed)
        for time_index, component in enumerate(components):
            reduced = (
                component["defender_total"]
                - component["defender_surfaces"][defender_id]
            )
            uncovered = np.exp(-component["suppression_strength"] * reduced)
            removed[:, time_index] = (
                np.sum(
                    component["intrinsic"] * uncovered[None, :, :],
                    axis=(1, 2),
                )
                / component["denominators"]
            )
        removed_horizon = _mean_axis(removed, times)
        impact = np.maximum(0.0, removed_horizon - base_horizon)
        rows.append(
            {
                "defender_id": defender_id,
                "direct_responsibility": float(impact[runner_index]),
                "derived_responsibility": float(
                    np.max(impact[other_indices]) if other_indices else 0.0
                ),
            }
        )
    direct_scale = max((row["direct_responsibility"] for row in rows), default=0.0)
    derived_scale = max((row["derived_responsibility"] for row in rows), default=0.0)
    for row in rows:
        direct = row["direct_responsibility"] / max(direct_scale, 1e-12)
        derived = row["derived_responsibility"] / max(derived_scale, 1e-12)
        row["joint_responsibility_score"] = float(min(direct, derived) + 0.25 * max(direct, derived))
    rows.sort(
        key=lambda row: (
            -row["joint_responsibility_score"],
            -row["direct_responsibility"],
            -row["derived_responsibility"],
            row["defender_id"],
        )
    )
    return rows


def _score_actions(
    actions,
    defender_id,
    components,
    attacker_ids,
    influence_config,
    batch_size,
    progress_every,
):
    values = np.zeros(
        (len(attacker_ids), len(actions), len(components)), dtype=np.float32
    )
    for time_index, component in enumerate(components):
        frame = component["state"].frame
        velocities = dict(component["state"].velocities)
        fixed = (
            component["defender_total"]
            - component["defender_surfaces"][defender_id]
        )
        intrinsic_flat = component["intrinsic"].reshape(len(attacker_ids), -1)
        for start in range(0, len(actions), batch_size):
            batch = actions[start : start + batch_size]
            uncovered = []
            for action in batch:
                x, y, vx, vy = interpolate_path_state(
                    action.full_path_xy,
                    action.response_path_times_s,
                    component["state"].time_s,
                )
                virtual_frame = frame.with_player(
                    defender_id,
                    replace(
                        frame.players[defender_id],
                        x=x,
                        y=y,
                        speed=math.hypot(vx, vy) * 3.6,
                    ),
                )
                virtual_velocities = dict(velocities)
                virtual_velocities[defender_id] = VelocityEstimate(
                    vx, vy, math.hypot(vx, vy), 0, 0.0
                )
                virtual = _surface(
                    virtual_frame,
                    virtual_velocities,
                    defender_id,
                    component["xgrid"],
                    component["ygrid"],
                    influence_config,
                )
                uncovered.append(
                    np.exp(
                        -influence_config.defender_suppression_strength
                        * (fixed + virtual)
                    ).reshape(-1)
                )
            uncovered_array = np.asarray(uncovered, dtype=float)
            scores = (
                uncovered_array @ intrinsic_flat.T
            ) / component["denominators"][None, :]
            values[:, start : start + len(batch), time_index] = scores.T
            if progress_every > 0 and (start + len(batch)) % progress_every < batch_size:
                print(
                    f"  t={component['state'].time_s:.1f}s · "
                    f"{start + len(batch)}/{len(actions)} endpoints",
                    flush=True,
                )
    return values


def _action_payload(action, index, direct, derived, beneficiary_id, pareto, regret, direct_trace, derived_trace):
    return {
        "index": int(index),
        "action_id": action.action_id,
        "endpoint_x": float(action.endpoint_x),
        "endpoint_y": float(action.endpoint_y),
        "times_s": [float(value) for value in action.response_path_times_s],
        "path_xy": [[float(x), float(y)] for x, y in action.full_path_xy],
        "effort_m2ps3": float(action.base_action.motion.effort_m2ps3),
        "direct_threat": float(direct),
        "derived_threat": float(derived),
        "beneficiary_id": beneficiary_id,
        "pareto": bool(pareto),
        "normalized_compromise_regret": float(regret),
        "direct_trace": [float(value) for value in direct_trace],
        "derived_trace": [float(value) for value in derived_trace],
    }


def _prepare_scene(row, review, frame_map, metadata, args):
    onset = int(row.frame_id)
    runner_id = str(row.player_id)
    team_id = str(row.team_id)
    direction = int(row.attacking_direction)
    seconds_to_shot = (int(row.shot_frame_id) - onset) / FPS
    observed_end, observed_reason = _observed_termination(
        frame_map,
        onset,
        min(args.max_horizon_seconds, seconds_to_shot),
        runner_id,
        team_id,
    )
    horizon = choose_adaptive_horizon(
        seconds_to_shot,
        observed_termination_seconds=observed_end,
        config=AdaptiveHorizonConfig(maximum_seconds=args.max_horizon_seconds),
    )
    times = np.asarray(horizon.evaluation_times_s, dtype=float)
    background = prepare_observed_background_sequence(
        frame_map,
        onset,
        DefenderBestResponseConfig(evaluation_times_s=horizon.evaluation_times_s),
    )
    attacking_goalkeeper = metadata.goalkeeper_id(team_id)
    attacker_ids = tuple(
        sorted(
            player_id
            for player_id, player in frame_map[onset].players.items()
            if player.team_id == team_id and player_id != attacking_goalkeeper
        )
    )
    defending_team_ids = {
        player.team_id
        for player in frame_map[onset].players.values()
        if player.team_id != team_id
    }
    defending_goalkeepers = {
        goalkeeper_id
        for defender_team_id in defending_team_ids
        if (goalkeeper_id := metadata.goalkeeper_id(defender_team_id)) is not None
    }
    defender_ids = tuple(
        sorted(
            player_id
            for player_id, player in frame_map[onset].players.items()
            if player.team_id != team_id
        )
    )
    candidate_defender_ids = tuple(
        player_id for player_id in defender_ids if player_id not in defending_goalkeepers
    )
    influence_config = GoalWeightedInfluenceConfig(grid_resolution_m=1.0)
    xgrid, ygrid = influence_pitch_grid(influence_config)
    components, observed = _prepare_components(
        background,
        attacker_ids,
        defender_ids,
        direction,
        xgrid,
        ygrid,
        influence_config,
    )
    # Relevant-defender selection is anchored at t=0 and uses only pre-onset
    # velocity history. Observed future is reserved for retrospective scoring,
    # not for choosing who is allowed to respond.
    decision_velocities = estimate_frame_velocities(
        frame_map.values(),
        onset,
        ArrivalModelConfig(history_seconds=0.4),
    )
    decision_background = ObservedBackgroundSequence(
        decision_frame_id=onset,
        states=(
            ObservedBackgroundState(
                time_s=0.0,
                frame=frame_map[onset],
                velocities=decision_velocities,
            ),
        ),
    )
    decision_components, decision_observed = _prepare_components(
        decision_background,
        attacker_ids,
        defender_ids,
        direction,
        xgrid,
        ygrid,
        influence_config,
    )
    rankings = _rank_defenders(
        decision_components,
        decision_observed,
        attacker_ids,
        runner_id,
        candidate_defender_ids,
        np.asarray((0.0,), dtype=float),
    )
    defender_id = str(rankings[0]["defender_id"])
    history = tuple(frame_map[index] for index in range(onset - 10, onset + 1))
    response_config = DefenderResponseConfig(
        horizon_seconds=horizon.seconds,
        response_delay_seconds=0.2,
    )
    reachability_config = SteeringReachabilityConfig(
        state_position_resolution_m=1.5,
        state_speed_resolution_mps=1.0,
        state_heading_bins=48,
        control_direction_count=12,
        plant_cut_direction_count=12,
        variants_per_endpoint=1,
    )
    generated = generate_defender_response_actions(
        frame_map[onset],
        history,
        defender_id,
        response_config,
        reachability_config,
    )
    actions = _unique_endpoint_actions(generated.actions)
    values = _score_actions(
        actions,
        defender_id,
        components,
        attacker_ids,
        influence_config,
        args.batch_size,
        args.progress_every,
    )
    horizon_values = _mean_axis(values, times)
    runner_index = attacker_ids.index(runner_id)
    other_indices = [index for index in range(len(attacker_ids)) if index != runner_index]
    direct = np.asarray(horizon_values[runner_index], dtype=float)
    derived_matrix = horizon_values[other_indices]
    beneficiary_rows = np.argmax(derived_matrix, axis=0)
    derived = np.max(derived_matrix, axis=0)
    beneficiary_ids = [attacker_ids[other_indices[index]] for index in beneficiary_rows]
    tradeoff = analyze_pair_tradeoff(direct, derived)
    pareto = set(pareto_frontier_indices(direct, derived))
    direct_span = max(
        direct[tradeoff.beneficiary_action_index]
        - direct[tradeoff.direct_action_index],
        1e-12,
    )
    derived_span = max(
        derived[tradeoff.direct_action_index]
        - derived[tradeoff.beneficiary_action_index],
        1e-12,
    )
    regret = np.maximum(
        np.maximum(0.0, direct - direct[tradeoff.direct_action_index]) / direct_span,
        np.maximum(0.0, derived - derived[tradeoff.beneficiary_action_index])
        / derived_span,
    )
    derived_trace = np.max(values[other_indices], axis=0)
    response_payload = [
        _action_payload(
            action,
            index,
            direct[index],
            derived[index],
            beneficiary_ids[index],
            index in pareto,
            regret[index],
            values[runner_index, index],
            derived_trace[index],
        )
        for index, action in enumerate(actions)
    ]
    observed_horizon = _mean_axis(observed, times)
    observed_derived_row = int(np.argmax(observed_horizon[other_indices]))
    observed_beneficiary_id = attacker_ids[other_indices[observed_derived_row]]
    observed_direct = float(observed_horizon[runner_index])
    observed_derived = float(observed_horizon[other_indices[observed_derived_row]])
    dense_start = onset - int(round(args.pre_onset_seconds * FPS))
    dense_end = onset + int(round(horizon.seconds * FPS))
    actual_defender_path = [
        [
            (frame_id - onset) / FPS,
            float(frame_map[frame_id].players[defender_id].x),
            float(frame_map[frame_id].players[defender_id].y),
        ]
        for frame_id in range(onset, dense_end + 1)
    ]
    actual_endpoint = actual_defender_path[-1]
    nearest_actual = min(
        range(len(actions)),
        key=lambda index: math.hypot(
            actions[index].endpoint_x - actual_endpoint[1],
            actions[index].endpoint_y - actual_endpoint[2],
        ),
    )
    named = {
        "direct_minimum": int(tradeoff.direct_action_index),
        "derived_minimum": int(tradeoff.beneficiary_action_index),
        "compromise": int(tradeoff.compromise_action_index),
        "joint_sum_minimum": int(np.argmin(direct + derived)),
        "nearest_actual_endpoint": int(nearest_actual),
    }
    for ranking in rankings:
        ranking["defender_name"] = metadata.players[ranking["defender_id"]].short_name
    return {
        "match_id": str(row.match_id),
        "match_label": f"{metadata.home_team_name} vs {metadata.away_team_name}",
        "onset_frame_id": onset,
        "shot_frame_id": int(row.shot_frame_id),
        "runner_id": runner_id,
        "runner_name": metadata.players[runner_id].short_name,
        "attacking_team_id": team_id,
        "attacking_direction": direction,
        "defender_id": defender_id,
        "defender_name": metadata.players[defender_id].short_name,
        "review": {
            "onset": str(review.onset_review),
            "possession": str(review.possession_review),
            "interaction": str(review.interaction_review),
            "note": "" if pd.isna(review.note) else str(review.note),
        },
        "horizon_seconds": float(horizon.seconds),
        "horizon_reason": (
            observed_reason
            if horizon.limiting_reason == "observed_termination"
            else horizon.limiting_reason
        ),
        "seconds_to_shot": float(seconds_to_shot),
        "pre_onset_seconds": float(args.pre_onset_seconds),
        "evaluation_times_s": [float(value) for value in times],
        "candidate_defenders": rankings,
        "generated_path_variant_count": len(generated.actions),
        "evaluated_endpoint_count": len(actions),
        "named_response_indices": named,
        "pareto_response_count": len(pareto),
        "tradeoff": asdict(tradeoff),
        "responses": response_payload,
        "actual_response": {
            "path_txy": actual_defender_path,
            "endpoint_x": float(actual_endpoint[1]),
            "endpoint_y": float(actual_endpoint[2]),
            "direct_threat": observed_direct,
            "derived_threat": observed_derived,
            "beneficiary_id": observed_beneficiary_id,
            "direct_trace": [float(value) for value in observed[runner_index]],
            "derived_trace": [
                float(value) for value in observed[other_indices[observed_derived_row]]
            ],
        },
        "background_frames": [
            {
                "time_s": (frame_id - onset) / FPS,
                "players": [
                    [player_id, player.team_id, float(player.x), float(player.y)]
                    for player_id, player in frame_map[frame_id].players.items()
                ],
                "ball": (
                    [float(frame_map[frame_id].ball.x), float(frame_map[frame_id].ball.y)]
                    if frame_map[frame_id].ball is not None
                    else None
                ),
            }
            for frame_id in range(dense_start, dense_end + 1)
        ],
    }


def main() -> None:
    args = parse_args()
    scene_specs = args.scene or list(DEFAULT_SCENES)
    audit = pd.read_csv(
        args.audit_csv,
        dtype={"match_id": str, "player_id": str, "team_id": str},
    )
    reviews = pd.read_csv(
        args.review_csv,
        dtype={"match_id": str, "runner_id": str},
    )
    selected = []
    for spec in scene_specs:
        match_text, frame_text = spec.rsplit(":", 1)
        match_id = normalize_bundesliga_match_id(match_text)
        frame_id = int(frame_text)
        rows = audit[(audit.match_id == match_id) & (audit.frame_id == frame_id)]
        if len(rows) != 1:
            raise ValueError(f"scene must match exactly one audit row: {spec}")
        review = reviews[
            (reviews.match_id == match_id)
            & (reviews.onset_frame_id == frame_id)
            & (reviews.runner_id == str(rows.iloc[0].player_id))
        ]
        if len(review) != 1:
            raise ValueError(f"scene must match exactly one review row: {spec}")
        selected.append((rows.iloc[0], review.iloc[0]))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for number, (row, review) in enumerate(selected, start=1):
        started = time.perf_counter()
        files = find_bundesliga_files(args.data_dir, str(row.match_id))
        metadata = load_bundesliga_match_metadata(files["matchinfo"])
        onset = int(row.frame_id)
        maximum_end = min(
            int(row.shot_frame_id),
            onset + int(math.ceil(args.max_horizon_seconds * FPS)),
        )
        frame_map = load_bundesliga_frames(
            files["positions"],
            range(
                onset - max(10, int(round(args.pre_onset_seconds * FPS))),
                maximum_end + 1,
            ),
        )
        print(
            f"[{number}/{len(selected)}] {row.match_id} frame {onset} · "
            f"{metadata.players[str(row.player_id)].short_name}",
            flush=True,
        )
        result = _prepare_scene(row, review, frame_map, metadata, args)
        results.append(result)
        print(
            f"  H={result['horizon_seconds']:.1f}s ({result['horizon_reason']}) · "
            f"D={result['defender_name']} · {result['evaluated_endpoint_count']} endpoints · "
            f"Pareto {result['pareto_response_count']} · "
            f"{time.perf_counter()-started:.1f}s",
            flush=True,
        )
    output = args.output_dir / "direct_derived_response_maps.json"
    output.write_text(json.dumps(results, ensure_ascii=False), encoding="utf-8")
    summary_rows = []
    for result in results:
        direct = result["responses"][result["named_response_indices"]["direct_minimum"]]
        derived = result["responses"][result["named_response_indices"]["derived_minimum"]]
        summary_rows.append(
            {
                "match_id": result["match_id"],
                "onset_frame_id": result["onset_frame_id"],
                "runner_id": result["runner_id"],
                "runner_name": result["runner_name"],
                "defender_id": result["defender_id"],
                "defender_name": result["defender_name"],
                "horizon_seconds": result["horizon_seconds"],
                "horizon_reason": result["horizon_reason"],
                "endpoint_count": result["evaluated_endpoint_count"],
                "pareto_count": result["pareto_response_count"],
                "direct_branch_gap": derived["direct_threat"] - direct["direct_threat"],
                "derived_branch_gap": direct["derived_threat"] - derived["derived_threat"],
                "interaction_review": result["review"]["interaction"],
            }
        )
    pd.DataFrame(summary_rows).to_csv(args.output_dir / "summary.csv", index=False)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "direct-derived-response-map-v0.1",
                "status": "scene_diagnostic_not_final_value_model",
                "scene_count": len(results),
                "horizon": asdict(
                    AdaptiveHorizonConfig(maximum_seconds=args.max_horizon_seconds)
                ),
                "influence": asdict(GoalWeightedInfluenceConfig()),
                "reachability": asdict(
                    SteeringReachabilityConfig(
                        state_position_resolution_m=1.5,
                        state_speed_resolution_mps=1.0,
                        state_heading_bins=48,
                        control_direction_count=12,
                        plant_cut_direction_count=12,
                        variants_per_endpoint=1,
                    )
                ),
                "notes": [
                    "Threat core is the unchanged goal-weighted residual influence diagnostic.",
                    "One minimum-effort feasible path is evaluated per unique 1 m endpoint cell.",
                    "Actual defender trajectory is shown separately and never treated as generated.",
                    "All non-intervened players, including the focal runner, retain observed paths.",
                    "Relevant-defender ranking uses only the t=0 state and pre-onset velocity history.",
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"output: {output.resolve()}")


if __name__ == "__main__":
    main()
