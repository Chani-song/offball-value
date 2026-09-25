#!/usr/bin/env python3
"""Evaluate the focal-runner versus named-beneficiary defensive dilemma."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import json
import math
from pathlib import Path

from offball_value.bundesliga import (
    find_bundesliga_files,
    load_bundesliga_frames,
    load_bundesliga_match_metadata,
    normalize_bundesliga_match_id,
)
from offball_value.defender_response import (
    DefenderResponseConfig,
    generate_defender_response_actions,
)
from offball_value.pass_window_search import (
    TargetedPassWindowResponseIncumbents,
    evaluate_pass_window_response,
    search_targeted_pass_window_response_actions,
    targeted_incumbents_from_response_traces,
)
from offball_value.pass_window_value import prepare_pass_release_background_sequence
from offball_value.steering_reachable import SteeringReachabilityConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--match-id", default="J03WOH")
    parser.add_argument("--frame-id", type=int, default=68836)
    parser.add_argument("--attack-rank", type=int, default=1)
    parser.add_argument("--beneficiary-id", default="DFL-OBJ-002FXT")
    parser.add_argument("--nearest-beneficiary-defenders", type=int, default=3)
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data/raw/bundesliga-integrated")
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/processed/attacker_maximin_v0_1"),
    )
    parser.add_argument("--control-distance", type=float, default=1.5)
    parser.add_argument("--obso-batch-size", type=int, default=8)
    parser.add_argument("--progress-every", type=int, default=250)
    return parser.parse_args()


def _option_payload(item, metadata):
    result = asdict(item)
    result["player_name"] = metadata.players[item.player_id].short_name
    return result


def _trace_payload(trace, metadata):
    return {
        "focal_peak": trace.focal_peak,
        "focal_peak_time_s": trace.focal_peak_time_s,
        "other_peak": trace.other_peak,
        "other_peak_time_s": trace.other_peak_time_s,
        "other_peak_player_id": trace.other_peak_player_id,
        "other_peak_player_name": (
            metadata.players[trace.other_peak_player_id].short_name
            if trace.other_peak_player_id is not None
            else None
        ),
        "team_peak": trace.team_peak,
        "player_windows": [
            {
                **asdict(item),
                "player_name": metadata.players[item.player_id].short_name,
            }
            for item in trace.player_windows
        ],
        "points": [
            {
                "release_time_s": point.release_time_s,
                "ball_owner_id": point.ball_owner_id,
                "ball_owner_name": metadata.players[point.ball_owner_id].short_name,
                "owner_ball_distance_m": point.owner_ball_distance_m,
                "controlled": point.controlled,
                "team_maximum": point.team_maximum,
                "focal": _option_payload(point.focal, metadata),
                "best_other": (
                    _option_payload(point.best_other, metadata)
                    if point.best_other is not None
                    else None
                ),
                "player_options": [
                    _option_payload(item, metadata) for item in point.player_options
                ],
            }
            for point in trace.points
        ],
    }


def _selected_action(action_set, action_id):
    return next(action for action in action_set.actions if action.action_id == action_id)


def main() -> None:
    args = parse_args()
    match_id = normalize_bundesliga_match_id(args.match_id)
    directory = args.input_dir / match_id
    path = directory / f"frame_{args.frame_id}_attacker_maximin.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    attacks = sorted(
        payload["searched_attacks"],
        key=lambda item: item["best_response"]["value"],
        reverse=True,
    )
    attack = attacks[args.attack_rank - 1]
    pass_window = attack["pass_window_dilemma"]
    files = find_bundesliga_files(args.data_dir, match_id)
    metadata = load_bundesliga_match_metadata(files["matchinfo"])
    frame_id = int(payload["frame_id"])
    frames = load_bundesliga_frames(
        files["positions"], range(frame_id - 10, frame_id + 61)
    )
    history = tuple(frames[index] for index in range(frame_id - 10, frame_id + 1))
    decision = frames[frame_id]
    team_id = str(payload["attacking_team_id"])
    attacker_id = str(payload["attacker_id"])
    beneficiary_id = str(args.beneficiary_id)
    if beneficiary_id not in decision.players:
        raise KeyError(f"beneficiary {beneficiary_id} is missing")
    if decision.players[beneficiary_id].team_id != team_id:
        raise ValueError("beneficiary must be an attacking teammate")
    direction = int(payload["attacking_direction"])
    owner_id = str(pass_window["ball_owner_id"])
    background = prepare_pass_release_background_sequence(
        frames, frame_id, tuple(pass_window["release_times_s"])
    )
    goalkeeper_ids = tuple(
        goalkeeper_id
        for team_key in metadata.teams
        if (goalkeeper_id := metadata.goalkeeper_id(team_key)) is not None
    )
    reachability = SteeringReachabilityConfig(**payload["parameters"]["reachability"])
    response_config = DefenderResponseConfig(
        **payload["parameters"]["defender_response"]
    )

    existing_ids = [item["defender_id"] for item in attack["relevant_defenders"]]
    beneficiary = decision.players[beneficiary_id]
    nearest = sorted(
        (
            math.hypot(state.x - beneficiary.x, state.y - beneficiary.y),
            player_id,
        )
        for player_id, state in decision.players.items()
        if state.team_id != team_id and player_id not in goalkeeper_ids
    )[: args.nearest_beneficiary_defenders]
    defender_ids = list(dict.fromkeys(existing_ids + [item[1] for item in nearest]))
    print(
        "defender pool: "
        + ", ".join(metadata.players[item].short_name for item in defender_ids),
        flush=True,
    )
    per_defender = {}
    incumbents: TargetedPassWindowResponseIncumbents | None = None
    total_evaluated = 0
    for index, defender_id in enumerate(defender_ids, start=1):
        print(
            f"[{index}/{len(defender_ids)}] {metadata.players[defender_id].short_name}",
            flush=True,
        )
        action_set = generate_defender_response_actions(
            decision, history, defender_id, response_config, reachability
        )
        # A per-defender seed must be one of that defender's own feasible actions.
        seed_action = action_set.actions[0]
        seed_trace = evaluate_pass_window_response(
            background,
            attacker_id,
            attack["attack_path_xy"],
            payload["attack_path_times_s"],
            team_id,
            direction,
            goalkeeper_ids,
            owner_id,
            seed_action,
            control_distance_m=args.control_distance,
        )
        local_seed = targeted_incumbents_from_response_traces(
            [(seed_action, seed_trace)], beneficiary_id
        )
        local = search_targeted_pass_window_response_actions(
            background,
            attacker_id,
            attack["attack_path_xy"],
            payload["attack_path_times_s"],
            team_id,
            direction,
            goalkeeper_ids,
            owner_id,
            action_set.actions,
            local_seed,
            control_distance_m=args.control_distance,
            obso_batch_size=args.obso_batch_size,
            progress_every=args.progress_every,
        )
        total_evaluated += local.evaluated_response_count
        if incumbents is None:
            incumbents = local
        else:
            incumbents = TargetedPassWindowResponseIncumbents(
                direct_cover=min(
                    incumbents.direct_cover,
                    local.direct_cover,
                    key=lambda item: item.rank_key,
                ),
                beneficiary_cover=min(
                    incumbents.beneficiary_cover,
                    local.beneficiary_cover,
                    key=lambda item: item.rank_key,
                ),
                dilemma_best=min(
                    incumbents.dilemma_best,
                    local.dilemma_best,
                    key=lambda item: item.rank_key,
                ),
                beneficiary_id=beneficiary_id,
                evaluated_response_count=total_evaluated,
            )
        per_defender[defender_id] = {
            "defender_name": metadata.players[defender_id].short_name,
            "current_distance_to_runner_m": math.hypot(
                decision.players[defender_id].x - decision.players[attacker_id].x,
                decision.players[defender_id].y - decision.players[attacker_id].y,
            ),
            "current_distance_to_beneficiary_m": math.hypot(
                decision.players[defender_id].x - beneficiary.x,
                decision.players[defender_id].y - beneficiary.y,
            ),
            "direct_cover": asdict(local.direct_cover),
            "beneficiary_cover": asdict(local.beneficiary_cover),
            "dilemma_best": asdict(local.dilemma_best),
        }
        del action_set
        gc.collect()

    if incumbents is None:
        raise ValueError("defender pool is empty")

    response_payloads = {}
    for key, selected in (
        ("direct_cover", incumbents.direct_cover),
        ("beneficiary_cover", incumbents.beneficiary_cover),
        ("dilemma_best", incumbents.dilemma_best),
    ):
        action_set = generate_defender_response_actions(
            decision, history, selected.defender_id, response_config, reachability
        )
        action = _selected_action(action_set, selected.defender_action_id)
        trace = evaluate_pass_window_response(
            background,
            attacker_id,
            attack["attack_path_xy"],
            payload["attack_path_times_s"],
            team_id,
            direction,
            goalkeeper_ids,
            owner_id,
            action,
            control_distance_m=args.control_distance,
        )
        beneficiary_window = next(
            item for item in trace.player_windows if item.player_id == beneficiary_id
        )
        response_payloads[key] = {
            "selection": asdict(selected),
            "defender_name": metadata.players[selected.defender_id].short_name,
            "defender_path_times_s": list(action.response_path_times_s),
            "defender_path_xy": [list(item) for item in action.full_path_xy],
            "focal_peak": trace.focal_peak,
            "beneficiary_peak": beneficiary_window.peak_value,
            "beneficiary_peak_time_s": beneficiary_window.peak_time_s,
            "dilemma_value": max(trace.focal_peak, beneficiary_window.peak_value),
            "trace": _trace_payload(trace, metadata),
        }
        del action_set
        gc.collect()

    nearest_payload = [
        {
            "defender_id": defender_id,
            "defender_name": metadata.players[defender_id].short_name,
            "distance_m": distance,
        }
        for distance, defender_id in nearest
    ]
    attack["targeted_pass_window_dilemma"] = {
        "schema_version": "targeted-pass-window-dilemma-v0.1",
        "beneficiary_id": beneficiary_id,
        "beneficiary_name": metadata.players[beneficiary_id].short_name,
        "definition": "min over one feasible defender response of max(focal runner option, named beneficiary option) across post-onset controlled pass releases",
        "generic_other_is_diagnostic_only": True,
        "release_times_s": pass_window["release_times_s"],
        "ball_owner_id": owner_id,
        "defender_pool_ids": defender_ids,
        "nearest_beneficiary_defenders": nearest_payload,
        "per_defender": per_defender,
        "responses": response_payloads,
        "evaluated_response_count": incumbents.evaluated_response_count,
    }
    payload["schema_version"] = "attacker-maximin-v0.4-targeted-dilemma"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"updated: {path.resolve()}")


if __name__ == "__main__":
    main()
