#!/usr/bin/env python3
"""Re-evaluate audited attack 1 over controlled pass-release instants."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import json
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
from offball_value.pass_dynamics import detect_kick_frame
from offball_value.pass_window_search import (
    evaluate_pass_window_response,
    incumbents_from_response_traces,
    search_pass_window_response_actions,
)
from offball_value.pass_window_value import prepare_pass_release_background_sequence
from offball_value.steering_reachable import SteeringReachabilityConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--match-id", default="J03WOH")
    parser.add_argument("--frame-id", type=int, default=68836)
    parser.add_argument("--attack-rank", type=int, default=1)
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


def trace_payload(trace, metadata):
    def option(item):
        result = asdict(item)
        result["player_name"] = metadata.players[item.player_id].short_name
        return result

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
        "largest_teammate_gain_player_id": trace.largest_teammate_gain_player_id,
        "largest_teammate_gain_player_name": (
            metadata.players[trace.largest_teammate_gain_player_id].short_name
            if trace.largest_teammate_gain_player_id is not None
            else None
        ),
        "largest_teammate_gain": trace.largest_teammate_gain,
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
                "ball_owner_name": (
                    metadata.players[point.ball_owner_id].short_name
                    if point.ball_owner_id in metadata.players
                    else None
                ),
                "owner_ball_distance_m": point.owner_ball_distance_m,
                "controlled": point.controlled,
                "team_maximum": point.team_maximum,
                "focal": option(point.focal),
                "best_other": option(point.best_other) if point.best_other else None,
                "player_options": [option(item) for item in point.player_options],
            }
            for point in trace.points
        ],
    }


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
    direction = int(payload["attacking_direction"])
    owner_id = min(
        (
            (
                ((state.x - decision.ball.x) ** 2 + (state.y - decision.ball.y) ** 2),
                player_id,
            )
            for player_id, state in decision.players.items()
            if state.team_id == team_id
        )
    )[1]
    kick = detect_kick_frame(
        frames.values(),
        owner_id,
        frame_id,
    )
    kick_time = (kick.kick_frame_id - frame_id) / 25.0
    release_times = tuple(
        sorted(
            set(
                [0.0]
                + [value for value in (0.2, 0.4) if value < kick_time]
                + [kick_time]
            )
        )
    )
    background = prepare_pass_release_background_sequence(
        frames, frame_id, release_times
    )
    goalkeeper_ids = tuple(
        goalkeeper_id
        for team_key in metadata.teams
        if (goalkeeper_id := metadata.goalkeeper_id(team_key)) is not None
    )
    reachability = SteeringReachabilityConfig(
        **payload["parameters"]["reachability"]
    )
    response_config = DefenderResponseConfig(
        **payload["parameters"]["defender_response"]
    )

    seed_entries = (
        attack.get("pass_window_dilemma", {}).get("responses")
        or attack["dilemma_responses"]
    )
    seed_traces = []
    seed_actions = {}
    for response in seed_entries.values():
        defender_id = str(response["selection"]["defender_id"])
        if defender_id not in seed_actions:
            action_set = generate_defender_response_actions(
                decision,
                history,
                defender_id,
                response_config,
                reachability,
            )
            seed_actions[defender_id] = {
                action.action_id: action for action in action_set.actions
            }
        action = seed_actions[defender_id][response["selection"]["defender_action_id"]]
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
        seed_traces.append((action, trace))
    incumbents = incumbents_from_response_traces(seed_traces)
    print(
        f"kick={kick.kick_frame_id} t={kick_time:.2f}; releases={release_times}; "
        f"seeds R={incumbents.direct_cover.value:.6f} "
        f"O={incumbents.other_cover.value:.6f} "
        f"global={incumbents.global_best.value:.6f}",
        flush=True,
    )

    relevant_ids = [item["defender_id"] for item in attack["relevant_defenders"]]
    for index, defender_id in enumerate(relevant_ids, start=1):
        print(
            f"[{index}/{len(relevant_ids)}] {metadata.players[defender_id].short_name}",
            flush=True,
        )
        action_set = generate_defender_response_actions(
            decision,
            history,
            defender_id,
            response_config,
            reachability,
        )
        incumbents = search_pass_window_response_actions(
            background,
            attacker_id,
            attack["attack_path_xy"],
            payload["attack_path_times_s"],
            team_id,
            direction,
            goalkeeper_ids,
            owner_id,
            action_set.actions,
            incumbents,
            control_distance_m=args.control_distance,
            obso_batch_size=args.obso_batch_size,
            progress_every=args.progress_every,
        )
        del action_set
        gc.collect()

    response_payloads = {}
    for key, selected in (
        ("direct_cover", incumbents.direct_cover),
        ("other_cover", incumbents.other_cover),
        ("global_best", incumbents.global_best),
    ):
        action_set = generate_defender_response_actions(
            decision,
            history,
            selected.defender_id,
            response_config,
            reachability,
        )
        action = next(
            item
            for item in action_set.actions
            if item.action_id == selected.defender_action_id
        )
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
        response_payloads[key] = {
            "selection": asdict(selected),
            "defender_name": metadata.players[selected.defender_id].short_name,
            "defender_path_times_s": list(action.response_path_times_s),
            "defender_path_xy": [list(point) for point in action.full_path_xy],
            "trace": trace_payload(trace, metadata),
        }
        del action_set
        gc.collect()
    attack["pass_window_dilemma"] = {
        "schema_version": "pass-window-dilemma-v0.1",
        "ball_owner_id": owner_id,
        "ball_owner_name": metadata.players[owner_id].short_name,
        "kick_frame_id": kick.kick_frame_id,
        "kick_time_s": kick_time,
        "kick_detected": kick.detected,
        "release_times_s": list(release_times),
        "control_distance_m": args.control_distance,
        "offside_rule": "evaluated at each candidate pass-release instant",
        "responses": response_payloads,
        "evaluated_response_count": incumbents.evaluated_response_count,
    }
    payload["schema_version"] = "attacker-maximin-v0.3-pass-window"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"updated: {path.resolve()}")


if __name__ == "__main__":
    main()
