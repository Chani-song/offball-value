#!/usr/bin/env python3
"""Evaluate threat-minimizing feasible responses for one audited attack path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import pandas as pd

from offball_value.bundesliga import (
    find_bundesliga_files,
    infer_attacking_direction,
    load_bundesliga_frames,
    load_bundesliga_match_metadata,
    normalize_bundesliga_match_id,
)
from offball_value.defender_best_response import (
    DefenderBestResponseConfig,
    evaluate_defender_trajectory,
    prepare_observed_background_sequence,
    search_defender_best_response,
)
from offball_value.defender_response import (
    DefenderResponseConfig,
    generate_defender_response_actions,
)
from offball_value.steering_reachable import SteeringReachabilityConfig


ATTACK_TIMES_S = (0.0, 0.4, 0.8, 1.2, 1.6, 2.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw/bundesliga-integrated"),
    )
    parser.add_argument("--match-id", default="J03WMX")
    parser.add_argument("--frame-id", type=int, default=14913)
    parser.add_argument("--player-id", default="DFL-OBJ-002G4A")
    parser.add_argument(
        "--endpoint-dir",
        type=Path,
        default=Path("data/processed/attacker_endpoints_v0_6"),
    )
    parser.add_argument(
        "--defender-summary-dir",
        type=Path,
        default=Path("data/processed/defender_response_audit_v0_1"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/defender_best_response_v0_1"),
    )
    parser.add_argument("--response-delay", type=float, default=0.2)
    parser.add_argument(
        "--defender-id",
        action="append",
        default=[],
        help="Repeat to override the three audited relevant defenders.",
    )
    return parser.parse_args()


def _path_from_row(row: pd.Series) -> tuple[tuple[float, float], ...]:
    points = json.loads(str(row["path_xy"]))
    return (
        (float(row["start_x"]), float(row["start_y"])),
        *tuple((float(point[0]), float(point[1])) for point in points),
    )


def _straight_action(actions: pd.DataFrame) -> pd.Series:
    eligible = actions[actions["optimization_eligible"].fillna(False)].copy()
    continuous = eligible[eligible["maneuver_type"] == "continuous_steering"]
    if continuous.empty:
        raise ValueError("No continuous-steering attack action is available")
    return continuous.sort_values(
        ["control_effort_m2ps3", "grid_snap_distance_m", "action_id"]
    ).iloc[0]


def _audited_defender_ids(
    path: Path,
    frame_id: int,
) -> tuple[str, ...]:
    rows = pd.read_csv(path)
    rows = rows[
        (rows["frame_id"] == frame_id)
        & (rows["attack_label"] == "straight/coast")
    ].sort_values("candidate_rank")
    if rows.empty:
        raise ValueError("No straight/coast defender shortlist exists")
    return tuple(rows["defender_id"].astype(str).drop_duplicates())


def _trace_payload(trace) -> dict[str, object]:
    return {
        "action_id": trace.action_id,
        "horizon_peak": trace.horizon_peak,
        "horizon_mean": trace.horizon_mean,
        "terminal": trace.terminal,
        "effort_m2ps3": trace.effort_m2ps3,
        "points": [
            {
                "time_s": point.time_s,
                "maximum_obso": point.maximum_obso,
                "maximum_x": point.maximum_x,
                "maximum_y": point.maximum_y,
                "evaluated_cell_count": point.evaluated_cell_count,
            }
            for point in trace.points
        ],
    }


def main() -> None:
    args = parse_args()
    match_id = normalize_bundesliga_match_id(args.match_id)
    files = find_bundesliga_files(args.data_dir, match_id)
    metadata = load_bundesliga_match_metadata(files["matchinfo"])
    endpoint_path = args.endpoint_dir / match_id / "attacker_endpoint_actions.csv"
    endpoint_rows = pd.read_csv(endpoint_path)
    attack_rows = endpoint_rows[
        (endpoint_rows["frame_id"] == args.frame_id)
        & (endpoint_rows["player_id"] == args.player_id)
    ]
    attack_row = _straight_action(attack_rows)
    attack_path = _path_from_row(attack_row)

    if args.defender_id:
        defender_ids = tuple(dict.fromkeys(args.defender_id))
    else:
        summary_path = (
            args.defender_summary_dir / match_id / "defender_response_summary.csv"
        )
        defender_ids = _audited_defender_ids(summary_path, args.frame_id)

    frame_ids = range(args.frame_id - 10, args.frame_id + 51)
    frames = load_bundesliga_frames(files["positions"], frame_ids)
    decision = frames[args.frame_id]
    history = tuple(frames[target] for target in range(args.frame_id - 10, args.frame_id + 1))
    attacking_team_id = decision.players[args.player_id].team_id
    attacking_direction = infer_attacking_direction(
        decision, attacking_team_id, metadata
    )
    goalkeeper_ids = tuple(
        goalkeeper_id
        for team_id in metadata.teams
        if (goalkeeper_id := metadata.goalkeeper_id(team_id)) is not None
    )
    best_config = DefenderBestResponseConfig()
    background = prepare_observed_background_sequence(
        frames, args.frame_id, best_config
    )
    response_config = DefenderResponseConfig(
        response_delay_seconds=args.response_delay
    )
    reachability_config = SteeringReachabilityConfig()

    output_dir = args.output_dir / match_id
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    payload: dict[str, object] = {
        "match_id": match_id,
        "frame_id": args.frame_id,
        "attacker_id": args.player_id,
        "attacker_action_id": str(attack_row["action_id"]),
        "attacker_path_times_s": ATTACK_TIMES_S,
        "attacker_path_xy": attack_path,
        "background_model": "observed_future_nonintervened_players_and_ball",
        "objective": "lexicographic_min(horizon_peak, horizon_mean, terminal, effort)",
        "defenders": [],
    }
    for defender_index, defender_id in enumerate(defender_ids, start=1):
        name = metadata.players[defender_id].short_name
        print(
            f"[{defender_index}/{len(defender_ids)}] generating feasible responses: {name}",
            flush=True,
        )
        action_set = generate_defender_response_actions(
            decision,
            history,
            defender_id,
            response_config,
            reachability_config,
        )
        started = time.perf_counter()
        search = search_defender_best_response(
            background,
            args.player_id,
            attack_path,
            ATTACK_TIMES_S,
            attacking_team_id,
            attacking_direction,
            goalkeeper_ids,
            action_set.actions,
            config=best_config,
            progress_every=100,
        )
        elapsed = time.perf_counter() - started
        actual = evaluate_defender_trajectory(
            background,
            args.player_id,
            attack_path,
            ATTACK_TIMES_S,
            attacking_team_id,
            attacking_direction,
            goalkeeper_ids,
            action=None,
            defender_id=defender_id,
            action_id="observed-defender",
            config=best_config,
        )
        best_action = next(
            action
            for action in action_set.actions
            if action.action_id == search.best.action_id
        )
        defender_payload = {
            "defender_id": defender_id,
            "defender_name": name,
            "feasible_action_count": len(action_set.actions),
            "unique_endpoint_count": action_set.unique_endpoint_count,
            "terminal_spread": search.terminal_spread,
            "fully_evaluated_action_count": len(search.fully_evaluated_traces),
            "evaluation_seconds": elapsed,
            "best": _trace_payload(search.best),
            "best_path_times_s": best_action.response_path_times_s,
            "best_path_xy": best_action.full_path_xy,
            "observed_defender_background": _trace_payload(actual),
        }
        payload["defenders"].append(defender_payload)
        rows.append(
            {
                "match_id": match_id,
                "frame_id": args.frame_id,
                "attacker_id": args.player_id,
                "attacker_action_id": attack_row["action_id"],
                "defender_id": defender_id,
                "defender_name": name,
                "feasible_action_count": len(action_set.actions),
                "unique_endpoint_count": action_set.unique_endpoint_count,
                "terminal_spread": search.terminal_spread,
                "fully_evaluated_action_count": len(search.fully_evaluated_traces),
                "best_action_id": search.best.action_id,
                "best_horizon_peak": search.best.horizon_peak,
                "best_horizon_mean": search.best.horizon_mean,
                "best_terminal": search.best.terminal,
                "observed_horizon_peak": actual.horizon_peak,
                "observed_horizon_mean": actual.horizon_mean,
                "observed_terminal": actual.terminal,
                "evaluation_seconds": elapsed,
            }
        )
        print(
            f"{name}: best peak={search.best.horizon_peak:.9f}, "
            f"observed={actual.horizon_peak:.9f}, "
            f"terminal spread={search.terminal_spread:.3g}, "
            f"full traces={len(search.fully_evaluated_traces)}, {elapsed:.1f}s",
            flush=True,
        )

    csv_path = output_dir / f"frame_{args.frame_id}_best_responses.csv"
    json_path = output_dir / f"frame_{args.frame_id}_best_responses.json"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    overall = min(
        payload["defenders"],
        key=lambda item: (
            item["best"]["horizon_peak"],
            item["best"]["horizon_mean"],
            item["best"]["terminal"],
            item["best"]["effort_m2ps3"],
            item["defender_id"],
        ),
    )
    payload["overall_best_defender_id"] = overall["defender_id"]
    payload["overall_best_defender_name"] = overall["defender_name"]
    payload["overall_best_action_id"] = overall["best"]["action_id"]
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"summary: {csv_path}")
    print(f"paths:   {json_path}")


if __name__ == "__main__":
    main()
