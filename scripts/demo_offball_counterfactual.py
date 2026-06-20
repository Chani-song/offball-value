from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Iterable

import pandas as pd

from offball_value.adapters import load_metrica_frames
from offball_value.baseline import PlayerState, compute_toy_option_score, detect_candidate_run
from offball_value.possession import infer_ball_carrier


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def to_player_state(p) -> PlayerState:
    return PlayerState(
        player_id=str(p.player_id),
        x=float(p.x),
        y=float(p.y),
    )


def split_teams(frame, attacking_team: str = "Home"):
    attackers = [to_player_state(p) for p in frame.players if p.team == attacking_team]
    defenders = [to_player_state(p) for p in frame.players if p.team != attacking_team]
    return attackers, defenders


def best_teammate_option(
    teammates: list[PlayerState],
    defenders: list[PlayerState],
    excluded_id: str | None = None,
    excluded_ids: set[str] | None = None,
):
    resolved_excluded_ids = set(excluded_ids or set())
    if excluded_id is not None:
        resolved_excluded_ids.add(excluded_id)

    candidates = []
    for tm in teammates:
        if tm.player_id in resolved_excluded_ids:
            continue
        score = compute_toy_option_score(tm, defenders, attacking_direction=1)
        candidates.append((tm.player_id, score))
    if not candidates:
        return None, None
    return max(candidates, key=lambda x: x[1])


def build_player_map(players: Iterable[PlayerState]) -> dict[str, PlayerState]:
    return {p.player_id: p for p in players}


def move_closest_defender_counterfactually(
    defenders_start: dict[str, PlayerState],
    defenders_end: list[PlayerState],
    runner_start: PlayerState,
    runner_end: PlayerState,
) -> list[PlayerState]:
    """
    Simple counterfactual:
    find the closest defender to the runner at the end frame,
    then partially pull that defender back toward the start-frame location.
    """
    if not defenders_end:
        return defenders_end

    closest = min(
        defenders_end,
        key=lambda d: (d.x - runner_end.x) ** 2 + (d.y - runner_end.y) ** 2,
    )

    cf_defenders = deepcopy(defenders_end)

    for d in cf_defenders:
        if d.player_id == closest.player_id and d.player_id in defenders_start:
            d0 = defenders_start[d.player_id]
            d.x = 0.7 * d0.x + 0.3 * d.x
            d.y = 0.7 * d0.y + 0.3 * d.y

    return cf_defenders


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", type=int, default=1)
    parser.add_argument("--horizon", type=int, default=25)
    parser.add_argument("--attacking-team", type=str, default="Home")
    parser.add_argument("--min-run-distance", type=float, default=0.04)
    parser.add_argument("--ball-control-distance", type=float, default=None)
    parser.add_argument("--allow-uncontrolled", action="store_true")
    parser.add_argument("--top-k", type=int, default=30)
    args = parser.parse_args()

    frames = load_metrica_frames(game=args.game)
    print(f"Loaded {len(frames)} frame snapshots from Metrica game {args.game}")

    results = []

    for idx in range(0, len(frames) - args.horizon, args.horizon):
        f0 = frames[idx]
        f1 = frames[idx + args.horizon]
        start_carrier = infer_ball_carrier(f0, max_distance=args.ball_control_distance)
        end_carrier = infer_ball_carrier(f1, max_distance=args.ball_control_distance)

        if not args.allow_uncontrolled:
            if start_carrier is None or end_carrier is None:
                continue
            if start_carrier.team != args.attacking_team or end_carrier.team != args.attacking_team:
                continue

        attackers0, defenders0 = split_teams(f0, attacking_team=args.attacking_team)
        attackers1, defenders1 = split_teams(f1, attacking_team=args.attacking_team)

        if len(attackers0) < 3 or len(attackers1) < 3 or len(defenders0) < 3 or len(defenders1) < 3:
            continue

        atk0_map = build_player_map(attackers0)
        atk1_map = build_player_map(attackers1)
        def0_map = build_player_map(defenders0)

        for runner_id, s0 in atk0_map.items():
            if start_carrier is not None and runner_id == start_carrier.player_id:
                continue
            if end_carrier is not None and runner_id == end_carrier.player_id:
                continue

            s1 = atk1_map.get(runner_id)
            if s1 is None:
                continue

            if not detect_candidate_run(
                s0.x, s0.y, s1.x, s1.y,
                min_distance=args.min_run_distance,
            ):
                continue

            best_actual_id, best_actual_score = best_teammate_option(
                teammates=attackers1,
                defenders=defenders1,
                excluded_ids={
                    player_id
                    for player_id in [
                        runner_id,
                        end_carrier.player_id if end_carrier is not None else None,
                    ]
                    if player_id is not None
                },
            )
            if best_actual_score is None:
                continue

            cf_defenders = move_closest_defender_counterfactually(
                defenders_start=def0_map,
                defenders_end=defenders1,
                runner_start=s0,
                runner_end=s1,
            )

            best_cf_id, best_cf_score = best_teammate_option(
                teammates=attackers1,
                defenders=cf_defenders,
                excluded_ids={
                    player_id
                    for player_id in [
                        runner_id,
                        end_carrier.player_id if end_carrier is not None else None,
                    ]
                    if player_id is not None
                },
            )
            if best_cf_score is None:
                continue

            delta = float(best_actual_score - best_cf_score)
            movement = ((s1.x - s0.x) ** 2 + (s1.y - s0.y) ** 2) ** 0.5

            results.append(
                {
                    "match_id": f0.match_id,
                    "frame_start": f0.frame_id,
                    "frame_end": f1.frame_id,
                    "time_start_s": f0.time_s,
                    "time_end_s": f1.time_s,
                    "ball_carrier_start": start_carrier.player_id if start_carrier is not None else None,
                    "ball_carrier_end": end_carrier.player_id if end_carrier is not None else None,
                    "runner_id": runner_id,
                    "runner_start_x": s0.x,
                    "runner_start_y": s0.y,
                    "runner_end_x": s1.x,
                    "runner_end_y": s1.y,
                    "runner_movement": movement,
                    "best_actual_receiver": best_actual_id,
                    "best_actual_score": best_actual_score,
                    "best_counterfactual_receiver": best_cf_id,
                    "best_counterfactual_score": best_cf_score,
                    "draft_offball_value": delta,
                }
            )

    out = pd.DataFrame(results)

    if out.empty:
        print("No candidate runs found.")
        return

    out = out.sort_values("draft_offball_value", ascending=False).reset_index(drop=True)

    out_file = OUT_DIR / f"demo_offball_counterfactual_game{args.game}.csv"
    out.to_csv(out_file, index=False)

    print("\nTop results:")
    print(out.head(args.top_k).to_string(index=False))
    print(f"\nSaved {len(out)} rows to {out_file}")


if __name__ == "__main__":
    main()
