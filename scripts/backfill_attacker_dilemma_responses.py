#!/usr/bin/env python3
"""Add R-cover, O-cover, and global defensive responses to a v0.1 audit."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import json
from pathlib import Path

from offball_value.action_value import decompose_threat_by_nearest_attacker
from offball_value.attacker_maximin import (
    TerminalBranchMinimum,
    search_terminal_branch_responses,
)
from offball_value.bundesliga import (
    find_bundesliga_files,
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
)
from offball_value.reference_obso import evaluate_reference_obso
from offball_value.steering_reachable import SteeringReachabilityConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--match-id", default="J03WOH")
    parser.add_argument("--frame-id", type=int, default=68836)
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data/raw/bundesliga-integrated")
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/processed/attacker_maximin_v0_1"),
    )
    parser.add_argument("--obso-batch-size", type=int, default=8)
    parser.add_argument("--progress-every", type=int, default=250)
    parser.add_argument(
        "--attack-index",
        type=int,
        default=0,
        help="One-based searched-attack index; 0 processes every incomplete item.",
    )
    return parser.parse_args()


def _decomposition(
    background,
    item,
    response,
    attack_times,
    attacking_team_id,
    attacking_direction,
    attacker_id,
    goalkeeper_ids,
):
    frame, velocities = build_local_counterfactual_state(
        background,
        attacker_id,
        item["attack_path_xy"],
        attack_times,
        defender_id=response.defender_id,
        defender_path_xy=response.full_path_xy,
        defender_path_times_s=response.response_path_times_s,
    )
    surface = evaluate_reference_obso(
        frame,
        attacking_team_id,
        attacking_direction,
        velocities=velocities,
        goalkeeper_ids=goalkeeper_ids,
        apply_offside=True,
    )
    return asdict(
        decompose_threat_by_nearest_attacker(
            surface,
            frame,
            attacking_team_id,
            attacker_id,
            attacking_direction,
        )
    )


def _response_payload(
    selected,
    action_sets,
    metadata,
    background,
    item,
    attack_times,
    attacking_team_id,
    attacking_direction,
    attacker_id,
    goalkeeper_ids,
):
    response = next(
        action
        for action in action_sets[selected.defender_id].actions
        if action.action_id == selected.defender_action_id
    )
    return {
        "selection": asdict(selected),
        "defender_name": metadata.players[selected.defender_id].short_name,
        "defender_path_times_s": list(response.response_path_times_s),
        "defender_path_xy": [list(point) for point in response.full_path_xy],
        "decomposition": _decomposition(
            background,
            item,
            response,
            attack_times,
            attacking_team_id,
            attacking_direction,
            attacker_id,
            goalkeeper_ids,
        ),
    }


def main() -> None:
    args = parse_args()
    match_id = normalize_bundesliga_match_id(args.match_id)
    directory = args.input_dir / match_id
    input_path = directory / f"frame_{args.frame_id}_attacker_maximin.json"
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    files = find_bundesliga_files(args.data_dir, match_id)
    metadata = load_bundesliga_match_metadata(files["matchinfo"])
    frame_id = int(payload["frame_id"])
    frames = load_bundesliga_frames(
        files["positions"], range(frame_id - 10, frame_id + 51)
    )
    history = tuple(frames[index] for index in range(frame_id - 10, frame_id + 1))
    decision = frames[frame_id]
    terminal_background = prepare_observed_background_sequence(
        frames, frame_id, DefenderBestResponseConfig()
    ).states[-1]
    attacking_team_id = str(payload["attacking_team_id"])
    attacking_direction = int(payload["attacking_direction"])
    attacker_id = str(payload["attacker_id"])
    attack_times = tuple(float(value) for value in payload["attack_path_times_s"])
    goalkeeper_ids = tuple(
        goalkeeper_id
        for team_id in metadata.teams
        if (goalkeeper_id := metadata.goalkeeper_id(team_id)) is not None
    )
    reachability = SteeringReachabilityConfig(
        **payload["parameters"]["reachability"]
    )
    response_config = DefenderResponseConfig(
        **payload["parameters"]["defender_response"]
    )
    all_items = payload["searched_attacks"]
    indexed_items = list(enumerate(all_items, start=1))
    if args.attack_index:
        if not 1 <= args.attack_index <= len(all_items):
            raise ValueError("attack-index lies outside searched attacks")
        indexed_items = [indexed_items[args.attack_index - 1]]
    for index, item in indexed_items:
        if "dilemma_responses" in item:
            print(f"[{index}/{len(all_items)}] already complete", flush=True)
            continue
        best = item["best_response"]
        split = item["best_response_decomposition"]
        direct_seed = TerminalBranchMinimum(
            branch="R",
            value=float(split["focal_maximum"]),
            maximum_x=float(split["focal_maximum_x"]),
            maximum_y=float(split["focal_maximum_y"]),
            evaluated_cell_count=0,
            defender_id=str(best["defender_id"]),
            defender_action_id=str(best["defender_action_id"]),
            defender_effort_m2ps3=float(best["defender_effort_m2ps3"]),
        )
        other_seed = TerminalBranchMinimum(
            branch="O",
            value=float(split["other_maximum"]),
            maximum_x=float(split["other_maximum_x"]),
            maximum_y=float(split["other_maximum_y"]),
            evaluated_cell_count=0,
            defender_id=str(best["defender_id"]),
            defender_action_id=str(best["defender_action_id"]),
            defender_effort_m2ps3=float(best["defender_effort_m2ps3"]),
        )
        global_payload = {
                "selection": {
                    "branch": "max(R,O)",
                    "value": best["value"],
                    "maximum_x": best["maximum_x"],
                    "maximum_y": best["maximum_y"],
                    "evaluated_cell_count": best["evaluated_cell_count"],
                    "defender_id": best["defender_id"],
                    "defender_action_id": best["defender_action_id"],
                    "defender_effort_m2ps3": best["defender_effort_m2ps3"],
                },
                "defender_name": item["best_defender_name"],
                "defender_path_times_s": item["best_defender_path_times_s"],
                "defender_path_xy": item["best_defender_path_xy"],
                "decomposition": item["best_response_decomposition"],
        }
        direct_best = direct_seed
        other_best = other_seed
        direct_payload = {**global_payload, "selection": asdict(direct_seed)}
        other_payload = {**global_payload, "selection": asdict(other_seed)}
        evaluated_count = 0
        print(
            f"[{index}/{len(all_items)}] branch search · fixed rank {item['fixed_rank']}",
            flush=True,
        )
        for defender_number, defender in enumerate(
            item["relevant_defenders"], start=1
        ):
            defender_id = str(defender["defender_id"])
            print(
                f"  [{defender_number}/{len(item['relevant_defenders'])}] "
                f"{metadata.players[defender_id].short_name}",
                flush=True,
            )
            action_set = generate_defender_response_actions(
                decision,
                history,
                defender_id,
                response_config,
                reachability,
            )
            one_set = {defender_id: action_set}
            result = search_terminal_branch_responses(
                terminal_background,
                attacker_id,
                item["attack_path_xy"],
                attack_times,
                attacking_team_id,
                attacking_direction,
                goalkeeper_ids,
                one_set,
                initial_direct_cover=direct_best,
                initial_other_cover=other_best,
                obso_batch_size=args.obso_batch_size,
                progress_every=args.progress_every,
            )
            evaluated_count += result.evaluated_response_count
            if result.direct_cover.defender_action_id != direct_best.defender_action_id:
                direct_best = result.direct_cover
                direct_payload = _response_payload(
                    direct_best,
                    one_set,
                    metadata,
                    terminal_background,
                    item,
                    attack_times,
                    attacking_team_id,
                    attacking_direction,
                    attacker_id,
                    goalkeeper_ids,
                )
            if result.other_cover.defender_action_id != other_best.defender_action_id:
                other_best = result.other_cover
                other_payload = _response_payload(
                    other_best,
                    one_set,
                    metadata,
                    terminal_background,
                    item,
                    attack_times,
                    attacking_team_id,
                    attacking_direction,
                    attacker_id,
                    goalkeeper_ids,
                )
            del result, one_set, action_set
            gc.collect()
        item["dilemma_responses"] = {
            "direct_cover": direct_payload,
            "other_cover": other_payload,
            "global_best": global_payload,
        }
        item["branch_response_evaluated_count"] = evaluated_count
        input_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(
            f"R-min={direct_best.value:.6f}; "
            f"O-min={other_best.value:.6f}",
            flush=True,
        )
    payload["schema_version"] = "attacker-maximin-v0.2-dilemma"
    input_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"updated: {input_path.resolve()}")


if __name__ == "__main__":
    main()
