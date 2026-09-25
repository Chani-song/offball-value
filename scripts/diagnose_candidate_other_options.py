#!/usr/bin/env python3
"""Print per-attacker OBSO branches through one audited response."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from offball_value.bundesliga import (
    find_bundesliga_files,
    load_bundesliga_frames,
    load_bundesliga_match_metadata,
)
from offball_value.defender_best_response import (
    DefenderBestResponseConfig,
    build_local_counterfactual_state,
    prepare_observed_background_sequence,
)
from offball_value.reference_obso import (
    evaluate_reference_obso,
    offside_attacker_ids,
)


def main() -> None:
    path = Path(
        "data/processed/attacker_maximin_v0_1/DFL-MAT-J03WOH/"
        "frame_68836_attacker_maximin.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    files = find_bundesliga_files(
        Path("data/raw/bundesliga-integrated"), payload["match_id"]
    )
    metadata = load_bundesliga_match_metadata(files["matchinfo"])
    frame_id = int(payload["frame_id"])
    frames = load_bundesliga_frames(
        files["positions"], range(frame_id - 10, frame_id + 51)
    )
    sequence = prepare_observed_background_sequence(
        frames, frame_id, DefenderBestResponseConfig()
    )
    attack = sorted(
        payload["searched_attacks"],
        key=lambda item: item["best_response"]["value"],
        reverse=True,
    )[0]
    response = attack["dilemma_responses"]["global_best"]
    team_id = str(payload["attacking_team_id"])
    direction = int(payload["attacking_direction"])
    focal_id = str(payload["attacker_id"])
    kownacki_id = "DFL-OBJ-002FXT"
    goalkeeper_ids = tuple(
        goalkeeper_id
        for team_key in metadata.teams
        if (goalkeeper_id := metadata.goalkeeper_id(team_key)) is not None
    )
    print("time  status  Kownacki     best other   player               team max")
    for background in sequence.states:
        frame, velocities = build_local_counterfactual_state(
            background,
            focal_id,
            attack["attack_path_xy"],
            payload["attack_path_times_s"],
            defender_id=response["selection"]["defender_id"],
            defender_path_xy=response["defender_path_xy"],
            defender_path_times_s=response["defender_path_times_s"],
        )
        ball_xy = (frame.ball.x, frame.ball.y)
        offside = offside_attacker_ids(frame, team_id, direction, ball_xy)
        surface = evaluate_reference_obso(
            frame,
            team_id,
            direction,
            velocities=velocities,
            goalkeeper_ids=goalkeeper_ids,
            apply_offside=True,
        )
        eligible_ids = tuple(
            sorted(
                player_id
                for player_id, state in frame.players.items()
                if state.team_id == team_id and player_id not in offside
            )
        )
        xx, yy = np.meshgrid(surface.xgrid, surface.ygrid)
        grid = np.column_stack([xx.ravel(), yy.ravel()])
        positions = np.asarray(
            [(frame.players[player_id].x, frame.players[player_id].y) for player_id in eligible_ids]
        )
        owners = np.argmin(
            np.linalg.norm(grid[:, None, :] - positions[None, :, :], axis=2), axis=1
        ).reshape(surface.obso.shape)
        values = {
            player_id: float(
                np.max(np.where(owners == index, surface.obso, -np.inf))
            )
            for index, player_id in enumerate(eligible_ids)
        }
        other_values = {
            player_id: value
            for player_id, value in values.items()
            if player_id != focal_id
        }
        other_id = max(other_values, key=other_values.get)
        status = "OFF" if kownacki_id in offside else "ON"
        print(
            f"{background.time_s:3.1f}   {status:3s}   "
            f"{values.get(kownacki_id, 0.0):.9f}  "
            f"{other_values[other_id]:.9f}  "
            f"{metadata.players[other_id].short_name:20s} "
            f"{surface.maximum:.9f}"
        )


if __name__ == "__main__":
    main()
