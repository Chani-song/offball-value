from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd


@dataclass
class PlayerState:
    player_id: str
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0



def euclidean(x1: float, y1: float, x2: float, y2: float) -> float:
    return float(np.hypot(x1 - x2, y1 - y2))



def nearest_defender_distance(player: PlayerState, defenders: Iterable[PlayerState]) -> float:
    distances = [euclidean(player.x, player.y, d.x, d.y) for d in defenders]
    return min(distances) if distances else np.nan



def forward_progress(player: PlayerState, attacking_direction: int = 1) -> float:
    # assumes normalized x where larger x is closer to the opponent goal for attacking_direction=1
    return attacking_direction * player.x



def compute_toy_option_score(
    receiver: PlayerState,
    defenders: Iterable[PlayerState],
    attacking_direction: int = 1,
    field_length: float = 1.0,
) -> float:
    """Very rough score for a receiver option.

    Higher is better when the receiver is more advanced and farther from the nearest defender.
    Coordinates are expected to be normalized roughly into [0, 1] for x and y.
    """
    nd = nearest_defender_distance(receiver, defenders)
    progress = forward_progress(receiver, attacking_direction=attacking_direction)
    goal_bonus = max(0.0, progress)
    return 0.6 * goal_bonus + 0.4 * float(nd)



def counterfactual_runner_delta(
    receiver_actual: PlayerState,
    receiver_counterfactual: PlayerState,
    defenders_actual: Iterable[PlayerState],
    defenders_counterfactual: Iterable[PlayerState],
    attacking_direction: int = 1,
) -> float:
    actual = compute_toy_option_score(receiver_actual, defenders_actual, attacking_direction=attacking_direction)
    cf = compute_toy_option_score(receiver_counterfactual, defenders_counterfactual, attacking_direction=attacking_direction)
    return actual - cf



def detect_candidate_run(
    start_x: float,
    start_y: float,
    end_x: float,
    end_y: float,
    min_distance: float = 0.03,
) -> bool:
    return euclidean(start_x, start_y, end_x, end_y) >= min_distance



def naive_counterfactual_position(start_x: float, start_y: float, end_x: float, end_y: float, alpha: float = 0.2) -> tuple[float, float]:
    """Keep the player close to the start position.

    alpha=0 returns a complete freeze. alpha=1 returns the actual end position.
    """
    cf_x = (1 - alpha) * start_x + alpha * end_x
    cf_y = (1 - alpha) * start_y + alpha * end_y
    return float(cf_x), float(cf_y)
