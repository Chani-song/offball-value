"""Pass value and legality with the whole defence on the pitch.

The imported evaluator (`defensive_positioning.payoff`) sees one defender:
the one the game controls. That defender fills both of the pass model's
defender roles, is the whole of the threat model's "room", and is the offside
line. On our scenes that is wrong in three measurable ways -- the defender
nearest the receiver is someone else 42-71% of the time, and the runner is
flagged offside against the single-defender line in about 18% of pairs where
the real second-last defender keeps him on.

The fix keeps every one of the collaborator's models and changes only which
defenders they are shown:

  pass model   given the controlled defender AND the background defenders.
               It was built for this: it selects the defender nearest the
               passer and the one nearest the receiver from however many it
               is handed, and it was fitted on every visible defender.
  offside      the collaborator's own full-game rule, `rules.offside`, which
               takes the second-last opponent. His 2v1 solver uses the
               reduced convention of one defender as the line; with the
               defence present that convention is no longer needed.
  threat       `value.positional_threat` measures "room" as the distance to
               one defender. Here it is the distance to the NEAREST defender.
               This is the only change to one of his formulas; every
               coefficient is his.

Each function reduces exactly to the imported one when the background is
empty. `tests/test_equivalence.py` checks that bit for bit.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from defensive_positioning.expected_pass import FAMILIES
from defensive_positioning.rules import inside_pitch, offside, offside_against_line
from defensive_positioning.value import location_core

from .agile_motion import AGILE
from .physics_pass import reach_time
from .run_passes import target_of

# ---- threat variant "epv_time" (2026-09-27, after the through-ball diagnostic)
#
# 준현's threat is 0.2 + 0.8 * location * (0.55 + 0.25 room + 0.2 location(support)
# support_room). Its location term rises 13 % over the 8 m from 70 to 78 m and its
# room term is direction-blind, so a runner free in behind is worth little more
# than one marked at his feet; swapping the room term alone moved the in-behind /
# to-feet ratio from 1.09 to 1.07. Two changes, the bracket and its three
# coefficients kept:
#   location  the public PAUSA EPV grid (data/static/EPV_grid.csv, Apache 2.0),
#             normalised to its maximum: 70 -> 78 m is x1.6 there. The 0.2
#             floor is dropped: grid values in the final third are 0.02-0.2,
#             and a 0.2 floor would swamp them.
#   room      1 - exp(-d / 9) as before, but d is the distance a defender
#             starting from rest covers in the MARGIN: the first defender's
#             arrival at the target minus the receiver's, by the same agile
#             race as pass model A (4.5 m/s^2 up to 9 m/s). For a standing
#             defender and a receiver already on the spot the margin is his
#             own run time, so d is his distance and the room is exactly 준현's;
#             a defender running away gains margin and room, one closing in
#             loses it. No new number enters.
# Support and its room are as before. Retention (2v1) prices the carrier the
# same way, so the pass / keep comparison stays on one scale.
EPV_GRID_PATH = Path(__file__).resolve().parents[2] / "data/static/EPV_grid.csv"
_EPV = None
RACE_PLAYER = {"reaction_s": AGILE.defender_delay_s, "speed_up": AGILE.speed_up,
               "braking": AGILE.braking, "plant_min_speed": AGILE.plant_min_speed,
               "plant_braking": AGILE.plant_braking, "plant_seconds": AGILE.plant_seconds,
               "max_speed": 9.0}
THREATS = ("andrew", "epv_time")


def epv_grid():
    global _EPV
    if _EPV is None:
        grid = np.loadtxt(EPV_GRID_PATH, delimiter=",")
        _EPV = grid / grid.max()
    return _EPV


def epv_location(points, scenario):
    """Normalised EPV at corner-origin points [..., 2]; the grid attacks toward +x."""
    grid = epv_grid() if scenario.attack_direction == 1 else np.fliplr(epv_grid())
    rows, cols = grid.shape
    p = np.asarray(points, dtype=float)
    j = np.clip((p[..., 0] / scenario.pitch_length * cols).astype(int), 0, cols - 1)
    i = np.clip((p[..., 1] / scenario.pitch_width * rows).astype(int), 0, rows - 1)
    return grid[i, j]


def distance_from_rest(seconds):
    """Metres a player covers from rest in `seconds` under the agile limits."""
    t = np.maximum(np.asarray(seconds, dtype=float), 0.0)
    a, top = RACE_PLAYER["speed_up"], RACE_PLAYER["max_speed"]
    t_top = top / a
    return np.where(t <= t_top, 0.5 * a * t * t, 0.5 * a * t_top * t_top + top * (t - t_top))


def threat_epv_time(target, support, defenders, defender_velocities, receiver, receiver_velocity,
                    scenario):
    target = np.asarray(target, dtype=float)
    support = np.asarray(support, dtype=float)
    defenders = np.asarray(defenders, dtype=float)
    t_def = reach_time(defenders, defender_velocities, target[..., None, :], RACE_PLAYER).min(axis=-1)
    t_rec = reach_time(receiver, receiver_velocity, target, RACE_PLAYER)
    room = 1 - np.exp(-distance_from_rest(t_def - t_rec) / 9.0)
    to_support = np.linalg.norm(support[..., None, :] - defenders, axis=-1).min(axis=-1)
    support_room = 1 - np.exp(-to_support / 9.0)
    return epv_location(target, scenario) * (
        0.55 + 0.25 * room + 0.2 * epv_location(support, scenario) * support_room)


def threat_of(threat, target, support, defenders, defender_velocities, receiver, receiver_velocity,
              scenario):
    if threat == "andrew":
        return possession_value_all(target, support, defenders, scenario)
    if threat == "epv_time":
        return threat_epv_time(target, support, defenders, defender_velocities, receiver,
                               receiver_velocity, scenario)
    raise ValueError(f"unknown threat {threat!r}; one of {THREATS}")


def positional_threat_all(ball, support, defenders, scenario):
    """`value.positional_threat` with room measured to the nearest defender.

    ball, support: [..., 2]. defenders: [..., m, 2], m >= 1.
    With m == 1 this is the imported function exactly.
    """
    ball = np.asarray(ball, dtype=float)
    support = np.asarray(support, dtype=float)
    defenders = np.asarray(defenders, dtype=float)
    to_ball = np.linalg.norm(ball[..., None, :] - defenders, axis=-1).min(axis=-1)
    to_support = np.linalg.norm(support[..., None, :] - defenders, axis=-1).min(axis=-1)
    room = 1 - np.exp(-to_ball / 9.0)
    support_room = 1 - np.exp(-to_support / 9.0)
    return 0.2 + 0.8 * location_core(ball, scenario) * (
        0.55 + 0.25 * room + 0.2 * location_core(support, scenario) * support_room)


def possession_value_all(ball, support, defenders, scenario):
    """`payoff.possession_value`'s contract, for a defender set."""
    result = np.asarray(positional_threat_all(ball, support, defenders, scenario), dtype=float)
    shape = np.broadcast_shapes(np.shape(ball)[:-1], np.shape(support)[:-1],
                                np.shape(defenders)[:-2])
    result = np.broadcast_to(result, shape)
    if not np.isfinite(result).all() or np.any(result < 0):
        raise ValueError("possession value must be finite and nonnegative")
    return result


def defence(defender, defender_velocity, background, background_velocity):
    """Controlled defender first, then the background: [..., 1 + m, 2] each."""
    defender = np.asarray(defender, dtype=float)
    defender_velocity = np.asarray(defender_velocity, dtype=float)
    background = np.asarray(background, dtype=float).reshape(-1, 2)
    background_velocity = np.asarray(background_velocity, dtype=float).reshape(-1, 2)
    if background.shape != background_velocity.shape:
        raise ValueError("one velocity per background defender")
    lead = np.broadcast_shapes(defender.shape[:-1], defender_velocity.shape[:-1])
    positions = np.concatenate(
        [np.broadcast_to(defender, lead + (2,))[..., None, :],
         np.broadcast_to(background, lead + background.shape)], axis=-2)
    velocities = np.concatenate(
        [np.broadcast_to(defender_velocity, lead + (2,))[..., None, :],
         np.broadcast_to(background_velocity, lead + background.shape)], axis=-2)
    return positions, velocities


def offside_flags(scenario, receiver, ball, defenders):
    """Full-game rule with two or more defenders; the imported reduced rule with one."""
    receiver = np.asarray(receiver, dtype=float)
    ball = np.asarray(ball, dtype=float)
    if defenders.shape[-2] >= 2:
        return offside(receiver[..., 0], ball[..., 0], defenders[..., 0],
                       scenario.attack_direction, scenario.pitch_length)
    return offside_against_line(receiver[..., 0], ball[..., 0], defenders[..., 0, 0],
                                scenario.attack_direction, scenario.pitch_length)


def release_payoffs_with_background(scenario, config, model, carrier, receiver, defender,
                                    carrier_velocity, receiver_velocity, defender_velocity,
                                    background, background_velocity, *, return_legal=False,
                                    threat="andrew"):
    """`payoff.release_payoffs`, line for line, with the defence as a set.

    Differences from the imported function, and nothing else:
      - defenders = controlled defender + background
      - offside against that set's second-last defender
      - threat's room to that set's nearest defender
    """
    defenders, velocities = defence(defender, defender_velocity,
                                    background, background_velocity)
    outputs, legalities = [], []
    flagged = (offside_flags(scenario, receiver, carrier, defenders)
               if config.enforce_offside else False)
    for choice in config.passes:
        target = target_of(choice, receiver, scenario.attack_direction, receiver_velocity, scenario)
        legal = inside_pitch(target, scenario.pitch_length, scenario.pitch_width) & ~np.asarray(flagged)
        completion = np.asarray(model.predict(
            carrier, receiver, defenders, carrier_velocity,
            receiver_velocity, velocities,
            target, FAMILIES.index(choice.family), scenario.attack_direction,
            scenario.pitch_length, scenario.pitch_width))
        if not np.isfinite(completion).all() or np.any((completion < 0) | (completion > 1)):
            raise ValueError("the pass model must return finite probabilities in [0, 1]")
        value = threat_of(threat, target, carrier, defenders, velocities, receiver, receiver_velocity,
                          scenario)
        outputs.append(np.where(legal, completion * value, 0.0))
        legalities.append(np.broadcast_to(legal, outputs[-1].shape))
    values = np.stack(outputs, axis=-1)
    return (values, np.stack(legalities, axis=-1)) if return_legal else values


def retention_payoff_with_background(scenario, carrier, receiver, defender, background, *,
                                     threat="andrew", carrier_velocity=None, defender_velocity=None,
                                     background_velocity=None):
    """`payoff.retention_payoff`: the carrier keeps the ball, threat by nearest defender.
    With threat "epv_time" the carrier is the receiver of his own ball (margin = the
    first defender's arrival minus his own reaction), so velocities are needed."""
    if threat == "andrew":
        defenders, _ = defence(defender, np.zeros_like(np.asarray(defender, dtype=float)),
                               background, np.zeros_like(np.asarray(background, dtype=float)))
        return possession_value_all(carrier, receiver, defenders, scenario)
    if carrier_velocity is None or defender_velocity is None or background_velocity is None:
        raise ValueError("retention with threat 'epv_time' needs velocities")
    defenders, velocities = defence(defender, defender_velocity, background, background_velocity)
    return threat_of(threat, carrier, receiver, defenders, velocities, carrier, carrier_velocity, scenario)
