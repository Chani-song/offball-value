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

A COPY of andrew-passer2on1/passer2on1/payoff.py, kept identical so the 2v1
and 3v1 games price a pass the same way. andrew-fixedpasser/tests/test_game.py
fails if the two ever give different numbers.
"""

from __future__ import annotations

import numpy as np

from defensive_positioning.expected_pass import FAMILIES
from defensive_positioning.rules import inside_pitch, offside, offside_against_line
from defensive_positioning.value import location_core


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
                                    background, background_velocity, *, return_legal=False):
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
        target = choice.target(receiver, scenario.attack_direction)
        legal = inside_pitch(target, scenario.pitch_length, scenario.pitch_width) & ~np.asarray(flagged)
        completion = np.asarray(model.predict(
            carrier, receiver, defenders, carrier_velocity,
            receiver_velocity, velocities,
            target, FAMILIES.index(choice.family), scenario.attack_direction,
            scenario.pitch_length, scenario.pitch_width))
        if not np.isfinite(completion).all() or np.any((completion < 0) | (completion > 1)):
            raise ValueError("the pass model must return finite probabilities in [0, 1]")
        threat = possession_value_all(target, carrier, defenders, scenario)
        outputs.append(np.where(legal, completion * threat, 0.0))
        legalities.append(np.broadcast_to(legal, outputs[-1].shape))
    values = np.stack(outputs, axis=-1)
    return (values, np.stack(legalities, axis=-1)) if return_legal else values


def retention_payoff_with_background(scenario, carrier, receiver, defender, background):
    """`payoff.retention_payoff`: the carrier keeps the ball, threat by nearest defender."""
    defenders, _ = defence(defender, np.zeros_like(np.asarray(defender, dtype=float)),
                           background, np.zeros_like(np.asarray(background, dtype=float)))
    return possession_value_all(carrier, receiver, defenders, scenario)
