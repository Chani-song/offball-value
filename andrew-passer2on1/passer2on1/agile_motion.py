"""Agile movement for the imported solver: our earlier prototype's motion
model, in the solver's command form.

Why. The imported model caps every change of velocity at one 3.8 m/s^2 --
speeding up, braking and turning alike -- a value tuned on synthetic states.
A defender who picks a new direction then ends a 0.6 s turn at most
0.5 * 3.8 * 0.6^2 ~ 0.7 m from where his old command would have put him, so
his decisions barely show. The local-game prototype we used before adopting
the solver (src/offball_value/steering_reachable.py,
defender_trajectory_search.py) separated the three. The reviewer, having
used it, chose its values (2026-09-25):

    speeding up       4.5 m/s^2   tangential, forward
    braking           6.0 m/s^2   tangential, backward
    turning           6.0 m/s^2   normal
    plant-and-cut     brake to a stop at up to 9.0 m/s^2, stand 0.1 s, go
                      again -- when running at 2 m/s or more and the new
                      direction is at least 45 deg from the current one
    reaction delay    0.2 s, the defender only, at the run's onset
    top speed         9.0 m/s, except the 2v1 carrier on the ball at 7.2.
                      Top speed is per body, in the scenario, not here.

The prototype gives these numbers no documented data source either. They are
the reviewer's choice from having used it, recorded as such.

Commands are unchanged. A command still means "go to full speed in this
compass direction": desired velocity = direction x max speed x attack
direction. Each physics step moves the velocity toward the desired one, now
capped by an ellipse in the player's own frame instead of a circle. The
prototype's control set (_moving_controls) is the boundary of this same
ellipse: 4.5 forward, 6.0 backward, 6.0 sideways. With all three equal the
ellipse is the imported circle, and the tests check that it reproduces the
imported motion.

Plant-and-cut. The prototype enumerated it as a separate manoeuvre with
variants: braking 6 / 7.5 / 9, plant 0.1 / 0.2 / 0.3 s, re-acceleration
2.25 / 4.5. Every command here is full effort, so the full-effort member is
used: 9.0, 0.1 s, 4.5. When it applies, both ways of carrying out the
command are simulated over the turn. The one ending the turn with its
velocity nearer the commanded velocity is taken, which is the quantity the
imported controller steers by. In practice the plant wins sharp reversals and
the curve wins moderate cuts. A plant that straddles the end of a turn does
not carry its remaining standing time over: the next turn starts from the
state reached.

Reaction delay. For the first 0.2 s after onset the defender keeps his onset
velocity, and his first command acts from then on. It applies only at onset.
Later on, the game's simultaneous turns already stop him seeing an attacking
change before the next decision, so another delay would count that reaction
twice.

Everything else is the imported build_layers: the same successor structure,
exact merging of identical states, pitch clipping, and paths for tackle
integration.

This file is kept byte-identical in andrew-passer2on1/passer2on1,
andrew-fixedpasser/fixedpasser and src/offball_value (the review pages
rebuild layers with it). andrew-passer2on1/tests/test_agile_motion.py
checks that.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from defensive_positioning.models import GameConfig, PlayerState, Scenario
from defensive_positioning.motion import Layer
from defensive_positioning.motion import build_layers as imported_build_layers


@dataclass(frozen=True)
class Physics:
    name: str
    speed_up: float
    braking: float
    turning: float
    plant_cut: bool
    plant_min_speed: float
    plant_min_angle_deg: float
    plant_braking: float
    plant_seconds: float
    defender_delay_s: float

    def as_dict(self) -> dict:
        return asdict(self)


AGILE = Physics(name="agile", speed_up=4.5, braking=6.0, turning=6.0, plant_cut=True,
                plant_min_speed=2.0, plant_min_angle_deg=45.0, plant_braking=9.0,
                plant_seconds=0.1, defender_delay_s=0.2)
ANDREW = {"name": "andrew"}


def resolve(physics) -> Physics | None:
    """None or "andrew" -> None (the imported motion, untouched); "agile", a
    Physics, or its dict (as stored in a manifest) -> Physics."""
    if physics is None or physics == "andrew":
        return None
    if isinstance(physics, Physics):
        return physics
    if physics == "agile":
        return AGILE
    if isinstance(physics, dict):
        return None if physics.get("name") == "andrew" else Physics(**physics)
    raise ValueError(f"unknown physics {physics!r}")


def describe(physics) -> dict:
    p = resolve(physics)
    return dict(ANDREW) if p is None else p.as_dict()


def _steps(seconds: float, config: GameConfig) -> int:
    n = seconds / config.physics_step
    if not np.isclose(n, round(n), atol=1e-9, rtol=0):
        raise ValueError(f"{seconds} s is not a multiple of the physics step")
    return int(round(n))


def steer(v, desired, physics: Physics, dt: float):
    """One physics step toward `desired`, the change capped by the ellipse."""
    change = desired - v
    norm = float(np.linalg.norm(change))
    if norm <= 1e-15:
        return v + change
    speed = float(np.linalg.norm(v))
    if speed <= 1e-12:
        # from rest every change is speeding up
        q = norm / (physics.speed_up * dt)
    else:
        t = v / speed
        along = float(change @ t)
        side = float(np.linalg.norm(change - along * t))
        cap = physics.speed_up if along >= 0.0 else physics.braking
        q = math.hypot(along / (cap * dt), side / (physics.turning * dt))
    return v + change * min(1.0, 1.0 / q)


def _plant_applies(v, desired, physics: Physics) -> bool:
    speed, want = float(np.linalg.norm(v)), float(np.linalg.norm(desired))
    if not physics.plant_cut or speed < physics.plant_min_speed or want <= 1e-12:
        return False
    cosine = float(v @ desired) / (speed * want)
    return cosine <= math.cos(math.radians(physics.plant_min_angle_deg)) + 1e-12


def _run(position, velocity, desired, scenario, config, physics, hold, plant):
    dt = config.physics_step
    p, v = np.array(position, dtype=float), np.array(velocity, dtype=float)
    path = [p.copy()]
    phase, standing = None, 0
    stand_steps = _steps(physics.plant_seconds, config)
    for i in range(_steps(config.step_seconds, config)):
        if i >= hold:
            if plant and phase is None:
                if not _plant_applies(v, desired, physics):
                    return None
                phase = "brake"
            if phase == "brake":
                speed = float(np.linalg.norm(v))
                drop = physics.plant_braking * dt
                if speed <= drop:
                    v, phase = np.zeros(2), "stand"
                else:
                    v = v * (1.0 - drop / speed)
            elif phase == "stand":
                standing += 1
                if standing >= stand_steps:
                    phase = "go"
            else:
                v = steer(v, desired, physics, dt)
        proposed = p + v * dt
        p = np.clip(proposed, [0.0, 0.0], [scenario.pitch_length, scenario.pitch_width])
        v = np.where(p != proposed, 0.0, v)
        path.append(p.copy())
    return p, v, np.array(path)


def advance(position, velocity, player: PlayerState, command: int, scenario: Scenario,
            config: GameConfig, physics: Physics, hold_s: float = 0.0, command_set=None, k: int = 0):
    """The imported motion.advance with the agile limits. Same return.

    With `command_set` (this body's relative commands) the desired velocity is
    the command's direction from this state at full speed; without, the compass."""
    if command_set is None:
        desired = (np.asarray(config.directions[command]) * player.maximum_speed
                   * scenario.attack_direction)
    else:
        desired = command_direction(command_set[command], position, velocity, k, config, scenario,
                                    player.maximum_speed) * player.maximum_speed
    hold = _steps(hold_s, config)
    curve = _run(position, velocity, desired, scenario, config, physics, hold, plant=False)
    cut = _run(position, velocity, desired, scenario, config, physics, hold, plant=True)
    if cut is not None and (np.linalg.norm(cut[1] - desired)
                            < np.linalg.norm(curve[1] - desired) - 1e-12):
        return cut
    return curve


def build_layers(player: PlayerState, scenario: Scenario, config: GameConfig,
                 physics=None, delay_s: float = 0.0, command_set=None):
    """motion.build_layers; with physics None and no command set it IS the imported function.
    `command_set`: this body's relative commands (one per compass slot), or None."""
    physics = resolve(physics)
    if command_set is not None:
        if len(command_set) != len(config.directions):
            raise ValueError("a relative command set must have one command per compass slot")
        if command_set[0].kind != "stop":
            raise ValueError("command 0 must be the stop; the layers start every body on it")
        if physics is None:
            raise ValueError("relative commands need an explicit physics (use 'agile')")
    if physics is None:
        if delay_s:
            raise ValueError("the imported motion has no reaction delay")
        return imported_build_layers(player, scenario, config)
    if delay_s > config.step_seconds + 1e-12:
        raise ValueError("the reaction delay is longer than the first turn")
    layers = [Layer(np.array([player.position]), np.array([player.velocity]),
                    np.array([0], dtype=int))]
    actions = len(config.directions)
    for k in range(config.steps):
        current = layers[-1]
        seen, positions, velocities, commands, paths = {}, [], [], [], []
        successor = np.empty((len(current), actions), dtype=int)
        for state in range(len(current)):
            state_paths = []
            for action in range(actions):
                p, v, path = advance(current.position[state], current.velocity[state],
                                     player, action, scenario, config, physics,
                                     hold_s=delay_s if k == 0 else 0.0, command_set=command_set, k=k)
                key = (*p, *v, action)
                if key not in seen:
                    seen[key] = len(positions)
                    positions.append(p)
                    velocities.append(v)
                    commands.append(action)
                successor[state, action] = seen[key]
                state_paths.append(path)
            paths.append(state_paths)
        current.successor = successor
        current.paths = np.array(paths)
        layers.append(Layer(np.array(positions), np.array(velocities), np.array(commands)))
        if len(layers[-1]) > config.max_joint_states:
            raise ValueError("one player's state count already exceeds max_joint_states")
    return layers


def layers_for(scenario: Scenario, config: GameConfig, physics=None, commands=None):
    """The three strategic bodies' layers in slot order (carrier, receiver,
    defender). The reaction delay goes to the defender slot only. `commands`:
    relative command sets by slot (relative_commands), or None for the compass."""
    physics = resolve(physics)
    delay = physics.defender_delay_s if physics is not None else 0.0
    cmd = commands or {}
    return (build_layers(scenario.carrier, scenario, config, physics, command_set=cmd.get("carrier")),
            build_layers(scenario.receiver, scenario, config, physics, command_set=cmd.get("receiver")),
            build_layers(scenario.defender, scenario, config, physics, delay_s=delay,
                         command_set=cmd.get("defender")))


# ---------------------------------------------------------------------------
# Relative commands (2026-09-27, meeting of 2026-09-25 and the reviewer's rule)
#
# Compass commands have no diagonals and no meaning: "west" is only "toward the
# runner" by accident, and the page had to guess names for them. A relative
# command names a football intention and turns it into a direction FROM THE
# BODY'S OWN STATE, so the solver's structure (each body's states built on its
# own, then combined) is untouched. Every anchor is fixed when the game is
# built: another body's position is its onset position carried on at its onset
# velocity, the 3v1 passer is his real track, the goal is the goal. That is
# my approximation, chosen to keep the bodies independent; over 1.8 s the
# extrapolation is close.
#
# The runner-related defender command is INTERCEPT, not pursuit. It aims at
# the earliest point on the runner's projected line (onset position + onset
# velocity x t) that the defender can reach by the time the runner gets there,
# or at the line's end within the horizon if he cannot. A defender told to
# intercept a run in behind therefore drops toward the space the run is going
# to, never at the runner's back.
#
# Command 0 is always the stop, as the compass had it: the layer builder starts
# every body on command 0.

GOAL_HALF_WIDTH = 3.66


@dataclass(frozen=True)
class Command:
    """One relative command of one body. `anchor` holds what the kind needs:
    toward/between: [steps + 1, 2] points per instant; intercept: (origin,
    velocity) of the run line; fixed: a unit vector; stop/continue: nothing."""
    name: str
    kind: str
    anchor: tuple = ()

    def as_dict(self) -> dict:
        return {"name": self.name, "kind": self.kind}


def goal_of(scenario: Scenario):
    return np.array([scenario.pitch_length if scenario.attack_direction == 1 else 0.0,
                     scenario.pitch_width / 2.0])


def _unit(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else np.zeros(2)


def _clip(p, scenario: Scenario):
    return np.clip(p, [0.0, 0.0], [scenario.pitch_length, scenario.pitch_width])


def intercept_point(position, k, run_origin, run_velocity, config: GameConfig, max_speed: float,
                    scenario: Scenario):
    """Earliest point on the run line the body can reach in time, else its end."""
    t0 = config.times[k]
    horizon = config.steps * config.step_seconds
    for t in np.arange(config.times[min(k + 1, config.steps)], horizon + 1e-9, 0.1):
        point = _clip(run_origin + run_velocity * t, scenario)
        if np.linalg.norm(point - position) <= max_speed * (t - t0):
            return point
    return _clip(run_origin + run_velocity * horizon, scenario)


def command_direction(command: Command, position, velocity, k: int, config: GameConfig,
                      scenario: Scenario, max_speed: float):
    """Unit direction (zero for the stop) of `command` for a body at this state."""
    p = np.asarray(position, dtype=float)
    if command.kind == "stop":
        return np.zeros(2)
    if command.kind == "continue":
        u = _unit(np.asarray(velocity, dtype=float))
        return u if np.linalg.norm(velocity) >= 0.5 else np.array([float(scenario.attack_direction), 0.0])
    if command.kind == "fixed":
        return np.asarray(command.anchor, dtype=float)
    ahead = min(k + 1, config.steps)
    if command.kind == "toward":
        target = np.asarray(command.anchor[ahead], dtype=float)
    elif command.kind == "intercept":
        origin, run = (np.asarray(a, dtype=float) for a in command.anchor)
        target = intercept_point(p, k, origin, run, config, max_speed, scenario)
    elif command.kind == "between":
        first, second = command.anchor
        a = np.asarray(first[ahead], dtype=float)
        if isinstance(second, Command):
            origin, run = (np.asarray(x, dtype=float) for x in second.anchor)
            b = intercept_point(p, k, origin, run, config, max_speed, scenario)
        else:
            b = np.asarray(second[ahead], dtype=float)
        target = 0.5 * (a + b)
    else:
        raise ValueError(f"unknown command kind {command.kind}")
    gap = target - p
    return _unit(gap) if np.linalg.norm(gap) > 0.5 else np.zeros(2)


def _track(position, velocity, config: GameConfig, scenario: Scenario):
    """[steps + 1, 2]: a body carried on at its onset velocity, kept on the pitch."""
    p, v = np.asarray(position, dtype=float), np.asarray(velocity, dtype=float)
    return tuple(tuple(_clip(p + v * t, scenario)) for t in config.times)


def _lateral_away(run_velocity, from_point, of_point, scenario: Scenario):
    """Unit vector perpendicular to the run, on the side away from `from_point`."""
    u = _unit(np.asarray(run_velocity, dtype=float))
    if np.linalg.norm(u) < 1e-9:
        u = np.array([float(scenario.attack_direction), 0.0])
    n = np.array([-u[1], u[0]])
    return tuple(n if np.dot(n, np.asarray(of_point) - np.asarray(from_point)) >= 0 else -n)


def relative_commands(record: dict, config: GameConfig, kind: str) -> dict:
    """Command sets by solver slot (carrier, receiver, defender) for one record.

    kind "2v1": carrier slot = the ball carrier, receiver slot = the runner.
    kind "3v1": carrier slot = the runner, receiver slot = the beneficiary,
    the passer scripted from record["passer"]. Deterministic in the record, so
    the readers rebuild the same sets.
    """
    from defensive_positioning.equilibrium_clips import scenario_from
    sc = scenario_from(record["scenario"])
    goal = tuple(tuple(goal_of(sc)) for _ in config.times)
    stop = Command("stop", "stop")
    cont = Command("keep going", "continue")
    to_goal = Command("toward goal", "toward", goal)
    d0 = np.asarray(sc.defender.position, dtype=float)
    if kind == "2v1":
        ball_track = _track(sc.carrier.position, sc.carrier.velocity, config, sc)
        runner = sc.receiver
        run_line = (tuple(runner.position), tuple(runner.velocity))
        intercept = Command("cut off runner", "intercept", run_line)
        carrier_cmds = (stop, cont, to_goal,
                        Command("left", "fixed", (0.0, float(sc.attack_direction))),
                        Command("right", "fixed", (0.0, -float(sc.attack_direction))))
        runner_cmds = (stop, cont, to_goal, Command("toward ball", "toward", ball_track),
                       Command("sideways", "fixed", _lateral_away(runner.velocity, d0, runner.position, sc)))
        defender_cmds = (stop, Command("toward ball", "toward", ball_track), intercept,
                         Command("hold between", "between", (ball_track, intercept)), to_goal)
        return {"carrier": carrier_cmds, "receiver": runner_cmds, "defender": defender_cmds}
    if kind == "3v1":
        passer = np.asarray(record["passer"]["positions"], dtype=float)
        if passer.shape[0] < config.steps + 1:
            raise ValueError("passer track shorter than the game")
        ball_track = tuple(tuple(float(x) for x in passer[k]) for k in range(config.steps + 1))
        runner, bene = sc.carrier, sc.receiver
        run_line = (tuple(runner.position), tuple(runner.velocity))
        intercept = Command("cut off runner", "intercept", run_line)
        bene_track = _track(bene.position, bene.velocity, config, sc)
        runner_cmds = (stop, cont, to_goal, Command("toward ball", "toward", ball_track),
                       Command("sideways", "fixed", _lateral_away(runner.velocity, d0, runner.position, sc)))
        bene_cmds = (stop, cont, to_goal, Command("toward ball", "toward", ball_track),
                     Command("sideways", "fixed", _lateral_away(bene.velocity, d0, bene.position, sc)))
        defender_cmds = (stop, intercept, Command("toward beneficiary", "toward", bene_track),
                         Command("hold between", "between", (bene_track, intercept)), to_goal)
        return {"carrier": runner_cmds, "receiver": bene_cmds, "defender": defender_cmds}
    raise ValueError(f"unknown game kind {kind}")


def command_names(commands: dict | None) -> dict | None:
    return None if commands is None else {slot: [c.name for c in cmds] for slot, cmds in commands.items()}
