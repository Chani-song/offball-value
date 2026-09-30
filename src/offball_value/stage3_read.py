"""Reading the imported solver's output in football terms.

The solver answers in compass moves -- hold, or step east, north, west,
south -- because that is its action set. The question being asked of it is
whether a defender goes to the ball, goes with the runner, drops toward his
goal, or holds. `name_move` translates one into the other: each compass move
is named for whichever of those lines it points along most, by cosine, with
"sideways" when it points along none of them by at least 0.3.

A state counts as SPLIT when even the defender's likeliest move is below
`below` (0.8). Exact equilibria can put 99.5% on one move and 0.5% on
another; a has-more-than-one-positive-probability test calls that torn, and
on the pitch it is a decision. Measured on the first 167 scenes this halved
the apparent dilemma rate and replaced half the top fifteen.

At the root the moves are also merged by name, because a defender split
55/45 between east and south may be heading for the ball either way --
choosing an angle, not choosing between the ball and the man. `fsplit0` is
1 minus the heaviest merged share. Later steps have no positions saved
alongside their policies, so they are split by compass only.
"""

from __future__ import annotations

import math

import numpy as np

from .agile_motion import ANDREW, command_direction, command_names, describe, layers_for, relative_commands
from .run_passes import passes_named, target_of

SOLVER_DIRS = ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0))


def world_direction(k: int, attack_direction: int) -> tuple[float, float]:
    """Compass command k as a direction on the pitch.

    The solver's commands are relative to the attack: `motion.advance` sets
    the desired velocity to directions[k] * max_speed * attack_direction, so
    when the attack runs toward decreasing x every command is turned half a
    circle. Reading directions[k] as absolute gets every scene attacking that
    way backwards -- checked against 1,529 sampled defender moves, the
    unturned reading agreed with the actual displacement 6% of the time on
    those scenes, the turned one as often as on the others.
    """
    ux, uy = SOLVER_DIRS[k]
    return (ux * attack_direction, uy * attack_direction)


def name_move_targets(u, defender, targets) -> str:
    """Name compass move u for whichever target it points along most.

    targets: ordered (name, position) pairs. Ties go to the later target, as
    they always have. "sideways" when no cosine reaches 0.3.
    """
    if tuple(u) == (0.0, 0.0):
        # a target velocity of zero: brake, not "stand" -- a running defender
        # keeps sliding the same way while he slows
        return "slow down"
    best, score = "sideways", 0.3
    for name, tgt in targets:
        vx, vy = tgt[0] - defender[0], tgt[1] - defender[1]
        n = math.hypot(vx, vy)
        if n > 1e-6:
            c = (u[0] * vx + u[1] * vy) / n
            if c >= score:
                best, score = name, c
    return best


def name_move(u, defender, carrier, runner, goal) -> str:
    return name_move_targets(u, defender, (("toward ball", carrier), ("toward runner", runner),
                                           ("toward goal", goal)))


def config_from_manifest(manifest: dict):
    """The solved study's GameConfig, pass set included (run passes since 2026-09-27)."""
    from defensive_positioning.models import GameConfig
    return GameConfig(**{k: v for k, v in manifest["config"].items()
                         if k in ("steps", "step_seconds", "physics_step")},
                      passes=passes_named(manifest.get("passes", "andrew")))


def commands_for(record: dict, config, manifest: dict, kind: str):
    """The study's relative command sets for one starting record, or None for the compass."""
    return relative_commands(record, config, kind) if manifest.get("commands") == "relative" else None


def study_kind(snapshot: dict) -> str:
    """'3v1' when the rollout names a scripted passer, '2v1' otherwise."""
    return "3v1" if "passer" in snapshot else "2v1"


def targets_for(snapshot: dict, goal) -> tuple:
    """What a defender's move can be toward, in the order ties resolve.

    2v1: the ball carrier IS the beneficiary, so "toward the ball" covers him.
    3v1: the ball (scripted passer), the runner and the beneficiary are three.
    """
    if study_kind(snapshot) == "3v1":
        return (("toward ball", snapshot["passer"]), ("toward runner", snapshot["runner"]),
                ("toward beneficiary", snapshot["beneficiary"]), ("toward goal", goal))
    return (("toward ball", snapshot["carrier"]), ("toward runner", snapshot["receiver"]),
            ("toward goal", goal))


def _goal(scenario: dict):
    return (float(scenario["pitch_length"]) if int(scenario["attack_direction"]) == 1 else 0.0,
            float(scenario["pitch_width"]) / 2.0)


def choices_at(policy, snapshot: dict, scenario: dict) -> list[tuple[int, str, float]]:
    """(compass index, football name, probability) for one defender policy row."""
    ad = int(scenario["attack_direction"])
    targets = targets_for(snapshot, _goal(scenario))
    out = [(k, name_move_targets(world_direction(k, ad), snapshot["defender"], targets), float(p))
           for k, p in enumerate(policy) if p >= 0.01]
    return sorted(out, key=lambda t: -t[2])


def root_choices(state: dict) -> list[tuple[int, str, float]]:
    """(compass index, football name, probability), heaviest first."""
    return choices_at(state["root_defender"], state["rollouts"][0][0], state["scenario"])


def merged(choices) -> list[tuple[str, float]]:
    agg: dict[str, float] = {}
    for _, name, p in choices:
        agg[name] = agg.get(name, 0.0) + p
    return sorted(agg.items(), key=lambda kv: -kv[1])


def split_by_step(policy_npz, steps: int, below: float = 0.8) -> list[dict]:
    """Per decision step: share of reachable states split, loosely mixed, and
    the mean probability of the defender's likeliest move."""
    out = []
    with np.load(policy_npz) as z:
        for k in range(steps):
            d = z[f"defender_{k}"].reshape(-1, len(SOLVER_DIRS))
            live = d.sum(-1) > 1e-9
            p = d[live]
            if not len(p):
                out.append({"split": float("nan"), "loose": float("nan"),
                            "top": float("nan"), "n": 0})
                continue
            top = p.max(-1)
            out.append({"split": float((top < below).mean()),
                        "loose": float(((p > 1e-9).sum(-1) > 1).mean()),
                        "top": float(top.mean()), "n": int(live.sum())})
    return out


def representative_rollout(state: dict):
    """The sampled play the pages animate: the first whose defender opens with
    the likeliest root move, so what is shown is the typical line of play."""
    rolls = state["rollouts"]
    root = choices_at(state["root_defender"], rolls[0][0], state["scenario"])
    want = root[0][0] if root else None
    return next((r for r in rolls if r and r[0].get("commands", {}).get("defender") == want),
                rolls[0])


def check_layers(layers, z, config) -> None:
    """The rebuilt layers must be the solver's: same state count per body at
    every instant. Movement built with another physics than the solve used
    gives other states, and reading a policy through them would silently
    return another state's answer -- this fails instead."""
    for k in range(config.steps + 1):
        rebuilt = tuple(len(layer[k]) for layer in layers)
        saved = (z[f"defender_{k}"].shape[:3] if k < config.steps
                 else z["terminal_release"].shape)
        if rebuilt != tuple(saved):
            raise RuntimeError(f"rebuilt movement layers {rebuilt} do not match the saved "
                               f"policy {tuple(saved)} at step {k}: wrong physics?")


def check_commands(state: dict, commands) -> None:
    """The layers must be rebuilt with the command sets the solve used."""
    solved_with = state.get("commands", "compass")
    rebuilt = command_names(commands) or "compass"
    if solved_with != rebuilt:
        raise RuntimeError(f"solved with commands {solved_with}, rebuilding with {rebuilt}")


def check_physics(state: dict, layers, physics) -> None:
    """The layers must be rebuilt with the physics the solve used. Two checks:
    the physics recorded in the solved state (outputs from before the field
    existed were all solved with the imported motion), and every saved
    rollout walked through the rebuilt layers landing where the solver put
    it. State counts alone cannot tell two physics apart."""
    solved_with = state.get("physics", ANDREW)
    if solved_with != describe(physics):
        raise RuntimeError(f"solved with physics {solved_with}, rebuilding with {describe(physics)}")
    for roll in state.get("rollouts", []):
        keys = (("runner", "beneficiary", "defender") if study_kind(roll[0]) == "3v1"
                else ("carrier", "receiver", "defender"))
        index = (0, 0, 0)
        for k, snap in enumerate(roll):
            for slot, key in enumerate(keys):
                err = float(np.hypot(*(layers[slot][k].position[index[slot]] - np.asarray(snap[key]))))
                if err > 1e-6:
                    raise RuntimeError(f"rebuilt layers disagree with a saved rollout at step {k}, "
                                       f"{key}: {err:.2e} m -- wrong physics?")
            if snap.get("event") != "move":
                break
            cmd = snap["commands"]
            index = tuple(int(layers[s][k].successor[index[s], cmd[key]]) for s, key in enumerate(keys))


def decision_path(state: dict, policy_npz, config, physics=None, commands=None) -> dict:
    """The defender's policy at every decision point of the representative play.

    The saved policies hold an answer for every state; the play visits a few.
    To know which, the three strategic bodies' movement layers are rebuilt --
    cheap, a fraction of a second, because they carry no payoffs -- and walked
    with the play's own commands. Each visited state's position is checked
    against the play's snapshot, so a mismatch fails loudly instead of reading
    another state's answer.
    """
    from defensive_positioning.equilibrium_clips import scenario_from

    scenario = state["scenario"]
    sc = scenario_from(scenario)
    layers = layers_for(sc, config, physics, commands)
    check_physics(state, layers, physics)
    check_commands(state, commands)
    roll = representative_rollout(state)
    kind = study_kind(roll[0])
    slot_keys = (("runner", "beneficiary", "defender") if kind == "3v1"
                 else ("carrier", "receiver", "defender"))
    points, index = [], (0, 0, 0)
    with np.load(policy_npz) as z:
        check_layers(layers, z, config)
        for k, snap in enumerate(roll):
            for slot, key in enumerate(slot_keys):
                err = float(np.hypot(*(layers[slot][k].position[index[slot]] - np.asarray(snap[key]))))
                if err > 1e-6:
                    raise RuntimeError(f"rebuilt layer disagrees with the play at step {k}, "
                                       f"{key}: {err:.2e} m")
            if snap.get("event") != "move":
                break
            policy = z[f"defender_{k}"][index]
            attack = z[f"attack_{k}"][index]
            ch = choices_at(policy, snap, scenario)
            agg = merged(ch)
            points.append({"step": k, "t": float(snap["time"]),
                           "choices": [(int(i), n, round(p, 4)) for i, n, p in ch],
                           "merged": [(n, round(p, 4)) for n, p in agg],
                           "split": 1.0 - agg[0][1] if agg else 0.0,
                           "release_probability": round(float(attack[-1]), 4)})
            cmd = snap["commands"]
            moves = tuple(cmd[key] for key in slot_keys)
            index = tuple(int(layers[s][k].successor[index[s], moves[s]]) for s in range(3))
    last = roll[-1]
    end = {"event": last.get("event"), "t": float(last["time"]), "to": last.get("to"),
           "family": last.get("family"), "target": last.get("target")}
    return {"kind": kind, "points": points, "end": end, "rollout": roll,
            "max_split": max((p["split"] for p in points), default=0.0)}


def modal_path(state: dict, policy_npz, config, passer_track=None, physics=None, commands=None) -> dict:
    """The most likely line of play: at every decision both sides take the
    action their equilibrium policy weights most.

    The sampled rollouts are honest draws from mixed strategies, and a draw
    can land on a 23% move -- shown against a caption naming a 36% one, that
    reads as the page contradicting itself. The modal line never does: the
    defender always takes the heaviest arrow on screen. It is an
    illustration of the policy, not a sample from it, and is labelled so.

    Positions come from the rebuilt movement layers; releases are decoded from
    the saved pass choices (receiver * passes + choice in the 3v1 game). For
    the 3v1 game `passer_track` [steps + 1, 2] supplies the scripted passer.
    """
    from defensive_positioning.equilibrium_clips import scenario_from

    scenario = state["scenario"]
    sc = scenario_from(scenario)
    layers = layers_for(sc, config, physics, commands)
    check_physics(state, layers, physics)
    check_commands(state, commands)
    kind = study_kind(state["rollouts"][0][0])
    stakes_by_step = {m["step"]: m for m in state.get("modal_line", [])}
    defender_cmds = (commands or {}).get("defender")
    actions = len(config.directions)
    passes = config.passes
    points, snaps, index = [], [], (0, 0, 0)
    end = None
    with np.load(policy_npz) as z:
        check_layers(layers, z, config)
        for k in range(config.steps + 1):
            snap = {"time": float(config.times[k]), "defender": layers[2][k].position[index[2]].tolist(),
                    "defender_velocity": layers[2][k].velocity[index[2]].tolist()}
            if kind == "3v1":
                snap.update(runner=layers[0][k].position[index[0]].tolist(),
                            beneficiary=layers[1][k].position[index[1]].tolist(),
                            passer=list(map(float, passer_track[k])))
            else:
                snap.update(carrier=layers[0][k].position[index[0]].tolist(),
                            receiver=layers[1][k].position[index[1]].tolist())
            snaps.append(snap)
            code = int(z[f"pass_index_{k}"][index])
            if k == config.steps:
                if bool(z["terminal_release"][index]) and code >= 0:
                    end = ("release", code)
                else:
                    end = ("no_pass" if kind == "3v1" else "retain", None)
                break
            dpol, apol = z[f"defender_{k}"][index], z[f"attack_{k}"][index]
            if defender_cmds is not None:
                # relative commands name themselves; their directions come from this state
                d_pos, d_vel = layers[2][k].position[index[2]], layers[2][k].velocity[index[2]]
                ch = sorted([(int(i), defender_cmds[i].name, float(p)) for i, p in enumerate(dpol) if p >= 0.01],
                            key=lambda t: -t[2])
                directions = [command_direction(c, d_pos, d_vel, k, config, sc, sc.defender.maximum_speed).tolist()
                              for c in defender_cmds]
            else:
                ch = choices_at(dpol, snap, scenario)
                directions = [list(world_direction(i, int(scenario["attack_direction"]))) for i in range(len(dpol))]
            agg = merged(ch)
            defend = int(np.argmax(dpol))
            attack = int(np.argmax(apol))
            m = stakes_by_step.get(k)
            stakes = None
            if m is not None:
                cs, rs = m["carrier_slot_values"], m["receiver_slot_values"]
                stakes = {"defender": m["defender_pure_loss"],
                          ("runner" if kind == "3v1" else "carrier"): max(cs) - min(cs),
                          ("beneficiary" if kind == "3v1" else "runner"): max(rs) - min(rs)}
            points.append({"step": k, "t": snap["time"],
                           "choices": [(int(i), n, round(p, 4)) for i, n, p in ch],
                           "merged": [(n, round(p, 4)) for n, p in agg],
                           "split": 1.0 - agg[0][1] if agg else 0.0,
                           "chosen": int(defend),
                           "chosen_name": next((n for i, n, _ in ch if i == defend), "?"),
                           "directions": [[round(x, 4) for x in u] for u in directions],
                           "stakes": stakes,
                           "release_probability": round(float(apol[-1]), 4)})
            if attack == actions ** 2 and code >= 0:
                end = ("release", code)
                break
            first, second = divmod(attack, actions)
            index = (int(layers[0][k].successor[index[0], first]),
                     int(layers[1][k].successor[index[1], second]),
                     int(layers[2][k].successor[index[2], defend]))
    last = snaps[-1]
    out_end = {"event": end[0], "t": last["time"], "to": None, "family": None, "target": None}
    if end[0] == "release":
        who = ("runner", "beneficiary")[end[1] // len(passes)] if kind == "3v1" else "runner"
        choice = passes[end[1] % len(passes)]
        slot = (0 if who == "runner" else 1) if kind == "3v1" else 1
        k_end = len(snaps) - 1
        receiver_pos = layers[slot][k_end].position[index[slot]]
        receiver_vel = layers[slot][k_end].velocity[index[slot]]
        target = target_of(choice, receiver_pos, int(scenario["attack_direction"]), receiver_vel, sc)
        out_end.update(to=who, family=choice.family, target=np.asarray(target).tolist())
    return {"kind": kind, "points": points, "end": out_end, "rollout": snaps,
            "max_split": max((p["split"] for p in points), default=0.0), "line": "modal"}
