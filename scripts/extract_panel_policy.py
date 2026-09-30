#!/usr/bin/env python3
"""One game's opening decision, as a figure panel needs it.

For a solved study and one starting record (by its provenance code): each of
the three bodies' position at the start, where each of its five commands puts
it 0.6 s later and how he is moving then (the solver's own movement layers,
momentum included), each
command's aim at the start (the unit direction the solver's movement
steers him toward, zero for the stop), each command's equilibrium probability (the defender's row; for the two attackers
the joint attack policy summed over the other attacker), the probability of a
pass at the start and where it would go, and the game's value.

Runs against whichever code tree is on PYTHONPATH -- the meeting-era tree
(compass commands, axis passes) or the v3 tree (relative commands, run
passes) -- so each study is read with the code that solved it.

Usage:
    PYTHONPATH=<tree>/src python scripts/extract_panel_policy.py \\
        --solved <study> --code S05 --label "meeting" --output panel.json
"""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

import numpy as np

from defensive_positioning.equilibrium_clips import scenario_from
from defensive_positioning.models import GameConfig
from offball_value import agile_motion as am
from offball_value import stage3_read as s3

try:            # v3 tree (2026-09-27): the study's pass set and relative commands
    from offball_value.stage3_read import commands_for, config_from_manifest
    from offball_value.run_passes import target_of
    from offball_value.agile_motion import command_direction
except ImportError:
    commands_for = config_from_manifest = target_of = command_direction = None

SLOTS = ("carrier", "receiver", "defender")
ROLES = {"2v1": ("ball carrier", "runner", "defender"), "3v1": ("runner", "beneficiary", "defender")}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--solved", type=Path, required=True)
    p.add_argument("--code", required=True, help="the starting record's provenance code")
    p.add_argument("--label", required=True)
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def pass_where(choice) -> str:
    """'4 m ahead', 'feet', '4 m left' for an axis pass (attacking axes); 'run +8 / side 0 / goal 0' for a run pass."""
    if not hasattr(choice, "offset"):
        return f"run {choice.along:+g} / side {choice.lateral:+g} / goal {choice.goalward:+g}"
    along, side = (float(x) for x in choice.offset)
    parts = ([f"{abs(along):g} m {'ahead' if along > 0 else 'behind'}"] if along else []) + \
            ([f"{abs(side):g} m {'left' if side > 0 else 'right'}"] if side else [])
    return " ".join(parts) or "feet"


def compass_name(u, direction) -> str:
    """A compass move named against the attack: forward, back, left, right (attack runs +x)."""
    x, y = float(u[0]) * direction, float(u[1]) * direction
    if abs(x) < 1e-9 and abs(y) < 1e-9:
        return "stop"
    if abs(x) >= abs(y):
        return "forward" if x > 0 else "back"
    return "left" if y > 0 else "right"


def study_config(manifest):
    return (config_from_manifest(manifest) if config_from_manifest else
            GameConfig(**{k: v for k, v in manifest["config"].items()
                          if k in ("steps", "step_seconds", "physics_step")}))


def main() -> None:
    args = parse_args()
    manifest = json.loads((args.solved / "manifest.json").read_text())
    rec = next(r for r in json.loads((args.solved / "starting_states.json").read_text())["states"]
               if r["provenance"].get("code") == args.code)
    out = panel(args.solved, manifest, study_config(manifest), rec, args.label)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    d = out["bodies"]["defender"]["options"]
    release = out["release"]
    print(f"{args.label} {args.code}: value {out['value']:.3f} · defender "
          + ", ".join(f"{o['name']} {o['prob']:.0%}" for o in sorted(d, key=lambda o: -o['prob']) if o["prob"] > 0.01)
          + (f" · pass now {release['prob']:.0%}" if release["prob"] > 0.01 else ""))


def panel(solved: Path, manifest: dict, config, rec: dict, label: str) -> dict:
    """One starting record's opening decision (see the module docstring)."""
    state = json.loads((solved / "states" / f"state_{rec['index']:03d}.json").read_text())
    kind = "3v1" if rec.get("passer") else "2v1"
    sc = scenario_from(rec["scenario"])
    direction = int(rec["scenario"]["attack_direction"])
    commands = (commands_for(rec, config, manifest, kind) if commands_for else None)
    physics = manifest.get("physics")
    if "commands" in inspect.signature(am.layers_for).parameters:
        layers = am.layers_for(sc, config, physics, commands)
    else:
        layers = am.layers_for(sc, config, physics)
    with np.load(solved / "policies" / f"state_{rec['index']:03d}.npz") as z:
        dpol = np.asarray(z["defender_0"]).reshape(-1)
        apol = np.asarray(z["attack_0"]).reshape(-1)
        code = int(np.asarray(z["pass_index_0"]).reshape(-1)[0])
    actions = len(config.directions)
    joint = apol[:actions * actions].reshape(actions, actions)      # [carrier slot, receiver slot]
    probs = (joint.sum(axis=1), joint.sum(axis=0), dpol)

    pos0 = {s: layers[i][0].position[0] for i, s in enumerate(SLOTS)}
    snapshot = ({"defender": pos0["defender"].tolist(), "carrier": pos0["carrier"].tolist(),
                 "receiver": pos0["receiver"].tolist()} if kind == "2v1" else
                {"defender": pos0["defender"].tolist(), "runner": pos0["carrier"].tolist(),
                 "beneficiary": pos0["receiver"].tolist(),
                 "passer": [float(x) for x in rec["passer"]["positions"][0]]})
    bodies = {}
    for i, slot in enumerate(SLOTS):
        options = []
        for c in range(actions):
            nxt = int(layers[i][0].successor[0, c])
            end = layers[i][1].position[nxt]
            if commands is not None:
                name = commands[slot][c].name
            elif slot == "defender":
                name = s3.name_move_targets(s3.world_direction(c, direction), snapshot["defender"],
                                            s3.targets_for(snapshot, s3._goal(rec["scenario"])))
            else:
                name = compass_name(s3.world_direction(c, direction), direction)
            if commands is not None:
                body = (sc.carrier, sc.receiver, sc.defender)[i]
                aim = command_direction(commands[slot][c], layers[i][0].position[0], layers[i][0].velocity[0], 0,
                                        config, sc, body.maximum_speed)
            else:
                aim = s3.world_direction(c, direction)
            options.append({"name": name, "prob": float(probs[i][c]), "end": [float(x) for x in end],
                            "end_velocity": [float(x) for x in layers[i][1].velocity[nxt]],
                            # the solver's own 0.6 s path under this command (the physics steps)
                            "path": ([[float(v) for v in pt] for pt in layers[i][0].paths[0, c]]
                                     if layers[i][0].paths is not None else None),
                            "aim": [float(x) for x in aim]})
        bodies[slot] = {"role": ROLES[kind][i], "pos": [float(x) for x in pos0[slot]],
                        "velocity": [float(x) for x in layers[i][0].velocity[0]], "options": options}
    npass = len(config.passes)

    def aimed(code):
        """(receiving role, target, choice) of pass candidate `code` (the imported pass_index coding)."""
        choice = config.passes[code % npass]
        to_slot = ("receiver" if kind == "2v1" else ("carrier" if code // npass == 0 else "receiver"))
        slot_i = SLOTS.index(to_slot)
        rpos, rvel = layers[slot_i][0].position[0], layers[slot_i][0].velocity[0]
        target = (target_of(choice, rpos, direction, rvel, sc) if target_of else choice.target(rpos, direction))
        return bodies[to_slot]["role"], [float(x) for x in np.asarray(target)], choice

    # multi-pass games (2026-09-29) give every pass candidate its own column; the imported one column
    multi = bool(state.get("multi_pass"))
    pass_probs = apol[actions * actions:] if multi else apol[-1:]
    release = {"prob": float(pass_probs.sum()), "to": None, "target": None}
    passes = []
    if multi:
        for j in np.flatnonzero(pass_probs > 1e-9):
            to, target, choice = aimed(int(j))
            passes.append({"to": to, "prob": float(pass_probs[j]), "target": target, "family": choice.family,
                           "where": pass_where(choice)})
        passes.sort(key=lambda q: -q["prob"])
        if passes:
            release.update(to=passes[0]["to"], target=passes[0]["target"])
    elif code >= 0:
        to, target, _ = aimed(code)
        release.update(to=to, target=target)
    slot_values = (state.get("modal_line") or [{}])[0]
    return {"label": label, "code": rec["provenance"].get("code"), "kind": kind, "attack_direction": direction,
            "start_frame": int(rec["provenance"]["onset_frame_id"]), "value": float(state["value"]),
            "pass_model": state.get("pass_model"), "bodies": bodies, "release": release,
            "ball": [float(x) for x in (rec["passer"]["positions"][0] if kind == "3v1" else pos0["carrier"])],
            # value of each command of each slot against the others' equilibrium play (v3 studies only)
            "slot_values": {k: slot_values[k] for k in ("carrier_slot_values", "receiver_slot_values",
                                                        "defender_values") if k in slot_values},
            "release_steps": rec.get("release_steps"), "passes": passes, "provenance": rec["provenance"]}


if __name__ == "__main__":
    main()
