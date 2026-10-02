#!/usr/bin/env python3
"""The defender's start moved over a grid (2026-09-30, for the Figure 2 alternatives): the same attack, the
defender elsewhere.

Copies one solved starting record (S05 at 0.0 s from the full 2v1 study that Figure 2 shows) once per point
of a lattice and moves ONLY the controlled defender's start position there. Everything else is the record
as solved: ball carrier and runner position and velocity, the defender's velocity and limits, the ten
background defenders' real tracks, the pitch, the attack direction. The solver then builds a fresh game at
each point (the same flags as the source study, jobs/solve_defender_grid.sbatch), so which moves are
possible and what they are worth are evaluated from the new start.

Lattice (the defaults: the first, 2 m version; the abstract's Figure 2 uses --spacing 1 with the ranges in
jobs/solve_defender_grid.sbatch, which cover each moment's panel): 2 m apart, laid through the real defender's exact position so that point (0, 0) IS the observed
start (solved at the exact coordinates, not snapped); forward steps i = -4..+3 (toward the goal attacked),
left steps j = -4..+2 (attack's left = up in the figures) -- from level with the ball carrier to 6 m
goal-side of the real defender, and from the carrier's line to the runner's. A point is dropped (and listed
as such, not solved) when it is off the pitch or within the solver's tackle radius (1.0 m, GameConfig) of the
ball carrier or the runner: two bodies cannot start there.

--player runner (2026-10-01, the user asked for the runner's version of Figure 2's flow + value background):
the same lattice laid through the real RUNNER (the 2v1 receiver slot) instead, only his start moved, same
velocity ("the same run, from somewhere else"); a point is dropped within 1.0 m of the ball
carrier or the defender. The pass candidates follow him (run_passes: aimed from the receiver's own position and
velocity), and the solver enforces offside itself. Codes carry R instead of D; files are states_runner_grid_*.
The default (--player defender) writes exactly what it wrote before.

Usage:
    python scripts/build_defender_grid.py --states <study>/starting_states.json --code S05 --out <dir>
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

TACKLE_RADIUS = 1.0  # m, GameConfig.tackle_radius
EDGE = 0.05          # m off the touchline, as build_stage3_states keeps bodies


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--states", type=Path, required=True, help="the solved study's starting_states.json")
    p.add_argument("--code", default="S05", help="provenance code of the record to copy")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--spacing", type=float, default=2.0)
    p.add_argument("--forward", type=int, nargs=2, default=(-4, 3), help="i range (goalward steps)")
    p.add_argument("--left", type=int, nargs=2, default=(-4, 2), help="j range (steps to the attack's left)")
    p.add_argument("--skip-states", type=Path, nargs="+", default=None,
                   help="states files already solved (e.g. the 2 m lattice): their starts are left out here")
    p.add_argument("--split", type=int, default=1, help="write the starts as this many part files")
    p.add_argument("--tag", default="", help="added to every code, e.g. '1m' (codes stay unique across runs)")
    p.add_argument("--player", choices=("defender", "runner"), default="defender", help="whose start moves")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    source = json.loads(args.states.read_text())
    meta = {k: v for k, v in source.items() if k != "states"}
    rec = next(r for r in source["states"] if r["provenance"]["code"] == args.code)
    sc = rec["scenario"]
    sign = int(sc["attack_direction"])
    slot = "defender" if args.player == "defender" else "receiver"      # the scenario's key for the moved body
    dx, dy = sc[slot]["position"]
    attackers = ({"ball carrier": sc["carrier"]["position"], "runner": sc["receiver"]["position"]}
                 if args.player == "defender" else
                 {"ball carrier": sc["carrier"]["position"], "defender": sc["defender"]["position"]})
    states, dropped, skipped = [], [], 0
    done = [r["scenario"][slot]["position"] for f in (args.skip_states or [])
            for r in json.loads(f.read_text())["states"]]
    for i in range(args.forward[0], args.forward[1] + 1):
        for j in range(args.left[0], args.left[1] + 1):
            # steps along the attack (i) and to its left (j); the solver's frame is mirrored when sign = -1
            x, y = dx + sign * args.spacing * i, dy + sign * args.spacing * j
            why = None
            if not (EDGE <= x <= sc["pitch_length"] - EDGE and EDGE <= y <= sc["pitch_width"] - EDGE):
                why = "off the pitch"
            for role, pos in attackers.items():
                if math.dist((x, y), pos) < TACKLE_RADIUS:
                    why = f"within {TACKLE_RADIUS:g} m (tackle radius) of the {role}"
            if why:
                dropped.append({"i": i, "j": j, "position": [x, y], "why": why})
                continue
            if any(math.dist((x, y), d) < 1e-6 for d in done):     # solved in the earlier run
                skipped += 1
                continue
            r = copy.deepcopy(rec)
            r["scenario"][slot]["position"] = [x, y]
            r["provenance"] = dict(rec["provenance"],
                                   code=f"{args.code}#{args.player[0].upper()}{args.tag}{i:+d},{j:+d}",
                                   grid={"what": f"{args.player} start moved", "i": i, "j": j,
                                         "spacing_m": args.spacing, f"real_{args.player}": [dx, dy],
                                         "observed": i == 0 and j == 0})
            states.append(r)
    args.out.mkdir(parents=True, exist_ok=True)
    parts = [states[k::args.split] for k in range(args.split)]
    for n, part in enumerate(parts):
        for k, r in enumerate(part):
            r["index"] = k
        out = dict(meta, grid={"what": (f"the controlled defender's start moved over a lattice; everything else as in "
                                        "the source record" if args.player == "defender" else
                                        "the runner's start moved over a lattice (same velocity); everything else as "
                                        "in the source record"),
                               **({} if args.player == "defender" else {"player": args.player}),
                               "source": str(args.states), "source_code": args.code, "spacing_m": args.spacing,
                               "forward": list(args.forward), "left": list(args.left), "dropped": dropped,
                               "skipped_already_solved": [str(f) for f in args.skip_states] if args.skip_states else None,
                               "part": f"{n + 1}/{args.split}"},
                   states=part)
        name = f"states_{args.player}_grid_{args.code}{('_' + args.tag) if args.tag else ''}"
        path = args.out / (f"{name}.json" if args.split == 1 else f"{name}_part{n + 1}.json")
        path.write_text(json.dumps(out))
        obs = [r["index"] for r in part if r["provenance"]["grid"]["observed"]]
        print(f"{path}: {len(part)} {args.player} starts" + (f" (observed = index {obs})" if obs else ""))
    print(f"{len(states)} new starts, {skipped} already solved, {len(dropped)} dropped: "
          + "; ".join(f"({d['i']:+d},{d['j']:+d}) {d['why']}" for d in dropped))


if __name__ == "__main__":
    main()
