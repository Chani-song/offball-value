#!/usr/bin/env python3
"""Read a defender-grid study (scripts/build_defender_grid.py) into one file: for every lattice point, the
defender's opening mixed strategy and what each of his commands does in the first 0.6 s.

The solver's strategies are behaviour strategies: a mixed choice at every decision state of the tree (the
policy npz keeps them all: defender_k / attack_k for decision k). The opening row (defender_0, the same as
the state file's root_defender) IS the probability of each first move, so nothing is summed over sequences;
the npz files stay in the study directory untouched.

Action ids are the solver's five commands, relative to the attack (stage3_read.world_direction):
0 stop (target velocity 0: he brakes, sliding on while he slows), 1 toward the goal attacked, 2 to the
attack's left (up in the figures), 3 away from that goal, 4 to the attack's right (down). The solver's own
football names ("toward ball", "toward runner", "toward goal", "sideways") are the target a command points
at most from THAT start (stage3_read.name_move_targets), so the same command changes name across the grid and
two commands can share one; they are stored per point for reference and never used to merge commands.

Per point: status (solved / unsolved: no state file / dropped at build, with why), the start (pitch metres,
attack to the right, from the centre spot, and the solver's frame), the defender's 5 probabilities (their
sum), each command's solver 0.6 s path, end and end speed, the attack's full opening policy (25 joint moves
carrier x runner + one column per pass candidate, with each played pass's target), the game value and the
solver's certificate gap. Uses extract_panel_policy.panel, so everything is read as the figure panels read it.

Usage:
    PYTHONPATH=src python scripts/extract_defender_grid.py --solved <study> --output <json> [--workers 8]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_panel_policy import compass_name, panel, study_config  # noqa: E402
from defensive_positioning.equilibrium_clips import scenario_from  # noqa: E402
from offball_value import agile_motion as am  # noqa: E402
from offball_value import stage3_read as s3  # noqa: E402

ACTIONS = [  # command index -> (id, label in the figures, compass reading it must have)
    ("stop", "Stop", "stop"),
    ("goalward", "Toward goal", "forward"),
    ("left", "Up (attack's left)", "left"),
    ("upfield", "Away from goal", "back"),
    ("right", "Down (attack's right)", "right"),
]
SOLVER_NAMES = {"slow down": "stop (brake)", "toward goal": "toward goal", "toward ball": "toward ball",
                "toward runner": "toward runner", "sideways": "sideways"}
_STUDY = {}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--solved", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--grid-states", type=Path, default=None,
                   help="the states file build_defender_grid.py wrote: its grid facts (spacing, dropped starts); "
                        "the solver does not copy them into <study>/starting_states.json")
    p.add_argument("--only", nargs="*", default=None, help="codes to read (testing)")
    return p.parse_args()


def pass_run(rec: dict, target, config, physics) -> dict:
    """The runner's first 0.6 s when the attack plays a pass to `target` (solver frame), for the runner grid's flow
    (2026-10-01, the user's option a). The pass column has no runner path of its own: the solver prices the pass
    from the race, where the receiver reaches a point by the agile motion heading for it at once
    (physics_pass.reach_time). So: the same physics step as his commands (agile_motion._run, curve or plant cut
    as agile_motion.advance picks), aimed at the target at full speed, from his start; the move ends at the path
    point nearest the target (he waits there if he gets there inside the 0.6 s)."""
    sc = scenario_from(rec["scenario"])
    r0 = np.asarray(rec["scenario"]["receiver"]["position"], float)
    v0 = np.asarray(rec["scenario"]["receiver"]["velocity"], float)
    gap = np.asarray(target, float) - r0
    desired = gap / max(float(np.linalg.norm(gap)), 1e-12) * float(rec["scenario"]["receiver"]["maximum_speed"])
    curve = am._run(r0, v0, desired, sc, config, physics, 0, plant=False)
    cut = am._run(r0, v0, desired, sc, config, physics, 0, plant=True)
    run = cut if (cut is not None and np.linalg.norm(cut[1] - desired) < np.linalg.norm(curve[1] - desired) - 1e-12) \
        else curve
    path = run[2]
    k = int(np.argmin(np.linalg.norm(path - np.asarray(target, float), axis=1)))
    return {"path": path[:k + 1].tolist(), "end": path[k].tolist(), "reached_target": k < len(path) - 1}


def read_one(job):
    solved, rec = job
    solved = Path(solved)
    sc = rec["scenario"]
    sign = int(sc["attack_direction"])
    norm = lambda c: [sign * (c[0] - 52.5), sign * (c[1] - 34.0)]
    grid = rec["provenance"].get("grid") or {"i": 0, "j": 0, "observed": True}
    base = {"i": grid["i"], "j": grid["j"], "observed": bool(grid.get("observed")), "code": rec["provenance"]["code"],
            "index": rec["index"], "defender_start": norm(sc["defender"]["position"]),
            "defender_start_solver": list(sc["defender"]["position"])}
    state_file = solved / "states" / f"state_{rec['index']:03d}.json"
    if not state_file.exists():
        return dict(base, status="unsolved", why="no state file (the solver did not finish or certify it)")
    if str(solved) not in _STUDY:
        manifest = json.loads((solved / "manifest.json").read_text())
        _STUDY[str(solved)] = (manifest, study_config(manifest))
    manifest, config = _STUDY[str(solved)]
    st = json.loads(state_file.read_text())
    p = panel(solved, manifest, config, rec, solved.name)
    d = p["bodies"]["defender"]
    with np.load(solved / "policies" / f"state_{rec['index']:03d}.npz") as z:
        dpol = np.asarray(z["defender_0"], float).reshape(-1)
        apol = np.asarray(z["attack_0"], float).reshape(-1)
    commands = []
    for c, o in enumerate(d["options"]):
        u = s3.world_direction(c, sign)
        if compass_name(u, sign) != ACTIONS[c][2]:
            raise SystemExit(f"command {c} reads {compass_name(u, sign)}, expected {ACTIONS[c][2]}")
        commands.append({"command": c, "id": ACTIONS[c][0], "label": ACTIONS[c][1], "prob": float(dpol[c]),
                         "solver_name": SOLVER_NAMES.get(o["name"], o["name"]),
                         "path": [norm(pt) for pt in o["path"]], "end": norm(o["end"]),
                         "end_speed": math.hypot(*o["end_velocity"])})
        if abs(o["prob"] - dpol[c]) > 1e-12:
            raise SystemExit(f"{rec['provenance']['code']}: panel prob {o['prob']} != npz {dpol[c]}")
    n = 5
    joint = apol[:n * n].reshape(n, n)
    attack = {"root": apol.tolist(), "joint_carrier_x_runner": joint.tolist(),
              "carrier_marginal": joint.sum(axis=1).tolist(), "runner_marginal": joint.sum(axis=0).tolist(),
              "pass_columns": [{"column": int(k), "prob": float(apol[n * n + k])} for k in np.flatnonzero(apol[n * n:] > 1e-12)],
              "passes_played": [dict(q, target=norm(q["target"])) for q in p["passes"]],
              "carrier_paths": [[norm(pt) for pt in o["path"]] for o in p["bodies"]["carrier"]["options"]],
              "runner_paths": [[norm(pt) for pt in o["path"]] for o in p["bodies"]["receiver"]["options"]],
              "runner_ends": [norm(o["end"]) for o in p["bodies"]["receiver"]["options"]]}
    physics = am.resolve(manifest.get("physics"))
    for q in attack["passes_played"]:            # targets in p["passes"] are the solver frame; q's are normed
        raw = next(r["target"] for r in p["passes"] if r["prob"] == q["prob"] and norm(r["target"]) == q["target"])
        run = pass_run(rec, raw, config, physics)
        q["runner_run"] = {"path": [norm(pt) for pt in run["path"]], "end": norm(run["end"]),
                           "reached_target": run["reached_target"]}
    return dict(base, status="solved", value=float(st["value"]), max_local_gap=float(st["max_local_gap"]),
                defender_prob_sum=float(dpol.sum()), attack_prob_sum=float(apol.sum()),
                defender=commands, attack=attack,
                carrier_start=norm(sc["carrier"]["position"]), runner_start=norm(sc["receiver"]["position"]),
                carrier_velocity=[sign * v for v in sc["carrier"]["velocity"]],
                runner_velocity=[sign * v for v in sc["receiver"]["velocity"]],
                defender_velocity=[sign * v for v in sc["defender"]["velocity"]],
                ball=norm(p["ball"]), start_frame=p["start_frame"], attack_direction=sign)


def main() -> None:
    args = parse_args()
    starts = json.loads((args.solved / "starting_states.json").read_text())
    recs = [r for r in starts["states"] if args.only is None or r["provenance"]["code"] in args.only]
    jobs = [(str(args.solved), r) for r in recs]
    if args.workers > 1:
        with ProcessPoolExecutor(args.workers) as pool:
            points = list(pool.map(read_one, jobs))
    else:
        points = [read_one(j) for j in jobs]
    grid = starts.get("grid") or (json.loads(args.grid_states.read_text()).get("grid", {}) if args.grid_states else {})
    if not grid:
        print("WARNING: no grid facts (pass --grid-states): dropped starts are not listed")
    sign = int(recs[0]["scenario"]["attack_direction"])
    moved = grid.get("player", "defender")          # build_defender_grid.py --player runner (2026-10-01) moves the runner
    for dpt in grid.get("dropped", []):
        points.append({"i": dpt["i"], "j": dpt["j"], "observed": False, "status": "dropped", "why": dpt["why"],
                       f"{moved}_start": [sign * (dpt["position"][0] - 52.5), sign * (dpt["position"][1] - 34.0)],
                       f"{moved}_start_solver": dpt["position"]})
    points.sort(key=lambda q: (q["i"], q["j"]))
    manifest = json.loads((args.solved / "manifest.json").read_text())
    out = {"study": str(args.solved), "grid": grid, "manifest": manifest,
           "actions": [{"command": c, "id": a, "label": lab, "compass": comp,
                        "definition": ("target velocity zero: the defender brakes (at the solver's braking limit) "
                                       "and slides on the way he was running until he stops" if c == 0 else
                                       f"steer toward the attack's {comp} at full speed within the agile limits "
                                       "(momentum, braking, turning, plant cut)")}
                       for c, (a, lab, comp) in enumerate(ACTIONS)],
           "frame": "pitch metres from the centre spot, attack to the right (+x toward the goal attacked, "
                    "+y the attack's left = up in the figures)",
           "points": points}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    solved = [q for q in points if q["status"] == "solved"]
    print(f"{args.output}: {len(solved)} solved, {sum(q['status'] == 'unsolved' for q in points)} unsolved, "
          f"{sum(q['status'] == 'dropped' for q in points)} dropped")
    for q in solved:
        if q["observed"]:
            print("observed:", ", ".join(f"{c['id']} {c['prob']:.4f}" for c in q["defender"]))


if __name__ == "__main__":
    main()
