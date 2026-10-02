#!/usr/bin/env python3
"""Checks of a defender-grid result (scripts/extract_defender_grid.py) before its figures are trusted:

  1. every solved start: the defender's five opening probabilities (and the attack's opening policy) sum to 1;
  2. the observed start reproduces the existing S05 analysis: its opening policies and game value against the
     reference study's state for the same record (full 2v1 study, the one Figure 2 shows); if they differ, the
     starting records and the two studies' settings are compared so the difference can be explained;
  3. the solver's certificate gap at every start, the solved / unsolved / dropped counts.

Also writes every start's numbers as one CSV (probabilities per command, the solver's own name for each
command at that start, value, gap, status).

Usage:
    python scripts/check_defender_grid.py --grid <grid json> --reference <study> --code S05 --out <dir>
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

TOL = 1e-9


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--grid", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True, help="the study the observed record was solved in")
    p.add_argument("--code", default="S05")
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    g = json.loads(args.grid.read_text())
    pts = g["points"]
    solved = [p for p in pts if p["status"] == "solved"]
    report = {"counts": {s: sum(p["status"] == s for p in pts) for s in ("solved", "unsolved", "dropped")},
              "unsolved": [{"i": p["i"], "j": p["j"], "why": p["why"]} for p in pts if p["status"] == "unsolved"],
              "dropped": [{"i": p["i"], "j": p["j"], "why": p["why"]} for p in pts if p["status"] == "dropped"]}
    dsum = [abs(sum(c["prob"] for c in p["defender"]) - 1.0) for p in solved]
    asum = [abs(p["attack_prob_sum"] - 1.0) for p in solved]
    report["defender_sum_max_error"] = max(dsum) if dsum else None
    report["attack_sum_max_error"] = max(asum) if asum else None
    report["sums_ok"] = bool(dsum) and max(dsum) < 1e-6 and max(asum) < 1e-6
    report["max_certificate_gap"] = max(p["max_local_gap"] for p in solved) if solved else None
    report["negative_probabilities"] = sum(c["prob"] < -TOL for p in solved for c in p["defender"])
    # 2. the observed start against the reference study
    obs = next((p for p in solved if p["observed"]), None)
    ref_starts = json.loads((args.reference / "starting_states.json").read_text())["states"]
    ref = next(r for r in ref_starts if r["provenance"]["code"] == args.code)
    ref_state = json.loads((args.reference / "states" / f"state_{ref['index']:03d}.json").read_text())
    cmp = {"reference_study": str(args.reference), "reference_index": ref["index"]}
    if obs is None:
        cmp["result"] = "the observed start was not solved"
    else:
        rd, ra = np.array(ref_state["root_defender"], float), np.array(ref_state["root_attack"], float)
        od = np.array([c["prob"] for c in obs["defender"]], float)
        oa = np.array(obs["attack"]["root"], float)
        cmp.update(defender_observed=od.tolist(), defender_reference=rd.tolist(),
                   defender_max_abs_diff=float(np.max(np.abs(od - rd))),
                   attack_max_abs_diff=float(np.max(np.abs(oa - ra))) if oa.shape == ra.shape else "shapes differ",
                   value_observed=obs["value"], value_reference=float(ref_state["value"]),
                   value_abs_diff=abs(obs["value"] - float(ref_state["value"])))
        same = (cmp["defender_max_abs_diff"] < 1e-6 and isinstance(cmp["attack_max_abs_diff"], float)
                and cmp["attack_max_abs_diff"] < 1e-6 and cmp["value_abs_diff"] < 1e-6)
        cmp["result"] = "matches the existing S05 analysis" if same else "DIFFERS from the existing S05 analysis"
        if not same:     # explain: the inputs and the settings side by side
            grid_starts = json.loads((Path(g["study"]) / "starting_states.json").read_text())["states"]
            mine = next(r for r in grid_starts if r["provenance"]["code"] == obs["code"])
            cmp["scenario_identical"] = mine["scenario"] == ref["scenario"]
            cmp["background_identical"] = mine.get("background") == ref.get("background")
            ref_man = json.loads((args.reference / "manifest.json").read_text())
            keys = set(ref_man) | set(g["manifest"])
            cmp["manifest_differences"] = {k: [ref_man.get(k), g["manifest"].get(k)] for k in sorted(keys)
                                           if k not in ("created_utc", "states", "state_file_sha256", "workers")
                                           and ref_man.get(k) != g["manifest"].get(k)}
    report["observed_vs_reference"] = cmp
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{args.code}_verification.json").write_text(json.dumps(report, indent=1))
    # every start's numbers, one row each
    ids = [a["id"] for a in g["actions"]]
    with (args.out / f"{args.code}_grid_probabilities.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["i_goalward_steps", "j_up_steps", "defender_x_m", "defender_y_m", "observed", "status", "why"]
                   + [f"p_{a}" for a in ids] + [f"solver_name_{a}" for a in ids]
                   + ["value", "certificate_gap", "code"])
        for p in pts:
            row = [p["i"], p["j"], f"{p['defender_start'][0]:.3f}", f"{p['defender_start'][1]:.3f}", p["observed"],
                   p["status"], p.get("why", "")]
            if p["status"] == "solved":
                row += [f"{c['prob']:.12g}" for c in p["defender"]] + [c["solver_name"] for c in p["defender"]]
                row += [f"{p['value']:.9g}", f"{p['max_local_gap']:.3g}", p["code"]]
            else:
                row += [""] * (2 * len(ids)) + ["", "", p.get("code", "")]
            w.writerow(row)
    print(json.dumps({k: v for k, v in report.items() if k not in ("dropped",)}, indent=1))


if __name__ == "__main__":
    main()
