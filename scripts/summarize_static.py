#!/usr/bin/env python3
"""Summary of the whole-window static counterfactuals (scripts/static_counterfactual.py), 2026-09-30.

Per moment (V equilibrium, S defender held + attack's best, R the static-best plan answered, A real attack held):
  situation overestimated   (S - V) / V   what a static analysis promises the attack beyond the true value
  best action overrated     (S - R) / S   how much of the static-best plan's value a responsive defender takes
  defence could improve     (S - V) / S   the real defence's exploitability, against the equilibrium defence
  real attack short of V    (V - A) / V   what the real attack, answered, falls short of the equilibrium -- over V,
                                          since A can be 0: a defender who knows a 3v1 attack's whole real path can
                                          step up so every pass is offside at the horizon, where keeping scores 0
All held sides follow their real commands at every turn; the free side best-responds knowing them.

Usage: python scripts/summarize_static.py data/processed/eval_v1/static_full
"""

from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

SHOWCASE = ("S05", "S13", "S34", "S53", "S15", "S20", "S36", "S44")
SURE = ("S46", "S48", "S58", "S66", "S72")


def main() -> None:
    folder = Path(sys.argv[1])
    rows = [json.loads(f.read_text()) for f in sorted(folder.glob("*.json"))]
    check = max(abs(r["V"] - r["V_saved"]) for r in rows)
    print(f"moments {len(rows)}; equilibrium value rebuilt vs saved: max difference {check:.1e}")
    order = sum(r["S"] >= r["V"] - 1e-9 >= r["A"] - 2e-9 and r["R"] <= r["S"] + 1e-9 for r in rows)
    print(f"order S >= V >= A and R <= S holds in {order}/{len(rows)}")
    metrics = {"situation overestimated (S-V)/V": lambda r: (r["S"] - r["V"]) / r["V"],
               "best action overrated (S-R)/S": lambda r: (r["S"] - r["R"]) / r["S"],
               "defence could improve (S-V)/S": lambda r: (r["S"] - r["V"]) / r["S"],
               "real attack short of V (V-A)/V": lambda r: (r["V"] - r["A"]) / r["V"]}
    subsets = {"all": rows, "2v1": [r for r in rows if r["kind"] == "2v1"], "3v1": [r for r in rows if r["kind"] == "3v1"],
               "showcase+sure (13)": [r for r in rows if r["code"].split("@")[0] in SHOWCASE + SURE]}
    out = {}
    for name, sub in subsets.items():
        print(f"\n== {name} ({len(sub)} moments)")
        out[name] = {}
        for label, f in metrics.items():
            v = [f(r) for r in sub]
            top = max(sub, key=f)["code"]
            out[name][label] = {"mean": st.mean(v), "median": st.median(v), "max": max(v), "max_at": top}
            print(f"   {label:34s} mean {100*st.mean(v):5.1f}%  median {100*st.median(v):5.1f}%  max {100*max(v):5.1f}% ({top})")
    print("\nreal passes in the held attack:", sum(bool(r["attack_real_passes"]) for r in rows))
    (folder.parent / "static_full_summary.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
