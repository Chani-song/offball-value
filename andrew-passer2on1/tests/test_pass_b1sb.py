#!/usr/bin/env python3
"""Is the fitted B1-SB model usable by the solver? (structural checks only)

  1  loading      B1SB_all.json loads with Andrew's ExpectedPass.load, no proxy
                  opt-in, in the velocity schema
  2  independent  every training row is a StatsBomb match from the four men's
                  club competitions -- none of our DFL matches
  3  game shape   a real 3v1 state's [n, 1 + m, 2] defender array gives finite
                  probabilities in [0, 1], vectorised = one by one
  4  report       the transfer scores on our Bundesliga passes exist and are finite

Run:  PYTHONPATH=andrew-passer2on1:src python \\
      andrew-passer2on1/tests/test_pass_b1sb.py data/processed/pass_models
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

from defensive_positioning.expected_pass import ExpectedPass

ROOT = Path(__file__).resolve().parents[2]
CLUB_MEN = {"1. Bundesliga", "La Liga", "Ligue 1", "Major League Soccer"}


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        sys.exit(1)


def main() -> None:
    models = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data/processed/pass_models"
    print("1  loading")
    model = ExpectedPass.load(models / "B1SB_all.json")
    check("loads with Andrew's loader, velocity schema", model.feature_set == "velocity")

    print("2  independent of our matches")
    comps, ids, n = set(), set(), 0
    for line in (models / "statsbomb_passes.jsonl").read_text().splitlines():
        r = json.loads(line)
        comps.add(r["competition"])
        ids.add(r["match_id"])
        n += 1
    check("men's club competitions only", comps <= CLUB_MEN, ", ".join(sorted(comps)))
    check("no DFL match among the training rows", not any(str(i).startswith("DFL") for i in ids),
          f"{n:,} passes, {len(ids)} matches")

    print("3  game shape")
    rec = json.loads((ROOT / "data/processed/stage3/fixedpasser_states_agile06_final2m.json")
                     .read_text())["states"][0]
    sc = rec["scenario"]
    rng = np.random.default_rng(5)
    n = 257
    passer, pv = np.asarray(rec["passer"]["positions"][0]), np.asarray(rec["passer"]["velocities"][0])
    recv = np.asarray(sc["receiver"]["position"]) + rng.normal(0, 2, (n, 2))
    rv = rng.normal(0, 3, (n, 2))
    bg, bgv = np.asarray(rec["background"]["positions"][0]), np.asarray(rec["background"]["velocities"][0])
    d = np.concatenate([(np.asarray(sc["defender"]["position"]) + rng.normal(0, 2, (n, 2)))[:, None, :],
                        np.broadcast_to(bg, (n,) + bg.shape)], axis=1)
    dv = np.concatenate([rng.normal(0, 3, (n, 1, 2)), np.broadcast_to(bgv, (n,) + bgv.shape)], axis=1)
    t = recv + np.array([4.0, 0.0]) * sc["attack_direction"]
    p = np.asarray(model.predict(passer, recv, d, pv, rv, dv, t, 0, sc["attack_direction"], 105.0, 68.0))
    one = np.array([float(model.predict(passer, recv[i], d[i], pv, rv[i], dv[i], t[i], 0,
                                        sc["attack_direction"], 105.0, 68.0)) for i in range(n)])
    check("shape, range, vectorised = one by one",
          p.shape == (n,) and np.isfinite(p).all() and 0 <= p.min() and p.max() <= 1
          and np.abs(p - one).max() < 1e-9, f"{p.min():.3f}-{p.max():.3f}")

    print("4  report")
    rep = json.loads((models / "report_b1sb.json").read_text())
    tr = rep["transfer"][1]
    check("transfer scores finite", all(math.isfinite(tr[k]) for k in ("auc", "brier", "log_loss")),
          f"AUC {tr['auc']:.3f}, Brier {tr['brier']:.4f}, mean {tr['mean_predicted']:.3f}")
    print("     through-ball probe (reported, not required): "
          + " → ".join(f"{r['B1-SB']:.3f}" for r in rep["probe"]))
    print("\nall B1-SB checks passed")


if __name__ == "__main__":
    main()
