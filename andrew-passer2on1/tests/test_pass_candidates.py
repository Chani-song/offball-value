#!/usr/bin/env python3
"""Are the fitted pass-model candidates usable by the solver?

Structural checks only -- a failure stops the overnight chain before any
solve. How good the candidates are is the fit report's business, not this.

  1  copies      physics_pass.py byte-identical in its three homes
  2  loading     every router file names models that load through
                 physics_pass.load_pass_model; B1 loads through 준현's own
                 ExpectedPass.load without the proxy opt-in (its targets are
                 intended-at-kick, not proxies)
  3  cross-fit   the routers cover every training match, each with a model
                 whose metadata says it was NOT trained on that match
  4  game shape  both candidates evaluate a real 3v1 state's defender array
                 ([n, 1 + m, 2]) and return finite probabilities in [0, 1],
                 matching a one-by-one evaluation
  5  velocity    A's through-ball probability rises when the runner sprints
                 into the target and rises again when the defender runs the
                 other way; the positions-only baseline cannot move at all

Run:  PYTHONPATH=andrew-passer2on1:andrew-fixedpasser:src \\
      .venv-delta/bin/python andrew-passer2on1/tests/test_pass_candidates.py data/processed/pass_models
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from defensive_positioning.expected_pass import ExpectedPass
from passer2on1.physics_pass import PhysicsRacePass, load_pass_model

ROOT = Path(__file__).resolve().parents[2]


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        sys.exit(1)


def main() -> None:
    models = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data/processed/pass_models"
    print("1  copies")
    files = [ROOT / "andrew-passer2on1/passer2on1/physics_pass.py",
             ROOT / "andrew-fixedpasser/fixedpasser/physics_pass.py",
             ROOT / "src/offball_value/physics_pass.py"]
    check("physics_pass.py byte-identical", len({f.read_bytes() for f in files}) == 1)

    print("2, 3  loading and cross-fitting")
    report = json.loads((models / "report.json").read_text())
    matches = report["matches"]
    loaded = {}
    for name in ("A", "B1"):
        router = json.loads((models / f"{name}_crossfit.json").read_text())
        check(f"{name} router covers the {len(matches)} training matches",
              sorted(router["by_match"]) == sorted(matches))
        for m, f in router["by_match"].items():
            raw = json.loads((models / f).read_text())
            meta = raw.get("metadata", {})
            check(f"{name} for {m} was not trained on it",
                  meta.get("held_out_match") == m and m not in meta.get("trained_on", [m]))
            load_pass_model(models / f)
        loaded[name] = load_pass_model(models / router["default"])
    check("B1 loads with 준현's loader, no proxy opt-in",
          isinstance(ExpectedPass.load(models / "B1_all.json"), ExpectedPass))
    check("A is the physics model", isinstance(loaded["A"], PhysicsRacePass))

    print("4  game shapes")
    rec = json.loads((ROOT / "data/processed/stage3/fixedpasser_states_agile06_final2m.json")
                     .read_text())["states"][0]
    sc = rec["scenario"]
    rng = np.random.default_rng(3)
    n = 257
    passer = np.asarray(rec["passer"]["positions"][0])
    pv = np.asarray(rec["passer"]["velocities"][0])
    recv = np.asarray(sc["receiver"]["position"]) + rng.normal(0, 2, (n, 2))
    rv = rng.normal(0, 3, (n, 2))
    bg = np.asarray(rec["background"]["positions"][0])
    bgv = np.asarray(rec["background"]["velocities"][0])
    dpos = np.asarray(sc["defender"]["position"]) + rng.normal(0, 2, (n, 2))
    dvel = rng.normal(0, 3, (n, 2))
    defenders = np.concatenate([dpos[:, None, :], np.broadcast_to(bg, (n,) + bg.shape)], axis=1)
    dv = np.concatenate([dvel[:, None, :], np.broadcast_to(bgv, (n,) + bgv.shape)], axis=1)
    target = recv + np.array([4.0, 0.0]) * sc["attack_direction"]
    for name, model in loaded.items():
        p = np.asarray(model.predict(passer, recv, defenders, pv, rv, dv, target, 0,
                                     sc["attack_direction"], 105.0, 68.0))
        one = np.array([float(model.predict(passer, recv[i], defenders[i], pv, rv[i], dv[i], target[i],
                                            0, sc["attack_direction"], 105.0, 68.0)) for i in range(n)])
        check(f"{name}: shape, range, vectorised = one by one",
              p.shape == (n,) and np.isfinite(p).all() and p.min() >= 0 and p.max() <= 1
              and np.abs(p - one).max() < 1e-9, f"{p.min():.3f}-{p.max():.3f}")

    print("5  velocity")
    probe = {row["상황"]: row for row in report["probe"]}
    rows = list(probe.values())
    check("A rises with the runner's sprint and the defender running away",
          rows[0]["A"] < rows[1]["A"] < rows[2]["A"], " → ".join(f"{r['A']:.3f}" for r in rows))
    check("the positions-only baseline does not move", len({round(r["baseline"], 9) for r in rows}) == 1)
    print("     B1 (reported, not required): " + " → ".join(f"{r['B1']:.3f}" for r in rows))
    print("\nall pass-candidate checks passed")


if __name__ == "__main__":
    main()
