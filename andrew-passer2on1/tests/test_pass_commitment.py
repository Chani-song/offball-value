"""Checks for the committed first defender in physics_pass (2026-09-29).

  1 unchanged   committed_s=None: A and A-sym predictions equal the module before the change, on random
                states and on the fitted models
                (needs that module's backup, BEFORE below; skipped where it is absent)
  2 meaning     a first defender "committed" to keeping his velocity -- given at p + v*tau with velocity v,
                committed_s = tau, no reaction of his own -- is exactly the fitted model's defender
  3 matters     committed to running at the target, he is faster there than the fitted model's defender

Run: PYTHONPATH=andrew-passer2on1 python andrew-passer2on1/tests/test_pass_commitment.py
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np

from passer2on1.physics_pass import load_pass_model

ROOT = Path(os.environ.get("OFFBALL_DATA_ROOT", "/scratch/bbmr/kseo1/offball-value"))
BEFORE = Path("/work/hdd/bbmr/kseo1/offball-out/backups/physics_pass_before_commitment_20260929.py")
MODELS = {"A": ROOT / "data/processed/pass_models/A_all.json",
          "A-sym": ROOT / "data/processed/pass_models_sym/Asym_all.json"}
failures = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        failures.append(name)


def old_module():
    spec = importlib.util.spec_from_file_location("physics_pass_before", BEFORE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def states(rng, n=4000, m=4):
    pos = lambda *s: rng.uniform([0, 0], [105, 68], size=s + (2,))
    vel = lambda *s: rng.normal(0, 3.5, size=s + (2,))
    return dict(carrier=pos(n), receiver=pos(n), defenders=pos(n, m), carrier_velocity=vel(n),
                receiver_velocity=vel(n), defender_velocities=vel(n, m), target=pos(n))


def main() -> None:
    rng = np.random.default_rng(20260929)
    old = old_module() if BEFORE.exists() else None
    s = states(rng)
    print("1  unchanged")
    if old is None:     # the pre-change module is a local backup (BEFORE), not part of the repository
        print(f"  SKIP  no copy of the module before the change at {BEFORE}")
    for name, path in (MODELS.items() if old is not None else ()):
        new_model, old_model = load_pass_model(path), old.load_pass_model(path)
        a = new_model.predict(**s)
        b = old_model.predict(**s)
        check(f"{name}: committed_s=None equals the module before", np.array_equal(a, b),
              f"max diff {float(np.max(np.abs(a - b))):.1e}")

    print("2  meaning")
    for name, path in MODELS.items():
        model = load_pass_model(path)
        tau = model.player["reaction_s"]
        d, dv = s["defenders"].copy(), s["defender_velocities"]
        d[:, 0] = d[:, 0] + dv[:, 0] * tau            # where keeping his velocity through the reaction leaves him
        base = model.margins(s["carrier"], s["receiver"], s["defenders"], s["receiver_velocity"], dv, s["target"])
        kept = model.margins(s["carrier"], s["receiver"], d, s["receiver_velocity"], dv, s["target"], committed_s=tau)
        worst = max(float(np.max(np.abs(x - y))) for x, y in zip(base, kept))
        check(f"{name}: committed to his own velocity = the fitted defender", worst <= 1e-12, f"max diff {worst:.1e}")

    print("3  matters")
    model = load_pass_model(MODELS["A-sym"])
    tau = model.player["reaction_s"]
    carrier, receiver, target = np.array([[30.0, 30.0]]), np.array([[50.0, 30.0]]), np.array([[50.0, 40.0]])
    d, dv = np.array([[[46.0, 34.0]]]), np.array([[[0.0, -5.0]]])     # running away from the target
    toward = (target - d[:, 0]) / np.linalg.norm(target - d[:, 0])
    committed = d.copy(); committed[:, 0] += toward * 1.0              # he has turned and gone 1 m toward it
    fitted = model.margins(carrier, receiver, d, np.zeros((1, 2)), dv, target)[0]
    turned = model.margins(carrier, receiver, committed, np.zeros((1, 2)), (toward * 5.0)[:, None, :], target,
                           committed_s=tau)[0]
    check("a defender committed toward the target shrinks the receiver's margin", float(turned[0]) < float(fitted[0]),
          f"{float(fitted[0]):.2f} s -> {float(turned[0]):.2f} s")

    print("\nall commitment checks passed" if not failures else f"\nFAILED: {failures}")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
