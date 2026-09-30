"""Checks for background tackles in passer2on1.game (2026-09-29): the other defenders can take the ball.

  1 off         background_tackles=False: survival exactly as before
  2 nobody      no background defenders: switching it on changes nothing
  3 lower       on (S05): survival never higher, lower somewhere; a defender standing on the carrier's path
                takes the ball far more often than one 20 m away
  4 value       on (S05, meeting pricing): the game is worth no more to the attack

Run: PYTHONPATH=andrew-passer2on1:src python andrew-passer2on1/tests/test_background_tackles.py
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import numpy as np

from defensive_positioning.markov import solve_markov_game
from defensive_positioning.models import GameConfig
from passer2on1.game import background_survival
from passer2on1.physics_pass import load_pass_model
from passer2on1.solve import game_from

ROOT = Path(os.environ.get("OFFBALL_DATA_ROOT", Path(__file__).resolve().parents[2]))
rec = next(r for r in json.loads((ROOT / "data/processed/showcase_v1/figure_2v1/states_2v1_all.json").read_text())["states"]
           if r["provenance"]["code"] == "S05")
PHYSICS = json.loads((ROOT / "data/processed/physics_limits/agile_p999_nodelay.json").read_text())
CONFIG = GameConfig(steps=3, step_seconds=0.6, physics_step=0.025)
MODEL = load_pass_model(ROOT / "andrew/models/experimental_pass.json", True)
failures = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        failures.append(name)


print("1  off")
base = game_from(rec, MODEL, CONFIG, PHYSICS, "compass", "andrew")
check("default leaves survival as it was", not base.background_tackles)
print("2  nobody")
empty = copy.deepcopy(rec); empty["background"] = None
a = game_from(empty, MODEL, CONFIG, PHYSICS, "compass", "andrew")
b = game_from(empty, MODEL, CONFIG, PHYSICS, "compass", "andrew", background_tackles=True)
check("no background: identical survival", all(np.array_equal(x, y) for x, y in zip(a.survival, b.survival)))
print("3  lower")
on = game_from(rec, MODEL, CONFIG, PHYSICS, "compass", "andrew", background_tackles=True)
never_higher = all(np.all(y <= x + 1e-15) for x, y in zip(base.survival, on.survival))
lower = min(float(np.min(y / np.maximum(x, 1e-300))) for x, y in zip(base.survival, on.survival))
check("never higher, lower somewhere", never_higher and lower < 1.0, f"smallest ratio {lower:.3f}")
layer = base.carrier[0]
path = layer.paths[0, 1]                                    # the root carrier's forward path
on_path = background_survival(layer, path[len(path) // 2][None], path[len(path) // 2][None], CONFIG)[0, 1]
far = background_survival(layer, path[0][None] + 20.0, path[0][None] + 20.0, CONFIG)[0, 1]
check("a defender on the path tackles, one 20 m off does not", on_path < 0.9 and far > 0.999999,
      f"keep {on_path:.3f} vs {far:.6f}")
print("4  value")
v0, v1 = solve_markov_game(base).root_value, solve_markov_game(on).root_value
check("worth no more to the attack", v1 <= v0 + 1e-12, f"{v0:.4f} -> {v1:.4f}")
print("\nall background-tackle checks passed" if not failures else f"\nFAILED: {failures}")
raise SystemExit(1 if failures else 0)
