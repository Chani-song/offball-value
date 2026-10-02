"""The numeric inputs committed with the code: the A and A-sym pass models, the measured movement limits and
the static EPV grid. The solver jobs read these files by path (jobs/*.sbatch); these checks need no tracking."""

from __future__ import annotations

import csv
import json
import math
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MATCHES = {"DFL-MAT-J03WMX", "DFL-MAT-J03WOH", "DFL-MAT-J03WOY", "DFL-MAT-J03WPY", "DFL-MAT-J03WQQ",
           "DFL-MAT-J03WR9"}           # the pass-model matches; DFL-MAT-J03WN1 is excluded (build_pass_dataset.py)


def load(path: Path) -> dict:
    return json.loads(path.read_text())


class PassModelFiles(unittest.TestCase):
    def check_family(self, folder: str, prefix: str, kind: str) -> None:
        base = ROOT / "data/processed" / folder
        router = load(base / f"{prefix}_crossfit.json")
        self.assertEqual(router["kind"], "per_match")
        self.assertEqual(set(router["by_match"]), MATCHES)
        for match, name in [*router["by_match"].items(), (None, router["default"])]:
            model = load(base / name)
            self.assertEqual(model["kind"], kind, name)
            self.assertEqual(len(model["coef"]), 3, name)
            self.assertTrue(all(math.isfinite(c) for c in model["coef"]), name)
            speed = model["ball_speed"]
            self.assertLess(speed["min"], speed["max"], name)
            trained = set(model["metadata"]["trained_on"])
            # a scene from match M is priced by a model that never saw M
            self.assertEqual(trained, MATCHES - {match} if match else MATCHES, name)

    def test_a(self) -> None:
        self.check_family("pass_models", "A", "physics_race_logit")

    def test_a_sym(self) -> None:
        # the model every solver job uses (--model .../Asym_crossfit.json)
        self.check_family("pass_models_sym", "Asym", "physics_race_sym_logit")


class MovementLimits(unittest.TestCase):
    def test_agile_p999(self) -> None:
        limits = load(ROOT / "data/processed/physics_limits/agile_p999_nodelay.json")
        self.assertEqual(limits["name"], "agile")
        for key in ("speed_up", "braking", "turning", "plant_braking", "plant_seconds"):
            self.assertGreater(limits[key], 0.0, key)
        self.assertEqual(limits["defender_delay_s"], 0.0)


class EPVGrid(unittest.TestCase):
    def test_shape_and_range(self) -> None:
        with (ROOT / "data/static/EPV_grid.csv").open() as f:
            rows = list(csv.reader(f))
        self.assertEqual(len(rows), 32)
        self.assertTrue(all(len(r) == 50 for r in rows))
        values = [float(v) for r in rows for v in r]
        self.assertTrue(all(0.0 <= v <= 1.0 for v in values))


if __name__ == "__main__":
    unittest.main()
