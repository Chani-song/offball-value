"""Map a raw delivery score onto an observed completion rate.

The mechanistic delivery chain is a product of five probabilities and sits far
below reality: median 0.32 on real Bundesliga passes that complete 81 % of the
time. Its ORDER carries information (the raw score's lowest decile completes
66 %, its highest 98 %), so the repair is a monotone map from the raw score to
the rate, fitted by isotonic regression on observed passes and held out by
match. scripts/calibrate_hybrid_delivery.py fits it and writes the JSON this
loads.

Applied to PASS deliveries only. Carry retention and the terminal-structure
floor are different quantities on different scales; the fit says nothing about
them, and reusing it there would be a category error.

MEASURED SIDE EFFECT, not a detail: isotonic preserves rank exactly but not
spacing, and the spacing is what an argmax over P x G x A consumes. On the
counterfactual-defender probe the calibrated model keeps only a third of the
raw response to a defender placed on the lane (-0.077 against -0.208). Turning
this on trades level for sensitivity. It is off by default for that reason.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

_CACHE: dict[str, tuple[np.ndarray, np.ndarray]] = {}


def load_calibration(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    key = str(path)
    if key not in _CACHE:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        xs = np.asarray(payload["isotonic_x"], dtype=float)
        ys = np.asarray(payload["isotonic_y"], dtype=float)
        if xs.size == 0 or xs.size != ys.size:
            raise ValueError(f"calibration at {path} has no usable knots")
        if np.any(np.diff(ys) < -1e-9):
            raise ValueError(f"calibration at {path} is not monotone")
        _CACHE[key] = (xs, ys)
    return _CACHE[key]


def apply_calibration(path: str | Path, value: float) -> float:
    xs, ys = load_calibration(path)
    index = int(np.clip(np.searchsorted(xs, float(value), side="right") - 1,
                        0, xs.size - 1))
    return float(np.clip(ys[index], 0.0, 1.0))
