"""Does the coupled rule need the attacker's real future, or just the coupling?

The coupled cell runs to the scene horizon (~2.8 s) where R6 stopped at
1.5 s, so it reads about twice as much observed attacker movement. That is
a fair objection: maybe the extra point comes from seeing more of what
actually happened, not from lining the two clocks up.

Two controls answer it. Replace the observed attacker path with

  constant velocity  extrapolated from the onset state alone
  frozen             held at the onset position

and re-run both the decoupled (R6-style, static Q) and the coupled rule at
each horizon. If the coupled rule beats the decoupled one under the SAME
motion model, the coupling is doing the work; if the gap only appears with
observed motion at the long horizon, the extra future is.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

_spec = importlib.util.spec_from_file_location(
    "decompose_clock_coupling", ROOT / "scripts" / "decompose_clock_coupling.py"
)
_decompose = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_decompose)


def hits(games, *, value, horizon_cap, region="loss"):
    total_hits = 0
    for _, cache, answer in games:
        times = np.array(cache["times"])
        keep = times <= horizon_cap + 1e-9
        weight = cache["weights"][region][keep]
        denominator = float(weight.sum())
        scores = {}
        for option_id in cache["options"]:
            occupation = cache["occupation"][region][option_id][keep]
            series = occupation * (
                cache["static_value"][option_id]
                if value == "static"
                else cache["coupled_value"][option_id][keep]
            )
            scores[option_id] = (
                float((weight * series).sum() / denominator)
                if denominator > 1e-12
                else 0.0
            )
        if scores and max(scores, key=lambda k: (scores[k], k)) == answer:
            total_hits += 1
    return total_hits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audit_dirs", nargs="+")
    options = parser.parse_args()

    labels = _decompose._scorer.load_labels()
    scenes = {}
    for directory in options.audit_dirs:
        for scene in json.loads(
            (Path(directory) / "local_game_payoff_audits.json").read_text()
        ):
            scenes[(str(scene["match_id"]), str(scene["onset_frame_id"]))] = scene

    print(f"{'공격수 궤적':<16} {'창':<6} {'분리 Q':>7} {'결합 Q':>7} {'차이':>6}")
    for motion, title in (
        ("observed", "관측 (실제)"),
        ("constant_velocity", "등속 외삽"),
        ("frozen", "온셋 고정"),
    ):
        games = []
        for (match_id, frame_id, defender_id), answer in sorted(labels.items()):
            scene = scenes.get((match_id, frame_id))
            if scene is None:
                continue
            cache = _decompose.build_cache(scene, defender_id, motion)
            if cache is not None:
                games.append((frame_id, cache, answer))
        for cap, shown in ((1.5, "1.5초"), (99.0, "전체")):
            static = hits(games, value="static", horizon_cap=cap)
            coupled = hits(games, value="coupled", horizon_cap=cap)
            print(
                f"{title:<16} {shown:<6} {static:>7} {coupled:>7} "
                f"{coupled - static:>+6}"
            )
    print(f"\n(n = {len(games)})")


if __name__ == "__main__":
    main()
