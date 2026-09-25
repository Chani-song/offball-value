"""Is the coupled rule's round-1 score a plateau or a spike?

The four-axis decomposition picked one cell — instantaneous loss region, the
scene's full horizon, area-weighted mean, Q read at the same instant. Before
that cell is worth a round-2 look it has to survive its own free parameters:

  value window  how far from an instant a priced pass still counts as
                arriving "then" (0.4 s was a guess)
  sample step   how finely the reaction is sampled (0.25 s, inherited)
  horizon cap   truncating the scene horizon, which is what separates the
                winning cell from the losing single-axis change

A rule that only scores well at one setting of a constant nobody derived is
a fitted rule. A rule that holds across the range is a rule.
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

_probe = importlib.util.spec_from_file_location(
    "decompose_clock_coupling", ROOT / "scripts" / "decompose_clock_coupling.py"
)
_decompose = importlib.util.module_from_spec(_probe)
_probe.loader.exec_module(_decompose)

from offball_value.coupled_beneficiary import value_at_times  # noqa: E402


def score(games, *, window_s, horizon_cap, region="loss", value="coupled"):
    """Round-1 hits for the coupled cell at one setting of its constants."""
    hits, flips = 0, []
    for frame_id, cache, answer, curves in games:
        times = np.array(cache["times"])
        keep = times <= horizon_cap + 1e-9
        weight = cache["weights"][region][keep]
        total = float(weight.sum())
        scores = {}
        for option_id in cache["options"]:
            occupation = cache["occupation"][region][option_id][keep]
            if value == "static":
                series = occupation * cache["static_value"][option_id]
            else:
                series = occupation * np.array(
                    [
                        _decompose._value_near(curves[option_id], t, window_s)
                        for t in times[keep]
                    ]
                )
            scores[option_id] = (
                float((weight * series).sum() / total) if total > 1e-12 else 0.0
            )
        prediction = max(scores, key=lambda k: (scores[k], k)) if scores else ""
        if prediction == answer:
            hits += 1
        else:
            flips.append((frame_id, cache["defender_name"], answer, prediction, cache))
    return hits, flips


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

    games = []
    for (match_id, frame_id, defender_id), answer in sorted(labels.items()):
        scene = scenes.get((match_id, frame_id))
        if scene is None:
            continue
        cache = _decompose.build_cache(scene, defender_id)
        if cache is None:
            continue
        defender = next(
            row
            for row in scene["candidate_defenders"]
            if str(row["defender_id"]) == defender_id
        )
        response = next(
            row
            for row in defender["responses"]
            if str(row["response_id"]) == str(defender.get("direct_best_response_id"))
        )
        curves = {
            option_id: value_at_times(response["cells"][option_id].get("candidate_grid") or [])
            for option_id in cache["options"]
        }
        games.append((frame_id, cache, answer, curves))

    horizons = [float(cache["horizon"]) for _, cache, _, _ in games]
    print(f"n = {len(games)}   장면 지평 {min(horizons):.2f}~{max(horizons):.2f}초\n")

    print("가치 창(window) 민감도 — 지평은 장면 전체")
    for window_s in (0.1, 0.15, 0.25, 0.4, 0.6, 0.8, 1.0, 1.5):
        hits, _ = score(games, window_s=window_s, horizon_cap=99.0)
        print(f"  ±{window_s:>4.2f}초   {hits:>2} / {len(games)}")

    print("\n지평 절단 민감도 — 창은 ±0.4초")
    for cap in (1.0, 1.5, 2.0, 2.5, 3.0, 99.0):
        hits, _ = score(games, window_s=0.4, horizon_cap=cap)
        shown = "전체" if cap > 10 else f"{cap:.1f}초"
        print(f"  {shown:>5}   {hits:>2} / {len(games)}")

    print("\n대조: 같은 격자에서 결합을 끄면 (static Q)")
    for cap in (1.5, 2.0, 2.5, 99.0):
        hits, _ = score(games, window_s=0.4, horizon_cap=cap, value="static")
        shown = "전체" if cap > 10 else f"{cap:.1f}초"
        print(f"  {shown:>5}   {hits:>2} / {len(games)}")

    print("\n구역 정의 민감도 (결합 on, 전체 지평, ±0.4초)")
    for region in ("loss", "trail"):
        hits, _ = score(games, window_s=0.4, horizon_cap=99.0, region=region)
        print(f"  {region:<6} {hits:>2} / {len(games)}")

    hits, flips = score(games, window_s=0.4, horizon_cap=99.0)
    base_hits, base_flips = score(games, window_s=0.4, horizon_cap=1.5, value="static")
    print(f"\nR6 오답 {len(base_flips)}개:")
    for frame_id, name, answer, prediction, cache in base_flips:
        print(
            f"  {frame_id}/{name:<16} 정답={cache['names'].get(answer, answer):<20} "
            f"R6={cache['names'].get(prediction, prediction)}"
        )
    print(f"결합 규칙 오답 {len(flips)}개:")
    for frame_id, name, answer, prediction, cache in flips:
        print(
            f"  {frame_id}/{name:<16} 정답={cache['names'].get(answer, answer):<20} "
            f"결합={cache['names'].get(prediction, prediction)}"
        )


if __name__ == "__main__":
    main()
