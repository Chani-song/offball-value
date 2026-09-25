"""Isolate what the two-clock coupling actually costs or buys.

The first coupled attempt (R9) scored 22 on round 1 against R6's 23, but it
changed four things at once, so that number says nothing about the coupling
itself. This script varies them one at a time on a shared per-scene cache:

  region   instantaneous loss (R5's own structure) | cumulative max trail
  horizon  1.5 s (R6's constant)                   | the scene's full horizon
  aggregate area-weighted mean over instants       | max over instants
  value    static Q (max over all release times)   | Q(t) at that instant

Cell (loss, 1.5, mean, static) IS R6 and must reproduce 23 — that is the
sanity check that the cache is faithful before any other cell is believed.

The claim under test is the reviewer's (2026-09-09): occupying the space and
receiving the ball there should refer to the same moment. Only the `value`
axis tests it; the other three are controls that stop a win or a loss from
being credited to the wrong change.

Usage:
    python scripts/decompose_clock_coupling.py <audit_dir> [...] [--json OUT]
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

from offball_value.assignment_rule import attacking_team_id, onset_state  # noqa: E402
from offball_value.coupled_beneficiary import value_at_times  # noqa: E402
from offball_value.vacated_space import (  # noqa: E402
    _region_grid,
    _state_v,
    _state_xy,
    coverage_field,
)

_spec = importlib.util.spec_from_file_location(
    "score_vacated_space_rule", ROOT / "scripts" / "score_vacated_space_rule.py"
)
_scorer = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_scorer)

STEP_S = 0.25
SHORT_HORIZON_S = 1.5
# How far from an instant a priced pass still counts as arriving "then".
VALUE_WINDOW_S = 0.4


def _sampler(rows):
    def at(time_s: float) -> tuple[float, float]:
        previous = rows[0]
        for row in rows:
            if row[0] >= time_s:
                if row[0] == previous[0]:
                    return (row[1], row[2])
                fraction = (time_s - previous[0]) / (row[0] - previous[0])
                return (
                    previous[1] + fraction * (row[1] - previous[1]),
                    previous[2] + fraction * (row[2] - previous[2]),
                )
            previous = row
        return (rows[-1][1], rows[-1][2])

    def sampled(time_s: float):
        now = at(time_s)
        before = at(max(0.0, time_s - 0.2))
        return now, ((now[0] - before[0]) / 0.2, (now[1] - before[1]) / 0.2)

    return sampled


def _attacker_sampler(scene: dict, motion: str = "observed", onset=None):
    """Where an attacker is at ``time_s``.

    ``motion`` selects how much of the future the rule is allowed to see:
    ``observed`` reads the tracking data; ``constant_velocity`` extrapolates
    from the onset state; ``frozen`` holds him at the onset position. The
    coupled rule runs to the scene horizon rather than 1.5 s, so it reads
    more real future than R6 did — these two controls measure whether that
    extra future, rather than the coupling, is doing the work.
    """
    if motion in ("constant_velocity", "frozen"):
        speed = 0.0 if motion == "frozen" else 1.0

        def extrapolated(player_id: str, time_s: float):
            row = onset.get(player_id)
            if row is None:
                return (0.0, 0.0), (0.0, 0.0)
            px, py = _state_xy(row)
            vx, vy = _state_v(row)
            return (
                (px + speed * vx * time_s, py + speed * vy * time_s),
                (speed * vx, speed * vy),
            )

        return extrapolated

    frames = sorted(
        (
            float(frame["relative_time_s"]),
            {str(p[0]): (float(p[2]), float(p[3])) for p in frame["players"]},
        )
        for frame in scene["background_frames"]
    )

    def attacker_at(player_id: str, time_s: float):
        previous = current = frames[0]
        for frame in frames:
            current = frame
            if frame[0] >= time_s:
                break
            previous = frame
        if player_id not in current[1] or player_id not in previous[1]:
            fallback = (
                current[1].get(player_id) or previous[1].get(player_id) or (0.0, 0.0)
            )
            return fallback, (0.0, 0.0)
        if current[0] == previous[0]:
            return current[1][player_id], (0.0, 0.0)
        span = current[0] - previous[0]
        fraction = (time_s - previous[0]) / span
        start, end = previous[1][player_id], current[1][player_id]
        return (
            (
                start[0] + fraction * (end[0] - start[0]),
                start[1] + fraction * (end[1] - start[1]),
            ),
            ((end[0] - start[0]) / span, (end[1] - start[1]) / span),
        )

    return attacker_at


def _value_near(curve, time_s: float, window_s: float = VALUE_WINDOW_S) -> float:
    """Best Q the pricing stage actually found for a ball arriving near ``t``."""
    if not curve:
        return 0.0
    nearby = [value for t, value in curve if abs(t - time_s) <= window_s]
    if nearby:
        return max(nearby)
    nearest = min(curve, key=lambda item: abs(item[0] - time_s))
    return 0.0 if abs(nearest[0] - time_s) > 1.0 else nearest[1]


def build_cache(scene: dict, defender_id: str, motion: str = "observed") -> dict | None:
    """Per-instant occupation and value tensors for one labelled game."""
    state = onset_state(scene)
    if defender_id not in state:
        return None
    defender = next(
        (
            row
            for row in scene["candidate_defenders"]
            if str(row["defender_id"]) == defender_id
        ),
        None,
    )
    if defender is None:
        return None
    path = next(
        (
            row["path_txy"]
            for row in defender["responses"]
            if row.get("kind") == "target_conditioned_baseline"
        ),
        None,
    )
    if not path:
        return None
    response = next(
        (
            row
            for row in defender["responses"]
            if str(row["response_id"]) == str(defender.get("direct_best_response_id"))
        ),
        None,
    )
    if response is None:
        return None

    runner = str(scene["runner_id"])
    team = attacking_team_id(scene)
    priced = {
        option_id: cell
        for option_id, cell in response["cells"].items()
        if cell.get("legal") is not False
        and cell.get("q") is not None
        and option_id != runner
    }
    if not priced:
        return None

    defender_at = _sampler([(float(p[0]), float(p[1]), float(p[2])) for p in path])
    attacker_at = _attacker_sampler(scene, motion, state)

    horizon = float(scene["horizon_seconds"])
    times: list[float] = []
    time_s = STEP_S
    while time_s <= horizon + 1e-9:
        times.append(round(time_s, 6))
        time_s = round(time_s + STEP_S, 6)

    xs, ys = _region_grid(state[defender_id])
    others = np.zeros((len(ys), len(xs)))
    for player_id, row in state.items():
        if str(row["team"]) == team or player_id == defender_id:
            continue
        np.maximum(
            others, coverage_field(_state_xy(row), _state_v(row), xs, ys), out=others
        )
    before = np.maximum(
        others,
        coverage_field(
            _state_xy(state[defender_id]), _state_v(state[defender_id]), xs, ys
        ),
    )

    options = sorted(priced)
    # occupation[region][option][instant]; weight[region][instant]
    occupation = {
        "loss": {option_id: [] for option_id in options},
        "trail": {option_id: [] for option_id in options},
    }
    weights = {"loss": [], "trail": []}
    trail = np.zeros_like(before)
    for time_s in times:
        after_xy, after_v = defender_at(time_s)
        loss = np.clip(
            before - np.maximum(others, coverage_field(after_xy, after_v, xs, ys)),
            0.0,
            None,
        )
        np.maximum(trail, loss, out=trail)
        regions = {"loss": loss, "trail": trail.copy()}
        for key, region in regions.items():
            total = float(region.sum())
            weights[key].append(total)
            for option_id in options:
                if total <= 1e-12:
                    occupation[key][option_id].append(0.0)
                    continue
                xy, velocity = attacker_at(option_id, time_s)
                occupation[key][option_id].append(
                    float((region * coverage_field(xy, velocity, xs, ys)).sum() / total)
                )

    curves = {
        option_id: value_at_times(cell.get("candidate_grid") or [])
        for option_id, cell in priced.items()
    }
    return {
        "times": times,
        "options": options,
        "weights": {key: np.array(value) for key, value in weights.items()},
        "occupation": {
            key: {o: np.array(v) for o, v in inner.items()}
            for key, inner in occupation.items()
        },
        "static_value": {o: float(cell["q"]) for o, cell in priced.items()},
        "coupled_value": {
            o: np.array([_value_near(curves[o], t) for t in times]) for o in options
        },
        "defender_name": str(defender["defender_name"]),
        "names": {str(o["option_id"]): str(o["option_name"]) for o in defender["options"]},
        "horizon": horizon,
    }


def predict(
    cache: dict, region: str, horizon: str, aggregate: str, value: str
) -> str:
    times = np.array(cache["times"])
    keep = (
        times <= SHORT_HORIZON_S + 1e-9 if horizon == "short" else np.ones_like(times, bool)
    )
    weight = cache["weights"][region][keep]
    scores: dict[str, float] = {}
    for option_id in cache["options"]:
        occupation = cache["occupation"][region][option_id][keep]
        if value == "static":
            series = occupation * cache["static_value"][option_id]
        else:
            series = occupation * cache["coupled_value"][option_id][keep]
        if aggregate == "mean":
            total = float(weight.sum())
            scores[option_id] = (
                float((weight * series).sum() / total) if total > 1e-12 else 0.0
            )
        else:
            scores[option_id] = float(series.max()) if series.size else 0.0
    if not scores:
        return ""
    return max(scores, key=lambda key: (scores[key], key))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audit_dirs", nargs="+")
    parser.add_argument("--json", default=None)
    options = parser.parse_args()

    labels = _scorer.load_labels()
    scenes: dict[tuple[str, str], dict] = {}
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
        cache = build_cache(scene, defender_id)
        if cache is None:
            continue
        games.append((frame_id, cache, answer))
    print(f"채점 가능한 장면 {len(games)}개\n", flush=True)

    grid = []
    for region in ("loss", "trail"):
        for horizon in ("short", "full"):
            for aggregate in ("mean", "max"):
                for value in ("static", "coupled"):
                    hits = 0
                    correct = []
                    for frame_id, cache, answer in games:
                        ok = predict(cache, region, horizon, aggregate, value) == answer
                        hits += ok
                        correct.append(ok)
                    grid.append(
                        {
                            "region": region,
                            "horizon": horizon,
                            "aggregate": aggregate,
                            "value": value,
                            "hits": hits,
                            "correct": correct,
                        }
                    )

    def label(cell) -> str:
        return (
            f"{cell['region']:<5} {cell['horizon']:<5} "
            f"{cell['aggregate']:<4} {cell['value']:<7}"
        )

    baseline = next(
        c
        for c in grid
        if (c["region"], c["horizon"], c["aggregate"], c["value"])
        == ("loss", "short", "mean", "static")
    )
    print(f"R6 재현 (loss/short/mean/static) = {baseline['hits']} / {len(games)}")
    print("  ↑ 23이어야 캐시가 R6에 충실한 것 — 아니면 아래 표는 전부 무효\n")

    print(f"{'구역':<5} {'창':<5} {'집계':<4} {'가치':<7}  점수")
    for cell in sorted(grid, key=lambda c: -c["hits"]):
        mark = "   <-- R6" if cell is baseline else ""
        print(f"{label(cell)}  {cell['hits']:>2}{mark}")

    print("\n한 축씩만 R6에서 바꿨을 때:")
    for axis, other in (
        ("region", "trail"),
        ("horizon", "full"),
        ("aggregate", "max"),
        ("value", "coupled"),
    ):
        key = {
            "region": "loss",
            "horizon": "short",
            "aggregate": "mean",
            "value": "static",
        }
        key[axis] = other
        cell = next(
            c
            for c in grid
            if (c["region"], c["horizon"], c["aggregate"], c["value"])
            == (key["region"], key["horizon"], key["aggregate"], key["value"])
        )
        delta = cell["hits"] - baseline["hits"]
        print(f"  {axis:<9} -> {other:<8} {cell['hits']:>2}  ({delta:+d})")

    if options.json:
        Path(options.json).write_text(
            json.dumps(
                {
                    "n": len(games),
                    "frames": [frame_id for frame_id, _, _ in games],
                    "grid": grid,
                },
                indent=2,
            )
        )
        print(f"\n저장: {options.json}")


if __name__ == "__main__":
    main()
