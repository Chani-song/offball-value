"""Score R9 (coupled clocks) against R6 (decoupled) on a labelled set.

R6 pairs a 1.5 s time-average of occupation with a max-over-release-times Q.
R9 asks for one instant that satisfies both. Everything else — the defender's
runner-tracking response, the coverage geometry, the labels — is identical, so
the difference between the two columns is the coupling and nothing else.

Usage:
    python scripts/score_coupled_beneficiary.py <audit_dir> [<audit_dir> ...]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from offball_value.assignment_rule import attacking_team_id, onset_state  # noqa: E402
from offball_value.coupled_beneficiary import rule_r9, value_at_times  # noqa: E402
from offball_value.vacated_space import rule_r5  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "score_vacated_space_rule", ROOT / "scripts" / "score_vacated_space_rule.py"
)
_scorer = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_scorer)


def _path_lookup(rows):
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


def absolute_lookups(scene: dict, defender: dict):
    """(defender_at, attacker_at) keyed on absolute seconds from the onset.

    R6's scorer indexes attackers by a fraction of its fixed window; R9 needs
    real time, because the whole point is to line the occupation instant up
    with the pass-arrival instant.
    """
    path = next(
        (
            row["path_txy"]
            for row in defender["responses"]
            if row.get("kind") == "target_conditioned_baseline"
        ),
        None,
    )
    if not path:
        return None, None
    defender_at = _path_lookup([(float(p[0]), float(p[1]), float(p[2])) for p in path])

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

    return defender_at, attacker_at


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audit_dirs", nargs="+")
    parser.add_argument("--exclude-frame", action="append", default=[])
    options = parser.parse_args()

    labels = _scorer.load_labels()
    scenes: dict[tuple[str, str], dict] = {}
    for directory in options.audit_dirs:
        payload = json.loads(
            (Path(directory) / "local_game_payoff_audits.json").read_text()
        )
        for scene in payload:
            scenes[(str(scene["match_id"]), str(scene["onset_frame_id"]))] = scene

    hits = {"R6": 0, "R9": 0}
    total = 0
    misses: dict[str, list] = {"R6": [], "R9": []}
    table: dict[str, list[str]] = {}

    for (match_id, frame_id, defender_id), answer in sorted(labels.items()):
        if frame_id in options.exclude_frame:
            continue
        scene = scenes.get((match_id, frame_id))
        if scene is None:
            continue
        state = onset_state(scene)
        if defender_id not in state:
            continue
        samples, attacker_fraction = _scorer.reaction_samples(scene, defender_id)
        if not samples:
            continue
        defender = next(
            row
            for row in scene["candidate_defenders"]
            if str(row["defender_id"]) == defender_id
        )
        response = next(
            (
                row
                for row in defender["responses"]
                if str(row["response_id"]) == str(defender.get("direct_best_response_id"))
            ),
            None,
        )
        if response is None:
            continue
        defender_at, attacker_at = absolute_lookups(scene, defender)
        if defender_at is None:
            continue

        total += 1
        team = attacking_team_id(scene)
        runner = str(scene["runner_id"])
        carrier = str(scene["carrier_id"])
        names = {str(o["option_id"]): str(o["option_name"]) for o in defender["options"]}

        priced = {
            option_id: cell
            for option_id, cell in response["cells"].items()
            if cell.get("legal") is not False
            and cell.get("q") is not None
            and option_id != runner
        }
        values = {option_id: float(cell["q"]) for option_id, cell in priced.items()}
        curves = {
            option_id: value_at_times(cell.get("candidate_grid") or [])
            for option_id, cell in priced.items()
        }

        _, detail = rule_r5(
            state, team, runner, carrier, defender_id, samples, attacker_fraction
        )
        occupation = {k: v for k, v in detail.items() if not k.startswith("_")}
        r6_scores = {
            option_id: occupation.get(option_id, 0.0) * value
            for option_id, value in values.items()
        }
        r6 = max(r6_scores, key=lambda k: (r6_scores[k], k)) if r6_scores else ""

        r9, r9_detail = rule_r9(
            state,
            team,
            runner,
            carrier,
            defender_id,
            defender_at,
            attacker_at,
            curves,
            float(scene["horizon_seconds"]),
        )

        for name, prediction in (("R6", r6), ("R9", r9)):
            if prediction == answer:
                hits[name] += 1
            else:
                misses[name].append(
                    (
                        frame_id,
                        str(defender["defender_name"]),
                        names.get(answer, answer),
                        names.get(prediction, prediction or "(none)"),
                    )
                )
        table[f"{frame_id}/{defender['defender_name']}"] = [
            "O" if r6 == answer else "X",
            "O" if r9 == answer else "X",
            f"{r9_detail.get('_vacated_area', 0.0):7.1f}",
        ]

    print(f"n = {total}\n")
    print(f"  R6 (two separate clocks)  {hits['R6']}")
    print(f"  R9 (one coupled instant)  {hits['R9']}\n")

    both = sum(1 for row in table.values() if row[0] == "O" and row[1] == "O")
    only6 = sum(1 for row in table.values() if row[0] == "O" and row[1] == "X")
    only9 = sum(1 for row in table.values() if row[0] == "X" and row[1] == "O")
    neither = sum(1 for row in table.values() if row[0] == "X" and row[1] == "X")
    print(f"  agreement: both {both} / R6 only {only6} / R9 only {only9} / both wrong {neither}\n")

    print("  per scene (R6, R9, vacated area)")
    for key, row in sorted(table.items()):
        mark = "  <-- differs" if row[0] != row[1] else ""
        print(f"    {key:<40} {row[0]} {row[1]} {row[2]}{mark}")

    for name in ("R6", "R9"):
        print(f"\n  {name} misses:")
        for frame_id, defender_name, answer, prediction in sorted(misses[name]):
            print(f"    {frame_id}/{defender_name:<18} answer={answer:<20} {name}={prediction}")


if __name__ == "__main__":
    main()
