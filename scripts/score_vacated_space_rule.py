"""Score the vacated-space beneficiary rule (R6) against human labels.

R6 is the reviewer's own formulation, arrived at over the 2026-09-08
session and refined by his reading of the misses:

  beneficiary = argmax over attackers of
      (occupation of the space the reacting defender vacates)  x  Q

- The vacated region accumulates along the defender's whole reaction path
  ("헨젤과 그레텔에 과자 흘리듯"), not at one instant.
- Both sides advance through time: attackers are scored from where they
  actually are at each sampled instant, which is what separates a player
  drifting out of the opening space from one running into it.
- Coverage is position+velocity only (ball excluded), per his definition.
- Q = P x G x A supplies "can he actually receive it there, and is it
  dangerous" — the part pure geometry cannot see. Occupation alone scores
  20/24 on the development set and Q alone scores 7/24; the product is
  what reaches 21+, so neither factor is carrying the other.

Usage:
    python scripts/score_vacated_space_rule.py AUDIT_DIR [AUDIT_DIR ...]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from offball_value.assignment_rule import attacking_team_id, onset_state, rule_r1
from offball_value.vacated_space import rule_r5

REVIEWS = Path("examples/research_audit/human_reviews/shot_context")
HORIZON_S = 1.5
STEP_S = 0.25


def load_labels() -> dict[tuple[str, str, str], str]:
    """Named-beneficiary labels, with the reviewer's confirmed corrections."""
    labels: dict[tuple[str, str, str], str] = {}
    for name in (
        "derived_beneficiary_blind_labels_v1.csv",
        "derived_beneficiary_represent_round1fix.csv",
        "derived_beneficiary_round2_blind_labels.csv",
    ):
        path = REVIEWS / name
        if not path.exists():
            continue
        for row in csv.DictReader(path.open(encoding="utf-8-sig")):
            if row.get("repeat_tag"):
                continue
            if row["defender_reacts"] != "yes":
                continue
            if row["beneficiary"] in ("", "none"):
                continue
            labels[
                (row["match_id"], row["onset_frame_id"], row["defender_id"])
            ] = row["beneficiary"]
    # 131237: reviewer confirmed Hendrix on 2026-09-06 (post-hoc label fix,
    # never counted as blind evidence — see the manifest's history note).
    for key in list(labels):
        if key[1] == "131237":
            labels[key] = "DFL-OBJ-J0130T"
    return labels


def reaction_samples(scene: dict, defender_id: str):
    """Geometric runner-tracking reaction, sampled, plus attacker lookup.

    The reaction is the ``focus_runner`` target-conditioned baseline: a pure
    goal-side pursuit of the runner, so it never depends on the value model
    the rule is later multiplied by.
    """
    defender = next(
        (
            row
            for row in scene["candidate_defenders"]
            if str(row["defender_id"]) == defender_id
        ),
        None,
    )
    if defender is None:
        return None, None
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
    rows = [(float(p[0]), float(p[1]), float(p[2])) for p in path]

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

    samples, times, time_s = [], [], STEP_S
    while time_s <= HORIZON_S + 1e-9:
        now = at(time_s)
        before = at(max(0.0, time_s - 0.2))
        samples.append(
            (now, ((now[0] - before[0]) / 0.2, (now[1] - before[1]) / 0.2))
        )
        times.append(time_s)
        time_s = round(time_s + STEP_S, 6)

    frames = sorted(
        (
            float(frame["relative_time_s"]),
            {str(p[0]): (float(p[2]), float(p[3])) for p in frame["players"]},
        )
        for frame in scene["background_frames"]
    )

    def attacker_at(player_id: str, fraction: float):
        # rule_r5 hands us fraction = (index + 1) / len(samples), so recovering
        # the index needs the -1. Without it every attacker was read a quarter
        # second AHEAD of the defender he was being compared against, which
        # gave attackers a uniform head start (found 2026-09-09; worth one
        # round-2 scene by luck, neutral on round 1).
        index = min(max(int(round(fraction * len(times))) - 1, 0), len(times) - 1)
        target = times[index] if times else 0.0
        previous = frames[0]
        current = frames[0]
        for frame in frames:
            current = frame
            if frame[0] >= target:
                break
            previous = frame
        if player_id not in current[1] or player_id not in previous[1]:
            fallback = current[1].get(player_id) or previous[1].get(player_id) or (0.0, 0.0)
            return fallback, (0.0, 0.0)
        if current[0] == previous[0]:
            return current[1][player_id], (0.0, 0.0)
        span = current[0] - previous[0]
        fraction_t = (target - previous[0]) / span
        start = previous[1][player_id]
        end = current[1][player_id]
        return (
            (
                start[0] + fraction_t * (end[0] - start[0]),
                start[1] + fraction_t * (end[1] - start[1]),
            ),
            ((end[0] - start[0]) / span, (end[1] - start[1]) / span),
        )

    return samples, attacker_at


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audit_dirs", nargs="+")
    args = parser.parse_args()
    labels = load_labels()

    scenes: dict[tuple[str, str], dict] = {}
    for directory in args.audit_dirs:
        payload = Path(directory) / "local_game_payoff_audits.json"
        for scene in json.loads(payload.read_text(encoding="utf-8")):
            scenes[(str(scene["match_id"]), str(scene["onset_frame_id"]))] = scene

    totals = {"R1": 0, "occupation only": 0, "Q only": 0, "R6 = occupation x Q": 0}
    scored = 0
    misses: list[str] = []
    for (match_id, frame_id, defender_id), answer in labels.items():
        scene = scenes.get((match_id, frame_id))
        if scene is None:
            continue
        state = onset_state(scene)
        if defender_id not in state:
            continue
        samples, attacker_at = reaction_samples(scene, defender_id)
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

        scored += 1
        team = attacking_team_id(scene)
        runner_id = str(scene["runner_id"])
        carrier_id = str(scene["carrier_id"])
        names = {
            str(o["option_id"]): str(o["option_name"]) for o in defender["options"]
        }

        totals["R1"] += (
            rule_r1(state, team, runner_id, carrier_id, defender_id)[0] == answer
        )
        occupation_pick, detail = rule_r5(
            state, team, runner_id, carrier_id, defender_id, samples, attacker_at
        )
        totals["occupation only"] += occupation_pick == answer
        occupation = {k: v for k, v in detail.items() if not k.startswith("_")}
        values = {
            option_id: float(cell["q"])
            for option_id, cell in response["cells"].items()
            if cell.get("legal") is not False
            and cell.get("q") is not None
            and option_id != runner_id
        }
        if values:
            q_pick = max(values, key=lambda k: (values[k], k))
            totals["Q only"] += q_pick == answer
            combined = {k: occupation.get(k, 0.0) * v for k, v in values.items()}
            pick = max(combined, key=lambda k: (combined[k], k))
            totals["R6 = occupation x Q"] += pick == answer
            if pick != answer:
                misses.append(
                    f"  {frame_id}/{defender['defender_name']}: "
                    f"label={names.get(answer, answer)} pick={names.get(pick, pick)}"
                )

    print(f"scored games: {scored}")
    for name, value in sorted(totals.items(), key=lambda kv: -kv[1]):
        print(f"  {name:<24}{value}")
    if misses:
        print("misses:")
        for line in misses:
            print(line)


if __name__ == "__main__":
    main()
