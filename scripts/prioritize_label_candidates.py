"""Rank unreviewed onset candidates by how much the two G surfaces disagree.

The v0.3.8 audit (docs/goal_danger_surface_v0_3_8_audit.md) established that
the seven existing labels cannot referee the goal-danger surface: no game in
the set has its decision turn on a wide-versus-central comparison, which is
exactly where the geometric proxy is demonstrably broken (it prices the
corner flag above the top of the D).  New labels are therefore worth far
more when they are drawn from scenes where the two surfaces ORDER the
attacking options differently -- reviewing those first buys the most
information per scene the human has to watch.

For every candidate this scores, at the onset frame, the attacking players'
goal danger under both surfaces and reports:

- ``ordering_disagreement``: fraction of attacker pairs whose ranking flips
  between the surfaces (0 = identical ordering, 1 = fully reversed).
- ``top_option_flips``: whether the single most dangerous attacker changes.
- ``wide_central_span``: how much lateral spread the attackers cover, since
  the surfaces differ most across the pitch width.

Usage:
    python scripts/prioritize_label_candidates.py \
        --audit-json data/processed/shot_context_run_onset_v0_2_expanded/shot_context_onset_audit.json \
        --previous-selection data/processed/shot_context_run_onset_v0_1/audit_selection.csv \
        --out data/processed/shot_context_run_onset_v0_2_expanded/label_priority.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from offball_value.obso import score_at_points

LEARNED_GRID = (
    "data/processed/progression_goal_danger_v0/progression_epv_grid_v0.csv"
)


def _onset_frame(scene: dict) -> dict:
    return min(scene["frames"], key=lambda f: abs(float(f["relative_time_s"])))


def _attacker_points(scene: dict) -> list[tuple[str, str, float, float]]:
    frame = _onset_frame(scene)
    team = str(scene["team_id"])
    rows = []
    for player in frame["players"]:
        if str(player[1]) != team:
            continue
        rows.append((str(player[0]), str(player[4]), float(player[2]), float(player[3])))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-json", type=Path, required=True)
    parser.add_argument("--previous-selection", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    seen: set[tuple[str, str]] = set()
    if args.previous_selection and args.previous_selection.exists():
        for row in csv.DictReader(args.previous_selection.open()):
            seen.add((str(row["match_id"]), str(row["frame_id"])))

    scenes = json.loads(args.audit_json.read_text(encoding="utf-8"))
    rows = []
    for scene in scenes:
        key = (str(scene["match_id"]), str(scene["onset_frame_id"]))
        attackers = _attacker_points(scene)
        if len(attackers) < 3:
            continue
        points = [(x, y) for _, _, x, y in attackers]
        direction = int(scene["attacking_direction"])
        geometric = score_at_points(points, direction, epv_grid_path=None)
        learned = score_at_points(points, direction, epv_grid_path=LEARNED_GRID)
        pairs = list(combinations(range(len(attackers)), 2))
        flips = sum(
            1
            for i, j in pairs
            if (geometric[i] - geometric[j]) * (learned[i] - learned[j]) < 0
        )
        top_geometric = max(range(len(attackers)), key=lambda i: geometric[i])
        top_learned = max(range(len(attackers)), key=lambda i: learned[i])
        ys = [y for _, _, _, y in attackers]
        rows.append(
            {
                "reviewed_before": key in seen,
                "match_id": scene["match_id"],
                "onset_frame_id": scene["onset_frame_id"],
                "runner_name": scene["runner_name"],
                "carrier_name": scene["carrier_name"],
                "seconds_before_shot": scene["seconds_before_shot"],
                "ordering_disagreement": round(flips / len(pairs), 4),
                "top_option_flips": int(top_geometric != top_learned),
                "top_geometric": attackers[top_geometric][1],
                "top_learned": attackers[top_learned][1],
                "wide_central_span_m": round(max(ys) - min(ys), 1),
                "attackers": len(attackers),
            }
        )

    fresh = [row for row in rows if not row["reviewed_before"]]
    fresh.sort(
        key=lambda row: (
            -row["top_option_flips"],
            -row["ordering_disagreement"],
            -row["wide_central_span_m"],
        )
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(fresh + [row for row in rows if row["reviewed_before"]])

    print(f"{len(rows)} candidates scored; {len(fresh)} never reviewed")
    print(f"wrote {args.out}\n")
    print("Review these first (highest information per scene):")
    print(
        f"{'#':>3} {'match':<18}{'frame':>8}  {'runner':<20}"
        f"{'pair-flips':>11}{'top?':>6}  geometric -> learned"
    )
    for index, row in enumerate(fresh[:14], start=1):
        print(
            f"{index:>3} {row['match_id']:<18}{row['onset_frame_id']:>8}  "
            f"{row['runner_name'][:19]:<20}{row['ordering_disagreement']:>11.3f}"
            f"{'YES' if row['top_option_flips'] else '-':>6}  "
            f"{row['top_geometric'][:16]} -> {row['top_learned'][:16]}"
        )


if __name__ == "__main__":
    main()
