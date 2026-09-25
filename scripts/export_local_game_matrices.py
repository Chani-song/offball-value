#!/usr/bin/env python3
"""Joint analysis with the game-theoretic solver (Junhyun quick win).

For every confirmed defender game, build the payoff matrix with UNFOLDED
attack columns — one column per (option, continuation type, lead, release /
carry event) — over the local pair, and solve it three ways:

1. pure defender security  min_d max_a Q  (the attacker best-responds to the
   defender's path; equals our reported oracle worst on the same pair);
2. pure attacker security  max_a min_d Q;
3. the mixed zero-sum equilibrium via the external LP solver.

The defender's mixing gain (1) − (3) is the value of randomizing the
defensive response; the equilibrium supports say which feasible paths and
which attacking continuations carry probability.  Missing candidates
(offside or out of horizon under that defender path) enter as Q = 0, which
is their actual value to the attack.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


def load_solver(junhyun_src: Path):
    sys.path.insert(0, str(junhyun_src))
    from defensive_positioning.search import solve_matrix_game  # type: ignore

    return solve_matrix_game


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audit-json",
        type=Path,
        default=Path(
            "data/processed/local_game_v0_3_canonical_grid/"
            "local_game_payoff_audits.json"
        ),
    )
    parser.add_argument(
        "--junhyun-src",
        type=Path,
        default=Path("/Users/kyuhyeokseo/Desktop/junhyun-mit-ssac2027/src"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/local_game_equilibria_v0_1"),
    )
    return parser.parse_args()


def column_key(option_id: str, row: dict[str, object]) -> tuple:
    if row["type"] == "carrier_carry":
        return (option_id, "carrier_carry", None, round(float(row["event_time"]), 3))
    return (
        option_id,
        str(row["type"]),
        None if row.get("lead") is None else float(row["lead"]),
        round(float(row["release"]), 3),
    )


def main() -> None:
    args = parse_args()
    solve_matrix_game = load_solver(args.junhyun_src)
    scenes = json.loads(args.audit_json.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    report = []
    for scene in scenes:
        for game in scene["candidate_defenders"]:
            search = [
                response
                for response in game["responses"]
                if response.get("is_search_candidate")
            ]
            local_ids = [str(option_id) for option_id in game["local_option_ids"]]

            columns: dict[tuple, int] = {}
            for response in search:
                for option_id in local_ids:
                    for row in response["cells"][option_id].get(
                        "candidate_grid", []
                    ):
                        columns.setdefault(
                            column_key(option_id, row), len(columns)
                        )
            if not columns:
                continue
            matrix = np.zeros((len(search), len(columns)), dtype=float)
            for row_index, response in enumerate(search):
                for option_id in local_ids:
                    for row in response["cells"][option_id].get(
                        "candidate_grid", []
                    ):
                        matrix[
                            row_index, columns[column_key(option_id, row)]
                        ] = float(row["q"])

            pure_defender = float(np.min(np.max(matrix, axis=1)))
            pure_attacker = float(np.max(np.min(matrix, axis=0)))
            defender_mix, attacker_mix, mixed_value = solve_matrix_game(matrix)
            mixed_value = float(mixed_value)

            names = {
                str(option["option_id"]): str(option["option_name"])
                for option in game["options"]
            }
            keys = list(columns)
            defender_support = [
                {
                    "response_id": str(search[i]["response_id"]),
                    "probability": round(float(p), 4),
                }
                for i, p in enumerate(defender_mix)
                if p > 1e-3
            ]
            attacker_support = [
                {
                    "option": names.get(keys[j][0], keys[j][0]),
                    "type": keys[j][1],
                    "lead": keys[j][2],
                    "release_or_event": keys[j][3],
                    "probability": round(float(p), 4),
                }
                for j, p in enumerate(attacker_mix)
                if p > 1e-3
            ]
            entry = {
                "scene": f"{scene['runner_name']}·{scene['onset_frame_id']}",
                "defender": str(game["defender_name"]),
                "matrix_shape": [len(search), len(columns)],
                "reported_oracle_worst": float(game["minimax_worst_q"]),
                "pure_defender_security": pure_defender,
                "pure_attacker_security": pure_attacker,
                "mixed_value": mixed_value,
                "defender_mixing_gain": pure_defender - mixed_value,
                "defender_support": sorted(
                    defender_support, key=lambda r: -r["probability"]
                )[:5],
                "attacker_support": sorted(
                    attacker_support, key=lambda r: -r["probability"]
                )[:5],
            }
            report.append(entry)
            np.savez_compressed(
                args.output_dir
                / f"matrix_{scene['onset_frame_id']}_{game['defender_id']}.npz",
                matrix=matrix,
                response_ids=np.asarray(
                    [str(r["response_id"]) for r in search]
                ),
                column_keys=np.asarray([str(k) for k in keys]),
                defender_mix=defender_mix,
                attacker_mix=attacker_mix,
            )
            print(
                f"{entry['scene']} · {entry['defender']}: "
                f"{len(search)}x{len(columns)} | pure_def "
                f"{pure_defender:.4f} | mixed {mixed_value:.4f} | "
                f"mixing gain {entry['defender_mixing_gain']:+.4f} | "
                f"supports D{len(defender_support)}/A{len(attacker_support)}"
            )

    (args.output_dir / "equilibrium_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nsaved {len(report)} games to {args.output_dir}")


if __name__ == "__main__":
    main()
