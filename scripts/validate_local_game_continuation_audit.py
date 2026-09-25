#!/usr/bin/env python3
"""Validate target-agnostic local-game audit invariants.

This checks the claims that can be established from a generated artifact.  It
does not judge whether a physically feasible path is tactically credible, or
whether the provisional P×G×A value is calibrated.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "audit_json",
        type=Path,
        help="Generated local_game_payoff_audits.json",
    )
    return parser.parse_args()


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _cell_q(response: dict[str, Any], option_id: str) -> float:
    return float(response["cells"][option_id]["q"])


def _best_response(
    responses: list[dict[str, Any]],
    option_id: str,
) -> dict[str, Any]:
    return min(
        responses,
        key=lambda response: (
            _cell_q(response, option_id),
            float(response["effort_m2ps3"]),
            str(response["response_id"]),
        ),
    )


def _validate_scene(scene: dict[str, Any]) -> int:
    _assert(
        scene["schema_version"] == "target-agnostic-local-game-audit-v0.2",
        "artifact is not target-agnostic v0.2",
    )
    payoff = scene["config"]["payoff"]
    structure = scene["config"]["structure"]
    _assert(
        payoff["response_search_mode"] == "target_agnostic",
        "response search mode must be target_agnostic",
    )
    _assert(
        payoff["local_game_selection_mode"] == "pairwise_cross_cost",
        "selection mode must be pairwise_cross_cost",
    )
    checked_cells = 0
    for defender in scene["candidate_defenders"]:
        label = f"{scene['runner_name']} × {defender['defender_name']}"
        responses = list(defender["responses"])
        response_by_id = {
            str(response["response_id"]): response for response in responses
        }
        search = [
            response
            for response in responses
            if bool(response["is_search_candidate"])
        ]
        references = [
            response
            for response in responses
            if not bool(response["is_search_candidate"])
        ]
        _assert(search, f"{label}: no optimizer response paths")
        _assert(
            all(
                response["kind"] == "target_agnostic_feasible"
                for response in search
            ),
            f"{label}: target-conditioned response entered optimizer",
        )
        _assert(
            all(
                response["kind"]
                in {
                    "actual_reference",
                    "target_conditioned_baseline",
                    "causal_policy_reference",
                }
                for response in references
            ),
            f"{label}: unexpected reference response kind",
        )
        _assert(
            defender["actual_response_id"] in response_by_id
            and not response_by_id[defender["actual_response_id"]][
                "is_search_candidate"
            ],
            f"{label}: actual defense entered optimizer",
        )
        _assert(
            defender["search_response_count"] == len(search),
            f"{label}: screened response count mismatch",
        )
        _assert(
            defender["raw_search_response_count"] >= len(search),
            f"{label}: raw response count is smaller than exact set",
        )

        options = list(defender["options"])
        option_by_id = {str(option["option_id"]): option for option in options}
        direct_id = str(defender["direct_option_id"])
        derived_value = defender["derived_option_id"]
        derived_id = str(derived_value) if derived_value is not None else None
        expected_local_ids = (
            [direct_id, derived_id] if derived_id is not None else [direct_id]
        )
        _assert(
            defender["local_option_ids"] == expected_local_ids,
            f"{label}: local option ids disagree with derived selection",
        )
        # Value layer must sit on exact minima; the displayed anchors may be
        # tail-sensible representatives within the anchor tie tolerance.
        tolerance = float(payoff.get("anchor_tie_tolerance", 0.0))

        def _within(display_q: float, exact_q: float) -> bool:
            return display_q <= exact_q + max(1e-9, exact_q * tolerance)

        direct_best = _best_response(search, direct_id)
        _assert(
            defender["cross_cost"]["runner_best_response_id"]
            == direct_best["response_id"],
            f"{label}: cross-cost runner minimum is not the exact minimum",
        )
        direct_display = response_by_id[str(defender["direct_best_response_id"])]
        _assert(
            _within(
                _cell_q(direct_display, direct_id),
                _cell_q(direct_best, direct_id),
            ),
            f"{label}: DIRECT MIN display anchor exceeds the tie tolerance",
        )
        if derived_id is not None:
            derived_best = _best_response(search, derived_id)
            _assert(
                defender["cross_cost"]["derived_best_response_id"]
                == derived_best["response_id"],
                f"{label}: cross-cost derived minimum is not the exact minimum",
            )
            derived_display = response_by_id[
                str(defender["derived_best_response_id"])
            ]
            _assert(
                _within(
                    _cell_q(derived_display, derived_id),
                    _cell_q(derived_best, derived_id),
                ),
                f"{label}: DERIVED MIN display anchor exceeds the tie tolerance",
            )
        else:
            _assert(
                defender["derived_best_response_id"] is None,
                f"{label}: runner-only game has a DERIVED MIN response",
            )

        exact_minimax_worst = min(
            max(_cell_q(response, option_id) for option_id in expected_local_ids)
            for response in search
        )
        if "exact_minimax_worst_q" in defender:
            _assert(
                math.isclose(
                    float(defender["exact_minimax_worst_q"]),
                    exact_minimax_worst,
                    abs_tol=1e-10,
                ),
                f"{label}: exact minimax worst disagrees with the cells",
            )
        minimax_display = response_by_id[str(defender["minimax_response_id"])]
        _assert(
            _within(
                max(
                    _cell_q(minimax_display, option_id)
                    for option_id in expected_local_ids
                ),
                exact_minimax_worst,
            ),
            f"{label}: MINIMAX display anchor exceeds the tie tolerance",
        )

        selected_strength = float(defender["cross_cost"]["strength"])
        candidate_strengths: list[tuple[float, str]] = []
        for option in options:
            option_id = str(option["option_id"])
            if (
                option_id == direct_id
                or float(option["peak_q"]) < float(payoff["minimum_peak_q"])
                or float(option["relative_effect"])
                < float(payoff["minimum_relative_response_effect"])
            ):
                continue
            option_best = _best_response(search, option_id)
            runner_cost = (
                _cell_q(option_best, direct_id)
                - _cell_q(direct_best, direct_id)
            )
            option_cost = (
                _cell_q(direct_best, option_id)
                - _cell_q(option_best, option_id)
            )
            runner_relative = runner_cost / max(
                _cell_q(option_best, direct_id), 1e-9
            )
            option_relative = option_cost / max(
                _cell_q(direct_best, option_id), 1e-9
            )
            candidate_strengths.append(
                (
                    min(max(0.0, runner_relative), max(0.0, option_relative)),
                    option_id,
                )
            )
        if bool(defender.get("derived_pinned")) or defender.get(
            "derived_selection_source"
        ):
            # The derived branch was chosen outside the cross-cost ranking —
            # either a human pin or the provisional assignment rule
            # (derived_selection_source). The ranking assertions do not
            # apply, but the selected id must be a real, priced option and
            # the rule case must preserve the value-based alternative field.
            _assert(
                derived_id in option_by_id,
                f"{label}: externally selected derived option is absent "
                "from the catalogue",
            )
            if defender.get("derived_selection_source"):
                _assert(
                    "derived_option_id_payoff" in defender,
                    f"{label}: assignment-rule selection must record the "
                    "payoff-based alternative",
                )
        elif candidate_strengths:
            maximum_strength = max(value for value, _ in candidate_strengths)
            _assert(
                math.isclose(
                    selected_strength, maximum_strength, abs_tol=1e-10
                ),
                f"{label}: selected derived option lacks maximum cross-cost strength",
            )
            _assert(
                any(
                    option_id == derived_id
                    and math.isclose(
                        strength, maximum_strength, abs_tol=1e-10
                    )
                    for strength, option_id in candidate_strengths
                ),
                f"{label}: derived option id disagrees with cross-cost ranking",
            )
            _assert(
                derived_id in option_by_id,
                f"{label}: derived option is absent from option catalogue",
            )
        else:
            _assert(
                derived_id is None and math.isclose(selected_strength, 0.0),
                f"{label}: a derived branch was forced without eligible options",
            )

        delay = float(structure["response_delay_seconds"])
        shared_prefix = [
            tuple(map(float, row))
            for row in search[0]["path_txy"]
            if float(row[0]) <= delay + 1e-9
        ]
        for response in search:
            path = [tuple(map(float, row)) for row in response["path_txy"]]
            prefix = [row for row in path if row[0] <= delay + 1e-9]
            _assert(
                len(prefix) == len(shared_prefix)
                and all(
                    all(math.isclose(a, b, abs_tol=1e-9) for a, b in zip(x, y))
                    for x, y in zip(prefix, shared_prefix)
                ),
                f"{label}: a path applies control inside response delay",
            )
            physical = response["physical_diagnostics"]
            bounds = (
                (
                    "maximum_path_speed_mps",
                    "maximum_speed_mps",
                ),
                (
                    "maximum_tangential_acceleration_mps2",
                    "maximum_acceleration_mps2",
                ),
                (
                    "maximum_tangential_deceleration_mps2",
                    "maximum_deceleration_mps2",
                ),
                (
                    "maximum_normal_acceleration_mps2",
                    "maximum_normal_acceleration_mps2",
                ),
            )
            for diagnostic, limit in bounds:
                _assert(
                    float(physical[diagnostic])
                    <= float(structure[limit]) + 1e-9,
                    f"{label}: {diagnostic} exceeds {limit}",
                )

        for response in responses:
            for option in options:
                cell = response["cells"][str(option["option_id"])]
                if cell["legal"] is False:
                    continue
                expected_q = (
                    float(cell["delivery"])
                    * float(cell["goal"])
                    * float(cell["accessibility"])
                )
                _assert(
                    math.isclose(float(cell["q"]), expected_q, abs_tol=1e-10),
                    f"{label}: Q != P×G×A for {response['response_id']}",
                )
                checked_cells += 1
    return checked_cells


def validate_artifact(path: Path) -> tuple[int, int, int]:
    scenes = json.loads(path.read_text(encoding="utf-8"))
    _assert(isinstance(scenes, list) and scenes, "audit must contain scenes")
    defender_count = sum(len(scene["candidate_defenders"]) for scene in scenes)
    checked_cells = sum(_validate_scene(scene) for scene in scenes)
    return len(scenes), defender_count, checked_cells


def main() -> None:
    args = parse_args()
    scenes, defenders, cells = validate_artifact(args.audit_json)
    print(
        "PASS · "
        f"{scenes} scene(s), {defenders} defender game(s), "
        f"{cells} legal P×G×A cells checked"
    )


if __name__ == "__main__":
    main()
