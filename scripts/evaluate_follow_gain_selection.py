"""Score pins-off derived selection under the follow-gain criterion.

The reviewer-defined objective (docs/derived_selection_criterion_v0_2.md):
the derived option is the player whose value differs most between the
defender following the runner and covering that player instead,

    k* = argmax_{k != runner} [ Q_k(d_follow-R) - Q_k(d_cover-k) ]

where d_follow-R is the feasible response minimising the runner's value and
d_cover-k the feasible response minimising option k's value.  Everything is
recomputed from the recorded response cells, so manifest pins never enter
the selection — they are only the labels being predicted.

Two formalisations of the same idea are reported side by side, because the
v0.3.8 audit showed they disagree and the seven labels cannot referee them:

- ``option`` (as originally implemented): difference of maxima,
  ``max_a Q_k(a|follow) - min_d max_a Q_k(a|d)``.  The two terms may be
  different attacking actions, so a small reweighting between actions can
  move the realised gain enormously.
- ``action``: difference first, maximum second,
  ``max_a [ Q_k(a|follow) - min_d Q_k(a|d) ]`` over actions matched across
  responses by identity.  This measures what the defender's choice costs a
  specific attacking action, which is what the criterion means.  Action
  identity is ambiguous, so both ``--action-key type`` (continuation channel
  only) and ``--action-key full`` (channel + lead + release time) are
  offered; report BOTH, never the flattering one.

Scores over seven labels are not validation: they are burnt (see
docs/goal_danger_surface_v0_3_8_audit.md section 6).

Usage:
    python scripts/evaluate_follow_gain_selection.py DIR [DIR ...]
    python scripts/evaluate_follow_gain_selection.py --mode action --action-key type DIR
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

MANIFEST = Path("examples/research_audit/manifests/confirmed_core_scenes.json")


def _cell_value(cell: dict) -> float:
    if cell is None or cell.get("legal") is False or cell.get("q") is None:
        return 0.0
    return float(cell["q"])


# `terminal_structure` is not an attacking action: it is a discounted
# post-horizon value floor with no release time, no receiver and no
# delivery. The v0.3.8 audit found it carrying 5 of 7 winning follow cells,
# with the whole follow-gain coming from where the DEFENDER finishes
# running (53833-F: follow and cover are literally the same cell). Crediting
# a defender with denying an action nobody can execute is a category error,
# so the criterion may exclude it while the value model keeps it.
STRUCTURAL_ROW_TYPES = {"terminal_structure"}


def _action_values(
    cell: dict,
    action_key: str,
    executable_only: bool = False,
) -> dict[tuple, float]:
    """Best q per attacking action inside one cell's candidate grid."""

    if cell is None or cell.get("legal") is False:
        return {}
    values: dict[tuple, float] = {}
    for row in cell.get("candidate_grid") or []:
        if executable_only and str(row["type"]) in STRUCTURAL_ROW_TYPES:
            continue
        if action_key == "type":
            key = (str(row["type"]),)
        else:
            key = (
                str(row["type"]),
                None if row.get("lead") is None else round(float(row["lead"]), 3),
                round(float(row.get("release") or 0.0), 3),
            )
        value = float(row["q"])
        if value > values.get(key, -1.0):
            values[key] = value
    return values


def _gains(
    responses: list[dict],
    option_ids: list[str],
    runner_id: str,
    mode: str,
    action_key: str,
    executable_only: bool = False,
) -> tuple[dict[str, float], int]:
    """Follow-gain per option, plus the index of the follow response."""

    runner_values = [_cell_value(r["cells"].get(runner_id)) for r in responses]
    follow_index = min(range(len(responses)), key=lambda i: runner_values[i])
    gains: dict[str, float] = {}
    for option_id in option_ids:
        if option_id == runner_id:
            continue
        if mode == "option":
            values = [_cell_value(r["cells"].get(option_id)) for r in responses]
            gains[option_id] = values[follow_index] - min(values)
            continue
        # Per-action: difference against the same action's best cover.
        follow_actions = _action_values(
            responses[follow_index]["cells"].get(option_id),
            action_key,
            executable_only,
        )
        if not follow_actions:
            gains[option_id] = 0.0
            continue
        cover_floor: dict[tuple, float] = {}
        for response in responses:
            for key, value in _action_values(
                response["cells"].get(option_id), action_key, executable_only
            ).items():
                if value < cover_floor.get(key, float("inf")):
                    cover_floor[key] = value
        gains[option_id] = max(
            value - cover_floor.get(key, 0.0)
            for key, value in follow_actions.items()
        )
    return gains, follow_index


def _score_dir(
    audit_dir: Path,
    labels: dict[str, dict[str, str]],
    mode: str,
    action_key: str,
    executable_only: bool = False,
) -> None:
    games = json.loads((audit_dir / "local_game_payoff_audits.json").read_text())
    hits = 0
    total = 0
    variant = mode + (f"/{action_key}" if mode == "action" else "")
    if executable_only:
        variant += "/executable"
    print(f"\n=== {audit_dir} · mode={variant} ===")
    for game in games:
        scene = str(game["onset_frame_id"])
        scene_labels = labels.get(scene, {})
        for defender in game["candidate_defenders"]:
            defender_id = str(defender["defender_id"])
            if defender_id not in scene_labels:
                continue
            label = scene_labels[defender_id]
            responses = [
                response
                for response in defender["responses"]
                if response.get("is_search_candidate", True)
            ] or defender["responses"]
            runner_id = str(game["runner_id"])
            option_ids = [
                str(option["option_id"]) for option in defender["options"]
            ]
            if runner_id not in option_ids:
                print(f"  {scene}/{defender_id}: runner cell missing — skipped")
                continue
            gains, _ = _gains(
                responses, option_ids, runner_id, mode, action_key, executable_only
            )
            pick = max(gains, key=lambda option_id: gains[option_id])
            ok = pick == label
            hits += ok
            total += 1
            names = {
                str(option["option_id"]): str(option.get("option_name", ""))
                for option in defender["options"]
            }
            ranked = sorted(gains.items(), key=lambda item: -item[1])[:3]
            ranked_text = ", ".join(
                f"{names.get(k, k) or k} {v:.3f}" for k, v in ranked
            )
            print(
                f"  {'O' if ok else 'X'} {scene}/{defender['defender_name']}: "
                f"pick={names.get(pick, pick) or pick} label={names.get(label, label) or label}"
                f"  [{ranked_text}]"
            )
    print(f"  pins-off score: {hits}/{total}   (NOT validation — see docs/goal_danger_surface_v0_3_8_audit.md §6)")


def main() -> None:
    args = [arg for arg in sys.argv[1:]]
    mode = "option"
    action_key = "type"
    executable_only = False
    dirs: list[str] = []
    index = 0
    while index < len(args):
        if args[index] == "--mode":
            mode = args[index + 1]
            index += 2
        elif args[index] == "--action-key":
            action_key = args[index + 1]
            index += 2
        elif args[index] == "--executable-only":
            executable_only = True
            index += 1
        elif args[index] == "--all":
            mode = "all"
            index += 1
        else:
            dirs.append(args[index])
            index += 1
    manifest = json.loads(MANIFEST.read_text())
    labels = {
        str(scene["onset_frame_id"]): {
            str(d): str(k)
            for d, k in (scene.get("confirmed_derived_ids") or {}).items()
        }
        for scene in manifest
    }
    variants = (
        [
            ("option", "type", False),
            ("action", "type", False),
            ("action", "full", False),
            ("action", "type", True),
            ("action", "full", True),
        ]
        if mode == "all"
        else [(mode, action_key, executable_only)]
    )
    for arg in dirs:
        for variant_mode, variant_key, variant_exec in variants:
            _score_dir(Path(arg), labels, variant_mode, variant_key, variant_exec)


if __name__ == "__main__":
    main()
