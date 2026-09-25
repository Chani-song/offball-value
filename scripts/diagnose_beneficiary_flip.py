#!/usr/bin/env python3
"""Why did a beneficiary answer change when only the delivery model changed?

The 90-scene xpass360 rebuild lost two labelled pairs against the hybrid
baseline and gained none. The vacated areas printed identically in both runs,
which already says the defender, his chosen response and the geometry are the
same — so loss(t) and occupation(t) in

    score(i) = Σ_t |loss(t)| · occupation_i(t) · Q_i(t) / Σ_t |loss(t)|

are unchanged and the flip lives entirely in Q. This prints the R9 score for
every candidate beneficiary under both builds side by side, with the Q and the
delivery that produced it, so the mechanism is visible rather than inferred.

The hypothesis it is meant to test: xpass360 puts 44 % of cells above P = 0.9
(hybrid: 3.5 %), so if delivery saturates, Q collapses toward G × A and the
argmax stops being driven by how well the ball can actually get there.

Usage:
    python scripts/diagnose_beneficiary_flip.py \
        --baseline data/processed/goalside_v1_chunk0 ... \
        --new out/scene_01 ... \
        --pair 105845 --pair 31860
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

_spec = importlib.util.spec_from_file_location(
    "score_coupled_beneficiary", ROOT / "scripts" / "score_coupled_beneficiary.py"
)
_coupled = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_coupled)
_scorer = _coupled._scorer


def load(dirs):
    scenes = {}
    for directory in dirs:
        payload = json.loads(
            (Path(directory) / "local_game_payoff_audits.json").read_text(
                encoding="utf-8"
            )
        )
        for scene in payload:
            scenes[(str(scene["match_id"]), str(scene["onset_frame_id"]))] = scene
    return scenes


def evaluate(scene, defender_id):
    """R9 pick plus, per option, its score, peak Q and the delivery behind it."""
    state = onset_state(scene)
    if defender_id not in state:
        return None
    samples, attacker_fraction = _scorer.reaction_samples(scene, defender_id)
    if not samples:
        return None
    defender = next(
        (r for r in scene["candidate_defenders"] if str(r["defender_id"]) == defender_id),
        None,
    )
    if defender is None:
        return None
    response = next(
        (
            r
            for r in defender["responses"]
            if str(r["response_id"]) == str(defender.get("direct_best_response_id"))
        ),
        None,
    )
    if response is None:
        return None
    defender_at, attacker_at = _coupled.absolute_lookups(scene, defender)
    if defender_at is None:
        return None

    runner = str(scene["runner_id"])
    priced = {
        option_id: cell
        for option_id, cell in response["cells"].items()
        if cell.get("legal") is not False
        and cell.get("q") is not None
        and option_id != runner
    }
    curves = {k: value_at_times(c.get("candidate_grid") or []) for k, c in priced.items()}
    pick, detail = rule_r9(
        state,
        attacking_team_id(scene),
        runner,
        str(scene["carrier_id"]),
        defender_id,
        defender_at,
        attacker_at,
        curves,
        float(scene["horizon_seconds"]),
    )
    names = {str(o["option_id"]): str(o["option_name"]) for o in defender["options"]}
    rows = {}
    for option_id, cell in priced.items():
        peak = max((q for _, q in curves.get(option_id, [])), default=0.0)
        rows[option_id] = {
            "name": names.get(option_id, option_id),
            "score": float(detail.get(option_id, 0.0)),
            "q": float(cell["q"]),
            "peak_q": peak,
            "delivery": float(cell.get("delivery") or 0.0),
            "goal": float(cell.get("goal") or 0.0),
            "access": float(cell.get("accessibility") or 0.0),
        }
    return pick, rows, str(defender["defender_name"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", nargs="+", required=True)
    parser.add_argument("--new", nargs="+", required=True)
    parser.add_argument("--pair", action="append", default=[], help="onset frame id")
    args = parser.parse_args()

    base_scenes = load(args.baseline)
    new_scenes = load(args.new)
    labels = _scorer.load_labels()

    for (match_id, frame_id, defender_id), answer in sorted(labels.items()):
        if args.pair and frame_id not in args.pair:
            continue
        base = base_scenes.get((match_id, frame_id))
        new = new_scenes.get((match_id, frame_id))
        if base is None or new is None:
            continue
        left = evaluate(base, defender_id)
        right = evaluate(new, defender_id)
        if left is None or right is None:
            continue
        base_pick, base_rows, defender_name = left
        new_pick, new_rows, _ = right

        print(f"\n{'=' * 78}")
        print(f"장면 {frame_id} · 수비수 {defender_name}")
        print(f"{'=' * 78}")
        everyone = sorted(
            set(base_rows) | set(new_rows),
            key=lambda k: -max(
                base_rows.get(k, {}).get("score", 0.0),
                new_rows.get(k, {}).get("score", 0.0),
            ),
        )
        label = base_rows.get(answer, new_rows.get(answer, {})).get("name", answer)
        print(f"정답: {label}")
        print(
            f"  하이브리드 선택: {base_rows.get(base_pick, {}).get('name', base_pick)}"
            f"  {'✓' if base_pick == answer else '✗'}"
        )
        print(
            f"  xpass360  선택: {new_rows.get(new_pick, {}).get('name', new_pick)}"
            f"  {'✓' if new_pick == answer else '✗'}"
        )
        print()
        print(
            f"{'선수':<22}{'R9(하이)':>10}{'R9(360)':>10}"
            f"{'Q하이':>8}{'Q360':>8}{'P하이':>8}{'P360':>8}{'G':>7}{'A':>7}"
        )
        for option_id in everyone:
            b = base_rows.get(option_id)
            n = new_rows.get(option_id)
            if b is None and n is None:
                continue
            src = b or n
            mark = ""
            if option_id == answer:
                mark = " ←정답"
            elif option_id in (base_pick, new_pick):
                mark = " ←선택"
            print(
                f"{src['name'][:20]:<22}"
                f"{(b or {}).get('score', 0.0):>10.4f}{(n or {}).get('score', 0.0):>10.4f}"
                f"{(b or {}).get('q', 0.0):>8.3f}{(n or {}).get('q', 0.0):>8.3f}"
                f"{(b or {}).get('delivery', 0.0):>8.3f}{(n or {}).get('delivery', 0.0):>8.3f}"
                f"{src.get('goal', 0.0):>7.3f}{src.get('access', 0.0):>7.3f}"
                f"{mark}"
            )


if __name__ == "__main__":
    main()
