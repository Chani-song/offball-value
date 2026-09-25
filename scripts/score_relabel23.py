"""Score every build against the re-labelled beneficiaries.

The first labelling screen drew players and the ball and nothing else. It did
not show the space the defender gives up as he chases, which is the very
quantity the beneficiary rule maximises, so a reviewer could only judge from
positions. The re-labelling screen draws that space instant by instant.

The two label sets therefore come from different information, not different
reviewers, and a change between them is an instrument upgrade rather than a
reversal. What this script exists to prevent is the other trap: re-judging
only the scenes where the model disagreed would let every correction run in
the model's favour by construction. All 23 defender games are re-judged, so
a correction can move either way, and this reports which way they moved.

Usage:
    python scripts/score_relabel23.py NEW_LABELS.csv [--builds a b c]
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from collections import Counter
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

DEFAULT_BUILDS = (
    ("투시 · 합집합없음", "round2_payoff_geometric"),
    ("투시 · 합집합", "vacated_union_round2"),
    ("예측 · 합집합", "kin_union"),
    ("예측 · 합집합없음", "kin_nounion"),
)


def read_labels(path: Path) -> dict[tuple[str, str, str], str]:
    labels = {}
    for row in csv.DictReader(path.open(encoding="utf-8-sig")):
        if row.get("repeat_tag"):
            continue
        if row.get("defender_reacts") != "yes":
            continue
        beneficiary = (row.get("beneficiary") or "").strip()
        if beneficiary in ("", "none"):
            continue
        labels[
            (row["match_id"], row["onset_frame_id"], row["defender_id"])
        ] = beneficiary
    return labels


def load_build(name: str) -> dict[tuple[str, str], dict]:
    games = {}
    for index in (0, 1, 2):
        path = ROOT / "data/processed" / f"{name}_chunk{index}" / "local_game_payoff_audits.json"
        if not path.exists():
            continue
        for game in json.loads(path.read_text()):
            games[(str(game["match_id"]), str(game["onset_frame_id"]))] = game
    return games


def pick(game, defender_id: str) -> str | None:
    """What the coupled rule answers for this defender in this build."""
    defender = next(
        (
            row
            for row in game["candidate_defenders"]
            if str(row["defender_id"]) == str(defender_id)
        ),
        None,
    )
    if defender is None:
        return None
    direct = next(
        (
            row
            for row in defender["responses"]
            if str(row["response_id"]) == str(defender.get("direct_best_response_id"))
        ),
        None,
    )
    if direct is None:
        return None
    defender_at, attacker_at = _coupled.absolute_lookups(game, defender)
    if defender_at is None:
        return None
    runner_id = str(game["runner_id"])
    curves = {
        option_id: value_at_times(cell.get("candidate_grid") or [])
        for option_id, cell in direct["cells"].items()
        if cell.get("legal") is not False
        and cell.get("q") is not None
        and option_id != runner_id
    }
    if not curves:
        return None
    beneficiary, _ = rule_r9(
        onset_state(game),
        attacking_team_id(game),
        runner_id,
        str(game["carrier_id"]),
        str(defender_id),
        defender_at,
        attacker_at,
        curves,
        float(game["horizon_seconds"]),
    )
    return str(beneficiary) or None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("labels", type=Path)
    parser.add_argument(
        "--old-labels",
        action="store_true",
        default=True,
        help="Also report how the labels themselves moved (default on).",
    )
    options = parser.parse_args()

    new = read_labels(options.labels)
    old = _coupled._scorer.load_labels()
    print(f"새 라벨 {len(new)}개 · 이전 라벨 {len(old)}개\n")

    builds = {name: load_build(directory) for name, directory in DEFAULT_BUILDS}
    picks: dict[str, dict] = {}
    for name, games in builds.items():
        answers = {}
        for (match_id, frame_id, defender_id) in new:
            game = games.get((str(match_id), str(frame_id)))
            if game is None:
                continue
            answer = pick(game, defender_id)
            if answer:
                answers[(match_id, frame_id, defender_id)] = answer
        picks[name] = answers

    print("① 새 라벨로 채점한 네 빌드\n")
    print(f"{'빌드':<22}{'정확도':>10}{'  (이전 라벨 기준)':>18}")
    for name, _ in DEFAULT_BUILDS:
        answers = picks[name]
        hit_new = sum(1 for key, value in answers.items() if value == new[key])
        shared = [k for k in answers if k in old]
        hit_old = sum(1 for key in shared if answers[key] == old[key])
        print(
            f"{name:<22}{hit_new:>4}/{len(answers):<5}"
            f"{hit_old:>10}/{len(shared)}"
        )

    print("\n② 라벨 자체가 어떻게 움직였나 — 편향 점검\n")
    both = [key for key in new if key in old]
    changed = [key for key in both if new[key] != old[key]]
    print(f"   두 라벨 집합에 모두 있는 조합 {len(both)}개 중 바뀐 것 {len(changed)}개")
    if changed:
        # Did the corrections favour the model, or cut both ways?
        toward = away = neither = 0
        reference = picks["예측 · 합집합"]
        for key in changed:
            model = reference.get(key)
            if model is None:
                neither += 1
            elif new[key] == model and old[key] != model:
                toward += 1
            elif old[key] == model and new[key] != model:
                away += 1
            else:
                neither += 1
        print(f"     모델 쪽으로 이동 {toward}   모델 반대로 이동 {away}   무관 {neither}")
        if away == 0 and toward > 0:
            print("     ⚠ 모든 정정이 모델에 유리한 방향입니다 — 편향을 의심해야 합니다.")
        elif toward and away:
            print("     ✓ 양방향으로 움직였습니다 — 선택 편향의 징후가 없습니다.")
        print("\n   바뀐 조합:")
        names = {}
        for game in builds["예측 · 합집합"].values():
            for frame in game["background_frames"][:1]:
                for player in frame["players"]:
                    names[str(player[0])] = player[4]
        for key in changed:
            print(
                f"     {key[1]:<8} {names.get(old[key], old[key][-6:]):<18}"
                f" → {names.get(new[key], new[key][-6:])}"
            )

    print("\n③ 분포")
    print(f"   새 라벨 reacts=yes 조합 {len(new)}개")
    counts = Counter(
        "캐리어" if key[0] and new[key] == str(
            next(
                (
                    game["carrier_id"]
                    for game in builds["예측 · 합집합"].values()
                    if str(game["onset_frame_id"]) == key[1]
                ),
                "",
            )
        ) else "다른 공격수"
        for key in new
    )
    print(f"   {dict(counts)}")


if __name__ == "__main__":
    main()
