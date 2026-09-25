#!/usr/bin/env python3
"""Compare the two delivery-model arms over the full scene pool.

The 42-pair evaluation could not separate hybrid (34) from xpass360 (33): one
answer is noise. This does not manufacture a winner where the labels cannot
support one. It reports two different things and keeps them apart:

  1. AGREEMENT -- how often the two arms name the same beneficiary. This needs
     no labels and covers every scene. If agreement is high the choice barely
     matters; if it is low the choice matters and the labels have to settle it.
  2. ACCURACY  -- on the subset that carries human labels only. Human labels
     are the evaluation and never enter the model.

Usage:
    python scripts/compare_full_arms.py \
        --arm hybrid /work/hdd/bbmr/kseo1/offball-out/full_hybrid \
        --arm xpass360 /work/hdd/bbmr/kseo1/offball-out/full_xpass360
"""

from __future__ import annotations

import argparse
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


def load_arm(root: Path) -> dict[tuple[str, str], dict]:
    games: dict[tuple[str, str], dict] = {}
    for path in sorted(root.glob("scene_*/local_game_payoff_audits.json")):
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError:
            print(f"  ! 깨진 JSON: {path}")
            continue
        for game in payload:
            games[(str(game["match_id"]), str(game["onset_frame_id"]))] = game
    return games


def pick(game: dict, defender_id: str) -> str | None:
    """The coupled R9 answer for this defender in this build."""
    defender = next(
        (r for r in game["candidate_defenders"] if str(r["defender_id"]) == str(defender_id)),
        None,
    )
    if defender is None:
        return None
    direct = next(
        (
            r
            for r in defender["responses"]
            if str(r["response_id"]) == str(defender.get("direct_best_response_id"))
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--arm", nargs=2, action="append", metavar=("NAME", "DIR"), required=True)
    p.add_argument("--exclude-frame", action="append", default=["57121"],
                   help="Onset frames excluded from scoring (default: 57121, label contaminated).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    excluded = {str(f) for f in args.exclude_frame}
    arms = {name: load_arm(Path(d)) for name, d in args.arm}
    for name, games in arms.items():
        print(f"{name:12} 장면 {len(games)}개")
    names = list(arms)
    if len(names) != 2:
        raise SystemExit("두 arm이 필요합니다")
    a, b = names

    shared = sorted(set(arms[a]) & set(arms[b]))
    print(f"\n두 arm 공통 장면: {len(shared)}개 (제외 프레임 {sorted(excluded)})")

    labels = _coupled._scorer.load_labels()
    print(f"사람 라벨 {len(labels)}개 보유\n")

    agree = disagree = 0
    only_a = only_b = neither = 0
    hits = Counter()
    labelled_pairs = 0
    disagreements = []

    for key in shared:
        if key[1] in excluded:
            continue
        ga, gb = arms[a][key], arms[b][key]
        defenders = {str(r["defender_id"]) for r in ga["candidate_defenders"]}
        defenders &= {str(r["defender_id"]) for r in gb["candidate_defenders"]}
        for did in sorted(defenders):
            pa, pb = pick(ga, did), pick(gb, did)
            if pa is None and pb is None:
                neither += 1
                continue
            if pa is None:
                only_b += 1
                continue
            if pb is None:
                only_a += 1
                continue
            if pa == pb:
                agree += 1
            else:
                disagree += 1
                disagreements.append((key, did, pa, pb))
            lab = labels.get((key[0], key[1], did))
            if lab:
                labelled_pairs += 1
                if pa == lab:
                    hits[a] += 1
                if pb == lab:
                    hits[b] += 1

    total = agree + disagree
    print("① 일치도 (라벨 불필요, 전 장면)")
    print(f"   두 arm 모두 답을 낸 수비수 게임: {total}")
    print(f"   같은 수혜자: {agree} ({agree / total:.1%})" if total else "   없음")
    print(f"   다른 수혜자: {disagree} ({disagree / total:.1%})" if total else "")
    print(f"   한쪽만 답함: {a} {only_a} / {b} {only_b}   둘 다 못함: {neither}")

    print("\n② 정확도 (사람 라벨 있는 부분집합만)")
    print(f"   라벨 붙은 수비수 게임: {labelled_pairs}")
    if labelled_pairs:
        for name in names:
            print(f"   {name:12} {hits[name]:>3}/{labelled_pairs}  ({hits[name] / labelled_pairs:.1%})")
        diff = hits[a] - hits[b]
        print(f"   차이: {diff:+d}")
        if abs(diff) <= 2:
            print("   → 이 표본에서 두 arm을 구분할 수 없습니다.")
    else:
        print("   라벨이 붙은 게임이 없습니다 (신규 장면만 빌드된 경우).")

    if disagreements:
        print(f"\n③ 불일치 예시 (앞 10개 / 총 {len(disagreements)})")
        for (mid, frame), did, pa, pb in disagreements[:10]:
            lab = labels.get((mid, frame, did), "")
            tag = f"  라벨={lab}" if lab else ""
            print(f"   {mid} f{frame} 수비수 {did}: {a}={pa}  {b}={pb}{tag}")


if __name__ == "__main__":
    main()
