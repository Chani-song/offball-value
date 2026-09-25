"""What the defender actually gives up by committing to one man.

This replaces an ad-hoc measurement that was wrong. I had identified "the
path that covers the runner" as the one whose ENDPOINT lands nearest him,
but the endpoint is at the 2.8 s horizon while the pass may arrive at 1.0 s,
so endpoint proximity is not coverage. That mistake produced a difference of
0.001 between committing to one man or the other, and the false conclusion
that the defender is powerless.

The correct identification is the path that actually minimises that option's
Q, which is what the audits let us read directly:

    commit to runner  = the response minimising Q_runner
    commit to other   = the response minimising Q_beneficiary

and the trade-off is what each commitment concedes on the other side:

    runner_cost = Q_runner(commit to other) - Q_runner(commit to runner)
    other_cost  = Q_other(commit to runner) - Q_other(commit to other)

Both large is the dilemma: whichever man he takes, the other gains.

Also reported, because they are different claims that were being conflated:

  - the knee, min over responses of max(Q_runner, Q_other): what he gets if
    he plays the best compromise rather than committing;
  - the gap to the single-target floor, which is zero whenever a compromise
    holds both at the level of the harder man alone - that is a real "no
    dilemma here" verdict, not a measurement artefact;
  - the correlation of Q_runner and Q_other over ALL responses, which is
    near zero because most paths do nothing useful and drown the extremes.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from offball_value.assignment_rule import attacking_team_id, onset_state  # noqa: E402
from offball_value.coupled_beneficiary import rule_r9, value_at_times  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "score_coupled_beneficiary", ROOT / "scripts" / "score_coupled_beneficiary.py"
)
_coupled = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_coupled)

REVIEWS = ROOT / "examples/research_audit/human_reviews/shot_context"
QC_FILES = [
    "settled_possession_onset_v0_1_reviews.csv",
    "shot_context_onset_v0_2_reviews.csv",
]


def load_scene_qc():
    out = {}
    for name in QC_FILES:
        path = REVIEWS / name
        if not path.exists():
            continue
        for row in csv.DictReader(path.open(encoding="utf-8-sig")):
            out[(row["match_id"], row["onset_frame_id"])] = (
                row.get("interaction_review") or ""
            ).strip()
    return out


def auc(scores, labels):
    positives = [s for s, y in zip(scores, labels) if y == 1]
    negatives = [s for s, y in zip(scores, labels) if y == 0]
    if not positives or not negatives:
        return float("nan")
    return sum(
        1.0 if p > n else 0.5 if p == n else 0.0 for p in positives for n in negatives
    ) / (len(positives) * len(negatives))


def defender_row(scene, defender):
    state = onset_state(scene)
    defender_id = str(defender["defender_id"])
    if defender_id not in state:
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
    defender_at, attacker_at = _coupled.absolute_lookups(scene, defender)
    if defender_at is None:
        return None
    runner = str(scene["runner_id"])
    curves = {
        option_id: value_at_times(cell.get("candidate_grid") or [])
        for option_id, cell in direct["cells"].items()
        if cell.get("legal") is not False
        and cell.get("q") is not None
        and option_id != runner
    }
    if not curves:
        return None
    beneficiary, _ = rule_r9(
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
    if not beneficiary:
        return None

    points = []
    for response in defender["responses"]:
        cells = response.get("cells") or {}
        a, b = cells.get(runner) or {}, cells.get(beneficiary) or {}
        if a.get("q") is None or b.get("q") is None:
            continue
        if a.get("legal") is False or b.get("legal") is False:
            continue
        points.append((float(a["q"]), float(b["q"])))
    if len(points) < 8:
        return None

    runner_values = np.array([p[0] for p in points])
    other_values = np.array([p[1] for p in points])
    at_runner = min(points, key=lambda p: p[0])
    at_other = min(points, key=lambda p: p[1])
    floor_runner, floor_other = at_runner[0], at_other[1]
    knee = min(max(a, b) for a, b in points)

    runner_cost = at_other[0] - floor_runner
    other_cost = at_runner[1] - floor_other
    correlation = (
        float(np.corrcoef(runner_values, other_values)[0, 1])
        if runner_values.std() > 1e-9 and other_values.std() > 1e-9
        else float("nan")
    )
    return {
        # The four cells, stored directly rather than reconstructed.
        "러너커밋 → Q러너": at_runner[0],
        "러너커밋 → Q수혜자": at_runner[1],
        "수혜자커밋 → Q러너": at_other[0],
        "수혜자커밋 → Q수혜자": at_other[1],
        # What each commitment concedes, in absolute Q and relative to the
        # level he could have held that man at.
        "러너 포기 비용": runner_cost,
        "수혜자 포기 비용": other_cost,
        "러너 포기 비용 (상대)": runner_cost / floor_runner if floor_runner > 1e-9 else 0.0,
        "수혜자 포기 비용 (상대)": other_cost / floor_other if floor_other > 1e-9 else 0.0,
        # The dilemma proper: BOTH commitments are expensive.
        "두 비용의 최솟값": min(runner_cost, other_cost),
        "두 상대비용의 최솟값": min(
            runner_cost / floor_runner if floor_runner > 1e-9 else 0.0,
            other_cost / floor_other if floor_other > 1e-9 else 0.0,
        ),
        # The compromise story.
        "무릎값": knee,
        "간극 (무릎 − 단독바닥)": knee - max(floor_runner, floor_other),
        "러너 단독바닥": floor_runner,
        "수혜자 단독바닥": floor_other,
        "전체 경로 상관": correlation,
        "응답 수": float(len(points)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audit_dirs", nargs="+")
    options = parser.parse_args()

    qc = load_scene_qc()
    settled = {
        (row["match_id"], row["onset_frame_id"])
        for row in csv.DictReader(
            (REVIEWS / "settled_possession_onset_v0_1_reviews.csv").open(
                encoding="utf-8-sig"
            )
        )
    }
    scenes = {}
    for directory in options.audit_dirs:
        for scene in json.loads(
            (Path(directory) / "local_game_payoff_audits.json").read_text()
        ):
            scenes[(str(scene["match_id"]), str(scene["onset_frame_id"]))] = scene

    per_defender, scene_best = [], {}
    for key, scene in sorted(scenes.items()):
        rows = [
            row
            for row in (
                defender_row(scene, defender)
                for defender in scene["candidate_defenders"]
            )
            if row is not None
        ]
        if not rows:
            continue
        verdict = qc.get(key)
        for row in rows:
            per_defender.append((verdict, row))
        scene_best[key] = (
            verdict,
            key in settled,
            {name: max(r[name] for r in rows) for name in rows[0]},
        )

    print(f"수비수-게임 {len(per_defender)}개, 장면 {len(scene_best)}개\n")

    print("① 커밋 4칸표 (중앙값)")
    take = lambda name: np.array([r[name] for _, r in per_defender])
    print(f"{'':<26} {'Q 러너':>10} {'Q 수혜자':>10}")
    print(
        f"{'러너를 최대한 막으면':<26} "
        f"{np.median(take('러너커밋 → Q러너')):>10.4f} "
        f"{np.median(take('러너커밋 → Q수혜자')):>10.4f}"
    )
    print(
        f"{'수혜자를 최대한 막으면':<26} "
        f"{np.median(take('수혜자커밋 → Q러너')):>10.4f} "
        f"{np.median(take('수혜자커밋 → Q수혜자')):>10.4f}"
    )
    print()
    for name in ("러너 포기 비용", "수혜자 포기 비용"):
        values = take(name)
        relative = take(f"{name} (상대)")
        print(
            f"   {name:<16} 중앙 +{np.median(values):.4f}  "
            f"수준 대비 {np.median(relative):.0%}  양수 {np.mean(values > 0):.0%}"
        )
    both = (take("러너 포기 비용 (상대)") > 0.1) & (take("수혜자 포기 비용 (상대)") > 0.1)
    print(f"   둘 다 10% 넘게 비싼 수비수: {both.mean():.0%}")
    print(f"   간극 > 0 인 수비수:        {np.mean(take('간극 (무릎 − 단독바닥)') > 1e-9):.0%}")
    correlations = take("전체 경로 상관")
    correlations = correlations[~np.isnan(correlations)]
    print(f"   전체 경로 상관 중앙:        {np.median(correlations):+.3f}  (극단이 묻힌 값)")

    print("\n② 형님 판정별 (수비수 단위 중앙값)")
    print(f"{'판정':<10} {'n':>5} {'러너 포기':>10} {'수혜자 포기':>12} {'두 비용 최소':>13} {'간극>0':>8}")
    for verdict in ("clear", "possible", "unclear", "none"):
        sub = [r for v, r in per_defender if v == verdict]
        if not sub:
            continue
        print(
            f"{verdict:<10} {len(sub):>5} "
            f"{np.median([r['러너 포기 비용 (상대)'] for r in sub]):>9.0%} "
            f"{np.median([r['수혜자 포기 비용 (상대)'] for r in sub]):>11.0%} "
            f"{np.median([r['두 상대비용의 최솟값'] for r in sub]):>12.0%} "
            f"{np.mean([r['간극 (무릎 − 단독바닥)'] > 1e-9 for r in sub]):>7.0%}"
        )

    print("\n③ 장면 수준 AUC")
    names = [n for n in next(iter(scene_best.values()))[2] if n != "응답 수"]
    # The verdict table shows `clear` strong on every trade-off column and
    # `possible` the weakest of all four, below `none`. Pooling them as one
    # positive class mixes a signal with an anti-signal, so the contrasts are
    # reported separately rather than only in the pooled form.
    contrasts = (
        (("clear", "possible"), ("unclear", "none"), "clear+possible vs unclear+none"),
        (("clear",), ("none",), "clear vs none"),
        (("clear",), ("unclear", "none"), "clear vs unclear+none"),
        (("possible",), ("none",), "possible vs none"),
    )
    for pool, label in ((None, "전체"), (True, "정착만")):
      for positives, negatives, contrast in contrasts:
        rows = [
            (values, 1 if verdict in positives else 0)
            for verdict, is_settled, values in scene_best.values()
            if verdict in positives + negatives
            and (pool is None or is_settled)
        ]
        if len(rows) < 20:
            continue
        truth = [t for _, t in rows]
        scored = sorted(
            ((auc([v[n] for v, _ in rows], truth), n) for n in names), reverse=True
        )
        print(f"\n   [{label}] {contrast}  n={len(rows)} 양성 {sum(truth)}")
        for value, name in scored[:5]:
            print(f"     {name:<24} {value:>6.3f}")


if __name__ == "__main__":
    main()
