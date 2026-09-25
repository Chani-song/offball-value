"""The dilemma as the defender's own minimax, with no marking term.

The reviewer rejected the previous score (2026-09-09):

  점수에서 (반응 수비수 후보인 A가 오프더볼 러너 X를 얼마나 바짝 따라붙을 수
  있나)가 들어가는데 이게 왜 들어가는 거야. ... A의 수비 궤적에 따라서
  min(X의 위협, Y의 위협)이 가장 작아지는 수비 방식이 A의 최적 수비 방식일
  거고, 이 때 그 값이 얼마인지. 그 값이 A의 딜레마 정도 아닐까?

He is right. Marking quality answers "is this the reacting defender", not
"how bad is his dilemma", and I put it in because it raised the AUC from
0.655 to 0.698 — a score-driven justification the project explicitly
forbids.

The principled quantity is his: A chooses his path, the attack then takes
whichever of X or Y is better, so A minimises the MAXIMUM and the dilemma is
that value.

    knee = min over A's paths of  max(Q_X, Q_Y)

WHY THE MARKING TERM WAS PROPPING IT UP. A defender with no influence over
either option leaves both threats unchanged whatever he does, so his knee is
high simply because X and Y are dangerous — not because he faces a choice.
The fix should be principled, not a marking patch: measure how much A can
move things at all, and ask whether the knee is high RELATIVE to that.

    reach_X = spread of Q_X over A's paths      (max - min)
    reach_Y = spread of Q_Y over A's paths
    floor   = max(min Q_X, min Q_Y)             best against one alone

Candidates scored here, all free of any marking quantity:

    knee                          his proposal, literally
    knee - floor                  how much the PAIR costs over one alone
    (knee - floor) / knee         the same, relative
    knee x min(reach_X, reach_Y)  knee, discounted when A is powerless
    knee restricted to defenders who can move both threats at all
"""

from __future__ import annotations

import argparse
import csv
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

REVIEWS = ROOT / "examples/research_audit/human_reviews/shot_context"
QC_FILES = [
    "settled_possession_onset_v0_1_reviews.csv",
    "shot_context_onset_v0_2_reviews.csv",
]
LABEL_FILES = [
    "derived_beneficiary_blind_labels_v1.csv",
    "derived_beneficiary_represent_round1fix.csv",
    "derived_beneficiary_round2_blind_labels.csv",
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


def load_pair_labels():
    out = {}
    for name in LABEL_FILES:
        path = REVIEWS / name
        if not path.exists():
            continue
        for row in csv.DictReader(path.open(encoding="utf-8-sig")):
            verdict = (row.get("defender_reacts") or "").strip().lower()
            if verdict not in ("yes", "no"):
                continue
            named = (row.get("beneficiary") or "").strip() not in ("", "none")
            out[(row["match_id"], row["onset_frame_id"], row["defender_id"])] = int(
                verdict == "yes" and named
            )
    return out


def auc(scores, labels):
    positives = [s for s, y in zip(scores, labels) if y == 1]
    negatives = [s for s, y in zip(scores, labels) if y == 0]
    if not positives or not negatives:
        return float("nan")
    return sum(
        1.0 if p > n else 0.5 if p == n else 0.0 for p in positives for n in negatives
    ) / (len(positives) * len(negatives))


def pair_measures(scene, defender):
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
    if len(points) < 3:
        return None

    knee = min(max(x, y) for x, y in points)
    floor_x = min(x for x, _ in points)
    floor_y = min(y for _, y in points)
    floor = max(floor_x, floor_y)
    reach_x = max(x for x, _ in points) - floor_x
    reach_y = max(y for _, y in points) - floor_y
    reach = min(reach_x, reach_y)
    commit_x = min(y for x, y in points if x <= floor_x + 1e-12)
    commit_y = min(x for x, y in points if y <= floor_y + 1e-12)

    return {
        "① 무릎값 (형님 제안)": knee,
        "② 무릎값 − 단독바닥": knee - floor,
        "③ (무릎값 − 바닥) / 무릎값": (knee - floor) / knee if knee > 1e-9 else 0.0,
        "④ 무릎값 × 영향력": knee * reach,
        "⑤ 영향력 (양쪽 최소 변동폭)": reach,
        "⑥ 두 커밋의 최솟값": min(commit_x, commit_y),
        "⑦ 무릎값 × √영향력": knee * (reach**0.5),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audit_dirs", nargs="+")
    options = parser.parse_args()

    qc = load_scene_qc()
    pair_labels = load_pair_labels()
    scenes = {}
    for directory in options.audit_dirs:
        for scene in json.loads(
            (Path(directory) / "local_game_payoff_audits.json").read_text()
        ):
            scenes[(str(scene["match_id"]), str(scene["onset_frame_id"]))] = scene

    pair_cols, pair_truth = {}, []
    scene_best = {}
    for key, scene in sorted(scenes.items()):
        rows = []
        for defender in scene["candidate_defenders"]:
            row = pair_measures(scene, defender)
            if row is None:
                continue
            rows.append(row)
            label = pair_labels.get((key[0], key[1], str(defender["defender_id"])))
            if label is not None:
                pair_truth.append(label)
                for name, value in row.items():
                    pair_cols.setdefault(name, []).append(value)
        if rows:
            scene_best[key] = {
                name: max(r[name] for r in rows) for name in rows[0]
            }

    print(f"쌍 {len(pair_truth)}개 (라벨 있음)\n")
    print(f"{'측정치':<26} {'AUC 쌍 딜레마':>13}")
    for name in sorted(pair_cols, key=lambda n: -auc(pair_cols[n], pair_truth)):
        print(f"{name:<26} {auc(pair_cols[name], pair_truth):>13.3f}")

    positive = [v for k, v in scene_best.items() if qc.get(k) in ("clear", "possible")]
    negative = [v for k, v in scene_best.items() if qc.get(k) in ("unclear", "none")]
    only_none = [v for k, v in scene_best.items() if qc.get(k) == "none"]
    if positive and negative:
        print(
            f"\n장면 수준 (후보 중 최댓값)   양성 {len(positive)} / "
            f"음성 {len(negative)}  (그중 none만 {len(only_none)})"
        )
        print(f"{'측정치':<26} {'vs unclear+none':>16} {'vs none만':>12}")
        names = list(positive[0])
        scored = []
        for name in names:
            wide = auc(
                [r[name] for r in positive] + [r[name] for r in negative],
                [1] * len(positive) + [0] * len(negative),
            )
            tight = auc(
                [r[name] for r in positive] + [r[name] for r in only_none],
                [1] * len(positive) + [0] * len(only_none),
            )
            scored.append((wide, tight, name))
        for wide, tight, name in sorted(scored, reverse=True):
            print(f"{name:<26} {wide:>16.3f} {tight:>12.3f}")


if __name__ == "__main__":
    main()
