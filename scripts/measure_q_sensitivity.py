"""How much does Q actually move when the defender moves?

The structural gap (knee minus the single-option floor) turned out to be
EXACTLY ZERO in more than half of defender-games: the defender's best answer
to the pair is as good as his best answer to the harder man alone, so by the
model's own reckoning there is no two-sided structure to find.

Two explanations were left open. Either the trajectory lattice is permissive
enough that a compromise path nearly always exists, or Q barely responds to
where the defender goes — in which case every path scores about the same and
no trade-off can appear.

This measures the second directly. For each (defender, option) the audits
store Q under every priced response, so the question is simply how wide that
spread is against the level of Q itself:

    span   = max Q - min Q over the defender's responses
    level  = median Q over those responses
    ratio  = span / level

A ratio near zero means the defender is nearly irrelevant to that option and
no dilemma can be expressed. It also reframes yesterday's finding that the
cheap screen proxy has a within-option rank correlation of only 0.153 with
real Q: if the true spread is small, that correlation is noise-limited by
construction rather than a failure of the proxy.

Reported separately for the runner, the R9 beneficiary, and all options, and
split by whether the reviewer called the scene a dilemma.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import statistics
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


def beneficiary_of(scene, defender):
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
    pick, _ = rule_r9(
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
    return pick or None


def spreads(defender, option_id):
    values = []
    for response in defender["responses"]:
        cell = (response.get("cells") or {}).get(option_id) or {}
        if cell.get("q") is None or cell.get("legal") is False:
            continue
        values.append(float(cell["q"]))
    if len(values) < 5:
        return None
    values = np.array(values)
    level = float(np.median(values))
    return {
        "span": float(values.max() - values.min()),
        "level": level,
        "ratio": float((values.max() - values.min()) / level) if level > 1e-9 else 0.0,
        "iqr_ratio": (
            float((np.quantile(values, 0.75) - np.quantile(values, 0.25)) / level)
            if level > 1e-9
            else 0.0
        ),
        "n": len(values),
    }


def report(title, rows):
    if not rows:
        print(f"{title}: 표본 없음")
        return
    ratios = sorted(r["ratio"] for r in rows)
    iqrs = sorted(r["iqr_ratio"] for r in rows)
    spans = sorted(r["span"] for r in rows)
    levels = sorted(r["level"] for r in rows)
    print(
        f"{title:<28} n={len(rows):<5} "
        f"폭/수준 중앙 {statistics.median(ratios):.2f}  "
        f"사분위폭/수준 {statistics.median(iqrs):.2f}  "
        f"폭 {statistics.median(spans):.4f}  수준 {statistics.median(levels):.4f}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audit_dirs", nargs="+")
    options = parser.parse_args()

    qc = load_scene_qc()
    scenes = {}
    for directory in options.audit_dirs:
        for scene in json.loads(
            (Path(directory) / "local_game_payoff_audits.json").read_text()
        ):
            scenes[(str(scene["match_id"]), str(scene["onset_frame_id"]))] = scene

    runner_rows, beneficiary_rows, all_rows = [], [], []
    dilemma_rows, quiet_rows = [], []
    for key, scene in sorted(scenes.items()):
        verdict = qc.get(key)
        runner = str(scene["runner_id"])
        for defender in scene["candidate_defenders"]:
            beneficiary = beneficiary_of(scene, defender)
            for option_id in (defender["responses"][0].get("cells") or {}):
                row = spreads(defender, option_id)
                if row is None:
                    continue
                all_rows.append(row)
                if option_id == runner:
                    runner_rows.append(row)
                    if verdict in ("clear", "possible"):
                        dilemma_rows.append(row)
                    elif verdict in ("unclear", "none"):
                        quiet_rows.append(row)
                if beneficiary and option_id == beneficiary:
                    beneficiary_rows.append(row)

    print("수비수가 움직일 때 Q가 얼마나 변하나")
    print("(폭 = 응답들 사이 Q 최대−최소, 수준 = Q 중앙값)\n")
    report("모든 옵션", all_rows)
    report("러너", runner_rows)
    report("R9 수혜자", beneficiary_rows)
    print()
    report("러너 — 딜레마 장면", dilemma_rows)
    report("러너 — 딜레마 아닌 장면", quiet_rows)

    print("\n폭/수준 비율의 분포 (모든 옵션)")
    ratios = np.array([r["ratio"] for r in all_rows])
    for threshold in (0.05, 0.1, 0.2, 0.5, 1.0):
        print(f"  비율 < {threshold:<4} : {(ratios < threshold).mean():.0%}")
    print(
        f"\n  해석: 비율 0.1 이면 수비수가 무엇을 하든 Q가 10%밖에 안 움직인다는 뜻."
    )


if __name__ == "__main__":
    main()
