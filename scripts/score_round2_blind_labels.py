"""Score the ROUND-2 blind labels against both pre-registration locks.

Written and frozen BEFORE any round-2 label exists (the reviewer is
labelling as this is committed), so the scoring procedure is itself part
of the pre-registration. Nothing here may change once the label CSV is
produced.

Competitors and their locks:
- Stage 1 (hash 55fd27adf6623edd, docs/preregistration_round2_v1_0.json):
  R0 nearest-man, R1 assignment rule, always-carrier — flat rows with
  *_pred_id per (match, scene, defender), all 90 scenes.
- Stage 2 (hash 15180553bf9caf6c, docs/preregistration_round2_stage2_v1_0.json):
  six Q-variants + shipped incumbent per surface, standard lock format,
  the 29 QC-passing scenes, with `scorable` false on tie-excluded games.

Pre-declared scoring rules (same as round 1):
- 1.0 for matching `beneficiary`, 0.5 for matching `beneficiary_alt`.
- Rows with defender_reacts=no/unsure or beneficiary in {none, empty} are
  whole-family failures, reported separately, scored for nobody.
- Rows with repeat_tag are test-retest only.
- PRIMARY RACE: all competitors compared on the COMMON game set = labelled
  rows that are stage-2-scorable (so every competitor answers the same
  games). Stage-1 rules additionally reported on the full labelled set as
  a secondary table (their predictions exist everywhere).
- Adoption rule: unique top scorer AND lead >= 2.0 AND >= 12 games scored
  on the primary set; otherwise adopt nothing.

Usage:
    python scripts/score_round2_blind_labels.py --labels HIS.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

STAGE1 = Path("docs/preregistration_round2_v1_0.json")
STAGE2 = Path("docs/preregistration_round2_stage2_v1_0.json")
AUDITS = (
    Path("data/processed/round2_pass_geometric_g/local_game_payoff_audits.json"),
    Path("data/processed/round2_pass_learned_g/local_game_payoff_audits.json"),
)


def _clean(value: str | None) -> str:
    return str(value or "").strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, required=True)
    args = parser.parse_args()

    stage1 = json.loads(STAGE1.read_text(encoding="utf-8"))
    stage2 = json.loads(STAGE2.read_text(encoding="utf-8"))
    rows = list(csv.DictReader(args.labels.open(encoding="utf-8-sig")))

    id_to_name: dict[str, str] = {}
    for audit in AUDITS:
        for scene in json.loads(audit.read_text(encoding="utf-8")):
            for defender in scene["candidate_defenders"]:
                for option in defender["options"]:
                    id_to_name[str(option["option_id"])] = str(
                        option.get("option_name") or option["option_id"]
                    )

    stage1_by_key = {
        (r["match_id"], str(r["onset_frame_id"]), r["defender_id"]): r
        for r in stage1["predictions"]
    }
    stage2_by_key: dict[tuple[str, str, str], dict[str, dict]] = defaultdict(dict)
    for surface, games in stage2["surfaces"].items():
        for key, record in games.items():
            match_id, scene, defender = key.split(":")
            stage2_by_key[(match_id, scene, defender)][surface] = record

    def points(pick: str, answer: str, alt: str) -> float:
        if pick == answer:
            return 1.0
        if alt and pick == alt:
            return 0.5
        return 0.0

    family_failures: list[str] = []
    repeats: dict[tuple[str, str, str], dict[str, dict]] = defaultdict(dict)
    primary: dict[str, float] = defaultdict(float)
    secondary_stage1: dict[str, float] = defaultdict(float)
    primary_n = 0
    secondary_n = 0
    unmatched: list[str] = []

    for row in rows:
        key = (row["match_id"], _clean(row["onset_frame_id"]), row["defender_id"])
        tag = _clean(row.get("repeat_tag"))
        repeats[key][tag or "first"] = row
        if tag:
            continue
        answer_id = _clean(row["beneficiary"])
        if row["defender_reacts"] != "yes" or answer_id in {"", "none"}:
            family_failures.append(
                f"  {key[1]}/{row['defender_name']}: reacts={row['defender_reacts']} "
                f"beneficiary={answer_id or '—'}"
            )
            continue
        alt_id = _clean(row.get("beneficiary_alt"))
        answer_name = id_to_name.get(answer_id, answer_id)
        alt_name = id_to_name.get(alt_id, alt_id) if alt_id else ""

        s1 = stage1_by_key.get(key)
        if s1 is None:
            unmatched.append(f"  {key} — no stage-1 prediction")
            continue
        secondary_n += 1
        for rule, field in (
            ("R0/nearest", "r0_pred_id"),
            ("R1/assignment", "r1_pred_id"),
            ("always-carrier", "always_carrier_pred_id"),
        ):
            secondary_stage1[rule] += points(str(s1[field]), answer_id, alt_id)

        s2 = stage2_by_key.get(key)
        if not s2:
            continue
        if not all(record.get("scorable", True) for record in s2.values()):
            continue
        primary_n += 1
        for rule, field in (
            ("R0/nearest", "r0_pred_id"),
            ("R1/assignment", "r1_pred_id"),
            ("always-carrier", "always_carrier_pred_id"),
        ):
            primary[rule] += points(str(s1[field]), answer_id, alt_id)
        for surface, record in s2.items():
            for variant, prediction in record["predictions"].items():
                primary[f"{surface}:{variant}"] += points(
                    str(prediction["pick"]), answer_name, alt_name
                )

    print(f"labels: {len(rows)} rows")
    print(f"family failures (none / no-dilemma / unsure): {len(family_failures)}")
    for line in family_failures:
        print(line)
    if unmatched:
        print("UNMATCHED (investigate before interpreting):")
        for line in unmatched:
            print(line)

    print("\n=== 반복 씬 test-retest ===")
    agree = total = 0
    for key, answers in repeats.items():
        if "first" in answers and len(answers) > 1:
            for tag, row in answers.items():
                if tag == "first":
                    continue
                total += 1
                same = _clean(answers["first"]["beneficiary"]) == _clean(
                    row["beneficiary"]
                )
                agree += same
                print(f"  {key[1]}/{row['defender_name']}: {'일치' if same else '불일치'}")
    if total:
        print(f"  test-retest: {agree}/{total}")

    print(f"\n=== PRIMARY RACE (common stage-2-scorable set, n={primary_n}) ===")
    ranked = sorted(primary.items(), key=lambda kv: (-kv[1], kv[0]))
    for rule, score in ranked:
        print(f"  {rule:<40} {score:.1f}")
    if len(ranked) >= 2:
        lead = ranked[0][1] - ranked[1][1]
        unique = ranked[0][1] > ranked[1][1]
        adopted = unique and lead >= 2.0 and primary_n >= 12
        print(
            f"  -> adoption: top={'unique' if unique else 'TIED'} lead={lead:.1f} "
            f"n={primary_n} => "
            f"{'ADOPT ' + ranked[0][0] if adopted else 'INCONCLUSIVE — adopt nothing'}"
        )

    print(f"\n=== stage-1 rules on ALL labelled games (secondary, n={secondary_n}) ===")
    for rule, score in sorted(secondary_stage1.items(), key=lambda kv: -kv[1]):
        print(f"  {rule:<40} {score:.1f}")


if __name__ == "__main__":
    main()
