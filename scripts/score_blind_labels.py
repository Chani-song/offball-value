"""Score the blind derived-beneficiary labels against the locked predictions.

Written and committed BEFORE any Pass-2 label exists, so the scoring
procedure itself is pre-registered: nothing here may be changed after the
label CSV is produced. Rules implemented exactly as declared in the locked
JSON (`adoption_rule` / `scoring` fields):

- 1.0 for matching ``beneficiary``, 0.5 for matching ``beneficiary_alt``.
- Rows with ``beneficiary = none`` or ``defender_reacts = no`` are reported
  separately as whole-family outcomes, never scored for any variant.
- Games whose locked record has ``scorable = false`` (tied follow response)
  are excluded.
- Scenes listed in ``--consistency-scene`` (the previously-labelled
  development scenes) are excluded from the race and reported as a
  label-consistency measurement against the old manifest pins.
- Rows with a non-empty ``repeat_tag`` are excluded from the race and
  paired with their originals to report test-retest agreement.
- Adoption: a variant is adopted only if it is the unique top scorer,
  leads the runner-up by >= 2.0 points, and >= 12 games were scored.

Usage:
    python scripts/score_blind_labels.py \
        --labels derived_beneficiary_blind_labels.csv \
        --preregistration docs/preregistration_derived_criterion_v1_1.json \
        --consistency-scene 53844 --consistency-scene 53833 ...
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

MANIFEST = Path("examples/research_audit/manifests/confirmed_core_scenes.json")


def _name_of(labels_row: dict, field: str) -> str:
    return str(labels_row.get(field) or "").strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--consistency-scene", action="append", default=[])
    parser.add_argument(
        "--audit-json",
        action="append",
        default=[],
        help=(
            "Audit payloads used to resolve option ids to the display names "
            "stored in the lock. Mechanical fix applied after labels arrived: "
            "the lock records picks by NAME while the blind CSV records "
            "beneficiary by ID; this adds the id<->name table and changes no "
            "scoring rule."
        ),
    )
    args = parser.parse_args()

    locked = json.loads(args.preregistration.read_text(encoding="utf-8"))
    rows = list(csv.DictReader(args.labels.open(encoding="utf-8-sig")))
    consistency_scenes = {str(scene) for scene in args.consistency_scene}

    id_to_name: dict[str, str] = {}
    for audit_path in args.audit_json:
        for scene in json.loads(Path(audit_path).read_text(encoding="utf-8")):
            for defender in scene["candidate_defenders"]:
                for option in defender["options"]:
                    id_to_name[str(option["option_id"])] = str(
                        option.get("option_name") or option["option_id"]
                    )

    # Old pins, for the consistency report only.
    pins: dict[tuple[str, str], str] = {}
    if MANIFEST.exists():
        for scene in json.loads(MANIFEST.read_text()):
            for defender_id, option_id in (
                scene.get("confirmed_derived_ids") or {}
            ).items():
                pins[(str(scene["onset_frame_id"]), str(defender_id))] = str(option_id)

    scores: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    counted: dict[str, int] = defaultdict(int)
    picks_by_game: dict[str, dict[tuple[str, str], tuple[str, float]]] = defaultdict(dict)
    family_failures: list[str] = []
    consistency: list[str] = []
    repeats: dict[tuple[str, str, str], dict[str, dict]] = defaultdict(dict)
    unmatched: list[str] = []

    # Option-name -> option-id resolution has to go through the locked file's
    # names, since the blind CSV records ids for beneficiary but names are
    # what the researcher saw. The blind screen stores option_id directly.
    for row in rows:
        scene = str(row["onset_frame_id"])
        defender = str(row["defender_id"])
        tag = str(row.get("repeat_tag") or "")
        game_key = None
        records = {}
        for surface, games in locked["surfaces"].items():
            for key, record in games.items():
                _, key_scene, key_defender = key.split(":")
                if key_scene == scene and key_defender == defender:
                    game_key = key
                    records[surface] = record
        repeats[(row["match_id"], scene, defender)][tag or "first"] = row
        if tag:
            continue
        if scene in consistency_scenes:
            pin = pins.get((scene, defender))
            answer = _name_of(row, "beneficiary")
            consistency.append(
                f"  {scene}/{row['defender_name']}: blind={answer or '—'} "
                f"old_pin={pin or '—'} reacts={row['defender_reacts']}"
            )
            continue
        if row["defender_reacts"] == "no" or _name_of(row, "beneficiary") in {"", "none"}:
            family_failures.append(
                f"  {scene}/{row['defender_name']}: reacts={row['defender_reacts']} "
                f"beneficiary={_name_of(row, 'beneficiary') or '—'}"
            )
            continue
        if game_key is None:
            unmatched.append(f"  {scene}/{defender} — no locked prediction")
            continue

        for surface, record in records.items():
            if not record.get("scorable", True):
                continue
            counted[surface] += 1
            answer = id_to_name.get(
                _name_of(row, "beneficiary"), _name_of(row, "beneficiary")
            )
            answer_alt = id_to_name.get(
                _name_of(row, "beneficiary_alt"), _name_of(row, "beneficiary_alt")
            )
            for variant, prediction in record["predictions"].items():
                pick = str(prediction["pick"])
                if pick == answer:
                    gained = 1.0
                elif answer_alt and pick == answer_alt:
                    gained = 0.5
                else:
                    gained = 0.0
                scores[surface][variant] += gained
                picks_by_game[surface][(scene, defender, variant)] = (pick, gained)

    print(f"labels: {len(rows)} rows")
    print(f"consistency-only rows: {len(consistency)}")
    print(f"family failures (none / no-dilemma): {len(family_failures)}")
    for line in family_failures:
        print(line)
    if unmatched:
        print("UNMATCHED rows (investigate before interpreting anything):")
        for line in unmatched:
            print(line)

    print("\n=== 일관성 (개발 씬, 경쟁 제외) ===")
    for line in consistency:
        print(line)

    print("\n=== 반복 씬 test-retest ===")
    agree = total = 0
    for key, answers in repeats.items():
        if "first" in answers and len(answers) > 1:
            for tag, row in answers.items():
                if tag == "first":
                    continue
                total += 1
                same = _name_of(answers["first"], "beneficiary") == _name_of(row, "beneficiary")
                agree += same
                print(f"  {key[1]}/{row['defender_name']}: {'일치' if same else '불일치'}")
    if total:
        print(f"  test-retest 일치율: {agree}/{total}")

    print("\n=== 기준 경쟁 ===")
    for surface, variant_scores in scores.items():
        n = counted[surface]
        ranked = sorted(variant_scores.items(), key=lambda kv: -kv[1])
        print(f"\n[{surface}] scored games: {n}")
        for variant, score in ranked:
            print(f"  {variant:<32} {score:.1f}")
        if len(ranked) >= 2:
            lead = ranked[0][1] - ranked[1][1]
            unique = ranked[0][1] > ranked[1][1]
            adopted = unique and lead >= 2.0 and n >= 12
            print(
                f"  -> adoption rule: top={'unique' if unique else 'TIED'}, "
                f"lead={lead:.1f}, n={n} => "
                f"{'ADOPT ' + ranked[0][0] if adopted else 'INCONCLUSIVE — adopt nothing'}"
            )


if __name__ == "__main__":
    main()
