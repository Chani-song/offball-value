"""Assemble the Pass-2 labelling package from the gate-passing scenes.

Inputs: the Pass-1 review CSV (already frozen), the per-surface audit
artifacts for every gate-passing scene, and the repeat-scene choice.
Outputs, in order:

1. merged per-surface audit payloads over the 13 new passing scenes + the
   6 previously-labelled development scenes (race-excluded, consistency
   measurement only);
2. the pre-registration lock v1.1 over exactly those scenes;
3. the blind labelling screen over all 19, with three repeat scenes
   appended for the test-retest measurement.

Run only AFTER Pass 1 is frozen and BEFORE any Pass-2 label exists.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from offball_value.blind_derived_review import render_blind_derived_review

PASS1 = Path(
    "examples/research_audit/human_reviews/shot_context/shot_context_onset_v0_2_reviews.csv"
)
GATE = {"clear", "possible"}
# Development scenes: previously pinned, excluded from the criterion race,
# relabelled blind purely to measure label consistency.
CONSISTENCY_SCENES = {"53844", "53833", "144747", "131237", "29751", "61841"}
# Repeats for the test-retest noise floor: three new scenes, re-shown at the
# end of the session.
REPEAT_SCENES = {"49266", "135654", "61802"}

SOURCES = {
    "geometric": [
        "data/processed/candidate_scenes_all27_geometric_g",
        "data/processed/candidate_scenes_pass1extra_geometric_g",
        "data/processed/local_game_v0_3_6_carry_access",
        "data/processed/scene_144747_geometric_g",
    ],
    "learned": [
        "data/processed/candidate_scenes_all27_learned_g",
        "data/processed/candidate_scenes_pass1extra_learned_g",
        "data/processed/local_game_v0_3_8_data_g",
        "data/processed/scene_144747_learned_g",
    ],
}


def main() -> None:
    passing = {
        str(row["onset_frame_id"])
        for row in csv.DictReader(PASS1.open(encoding="utf-8-sig"))
        if row["onset_review"] == "correct"
        and row["possession_review"] == "settled"
        and row["interaction_review"] in GATE
    }
    print(f"gate-passing scenes: {len(passing)}")

    merged: dict[str, list[dict]] = {}
    for surface, sources in SOURCES.items():
        seen: set[str] = set()
        scenes: list[dict] = []
        for source in sources:
            path = Path(source) / "local_game_payoff_audits.json"
            for scene in json.loads(path.read_text(encoding="utf-8")):
                sid = str(scene["onset_frame_id"])
                if sid in passing and sid not in seen:
                    seen.add(sid)
                    scenes.append(scene)
        missing = passing - seen
        if missing:
            raise SystemExit(f"{surface}: missing analysed scenes {sorted(missing)}")
        out = Path(f"data/processed/pass2_scenes_{surface}_g")
        out.mkdir(parents=True, exist_ok=True)
        (out / "local_game_payoff_audits.json").write_text(json.dumps(scenes))
        merged[surface] = scenes
        games = sum(len(scene["candidate_defenders"]) for scene in scenes)
        print(f"  {surface}: {len(scenes)} scenes, {games} defender-games -> {out}")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/preregister_derived_predictions.py",
            "--audit-dir", "data/processed/pass2_scenes_geometric_g:geometric",
            "--audit-dir", "data/processed/pass2_scenes_learned_g:learned",
            "--out", "docs/preregistration_derived_criterion_v1_1.md",
        ],
        capture_output=True,
        text=True,
    )
    print(result.stdout.strip())
    if result.returncode != 0:
        raise SystemExit(result.stderr)

    # Blind screen: geometric payloads (display only), sorted by match/frame,
    # with the repeat scenes appended under a tag.
    display = sorted(
        merged["geometric"],
        key=lambda scene: (str(scene["match_id"]), int(scene["onset_frame_id"])),
    )
    tail = [
        dict(scene, repeat_tag="R")
        for scene in display
        if str(scene["onset_frame_id"]) in REPEAT_SCENES
    ]
    html = render_blind_derived_review(display + tail)
    out = Path("data/processed/blind_derived_review/index.html")
    out.write_text(html, encoding="utf-8")
    rows = sum(len(scene["candidate_defenders"]) for scene in display + tail)
    print(
        f"blind screen: {len(display)} scenes + {len(tail)} repeats, "
        f"{rows} label rows -> {out}"
    )
    consistency_present = sorted(
        str(scene["onset_frame_id"])
        for scene in display
        if str(scene["onset_frame_id"]) in CONSISTENCY_SCENES
    )
    print(f"consistency (race-excluded) scenes included: {consistency_present}")


if __name__ == "__main__":
    main()
