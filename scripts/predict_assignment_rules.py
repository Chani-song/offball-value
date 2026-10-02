"""Deterministic beneficiary-assignment predictions (R0 / R1 / always-carrier).

Pre-registered predictor for round-2 counterfactual-assignment validation.
Given an onset-audit JSON (a list of scene payloads with ``frames``,
``runner_id``, ``carrier_id``, ``team_id`` / ``attacking_team_id`` and
``attacking_direction``), this CLI emits, for every (scene, candidate
defender) pair, the beneficiary predicted by three locked rules:

- ``always_carrier``: the carrier.
- ``R0`` (zero parameters): nearest non-runner attacker to the defender at
  the onset frame.
- ``R1`` (assignment rule): R0 plus a carrier-override and a guarded
  covered-exclusion (exact formulas below).

Candidate defenders per scene are the pipeline's structural selection:
``build_structural_local_game(scene, LocalGameStructureConfig(
candidate_defender_count=N))`` (default N=3), taken in structural order.

RULE SPEC (locked)
------------------
Common geometry, computable from the onset-audit payload alone:

- Onset frame f0 = the frame whose ``relative_time_s`` is nearest to 0.0
  (tie -> smaller ``|relative_time_s|``).
- Attackers = players in f0 with team == attacking team id; every other
  player row in f0 is treated as defending-team.
- Velocities (centred finite difference): f+ / f- = frames whose
  ``relative_time_s`` is nearest to +/- ``VEL_TARGET_S`` (ties -> smaller
  ``|t|``; on the 0.08 s grid this resolves to +/-0.16 s). For a player
  present in both: v = (p(f+) - p(f-)) / (t+ - t-); otherwise v = (0, 0).
- Candidate ranking for defender D: all attackers EXCLUDING the runner
  (the carrier IS a candidate), sorted by (distance to D rounded to 1e-9,
  then option id ascending). All ties everywhere break by
  lexicographically smallest option id -> fully deterministic; no
  randomness, no wall-clock dependence.

R0: the first candidate in that ranking.

R1, evaluated in this order:

(i) CARRIER-OVERRIDE: with d_cd = ||carrier - defender||, if
    0 < d_cd <= ``D_ADV_M`` and dot(v_carrier, u) >= ``V_ADV_MPS`` where
    u = (defender - carrier) / d_cd (the carrier's approach speed toward
    this defender), predict the carrier and stop.
(ii) COVERED-EXCLUSION (skip at most ONE candidate; never the carrier;
    never the defender's own tight man): with (d0, top) the
    nearest-ranked candidate, if top != carrier and d0 >= ``D_NEAR_M``
    and there EXISTS a defending-team player q != this defender with
    ||q - top|| <= ``R_COV_M`` and ||q - top|| < d0, predict the
    SECOND-ranked candidate. The existential condition is
    order-independent, hence deterministic.
(iii) Otherwise predict top (= R0's pick).

CONSTANT PROVENANCE (train fit, fully disclosed)
------------------------------------------------
All constants were fitted by exhaustive grid search on the 24 round-1
picks ONLY (21 named picks from
``derived_beneficiary_blind_labels_v1.csv`` -- repeat_tag empty,
defender_reacts=yes, beneficiary != none -- plus the 3 re-presentation
answers from ``derived_beneficiary_represent_round1fix.csv``), scored
against ``pass2_scenes_geometric_g/local_game_payoff_audits.json``
geometry. Round-2 labels do not exist; nothing from the round-2 audit
was used in fitting. Train scores (1.0 exact / 0.5 alt, out of 24):
R0 = 13.5, R1 = 19.0, always-carrier = 6.0. Sensitivity: one constant
at a time within +/-25% the R1 train score never drops below 17.0;
under JOINT perturbation of several constants the worst score in the
+/-25% box is 15.0/24 -- still above R0's 13.5 everywhere in the box.
Full provenance, per-pick table, rejected thin-window fits and known
misses: ``docs/preregistration_round2_v1_0.md``.

Usage:
    python scripts/predict_assignment_rules.py AUDIT_JSON \
        [--candidate-defender-count 3] [--output OUT_JSON]

Output: a JSON list (stdout, or ``--output``) with one row per
(scene, defender) in scene-file order then structural defender order.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Mapping, Sequence

_REPO_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

from offball_value.local_game_structure import (  # noqa: E402
    LocalGameStructureConfig,
    build_structural_local_game,
)

from offball_value.assignment_rule import (  # noqa: E402
    D_ADV_M,
    D_NEAR_M,
    R_COV_M,
    V_ADV_MPS,
    VEL_TARGET_S,
    attacking_team_id,
    carrier_advancing,
    onset_state,
    predict_scene,
    ranked_candidates,
    rule_r0,
    rule_r1,
)

__all__ = [
    "V_ADV_MPS", "D_ADV_M", "R_COV_M", "D_NEAR_M", "VEL_TARGET_S",
    "attacking_team_id", "onset_state", "ranked_candidates",
    "rule_r0", "carrier_advancing", "rule_r1", "predict_scene",
    "structural_defender_ids", "predict_audit", "main",
]


def structural_defender_ids(
    scene: Mapping[str, object], candidate_defender_count: int
) -> list[tuple[str, str]]:
    """(defender_id, defender_name) rows in the pipeline's structural order."""
    payload = build_structural_local_game(
        scene,
        LocalGameStructureConfig(candidate_defender_count=candidate_defender_count),
    )
    return [
        (str(row["defender_id"]), str(row["defender_name"]))
        for row in payload["candidate_defenders"]  # type: ignore[index]
    ]


def predict_audit(
    audit_path: str | Path, candidate_defender_count: int = 3
) -> list[dict[str, object]]:
    """Predictions for every (scene, structural defender) in an audit file."""
    with open(audit_path) as handle:
        scenes = json.load(handle)
    rows: list[dict[str, object]] = []
    for scene in scenes:
        defenders = structural_defender_ids(scene, candidate_defender_count)
        rows.extend(predict_scene(scene, [d[0] for d in defenders]))
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Deterministic R0 / R1 / always-carrier beneficiary predictions "
            "for an onset-audit JSON (constants locked on round-1 train data)."
        )
    )
    parser.add_argument("audit_json", help="onset-audit JSON (list of scenes)")
    parser.add_argument(
        "--candidate-defender-count",
        type=int,
        default=3,
        help="structural candidate defenders per scene (default 3)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="write the JSON rows here instead of stdout",
    )
    args = parser.parse_args(argv)

    rows = predict_audit(args.audit_json, args.candidate_defender_count)
    payload = json.dumps(rows, indent=1, sort_keys=True)
    if args.output:
        Path(args.output).write_text(payload + "\n")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
