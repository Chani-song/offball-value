"""Does the dilemma appear once the max over release times is removed?

Q is the maximum over each cell's candidate grid — median 17 rows spanning
release times, continuation types and leads. A defender can deny one pass
but not seventeen, so the maximum barely moves when he does: its
interquartile range across his own responses is 0.00 of the level. With the
defender unable to change Q, "covering one man" and "covering two" cost the
same and the structural gap is exactly zero in over half of defender-games.

Fixing the release time triples sensitivity, and sensitivity climbs with
release because a later pass gives the defender time to interfere. So this
re-scores the dilemma at each release separately:

    Q_r(option, response) = max over grid rows with release == r
    knee_r = min over responses of max(Q_r(runner), Q_r(beneficiary))

Prediction sealed beforehand in docs/release_fixed_dilemma_sealed.md: if
insensitivity is the bottleneck, AUC should RISE with release while the
danger baseline stays flat. No single release may be picked afterwards — the
claim is about the shape of the curve.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from collections import defaultdict
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


def release_values(response, option_id):
    """Best Q per release time for one (response, option), from the grid."""
    cell = (response.get("cells") or {}).get(option_id) or {}
    if cell.get("legal") is False:
        return {}
    best: dict[float, float] = {}
    for row in cell.get("candidate_grid") or ():
        key = round(float(row["release"]), 2)
        value = float(row["q"])
        if value > best.get(key, -1.0):
            best[key] = value
    return best


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audit_dirs", nargs="+")
    parser.add_argument("--pool", choices=["all", "settled", "shot"], default="all")
    options = parser.parse_args()

    qc = load_scene_qc()
    settled = set()
    for row in csv.DictReader(
        (REVIEWS / "settled_possession_onset_v0_1_reviews.csv").open(
            encoding="utf-8-sig"
        )
    ):
        settled.add((row["match_id"], row["onset_frame_id"]))

    scenes = {}
    for directory in options.audit_dirs:
        for scene in json.loads(
            (Path(directory) / "local_game_payoff_audits.json").read_text()
        ):
            scenes[(str(scene["match_id"]), str(scene["onset_frame_id"]))] = scene

    # per release: scene -> best knee over defenders, and the danger control
    knees: dict[float, list[float]] = defaultdict(list)
    dangers: dict[float, list[float]] = defaultdict(list)
    truths: dict[float, list[int]] = defaultdict(list)

    for key, scene in sorted(scenes.items()):
        verdict = qc.get(key)
        if verdict not in ("clear", "possible", "unclear", "none"):
            continue
        if options.pool == "settled" and key not in settled:
            continue
        if options.pool == "shot" and key in settled:
            continue
        runner = str(scene["runner_id"])
        per_release_knee: dict[float, float] = {}
        per_release_all: dict[float, list[float]] = defaultdict(list)
        for defender in scene["candidate_defenders"]:
            beneficiary = beneficiary_of(scene, defender)
            if not beneficiary:
                continue
            pairs: dict[float, list[tuple[float, float]]] = defaultdict(list)
            for response in defender["responses"]:
                a = release_values(response, runner)
                b = release_values(response, beneficiary)
                for release in set(a) & set(b):
                    pairs[release].append((a[release], b[release]))
                for option_id in response.get("cells") or {}:
                    for release, value in release_values(response, option_id).items():
                        per_release_all[release].append(value)
            for release, points in pairs.items():
                if len(points) < 5:
                    continue
                knee = min(max(x, y) for x, y in points)
                if knee > per_release_knee.get(release, -1.0):
                    per_release_knee[release] = knee
        label = 1 if verdict in ("clear", "possible") else 0
        for release, knee in per_release_knee.items():
            values = per_release_all.get(release)
            if not values:
                continue
            knees[release].append(knee)
            dangers[release].append(float(np.median(values)))
            truths[release].append(label)

    rng = np.random.default_rng(0)
    print(f"풀: {options.pool}\n")
    print(
        f"{'릴리즈':>7} {'장면':>6} {'양성':>5} {'무릎값':>8} {'위험도':>8} "
        f"{'차이':>8} {'차이 95% 구간':>22} {'차이<=0':>8}"
    )
    for release in sorted(knees):
        k = np.array(knees[release])
        d = np.array(dangers[release])
        t = np.array(truths[release])
        if len(t) < 20 or t.sum() < 5 or (1 - t).sum() < 5:
            continue
        ak, ad = auc(k, t), auc(d, t)
        diffs = []
        for _ in range(2000):
            index = rng.integers(0, len(t), len(t))
            if t[index].sum() in (0, len(index)):
                continue
            diffs.append(auc(k[index], t[index]) - auc(d[index], t[index]))
        diffs = np.array(diffs)
        print(
            f"{release:>7.2f} {len(t):>6} {int(t.sum()):>5} {ak:>8.3f} {ad:>8.3f} "
            f"{ak - ad:>+8.3f} "
            f"[{np.quantile(diffs, 0.025):>+7.3f}, {np.quantile(diffs, 0.975):>+7.3f}] "
            f"{(diffs <= 0).mean():>7.1%}"
        )


if __name__ == "__main__":
    main()
