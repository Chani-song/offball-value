"""Single declared opening of round 2 for the coupled-clock rule.

Scores exactly the definition locked in docs/round2_coupled_clock_prediction_sealed.md
— instantaneous loss region, the scene's own horizon, area-weighted mean,
Q read at the same instant with a +/-0.4 s tolerance — against R6, on the
round-2 blind labels.

No grid, no constants to vary: the two columns are the sealed rule and the
incumbent, and this script is meant to be run once.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

_spec = importlib.util.spec_from_file_location(
    "decompose_clock_coupling", ROOT / "scripts" / "decompose_clock_coupling.py"
)
_decompose = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_decompose)

# The scene the reviewer and I dissected together while developing R7; he
# ruled it contaminated for any later validation.
CONTAMINATED = {"57121"}


def predict(cache, *, value, horizon_cap):
    times = np.array(cache["times"])
    keep = times <= horizon_cap + 1e-9
    weight = cache["weights"]["loss"][keep]
    denominator = float(weight.sum())
    scores = {}
    for option_id in cache["options"]:
        occupation = cache["occupation"]["loss"][option_id][keep]
        series = occupation * (
            cache["static_value"][option_id]
            if value == "static"
            else cache["coupled_value"][option_id][keep]
        )
        scores[option_id] = (
            float((weight * series).sum() / denominator) if denominator > 1e-12 else 0.0
        )
    return max(scores, key=lambda k: (scores[k], k)) if scores else ""


def mcnemar_two_sided(only_a: int, only_b: int) -> float:
    n = only_a + only_b
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(only_a, only_b) + 1))
    return min(1.0, 2.0 * tail / (2.0**n))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audit_dirs", nargs="+")
    options = parser.parse_args()

    labels = _decompose._scorer.load_labels()
    scenes = {}
    for directory in options.audit_dirs:
        for scene in json.loads(
            (Path(directory) / "local_game_payoff_audits.json").read_text()
        ):
            scenes[(str(scene["match_id"]), str(scene["onset_frame_id"]))] = scene

    rows = []
    for (match_id, frame_id, defender_id), answer in sorted(labels.items()):
        scene = scenes.get((match_id, frame_id))
        if scene is None:
            continue
        cache = _decompose.build_cache(scene, defender_id)
        if cache is None:
            continue
        r6 = predict(cache, value="static", horizon_cap=1.5)
        r9 = predict(cache, value="coupled", horizon_cap=99.0)
        rows.append(
            {
                "frame": frame_id,
                "defender": cache["defender_name"],
                "answer": cache["names"].get(answer, answer),
                "r6": cache["names"].get(r6, r6 or "(없음)"),
                "r9": cache["names"].get(r9, r9 or "(없음)"),
                "r6_ok": r6 == answer,
                "r9_ok": r9 == answer,
                "contaminated": frame_id in CONTAMINATED,
            }
        )

    def report(subset, title):
        n = len(subset)
        r6 = sum(row["r6_ok"] for row in subset)
        r9 = sum(row["r9_ok"] for row in subset)
        both = sum(row["r6_ok"] and row["r9_ok"] for row in subset)
        only6 = sum(row["r6_ok"] and not row["r9_ok"] for row in subset)
        only9 = sum(row["r9_ok"] and not row["r6_ok"] for row in subset)
        neither = sum(not row["r6_ok"] and not row["r9_ok"] for row in subset)
        print(f"\n=== {title} (n = {n}) ===")
        print(f"  R6 (분리, T=1.5초)     {r6:>2}  ({r6 / n:.0%})")
        print(f"  R9 (결합, 전체 지평)   {r9:>2}  ({r9 / n:.0%})")
        print(f"  대응: 둘 다 {both} / R6만 {only6} / R9만 {only9} / 둘 다 오답 {neither}")
        print(f"  McNemar 양측 p = {mcnemar_two_sided(only6, only9):.4f}")

    primary = [row for row in rows if not row["contaminated"]]
    report(primary, "1차 판정 — 57121 제외")
    report(rows, "참고 — 전체")

    print("\n장면별")
    for row in sorted(rows, key=lambda r: r["frame"]):
        mark = "  <-- 다름" if row["r6_ok"] != row["r9_ok"] else ""
        flag = " *오염" if row["contaminated"] else ""
        print(
            f"  {row['frame']}/{row['defender']:<22} "
            f"R6 {'O' if row['r6_ok'] else 'X'}  R9 {'O' if row['r9_ok'] else 'X'}{mark}{flag}"
        )

    for key, title in (("r6", "R6"), ("r9", "R9")):
        misses = [row for row in rows if not row[f"{key}_ok"]]
        print(f"\n{title} 오답 {len(misses)}개:")
        for row in sorted(misses, key=lambda r: r["frame"]):
            flag = " *오염" if row["contaminated"] else ""
            print(
                f"  {row['frame']}/{row['defender']:<20} "
                f"정답={row['answer']:<22} {title}={row[key]}{flag}"
            )


if __name__ == "__main__":
    main()
