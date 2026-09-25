"""How much does response screening inflate the dilemma value?

The dilemma is A's own minimax: he picks a path, the attack takes the better
of X or Y, so

    knee = min over A's paths of max(Q_X, Q_Y)

But the audits do not store all the paths. The generator returns 160
representatives of the physical lattice, and a value-aware screen then keeps
about 39 of them — the per-option minimisers, the runner-plus-option
compromises, the minimax candidates, and the cheapest by effort. The screen
uses a CHEAP approximation of Q; the real Q is priced afterwards, only for
the survivors.

Because a minimum over a subset is never below the minimum over the whole
set, **every knee we have reported is an upper bound** — the dilemma is
overstated by however much the screen threw away. This measures that gap
directly: the same scenes rendered twice, identical in every respect except
that one run keeps all 160 responses and prices them in full.

Reads two output directories and compares knee per (scene, defender).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
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


def knee_of(scene, defender, beneficiary):
    """min over stored responses of max(Q_runner, Q_beneficiary), plus context."""
    runner = str(scene["runner_id"])
    points = []
    for response in defender["responses"]:
        cells = response.get("cells") or {}
        a, b = cells.get(runner) or {}, cells.get(beneficiary) or {}
        if a.get("q") is None or b.get("q") is None:
            continue
        if a.get("legal") is False or b.get("legal") is False:
            continue
        points.append((float(a["q"]), float(b["q"])))
    if len(points) < 2:
        return None
    return {
        "knee": min(max(x, y) for x, y in points),
        "floor_x": min(x for x, _ in points),
        "floor_y": min(y for _, y in points),
        "responses": len(points),
    }


def beneficiary_of(scene, defender):
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
    state = onset_state(scene)
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
        str(defender["defender_id"]),
        defender_at,
        attacker_at,
        curves,
        float(scene["horizon_seconds"]),
    )
    return pick or None


def load(directory):
    return {
        (str(s["match_id"]), str(s["onset_frame_id"])): s
        for s in json.loads(
            (Path(directory) / "local_game_payoff_audits.json").read_text()
        )
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("screened_dir")
    parser.add_argument("full_dir")
    options = parser.parse_args()

    screened, full = load(options.screened_dir), load(options.full_dir)
    shared = sorted(set(screened) & set(full))
    print(f"공통 장면 {len(shared)}개\n")

    rows = []
    for key in shared:
        scene_s, scene_f = screened[key], full[key]
        by_id_f = {
            str(d["defender_id"]): d for d in scene_f["candidate_defenders"]
        }
        for defender_s in scene_s["candidate_defenders"]:
            defender_id = str(defender_s["defender_id"])
            defender_f = by_id_f.get(defender_id)
            if defender_f is None:
                continue
            # Use the SCREENED run's beneficiary for both, so the comparison
            # isolates the response set and not a change of option.
            beneficiary = beneficiary_of(scene_s, defender_s)
            if beneficiary is None:
                continue
            left = knee_of(scene_s, defender_s, beneficiary)
            right = knee_of(scene_f, defender_f, beneficiary)
            if left is None or right is None:
                continue
            rows.append(
                (
                    key[1],
                    str(defender_s["defender_name"]),
                    left["responses"],
                    right["responses"],
                    left["knee"],
                    right["knee"],
                    left["floor_x"],
                    right["floor_x"],
                    left["floor_y"],
                    right["floor_y"],
                )
            )

    if not rows:
        print("비교 가능한 쌍이 없음")
        return

    print(
        f"{'장면':<8} {'수비수':<20} {'응답 수':>12}  "
        f"{'무릎값 선별':>10} {'무릎값 전체':>10} {'과대평가':>9}"
    )
    inflation = []
    for (
        frame,
        name,
        n_left,
        n_right,
        knee_left,
        knee_right,
        fx_l,
        fx_r,
        fy_l,
        fy_r,
    ) in rows:
        gap = knee_left - knee_right
        relative = gap / knee_left if knee_left > 1e-9 else 0.0
        inflation.append(relative)
        print(
            f"{frame:<8} {name[:19]:<20} {n_left:>5} -> {n_right:<4}  "
            f"{knee_left:>10.4f} {knee_right:>10.4f} {relative:>8.1%}"
        )

    print(
        f"\n무릎값 과대평가: 중앙 {statistics.median(inflation):.1%}  "
        f"평균 {statistics.mean(inflation):.1%}  "
        f"최대 {max(inflation):.1%}  최소 {min(inflation):.1%}"
    )
    changed = sum(1 for value in inflation if value > 0.01)
    print(f"1% 넘게 부풀려진 쌍: {changed}/{len(inflation)}")

    print("\n단독 바닥값도 같이 (선별 -> 전체)")
    for frame, name, _, _, _, _, fx_l, fx_r, fy_l, fy_r in rows:
        print(
            f"  {frame:<8} {name[:19]:<20} 러너 {fx_l:.4f} -> {fx_r:.4f}   "
            f"수혜자 {fy_l:.4f} -> {fy_r:.4f}"
        )


if __name__ == "__main__":
    main()
