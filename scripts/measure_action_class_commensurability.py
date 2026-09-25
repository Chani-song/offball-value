#!/usr/bin/env python3
"""Are pass options and carry options priced on the same scale?

Q = P x G x A is compared across candidates by argmax, so P has to mean the
same thing for every action class it ranks. It does not. In the shipped
pipeline:

    pass  P = xPass(geometry) x mechanistic chain     (hybrid)
            = a learned StatsBomb probability          (xpass360)
    carry P = path_min_retention
            = min_t  PROD_defenders logistic((arrival - 0.35) / 0.25)
                                                       (both builds)

Carry delivery has no learned component in either build and is flagged
`uncalibrated_carry_retention`. Under the hybrid, pass and carry at least
shared the same mechanistic machinery. Swapping the pass model to xpass360
moves passes onto a learned scale and leaves carries where they were, so the
exchange rate between the two action classes changes even though nothing
about carrying changed.

Section 4 tests that mechanism directly: multiply every PASS row of the
candidate grid by a constant k, leave carry and structural rows alone, and
rescore R9. If the beneficiary answers the swap lost come back at the k that
undoes the measured pass uplift, the flip was an exchange-rate artefact.

k is a DIAGNOSTIC, not a proposed fix. Choosing k by which value scores best
on the labels would be fitting the evaluation set; the legitimate repair is
to calibrate both delivery models against observed outcomes.

Usage:
    python scripts/measure_action_class_commensurability.py \
        --hybrid data/processed/goalside_v1_chunk{0,1,2} \
        --x360 out/scene_* \
        --output out/commensurability/report.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter, defaultdict
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
_scorer = _coupled._scorer

PASS_ROWS = {"through_ball_to_space", "receive_to_feet", "cutback_to_space"}
CARRY_ROWS = {"carry_into_space", "carrier_carry"}
STRUCT_ROWS = {"terminal_structure"}

EXCLUDE_FRAMES = {"57121"}          # label contaminated
SWEEP = [1.0, 0.9, 0.8, 0.7, 0.6, 0.55, 0.53, 0.5, 0.45, 0.4, 0.3]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--hybrid", type=Path, nargs="+", required=True)
    p.add_argument("--x360", type=Path, nargs="+", required=True)
    p.add_argument("--output", type=Path, default=None)
    return p.parse_args()


def action_class(row_type: str) -> str:
    if row_type in PASS_ROWS:
        return "pass"
    if row_type in CARRY_ROWS:
        return "carry"
    return "structural"


def scan(dirs, want=None):
    """Compact per-cell records for every scene; full scenes only for `want`."""
    cells = {}
    keep = {}
    for directory in dirs:
        path = directory / "local_game_payoff_audits.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        for scene in payload:
            key = (str(scene["match_id"]), str(scene["onset_frame_id"]))
            if want is not None and key in want:
                keep[key] = scene
            per = {}
            for defender in scene["candidate_defenders"]:
                did = str(defender["defender_id"])
                for response in defender["responses"]:
                    rid = str(response["response_id"])
                    for oid, cell in (response.get("cells") or {}).items():
                        if cell.get("legal") is False or cell.get("q") is None:
                            continue
                        per[(did, rid, oid)] = (
                            str(cell.get("continuation_type")),
                            float(cell["delivery"]),
                            float(cell["q"]),
                        )
            cells[key] = per
        del payload
    return cells, keep


def r9_pick(scene, defender_id, pass_scale=1.0):
    state = onset_state(scene)
    if defender_id not in state:
        return None
    defender = next(
        (r for r in scene["candidate_defenders"] if str(r["defender_id"]) == defender_id),
        None,
    )
    if defender is None:
        return None
    response = next(
        (
            r
            for r in defender["responses"]
            if str(r["response_id"]) == str(defender.get("direct_best_response_id"))
        ),
        None,
    )
    if response is None:
        return None
    defender_at, attacker_at = _coupled.absolute_lookups(scene, defender)
    if defender_at is None:
        return None
    runner = str(scene["runner_id"])
    curves, kinds = {}, {}
    for oid, cell in response["cells"].items():
        if cell.get("legal") is False or cell.get("q") is None or oid == runner:
            continue
        grid = []
        for row in cell.get("candidate_grid") or []:
            scale = pass_scale if action_class(str(row.get("type"))) == "pass" else 1.0
            grid.append({**row, "q": float(row["q"]) * scale})
        curves[oid] = value_at_times(grid)
        kinds[oid] = action_class(str(cell.get("continuation_type")))
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
    return pick, kinds


def main() -> None:
    args = parse_args()
    labels = _scorer.load_labels()
    want = {(m, f) for (m, f, _d) in labels}

    print("하이브리드 감사 읽는 중 ...", flush=True)
    hy_cells, hy_scenes = scan(args.hybrid, want)
    print("xpass360 감사 읽는 중 ...", flush=True)
    xp_cells, xp_scenes = scan(args.x360, want)
    common = sorted(set(hy_cells) & set(xp_cells))
    print(f"양쪽에 있는 장면 {len(common)}개\n", flush=True)
    out = {"scenes": len(common)}

    # ---- 1. 액션 클래스별 P 수준 ------------------------------------------
    print("=" * 84)
    print("[1] 액션 클래스별 배달 P — 같은 셀을 두 빌드에서")
    print("=" * 84)
    buckets = defaultdict(lambda: {"hy": [], "xp": []})
    for key in common:
        a, b = hy_cells[key], xp_cells[key]
        for ck in set(a) & set(b):
            cls = action_class(a[ck][0])
            buckets[cls]["hy"].append(a[ck][1])
            buckets[cls]["xp"].append(b[ck][1])
    print(f"  {'클래스':<14}{'n':>9}{'P하이중앙':>11}{'P360중앙':>11}{'배율':>8}{'불변':>8}")
    out["levels"] = {}
    for cls in ("pass", "carry", "structural"):
        if not buckets[cls]["hy"]:
            continue
        h = np.array(buckets[cls]["hy"])
        x = np.array(buckets[cls]["xp"])
        rec = {
            "n": int(h.size),
            "hybrid_median": float(np.median(h)),
            "x360_median": float(np.median(x)),
            "uplift": float(np.median(x) / max(np.median(h), 1e-9)),
            "unchanged_fraction": float(np.mean(np.abs(x - h) < 1e-9)),
        }
        out["levels"][cls] = rec
        print(f"  {cls:<14}{rec['n']:>9,}{rec['hybrid_median']:>11.3f}"
              f"{rec['x360_median']:>11.3f}{rec['uplift']:>8.2f}"
              f"{rec['unchanged_fraction']:>8.1%}")
    if "pass" in out["levels"] and "carry" in out["levels"]:
        rate_h = out["levels"]["pass"]["hybrid_median"] / out["levels"]["carry"]["hybrid_median"]
        rate_x = out["levels"]["pass"]["x360_median"] / out["levels"]["carry"]["x360_median"]
        out["exchange_rate"] = {"hybrid": rate_h, "x360": rate_x, "shift": rate_x / rate_h}
        print(f"\n  패스/캐리 교환비   하이브리드 {rate_h:.3f}  ->  xpass360 {rate_x:.3f}"
              f"   ({rate_x / rate_h:.2f}배 이동)")
        print(f"  되돌리는 k = {rate_h / rate_x:.3f}")

    # ---- 2~3. argmax 구성과 라벨 구성 --------------------------------------
    print("\n" + "=" * 84)
    print("[2] R9 가 고른 수혜자의 액션 클래스 / [3] 라벨 정답의 액션 클래스")
    print("=" * 84)
    pick_kind = {"hy": Counter(), "xp": Counter()}
    label_kind = Counter()
    scorable = []
    for (match_id, frame_id, defender_id), answer in sorted(labels.items()):
        if frame_id in EXCLUDE_FRAMES:
            continue
        key = (match_id, frame_id)
        if key not in hy_scenes or key not in xp_scenes:
            continue
        left = r9_pick(hy_scenes[key], defender_id)
        right = r9_pick(xp_scenes[key], defender_id)
        if left is None or right is None:
            continue
        scorable.append((key, defender_id, answer))
        pick_kind["hy"][left[1].get(left[0], "?")] += 1
        pick_kind["xp"][right[1].get(right[0], "?")] += 1
        label_kind[left[1].get(answer, "?")] += 1
    out["pick_kind"] = {k: dict(v) for k, v in pick_kind.items()}
    out["label_kind"] = dict(label_kind)
    out["scorable_pairs"] = len(scorable)
    print(f"\n  채점 가능 쌍 {len(scorable)}개 (57121 제외)")
    print(f"  {'':<22}{'pass':>9}{'carry':>9}{'structural':>12}")
    print(f"  {'R9 선택 (하이)':<20}{pick_kind['hy']['pass']:>9}"
          f"{pick_kind['hy']['carry']:>9}{pick_kind['hy']['structural']:>12}")
    print(f"  {'R9 선택 (360)':<20}{pick_kind['xp']['pass']:>9}"
          f"{pick_kind['xp']['carry']:>9}{pick_kind['xp']['structural']:>12}")
    print(f"  {'라벨 정답':<20}{label_kind['pass']:>9}"
          f"{label_kind['carry']:>9}{label_kind['structural']:>12}")

    # ---- 4. 교환비 스윕 -----------------------------------------------------
    print("\n" + "=" * 84)
    print("[4] 패스 행에 k 를 곱하고 R9 재채점 (진단용, 교정안 아님)")
    print("=" * 84)
    print(f"\n  {'k':>6}{'하이 정답':>10}{'360 정답':>10}   360 에서 되살아난 쌍")
    out["sweep"] = []
    base_wrong = None
    for k in SWEEP:
        hy_hit = xp_hit = 0
        xp_right = set()
        for key, defender_id, answer in scorable:
            lh = r9_pick(hy_scenes[key], defender_id, k)
            lx = r9_pick(xp_scenes[key], defender_id, k)
            if lh and lh[0] == answer:
                hy_hit += 1
            if lx and lx[0] == answer:
                xp_hit += 1
                xp_right.add((key[1], defender_id))
        if base_wrong is None:
            base_wrong = {
                (key[1], d) for key, d, a in scorable
            } - xp_right
        recovered = sorted(f"{f}" for f, d in (base_wrong & xp_right))
        out["sweep"].append({"k": k, "hybrid": hy_hit, "x360": xp_hit,
                             "recovered": recovered})
        print(f"  {k:>6.2f}{hy_hit:>10}{xp_hit:>10}   {', '.join(recovered) or '-'}")
        sys.stdout.flush()
    out["total_pairs"] = len(scorable)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(out, ensure_ascii=False, indent=1))
        print(f"\nsaved {args.output}")


if __name__ == "__main__":
    main()
