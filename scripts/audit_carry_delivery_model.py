#!/usr/bin/env python3
"""Does the carry (dribble) delivery model do any work at all?

The pass side of Q has had two models and an audit. The carry side has had
neither. Its delivery is

    path_min_retention = min_t PROD_defenders logistic((arrival_t - 0.35) / 0.25)

flagged `uncalibrated_carry_retention`, with no learned component in either
build. Two things follow that nobody has measured:

1. If that quantity sits near 1 for almost every carry, then for carry
   options Q = P x G x A collapses to G x A -- the same degeneracy the pass
   side was suspected of, on the other action class.
2. If it does not move when the candidate defender changes his response,
   the carry option is blind to the defender in exactly the way the hybrid
   pass model was accused of being.

Both are asked here against the audits already on disk. No model needed:
the carry delivery and its pressure components are recorded per cell.

Usage:
    python scripts/audit_carry_delivery_model.py --audits <dir> [<dir> ...]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

CARRY_TYPES = {"carrier_carry", "carry_into_space"}
PASS_TYPES = {"through_ball_to_space", "receive_to_feet", "cutback_to_space"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--audits", type=Path, nargs="+", required=True)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--label", type=str, default="")
    return p.parse_args()


def describe(name, values):
    a = np.asarray(values, float)
    if a.size == 0:
        return None
    return {
        "n": int(a.size),
        "median": float(np.median(a)),
        "mean": float(a.mean()),
        "p10": float(np.percentile(a, 10)),
        "p90": float(np.percentile(a, 90)),
        "above_0_9": float(np.mean(a > 0.9)),
        "above_0_95": float(np.mean(a > 0.95)),
        "std": float(a.std()),
    }


def main() -> None:
    args = parse_args()
    carry_p, pass_p = [], []
    carry_groups = defaultdict(list)
    pass_groups = defaultdict(list)
    weakest_times, q_carry, ga_carry = [], [], []

    for directory in args.audits:
        path = directory / "local_game_payoff_audits.json"
        if not path.exists():
            continue
        for scene in json.loads(path.read_text(encoding="utf-8")):
            key0 = f"{scene['match_id']}:{scene['onset_frame_id']}"
            for defender in scene["candidate_defenders"]:
                did = str(defender["defender_id"])
                for response in defender["responses"]:
                    for oid, cell in (response.get("cells") or {}).items():
                        if cell.get("legal") is False or cell.get("q") is None:
                            continue
                        ctype = str(cell.get("continuation_type"))
                        delivery = float(cell["delivery"])
                        if ctype in CARRY_TYPES:
                            carry_p.append(delivery)
                            carry_groups[(key0, did, oid)].append(delivery)
                            q_carry.append(float(cell["q"]))
                            ga_carry.append(
                                float(cell["goal"]) * float(cell["accessibility"])
                            )
                            dc = cell.get("delivery_components") or {}
                            t = dc.get("weakest_pressure_arrival_time_s")
                            if t is not None:
                                weakest_times.append(float(t))
                        elif ctype in PASS_TYPES:
                            pass_p.append(delivery)
                            pass_groups[(key0, did, oid)].append(delivery)

    print("=" * 84)
    print(f"캐리(드리블) 배달 모델 감사   {args.label}")
    print("=" * 84)
    out = {"label": args.label}

    print("\n[1] 배달 P 분포")
    print(f"  {'':<12}{'n':>9}{'중앙':>8}{'평균':>8}{'표준편차':>10}"
          f"{'10%':>8}{'90%':>8}{'>0.9':>8}{'>0.95':>8}")
    for name, values in (("캐리", carry_p), ("패스", pass_p)):
        d = describe(name, values)
        if d is None:
            continue
        out[name] = d
        print(f"  {name:<12}{d['n']:>9,}{d['median']:>8.3f}{d['mean']:>8.3f}"
              f"{d['std']:>10.3f}{d['p10']:>8.3f}{d['p90']:>8.3f}"
              f"{d['above_0_9']:>8.1%}{d['above_0_95']:>8.1%}")

    print("\n[2] 수비수가 대응을 바꿀 때 P 가 움직이나 (같은 장면·수비수·옵션)")
    print(f"  {'':<12}{'그룹':>9}{'중앙폭':>10}{'평균폭':>10}{'무반응<1%p':>13}")
    out["spread"] = {}
    for name, groups in (("캐리", carry_groups), ("패스", pass_groups)):
        spreads = [max(v) - min(v) for v in groups.values() if len(v) >= 4]
        if not spreads:
            continue
        a = np.array(spreads)
        out["spread"][name] = {
            "groups": int(a.size), "median": float(np.median(a)),
            "mean": float(a.mean()), "flat": float(np.mean(a < 0.01)),
        }
        print(f"  {name:<12}{a.size:>9,}{np.median(a):>10.3f}{a.mean():>10.3f}"
              f"{np.mean(a < 0.01):>13.1%}")

    if q_carry:
        q = np.array(q_carry)
        ga = np.array(ga_carry)
        c = float(np.corrcoef(q, ga)[0, 1])
        out["carry_q_vs_ga_corr"] = c
        out["carry_q_over_ga"] = describe("ratio", q / np.maximum(ga, 1e-9))
        print("\n[3] 캐리에서 Q 가 G×A 로 무너졌나")
        print(f"  corr(Q, G×A) = {c:.4f}   (1.000 에 가까우면 P 가 사실상 상수)")
        r = out["carry_q_over_ga"]
        print(f"  Q/(G×A) = P :  중앙 {r['median']:.3f} · 표준편차 {r['std']:.3f}"
              f" · 10~90% {r['p10']:.3f}~{r['p90']:.3f}")

    if weakest_times:
        t = np.array(weakest_times)
        out["weakest_pressure_arrival_s"] = describe("t", t)
        print("\n[4] 가장 약한 지점의 수비수 도착시간 (모델 입력)")
        print(f"  n={t.size:,} · 중앙 {np.median(t):.2f}s · 10~90% "
              f"{np.percentile(t, 10):.2f}~{np.percentile(t, 90):.2f}s")
        print(f"  임계 0.35s 이내로 압박 도달하는 비율: {np.mean(t <= 0.35):.1%}")
        print("  (logistic((t-0.35)/0.25) 이므로 t 가 1s 를 넘으면 P 는 사실상 1)")
        print(f"  t > 1.0s 비율: {np.mean(t > 1.0):.1%}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(out, ensure_ascii=False, indent=1))
        print(f"\nsaved {args.output}")


if __name__ == "__main__":
    main()
