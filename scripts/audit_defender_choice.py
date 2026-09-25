#!/usr/bin/env python3
"""Who gets picked as the reacting defender, and does the pick make sense?

Defender candidates are ranked by goal-side marking distance and pursuit
cost, both measured FROM the runner, and the top three are always taken.
Nothing asks the reverse question. A defender can be the closest man to this
runner and still be standing two metres from a different attacker he is
plainly marking, and he will not leave that man for a runner thirty metres
away. The local game then solves a bind nobody was ever in.

Five things are measured here, on builds that already exist:

  1 asymmetry    is the runner the nearest attacker to that defender, and if
                 not, how much nearer is the other one
  2 filters      how many triples and how many labelled games survive each
                 candidate filter, at several thresholds
  3 always-3     how often the third pick is far enough to be filler
  4 reaction     does the defender's own path bend toward the runner at all,
                 or is he picked while walking the other way
  5 goalkeepers  are keepers entering the candidate list

A filter is only worth adopting if it drops triples without dropping the
defenders the labels say actually reacted, so (2) reports both together.
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import math
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--build", type=Path,
                   default=Path("/work/hdd/bbmr/kseo1/offball-out/v7_r9_ssac"))
    return p.parse_args()


def load_labels() -> dict:
    spec = importlib.util.spec_from_file_location(
        "sv", "scripts/score_vacated_space_rule.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.load_labels()


def gather(build: Path) -> list[dict]:
    """One row per (scene, candidate defender), with the geometry we need."""
    rows = []
    for path in sorted(glob.glob(
            f"{build}/scene_*/local_game_payoff_audits.json")):
        try:
            payload = json.load(open(path))
        except Exception:
            continue
        if not payload:
            continue
        g = payload[0]
        frames = g["background_frames"]
        onset = g["onset_frame_id"]
        at = min(frames, key=lambda fr: abs(fr["frame_id"] - onset))
        att = str(g["attacking_team_id"])
        pos = {str(p[0]): (p[2], p[3]) for p in at["players"]}
        attackers = [str(p[0]) for p in at["players"] if str(p[1]) == att]
        runner = str(g["runner_id"])
        if runner not in pos:
            continue
        # where the defender is later, to see whether he moves toward the run
        end = frames[-1]
        pos_end = {str(p[0]): (p[2], p[3]) for p in end["players"]}

        for rank, d in enumerate(g.get("candidate_defenders") or []):
            did = str(d["defender_id"])
            if did not in pos:
                continue
            dx, dy = pos[did]
            dist = {a: math.hypot(pos[a][0] - dx, pos[a][1] - dy)
                    for a in attackers if a in pos}
            if runner not in dist:
                continue
            ordered = sorted(dist, key=lambda a: dist[a])
            to_runner = dist[runner]
            nearest = dist[ordered[0]]
            # closing speed toward the runner over the clip
            closed = None
            if did in pos_end and runner in pos_end:
                later = math.hypot(pos_end[runner][0] - pos_end[did][0],
                                   pos_end[runner][1] - pos_end[did][1])
                closed = to_runner - later
            rows.append({
                "match_id": str(g["match_id"]),
                "onset_frame_id": str(g["onset_frame_id"]),
                "defender_id": did,
                "rank": rank,
                "to_runner": to_runner,
                "nearest": nearest,
                "gap": to_runner - nearest,
                "runner_rank": ordered.index(runner),
                "closed": closed,
                "minimax": float(d.get("minimax_worst_q") or 0.0),
            })
    return rows


def main() -> None:
    args = parse_args()
    rows = gather(args.build)
    labels = set(load_labels())
    n = len(rows)
    print(f"후보 (장면 × 수비수) {n}개 · 라벨 {len(labels)}개\n")

    def labelled(keep) -> int:
        return len({(r["match_id"], r["onset_frame_id"], r["defender_id"])
                    for r in keep} & labels)

    base_lab = labelled(rows)
    print(f"현재 채점 가능한 라벨 {base_lab}개\n")

    # --- 1. asymmetry ------------------------------------------------------
    print("=" * 66)
    print("1. 그 수비수에게 러너는 몇 번째로 가까운 공격수인가")
    print("=" * 66)
    for k in range(4):
        c = sum(1 for r in rows if r["runner_rank"] == k)
        tag = f"{k+1}번째" if k < 3 else "4번째 이하"
        if k == 3:
            c = sum(1 for r in rows if r["runner_rank"] >= 3)
        print(f"   {tag:10} {c:5}  ({c/n:.0%})")

    # --- 2. filters --------------------------------------------------------
    print("\n" + "=" * 66)
    print("2. 후보 필터별 생존 (삼중항 / 라벨)")
    print("=" * 66)
    print(f"   {'필터':34} {'삼중항':>12} {'라벨':>10}")
    print(f"   {'현재 (필터 없음)':34} {n:6} (100%) {base_lab:6}/{base_lab}")
    tests = []
    for k in (1, 2):
        tests.append((f"러너가 최근접 {k}순위 이내",
                      lambda r, k=k: r["runner_rank"] < k))
    for delta in (3.0, 5.0, 8.0):
        tests.append((f"최근접보다 {delta:.0f}m 넘게 멀면 제외",
                      lambda r, d=delta: r["gap"] <= d))
    for cap in (15.0, 20.0, 25.0):
        tests.append((f"러너까지 {cap:.0f}m 초과 제외",
                      lambda r, c=cap: r["to_runner"] <= c))
    tests.append(("상위 2명만 (3번째 버림)", lambda r: r["rank"] < 2))
    tests.append(("최근접 2순위 이내 AND 20m 이내",
                  lambda r: r["runner_rank"] < 2 and r["to_runner"] <= 20.0))
    for name, fn in tests:
        keep = [r for r in rows if fn(r)]
        lab = labelled(keep)
        flag = "  <- 라벨 손실" if lab < base_lab else ""
        print(f"   {name:34} {len(keep):6} ({len(keep)/n:3.0%}) "
              f"{lab:6}/{base_lab}{flag}")

    # --- 3. always three ---------------------------------------------------
    print("\n" + "=" * 66)
    print("3. '무조건 3명'이 만들어내는 것 — 순위별 러너까지 거리")
    print("=" * 66)
    for k in range(3):
        sub = [r["to_runner"] for r in rows if r["rank"] == k]
        if not sub:
            continue
        a = np.array(sub)
        print(f"   {k+1}순위 수비수  중앙 {np.median(a):5.1f} m · "
              f"75% {np.percentile(a,75):5.1f} m · 최대 {a.max():5.1f} m · "
              f"20m 초과 {(a>20).mean():.0%}")

    # --- 4. does he react --------------------------------------------------
    print("\n" + "=" * 66)
    print("4. 이 수비수가 실제로 러너 쪽으로 좁히는가 (온셋 → +3초)")
    print("=" * 66)
    closed = np.array([r["closed"] for r in rows if r["closed"] is not None])
    if closed.size:
        print(f"   좁힘(양수) {(closed>0).mean():.0%} · "
              f"벌어짐(음수) {(closed<0).mean():.0%}")
        print(f"   중앙 {np.median(closed):+.1f} m · "
              f"5m 이상 벌어진 경우 {(closed<-5).sum()} "
              f"({(closed<-5).mean():.0%})")
        far = [r for r in rows if r["closed"] is not None
               and r["closed"] < 0 and r["runner_rank"] > 0]
        print(f"   러너가 최근접도 아니고 벌어지기까지 한 후보: "
              f"{len(far)} ({len(far)/n:.0%})")

    # --- 5. keepers --------------------------------------------------------
    print("\n" + "=" * 66)
    print("5. 골키퍼가 후보에 들어오나")
    print("=" * 66)
    deep = [r for r in rows if r["to_runner"] > 30]
    print(f"   러너까지 30m 초과인 후보 {len(deep)} ({len(deep)/n:.0%})")
    print("   (골키퍼 플래그가 payload에 없어 거리로만 봅니다. 30m 초과가")
    print("    많으면 키퍼나 반대편 수비가 섞였을 가능성이 큽니다.)")


if __name__ == "__main__":
    main()
