#!/usr/bin/env python3
"""Same static features, different arrival race: can xpass360 tell them apart?

The lane-feature distributions of our option catalogue and of real StatsBomb
passes are nearly identical (lane_blockers_2m mean 0.37 in both), yet xpass360
prices our cells at a median 0.966 against 0.823 on real passes. The proposed
explanation is a CONDITIONAL shift, not a covariate one: the same static
geometry means something different when the target is the future position of a
race rather than the present position of a settled receiver.

That explanation predicts something checkable. Hold the static features fixed
-- stratify on them -- and vary only the kinematic race. A model built from
static freeze frames has no feature that can respond; the mechanistic chain,
which carries receiver_first and the arrival margins, must.

If xpass360 still tracks receiver_first strongly WITHIN a static stratum, the
explanation is wrong and its apparent blindness was just static geometry
correlating with the race.

Usage:
    python scripts/test_conditional_shift.py --audits out/scene_* \
        --output out/delivery_analysis/conditional_shift.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from offball_value.xpass import lane_features, to_statsbomb_coordinates

PASS_TYPES = {"through_ball_to_space", "receive_to_feet", "cutback_to_space"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--audits", type=Path, nargs="+", required=True)
    p.add_argument("--calibration", type=Path,
                   default=Path("out/delivery_analysis/hybrid_calibration.json"))
    p.add_argument("--output", type=Path, default=None)
    return p.parse_args()


def frame_lookup(scene):
    out = {}
    for frame in scene["background_frames"]:
        players = {str(r[0]): (str(r[1]), float(r[2]), float(r[3]))
                   for r in frame["players"]}
        ball = frame.get("ball")
        out[round(float(frame["relative_time_s"]), 2)] = (
            players, (float(ball[0]), float(ball[1])) if ball else None)
    return out


def nearest_time(keys, t):
    return min(keys, key=lambda u: abs(u - t))


def main() -> None:
    args = parse_args()
    rows = []
    for directory in args.audits:
        path = directory / "local_game_payoff_audits.json"
        if not path.exists():
            continue
        for scene in json.loads(path.read_text(encoding="utf-8")):
            frames = frame_lookup(scene)
            keys = list(frames)
            attacking = str(scene["attacking_team_id"])
            direction = int(scene["attacking_direction"])
            for defender in scene["candidate_defenders"]:
                for response in defender["responses"]:
                    for _oid, cell in (response.get("cells") or {}).items():
                        if cell.get("legal") is False or cell.get("q") is None:
                            continue
                        if str(cell.get("continuation_type")) not in PASS_TYPES:
                            continue
                        dc = cell.get("delivery_components") or {}
                        rf = dc.get("receiver_first")
                        mech = dc.get("mechanistic_delivery")
                        if rf is None or mech is None:
                            continue
                        players, ball = frames[nearest_time(keys, float(cell["release_time_s"]))]
                        if ball is None:
                            continue
                        end = (float(cell["event_xy"][0]), float(cell["event_xy"][1]))
                        opponents = [(x, y) for _o, (team, x, y) in players.items()
                                     if team != attacking]
                        lf = lane_features(
                            to_statsbomb_coordinates(ball, direction),
                            to_statsbomb_coordinates(end, direction),
                            [to_statsbomb_coordinates(xy, direction) for xy in opponents])
                        rows.append((float(cell["delivery"]), float(mech), float(rf),
                                     float(dc.get("pass_distance_m") or 0.0),
                                     float(lf[2]), float(lf[1]), float(lf[4])))
        print(f"  [{directory.name}] 누적 {len(rows):,}", flush=True)

    A = np.array(rows)
    P360, MECH, RF, DIST, BLK2, PERP, TNO = (A[:, i] for i in range(7))
    print(f"\n패스 셀 {len(A):,}개\n")
    out = {"cells": int(len(A))}

    def corr(a, b):
        if a.size < 30 or a.std() < 1e-9 or b.std() < 1e-9:
            return np.nan
        return float(np.corrcoef(a, b)[0, 1])

    def rank(v):
        order = np.argsort(np.argsort(v))
        return order.astype(float)

    def spearman(a, b):
        return corr(rank(a), rank(b))

    # Isotonic is monotone, so it preserves RANK correlation exactly but can
    # crush Pearson. Which of the two matters is the whole question: argmax
    # over Q = P x G x A needs the spacing, not just the order.
    CAL = None
    if args.calibration and args.calibration.exists():
        c = json.loads(args.calibration.read_text(encoding="utf-8"))
        cx = np.array(c["isotonic_x"], float); cy = np.array(c["isotonic_y"], float)
        def CAL(v):  # noqa: E731
            idx = np.clip(np.searchsorted(cx, v, side="right") - 1, 0, len(cx) - 1)
            return cy[idx]

    print("=" * 84)
    print("[1] 전체에서의 상관 (층화 전)")
    print("=" * 84)
    print(f"  corr(xpass360, receiver_first) = {corr(P360, RF):+.3f}")
    print(f"  corr(역학,     receiver_first) = {corr(MECH, RF):+.3f}")
    out["raw"] = {"x360": corr(P360, RF), "mech": corr(MECH, RF)}

    print("\n" + "=" * 84)
    print("[2] 정적 특징을 고정하고(층화) 그 안에서만 본 상관")
    print("=" * 84)
    dist_e = [0, 8, 12, 16, 20, 25, 30, 200]
    perp_e = [0, 1, 2, 3, 5, 100]
    tno_e = [0, 4, 7, 10, 15, 100]
    strata = defaultdict(list)
    for i in range(len(A)):
        d = np.searchsorted(dist_e, DIST[i], "right")
        p_ = np.searchsorted(perp_e, PERP[i], "right")
        t = np.searchsorted(tno_e, TNO[i], "right")
        strata[(d, p_, t, int(min(BLK2[i], 2)))].append(i)

    c360, cmech, spread360, spreadmech, weights = [], [], [], [], []
    s360, smech, scal, ccal, spreadcal = [], [], [], [], []
    MECHCAL = CAL(MECH) if CAL is not None else None
    used = 0
    for _key, idx in strata.items():
        if len(idx) < 100:
            continue
        idx = np.array(idx)
        rf = RF[idx]
        if rf.std() < 0.05:
            continue
        used += len(idx)
        c1, c2 = corr(P360[idx], rf), corr(MECH[idx], rf)
        if np.isnan(c1) or np.isnan(c2):
            continue
        c360.append(c1); cmech.append(c2); weights.append(len(idx))
        s360.append(spearman(P360[idx], rf)); smech.append(spearman(MECH[idx], rf))
        if MECHCAL is not None:
            ccal.append(corr(MECHCAL[idx], rf))
            scal.append(spearman(MECHCAL[idx], rf))
        lo, hi = rf < 0.2, rf > 0.8
        if lo.sum() >= 20 and hi.sum() >= 20:
            spread360.append(P360[idx][hi].mean() - P360[idx][lo].mean())
            spreadmech.append(MECH[idx][hi].mean() - MECH[idx][lo].mean())
            if MECHCAL is not None:
                spreadcal.append(MECHCAL[idx][hi].mean() - MECHCAL[idx][lo].mean())
    w = np.array(weights, dtype=float)
    print(f"  사용된 층 {len(c360)}개 · 셀 {used:,}개 "
          f"(층당 100셀 이상, 층 내 receiver_first 표준편차 >= 0.05)")
    print(f"\n  층 내 corr(모델, receiver_first) 가중평균")
    print(f"    xpass360  {np.average(c360, weights=w):+.3f}")
    print(f"    역학       {np.average(cmech, weights=w):+.3f}")
    out["stratified"] = {"strata": len(c360), "cells": int(used),
                         "x360": float(np.average(c360, weights=w)),
                         "mech": float(np.average(cmech, weights=w))}
    print(f"\n  층 내 SPEARMAN(순위) 상관 — 등장성이 정확히 보존하는 양")
    print(f"    xpass360  {np.average(s360, weights=w):+.3f}")
    print(f"    역학       {np.average(smech, weights=w):+.3f}")
    out["stratified_spearman"] = {"x360": float(np.average(s360, weights=w)),
                                  "mech": float(np.average(smech, weights=w))}
    if ccal:
        wc = w[: len(ccal)]
        print(f"\n  역학을 캘리브레이션한 뒤")
        print(f"    Pearson  {np.average(ccal, weights=wc):+.3f}"
              f"   (보정 전 {np.average(cmech, weights=w):+.3f})")
        print(f"    Spearman {np.average(scal, weights=wc):+.3f}"
              f"   (보정 전 {np.average(smech, weights=w):+.3f})")
        out["stratified_calibrated"] = {
            "pearson": float(np.average(ccal, weights=wc)),
            "spearman": float(np.average(scal, weights=wc))}
        if spreadcal:
            print(f"\n  같은 층에서 receiver_first>0.8 과 <0.2 의 P 간격")
            print(f"    xpass360        {np.median(spread360):+.3f}")
            print(f"    역학 raw         {np.median(spreadmech):+.3f}")
            print(f"    역학 캘리브레이션    {np.median(spreadcal):+.3f}   <- argmax 가 쓰는 양")
            out["gap_calibrated_median"] = float(np.median(spreadcal))
    if spread360:
        s3, sm = np.array(spread360), np.array(spreadmech)
        print(f"\n  같은 층 안에서 receiver_first>0.8 과 <0.2 의 P 차이 (층 {len(s3)}개)")
        print(f"    xpass360  중앙 {np.median(s3):+.3f} · 평균 {s3.mean():+.3f}")
        print(f"    역학       중앙 {np.median(sm):+.3f} · 평균 {sm.mean():+.3f}")
        out["within_stratum_gap"] = {
            "n_strata": len(s3),
            "x360_median": float(np.median(s3)), "x360_mean": float(s3.mean()),
            "mech_median": float(np.median(sm)), "mech_mean": float(sm.mean())}

    print("\n" + "=" * 84)
    print("[3] 가장 큰 단일 층에서 receiver_first 구간별로")
    print("=" * 84)
    big = max((k for k in strata if len(strata[k]) >= 100),
              key=lambda k: len(strata[k]), default=None)
    if big is not None:
        idx = np.array(strata[big])
        print(f"  층 크기 {len(idx):,} · 거리 {DIST[idx].mean():.1f}m · "
              f"수직거리 {PERP[idx].mean():.1f}m · 목표최근접 {TNO[idx].mean():.1f}m")
        print(f"  {'receiver_first':<20}{'n':>8}{'xpass360':>11}{'역학':>10}")
        for lo, hi in ((0, .2), (.2, .4), (.4, .6), (.6, .8), (.8, 1.01)):
            m = (RF[idx] >= lo) & (RF[idx] < hi)
            if m.sum() < 10:
                continue
            print(f"  [{lo:.1f}, {hi:.1f})          {m.sum():>8,}"
                  f"{P360[idx][m].mean():>11.3f}{MECH[idx][m].mean():>10.3f}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(out, ensure_ascii=False, indent=1))
        print(f"\nsaved {args.output}")


if __name__ == "__main__":
    main()
