#!/usr/bin/env python3
"""The delivery-model comparison, split by what the cell actually is.

compare_delivery_models_on_audits.py measures how far P moves when the
candidate defender changes his response, and reports hybrid 0.000 against
xpass360 0.130. That population is 54 % `terminal_structure` cells, whose
hybrid delivery is

    0.4 * exp(-terminal_distance / 45)

with no defender term anywhere in it, so its spread is zero by construction
while the 360 model is recomputed on those same cells as if they were real
passes. This script keeps the original grouping and perturbation and adds the
two things the comparison needs to be read: the continuation type of each
cell, and how close the defender can actually get to the ball->target segment.

Carry cells stay excluded for the same reason as the original (their release
rides the carry path, so the 360 model would see a start the hybrid never
had) -- but they are counted, because "the pass model cannot be applied here
at all" is the commensurability finding, not a footnote.

Usage:
    python scripts/compare_delivery_models_split.py \
        --audits data/processed/goalside_v1_chunk{0,1,2} \
        --output out/delivery_split/report.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from offball_value.xpass import (
    lane_features,
    load_xpass_model,
    pass_features,
    to_statsbomb_coordinates,
)

PASS_TYPES = {"through_ball_to_space", "receive_to_feet", "cutback_to_space"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--audits", type=Path, nargs="+", required=True)
    p.add_argument(
        "--model",
        type=Path,
        default=Path("data/processed/xpass_360_nochoice/xpass_360_hist_gbdt.joblib"),
    )
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--min-cells", type=int, default=4)
    return p.parse_args()


def frame_lookup(scene):
    frames = {}
    for frame in scene["background_frames"]:
        players = {
            str(r[0]): (str(r[1]), float(r[2]), float(r[3])) for r in frame["players"]
        }
        ball = frame.get("ball")
        frames[round(float(frame["relative_time_s"]), 2)] = (
            players,
            (float(ball[0]), float(ball[1])) if ball else None,
        )
    return frames


def nearest_time(frames, t):
    return min(frames, key=lambda u: abs(u - t))


def goalkeeper_id(players, attacking_team, direction):
    defending = [(o, v) for o, v in players.items() if v[0] != attacking_team]
    if not defending:
        return None
    sign = 1 if int(direction) >= 0 else -1
    return max(defending, key=lambda item: sign * item[1][1])[0]


def path_position(path_txy, t):
    best = min(path_txy, key=lambda r: abs(float(r[0]) - t))
    return float(best[1]), float(best[2])


def segment_distance(p, a, b):
    p, a, b = np.asarray(p, float), np.asarray(a, float), np.asarray(b, float)
    ab = b - a
    denom = float(ab @ ab)
    if denom < 1e-9:
        return float(np.hypot(*(p - a)))
    u = float(np.clip((p - a) @ ab / denom, 0.0, 1.0))
    return float(np.hypot(*(p - (a + u * ab))))


def collect(scene, model):
    frames = frame_lookup(scene)
    attacking_team = str(scene["attacking_team_id"])
    direction = int(scene["attacking_direction"])
    zero_players, _ = frames[nearest_time(frames, 0.0)]
    keeper = goalkeeper_id(zero_players, attacking_team, direction)

    rows, skipped = [], Counter()
    for defender in scene["candidate_defenders"]:
        defender_id = str(defender["defender_id"])
        carrier_option_ids = {
            str(o["option_id"]) for o in defender["options"] if o.get("is_carrier")
        }
        for response in defender["responses"]:
            path_txy = response.get("path_txy") or []
            for option_id, cell in (response.get("cells") or {}).items():
                if not cell.get("legal") or cell.get("delivery") is None:
                    continue
                ctype = str(cell.get("continuation_type"))
                if option_id in carrier_option_ids:
                    skipped["carrier_option"] += 1
                    continue
                if ctype not in PASS_TYPES:
                    skipped[ctype] += 1
                    continue
                if not path_txy:
                    skipped["no_response_path"] += 1
                    continue
                release = float(cell["release_time_s"])
                target = (float(cell["event_xy"][0]), float(cell["event_xy"][1]))
                players, ball = frames[nearest_time(frames, release)]
                if ball is None:
                    skipped["no_ball"] += 1
                    continue
                defender_xy = path_position(path_txy, release)
                opponents = []
                for oid, (team, x, y) in players.items():
                    if team == attacking_team or oid == keeper:
                        continue
                    if oid == defender_id:
                        x, y = defender_xy
                    opponents.append((x, y))
                rows.append(
                    {
                        "scene": f"{scene['match_id']}:{scene['onset_frame_id']}",
                        "defender": defender_id,
                        "option": option_id,
                        "response": str(response["response_id"]),
                        "ctype": ctype,
                        "hybrid": float(cell["delivery"]),
                        "lane_distance_m": segment_distance(defender_xy, ball, target),
                        "start": ball,
                        "end": target,
                        "opponents": opponents,
                    }
                )
    if not rows:
        return rows, skipped

    features = np.stack(
        [
            np.concatenate(
                [
                    pass_features(
                        to_statsbomb_coordinates(r["start"], direction),
                        to_statsbomb_coordinates(r["end"], direction),
                        "Ground Pass",
                        False,
                    ),
                    lane_features(
                        to_statsbomb_coordinates(r["start"], direction),
                        to_statsbomb_coordinates(r["end"], direction),
                        [to_statsbomb_coordinates(xy, direction) for xy in r["opponents"]],
                    ),
                ]
            )
            for r in rows
        ]
    )
    features_full = features
    columns = getattr(model, "_offball_feature_columns", None)
    if columns is not None:
        features = features[:, columns]
    for r, v, f in zip(rows, model.predict_proba(features)[:, 1], features_full):
        r["x360"] = float(v)
        r["_lane"] = f[-6:]
        del r["opponents"]
    return rows, skipped


def summarize(values):
    a = np.asarray(values, float)
    return {
        "n": int(a.size),
        "median": float(np.median(a)),
        "mean": float(a.mean()),
        "p75": float(np.percentile(a, 75)),
        "p90": float(np.percentile(a, 90)),
        "flat_under_1pp": float(np.mean(a < 0.01)),
    }


def report(rows, skipped, min_cells):
    groups = defaultdict(list)
    for r in rows:
        groups[(r["scene"], r["defender"], r["option"])].append(r)
    kept = {k: v for k, v in groups.items() if len(v) >= min_cells}

    print("=" * 86)
    print("실제 패스 셀만으로 다시 잰 배달 모델 비교")
    print("=" * 86)
    print(f"\n패스 셀 {len(rows):,}개 · 그룹 {len(kept):,}개 (셀 {min_cells}개 이상)")
    print("\n제외된 셀 (원 감사는 이 중 terminal_structure 를 포함했다)")
    for key, n in skipped.most_common():
        print(f"  {key:<28}{n:>9,}")

    out = {"skipped": dict(skipped), "groups": len(kept), "cells": len(rows)}

    print("\n[1] 수비수가 대응을 바꿀 때 P 변동폭 — 경로 접근 가능 거리별")
    print(f"  {'':<26}{'그룹':>7}{'하이중앙':>10}{'360중앙':>10}{'하이평균':>10}{'360평균':>10}")
    bands = [(0.0, 2.0, "경로 2m 이내"), (0.0, 3.0, "경로 3m 이내"),
             (0.0, 5.0, "경로 5m 이내"), (10.0, 1e9, "경로 10m 밖(대조군)"),
             (0.0, 1e9, "전체")]
    out["bands"] = {}
    for lo, hi, label in bands:
        sel = [v for v in kept.values()
               if lo <= min(c["lane_distance_m"] for c in v) < hi]
        if not sel:
            continue
        h = [max(c["hybrid"] for c in v) - min(c["hybrid"] for c in v) for v in sel]
        x = [max(c["x360"] for c in v) - min(c["x360"] for c in v) for v in sel]
        print(f"  {label:<24}{len(sel):>7,}{np.median(h):>10.3f}{np.median(x):>10.3f}"
              f"{np.mean(h):>10.3f}{np.mean(x):>10.3f}")
        out["bands"][label] = {"hybrid": summarize(h), "x360": summarize(x)}

    print("\n[2] P 수준")
    out["levels"] = {}
    for name, label in (("hybrid", "하이브리드"), ("x360", "xpass360")):
        v = np.array([r[name] for r in rows])
        print(f"  {label:<12} 중앙 {np.median(v):.3f} · 평균 {v.mean():.3f}"
              f" · >0.9 {np.mean(v > 0.9):.1%}")
        out["levels"][name] = {"median": float(np.median(v)), "mean": float(v.mean()),
                               "above_0_9": float(np.mean(v > 0.9))}
    c = float(np.corrcoef([r["hybrid"] for r in rows], [r["x360"] for r in rows])[0, 1])
    print(f"  상관 {c:.3f}")
    out["correlation"] = c

    print("\n[3] 부호 검사: 수비수가 경로에서 멀수록 P가 높아야 한다")
    print(f"  {'':<26}{'쌍':>9}{'하이':>9}{'360':>9}")
    hh = xx = tot = 0
    for v in kept.values():
        for i in range(len(v)):
            for j in range(i + 1, len(v)):
                a, b = v[i], v[j]
                d = a["lane_distance_m"] - b["lane_distance_m"]
                if abs(d) < 0.5:
                    continue
                tot += 1
                far, near = (a, b) if d > 0 else (b, a)
                hh += far["hybrid"] >= near["hybrid"]
                xx += far["x360"] >= near["x360"]
    if tot:
        print(f"  {'부호 일치율':<24}{tot:>9,}{hh/tot:>9.1%}{xx/tot:>9.1%}")
        out["sign_agreement"] = {"pairs": tot, "hybrid": hh / tot, "x360": xx / tot}

    print("\n[4] 압축: 같은 셀에서 하이브리드 P 구간별로 360 이 얼마나 들어올리나")
    h = np.array([r["hybrid"] for r in rows])
    x = np.array([r["x360"] for r in rows])
    print(f"  {'P하이 구간':<14}{'n':>9}{'하이평균':>10}{'360평균':>10}{'배율':>8}")
    out["uplift_buckets"] = []
    edges = np.arange(0.0, 1.01, 0.1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (h >= lo) & (h < hi)
        if m.sum() < 20:
            continue
        ratio = float(x[m].mean() / max(h[m].mean(), 1e-9))
        out["uplift_buckets"].append(
            {"lo": float(lo), "hi": float(hi), "n": int(m.sum()),
             "hybrid_mean": float(h[m].mean()), "x360_mean": float(x[m].mean()),
             "uplift": ratio}
        )
        print(f"  [{lo:.1f},{hi:.1f})      {m.sum():>9,}{h[m].mean():>10.3f}"
              f"{x[m].mean():>10.3f}{ratio:>8.2f}")

    print("\n[4b] 우리 옵션 카탈로그의 레인 특징 분포 (StatsBomb 실제 패스와 비교용)")
    from offball_value.xpass import FEATURE_NAMES_360, lane_features, to_statsbomb_coordinates
    li = FEATURE_NAMES_360.index("lane_blockers_2m") - len(FEATURE_NAMES_360) + 6
    lanes = np.stack([r["_lane"] for r in rows if "_lane" in r]) if any("_lane" in r for r in rows) else None
    if lanes is not None:
        names = ["passer_nearest_opponent", "lane_min_perpendicular", "lane_blockers_2m",
                 "lane_blockers_4m", "target_nearest_opponent", "visible_opponents"]
        print(f"  {'특징':<26}{'중앙':>9}{'평균':>9}{'10%':>9}{'90%':>9}")
        out["our_lane_features"] = {}
        for k, nm in enumerate(names):
            v = lanes[:, k]
            out["our_lane_features"][nm] = {
                "median": float(np.median(v)), "mean": float(v.mean()),
                "p10": float(np.percentile(v, 10)), "p90": float(np.percentile(v, 90))}
            print(f"  {nm:<26}{np.median(v):>9.2f}{v.mean():>9.2f}"
                  f"{np.percentile(v,10):>9.2f}{np.percentile(v,90):>9.2f}")
        b2 = lanes[:, 2]
        print(f"\n  경로 2m 안 수비수 0명인 셀 비율: {np.mean(b2 < 1):.1%}"
              f"  (StatsBomb 실제 패스는 70.0%)")
        out["our_zero_blocker_fraction"] = float(np.mean(b2 < 1))

    print("\n[5] 한 응답 안에서 옵션들을 가르는 힘 (argmax 가 쓰는 것)")
    by_resp = defaultdict(list)
    for r in rows:
        by_resp[(r["scene"], r["defender"], r["response"])].append(r)
    hs, xs, hl, xl = [], [], [], []
    for group in by_resp.values():
        if len(group) < 3:
            continue
        a = np.array([g["hybrid"] for g in group])
        b = np.array([g["x360"] for g in group])
        hs.append(a.max() - a.min())
        xs.append(b.max() - b.min())
        hl.append(float(np.std(np.log(np.maximum(a, 1e-4)))))
        xl.append(float(np.std(np.log(np.maximum(b, 1e-4)))))
    if hs:
        hs, xs, hl, xl = map(np.array, (hs, xs, hl, xl))
        out["option_spread"] = {
            "responses": int(hs.size),
            "hybrid_range_median": float(np.median(hs)),
            "x360_range_median": float(np.median(xs)),
            "hybrid_logstd_median": float(np.median(hl)),
            "x360_logstd_median": float(np.median(xl)),
            "shrunk_fraction": float(np.mean(xl < hl)),
        }
        print(f"  응답 {hs.size:,}개 (옵션 3개 이상)")
        print(f"  {'':<22}{'하이':>10}{'360':>10}")
        print(f"  {'P 최대-최소 중앙':<20}{np.median(hs):>10.3f}{np.median(xs):>10.3f}")
        print(f"  {'log P 표준편차 중앙':<20}{np.median(hl):>10.3f}{np.median(xl):>10.3f}")
        print(f"\n  360 에서 log 퍼짐이 줄어든 응답 비율: {np.mean(xl < hl):.1%}")
    return out



def main() -> None:
    args = parse_args()
    model = load_xpass_model(args.model)
    rows, skipped = [], Counter()
    for path in args.audits:
        payload = json.loads(
            (path / "local_game_payoff_audits.json").read_text(encoding="utf-8")
        )
        for scene in payload:
            r, s = collect(scene, model)
            rows.extend(r)
            skipped.update(s)
        print(f"[{path.name}] 누적 패스 셀 {len(rows):,}", flush=True)
        del payload
    out = report(rows, skipped, args.min_cells)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(out, ensure_ascii=False, indent=1))
        print(f"\nsaved {args.output}")


if __name__ == "__main__":
    main()
