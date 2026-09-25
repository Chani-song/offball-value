#!/usr/bin/env python3
"""xPass with lane kinematics -- built so the features survive to inference.

The previous attempt added arrival times at the pass target and did not
transfer. This one adds only quantities anchored to the LANE, which the
pipeline computes identically for a pass that was never played, and excludes
the four features the pipeline cannot supply at all (height x3,
under_pressure), which is why the shipped xpass360 uses 17 of the 21.

Two things are reported besides the usual fit:

  the identity that broke the last attempt -- mate_time_to_target split by
  outcome, which should show completions sitting at ~0 by arithmetic;

  a transfer check on Bundesliga tracking under BOTH labels, this project's
  tracking-derived one and the DFL event file's own Evaluation, which agree
  86 % of the time. A conclusion that holds under only one of them is not a
  conclusion.

League context, since it is easy to misread the transfer number: StatsBomb's
men's passes complete at 0.868 and its 1. Bundesliga at 0.865, while these
scenes are 2. Bundesliga at 0.806 by the tracking label and 0.831 by the DFL
one. Some of any gap is the division, not the model.

Usage:
    python scripts/train_xpass_lane_kinematic.py --output data/processed/xpass_lane_kin
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from offball_value.lane_kinematics import LANE_KINEMATIC_NAMES, lane_kinematics
from offball_value.xpass import (
    FEATURE_NAMES_360,
    event_features_and_label,
    is_open_play_pass,
    lane_features,
)

MAX_GAP_S = 2.0
X_SCALE_M = 105.0 / 120.0
Y_SCALE_M = 68.0 / 80.0
WOMEN = ("Women's World Cup", "UEFA Women's Euro")
# The pipeline sends "Ground Pass", not under_pressure, for every counterfactual
# pass, so a model leaning on these is leaning on a constant. Single-feature
# AUC put height_high at 0.964. The shipped xpass360 drops them for this reason.
UNAVAILABLE = ("height_ground", "height_low", "height_high", "under_pressure")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path,
                   default=Path("data/raw/statsbomb-open-data/data"))
    p.add_argument("--output", type=Path, default=Path("data/processed/xpass_lane_kin"))
    p.add_argument("--include-women", action="store_true")
    return p.parse_args()


def to_m(xy):
    return (float(xy[0]) * X_SCALE_M, float(xy[1]) * Y_SCALE_M)


def auc(pred, label):
    pos, neg = pred[label > 0.5], pred[label <= 0.5]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(order.size, float)
    ranks[order] = np.arange(1, order.size + 1)
    return float((ranks[: pos.size].sum() - pos.size * (pos.size + 1) / 2)
                 / (pos.size * neg.size))


def matched_velocities(now, past, gap_s):
    """Velocities from a freeze-frame pair, re-matched by proximity within side."""
    pools = {True: [], False: []}
    for p in past:
        pools[bool(p.get("teammate"))].append(to_m(p["location"]))
    out = []
    for p in now:
        side = bool(p.get("teammate"))
        here = to_m(p["location"])
        pool = pools[side]
        if not pool:
            out.append((p, here, (0.0, 0.0)))
            continue
        j = min(range(len(pool)),
                key=lambda k: math.hypot(pool[k][0] - here[0], pool[k][1] - here[1]))
        was = pool.pop(j)
        out.append((p, here, ((here[0] - was[0]) / gap_s, (here[1] - was[1]) / gap_s)))
    return out


def main() -> None:
    from sklearn.ensemble import HistGradientBoostingClassifier
    import joblib

    args = parse_args()
    meta_by_id = {}
    for path in sorted((args.data_root / "matches").rglob("*.json")):
        for match in json.loads(path.read_text(encoding="utf-8")):
            meta_by_id[str(match["match_id"])] = match

    static_names = [n for n in FEATURE_NAMES_360 if n not in UNAVAILABLE]
    static_idx = [FEATURE_NAMES_360.index(n) for n in static_names]
    names = static_names + list(LANE_KINEMATIC_NAMES)

    X, y, fold = [], [], []
    mate_time_by_label = {0: [], 1: []}
    matches = sorted((args.data_root / "three-sixty").glob("*.json"))
    print(f"three-sixty 파일 {len(matches)}개")
    if not args.include_women:
        print(f"  여자 대회 제외: {list(WOMEN)}")
    print(f"  제외한 feature: {list(UNAVAILABLE)}")
    print(f"  feature {len(names)}개 = 정적 {len(static_names)} + 경로운동학 "
          f"{len(LANE_KINEMATIC_NAMES)}\n")

    for index, frames_path in enumerate(matches):
        events_path = args.data_root / "events" / frames_path.name
        if not events_path.exists():
            continue
        competition = (meta_by_id.get(frames_path.stem, {})
                       .get("competition", {}).get("competition_name", "?"))
        if not args.include_women and competition in WOMEN:
            continue
        try:
            ff = {str(r["event_uuid"]): r["freeze_frame"]
                  for r in json.loads(frames_path.read_text(encoding="utf-8"))
                  if r.get("freeze_frame")}
            events = json.loads(events_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue

        def secs(e):
            t = e.get("timestamp") or "0:0:0"
            h, m, s = t.split(":")
            return int(e.get("period", 1)) * 10000 + int(h) * 3600 + int(m) * 60 + float(s)

        seq = sorted(((secs(e), e) for e in events if str(e.get("id")) in ff),
                     key=lambda kv: kv[0])
        for i, (t_now, event) in enumerate(seq):
            if not is_open_play_pass(event) or i == 0:
                continue
            gap = t_now - seq[i - 1][0]
            if not (0.05 < gap <= MAX_GAP_S):
                continue
            base, label = event_features_and_label(event)
            start = (float(event["location"][0]), float(event["location"][1]))
            end = tuple(float(v) for v in event["pass"]["end_location"][:2])
            frame = ff[str(event["id"])]
            opponents_units = [(float(p["location"][0]), float(p["location"][1]))
                               for p in frame if not p.get("teammate")]
            static = np.concatenate([base, lane_features(start, end, opponents_units)])
            tracked = matched_velocities(frame, ff[str(seq[i - 1][1]["id"])], gap)
            defenders = [(pos, vel) for p, pos, vel in tracked
                         if not p.get("teammate") and not p.get("keeper")]
            actor = next((vel for p, _pos, vel in tracked if p.get("actor")), (0.0, 0.0))
            lane_kin = lane_kinematics(to_m(start), to_m(end), defenders, actor)
            X.append(np.concatenate([static[static_idx], lane_kin]))
            y.append(label)
            fold.append(index % 5)

            # Diagnostic: the identity that sank the previous feature set.
            mates = [pos for p, pos, _v in tracked
                     if p.get("teammate") and not p.get("actor")]
            if mates:
                em = to_m(end)
                mate_time_by_label[label].append(
                    min(math.hypot(m[0] - em[0], m[1] - em[1]) for m in mates))
        if (index + 1) % 100 == 0:
            print(f"  {index+1}/{len(matches)} · 표본 {len(X):,}", flush=True)

    X = np.stack(X)
    y = np.asarray(y, dtype=float)
    fold = np.asarray(fold)
    n_static = len(static_names)
    print(f"\n학습 표본 {len(X):,} · 성공률 {y.mean():.4f}")

    print(f"\n{'='*80}")
    print("[진단] 옛 feature 가 왜 전이되지 않았나")
    print("="*80)
    for lab, name in ((1, "성공한 패스"), (0, "실패한 패스")):
        d = np.array(mate_time_by_label[lab])
        print(f"  {name}: end_location 에서 가장 가까운 아군까지 거리 "
              f"중앙 {np.median(d):.2f} m · 1m 이내 {np.mean(d < 1.0):.1%}")
    print("  성공한 패스의 end_location 은 정의상 아군 발밑이다. 거기까지의")
    print("  '아군 도착 시간'은 축구가 아니라 산술이고, 목표가 가상의 점이 되는")
    print("  순간 의미가 사라진다. 경로 기준 feature 는 그 항등식을 갖지 않는다.")

    print(f"\n{'='*80}")
    print("5겹 교차검증 (경기 단위) · 보류 폴드에서만 평가")
    print("="*80)
    print(f"  {'':<30}{'AUC':>9}{'Brier':>10}{'폴드 표준편차':>15}")
    results = {}
    for name, cols in (("정적만 (현행 xpass360)", list(range(n_static))),
                       ("정적 + 경로운동학", list(range(X.shape[1])))):
        pred = np.zeros(len(X))
        per = []
        for f in sorted(set(fold)):
            fit, use = fold != f, fold == f
            model = HistGradientBoostingClassifier(
                max_depth=None, max_iter=300, learning_rate=0.08,
                min_samples_leaf=50, l2_regularization=1.0, random_state=0)
            model.fit(X[np.ix_(fit, cols)], y[fit])
            pred[use] = model.predict_proba(X[np.ix_(use, cols)])[:, 1]
            per.append(auc(pred[use], y[use]))
        a, b = auc(pred, y), float(np.mean((pred - y) ** 2))
        results[name] = {"auc": a, "brier": b, "fold_auc": per}
        print(f"  {name:<28}{a:>9.4f}{b:>10.4f}{np.std(per):>15.4f}")
    print(f"\n  기저율 Brier {y.mean()*(1-y.mean()):.4f}")
    gain = results["정적 + 경로운동학"]["auc"] - results["정적만 (현행 xpass360)"]["auc"]
    sd = np.std(results["정적 + 경로운동학"]["fold_auc"])
    print(f"  경로운동학이 더하는 AUC: {gain:+.4f}  (폴드 표준편차 {sd:.4f}"
          f" → {abs(gain)/max(sd,1e-9):.1f}σ)")

    print("\n  경로운동학 feature 단독 AUC")
    for k, name in enumerate(LANE_KINEMATIC_NAMES):
        v = X[:, n_static + k]
        print(f"    {name:<28}{auc(v, y):>8.3f}")

    print("\n전체 데이터로 최종 모델 적합 중 ...", flush=True)
    final = HistGradientBoostingClassifier(
        max_depth=None, max_iter=300, learning_rate=0.08,
        min_samples_leaf=50, l2_regularization=1.0, random_state=0)
    final.fit(X, y)
    args.output.mkdir(parents=True, exist_ok=True)
    joblib.dump(final, args.output / "xpass_lane_kin_gbdt.joblib")
    (args.output / "training_report.json").write_text(json.dumps({
        "feature_order": names, "n_static": n_static,
        "static_names": static_names,
        "lane_kinematic_names": list(LANE_KINEMATIC_NAMES),
        "dropped_unavailable": list(UNAVAILABLE),
        "include_women": bool(args.include_women),
        "samples": int(len(X)), "completion": float(y.mean()),
        "results": results,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
