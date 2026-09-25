#!/usr/bin/env python3
"""What does StatsBomb actually say about passing into an occupied lane?

Two numbers have been carried through this investigation without a source:
"96,639 passes" and "the real open-play completion rate is 0.83". Both are
checkable now that the open data is on disk, and the second one matters --
the hybrid's median P was judged against it.

More useful than either: the ground-truth completion curve against lane
occupancy, on the same 360 matches xpass360 was trained on, next to what
xpass360 predicts for those same passes. That is the reference any
recalibration of the delivery model has to hit.

Usage:
    python scripts/measure_statsbomb_lane_truth.py \
        --data-root data/raw/statsbomb-open-data/data \
        --output out/delivery_analysis/statsbomb_lane_truth.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from offball_value.xpass import (
    FEATURE_NAMES_360,
    event_features_and_label,
    is_open_play_pass,
    lane_features,
    load_xpass_model,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path,
                   default=Path("data/raw/statsbomb-open-data/data"))
    p.add_argument("--model", type=Path,
                   default=Path("data/processed/xpass_360_nochoice/"
                                "xpass_360_hist_gbdt.joblib"))
    p.add_argument("--output", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    frames_dir = args.data_root / "three-sixty"
    events_dir = args.data_root / "events"
    matches = sorted(frames_dir.glob("*.json"))
    print(f"three-sixty 파일 {len(matches)}개", flush=True)

    rows, labels = [], []
    n_pass_all = n_pass_with_frame = 0
    matched_files = 0
    for i, frames_path in enumerate(matches):
        events_path = events_dir / frames_path.name
        if not events_path.exists():
            continue
        try:
            frames = {
                str(r["event_uuid"]): r["freeze_frame"]
                for r in json.loads(frames_path.read_text(encoding="utf-8"))
                if r.get("freeze_frame")
            }
            events = json.loads(events_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        matched_files += 1
        for event in events:
            if not is_open_play_pass(event):
                continue
            n_pass_all += 1
            freeze = frames.get(str(event.get("id")))
            if not freeze:
                continue
            n_pass_with_frame += 1
            base, label = event_features_and_label(event)
            opponents = [
                (float(p["location"][0]), float(p["location"][1]))
                for p in freeze if not p.get("teammate", False)
            ]
            start = (float(event["location"][0]), float(event["location"][1]))
            end = tuple(float(v) for v in event["pass"]["end_location"][:2])
            rows.append(np.concatenate([base, lane_features(start, end, opponents)]))
            labels.append(label)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(matches)} · 누적 {len(rows):,}", flush=True)

    X = np.stack(rows)
    y = np.asarray(labels, dtype=float)
    out = {
        "three_sixty_files": len(matches),
        "matched_files": matched_files,
        "open_play_passes_in_those_matches": n_pass_all,
        "open_play_passes_with_freeze_frame": n_pass_with_frame,
        "completion_rate": float(y.mean()),
    }
    print("\n" + "=" * 80)
    print("[1] 모집단")
    print("=" * 80)
    print(f"  360 프레임이 있는 경기            {matched_files:,}")
    print(f"  그 경기의 오픈플레이 패스         {n_pass_all:,}")
    print(f"  그중 freeze_frame 이 붙은 패스    {n_pass_with_frame:,}  <- 학습 모집단")
    print(f"  실제 성공률                      {y.mean():.4f}")
    print(f"  (모델 리포트 train+test = 266,708)")

    col = {name: FEATURE_NAMES_360.index(name) for name in FEATURE_NAMES_360}
    model = load_xpass_model(args.model)
    keep = getattr(model, "_offball_feature_columns", None)
    pred = model.predict_proba(X if keep is None else X[:, keep])[:, 1]
    out["model_mean_prediction"] = float(pred.mean())

    def curve(title, values, edges, fmt="{:.1f}"):
        print(f"\n{title}")
        print(f"  {'구간':<18}{'n':>9}{'실제 성공률':>12}{'xpass360 예측':>14}{'차이':>9}")
        recs = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (values >= lo) & (values < hi)
            if m.sum() < 50:
                continue
            obs, prd = float(y[m].mean()), float(pred[m].mean())
            recs.append({"lo": float(lo), "hi": float(hi), "n": int(m.sum()),
                         "observed": obs, "predicted": prd})
            lab = f"[{fmt.format(lo)}, {fmt.format(hi)})"
            print(f"  {lab:<18}{m.sum():>9,}{obs:>12.3f}{prd:>14.3f}{prd-obs:>+9.3f}")
        return recs

    print("\n" + "=" * 80)
    print("[2] 진짜 성공률 곡선 vs xpass360 예측")
    print("=" * 80)
    out["by_lane_blockers_2m"] = curve(
        "경로 2m 안의 수비수 수", X[:, col["lane_blockers_2m"]],
        [0, 1, 2, 3, 4, 99], "{:.0f}")
    out["by_lane_min_perp"] = curve(
        "경로에서 가장 가까운 수비수까지 수직거리 (m)",
        X[:, col["lane_min_perpendicular"]], [0, 1, 2, 3, 5, 8, 100])
    out["by_length"] = curve(
        "패스 길이 (yd)", X[:, col["length"]], [0, 10, 20, 30, 40, 60, 200], "{:.0f}")

    print("\n" + "=" * 80)
    print("[2b] StatsBomb 실제 패스의 레인 특징 분포 (우리 카탈로그와 비교)")
    print("=" * 80)
    print(f"  {'특징':<26}{'중앙':>9}{'평균':>9}{'10%':>9}{'90%':>9}")
    out["statsbomb_lane_features"] = {}
    for nm in ("passer_nearest_opponent", "lane_min_perpendicular",
               "lane_blockers_2m", "lane_blockers_4m",
               "target_nearest_opponent", "visible_opponents"):
        v = X[:, col[nm]]
        out["statsbomb_lane_features"][nm] = {
            "median": float(np.median(v)), "mean": float(v.mean()),
            "p10": float(np.percentile(v, 10)), "p90": float(np.percentile(v, 90))}
        print(f"  {nm:<26}{np.median(v):>9.2f}{v.mean():>9.2f}"
              f"{np.percentile(v,10):>9.2f}{np.percentile(v,90):>9.2f}")
    b2 = X[:, col["lane_blockers_2m"]]
    print(f"\n  경로 2m 안 수비수 0명 비율: {np.mean(b2 < 1):.1%}")

    tno = X[:, col["target_nearest_opponent"]]
    print(f"\n  목표 지점에서 가장 가까운 수비수까지 거리별 실제 성공률")
    print(f"  {'구간(m)':<16}{'n':>9}{'실제':>9}{'예측':>9}")
    out["by_target_nearest"] = []
    for lo, hi in ((0,2),(2,4),(4,6),(6,9),(9,15),(15,100)):
        m = (tno >= lo) & (tno < hi)
        if m.sum() < 50: continue
        out["by_target_nearest"].append({"lo": lo, "hi": hi, "n": int(m.sum()),
            "observed": float(y[m].mean()), "predicted": float(pred[m].mean())})
        print(f"  [{lo}, {hi})          {m.sum():>9,}{y[m].mean():>9.3f}{pred[m].mean():>9.3f}")

    print("\n" + "=" * 80)
    print("[3] 실제 성공률이 가장 낮은 구간에서 모델이 어디까지 내려가나")
    print("=" * 80)
    worst = X[:, col["lane_blockers_2m"]] >= 3
    if worst.sum() >= 50:
        print(f"  경로 2m 안 수비수 3명 이상 (n={worst.sum():,})")
        print(f"    실제 성공률 {y[worst].mean():.3f} · xpass360 평균 {pred[worst].mean():.3f}")
        print(f"    xpass360 예측 10~90%: {np.percentile(pred[worst],10):.3f}"
              f" ~ {np.percentile(pred[worst],90):.3f}")
        out["worst_lane"] = {"n": int(worst.sum()),
                             "observed": float(y[worst].mean()),
                             "predicted": float(pred[worst].mean())}
    print(f"\n  전체 예측 범위 10~90%: {np.percentile(pred,10):.3f} ~ {np.percentile(pred,90):.3f}")
    print(f"  전체 실제 성공률 {y.mean():.3f} · 예측 평균 {pred.mean():.3f}")
    out["prediction_p10"] = float(np.percentile(pred, 10))
    out["prediction_p90"] = float(np.percentile(pred, 90))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(out, ensure_ascii=False, indent=1))
        print(f"\nsaved {args.output}")


if __name__ == "__main__":
    main()
