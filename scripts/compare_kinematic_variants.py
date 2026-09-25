#!/usr/bin/env python3
"""Do the retrained variants stop overshooting on the data we actually use?

The first kinematic model predicted 0.922 on real Bundesliga passes that
complete 0.807 -- worse than the static xpass360 it was meant to improve, and
worse than the base rate in Brier. Three suspects were separated:

  height_ground/low/high and under_pressure  the pipeline cannot supply either
      and sends "Ground Pass", not under_pressure, for every counterfactual
      pass. A single-feature audit put height_high alone at AUC 0.964, so the
      model leans hardest on a constant. The shipped xpass360 drops all four;
      the first kinematic build put them back by accident.
  women's competitions  26.5 % of the 360 passes, completing 0.784 against
      0.868, while these scenes are men's 2. Bundesliga.
  international tournaments  68 % of the remaining matches, at a different
      level from club football.

Each variant is scored here against the OBSERVED outcome on tracking, which is
the only test that matters: AUC inside StatsBomb says nothing about whether the
number transfers.

Usage:
    python scripts/compare_kinematic_variants.py
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
    "calibrate_hybrid_delivery", ROOT / "scripts" / "calibrate_hybrid_delivery.py"
)
_cal = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cal)

from offball_value.bundesliga import (  # noqa: E402
    find_bundesliga_files,
    infer_attacking_direction,
    load_bundesliga_events,
    load_bundesliga_frame_clock,
    load_bundesliga_frames,
    load_bundesliga_match_metadata,
)
from offball_value.kinematic_xpass import TrackedPlayer  # noqa: E402
from offball_value.observed_passes import (  # noqa: E402
    infer_intended_target,
    resolve_pass_target,
)
from offball_value.pass_dynamics import (  # noqa: E402
    ArrivalModelConfig,
    estimate_frame_velocities,
)
from offball_value.xpass import (  # noqa: E402
    FEATURE_NAMES_360,
    load_xpass_model,
    pass_features_360,
    predict_pass_success_360,
    to_statsbomb_coordinates,
)
from offball_value.kinematic_xpass import (  # noqa: E402
    KINEMATIC_FEATURE_NAMES,
    kinematic_features,
)

HISTORY_FRAMES = 10
FORWARD_FRAMES = 75
FORWARD_STRIDE = 2

VARIANTS = {
    "A 남자전체·feature제외": "data/processed/xpass_360_kin_men",
    "B 클럽리그·feature제외": "data/processed/xpass_360_kin_club",
    "C 옛설정(대조군)": "data/processed/xpass_360_kin_v1",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", type=Path, default=Path("data/raw/bundesliga-integrated"))
    p.add_argument("--output", type=Path,
                   default=Path("out/delivery_analysis/kinematic_variants.json"))
    return p.parse_args()


def load_variant(directory: Path):
    report = json.loads((directory / "training_report.json").read_text(encoding="utf-8"))
    model = load_xpass_model(directory / "xpass_360_kinematic_gbdt.joblib")
    all_names = list(FEATURE_NAMES_360) + list(KINEMATIC_FEATURE_NAMES)
    keep = [all_names.index(n) for n in report["feature_order"]]
    return model, keep, report


def predict(model, keep, ball, target, direction, tracked):
    static = pass_features_360(
        to_statsbomb_coordinates(ball, direction),
        to_statsbomb_coordinates(target, direction),
        "Ground Pass", False,
        [to_statsbomb_coordinates((p.x, p.y), direction)
         for p in tracked if not p.teammate],
    )
    full = np.concatenate([static, kinematic_features(list(tracked), ball, target)])
    return float(np.clip(model.predict_proba(full[keep].reshape(1, -1))[0, 1], 0.0, 1.0))


def main() -> None:
    args = parse_args()
    config = ArrivalModelConfig()
    variants = {}
    for name, path in VARIANTS.items():
        directory = Path(path)
        if (directory / "xpass_360_kinematic_gbdt.joblib").exists():
            variants[name] = load_variant(directory)
            print(f"  불러옴 {name}: feature {len(variants[name][1])}개")
    x360 = load_xpass_model(
        Path("data/processed/xpass_360_nochoice/xpass_360_hist_gbdt.joblib"))

    rows = []
    for match_id in [m for m in _cal.list_bundesliga_match_ids(args.raw_dir)
                     if m not in _cal.EXCLUDED_MATCHES]:
        files = find_bundesliga_files(args.raw_dir, match_id)
        meta = load_bundesliga_match_metadata(files["matchinfo"])
        clock = load_bundesliga_frame_clock(files["positions"])
        events = load_bundesliga_events(files["events"], clock)
        passes = events[
            (events["event_type"] == "pass")
            & (events["from_open_play"] == True)  # noqa: E712
            & events["frame_id"].notna()
            & events["player_id"].notna() & events["team_id"].notna()
        ].copy()
        if passes.empty:
            continue
        needed = set()
        for fid in passes["frame_id"].astype(int):
            needed.update(range(fid - HISTORY_FRAMES, fid + 1))
            needed.update(range(fid, fid + FORWARD_FRAMES + 1, FORWARD_STRIDE))
        frames = load_bundesliga_frames(files["positions"], needed)
        keepers = {k for t in (meta.home_team_id, meta.away_team_id)
                   if (k := meta.goalkeeper_id(t)) is not None}

        for _, ev in passes.iterrows():
            fid = int(ev["frame_id"])
            frame = frames.get(fid)
            if frame is None or frame.ball is None:
                continue
            passer, team = str(ev["player_id"]), str(ev["team_id"])
            if passer not in frame.players:
                continue
            window = [frames[f] for f in range(fid - HISTORY_FRAMES, fid + 1)
                      if f in frames]
            if len(window) < 3:
                continue
            intended = infer_intended_target(frames, fid, passer, team)
            observed = resolve_pass_target(frames, fid, passer, team,
                                           forward_frames=FORWARD_FRAMES,
                                           stride=FORWARD_STRIDE)
            if intended is None or observed is None or observed.travel_distance_m < 3.0:
                continue
            ball = (float(frame.ball.x), float(frame.ball.y))
            target = intended.target_xy
            direction = infer_attacking_direction(frame, team, meta)
            try:
                velocities = estimate_frame_velocities(window, fid, config)
            except (ValueError, KeyError):
                continue
            tracked = []
            for pid, p in frame.players.items():
                v = velocities.get(pid)
                tracked.append(TrackedPlayer(
                    float(p.x), float(p.y),
                    float(v.vx) if v is not None else 0.0,
                    float(v.vy) if v is not None else 0.0,
                    teammate=str(p.team_id) == team,
                    actor=pid == passer, keeper=pid in keepers))
            row = {"label": 1.0 if observed.completed else 0.0,
                   "distance_m": math.hypot(target[0] - ball[0], target[1] - ball[1])}
            try:
                for name, (model, keep, _r) in variants.items():
                    row[name] = predict(model, keep, ball, target, direction, tracked)
                opponents = [(p.x, p.y) for p in tracked
                             if not p.teammate and not p.keeper]
                row["xpass360 (참고)"] = float(predict_pass_success_360(
                    x360, ball, target, direction, opponents))
            except Exception:
                continue
            rows.append(row)
        print(f"  [{match_id}] 누적 {len(rows):,}", flush=True)
        del frames

    label = np.array([r["label"] for r in rows])
    names = list(variants) + ["xpass360 (참고)"]
    print(f"\n{'='*84}")
    print("실제 Bundesliga 패스에서의 예측 — 관측과 비교")
    print("="*84)
    print(f"  패스 {len(rows):,}개 · 실제 성공률 {label.mean():.4f}\n")
    print(f"  {'모델':<26}{'예측 평균':>11}{'Brier':>10}{'실제 대비':>11}")
    out = {"passes": len(rows), "completion": float(label.mean()), "models": {}}
    for name in names:
        v = np.array([r[name] for r in rows])
        brier = float(np.mean((v - label) ** 2))
        out["models"][name] = {"mean": float(v.mean()), "brier": brier,
                               "gap": float(v.mean() - label.mean())}
        print(f"  {name:<24}{v.mean():>11.3f}{brier:>10.4f}{v.mean()-label.mean():>+11.3f}")
    print(f"  {'(기저율 Brier)':<24}{'':>11}{label.mean()*(1-label.mean()):>10.4f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"\nsaved {args.output}")


if __name__ == "__main__":
    main()
