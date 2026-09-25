#!/usr/bin/env python3
"""Does "the runner is heading somewhere dangerous" pick better scenes?

The shortlist is ordered by how much the attack is worth under the
defender's best reply, and Chani's strong scenes sit in the bottom decile of
exactly that. The proposal here is a different quantity: rank by the threat
of the ground the RUNNER moves into, not by the value of the resulting
attack.

It is a genuinely different signal from the ones already tested.
`vacated_xt` scores the space the DEFENDER gives up; this scores where the
runner goes, whether or not any defender vacates anything.

Four forms, because "runs into a dangerous area" is ambiguous:

  xt_end     threat at the runner's position three seconds after onset
  xt_gain    that minus the threat where he started -- the run's progress
  xt_max     the best threat he passes through at any point
  xt_speed   gain divided by the time taken, i.e. how fast he is improving

Scored the same way as before: AUC against Chani's strong and medium
phases. Same caveat as before too -- a phase holds several triples and she
marked one moment, so the positive class carries noise that pushes every
AUC toward 0.5. A value near 0.5 is weak evidence of nothing; a value far
from it is worth acting on.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

from offball_value.obso import score_at_points


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--build", type=Path,
                   default=Path("/work/hdd/bbmr/kseo1/offball-out/v7_r9_ssac"))
    p.add_argument("--runner-score", type=Path,
                   default=Path("data/processed/chani_runner_score.csv"))
    p.add_argument("--output", type=Path,
                   default=Path("data/processed/runner_destination_xt.csv"))
    return p.parse_args()


def auc(scores: np.ndarray, positive: np.ndarray) -> float:
    pos, neg = scores[positive], scores[~positive]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    values = np.concatenate([pos, neg])
    order = np.argsort(values, kind="stable")
    raw = np.empty(order.size, dtype=float)
    raw[order] = np.arange(1, order.size + 1)
    ranks = pd.DataFrame({"v": values, "r": raw}).groupby("v")["r"].transform("mean")
    ranks = ranks.to_numpy()
    return float((ranks[:pos.size].sum() - pos.size * (pos.size + 1) / 2)
                 / (pos.size * neg.size))


def main() -> None:
    args = parse_args()
    rows = []
    for path in sorted(glob.glob(
            f"{args.build}/scene_*/local_game_payoff_audits.json")):
        try:
            payload = json.load(open(path))
        except Exception:
            continue
        if not payload:
            continue
        g = payload[0]
        runner = str(g["runner_id"])
        direction = int(g["attacking_direction"])
        frames = g["background_frames"]
        onset = g["onset_frame_id"]

        track = []
        for f in frames:
            for p in f["players"]:
                if str(p[0]) == runner:
                    track.append((float(f["relative_time_s"]),
                                  float(p[2]), float(p[3])))
                    break
        if len(track) < 3:
            continue
        onset_t = min(track, key=lambda r: abs(
            r[0] - next(float(f["relative_time_s"]) for f in frames
                        if f["frame_id"] == onset)))
        after = [r for r in track if r[0] >= onset_t[0]]
        if len(after) < 2:
            continue
        points = [(r[1], r[2]) for r in after]
        xt = np.asarray(score_at_points(points, direction), dtype=float)
        start, end = float(xt[0]), float(xt[-1])
        span = max(after[-1][0] - after[0][0], 1e-6)
        rows.append({
            "match_id": str(g["match_id"]),
            "onset_frame_id": int(g["onset_frame_id"]),
            "xt_end": end,
            "xt_gain": end - start,
            "xt_max": float(xt.max()),
            "xt_speed": (end - start) / span,
        })

    frame = pd.DataFrame(rows).drop_duplicates(["match_id", "onset_frame_id"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(f"장면 {len(frame)}개 계산 → {args.output}\n")

    cols = ["xt_end", "xt_gain", "xt_max", "xt_speed"]
    print("분포")
    print(frame[cols].describe(percentiles=[.5, .9]).round(4).to_string())

    runner_score = pd.read_csv(args.runner_score)
    for level in ("strong", "medium"):
        clips = runner_score[runner_score["effect"] == level]
        mask = np.zeros(len(frame), dtype=bool)
        for r in clips.itertuples():
            if pd.isna(getattr(r, "phase_start", None)):
                continue
            mask |= (
                (frame["match_id"] == r.match_id).to_numpy()
                & (frame["onset_frame_id"] >= float(r.phase_start)).to_numpy()
                & (frame["onset_frame_id"] <= float(r.phase_end)).to_numpy()
            )
        print(f"\n=== {level}: 전문가 국면 장면 {int(mask.sum())} / {len(frame)} ===")
        if mask.sum() < 3:
            print("  표본 부족")
            continue
        for column in cols:
            values = frame[column].to_numpy(dtype=float)
            ok = np.isfinite(values)
            value = auc(values[ok], mask[ok])
            if abs(value - 0.5) >= 0.15:
                note = "신호" if value > 0.5 else "신호 (뒤집어야)"
            elif abs(value - 0.5) >= 0.08:
                note = "약함"
            else:
                note = "무관"
            print(f"  {column:12} AUC {value:6.3f}   {note}")

    print("\n비교 기준 (같은 방식으로 이미 측정한 값)")
    print("  strong 에서 minimax_worst_q  AUC 0.247  (뒤집힌 신호)")
    print("  strong 에서 vacated_xt       AUC 0.231  (뒤집힌 신호)")
    print("  medium 에서는 전 지표가 0.47~0.54 로 무관했습니다.")


if __name__ == "__main__":
    main()
