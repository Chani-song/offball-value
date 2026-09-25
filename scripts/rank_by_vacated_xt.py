#!/usr/bin/env python3
"""Rank triples by the PAUSA threat of the space the defender vacates, and sanity-check.

`second_best_q` ranks by what the ATTACK is worth when the defender commits.
This ranks by what the defender GIVES UP: take the coverage he drops as he
chases the runner, weight each point of it by the PAUSA EPV surface, and sum.
A defender pulled out of a worthless corner scores low however far he runs; one
pulled off the penalty spot scores high even if he barely moves.

The two need not agree, and if they disagree the disagreement is the finding,
so both are reported side by side rather than blended.

Both are then checked against Chani's strong/medium. This is hygiene, not
selection: the criteria come from mechanism, and 88 single-reviewer labels with
a known role shift between her reading and ours cannot choose between them. A
criterion that lands near chance is suspect; one that beats chance is not
thereby proven.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from offball_value.assignment_rule import onset_state
from offball_value.bundesliga import FPS
from offball_value.obso import score_at_points
from offball_value.vacated_space import (
    _region_grid,
    _state_v,
    _state_xy,
    coverage_field,
)


def vacated_threat(game: dict, defender: dict) -> tuple[float, float, float]:
    """(threat-weighted vacated mass, raw vacated mass, mean threat over it)."""
    state = onset_state(game)
    did = str(defender["defender_id"])
    if did not in state:
        return (np.nan,) * 3
    attack = str(game["attacking_team_id"])
    # 수비수가 러너를 쫓는 경로. 빌드 출력에 runner_response_path_txy 키는 없고,
    # responses 안의 target_conditioned_baseline("focus_runner")이 그 경로다.
    chase = ()
    for resp in defender.get("responses") or ():
        if str(resp.get("response_id")) == "focus_runner":
            chase = resp.get("path_txy") or ()
            break
    if not chase:
        for resp in defender.get("responses") or ():
            if str(resp.get("kind")) == "target_conditioned_baseline":
                chase = resp.get("path_txy") or ()
                break
    if not chase or len(chase) < 2:
        return (np.nan,) * 3
    xs, ys = _region_grid(state[did])
    others = np.zeros((len(ys), len(xs)))
    for pid, row in state.items():
        if str(row["team"]) == attack or pid == did:
            continue
        np.maximum(others, coverage_field(_state_xy(row), _state_v(row), xs, ys), out=others)
    before = np.maximum(
        others, coverage_field(_state_xy(state[did]), _state_v(state[did]), xs, ys)
    )
    end = [float(v) for v in chase[-1]]
    end_xy = (end[1], end[2])
    prev = [float(v) for v in chase[max(0, len(chase) - 2)]]
    dt = max(end[0] - prev[0], 1e-3)
    end_v = ((end[1] - prev[1]) / dt, (end[2] - prev[2]) / dt)
    after = np.maximum(others, coverage_field(end_xy, end_v, xs, ys))
    loss = np.clip(before - after, 0.0, None)

    gx, gy = np.meshgrid(xs, ys)
    pts = np.column_stack([gx.ravel(), gy.ravel()])
    xt = score_at_points(pts, int(game["attacking_direction"])).reshape(loss.shape)
    mass = float(loss.sum())
    weighted = float((loss * xt).sum())
    return weighted, mass, (weighted / mass if mass > 1e-9 else 0.0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--build", type=Path,
                   default=Path("/work/hdd/bbmr/kseo1/offball-out/v5_hybrid"))
    p.add_argument("--ranking", type=Path, default=Path("data/processed/triple_ranking.csv"))
    p.add_argument("--output", type=Path, default=Path("data/processed/triple_ranking_xt.csv"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    paths = sorted(args.build.glob("scene_*/local_game_payoff_audits.json"))
    print(f"빌드 {len(paths)}개", flush=True)
    for i, path in enumerate(paths):
        if i % 100 == 0:
            print(f"  {i}/{len(paths)}", flush=True)
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        for game in payload:
            for d in game.get("candidate_defenders") or []:
                w, mass, mean = vacated_threat(game, d)
                rows.append({
                    "match_id": game["match_id"],
                    "onset_frame_id": game["onset_frame_id"],
                    "runner_id": str(game["runner_id"]),
                    "defender_id": str(d["defender_id"]),
                    "vacated_xt": w,
                    "vacated_mass": mass,
                    "vacated_mean_xt": mean,
                })
    xt = pd.DataFrame(rows)
    base = pd.read_csv(args.ranking)
    frame = base.merge(xt, on=["match_id", "onset_frame_id", "runner_id", "defender_id"], how="left")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(f"\n삼중항 {len(frame)}개, vacated_xt 계산됨 {frame['vacated_xt'].notna().sum()}개\n")

    print("=" * 74); print("vacated_xt 분포"); print("=" * 74)
    print(frame[["vacated_xt", "vacated_mass", "vacated_mean_xt"]]
          .describe(percentiles=[.5, .75, .9, .95, .99]).round(4).to_string())

    print("\n" + "=" * 74); print("기존 기준과의 상관"); print("=" * 74)
    cols = ["vacated_xt", "vacated_mass", "vacated_mean_xt",
            "second_best_q", "minimax_worst_q", "top_q", "option_spread"]
    print(frame[cols].corr().round(3).to_string())

    print("\n" + "=" * 74); print("상위 5%가 서로 겹치는가"); print("=" * 74)
    q = 0.05
    sets = {}
    for col in ("vacated_xt", "second_best_q", "minimax_worst_q"):
        thr = frame[col].quantile(1 - q)
        sets[col] = set(frame.index[frame[col] >= thr])
    for a in sets:
        for b in sets:
            if a < b:
                print(f"  {a:18} ∩ {b:18} : {len(sets[a] & sets[b]):>3}/{len(sets[a]):<3}"
                      f"  ({len(sets[a] & sets[b]) / len(sets[a]):.0%})")


if __name__ == "__main__":
    main()
