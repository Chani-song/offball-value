#!/usr/bin/env python3
"""Does the priced event happen IN the space the defender gave up?

R9 multiplies, at one instant, how much of the vacated region a player covers
by the threat of the ball arriving to him:

    score(i) = SUM_t |loss(t)| . occupation_i(t) . Q_i(t) / SUM_t |loss(t)|

The two factors are evaluated at the same TIME -- that was the R6 -> R9 fix --
but not at the same PLACE. Occupation is a soft coverage field with a fixed
10 m radius, so it credits a player for space he is merely near. Q is priced at
the cell's own event point, which for a carry is the carrier's observed path
extrapolated along his CURRENT heading by 2-6 m.

A carrier can therefore bank occupation for a region eight metres to his side
while the only carry the model can price runs forward, somewhere else. Nothing
in the value makes him travel to the space he is being credited for. For a
receiver the ball does that travelling and the delivery term prices it; for a
carrier the model prices keeping the ball, never getting there.

This measures the gap: where the priced event lands relative to the vacated
region, and the angle between where the carrier is heading and where the space
opened, split by action class.

Usage:
    python scripts/measure_vacated_space_reachability.py \
        --audits data/processed/goalside_v1_chunk0 ... \
        --output out/delivery_analysis/vacated_reachability.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from offball_value.assignment_rule import attacking_team_id, onset_state  # noqa: E402
from offball_value.coupled_beneficiary import (  # noqa: E402
    SAMPLE_STEP_SECONDS,
    rule_r9,
    value_at_times,
)
from offball_value.vacated_space import (  # noqa: E402
    _region_grid,
    _state_v,
    _state_xy,
    coverage_field,
)

_spec = importlib.util.spec_from_file_location(
    "score_coupled_beneficiary", ROOT / "scripts" / "score_coupled_beneficiary.py"
)
_coupled = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_coupled)
_scorer = _coupled._scorer

CARRY = {"carrier_carry", "carry_into_space"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--audits", type=Path, nargs="+", required=True)
    p.add_argument("--output", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    labels = _scorer.load_labels()
    scenes = {}
    for directory in args.audits:
        path = directory / "local_game_payoff_audits.json"
        if not path.exists():
            continue
        for scene in json.loads(path.read_text(encoding="utf-8")):
            scenes[(str(scene["match_id"]), str(scene["onset_frame_id"]))] = scene

    records = []
    for (match_id, frame_id, defender_id), answer in sorted(labels.items()):
        scene = scenes.get((match_id, frame_id))
        if scene is None:
            continue
        state = onset_state(scene)
        if defender_id not in state:
            continue
        defender = next((r for r in scene["candidate_defenders"]
                         if str(r["defender_id"]) == defender_id), None)
        if defender is None:
            continue
        response = next((r for r in defender["responses"]
                         if str(r["response_id"]) == str(defender.get("direct_best_response_id"))),
                        None)
        if response is None:
            continue
        defender_at, attacker_at = _coupled.absolute_lookups(scene, defender)
        if defender_at is None:
            continue
        runner = str(scene["runner_id"])
        priced = {k: v for k, v in response["cells"].items()
                  if v.get("legal") is not False and v.get("q") is not None and k != runner}
        curves = {k: value_at_times(v.get("candidate_grid") or []) for k, v in priced.items()}
        pick, _ = rule_r9(state, attacking_team_id(scene), runner, str(scene["carrier_id"]),
                          defender_id, defender_at, attacker_at, curves,
                          float(scene["horizon_seconds"]))

        xs, ys = _region_grid(state[defender_id])
        team = attacking_team_id(scene)
        others = np.zeros((len(ys), len(xs)))
        for pid, row in state.items():
            if str(row["team"]) == team or pid == defender_id:
                continue
            np.maximum(others, coverage_field(_state_xy(row), _state_v(row), xs, ys), out=others)
        before = np.maximum(others, coverage_field(
            _state_xy(state[defender_id]), _state_v(state[defender_id]), xs, ys))
        gx, gy = np.meshgrid(xs, ys)

        for option_id, cell in priced.items():
            ctype = str(cell.get("continuation_type"))
            kind = "carry" if ctype in CARRY else (
                "structural" if ctype == "terminal_structure" else "pass")
            event = cell.get("event_xy")
            if not event:
                continue
            ex, ey = float(event[0]), float(event[1])

            t = SAMPLE_STEP_SECONDS
            horizon = float(scene["horizon_seconds"])
            tot_w = 0.0
            occ_sum = 0.0
            cen_x = cen_y = 0.0
            while t <= horizon + 1e-9:
                axy, av = defender_at(t)
                loss = np.clip(before - np.maximum(
                    others, coverage_field(axy, av, xs, ys)), 0.0, None)
                w = float(loss.sum())
                if w > 1e-12:
                    tot_w += w
                    cen_x += float((loss * gx).sum())
                    cen_y += float((loss * gy).sum())
                    xy, vel = attacker_at(option_id, t)
                    occ_sum += float((loss * coverage_field(xy, vel, xs, ys)).sum())
                t = round(t + SAMPLE_STEP_SECONDS, 6)
            if tot_w <= 1e-12:
                continue
            cx, cy = cen_x / tot_w, cen_y / tot_w

            # Total vacated field over the window, to read the event point off.
            t = SAMPLE_STEP_SECONDS
            acc = np.zeros_like(before)
            while t <= horizon + 1e-9:
                axy, av = defender_at(t)
                acc += np.clip(before - np.maximum(
                    others, coverage_field(axy, av, xs, ys)), 0.0, None)
                t = round(t + SAMPLE_STEP_SECONDS, 6)
            peak = float(acc.max()) or 1.0
            ix = int(np.clip(np.searchsorted(xs, ex) - 1, 0, len(xs) - 1))
            iy = int(np.clip(np.searchsorted(ys, ey) - 1, 0, len(ys) - 1))
            inside = float(acc[iy, ix]) / peak

            start_xy, start_v = attacker_at(option_id, SAMPLE_STEP_SECONDS)
            bearing = math.degrees(math.atan2(cy - start_xy[1], cx - start_xy[0]))
            heading = math.degrees(math.atan2(start_v[1], start_v[0])) \
                if math.hypot(*start_v) > 0.5 else None
            angle = None
            if heading is not None:
                angle = abs((bearing - heading + 180.0) % 360.0 - 180.0)

            records.append({
                "frame": frame_id, "kind": kind,
                "is_pick": option_id == pick, "is_answer": option_id == answer,
                "occupation": occ_sum / tot_w,
                "event_inside_vacated": inside,
                "event_to_centroid_m": math.hypot(ex - cx, ey - cy),
                "player_to_centroid_m": math.hypot(start_xy[0] - cx, start_xy[1] - cy),
                "heading_angle_deg": angle,
            })
        print(f"  {frame_id} 처리", flush=True)

    print(f"\n{'='*84}")
    print("값이 매겨진 지점이 실제로 '비운 공간' 안인가")
    print("="*84)
    out = {"records": len(records), "by_kind": {}}
    groups = defaultdict(list)
    for r in records:
        groups[r["kind"]].append(r)
    print(f"  {'클래스':<12}{'n':>7}{'점유':>9}{'지점이 비운공간 안':>20}"
          f"{'지점-중심 거리':>16}{'선수-중심 거리':>16}")
    for kind in ("pass", "carry", "structural"):
        g = groups.get(kind) or []
        if not g:
            continue
        rec = {
            "n": len(g),
            "occupation": float(np.mean([x["occupation"] for x in g])),
            "inside": float(np.mean([x["event_inside_vacated"] for x in g])),
            "event_dist": float(np.median([x["event_to_centroid_m"] for x in g])),
            "player_dist": float(np.median([x["player_to_centroid_m"] for x in g])),
        }
        out["by_kind"][kind] = rec
        print(f"  {kind:<12}{rec['n']:>7}{rec['occupation']:>9.2f}{rec['inside']:>20.3f}"
              f"{rec['event_dist']:>16.1f}{rec['player_dist']:>16.1f}")

    print("\n  진행 방향과 비운 공간 방향의 각도 (캐리만 — 캐리는 이 방향으로만 뻗을 수 있다)")
    ang = [r["heading_angle_deg"] for r in groups.get("carry", [])
           if r["heading_angle_deg"] is not None]
    if ang:
        a = np.array(ang)
        out["carry_heading_angle"] = {
            "n": int(a.size), "median": float(np.median(a)),
            "within_45": float(np.mean(a <= 45)), "beyond_90": float(np.mean(a > 90))}
        print(f"    n={a.size} · 중앙 {np.median(a):.0f}도")
        print(f"    45도 이내(대략 그쪽으로 가는 중) {np.mean(a <= 45):.1%}")
        print(f"    90도 초과(반대쪽으로 가는 중)   {np.mean(a > 90):.1%}")

    picks = [r for r in records if r["is_pick"]]
    if picks:
        print(f"\n  R9 가 실제로 고른 옵션 {len(picks)}개")
        for kind in ("pass", "carry", "structural"):
            g = [r for r in picks if r["kind"] == kind]
            if g:
                print(f"    {kind:<12}{len(g):>4}개 · 지점이 비운공간 안 "
                      f"{np.mean([x['event_inside_vacated'] for x in g]):.3f}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(out, ensure_ascii=False, indent=1))
        print(f"\nsaved {args.output}")


if __name__ == "__main__":
    main()
