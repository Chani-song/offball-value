#!/usr/bin/env python3
"""R10 prototype: integrate threat over the vacated space itself.

R9 couples the clocks but not the places. Its occupation term is a 10 m soft
coverage field at the player's position; its Q is priced at the cell's own
event point, which measurement put a median 13-18 m away, with the priced
point scoring 0.041 (pass) and 0.031 (carry) on the normalised vacated field.
The two factors describe different locations, and G barely varies across seven
discrete candidates -- removing it changed none of 21 answers.

So price the space instead of a point:

    score(i) = SUM_t SUM_z  loss(t,z) . w_i(z,t) . P_i(z,t) . G(z)
                            ------------------------------------
                                   SUM_t SUM_z loss(t,z)

    loss(t,z)  how much coverage the defender gave up at that cell
    w_i(z,t)   share of i's influence sitting on that cell, normalised over
               his WHOLE field so it reads as "how much of this player's
               presence is in the vacated space" and cannot be inflated by a
               larger region
    P_i(z,t)   probability the ball reaches him there, priced at that cell
    G(z)       static threat of that cell

WHAT THIS IS NOT: an expected value. A receiver is served once, so summing
over cells and instants is an aggregate, not an expectation, and the same is
true of R9. It is a ranking score and should be called one.

KNOWN DOUBLE COUNT, left in deliberately so the ablations can show its size:
w uses velocity-oriented influence and P uses receiver arrival time, so the
receiver's velocity enters twice. The `--no-weight` arm drops w entirely and
is the control for it.

Carry options are scored on the pass machinery here, which is WRONG for them --
there is no point-wise "can he dribble there and keep it" function yet. Carry
picks from this prototype mean little; that gap is the next piece of work.

Usage:
    python scripts/prototype_integrated_beneficiary.py --audits <dir> ...
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from offball_value.assignment_rule import attacking_team_id, onset_state  # noqa: E402
from offball_value.carry_dynamics import carry_point_estimate  # noqa: E402
from offball_value.coupled_beneficiary import (  # noqa: E402
    SAMPLE_STEP_SECONDS,
    rule_r9,
    value_at_times,
)
from offball_value.obso import score_at_points  # noqa: E402
from offball_value.point_value import build_value_field, point_value  # noqa: E402
from offball_value.pass_dynamics import (  # noqa: E402
    ArrivalModelConfig,
    VelocityEstimate,
    point_reception_estimate,
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

LOSS_FLOOR = 0.05          # share of the instant's peak loss a cell must carry
MAX_CELLS = 120            # per instant, the strongest vacated cells


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--audits", type=Path, nargs="+", required=True)
    p.add_argument("--max-pairs", type=int, default=None)
    p.add_argument("--carry-speed-ratio", type=float, default=0.94,
                   help="measured by scripts/measure_carry_speed_paired.py")
    p.add_argument("--value-mode", choices=("xt", "sum", "mean"), default="xt",
                   help="xt: static grid at the point. sum/mean: threat of the "
                        "space the player would command there, defenders in.")
    return p.parse_args()


def frame_at(scene, time_s):
    """Nearest background frame as (players dict, ball xy)."""
    best = min(scene["background_frames"],
               key=lambda fr: abs(float(fr["relative_time_s"]) - time_s))
    players = {str(r[0]): (str(r[1]), float(r[2]), float(r[3])) for r in best["players"]}
    ball = best.get("ball")
    return players, ((float(ball[0]), float(ball[1])) if ball else None)


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

    arrival = ArrivalModelConfig()
    from offball_value.bundesliga import BundesligaFrame, BundesligaObjectState

    tally = {"R9": 0, "적분 전부": 0, "적분 G제거": 0, "적분 w제거": 0}
    rows = []
    started = time.time()
    pairs = [(k, v) for k, v in sorted(labels.items()) if k[1] != "57121"]
    if args.max_pairs:
        pairs = pairs[: args.max_pairs]

    for (match_id, frame_id, defender_id), answer in pairs:
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
        team = attacking_team_id(scene)
        direction = int(scene["attacking_direction"])
        options = [oid for oid, cell in response["cells"].items()
                   if cell.get("legal") is not False and cell.get("q") is not None
                   and oid != runner]
        if len(options) < 2:
            continue

        # R9, for reference on the same scene
        curves = {oid: value_at_times(response["cells"][oid].get("candidate_grid") or [])
                  for oid in options}
        r9_pick, _ = rule_r9(state, team, runner, str(scene["carrier_id"]), defender_id,
                             defender_at, attacker_at, curves, float(scene["horizon_seconds"]))

        xs, ys = _region_grid(state[defender_id])
        gx, gy = np.meshgrid(xs, ys)
        others = np.zeros((len(ys), len(xs)))
        for pid, row in state.items():
            if str(row["team"]) == team or pid == defender_id:
                continue
            np.maximum(others, coverage_field(_state_xy(row), _state_v(row), xs, ys), out=others)
        before = np.maximum(others, coverage_field(
            _state_xy(state[defender_id]), _state_v(state[defender_id]), xs, ys))

        totals = {m: {oid: 0.0 for oid in options} for m in ("full", "noG", "noW")}
        weight_total = 0.0
        horizon = float(scene["horizon_seconds"])
        t = SAMPLE_STEP_SECONDS
        while t <= horizon + 1e-9:
            axy, av = defender_at(t)
            loss = np.clip(before - np.maximum(others, coverage_field(axy, av, xs, ys)),
                           0.0, None)
            peak = float(loss.max())
            if peak <= 1e-9:
                t = round(t + SAMPLE_STEP_SECONDS, 6)
                continue
            mask = loss >= LOSS_FLOOR * peak
            idx = np.argwhere(mask)
            if len(idx) > MAX_CELLS:
                order = np.argsort(loss[mask])[::-1][:MAX_CELLS]
                idx = idx[order]
            cells = np.array([[gx[i, j], gy[i, j]] for i, j in idx], dtype=float)
            losses = np.array([loss[i, j] for i, j in idx], dtype=float)
            weight_total += float(losses.sum())
            goal = np.asarray(score_at_points(cells, direction), dtype=float)

            players, ball = frame_at(scene, t)
            if ball is None:
                t = round(t + SAMPLE_STEP_SECONDS, 6)
                continue
            snapshot = {}
            velocities = {}
            for pid, (tm, px, py) in players.items():
                snapshot[pid] = BundesligaObjectState(pid, tm, px, py)
                xy, vel = attacker_at(pid, t) if pid in state else ((px, py), (0.0, 0.0))
                velocities[pid] = VelocityEstimate(vel[0], vel[1],
                                                   float(np.hypot(*vel)), 2, 0.2)
            bframe = BundesligaFrame(str(scene["match_id"]), 0, 1, "", None, snapshot,
                                     BundesligaObjectState("ball", "", ball[0], ball[1]))
            vfield = None
            if args.value_mode != "xt":
                vfield = build_value_field(
                    [((float(p.x), float(p.y)),
                      (velocities[pid].vx, velocities[pid].vy))
                     for pid, p in snapshot.items() if p.team_id != team],
                    direction)

            for oid in options:
                if oid not in snapshot:
                    continue
                xy, vel = attacker_at(oid, t)
                field = coverage_field(xy, vel, xs, ys)
                share = field[idx[:, 0], idx[:, 1]] / max(float(field.sum()), 1e-9)
                # The carrier travels with the ball; everyone else is served
                # by one. Two different questions, and until now the carrier was
                # scored on the pass machinery, which cannot express either the
                # turn he must make or the race to where he is going.
                is_carrier = oid == str(scene["carrier_id"])
                try:
                    if is_carrier:
                        probs = np.array([
                            (lambda e: e.carry_probability if e else 0.0)(
                                carry_point_estimate(
                                    bframe, velocities, oid, team,
                                    (float(c[0]), float(c[1])), arrival,
                                    carry_speed_ratio=args.carry_speed_ratio))
                            for c in cells
                        ])
                    else:
                        probs = np.array([
                            point_reception_estimate(
                                bframe, velocities, oid, team, ball,
                                (float(c[0]), float(c[1])), arrival,
                                passer_id=str(scene["carrier_id"])).receive_probability
                            for c in cells
                        ])
                except (ValueError, KeyError):
                    continue
                if vfield is None:
                    value = goal
                else:
                    _pos, vel = attacker_at(oid, t)
                    value = np.array([
                        point_value(vfield, (float(c[0]), float(c[1])), vel,
                                    mode=args.value_mode)
                        for c in cells
                    ])
                totals["full"][oid] += float(np.sum(losses * share * probs * value))
                totals["noG"][oid] += float(np.sum(losses * share * probs))
                totals["noW"][oid] += float(np.sum(losses * probs * value)) / max(len(cells), 1)
            t = round(t + SAMPLE_STEP_SECONDS, 6)

        if weight_total <= 1e-9:
            continue
        picks = {m: max(d, key=lambda k: (d[k], k)) for m, d in totals.items()}
        tally["R9"] += r9_pick == answer
        tally["적분 전부"] += picks["full"] == answer
        tally["적분 G제거"] += picks["noG"] == answer
        tally["적분 w제거"] += picks["noW"] == answer
        rows.append((frame_id, r9_pick == answer, picks["full"] == answer))
        print(f"  {frame_id:<8} R9 {'O' if r9_pick==answer else 'X'} · "
              f"적분 {'O' if picks['full']==answer else 'X'}", flush=True)

    print(f"\nn = {len(rows)} · 소요 {time.time()-started:.0f}초")
    for name, hit in tally.items():
        print(f"  {name:<14}{hit:>4} / {len(rows)}")
    print(f"\n  캐리 속도 비율 {args.carry_speed_ratio:.2f} · 위험도 모드 {args.value_mode}")
    print("  (xt = 정적 격자 · sum = 지배 공간의 위협 합 · mean = 그 평균)")


if __name__ == "__main__":
    main()
