#!/usr/bin/env python3
"""Which (runner, defender) pairs are worth building a local game for?

The pipeline emits ~1,900 triples and a reviewer's first reaction to many of
them is "why is THIS defender implicated by THIS run?". That is a generation
problem, not a ranking problem: a pair that makes no sense should never
reach stage 3, and no ordering fixes it.

Candidates are currently ranked by goal-side marking distance measured from
the runner at onset, and the top three are always taken. Rank 1 is right
18/18 against the labels; rank 3 sits a median 12.4 m away and reaches 36 m.

What decides responsibility is where the RUN goes, not where the runner
stands at onset. A defender behind the runner can still be the one who
cuts the run off; a defender marking someone else will leave him if the run
passes right by; the man pressing the ball has to turn if the run goes in
behind him. Static gates (goal-side, nearest-attacker rank, presser) get all
three of those wrong, so they are not gates here -- they are reported only
as descriptions of the labelled defenders, to check the counterexamples
actually occur.

The gate measured is the run path itself:

  path_min   closest the runner comes to the defender's onset position at
             any point of the run
  path_end   distance from the defender to where the run finishes
  catch      can the defender reach the runner's position at some moment
             of the run, at 6 m/s after a 0.3 s reaction (metres of slack;
             positive means yes)
  line_min   same as path_min but using the runner's onset velocity
             extrapolated straight, i.e. intent at onset only

The runner's track is the move being responded to, so reading it forward is
legitimate. The defender's own movement is the response and is not used.
"""

from __future__ import annotations

import argparse
import collections
import glob
import importlib.util
import json
import math
from pathlib import Path

import numpy as np

DEF_SPEED = 6.0     # m/s, a defender reacting rather than sprinting flat out
REACTION_S = 0.3
PRESS_RADIUS = 5.0  # a defender this close to the ball is engaged with it


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--build", type=Path,
                   default=Path("/work/hdd/bbmr/kseo1/offball-out/v7_r9_ssac"))
    return p.parse_args()


def load_labels():
    spec = importlib.util.spec_from_file_location(
        "sv", "scripts/score_vacated_space_rule.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.load_labels()


def track(frames, pid, t0, horizon):
    out = []
    for f in frames:
        t = float(f["relative_time_s"]) - t0
        if t < -1e-9 or t > horizon + 1e-9:
            continue
        for p in f["players"]:
            if str(p[0]) == pid:
                out.append((t, float(p[2]), float(p[3])))
                break
    return out


def gather(build: Path) -> list[dict]:
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
        t0 = float(at["relative_time_s"])
        horizon = float(g["horizon_seconds"])
        att = str(g["attacking_team_id"])
        pos = {str(p[0]): (float(p[2]), float(p[3])) for p in at["players"]}
        attackers = [str(p[0]) for p in at["players"] if str(p[1]) == att]
        runner = str(g["runner_id"])
        if runner not in pos:
            continue
        run = track(frames, runner, t0, horizon)
        if len(run) < 3:
            continue
        rx, ry = run[0][1], run[0][2]
        # onset velocity from the first ~0.4 s of the run
        later = next((r for r in run if r[0] >= 0.4), run[-1])
        dt = max(later[0] - run[0][0], 1e-6)
        vx, vy = (later[1] - rx) / dt, (later[2] - ry) / dt
        gx, gy = float(g["goal_xy"][0]), float(g["goal_xy"][1])
        span = math.hypot(gx - rx, gy - ry)
        ux, uy = ((gx - rx) / span, (gy - ry) / span) if span > 1e-9 else (0.0, 0.0)
        bx, by = float(at["ball"][0]), float(at["ball"][1])
        defenders_all = [str(p[0]) for p in at["players"] if str(p[1]) != att]
        ball_d = {d: math.hypot(pos[d][0] - bx, pos[d][1] - by) for d in defenders_all}
        presser = min(ball_d, key=ball_d.get) if ball_d else None
        if presser is not None and ball_d[presser] > PRESS_RADIUS:
            presser = None

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
            d_t = [(t, math.hypot(x - dx, y - dy)) for t, x, y in run]
            path_min = min(dd for _, dd in d_t)
            path_end = d_t[-1][1]
            catch = max(DEF_SPEED * max(t - REACTION_S, 0.0) - dd for t, dd in d_t)
            line_min = min(
                math.hypot(rx + vx * t - dx, ry + vy * t - dy)
                for t in np.linspace(0.0, horizon, 30))
            rows.append({
                "key": (str(g["match_id"]), str(g["onset_frame_id"]), did),
                "scene": (str(g["match_id"]), str(g["onset_frame_id"]), runner),
                "rank": rank,
                "near": dist[runner],
                "path_min": path_min,
                "path_end": path_end,
                "catch": catch,
                "line_min": line_min,
                "own": ordered.index(runner),
                "side": (dx - rx) * ux + (dy - ry) * uy,
                "is_presser": did == presser,
            })
    return rows


def main() -> None:
    args = parse_args()
    rows = gather(args.build)
    labels = set(load_labels())
    n = len(rows)
    scenes = len({r["scene"] for r in rows})
    lab_rows = [r for r in rows if r["key"] in labels]
    base_lab = len({r["key"] for r in lab_rows})
    print(f"후보 {n}개 · 장면 {scenes}개 · 채점 가능한 라벨 {base_lab}개\n")

    print("=" * 72)
    print("라벨된 수비수(전문가가 '이 사람이 반응'이라 한 23명)의 모습")
    print("=" * 72)
    behind = sum(1 for r in lab_rows if r["side"] < 0)
    behind2 = sum(1 for r in lab_rows if r["side"] < -2)
    pressers = sum(1 for r in lab_rows if r["is_presser"])
    notown = sum(1 for r in lab_rows if r["own"] >= 1)
    notown2 = sum(1 for r in lab_rows if r["own"] >= 2)
    print(f"   러너보다 골에서 먼 쪽(뒤)에 선 사람   {behind:2}/{base_lab}  (2m 이상 뒤 {behind2})")
    print(f"   공 압박 중(공에서 {PRESS_RADIUS:.0f}m 이내 최근접)   {pressers:2}/{base_lab}")
    print(f"   러너가 그의 최근접 공격수가 아님        {notown:2}/{base_lab}  (3순위 이하 {notown2})")
    print("   → 0이 아니면 그 정적 게이트는 진짜 반응 수비수를 떨어뜨립니다.\n")
    for k in ("near", "path_min", "path_end", "catch", "line_min"):
        a = np.array([r[k] for r in lab_rows])
        print(f"   {k:9} 라벨  최소 {a.min():6.1f} · 중앙 {np.median(a):6.1f} · 최대 {a.max():6.1f}")

    print("\n" + "=" * 72)
    print("순위별 분포 (전체 후보)")
    print("=" * 72)
    for k in range(3):
        sub = [r for r in rows if r["rank"] == k]
        if not sub:
            continue
        pm = np.array([r["path_min"] for r in sub])
        ct = np.array([r["catch"] for r in sub])
        print(f"   {k+1}순위  path_min 중앙 {np.median(pm):5.1f} · 90% {np.percentile(pm,90):5.1f} "
              f"· 20m 초과 {(pm>20).mean():3.0%}   catch<0 비율 {(ct<0).mean():3.0%}")

    def survive(fn):
        keep = [r for r in rows if fn(r)]
        lab = len({r["key"] for r in keep} & labels)
        sc = len({r["scene"] for r in keep})
        per = collections.Counter(r["scene"] for r in keep)
        mean_per = len(keep) / max(len(per), 1)
        return len(keep), sc, lab, mean_per

    gates = [
        ("현재 (게이트 없음)", lambda r: True),
        ("near ≤ 20m (온셋 거리)", lambda r: r["near"] <= 20),
        ("path_min ≤ 20m", lambda r: r["path_min"] <= 20),
        ("path_min ≤ 15m", lambda r: r["path_min"] <= 15),
        ("path_min ≤ 12m", lambda r: r["path_min"] <= 12),
        ("path_min ≤ 10m", lambda r: r["path_min"] <= 10),
        ("path_min ≤ 8m", lambda r: r["path_min"] <= 8),
        ("path_end ≤ 15m", lambda r: r["path_end"] <= 15),
        ("path_end ≤ 12m", lambda r: r["path_end"] <= 12),
        ("catch ≥ 0 (따라잡을 수 있음)", lambda r: r["catch"] >= 0),
        ("catch ≥ −3m", lambda r: r["catch"] >= -3),
        ("catch ≥ −5m", lambda r: r["catch"] >= -5),
        ("line_min ≤ 15m (온셋 속도 직선)", lambda r: r["line_min"] <= 15),
        ("line_min ≤ 12m", lambda r: r["line_min"] <= 12),
        ("line_min ≤ 10m", lambda r: r["line_min"] <= 10),
    ]

    print("\n" + "=" * 72)
    print("게이트별 생존")
    print("=" * 72)
    print(f"   {'게이트':32} {'삼중항':>12} {'장면':>6} {'장면당':>6} {'라벨':>8}")
    for name, fn in gates:
        cnt, sc, lab, mp = survive(fn)
        flag = "" if lab == base_lab else f"  −{base_lab - lab}"
        print(f"   {name:32} {cnt:5} ({cnt/n:3.0%}) {sc:5} {mp:6.2f} "
              f"{lab:4}/{base_lab}{flag}")

    print("\n" + "=" * 72)
    print("장면당 수비수 수 — 라벨을 하나도 잃지 않는 가장 빡빡한 path_min")
    print("=" * 72)
    lab_pm = max(r["path_min"] for r in lab_rows)
    thr = math.ceil(lab_pm)
    print(f"   라벨 중 가장 먼 path_min = {lab_pm:.1f}m → 문턱 {thr}m")
    per = collections.Counter(r["scene"] for r in rows if r["path_min"] <= thr)
    dist = collections.Counter(per.values())
    for k in sorted(dist):
        print(f"   {k}명   장면 {dist[k]:4}  ({dist[k]/max(len(per),1):.0%})")
    print(f"   0명   장면 {scenes - len(per):4}")
    print(f"\n   사람 라벨 분포는 1명 68% · 2명 26% · 3명 6% 였습니다.")


if __name__ == "__main__":
    main()
