#!/usr/bin/env python3
"""Do the kinematic features predict the label, or restate it?

The 360 training set has only one location per pass: `pass.end_location`,
which is where the ball ENDED. On a completion that is a team-mate's feet, so
the arrival time of the nearest team-mate is near zero; on an interception it
is an opponent's feet, so the nearest defender's is. Every arrival-race feature
built against that point therefore carries the outcome inside it, and a model
fitted on them learns a near-tautology that means something else entirely when
the target is a point the option catalogue invented.

That is a claim with a number attached: if a single feature separates the
label almost perfectly on its own, it is not modelling the race, it is reading
the answer. Reported per feature, alongside the static ones for scale, and
split by competition so the women's and international share can be judged at
the same time.

Usage:
    python scripts/audit_kinematic_leakage.py
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from offball_value.kinematic_xpass import (
    KINEMATIC_FEATURE_NAMES,
    TrackedPlayer,
    kinematic_features,
)
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path,
                   default=Path("data/raw/statsbomb-open-data/data"))
    p.add_argument("--output", type=Path,
                   default=Path("out/delivery_analysis/kinematic_leakage.json"))
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


def match_velocities(now, past, gap_s):
    pools = {True: [], False: []}
    for p in past:
        pools[bool(p.get("teammate"))].append(to_m(p["location"]))
    out = []
    for p in now:
        side = bool(p.get("teammate"))
        here = to_m(p["location"])
        pool = pools[side]
        if not pool:
            out.append(TrackedPlayer(here[0], here[1], 0.0, 0.0, teammate=side,
                                     actor=bool(p.get("actor")),
                                     keeper=bool(p.get("keeper"))))
            continue
        j = min(range(len(pool)),
                key=lambda k: math.hypot(pool[k][0] - here[0], pool[k][1] - here[1]))
        was = pool.pop(j)
        out.append(TrackedPlayer(
            here[0], here[1], (here[0] - was[0]) / gap_s, (here[1] - was[1]) / gap_s,
            teammate=side, actor=bool(p.get("actor")), keeper=bool(p.get("keeper"))))
    return out


def main() -> None:
    args = parse_args()
    matches = {}
    for path in sorted((args.data_root / "matches").rglob("*.json")):
        for match in json.loads(path.read_text(encoding="utf-8")):
            matches[str(match["match_id"])] = match

    X, y, comp = [], [], []
    for frames_path in sorted((args.data_root / "three-sixty").glob("*.json")):
        events_path = args.data_root / "events" / frames_path.name
        if not events_path.exists():
            continue
        meta = matches.get(frames_path.stem, {})
        competition = meta.get("competition", {}).get("competition_name", "?")
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
            opponents = [(float(p["location"][0]), float(p["location"][1]))
                         for p in ff[str(event["id"])] if not p.get("teammate")]
            static = np.concatenate([base, lane_features(start, end, opponents)])
            tracked = match_velocities(ff[str(event["id"])], ff[str(seq[i-1][1]["id"])], gap)
            kin = kinematic_features(tracked, to_m(start), to_m(end))
            X.append(np.concatenate([static, kin]))
            y.append(label)
            comp.append(competition)
    X = np.stack(X)
    y = np.asarray(y, dtype=float)
    comp = np.asarray(comp)

    names = list(FEATURE_NAMES_360) + list(KINEMATIC_FEATURE_NAMES)
    n_static = len(FEATURE_NAMES_360)
    print(f"표본 {len(X):,} · 성공률 {y.mean():.4f}\n")

    print("=" * 76)
    print("feature 하나만으로 라벨을 얼마나 가르나 (AUC)")
    print("=" * 76)
    scored = [(abs(auc(X[:, k], y) - 0.5) + 0.5, names[k], k) for k in range(len(names))]
    scored.sort(reverse=True)
    print(f"  {'feature':<28}{'AUC':>8}{'종류':>10}")
    for value, name, k in scored[:14]:
        kind = "정적" if k < n_static else "운동학"
        print(f"  {name:<28}{value:>8.3f}{kind:>10}")
    print("\n  0.5 = 무의미. 0.9 이상이면 그 항 하나가 정답을 거의 다 안다는 뜻.")
    out = {"n": int(len(X)), "completion": float(y.mean()),
           "single_feature_auc": {name: float(v) for v, name, _ in scored}}

    print("\n" + "=" * 76)
    print("대회별 구성과 성공률")
    print("=" * 76)
    tally = Counter(comp)
    print(f"  {'대회':<34}{'패스':>9}{'비중':>8}{'성공률':>9}")
    for name, count in tally.most_common():
        m = comp == name
        print(f"  {name:<32}{count:>9,}{count/len(comp):>8.1%}{y[m].mean():>9.3f}")
    women = np.isin(comp, WOMEN)
    print(f"\n  여자 축구 {women.sum():,} ({women.mean():.1%}) · 성공률 {y[women].mean():.3f}")
    print(f"  남자 축구 {(~women).sum():,} · 성공률 {y[~women].mean():.3f}")
    out["women_fraction"] = float(women.mean())
    out["completion_women"] = float(y[women].mean())
    out["completion_men"] = float(y[~women].mean())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"\nsaved {args.output}")


if __name__ == "__main__":
    main()
