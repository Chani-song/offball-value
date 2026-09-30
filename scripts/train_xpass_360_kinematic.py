#!/usr/bin/env python3
"""xPass 360 with velocity features differenced from consecutive freeze frames.

A 360 freeze frame is a still, so xpass360 was built without any notion of who
gets to the ball first -- and measurement showed the cost: stratified on static
geometry its correlation with receiver_first is +0.267 where the mechanistic
chain holds +0.762, and in the largest stratum it is not even monotone.

But the frames are not isolated. Consecutive ones sit a median 0.97 s apart
(53 % of passes have a predecessor within 2 s), and degrading real tracking to
exactly that representation -- two snapshots, identities discarded, only the
players a camera would see -- still recovers receiver_first at correlation
0.934 against 25 fps truth. The identity problem is survivable because the
teammate flag narrows re-matching to one side.

So the race can be differenced out of the pair and handed to the model as
features. This builds that training set and asks the only question that
matters: do the kinematic features earn their place on top of the static ones
the current model already has, held out by match.

Usage:
    python scripts/train_xpass_360_kinematic.py \
        --output data/processed/xpass_360_kinematic
"""

from __future__ import annotations

import argparse
import json
import math
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
# The shipped xpass360 was trained on 17 of the 21 static features, dropping
# height_ground/low/high and under_pressure. That was not arbitrary: the
# pipeline cannot supply either -- every counterfactual pass goes in as a
# "Ground Pass" that is not under_pressure -- and a single-feature audit put
# height_high alone at AUC 0.964. Feeding a constant for the feature the model
# leans on hardest is how an inference set becomes uniformly optimistic. The
# first version of this script put them back in by accident.
DROP_FEATURES = ("height_ground", "height_low", "height_high", "under_pressure")
# Women's competitions are 26.5 % of the 360 passes and complete at 0.784
# against 0.868 for the men's, and the scenes this feeds are men's 2.
# Bundesliga. Kept as a switch rather than a silent filter.
WOMEN_COMPETITIONS = ("Women's World Cup", "UEFA Women's Euro")
PITCH_X_M = 105.0 / 120.0        # StatsBomb pitch units -> metres
PITCH_Y_M = 68.0 / 80.0
PASS_SPEED_MPS = 14.0
MAX_SPEED_MPS = 9.0
ACCEL_MPS2 = 3.5
REACTION_S = 0.10
TURN_PENALTY_S = 0.40

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path,
                   default=Path("data/raw/statsbomb-open-data/data"))
    p.add_argument("--output", type=Path, default=Path("data/processed/xpass_360_kinematic"))
    p.add_argument("--max-gap", type=float, default=MAX_GAP_S)
    p.add_argument("--keep-dropped", action="store_true",
                   help="keep height/under_pressure (the accidental v1 behaviour)")
    p.add_argument("--include-women", action="store_true")
    p.add_argument("--competitions", type=str, default="",
                   help="comma-separated competition names to keep; empty = all")
    return p.parse_args()


def to_m(xy):
    return (float(xy[0]) * PITCH_X_M, float(xy[1]) * PITCH_Y_M)


def match_velocities(now, past, gap_s):
    """Greedy nearest-neighbour within side. The teammate flag is the only
    identity a freeze frame gives, so it is the only one used."""
    out = []
    pools = {True: [], False: []}
    for p in past:
        pools[bool(p.get("teammate"))].append(to_m(p["location"]))
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
            here[0], here[1],
            (here[0] - was[0]) / gap_s, (here[1] - was[1]) / gap_s,
            teammate=side, actor=bool(p.get("actor")), keeper=bool(p.get("keeper"))))
    return out


def auc(pred, label):
    pos, neg = pred[label > 0.5], pred[label <= 0.5]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(order.size, float)
    ranks[order] = np.arange(1, order.size + 1)
    return float((ranks[: pos.size].sum() - pos.size * (pos.size + 1) / 2)
                 / (pos.size * neg.size))


def main() -> None:
    from sklearn.ensemble import HistGradientBoostingClassifier

    args = parse_args()
    frames_dir = args.data_root / "three-sixty"
    events_dir = args.data_root / "events"
    matches = sorted(frames_dir.glob("*.json"))
    meta_by_id = {}
    for path in sorted((args.data_root / "matches").rglob("*.json")):
        for match in json.loads(path.read_text(encoding="utf-8")):
            meta_by_id[str(match["match_id"])] = match
    keep_only = {c.strip() for c in args.competitions.split(",") if c.strip()}
    print(f"{len(matches)} three-sixty files", flush=True)
    if keep_only:
        print(f"  competitions restricted to: {sorted(keep_only)}")
    if not args.include_women:
        print(f"  women's competitions excluded: {list(WOMEN_COMPETITIONS)}")

    X, y, fold = [], [], []
    n_pass = n_paired = 0
    for index, frames_path in enumerate(matches):
        events_path = events_dir / frames_path.name
        if not events_path.exists():
            continue
        competition = (meta_by_id.get(frames_path.stem, {})
                       .get("competition", {}).get("competition_name", "?"))
        if not args.include_women and competition in WOMEN_COMPETITIONS:
            continue
        if keep_only and competition not in keep_only:
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
            if not is_open_play_pass(event):
                continue
            n_pass += 1
            if i == 0:
                continue
            t_prev, prev = seq[i - 1]
            gap = t_now - t_prev
            if not (0.05 < gap <= args.max_gap):
                continue
            n_paired += 1
            now_ff, past_ff = ff[str(event["id"])], ff[str(prev["id"])]
            base, label = event_features_and_label(event)
            start = (float(event["location"][0]), float(event["location"][1]))
            end = tuple(float(v) for v in event["pass"]["end_location"][:2])
            opponents = [(float(p["location"][0]), float(p["location"][1]))
                         for p in now_ff if not p.get("teammate")]
            static = np.concatenate([base, lane_features(start, end, opponents)])
            tracked = match_velocities(now_ff, past_ff, gap)
            kin = kinematic_features(tracked, to_m(start), to_m(end))
            X.append(np.concatenate([static, kin]))
            y.append(label)
            fold.append(index % 5)
        if (index + 1) % 100 == 0:
            print(f"  {index+1}/{len(matches)} · samples {len(X):,}", flush=True)

    X = np.stack(X)
    y = np.asarray(y, dtype=float)
    fold = np.asarray(fold)
    all_names = list(FEATURE_NAMES_360) + list(KINEMATIC_FEATURE_NAMES)
    if not args.keep_dropped:
        keep_idx = [i for i, n in enumerate(all_names) if n not in DROP_FEATURES]
        X = X[:, keep_idx]
        all_names = [all_names[i] for i in keep_idx]
        print(f"dropped features: {list(DROP_FEATURES)}")
    n_static = sum(1 for n in all_names if n in FEATURE_NAMES_360)
    print(f"\nopen-play passes {n_pass:,} · paired with the previous frame {n_paired:,} "
          f"({n_paired/max(n_pass,1):.1%}) · training samples {len(X):,}")
    print(f"completion rate {y.mean():.4f}")

    print(f"\n{'='*84}")
    print("5-fold cross-validation (split by match) · scored on held-out folds only")
    print("="*84)
    print(f"  {'':<28}{'AUC':>9}{'Brier':>10}{'fold std':>16}")
    results = {}
    # These names are the "results" keys in training_report.json, so they stay
    # as they are: "static only (current xpass360)" and "static + kinematic".
    for name, cols in (("static only (current xpass360)", list(range(n_static))),
                       ("static + kinematic", list(range(X.shape[1])))):
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
        print(f"  {name:<26}{a:>9.4f}{b:>10.4f}{np.std(per):>16.4f}")
    print(f"\n  base-rate Brier {y.mean()*(1-y.mean()):.4f}")
    d = results["static + kinematic"]["auc"] - results["static only (current xpass360)"]["auc"]
    print(f"  AUC added by kinematics: {d:+.4f}"
          f"  (fold std {np.std(results['static + kinematic']['fold_auc']):.4f})")

    print("\nfitting the final model on all data ...", flush=True)
    final = HistGradientBoostingClassifier(
        max_depth=None, max_iter=300, learning_rate=0.08,
        min_samples_leaf=50, l2_regularization=1.0, random_state=0)
    final.fit(X, y)
    args.output.mkdir(parents=True, exist_ok=True)
    import joblib
    joblib.dump(final, args.output / "xpass_360_kinematic_gbdt.joblib")
    print(f"saved {args.output/'xpass_360_kinematic_gbdt.joblib'}")

    (args.output / "training_report.json").write_text(json.dumps({
        "static_feature_names": list(FEATURE_NAMES_360),
        "kinematic_feature_names": list(KINEMATIC_FEATURE_NAMES),
        "feature_order": all_names,
        "dropped": [] if args.keep_dropped else list(DROP_FEATURES),
        "include_women": bool(args.include_women),
        "competitions": sorted(keep_only) or "all",
        "open_play_passes": n_pass, "paired": n_paired,
        "samples": int(len(X)), "completion": float(y.mean()),
        "max_gap_s": args.max_gap, "results": results,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nsaved {args.output/'training_report.json'}")


if __name__ == "__main__":
    main()
