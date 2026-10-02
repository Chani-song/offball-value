#!/usr/bin/env python3
"""Fit and compare the two pass-model candidates on our Bundesliga passes.

  baseline  andrew/models/experimental_pass.json -- Andrew's positions-only model,
            fitted on StatsBomb 360, unchanged. Scored on the same passes.
  B1        his own logistic on his own "velocity" feature set
            (expected_pass.pass_features, positions_only=False), refitted with
            L2 on our passes. The penalty is chosen by an inner
            leave-one-match-out search. Saved with his class, in his schema,
            so ExpectedPass.load reads it and the game needs no change.
  A         physics_pass.PhysicsRacePass: ball speed from a least-squares line
            on our launch speeds, the two race margins from the solver's own
            agile motion, and three logistic numbers.

Every score below is out of match: each match is predicted by models fitted
on the other five, never on itself.

The same rule decides what the SOLVER uses. A scene from match M is priced by
the model fitted without M (`<name>_without_<M>.json`), so no scene is valued
by a model that saw its own passes. That is what the `*_crossfit.json` router
files map. Scenes from a match with no passes here (DFL-MAT-J03WN1, excluded
for its early red card) get the model fitted on all six, which never saw
them either.

Family (ground / driven / lofted) is not observed in these rows. Every pass is
scored as ground, B1's two family columns are constant and get zero weight,
and A ignores family. In both candidates the solver's 18 passes act as 6
targets.

Usage:
    python scripts/fit_pass_candidates.py --passes data/processed/pass_models/passes.jsonl \
        --output data/processed/pass_models
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from defensive_positioning.expected_pass import FEATURE_SETS, ExpectedPass, pass_features
from offball_value.agile_motion import AGILE
from offball_value.physics_pass import KIND, MARGIN_CLIP_S, PhysicsRacePass

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "andrew/models/experimental_pass.json"
C_GRID = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
BANDS = (0, 10, 15, 20, 25, 30, 40, 200)
PLAYER = {"reaction_s": AGILE.defender_delay_s, "speed_up": AGILE.speed_up,
          "braking": AGILE.braking, "plant_min_speed": AGILE.plant_min_speed,
          "plant_braking": AGILE.plant_braking, "plant_seconds": AGILE.plant_seconds,
          "max_speed": 9.0}
# physically possible launch speeds, only to keep a mis-tracked ball out of the line fit
BALL_FIT_RANGE = (2.0, 40.0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--passes", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def arrays(r: dict):
    c, cv = np.array(r["passer"][:2]), np.array(r["passer"][2:])
    rr, rv = np.array(r["receiver"][:2]), np.array(r["receiver"][2:])
    opp = np.array(r["opponents"], dtype=float)
    return c, cv, rr, rv, opp[:, :2], opp[:, 2:4], np.array(r["target"]), int(r["attack_direction"])


def b1_features(rows) -> np.ndarray:
    out = []
    for r in rows:
        c, cv, rr, rv, d, dv, t, ad = arrays(r)
        out.append(pass_features(c, rr, d, cv, rv, dv, t, 0, ad))
    return np.array(out)


def baseline_predictions(rows) -> np.ndarray:
    model = ExpectedPass.load(BASELINE, allow_proxy=True)
    return np.array([float(model.predict(c, rr, d, cv, rv, dv, t, 0, ad))
                     for c, cv, rr, rv, d, dv, t, ad in map(arrays, rows)])


def standardise(x_train, x_test):
    mean, scale = x_train.mean(axis=0), x_train.std(axis=0)
    const = scale < 1e-9
    mean, scale = np.where(const, 0.0, mean), np.where(const, 1.0, scale)
    return (x_train - mean) / scale, (x_test - mean) / scale, mean, scale, const


def fit_logit(x, y, c):
    m = LogisticRegression(C=c, max_iter=5000)
    m.fit(x, y)
    return m


def log_loss(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def auc(p, y):
    pos, neg = p[y > 0.5], p[y <= 0.5]
    if not len(pos) or not len(neg):
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(order.size)
    ranks[order] = np.arange(1, order.size + 1)
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def fit_b1(x, y, groups):
    """Penalty by inner leave-one-match-out log-loss, then refit on all of x."""
    matches = sorted(set(groups))
    best = None
    for c in C_GRID:
        if len(matches) < 2:
            best = (0.0, 1.0)
            break
        losses = []
        for m in matches:
            tr, te = groups != m, groups == m
            xa, xb, *_ = standardise(x[tr], x[te])
            losses.append(log_loss(fit_logit(xa, y[tr], c).predict_proba(xb)[:, 1], y[te]) * te.sum())
        score = sum(losses) / len(y)
        if best is None or score < best[0]:
            best = (score, c)
    xs, _, mean, scale, const = standardise(x, x)
    model = fit_logit(xs, y, best[1])
    w = model.coef_[0].copy()
    w[const] = 0.0
    return ExpectedPass(mean, scale, w, float(model.intercept_[0]), {}, "velocity"), best[1]


def ball_line(rows):
    s = np.array([r["ball_speed"] for r in rows])
    d = np.array([r["pass_distance"] for r in rows])
    ok = (s >= BALL_FIT_RANGE[0]) & (s <= BALL_FIT_RANGE[1])
    slope, intercept = np.polyfit(d[ok], s[ok], 1)
    lo, hi = np.percentile(s[ok], [5, 95])
    return {"intercept": float(intercept), "slope": float(slope), "min": float(lo), "max": float(hi),
            "fitted_on": int(ok.sum())}


def a_margins(model: PhysicsRacePass, rows):
    out = []
    for c, cv, rr, rv, d, dv, t, ad in map(arrays, rows):
        mr, ml = model.margins(c, rr, d, rv, dv, t)
        out.append((float(mr), float(ml)))
    return np.clip(np.array(out), -MARGIN_CLIP_S, MARGIN_CLIP_S)


def fit_a(rows, y, meta):
    spec = {"kind": KIND, "coef": [0.0, 0.0, 0.0], "ball_speed": ball_line(rows),
            "player": PLAYER, "metadata": meta}
    x = a_margins(PhysicsRacePass(spec), rows)
    m = fit_logit(x, y, 1e4)
    spec["coef"] = [float(m.intercept_[0]), float(m.coef_[0][0]), float(m.coef_[0][1])]
    return PhysicsRacePass(spec)


def report(name, p, y, dist):
    bands = []
    for lo, hi in zip(BANDS[:-1], BANDS[1:]):
        k = (dist >= lo) & (dist < hi)
        if k.sum():
            bands.append({"band": f"[{lo},{hi})", "n": int(k.sum()),
                          "observed": float(y[k].mean()), "predicted": float(p[k].mean())})
    return {"name": name, "auc": auc(p, y), "brier": float(np.mean((p - y) ** 2)),
            "log_loss": log_loss(p, y), "mean_predicted": float(p.mean()), "bands": bands}


def probe(models: dict) -> list:
    """A through ball: runner 20 m ahead sprinting toward goal, target 8 m in front
    of him, a defender beside him -- standing, then running the other way."""
    c, cv = np.array([50.0, 34.0]), np.zeros(2)
    r, t = np.array([70.0, 34.0]), np.array([78.0, 34.0])
    out = []
    # The labels and the row key below ("situation") are written to report.json,
    # and andrew-passer2on1/tests/test_pass_candidates.py reads them back, so they
    # stay as they are. The three cases: everyone standing; runner at 7 m/s in
    # behind, defender standing; runner at 7 m/s in behind, defender at 3 m/s the
    # other way.
    for label, rv, dv in (("everyone standing", [0.0, 0.0], [0.0, 0.0]),
                          ("runner 7 m/s in behind · defender standing", [7.0, 0.0], [0.0, 0.0]),
                          ("runner 7 m/s in behind · defender 3 m/s the other way", [7.0, 0.0], [-3.0, 0.0])):
        d = np.array([[71.0, 36.0]])
        row = {"situation": label}
        for name, m in models.items():
            row[name] = float(m.predict(c, r, d, cv, np.array(rv), np.array([dv]), t, 0, 1))
        out.append(row)
    return out


def main() -> None:
    args = parse_args()
    rows = [json.loads(line) for line in args.passes.read_text().splitlines() if line.strip()]
    y = np.array([r["label"] for r in rows], dtype=float)
    groups = np.array([r["match_id"] for r in rows])
    dist = np.array([r["pass_distance"] for r in rows])
    matches = sorted(set(groups))
    args.output.mkdir(parents=True, exist_ok=True)
    print(f"passes {len(rows):,} · completion {y.mean():.4f} · matches {len(matches)}", flush=True)

    xb1 = b1_features(rows)
    assert xb1.shape[1] == len(FEATURE_SETS["velocity"])
    oof = {"baseline": baseline_predictions(rows), "B1": np.zeros(len(rows)), "A": np.zeros(len(rows))}
    meta_common = {"training": "our Bundesliga open-play passes, tracking labels, intended-at-kick targets",
                   "passes_file": str(args.passes), "passes": len(rows)}
    chosen_c = {}
    for m in matches if len(matches) > 1 else []:
        tr, te = groups != m, groups == m
        b1, c = fit_b1(xb1[tr], y[tr], groups[tr])
        chosen_c[m] = c
        oof["B1"][te] = b1.probability(xb1[te])
        a = fit_a([r for r, k in zip(rows, tr) if k], y[tr], {})
        oof["A"][te] = a.probability_from_margins(*a_margins(a, [r for r, k in zip(rows, te) if k]).T)
        for name, model in (("B1", b1), ("A", a)):
            held = {**meta_common, "held_out_match": m, "trained_on": [x for x in matches if x != m]}
            if name == "B1":
                model.metadata = {"kind": "fitted_logistic", "target_semantics": "intended_at_kick",
                                  "validation_status": "leave_one_match_out", "feature_set": "velocity",
                                  "l2_C": c, **held}
                model.save(args.output / f"B1_without_{m}.json")
            else:
                model.spec["metadata"] = held
                (args.output / f"A_without_{m}.json").write_text(json.dumps(model.spec, indent=2) + "\n")
        print(f"  fold {m}: passes {te.sum()} · B1 C={c}", flush=True)

    b1_all, c_all = fit_b1(xb1, y, groups)
    b1_all.metadata = {"kind": "fitted_logistic", "target_semantics": "intended_at_kick",
                       "validation_status": "leave_one_match_out", "feature_set": "velocity",
                       "l2_C": c_all, **meta_common, "trained_on": matches}
    b1_all.save(args.output / "B1_all.json")
    a_all = fit_a(rows, y, {**meta_common, "trained_on": matches})
    (args.output / "A_all.json").write_text(json.dumps(a_all.spec, indent=2) + "\n")
    for name in ("A", "B1"):
        router = {"kind": "per_match", "default": f"{name}_all.json",
                  "by_match": {m: f"{name}_without_{m}.json" for m in matches if len(matches) > 1},
                  "why": "a scene from match M is priced by the model fitted without M"}
        (args.output / f"{name}_crossfit.json").write_text(json.dumps(router, indent=2) + "\n")

    results = {"passes": len(rows), "completion": float(y.mean()), "matches": matches,
               "b1_C": {"per_fold": chosen_c, "all": c_all},
               "a_all": {"coef": a_all.spec["coef"], "ball_speed": a_all.spec["ball_speed"]},
               "out_of_match": [report(k, v, y, dist) for k, v in oof.items()] if len(matches) > 1 else [],
               "probe": probe({"baseline": ExpectedPass.load(BASELINE, allow_proxy=True),
                               "B1": b1_all, "A": a_all})}
    (args.output / "report.json").write_text(json.dumps(results, ensure_ascii=False, indent=1) + "\n")

    print(f"\nOut-of-match evaluation (each match predicted by a model trained on the other matches) · base-rate Brier {y.mean() * (1 - y.mean()):.4f}")
    print(f"  {'model':<10}{'AUC':>7}{'Brier':>9}{'logloss':>9}{'mean pred':>9}")
    for rep in results["out_of_match"]:
        print(f"  {rep['name']:<10}{rep['auc']:>7.3f}{rep['brier']:>9.4f}{rep['log_loss']:>9.4f}{rep['mean_predicted']:>9.3f}")
    if results["out_of_match"]:
        print("\n  actual / predicted by length band")
        names = [r["name"] for r in results["out_of_match"]]
        print("  " + f"{'band':<10}{'n':>5}{'actual':>7}" + "".join(f"{n:>10}" for n in names))
        for i, band in enumerate(results["out_of_match"][0]["bands"]):
            print("  " + f"{band['band']:<10}{band['n']:>5}{band['observed']:>7.3f}"
                  + "".join(f"{r['bands'][i]['predicted']:>10.3f}" for r in results["out_of_match"]))
    print(f"\nA fitted on all: coef {np.round(a_all.spec['coef'], 3).tolist()} · ball speed {a_all.spec['ball_speed']}")
    print(f"B1 fitted on all: C={c_all}")
    print("\nThrough ball in behind (pass to 8 m ahead of the runner)")
    for row in results["probe"]:
        print("  " + row["situation"] + " → " + " · ".join(f"{k} {v:.3f}" for k, v in row.items() if k != "situation"))
    print(f"\n→ {args.output}")


if __name__ == "__main__":
    main()
