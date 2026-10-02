#!/usr/bin/env python3
"""Fit candidate A-sym beside A on the same passes and compare them out of match.

A-sym (physics_pass.PhysicsRaceSymPass, 2026-09-28) is candidate A with the
receiver and the defenders held to one rule: a player counts as at a point only
if he can be there when the ball is -- running through it by then, or stopping on
it to wait. A lets a receiver who runs through the target early wait there for
the ball, while a defender who has passed it must come back. Same physics
constants, same two margins, same three logistic numbers fitted by maximum
likelihood, and each match predicted by the models fitted on the others -- the
procedure of fit_pass_candidates.py, whose helpers are used as they are.

Also reported, to see whether the passes the new rule changes behave as it says:
  receiver-changed  A-sym's take time differs from A's: the receiver runs through
                    the target before the ball and cannot stop on it
  defender-changed  the first defender at the target differs: one A counts there
                    ran through it too early to stop
  lane-changed      the same on the lane points
For each group, the observed completion next to A's and A-sym's out-of-match
predictions.

Writes the A-sym models to a NEW folder (default data/processed/pass_models_sym):
Asym_all.json, Asym_without_<match>.json, Asym_crossfit.json and report.json.
The A files in data/processed/pass_models are not touched.

Usage:
    PYTHONPATH=src:scripts python scripts/fit_pass_sym.py --passes data/processed/pass_models/passes.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fit_pass_candidates import BANDS, PLAYER, a_margins, arrays, ball_line, fit_logit, log_loss, report
from offball_value.physics_pass import KIND, KIND_SYM, PhysicsRacePass, PhysicsRaceSymPass

ROOT = Path(__file__).resolve().parents[1]
CLASSES = {"A": (KIND, PhysicsRacePass), "Asym": (KIND_SYM, PhysicsRaceSymPass)}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--passes", type=Path, required=True)
    p.add_argument("--output", type=Path, default=ROOT / "data/processed/pass_models_sym")
    p.add_argument("--reference", type=Path, default=ROOT / "data/processed/pass_models/report.json",
                   help="the A fit's own report, to check A is reproduced")
    return p.parse_args()


def fit(name, rows, y, meta):
    kind, cls = CLASSES[name]
    spec = {"kind": kind, "coef": [0.0, 0.0, 0.0], "ball_speed": ball_line(rows),
            "player": PLAYER, "metadata": meta}
    m = fit_logit(a_margins(cls(spec), rows), y, 1e4)
    spec["coef"] = [float(m.intercept_[0]), float(m.coef_[0][0]), float(m.coef_[0][1])]
    return cls(spec)


def changed(rows):
    """Per pass: which of the three race terms A-sym computes differently from A (coefficients do not enter)."""
    spec = {"coef": [0.0, 0.0, 0.0], "ball_speed": ball_line(rows), "player": PLAYER}
    a, s = PhysicsRacePass({**spec, "kind": KIND}), PhysicsRaceSymPass({**spec, "kind": KIND_SYM})
    out = []
    for c, cv, rr, rv, d, dv, t, ad in map(arrays, rows):
        length = np.hypot(*(t - c))
        tb = length / a.ball_speed(length)
        recv = not np.isclose(a._take(rr, rv, t, tb), s._take(rr, rv, t, tb))
        dfn = not np.isclose(a._there(d, dv, t[None, :], tb).min(), s._there(d, dv, t[None, :], tb).min())
        ma, ms = a.margins(c, rr, d, rv, dv, t), s.margins(c, rr, d, rv, dv, t)
        out.append((recv, dfn, not np.isclose(ma[1], ms[1])))
    return np.array(out)


def main() -> None:
    args = parse_args()
    rows = [json.loads(line) for line in args.passes.read_text().splitlines() if line.strip()]
    y = np.array([r["label"] for r in rows], dtype=float)
    groups = np.array([r["match_id"] for r in rows])
    dist = np.array([r["pass_distance"] for r in rows])
    matches = sorted(set(groups))
    args.output.mkdir(parents=True, exist_ok=True)
    meta = {"training": "our Bundesliga open-play passes, tracking labels, intended-at-kick targets",
            "passes_file": str(args.passes), "fit_by": "scripts/fit_pass_sym.py"}
    print(f"passes {len(rows):,} · completion {y.mean():.4f} · matches {len(matches)}", flush=True)

    oof = {name: np.zeros(len(rows)) for name in CLASSES}
    for m in matches:
        tr, te = groups != m, groups == m
        rows_tr = [r for r, k in zip(rows, tr) if k]
        rows_te = [r for r, k in zip(rows, te) if k]
        for name in CLASSES:
            held = {**meta, "held_out_match": m, "trained_on": [x for x in matches if x != m],
                    "passes": int(tr.sum())}
            model = fit(name, rows_tr, y[tr], held)
            oof[name][te] = model.probability_from_margins(*a_margins(model, rows_te).T)
            if name == "Asym":
                (args.output / f"Asym_without_{m}.json").write_text(json.dumps(model.spec, indent=2) + "\n")
        print(f"  fold {m}: passes {int(te.sum())}", flush=True)
    a_all = fit("Asym", rows, y, {**meta, "trained_on": matches, "passes": len(rows)})
    (args.output / "Asym_all.json").write_text(json.dumps(a_all.spec, indent=2) + "\n")
    router = {"kind": "per_match", "default": "Asym_all.json",
              "by_match": {m: f"Asym_without_{m}.json" for m in matches},
              "why": "a scene from match M is priced by the model fitted without M"}
    (args.output / "Asym_crossfit.json").write_text(json.dumps(router, indent=2) + "\n")

    reps = [report(name, oof[name], y, dist) for name in CLASSES]
    ref = json.loads(args.reference.read_text()) if args.reference.exists() else {}
    ref_a = next((r for r in ref.get("out_of_match", []) if r["name"] == "A"), None)
    flags = changed(rows)
    groups_rep = {}
    for j, label in enumerate(("receiver-changed", "defender-changed", "lane-changed")):
        k = flags[:, j]
        groups_rep[label] = {"n": int(k.sum()), "observed": float(y[k].mean()) if k.any() else None,
                             "A": float(oof["A"][k].mean()) if k.any() else None,
                             "Asym": float(oof["Asym"][k].mean()) if k.any() else None,
                             "log_loss_A": log_loss(oof["A"][k], y[k]) if k.any() else None,
                             "log_loss_Asym": log_loss(oof["Asym"][k], y[k]) if k.any() else None}
    untouched = ~flags.any(axis=1)
    groups_rep["unchanged"] = {"n": int(untouched.sum()), "observed": float(y[untouched].mean()),
                               "A": float(oof["A"][untouched].mean()), "Asym": float(oof["Asym"][untouched].mean())}
    result = {"passes": len(rows), "completion": float(y.mean()), "matches": matches,
              "out_of_match": reps, "a_reproduces_reference": (None if ref_a is None else
                                                               abs(ref_a["log_loss"] - reps[0]["log_loss"]) < 1e-9),
              "asym_all": {"coef": a_all.spec["coef"], "ball_speed": a_all.spec["ball_speed"]},
              "changed_groups": groups_rep}
    (args.output / "report.json").write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n")

    print(f"\nOut-of-match evaluation · base-rate Brier {y.mean() * (1 - y.mean()):.4f}")
    print(f"  {'model':<8}{'AUC':>7}{'Brier':>9}{'logloss':>9}{'mean pred':>9}")
    for rep in reps:
        print(f"  {rep['name']:<8}{rep['auc']:>7.3f}{rep['brier']:>9.4f}{rep['log_loss']:>9.4f}{rep['mean_predicted']:>9.3f}")
    if ref_a is not None:
        print(f"  (A reproduced: earlier report logloss {ref_a['log_loss']:.4f} → this run {reps[0]['log_loss']:.4f}"
              f" · {'match' if result['a_reproduces_reference'] else 'MISMATCH'})")
    print("\n  Observed / predicted by length band")
    print(f"  {'band':<10}{'n':>5}{'obs':>7}{'A':>8}{'Asym':>8}")
    for i, band in enumerate(reps[0]["bands"]):
        print(f"  {band['band']:<10}{band['n']:>5}{band['observed']:>7.3f}{band['predicted']:>8.3f}"
              f"{reps[1]['bands'][i]['predicted']:>8.3f}")
    print("\n  Passes the rule changes (observed completion · A prediction · A-sym prediction)")
    for label, g in groups_rep.items():
        if not g["n"]:
            print(f"  {label:<17} 0"); continue
        extra = (f" · logloss A {g['log_loss_A']:.3f} → A-sym {g['log_loss_Asym']:.3f}" if "log_loss_A" in g else "")
        print(f"  {label:<17} {g['n']:>4} · observed {g['observed']:.3f} · A {g['A']:.3f} · A-sym {g['Asym']:.3f}{extra}")
    print(f"\nA-sym fit on all passes: coefficients {np.round(a_all.spec['coef'], 3).tolist()}")
    print(f"→ {args.output}")


if __name__ == "__main__":
    main()
