#!/usr/bin/env python3
"""The abstract's evaluation numbers (2026-09-30): every real moment of the evaluation set, what the players
did against what the game says.

Each state of the evaluation set is one real moment (0 / 0.6 / 1.2 s of a scene) solved as its own game
(jobs/solve_evaluation.sbatch). Per moment this reads the opening payoff table the solver saved (solve.root_game:
M[defender command, attack column] with the rest of the game already solved, and its equilibrium p, q) and the
tracking 0.6 s later, and reports:

  observed    each player's real move = the command whose 0.6 s end (the solver's own movement, panel
              options' "end") is nearest where he really was 0.6 s later; the ball carrier's / passer's pass
              when the ball moved faster than any player can run with it (TOP_SPEED, as build_eval_states)
  value of an option (the solver's modal-line definitions)
              attack column a:  u[a] = p . M[:, a]  (against the defender's equilibrium)
              ball carrier / runner / beneficiary command: the best column with that command, the teammate
              best-responding (max over his commands); the ball carrier's "pass" = the best pass column
              defender command d: w[d] = M[d, :] . q (what the attack gets; lower is better for him)
  rank        1 + the number of legal options strictly better than the observed one (ties take the best
              place); the fractional rank gives tied options the mean of their places; rank_frac = (n -
              fractional rank) / (n - 1), 1 = the best option, 0 = the worst. Ties (TIE) are common: a player
              whose every command is worth the same (indifferent) is reported as such
  similarity  the equilibrium's probability of the observed option (its marginal for the player)
  regret      value of the best option minus the observed one's (for the defender: observed minus best)
  static vs responsive
              defender held to his observed command d_obs: the attack's best column there a* appears worth
              M[d_obs, a*]; once the defender may answer it is worth min_d M[d, a*]; loss = the difference.
              And the other way: the attack held to its observed column, the defender's best there d*,
              then the attack answers: max_a M[d*, a].
  equilibrium vs observed
              defence: max_a M[d_obs, a] (the attack's best against the real defender) vs the value V*
              attack:  u[a_obs] (the real attack against the equilibrium defender) vs V*
  dilemma     the defender's equilibrium mixes (more than one command above the solver's 1e-9, as
              solve.py's mixed_states); no saddle point: min_d max_a M - max_a min_d M > 1e-9

Moments where the observed command is unclear are not dropped: the distance to the nearest and the second
nearest command end is reported so the reader can see how well the real move matches the command set.

Usage:
    PYTHONPATH=src:scripts python scripts/analyze_eval.py \\
        --solved out/runs/eval_v1_2v1 out/runs/eval_v1_3v1 \\
        --panels data/processed/eval_v1/panels --output data/processed/eval_v1/analysis
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics as st
from collections import defaultdict
from pathlib import Path

import numpy as np

from build_stage3_states import corner
from offball_value.bundesliga import find_bundesliga_files, load_bundesliga_frames

FPS = 25
STEP_FRAMES = 15                   # 0.6 s
HALF_FRAMES = 4                    # ball speed over +-0.16 s (build_eval_states)
TOP_SPEED = 9.0                    # m/s, the states files' top player speed (build_eval_states)
MIXED = 1e-9                       # solve.py's mixed_states threshold (the LP's tolerance scale)
TIE = 1e-9                         # two option values this close are equal (the same tolerance scale)
SHOWCASE = ("S05", "S13", "S34", "S53", "S15", "S20", "S36", "S44")
SURE = ("S46", "S48", "S58", "S66", "S72")      # pipeline scenes the annotator marked 'sure' off-ball runs
DATA = Path(__file__).resolve().parents[1]      # the repository root


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--solved", type=Path, nargs="+", required=True)
    p.add_argument("--panels", type=Path, required=True)
    p.add_argument("--raw-dir", type=Path, default=DATA / "data/raw/bundesliga-integrated")
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def rank_of(values, legal, observed, lower_better=False):
    """(competition rank: 1 + options strictly better; fractional rank: ties share the mean of the places
    they occupy; n; tied: options equal to the observed one incl. itself; indifferent: all options equal).
    Equal = within TIE, the solver's tolerance scale."""
    vals = [(v if not lower_better else -v) for v, ok in zip(values, legal) if ok]
    mine = values[observed] if not lower_better else -values[observed]
    n = len(vals)
    better = int(sum(v > mine + TIE for v in vals))
    tied = int(sum(abs(v - mine) <= TIE for v in vals))
    return 1 + better, better + (tied + 1) / 2, n, tied, bool(max(vals) - min(vals) <= TIE)


def moment(state, rec, panel, frames):
    """Every number for one solved moment."""
    g = state["root_game"]
    M = np.array([[np.nan if x is None else x for x in row] for row in g["matrix"]])
    rows, cols = np.array(g["defender_legal"]), np.array(g["attack_legal"])
    p, q = np.array(state["root_defender"]), np.array(state["root_attack"])
    V = float(state["value"])
    kind = "3v1" if rec.get("passer") else "2v1"
    pv = rec["provenance"]
    A = 5
    moves_n = A * A
    Ml = np.where(rows[:, None] & cols[None, :], M, np.nan)
    u = np.nansum(p[:, None] * np.nan_to_num(Ml), axis=0)            # attack columns vs the defender's mix
    u = np.where(cols, u, np.nan)
    w = np.where(rows, np.nan_to_num(Ml) @ q, np.nan)                 # defender commands vs the attack's mix
    moves = u[:moves_n].reshape(A, A)
    pass_cols = np.flatnonzero(cols[moves_n:]) + moves_n
    best_pass = float(np.nanmax(u[pass_cols])) if len(pass_cols) else math.nan

    # observed: nearest command end per player, and whether a pass was played in the 0.6 s
    start = int(pv["onset_frame_id"])
    ids = ({"carrier": pv["carrier_id"], "receiver": pv["runner_id"], "defender": pv["defender_id"]} if kind == "2v1"
           else {"carrier": pv["runner_id"], "receiver": pv["beneficiary_id"], "defender": pv["defender_id"]})
    obs, fit = {}, {}
    later = frames.get(start + STEP_FRAMES)
    for slot, body in panel["bodies"].items():
        pid = ids[slot]
        real = np.array(corner((later.players[pid].x, later.players[pid].y)))
        d = sorted((float(np.linalg.norm(real - np.array(o["end"]))), i) for i, o in enumerate(body["options"]))
        obs[slot], fit[slot] = d[0][1], (round(d[0][0], 2), round(d[1][0], 2))
    speeds = []
    for f in range(start + 1, start + STEP_FRAMES + 1):
        a, b = frames.get(f - HALF_FRAMES), frames.get(f + HALF_FRAMES)
        if a is not None and b is not None and a.ball is not None and b.ball is not None:
            speeds.append(math.hypot(b.ball.x - a.ball.x, b.ball.y - a.ball.y) / (2 * HALF_FRAMES / FPS))
    passer_on_ball = kind == "2v1" or bool((rec.get("release_steps") or [True])[0])
    passed = passer_on_ball and bool(speeds) and max(speeds) > TOP_SPEED and len(pass_cols) > 0

    roles = ({"carrier": "ball carrier", "receiver": "runner", "defender": "defender"} if kind == "2v1"
             else {"carrier": "runner", "receiver": "beneficiary", "defender": "defender"})
    players = []
    # attackers' commands: best column with that command, the teammate best-responding
    carrier_vals = list(np.nanmax(moves, axis=1)) + ([best_pass] if kind == "2v1" else [])
    carrier_legal = [not math.isnan(v) for v in carrier_vals]
    carrier_prob = list(q[:moves_n].reshape(A, A).sum(axis=1)) + ([float(q[moves_n:].sum())] if kind == "2v1" else [])
    receiver_vals = list(np.nanmax(moves, axis=0))
    receiver_prob = list(q[:moves_n].reshape(A, A).sum(axis=0))
    c_obs = (A if (kind == "2v1" and passed) else obs["carrier"])
    for slot, vals, prob, o, lower in (("carrier", carrier_vals, carrier_prob, c_obs, False),
                                       ("receiver", receiver_vals, receiver_prob, obs["receiver"], False),
                                       ("defender", list(w), list(p), obs["defender"], True)):
        legal = [not (isinstance(v, float) and math.isnan(v)) for v in vals]
        if not legal[o]:
            continue
        rank, frank, n, tied, indifferent = rank_of(vals, legal, o, lower)
        frac = (n - frank) / (n - 1) if n > 1 else 1.0
        best = (min if lower else max)(v for v, ok in zip(vals, legal) if ok)
        players.append({"role": roles[slot], "observed": ("pass" if (slot == "carrier" and o == A) else int(o)),
                        "rank": int(rank), "rank_fractional": frank, "n": int(n), "tied": tied,
                        "indifferent": indifferent, "rank_frac": round(frac, 4), "similarity": round(float(prob[o]), 4),
                        "regret": round(float((vals[o] - best) if lower else (best - vals[o])), 5),
                        "fit_m": fit.get(slot)})

    # static vs responsive, both ways
    d_obs = obs["defender"]
    rowv = np.where(cols, M[d_obs], np.nan)
    a_star = int(np.nanargmax(rowv))
    appeared = float(rowv[a_star])
    responsive = float(np.nanmin(np.where(rows, M[:, a_star], np.nan)))
    a_obs = (int(pass_cols[np.nanargmax(u[pass_cols])]) if passed else obs["carrier"] * A + obs["receiver"])
    col = np.where(rows, M[:, a_obs], np.nan)
    d_star = int(np.nanargmin(col))
    def_appeared = float(col[d_star])
    def_responsive = float(np.nanmax(np.where(cols, M[d_star], np.nan)))
    Ms = M[np.ix_(rows, cols)]                                        # the legal table
    saddle_gap = float(Ms.max(axis=1).min() - Ms.min(axis=0).max())
    return {
        "code": pv["code"], "scene": pv["code"].split("@")[0], "kind": kind, "value": V, "passed": bool(passed),
        "defender_mixed": int((p > MIXED).sum() > 1), "attack_mixed": int((q > MIXED).sum() > 1),
        "no_saddle": int(saddle_gap > MIXED), "saddle_gap": round(saddle_gap, 6),
        "static_attack_appeared": appeared, "static_attack_responsive": responsive,
        "static_attack_loss": appeared - responsive, "static_attack_loss_rel": (appeared - responsive) / appeared,
        "static_defence_appeared": def_appeared, "static_defence_responsive": def_responsive,
        "static_defence_loss": def_responsive - def_appeared,
        "vs_real_defender": float(np.nanmax(rowv)), "defence_gain_rel": (float(np.nanmax(rowv)) - V) / float(np.nanmax(rowv)),
        "real_attack_vs_eq": float(u[a_obs]), "attack_gain_rel": (V - float(u[a_obs])) / float(u[a_obs]),
        "players": players,
    }


def summary(rows, label):
    if not rows:
        return {"subset": label, "moments": 0}
    med = lambda xs: float(st.median(xs)) if xs else math.nan
    mean = lambda xs: float(st.mean(xs)) if xs else math.nan
    by_role = defaultdict(list)
    for r in rows:
        for pl in r["players"]:
            by_role[pl["role"]].append(pl)
    allp = [pl for r in rows for pl in r["players"]]
    out = {"subset": label, "scenes": len({r["scene"] for r in rows}), "moments": len(rows),
           "player_decisions": len(allp),
           "defender_mixed_pct": 100 * mean([r["defender_mixed"] for r in rows]),
           "attack_mixed_pct": 100 * mean([r["attack_mixed"] for r in rows]),
           "no_saddle_pct": 100 * mean([r["no_saddle"] for r in rows]),
           "static_attack_loss_mean": mean([r["static_attack_loss"] for r in rows]),
           "static_attack_loss_rel_mean_pct": 100 * mean([r["static_attack_loss_rel"] for r in rows]),
           "static_attack_loss_rel_median_pct": 100 * med([r["static_attack_loss_rel"] for r in rows]),
           "static_defence_loss_mean": mean([r["static_defence_loss"] for r in rows]),
           "defence_gain_rel_max_pct": 100 * max(r["defence_gain_rel"] for r in rows),
           "defence_gain_rel_median_pct": 100 * med([r["defence_gain_rel"] for r in rows]),
           "attack_gain_rel_max_pct": 100 * max(r["attack_gain_rel"] for r in rows),
           "attack_gain_rel_median_pct": 100 * med([r["attack_gain_rel"] for r in rows]),
           "rank_median_all (competition: ties -> best place)": med([pl["rank"] for pl in allp]),
           "rank_fractional_median_all (ties share places)": med([pl["rank_fractional"] for pl in allp]),
           "indifferent_pct (every option worth the same)": 100 * mean([pl["indifferent"] for pl in allp]),
           "observed_tied_with_another_pct": 100 * mean([pl["tied"] > 1 for pl in allp]),
           "rank_frac_median_all": med([pl["rank_frac"] for pl in allp]),
           "similarity_mean_all": mean([pl["similarity"] for pl in allp]),
           "regret_mean_all": mean([pl["regret"] for pl in allp])}
    for role, pls in sorted(by_role.items()):
        out[f"{role}: n / rank median / fractional rank median / rank_frac median / similarity mean / regret mean / indifferent %"] = (
            len(pls), med([x["rank"] for x in pls]), med([x["rank_fractional"] for x in pls]),
            round(med([x["rank_frac"] for x in pls]), 3), round(mean([x["similarity"] for x in pls]), 3),
            round(mean([x["regret"] for x in pls]), 4), round(100 * mean([x["indifferent"] for x in pls]), 1))
    return out


def main() -> None:
    args = parse_args()
    rows, missing = [], []
    for solved in args.solved:
        recs = {r["index"]: r for r in json.loads((solved / "starting_states.json").read_text())["states"]}
        by_match = defaultdict(list)
        for idx, rec in recs.items():
            by_match[rec["provenance"]["match_id"]].append(rec)
        for match, group in sorted(by_match.items()):
            files = find_bundesliga_files(args.raw_dir, match)
            wanted = sorted({int(r["provenance"]["onset_frame_id"]) + d for r in group
                             for d in range(-HALF_FRAMES, STEP_FRAMES + HALF_FRAMES + 1)})
            frames = load_bundesliga_frames(files["positions"], wanted)
            for rec in group:
                sf = solved / "states" / f"state_{rec['index']:03d}.json"
                pf = args.panels / f"eval-{rec['provenance']['code']}.json"
                if not sf.exists() or not pf.exists():
                    missing.append(rec["provenance"]["code"])
                    continue
                rows.append(moment(json.loads(sf.read_text()), rec, json.loads(pf.read_text()), frames))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "moments.json").write_text(json.dumps(rows, indent=1))
    with (args.output / "players.csv").open("w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["code", "kind", "role", "observed", "rank", "rank_fractional", "n", "tied", "indifferent",
                     "rank_frac", "similarity", "regret",
                     "fit_nearest_m", "fit_second_m", "passed"])
        for r in rows:
            for pl in r["players"]:
                wr.writerow([r["code"], r["kind"], pl["role"], pl["observed"], pl["rank"], pl["rank_fractional"], pl["n"],
                             pl["tied"], pl["indifferent"], pl["rank_frac"],
                             pl["similarity"], pl["regret"], *(pl["fit_m"] or ("", "")), r["passed"]])
    subsets = {"all": rows,
               "showcase+sure (13 scenes)": [r for r in rows if r["scene"] in SHOWCASE + SURE],
               "2v1": [r for r in rows if r["kind"] == "2v1"], "3v1": [r for r in rows if r["kind"] == "3v1"]}
    sums = [summary(v, k) for k, v in subsets.items()]
    (args.output / "summary.json").write_text(json.dumps({"summaries": sums, "missing": missing}, indent=1))
    for s in sums:
        print(f"\n== {s['subset']}")
        for k, v in s.items():
            if k != "subset":
                print(f"   {k}: {round(v, 4) if isinstance(v, float) else v}")
    if missing:
        print("\nmissing (not solved or no panel):", " ".join(missing))


if __name__ == "__main__":
    main()
