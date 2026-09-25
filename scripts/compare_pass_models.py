#!/usr/bin/env python3
"""Same scenes, three pass models: what changes in the solver's answer?

For each game (2v1, 3v1) and each solved study -- the positions-only model the
solver shipped with, candidate A (physics race), candidate B1 (velocity
logistic on our passes) and B1-SB (the same logistic on StatsBomb 360) -- on
the triangle-2 m scene set:

  torn             scenes whose most-likely line is torn somewhere (likeliest
                   defender target under 80 %), as on the review pages
  retreat          on scenes where the runner REALLY went more than 3 m
                   goalward, how often the defender drops more than 1 m toward
                   his goal over the same time, real tracking vs solver
                   (2026-09-25: real 94-96 %, positions-only solver 58-59 %)
  ends             where the line ends: pass to runner / beneficiary, or kept

Usage:  PYTHONPATH=src:scripts python scripts/compare_pass_models.py
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from defensive_positioning.models import GameConfig
from offball_value.stage3_read import modal_path
from filter_stage3_states import positions_at

ROOT = Path(__file__).resolve().parents[1]
OUT = Path("/work/hdd/bbmr/kseo1/offball-out")
BUILD = OUT / "v7_r9_ssac"
D = ROOT / "data/processed/stage3"
STUDIES = {
    "2대1": ("passer2on1", {"지금 모델": ("stage3_passer2on1_agile", "passer2on1_results_agile.csv"),
                           "A 물리": ("stage3_passer2on1_agile_passA", "passer2on1_results_passA.csv"),
                           "B1 속도로지스틱": ("stage3_passer2on1_agile_passB1", "passer2on1_results_passB1.csv"),
                           "B1-SB 스탯츠밤": ("stage3_passer2on1_agile_passB1SB", "passer2on1_results_passB1SB.csv")}),
    "3대1": ("fixedpasser", {"지금 모델": ("stage3_fixedpasser_agile", "fixedpasser_results_agile.csv"),
                           "A 물리": ("stage3_fixedpasser_agile_passA", "fixedpasser_results_passA.csv"),
                           "B1 속도로지스틱": ("stage3_fixedpasser_agile_passB1", "fixedpasser_results_passB1.csv"),
                           "B1-SB 스탯츠밤": ("stage3_fixedpasser_agile_passB1SB", "fixedpasser_results_passB1SB.csv")}),
}


def study(game, solved: Path, results: Path, keep: set[int]) -> dict | None:
    if not (solved / "manifest.json").exists() or not results.exists():
        return None
    man = json.loads((solved / "manifest.json").read_text())
    cfg = GameConfig(**{k: v for k, v in man["config"].items() if k in ("steps", "step_seconds", "physics_step")})
    start = {s["index"]: s for s in json.loads((solved / "starting_states.json").read_text())["states"]}
    res = pd.read_csv(results)
    res = res[res["index"].isin(keep)]
    torn = int((res["path_max_split"] > 0.2).sum())
    real_d, sol_d, ends = [], [], Counter()
    for i in sorted(keep & set(start)):
        f = solved / "states" / f"state_{i:03d}.json"
        if not f.exists():
            continue
        st, rec = json.loads(f.read_text()), start[i]
        pr, ad = rec["provenance"], int(st["scenario"]["attack_direction"])
        p = modal_path(st, solved / "policies" / f"state_{i:03d}.npz", cfg,
                       passer_track=(rec.get("passer") or {}).get("positions"), physics=man.get("physics"))
        roll, T = p["rollout"], p["rollout"][-1]["time"]
        e = p["end"]
        ends[e["to"] if e["event"] == "release" else e["event"]] += 1
        if T <= 0:
            continue
        g = json.loads((BUILD / pr["scene_dir"] / "local_game_payoff_audits.json").read_text())[0]
        fr = g["background_frames"]
        t0 = float(min(fr, key=lambda x: abs(x["frame_id"] - g["onset_frame_id"]))["relative_time_s"])
        a0, a1 = positions_at(fr, t0, 0.0), positions_at(fr, t0, T)
        if (a1[pr["runner_id"]][0] - a0[pr["runner_id"]][0]) * ad > 3.0:
            real_d.append((a1[pr["defender_id"]][0] - a0[pr["defender_id"]][0]) * ad)
            sol_d.append((roll[-1]["defender"][0] - roll[0]["defender"][0]) * ad)
    real_d, sol_d = np.array(real_d), np.array(sol_d)
    return {"scenes": len(res), "torn": torn, "goalward_runs": len(real_d),
            "real_retreat": float(np.mean(real_d > 1)) if len(real_d) else None,
            "solver_retreat": float(np.mean(sol_d > 1)) if len(sol_d) else None,
            "solver_step_up": float(np.mean(sol_d < -1)) if len(sol_d) else None,
            "solver_drop_median_m": float(np.median(sol_d)) if len(sol_d) else None,
            "ends": dict(ends)}


def main() -> None:
    out = {}
    for label, (game, runs) in STUDIES.items():
        keep = {s["index"] for s in json.loads((D / f"{game}_states_agile06_final2m.json").read_text())["states"]}
        print(f"\n== {label} (삼각형 2 m 장면 {len(keep)}개) — 러너가 실제로 골문 쪽 3 m 넘게 뛴 장면에서 수비")
        print(f"   {'모델':<16}{'갈린 장면':>9}{'실제 물러남':>11}{'솔버 물러남':>11}{'솔버 전진':>9}{'솔버 중앙값':>11}   흐름의 끝")
        for name, (solved, results) in runs.items():
            r = study(game, OUT / solved, D / results, keep)
            out[f"{label} {name}"] = r
            if r is None:
                print(f"   {name:<16}(결과 없음)")
                continue
            pct = lambda v: "-" if v is None else f"{v:.0%}"
            print(f"   {name:<16}{r['torn']:>5}/{r['scenes']:<4}{pct(r['real_retreat']):>10}"
                  f"{pct(r['solver_retreat']):>11}{pct(r['solver_step_up']):>9}"
                  f"{(r['solver_drop_median_m'] or 0):>+10.1f}m   {r['ends']}")
    path = ROOT / "data/processed/pass_models/solver_comparison.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")
    print(f"\n→ {path}")


if __name__ == "__main__":
    main()
