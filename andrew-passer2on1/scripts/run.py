#!/usr/bin/env python3
"""Solve the 2v1 background game over a states file.

Writes the same study layout as `defensive_positioning.exact_study` --
manifest.json, starting_states.json, rows.jsonl, summary.json, states/,
policies/ -- which `scripts/extract_panel_policy.py` and `scripts/analyze_eval.py`
read. The imported code is not modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import platform
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy

from defensive_positioning.models import GameConfig

from passer2on1.agile_motion import describe
from passer2on1.run_passes import passes_named
from passer2on1.solve import solve_one


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--states", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--allow-proxy-labels", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--step-seconds", type=float, default=1.0)
    p.add_argument("--physics-step", type=float, default=0.025)
    p.add_argument("--physics", choices=("andrew", "agile"), default="andrew",
                   help="movement model: the imported one, or agile_motion.AGILE")
    p.add_argument("--physics-file", type=Path, default=None,
                   help="JSON of agile_motion.Physics fields; overrides --physics (2026-09-28: "
                        "no reaction delay for games started mid-play)")
    p.add_argument("--commands", choices=("compass", "relative"), default="compass",
                   help="movement commands: the imported compass, or agile_motion.relative_commands")
    p.add_argument("--threat", choices=("andrew", "epv_time"), default="andrew",
                   help="threat (xT) at the pass target: the imported one, or EPV location with a time-margin room")
    p.add_argument("--passes", choices=("andrew", "run"), default="andrew",
                   help="pass candidates: the imported axis offsets, or run_passes.RUN_PASSES")
    p.add_argument("--pass-reaction-s", type=float, default=None,
                   help="price a pass after the defender has carried out his command this long "
                        "(passer2on1.reactive_pass, 2026-09-29); default: the imported pricing")
    p.add_argument("--background-tackles", action="store_true",
                   help="the other defenders can tackle the carrier too (passer2on1.game, 2026-09-29)")
    p.add_argument("--multi-pass", action="store_true",
                   help="every pass candidate its own attack column, priced after --pass-reaction-s "
                        "(passer2on1.multi_pass, 2026-09-29)")
    p.add_argument("--terminal", choices=("pass", "xt"), default="pass",
                   help="the horizon's worth: the imported best pass there (pass), or no pass there, only "
                        "the ball's possession value (xt, 2026-09-29: every pass then faces a defender who "
                        "chooses at the same time)")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--indices", type=str, default=None,
                   help="comma-separated state indices to solve instead of the first --limit")
    return p.parse_args()


def model_router(path: Path):
    """--model is a pass model, or a per-match router written by
    scripts/fit_pass_candidates.py: a scene from match M is then priced by the
    model fitted without M's passes."""
    raw = json.loads(path.read_text())
    if raw.get("kind") != "per_match":
        return (lambda record: str(path)), None
    by = {m: str(path.parent / f) for m, f in raw["by_match"].items()}
    default = str(path.parent / raw["default"])
    for f in (default, *by.values()):
        if not Path(f).exists():
            raise SystemExit(f"router names a missing model: {f}")
    return (lambda record: by.get(str((record.get("provenance") or {}).get("match_id")), default)), \
        {**raw, "sha256": {Path(f).name: hashlib.sha256(Path(f).read_bytes()).hexdigest()
                           for f in (default, *by.values())}}


def main() -> None:
    args = parse_args()
    physics = json.loads(args.physics_file.read_text()) if args.physics_file else args.physics
    model_for, router = model_router(args.model)
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit("output must be new or empty; existing studies are never overwritten")
    config = GameConfig(steps=args.steps, step_seconds=args.step_seconds,
                        physics_step=args.physics_step, passes=passes_named(args.passes))
    if args.commands == "relative" and describe(physics)["name"] != "agile":
        raise SystemExit("relative commands need --physics agile")
    source = json.loads(args.states.read_text())
    bg_meta = source.get("background", {})
    if bg_meta and (bg_meta.get("steps") != args.steps
                    or abs(float(bg_meta.get("step_seconds", 0)) - args.step_seconds) > 1e-9):
        raise SystemExit(f"background tracks were built for {bg_meta.get('steps')} x "
                         f"{bg_meta.get('step_seconds')} s, not {args.steps} x {args.step_seconds} s")
    records = source["states"]
    if args.indices:
        wanted = {int(i) for i in args.indices.split(",")}
        records = [r for r in records if r["index"] in wanted]
    elif args.limit:
        records = records[:args.limit]
    for folder in ("states", "policies", "figures", "clips"):
        (args.output / folder).mkdir(parents=True, exist_ok=True)
    manifest = dict(
        created_utc=datetime.now(timezone.utc).isoformat(), config=asdict(config),
        physics=describe(physics), pass_model_router=router,
        commands=args.commands, threat=args.threat, passes=args.passes, pass_reaction_s=args.pass_reaction_s,
        multi_pass=args.multi_pass, background_tackles=args.background_tackles, terminal=args.terminal,
        workers=args.workers, states=len(records), driver="andrew-passer2on1/scripts/run.py",
        game="Background2on1Game", horizon_seconds=args.steps * args.step_seconds,
        state_file_sha256=hashlib.sha256(args.states.read_bytes()).hexdigest(),
        model_sha256=hashlib.sha256(args.model.read_bytes()).hexdigest(),
        versions=dict(python=platform.python_version(), numpy=np.__version__,
                      scipy=scipy.__version__))
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.output / "starting_states.json").write_text(json.dumps({"states": records}) + "\n")
    start, rows = time.perf_counter(), []
    with ProcessPoolExecutor(max_workers=args.workers,
                             mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = {pool.submit(solve_one, r, model_for(r), args.allow_proxy_labels,
                               config, str(args.output), physics, args.commands, args.threat,
                               args.pass_reaction_s, args.multi_pass, args.background_tackles,
                               args.terminal): r["index"]
                   for r in records}
        failed = []
        for fut in as_completed(futures):
            # one game the solver cannot certify must not throw away the others (2026-09-29: a night
            # run lost its reading step to one of 3,095 games missing the 1e-8 check by a hair)
            try:
                row = fut.result()
            except Exception as exc:  # noqa: BLE001 -- recorded and reported, never hidden
                failed.append({"index": futures[fut], "error": f"{type(exc).__name__}: {exc}"})
                print(f"FAILED state {futures[fut]:03d}: {failed[-1]['error']}", flush=True)
                continue
            rows.append(row)
            print(f"{len(rows)}/{len(records)}: state {row['index']:03d} "
                  f"value={row['value']:.6f} gap={row['certificate']['gap']:.2g} "
                  f"compute={row['timing']['compute_seconds']:.1f}s", flush=True)
    rows.sort(key=lambda r: r["index"])
    summary = dict(
        states=len(rows), horizon_seconds=args.steps * args.step_seconds,
        wall_seconds=time.perf_counter() - start,
        total_cpu_seconds=sum(r["timing"]["cpu_seconds"] for r in rows),
        max_certificate_gap=max(r["certificate"]["gap"] for r in rows),
        max_local_gap=max(r["max_local_gap"] for r in rows),
        policy_bytes=sum(r["policy_bytes"] for r in rows),
        failed=sorted(failed, key=lambda f: f["index"]))
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
