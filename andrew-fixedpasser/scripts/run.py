#!/usr/bin/env python3
"""Solve the 3v1 fixed-passer game over a states file.

Writes the same study layout as `defensive_positioning.exact_study` and our
`scripts/run_stage3.py` -- manifest.json, starting_states.json, rows.jsonl,
summary.json, states/, policies/ -- so `scripts/summarise_stage3.py` and the
review pages read it unchanged. The imported code is not modified.
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

from fixedpasser.agile_motion import describe
from fixedpasser.solve import solve_one


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
    model_for, router = model_router(args.model)
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit("output must be new or empty; existing studies are never overwritten")
    config = GameConfig(steps=args.steps, step_seconds=args.step_seconds,
                        physics_step=args.physics_step)
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
        physics=describe(args.physics), pass_model_router=router,
        workers=args.workers, states=len(records), driver="andrew-fixedpasser/scripts/run.py",
        game="FixedPasserGame", horizon_seconds=args.steps * args.step_seconds,
        state_file_sha256=hashlib.sha256(args.states.read_bytes()).hexdigest(),
        model_sha256=hashlib.sha256(args.model.read_bytes()).hexdigest(),
        versions=dict(python=platform.python_version(), numpy=np.__version__,
                      scipy=scipy.__version__))
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.output / "starting_states.json").write_text(json.dumps({"states": records}) + "\n")
    start, rows = time.perf_counter(), []
    with ProcessPoolExecutor(max_workers=args.workers,
                             mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = [pool.submit(solve_one, r, model_for(r), args.allow_proxy_labels,
                               config, str(args.output), args.physics) for r in records]
        for fut in as_completed(futures):
            row = fut.result()
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
        policy_bytes=sum(r["policy_bytes"] for r in rows))
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
