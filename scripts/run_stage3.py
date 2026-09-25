#!/usr/bin/env python3
"""Run the imported exact 2v1 solver with a turn length we choose.

`defensive_positioning.exact_study` fixes the game at three turns of 0.4 s --
`GameConfig(steps=3, step_seconds=.4, ...)` on one line of `main()`, with no
flag. That is 1.2 s of play. Our scenes are the 2.96 s after a run's onset,
so solving his default answers a question about the first 40% of the run,
before the run has usually got anywhere.

State count depends only on the number of turns (5^(3k) joint command
histories), so three turns of 1.0 s is the same size of game as three of
0.4 s and covers the whole scene. What it costs is resolution: the defender
re-decides at 0, 1 and 2 s rather than every 0.4 s. Four turns would give
both and is out of reach (5^12 = 244M states against a 2.1M bound).

This file does not modify the imported code. It calls the same `solve_one`
with a config of its own and writes the same artefacts, so everything that
reads an exact_study output reads this one.
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

from defensive_positioning.exact_study import overview, solve_one
from defensive_positioning.models import GameConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--states", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--allow-proxy-labels", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--step-seconds", type=float, default=1.0)
    p.add_argument("--physics-step", type=float, default=0.025)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--clips", action="store_true",
                   help="render mp4 clips for one state per stratum, as exact_study does")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit("output must be new or empty; existing studies are never overwritten")
    config = GameConfig(steps=args.steps, step_seconds=args.step_seconds,
                        physics_step=args.physics_step)
    records = json.loads(args.states.read_text())["states"]
    if args.limit:
        records = records[:args.limit]
    for folder in ("states", "policies", "figures", "clips"):
        (args.output / folder).mkdir(parents=True, exist_ok=True)
    manifest = dict(
        created_utc=datetime.now(timezone.utc).isoformat(), config=asdict(config),
        workers=args.workers, states=len(records), driver="scripts/run_stage3.py",
        horizon_seconds=args.steps * args.step_seconds,
        state_file_sha256=hashlib.sha256(args.states.read_bytes()).hexdigest(),
        model_sha256=hashlib.sha256(args.model.read_bytes()).hexdigest(),
        versions=dict(python=platform.python_version(), numpy=np.__version__,
                      scipy=scipy.__version__))
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.output / "starting_states.json").write_text(json.dumps({"states": records}, indent=2) + "\n")
    selected = ({next(r["index"] for r in records if r["stratum"] == s)
                 for s in {r["stratum"] for r in records}} if args.clips else set())
    start, rows = time.perf_counter(), []
    with ProcessPoolExecutor(max_workers=args.workers,
                             mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = [pool.submit(solve_one, r, str(args.model), args.allow_proxy_labels,
                               config, str(args.output), r["index"] in selected)
                   for r in records]
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
    overview(rows, args.output / "figures" / "overview.png")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
