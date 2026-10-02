#!/usr/bin/env python3
"""Figure 2's defender-start grids, for the public Nash view.

Reads the flattened CSVs from the 2026-10-01 vector-field bundle -- one row per
lattice start, which is all the demo draws -- rather than the full solver JSONs
(8-10 MB each, mostly per-command 0.6 s paths the demo has no use for).

Two quantities, both the bundle's own, neither recomputed here:

    value   the equilibrium value with the defender starting at that point.
            Figure 2's background shade; LOWER is better for the defence.
    dx, dy  sum over his commands of p x (end - start): his probability-weighted
            displacement over the first 0.6 s. Figure 2's arrow.

Coordinates are already the demo's: centre-spot origin, metres, attack to the
right, the same frame as the 2026-09-30 tracking CSVs.

The tracking-derived bundle never leaves `local_inputs/`; what this writes is
a lattice of numbers with no player positions in it beyond the defender's own
candidate starts, which are a grid, not a track.

Usage:
    python -m demo_viz.web.export_grid \
        --bundle local_inputs/offball_vectorfield_ranking_20261001/vector_field
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "demo_viz.web"

from .export_data import WEB_DATA

#: `<code>_defender_grid.csv` -> the moment it belongs to, in seconds.
MOMENTS = {"": 0.0, "@0.6": 0.6, "@1.2": 1.2}


def read_grid(path: Path) -> dict:
    """One moment's lattice: the solved starts, their value and their move."""

    points = []
    for row in csv.DictReader(path.open()):
        if row["status"] != "solved":
            continue
        points.append({
            "i": int(row["i"]),
            "j": int(row["j"]),
            "x": round(float(row["x"]), 3),
            "y": round(float(row["y"]), 3),
            "v": round(float(row["value"]), 5),
            "dx": round(float(row["dx"]), 4),
            "dy": round(float(row["dy"]), 4),
            **({"observed": True} if row.get("observed") == "1" else {}),
        })
    values = [p["v"] for p in points]
    return {
        "schema": "grid/1",
        "spacing_m": 1.0,
        "points": points,
        "value": {"min": min(values), "max": max(values)} if values else None,
        "source": {
            "bundle": "offball_vectorfield_ranking_20261001",
            "file": path.name,
            "value": "equilibrium value with the defender starting here; "
                     "lower is better for the defence",
            "move": "sum of p x (end - start) over his five commands, over 0.6 s",
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--code", default="S05")
    parser.add_argument("--out", type=Path, default=WEB_DATA / "grid")
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    written = total = 0
    for suffix, dt in sorted(MOMENTS.items(), key=lambda kv: kv[1]):
        source = args.bundle / f"{args.code}{suffix}_defender_grid.csv"
        if not source.exists():
            print(f"!! missing {source}", file=sys.stderr)
            continue
        payload = read_grid(source)
        payload["code"] = args.code
        payload["dt"] = dt
        path = args.out / f"{args.code}_{dt:.1f}.json"
        path.write_text(json.dumps(payload, separators=(",", ":")))
        size = path.stat().st_size
        total += size
        written += 1
        print(f"{path.name:20} {len(payload['points']):4} starts  {size / 1024:6.1f} KB")

    print(f"\n{written} grids, {total / 1024:.0f} KB total -> {args.out}")
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())
