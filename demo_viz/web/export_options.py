#!/usr/bin/env python3
"""Every feasible option at every solved moment, ranked as the paper ranks it.

The 2026-10-01 bundle's README states the rule outright: each option is valued
by "choosing this option and leaving the rest at equilibrium", which the panel
files carry as `slot_values`, and ties take the best position. That reproduces
`analysis/players.csv`'s stored `rank` for **198 of 203** rankable decisions --
every defender, every runner, every teammate. The five misses are all 2v1 ball
carriers and all exactly +1, which is the pass: his option set is six, and the
pass's value is not in `slot_values`.

So the carrier's moves are exported with `rank_partial`, saying their rank is
among his five moves and that a sixth option exists whose value the bundle does
not publish. Nothing is invented to fill it.

Which slot holds which role depends on the game, and getting this wrong is what
made an earlier audit believe the ranking was unreproducible:

    2v1   carrier slot = the ball carrier, receiver slot = the runner
    3v1   carrier slot = the RUNNER,       receiver slot = the teammate

Per option this writes the value, the rank, the equilibrium probability, the
solver's own 0.6 s path and its command name -- all read, none recomputed.

Usage:
    python -m demo_viz.web.export_options
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

from ..paper_story.bundle import BUNDLE, DEFAULT_ROOT, roots, to_scene_xy
from .export_data import WEB_DATA, slug

#: (game kind, the panel's slot) -> the role the demo calls it.
SLOT_ROLE = {
    ("2v1", "carrier"): "ball carrier",
    ("2v1", "receiver"): "runner",
    ("2v1", "defender"): "defender",
    ("3v1", "carrier"): "runner",
    ("3v1", "receiver"): "beneficiary",
    ("3v1", "defender"): "defender",
}

#: and which `slot_values` array holds that slot's option values.
SLOT_VALUES = {"carrier": "carrier_slot_values",
               "receiver": "receiver_slot_values",
               "defender": "defender_values"}


def _rank(values: list[float], index: int, minimising: bool) -> int:
    """Competition rank: ties take the best position, as the bundle does."""

    mine = values[index]
    better = (sum(1 for v in values if v < mine) if minimising
              else sum(1 for v in values if v > mine))
    return better + 1


def moment_options(panel: dict, observed: dict) -> dict:
    """One solved moment's options, per role, ranked."""

    kind = panel["kind"]
    values = panel.get("slot_values") or {}
    out = {}
    for slot, body in (panel.get("bodies") or {}).items():
        role = SLOT_ROLE.get((kind, slot))
        vals = list(values.get(SLOT_VALUES.get(slot, ""), []) or [])
        options = body.get("options") or []
        if not role or not vals or len(vals) != len(options):
            continue
        # the defender minimises the attack's value; everyone else maximises
        minimising = role == "defender"
        rows = []
        for index, option in enumerate(options):
            rows.append({
                "command": option.get("command", index),
                "label": option.get("label", ""),
                "value": round(float(vals[index]), 6),
                "rank": _rank(vals, index, minimising),
                "prob": round(float(option.get("prob", 0.0)), 6),
                "path": [to_scene_xy(p) for p in (option.get("path") or [])][::4],
                "end": to_scene_xy(option["end"]) if option.get("end") else None,
                "rests": bool(option.get("rests")),
            })
        rows.sort(key=lambda r: (r["rank"], r["command"]))
        entry = {"options": rows, "n_shown": len(rows)}
        # the observed decision for this role at this moment, as analyze_eval
        # recorded it -- including the carrier's sixth option, the pass
        seen = observed.get(role)
        if seen:
            entry["observed"] = seen
            if seen.get("n") and int(seen["n"]) > len(rows):
                entry["rank_partial"] = True
                entry["missing_option"] = ("the pass, whose value the bundle "
                                           "does not publish")
        out[role] = entry
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=WEB_DATA / "options")
    args = parser.parse_args(argv)

    root = DEFAULT_ROOT
    if not (root / "analysis" / "players.csv").exists():
        print(f"no {BUNDLE} bundle at {root}; nothing to export", file=sys.stderr)
        return 1
    folders = [folder for _, folder in roots(root)]     # the first bundle, then any later

    players = {}
    for folder in folders:
        with (folder / "analysis" / "players.csv").open() as handle:
            for row in csv.DictReader(handle):
                if row["role"] in players.get(row["code"], {}):
                    continue                            # an earlier bundle keeps the code
                players.setdefault(row["code"], {})[row["role"]] = {
                    "action": row["observed"],
                    "rank": int(row["rank"]),
                    "rank_fractional": float(row["rank_fractional"]),
                    "n": int(row["n"]),
                    "tied": int(row["tied"]),
                    "equilibrium_probability": float(row["similarity"]),
                    "regret": float(row["regret"]),
                }

    showcase = json.loads((WEB_DATA.parent / "data" / "submission_showcase.json").read_text())
    rows = showcase["scenes"] if isinstance(showcase, dict) else showcase
    by_scene: dict[str, dict] = {}
    for entry in rows:
        code, scene_id = entry.get("showcase_id"), entry.get("scene_id")
        if not code or not scene_id:
            continue
        moments = []
        for suffix, dt in (("", 0.0), ("@0.6", 0.6), ("@1.2", 1.2)):
            # the first bundle that has the moment (a later one may add moments to a scene)
            panel_file = next((f / "panels" / f"eval-{code}{suffix}.json" for f in folders
                               if (f / "panels" / f"eval-{code}{suffix}.json").exists()), None)
            if panel_file is None:
                continue
            panel = json.loads(panel_file.read_text())
            moments.append({
                "dt": dt,
                "roles": moment_options(panel, players.get(f"{code}{suffix}", {})),
            })
        if moments:
            by_scene[scene_id] = {"schema": "options/1", "code": code,
                                  "moments": moments}

    args.out.mkdir(parents=True, exist_ok=True)
    total = 0
    for scene_id, payload in sorted(by_scene.items()):
        path = args.out / f"{slug(scene_id)}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
        total += path.stat().st_size
        shown = sum(len(m["roles"]) for m in payload["moments"])
        print(f"{payload['code']:5} {len(payload['moments'])} moments, "
              f"{shown} role-moments, {path.stat().st_size / 1024:5.1f} KB")
    print(f"\n{len(by_scene)} files, {total / 1024:.0f} KB total -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
