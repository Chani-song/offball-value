#!/usr/bin/env python3
"""Concatenate a sweep arm's per-match CSVs into the layout the scorer wants.

score_against_chani2.py reads one attacking_phases.csv and one
shot_context_run_onsets.csv from a root directory; the sweeps write one of
each per match. This is that join, and nothing else -- no filtering, so a
mis-scored arm is the arm and not this step.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

NEEDED = ("attacking_phases.csv", "shot_context_run_onsets.csv")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("arm_dir", type=Path)
    args = p.parse_args()
    matches = sorted(d for d in args.arm_dir.iterdir() if d.is_dir())
    if not matches:
        print(f"{args.arm_dir}: 경기 디렉토리 없음")
        return 1
    for name in NEEDED:
        parts = [pd.read_csv(d / name) for d in matches if (d / name).exists()]
        if not parts:
            print(f"{args.arm_dir}: {name} 없음")
            return 1
        merged = pd.concat(parts, ignore_index=True)
        merged.to_csv(args.arm_dir / name, index=False)
        print(f"  {args.arm_dir.name}/{name}: {len(merged)}행 ({len(parts)}경기)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
