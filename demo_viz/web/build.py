#!/usr/bin/env python3
"""Assemble the static site into a deployable folder.

    python -m demo_viz.web.build                  # -> demo_viz/web_build/
    python -m demo_viz.web.build --out /tmp/site

The result is plain files: no bundler, no node, no runtime dependencies. The
GitHub Pages workflow uploads exactly this folder.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    __package__ = "demo_viz.web"

HERE = Path(__file__).resolve().parent
SITE = HERE / "site"
WEB_DATA = HERE.parent / "web_data"
DEFAULT_OUT = HERE.parent / "web_build"


def build(out: Path, data_dir: Path = WEB_DATA, clean: bool = True) -> Path:
    if clean and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copytree(SITE, out, dirs_exist_ok=True)
    target = out / "data"
    target.mkdir(parents=True, exist_ok=True)
    if data_dir.exists():
        for path in sorted(data_dir.glob("*.json")):
            shutil.copy2(path, target / path.name)
        # the OBSO surfaces live in their own folder and are fetched only when
        # a visitor picks the threat view, so they never touch the first load
        # curation metadata: small, loaded once with the scene index
        showcase = HERE.parent / "data" / "submission_showcase.json"
        if showcase.exists():
            shutil.copy2(showcase, target / "submission_showcase.json")
        # "obso" is the legacy reference stack: no longer reachable from the
        # public UI, kept so its parity tests still run against a built site.
        for extra in ("obso", "solver", "release"):
            source = data_dir / extra
            if source.exists():
                shutil.copytree(source, target / extra, dirs_exist_ok=True)
    return out


def report(out: Path) -> str:
    rows = []
    total = 0
    for path in sorted(out.rglob("*")):
        if path.is_file():
            size = path.stat().st_size
            total += size
            rows.append((str(path.relative_to(out)), size))
    shell = sum(size for name, size in rows if not name.startswith("data/"))
    scenes = [(name, size) for name, size in rows if name.startswith("data/")
              and not name.endswith("index.json")
              and not name.endswith("submission_showcase.json")
              and not name.startswith("data/obso/")
              and not name.startswith("data/solver/")
              and not name.startswith("data/release/")]
    obso = [(name, size) for name, size in rows if name.startswith("data/obso/")]
    solver = [(name, size) for name, size in rows if name.startswith("data/solver/")]
    release = [(name, size) for name, size in rows if name.startswith("data/release/")]
    index = sum(size for name, size in rows
                if name.endswith("data/index.json")
                or name.endswith("submission_showcase.json"))
    lines = [
        f"build       {out}",
        f"files       {len(rows)}",
        f"shell       {shell / 1024:8.1f} KB   (html + css + js, loaded once)",
        f"index       {index / 1024:8.1f} KB   (scene list, loaded once)",
        f"scenes      {len(scenes)} files, {sum(s for _, s in scenes) / 1024 / 1024:.2f} MB "
        f"total, {(sum(s for _, s in scenes) / max(len(scenes), 1)) / 1024:.0f} KB each "
        f"(loaded one at a time)",
        f"obso        {len(obso)} files, {sum(s for _, s in obso) / 1024 / 1024:.2f} MB "
        f"total, {(sum(s for _, s in obso) / max(len(obso), 1)) / 1024:.0f} KB each "
        f"(only when the threat view is picked)",
        f"solver      {len(solver)} files, {sum(s for _, s in solver) / 1024:.0f} KB "
        f"total (only when the solver layer is picked)",
        f"release     {len(release)} files, {sum(s for _, s in release) / 1024 / 1024:.2f} MB "
        f"total, {(sum(s for _, s in release) / max(len(release), 1)) / 1024:.0f} KB each "
        f"(only when the pass-model explorer is used)",
        f"initial     {(shell + index) / 1024:8.1f} KB   shell + index",
        f"total       {total / 1024 / 1024:8.2f} MB",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--data", type=Path, default=WEB_DATA)
    args = parser.parse_args(argv)

    if not (args.data / "index.json").exists():
        print(f"!! no data at {args.data}; run: python -m demo_viz.web.export_data",
              file=sys.stderr)
    out = build(args.out, args.data)
    print(report(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
