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


def build(out: Path, data_dir: Path = WEB_DATA, clean: bool = True,
          include_legacy_obso: bool = False) -> Path:
    """Assemble the site.

    ``include_legacy_obso`` ships the OBSO surfaces as well. They are ~5.5 MB
    and nothing in the public interface can request them -- OBSO was demoted to
    a reference diagnostic when the demo went solver-native -- so the public
    build leaves them out. The OBSO parity tests pass it to get a site they can
    exercise the browser implementation against.
    """

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
            _split_showcase(showcase, target, data_dir / "solver")
        extras = ["solver", "release", "story", "compare", "grid", "options"]
        if include_legacy_obso:
            extras.append("obso")
        for extra in extras:
            source = data_dir / extra
            if source.exists():
                shutil.copytree(source, target / extra, dirs_exist_ok=True)
    return out


#: Curation a public first paint never reads: the reviewers' scores, their
#: written note, and the evidence behind the scene mapping. Only the mapping's
#: verdict is kept, because that is what decides whether a scene opens at all.
#: The explorer fetches the rest from `showcase_detail.json` when it opens.
DETAIL_FIELDS = ("review", "annotation")


def _solved_moments(solver_dir: Path) -> dict[str, int]:
    """Scene id -> how many moments the bundle solved a game at, from the files.

    The scene list leads with the scenes that have an equilibrium, so it has to
    know which those are before any of them is fetched. Read from the exported
    panels themselves rather than declared anywhere: a scene has an equilibrium
    exactly when its solver file holds solved panels.
    """

    out: dict[str, int] = {}
    if not solver_dir.exists():
        return out
    for path in sorted(solver_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if payload.get("kind") != "bundle_panels" or not payload.get("available"):
            continue
        scene_id = payload.get("scene_id")
        panels = payload.get("panels") or []
        if scene_id and panels:
            out[scene_id] = len(panels)
    return out


def _split_showcase(source: Path, target: Path, solver_dir: Path | None = None) -> None:
    """The curated list, lean for the page and whole for the explorer."""

    payload = json.loads(source.read_text())
    rows = payload["scenes"] if isinstance(payload, dict) else payload
    solved = _solved_moments(solver_dir) if solver_dir else {}
    detail = {}
    lean = []
    for row in rows:
        keep = {k: v for k, v in row.items() if k not in DETAIL_FIELDS}
        # how many solved moments this scene has, so the list can lead with
        # the scenes the two claims can actually be read in
        keep["solved_moments"] = solved.get(row.get("scene_id"), 0)
        mapping = row.get("mapping")
        if isinstance(mapping, dict):
            keep["mapping"] = {"status": mapping.get("status")}
        lean.append(keep)
        extra = {k: row[k] for k in DETAIL_FIELDS if k in row}
        if isinstance(mapping, dict):
            extra["mapping"] = mapping
        if extra and row.get("showcase_id"):
            detail[row["showcase_id"]] = extra
    body = ({**payload, "scenes": lean} if isinstance(payload, dict) else lean)
    (target / "submission_showcase.json").write_text(
        json.dumps(body, ensure_ascii=False, separators=(",", ":")))
    (target / "showcase_detail.json").write_text(
        json.dumps(detail, ensure_ascii=False, separators=(",", ":")))


#: Modules the page imports dynamically, each used by one story mode. They sit
#: in the build like any other file but are never fetched until that mode is
#: opened, so counting them in the first load would overstate it.
ON_DEMAND = ("js/policy.js", "js/evalstrip.js", "js/figure.js",
             "js/arrows.js", "js/obso.js", "js/release.js",
             "js/ranking.js", "js/chart.js", "js/compare.js", "js/grid.js",
             "js/panel.js", "js/labels.js", "js/options.js",
             # the reviewers' scores and notes: the explorer's, fetched there
             "data/showcase_detail.json",
             # the method sheet: prose, fetched when Details is opened
             "details.html")


def report(out: Path) -> str:
    rows = []
    total = 0
    for path in sorted(out.rglob("*")):
        if path.is_file():
            size = path.stat().st_size
            total += size
            rows.append((str(path.relative_to(out)), size))
    shell = sum(size for name, size in rows
                if not name.startswith("data/") and name not in ON_DEMAND)
    deferred = sum(size for name, size in rows if name in ON_DEMAND)
    scenes = [(name, size) for name, size in rows if name.startswith("data/")
              and not name.endswith("index.json")
              and not name.endswith("submission_showcase.json")
              and not name.startswith("data/obso/")
              and not name.startswith("data/solver/")
              and not name.startswith("data/release/")
              and not name.startswith("data/story/")
              and not name.startswith("data/compare/")
              and not name.startswith("data/grid/")
              and not name.startswith("data/options/")]
    obso = [(name, size) for name, size in rows if name.startswith("data/obso/")]
    solver = [(name, size) for name, size in rows if name.startswith("data/solver/")]
    release = [(name, size) for name, size in rows if name.startswith("data/release/")]
    story = [(name, size) for name, size in rows if name.startswith("data/story/")
             and not name.endswith("contract.json")]
    compare = [(name, size) for name, size in rows if name.startswith("data/compare/")]
    grid = [(name, size) for name, size in rows if name.startswith("data/grid/")]
    options = [(name, size) for name, size in rows if name.startswith("data/options/")]
    index = sum(size for name, size in rows
                if name.endswith("data/index.json")
                or name.endswith("submission_showcase.json")
                or name.endswith("data/story/contract.json"))
    legacy = ([f"obso        {len(obso)} files, "
               f"{sum(s for _, s in obso) / 1024 / 1024:.2f} MB total   "
               f"(legacy, included only with --include-legacy-obso)"]
              if obso else [])
    lines = [
        f"build       {out}",
        f"files       {len(rows)}",
        f"shell       {shell / 1024:8.1f} KB   (html + css + js, loaded once)",
        f"on demand   {deferred / 1024:8.1f} KB   (policy, evaluation strip and "
        f"figure conventions, imported when their mode is opened)",
        f"index       {index / 1024:8.1f} KB   (scene list, loaded once)",
        f"scenes      {len(scenes)} files, {sum(s for _, s in scenes) / 1024 / 1024:.2f} MB "
        f"total, {(sum(s for _, s in scenes) / max(len(scenes), 1)) / 1024:.0f} KB each "
        f"(loaded one at a time)",
        f"solver      {len(solver)} files, {sum(s for _, s in solver) / 1024:.0f} KB "
        f"total (only when the solver layer is picked)",
        f"story       {len(story)} files, {sum(s for _, s in story) / 1024:.0f} KB "
        f"total, {(sum(s for _, s in story) / max(len(story), 1)) / 1024:.0f} KB each "
        f"(paper-story contract, one at a time with the scene)",
        f"grid        {len(grid)} files, {sum(s for _, s in grid) / 1024:.0f} KB "
        f"total (the defender field, only when Nash shows it)",
        f"options     {len(options)} files, {sum(s for _, s in options) / 1024:.0f} KB "
        f"total (ranked actions, only when Player evaluation opens)",
        f"compare     {len(compare)} files, {sum(s for _, s in compare) / 1024:.0f} KB "
        f"total (tracking evidence, only when Dilemma compares players)",
        f"release     {len(release)} files, {sum(s for _, s in release) / 1024 / 1024:.2f} MB "
        f"total, {(sum(s for _, s in release) / max(len(release), 1)) / 1024:.0f} KB each "
        f"(only when the pass-model explorer is used)",
        *legacy,
        f"initial     {(shell + index) / 1024:8.1f} KB   shell + index",
        f"total       {total / 1024 / 1024:8.2f} MB",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--data", type=Path, default=WEB_DATA)
    parser.add_argument("--include-legacy-obso", action="store_true",
                        help="also ship the legacy OBSO surfaces (~5.5 MB). For "
                             "internal debugging and the OBSO parity tests; the "
                             "public interface cannot request them.")
    args = parser.parse_args(argv)

    if not (args.data / "index.json").exists():
        print(f"!! no data at {args.data}; run: python -m demo_viz.web.export_data",
              file=sys.stderr)
    out = build(args.out, args.data,
                include_legacy_obso=args.include_legacy_obso)
    print(report(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
