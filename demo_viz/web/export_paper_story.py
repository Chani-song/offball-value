#!/usr/bin/env python3
"""Export the paper-story payload for each scene.

    python -m demo_viz.web.export_paper_story
    python -m demo_viz.web.export_paper_story --limit 3

Writes ``demo_viz/web_data/story/contract.json`` (the vocabulary, loaded once
with the shell) and ``demo_viz/web_data/story/<scene>.json`` (lazy, one at a
time with the scene).

WHAT IT WRITES TODAY
--------------------
Mostly explicit unavailable states, and that is the point. The player
evaluation the abstract describes is not implemented in the research code yet
(``demo_viz/PAPER_STORY_TRACE.md``), so every reserved field comes back with a
reason attached rather than a number. The interface renders those reasons.

Nothing here computes a paper quantity. It asks
:mod:`demo_viz.paper_story.adapter` for the scene and writes what it gets, so
when the evaluation pipeline lands the only thing that changes is the source
the adapter reads -- not this file, and not the browser.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    __package__ = "demo_viz.web"

from ..core.showcase import load as load_showcase  # noqa: E402
from ..paper_story import (  # noqa: E402
    Metric,
    build_scene,
    contract_payload,
    discover_source,
    equilibrium_for,
    validate_payload,
)
from .export_data import WEB_DATA, slug  # noqa: E402

STORY_DIR = WEB_DATA / "story"


def action_value_metric(scene_id: str, release_dir: Path) -> Metric:
    """Whether the solver's release chain was exported for this scene.

    The value is the chain itself, which is what the drill-down is about. It
    is available only when ``export_release.py`` actually ran here -- that
    needs the research stack and the fitted model, neither of which is
    committed.
    """

    path = Path(release_dir) / f"{slug(scene_id)}.json"
    if not path.exists():
        return Metric(
            name="action_value", label="Action value details",
            availability="artifact_missing",
            detail="The pass model was not exported for this build.")
    return Metric(
        name="action_value", label="Action value details",
        availability="available",
        value="legal × completion proxy × positional threat",
        provenance="solver", source="demo_viz/web/export_release.py",
        definition_version="release_payoffs_with_background")


def story_titles(showcase) -> dict[str, tuple[str | None, str | None]]:
    """Curated titles, keyed by scene id. Absent for uncurated scenes."""

    out: dict[str, tuple[str | None, str | None]] = {}
    for entry in showcase or ():
        if entry.scene_id:
            out[entry.scene_id] = (entry.story_title or None,
                                   entry.story_summary or None)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=STORY_DIR)
    parser.add_argument("--data", type=Path, default=WEB_DATA)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    index_path = args.data / "index.json"
    if not index_path.exists():
        print(f"!! no scene index at {index_path}; run export_data first",
              file=sys.stderr)
        return 1
    scenes = [entry["scene_id"] for entry in json.loads(index_path.read_text())["scenes"]]
    if args.limit:
        scenes = scenes[: args.limit]

    try:
        titles = story_titles(load_showcase())
    except Exception as error:                              # pragma: no cover
        print(f"!! showcase unreadable ({error}); exporting without titles",
              file=sys.stderr)
        titles = {}

    source = discover_source()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "contract.json").write_text(json.dumps(contract_payload(), indent=2))

    total = 0
    for scene_id in scenes:
        title, summary = titles.get(scene_id, (None, None))
        story = build_scene(
            scene_id,
            source=source,
            equilibrium=equilibrium_for(scene_id, args.data / "solver"),
            action_value=action_value_metric(scene_id, args.data / "release"),
            story_title=title, story_summary=summary,
        )
        payload = validate_payload(story.to_payload())
        path = args.out / f"{slug(scene_id)}.json"
        path.write_text(json.dumps(payload, separators=(",", ":"),
                                   ensure_ascii=False))
        total += path.stat().st_size

    print(f"evaluation source: {source.name} ({source.definition_version})")
    print(f"{len(scenes)} story files, {total / 1024:.0f} KB total, "
          f"{total / max(len(scenes), 1) / 1024:.0f} KB each (lazy) -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
