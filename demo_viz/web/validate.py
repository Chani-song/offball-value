#!/usr/bin/env python3
"""Check the browser maths against the Python implementation.

    python -m demo_viz.web.validate                  # 60 random cases
    python -m demo_viz.web.validate --cases 200 --tolerance 1e-9

Picks random scenes, frames, runners, defenders and beneficiaries; computes the
residual-space values in Python; runs the same cases through ``js/influence.js``
in headless Chrome; compares. The browser is the thing being tested, so the
comparison has to actually execute the JavaScript rather than re-read it.
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import http.server
import json
import random
import shutil
import socketserver
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    __package__ = "demo_viz.web"

from ..core.influence import InfluenceCache
from ..loader import load_scene
from .build import WEB_DATA, build
from .export_data import slug

HERE = Path(__file__).resolve().parent

CHROME_CANDIDATES = (
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/usr/bin/google-chrome",
    "/usr/bin/chromium",
    "/usr/bin/chromium-browser",
)

SAMPLE_EVERY = 5


def find_chrome() -> str | None:
    for candidate in CHROME_CANDIDATES:
        if Path(candidate).exists():
            return candidate
    return shutil.which("google-chrome") or shutil.which("chromium")


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):        # the request log is noise here
        pass


@contextlib.contextmanager
def serve(directory: Path):
    handler = functools.partial(_QuietHandler, directory=str(directory))
    with socketserver.TCPServer(("127.0.0.1", 0), handler) as httpd:
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        try:
            yield httpd.server_address[1]
        finally:
            httpd.shutdown()


def build_cases(scene_ids: list[str], count: int, seed: int) -> tuple[list[dict], dict]:
    """Random role triplets, with the Python answer attached."""

    rng = random.Random(seed)
    cases: list[dict] = []
    expected: dict[str, dict] = {}
    per_scene = max(1, count // max(len(scene_ids), 1))

    for scene_id in scene_ids:
        scene = load_scene(scene_id, quantities=False, surfaces=False)
        cache = InfluenceCache(scene, every=SAMPLE_EVERY, grid_resolution_m=2.0)
        attackers = [p.player_id for p in scene.players_on("attack") if not p.is_goalkeeper]
        defenders = [p.player_id for p in scene.players_on("defend") if not p.is_goalkeeper]
        if not attackers or not defenders:
            continue
        for _ in range(per_scene):
            slot = rng.randrange(len(cache.indices))
            frame_index = int(cache.indices[slot])
            n_ben = rng.choice([1, 1, 2])
            n_def = rng.choice([1, 1, 2])
            beneficiaries = rng.sample(attackers, min(n_ben, len(attackers)))
            chosen = rng.sample(defenders, min(n_def, len(defenders)))
            freeze_index = rng.randrange(0, scene.n_frames)
            baseline = rng.choice(["hold", "drift"])
            key = f"{scene_id}|{len(cases)}"

            swap = tuple((d, freeze_index, baseline) for d in chosen)
            factual = cache.combined(beneficiaries, slot)
            counter = cache.combined(beneficiaries, slot, swap)
            expected[key] = {
                "factual": factual.value,
                "counterfactual": counter.value,
                "gain": factual.value - counter.value,
                "peak": float(np.max(factual.field)),
            }
            cases.append({
                "key": key,
                "file": f"{slug(scene_id)}.json",
                "slot": slot,
                "frame_index": frame_index,
                "beneficiaries": beneficiaries,
                "defenders": list(chosen),
                "freeze_index": freeze_index,
                "baseline": baseline,
            })
    return cases, expected


def run_browser(build_dir: Path, chrome: str, timeout: int = 180) -> dict:
    with serve(build_dir) as port:
        command = [
            chrome, "--headless", "--disable-gpu", "--no-sandbox",
            "--virtual-time-budget=120000", "--dump-dom",
            f"http://127.0.0.1:{port}/harness.html",
        ]
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    dom = result.stdout
    start = dom.find('<pre id="out">')
    if start < 0:
        raise RuntimeError("harness did not render; is Chrome available?")
    start = dom.index(">", start) + 1
    end = dom.index("</pre>", start)
    payload = dom[start:end].strip()
    payload = payload.replace("&quot;", '"').replace("&amp;", "&").replace("&lt;", "<")
    if payload == "running":
        raise RuntimeError("harness timed out before finishing")
    return json.loads(payload)


def run_bench(index, args, chrome: str) -> int:
    """Browser latency: scene fetch, cache build, and role switching."""

    rng = random.Random(args.seed)
    files = [row["file"] for row in rng.sample(index, min(args.scenes, len(index)))]
    with tempfile.TemporaryDirectory() as tmp:
        build_dir = build(Path(tmp) / "site", args.data)
        # Inline the payloads and keep the work synchronous: with
        # --virtual-time-budget the clock stops during async work, and without
        # it --dump-dom fires before async work finishes.
        plan = {"scenes": [
            {"json": (args.data / name).read_text()} for name in files
        ]}
        page = (HERE / "bench.html").read_text().replace(
            "<body>",
            '<body><script type="application/json" id="plan">'
            + json.dumps(plan).replace("</", "<\\/")
            + "</script>",
        )
        (build_dir / "bench.html").write_text(page)
        with serve(build_dir) as port:
            command = [
                chrome, "--headless", "--disable-gpu", "--no-sandbox", "--dump-dom",
                f"http://127.0.0.1:{port}/bench.html",
            ]
            result = subprocess.run(command, capture_output=True, text=True, timeout=300)
    dom = result.stdout
    start = dom.index(">", dom.find('<pre id="out">')) + 1
    payload = json.loads(dom[start:dom.index("</pre>", start)].strip())
    if "error" in payload:
        print(f"!! browser error: {payload['error']}", file=sys.stderr)
        return 1

    fields = ["parse_ms", "cache_ms", "first_residual_ms", "role_switch_ms",
              "rank_defenders_ms", "rank_beneficiaries_ms"]
    print(f"browser latency over {len(payload['rows'])} scenes (median / worst, ms)")
    for field in fields:
        values = sorted(row[field] for row in payload["rows"])
        median = values[len(values) // 2]
        print(f"  {field:<24} {median:8.1f} {values[-1]:8.1f}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cases", type=int, default=60)
    parser.add_argument("--scenes", type=int, default=6)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--tolerance", type=float, default=1e-9,
                        help="relative tolerance on each scalar")
    parser.add_argument("--atol", type=float, default=1e-9,
                        help="absolute tolerance, in space units")
    parser.add_argument("--data", type=Path, default=WEB_DATA)
    parser.add_argument("--bench", action="store_true",
                        help="measure browser latency instead of checking values")
    args = parser.parse_args(argv)

    chrome = find_chrome()
    if chrome is None:
        print("!! no Chrome/Chromium found; cannot run the browser side", file=sys.stderr)
        return 2

    index_path = args.data / "index.json"
    if not index_path.exists():
        print(f"!! no exported data at {args.data}; run: "
              f"python -m demo_viz.web.export_data", file=sys.stderr)
        return 2
    index = json.loads(index_path.read_text())["scenes"]
    if args.bench:
        return run_bench(index, args, chrome)
    rng = random.Random(args.seed)
    scene_ids = [row["scene_id"] for row in rng.sample(index, min(args.scenes, len(index)))]
    print(f"scenes: {', '.join(scene_ids)}")

    cases, expected = build_cases(scene_ids, args.cases, args.seed)
    print(f"cases:  {len(cases)}")

    with tempfile.TemporaryDirectory() as tmp:
        build_dir = build(Path(tmp) / "site", args.data)
        shutil.copy2(HERE / "harness.html", build_dir / "harness.html")
        (build_dir / "fixtures.json").write_text(
            json.dumps({"every": SAMPLE_EVERY, "cases": cases})
        )
        payload = run_browser(build_dir, chrome)

    if "error" in payload:
        print(f"!! browser error: {payload['error']}", file=sys.stderr)
        return 1

    fields = ("factual", "counterfactual", "gain", "peak")
    worst_abs = {field: 0.0 for field in fields}
    worst_rel = {field: 0.0 for field in fields}
    failures = []
    for row in payload["results"]:
        reference = expected[row["key"]]
        for field in fields:
            want = reference[field]
            got = row[field]
            if got is None:
                continue
            absolute = abs(got - want)
            worst_abs[field] = max(worst_abs[field], absolute)
            # `gain` is a difference of two similar numbers, so a pure relative
            # measure blows up when the true gain is ~0; judge it the way
            # numpy.isclose does, with an absolute floor.
            if abs(want) > 1e-6:
                worst_rel[field] = max(worst_rel[field], absolute / abs(want))
            if absolute > args.atol + args.tolerance * abs(want):
                failures.append((row["key"], field, want, got, absolute))

    print("\nbrowser vs python, worst over all cases")
    print(f"  {'quantity':<16} {'abs':>12} {'rel':>12}")
    for field in fields:
        print(f"  {field:<16} {worst_abs[field]:12.3e} {worst_rel[field]:12.3e}")
    if failures:
        print(f"\n{len(failures)} case(s) outside atol={args.atol:g} + rtol={args.tolerance:g}:")
        for key, field, want, got, absolute in failures[:10]:
            print(f"  {key} {field}: python={want:.12f} js={got:.12f} abs={absolute:.2e}")
        return 1
    print(f"\nall {len(payload['results'])} cases agree "
          f"(atol={args.atol:g}, rtol={args.tolerance:g})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
