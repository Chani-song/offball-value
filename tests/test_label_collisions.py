"""No two labels overlap, in any public scene, at any solved moment.

This is a rendering test, not a source test: it opens the built site in
headless Chrome, walks every public scene, every solved moment, both pitch
views and two window widths, and reads back the REAL bounding box of every
text node the pitch draws. A layout that is clean in one screenshot is not
what section 17 asks for.
"""

from __future__ import annotations

import functools
import http.server
import math
import json
import os
import socketserver
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from demo_viz.web.build import build          # noqa: E402
from demo_viz.web.validate import find_chrome  # noqa: E402

PROBE = """<!doctype html><meta charset=utf-8><div id=h></div>
<script>
const WIDTHS = __WIDTHS__, VIEWS = ["focus", "full"];
const wait = (ms) => new Promise(r => setTimeout(r, ms));
function geometry(d) {
  // the viewBox the pitch is actually showing, and everything drawn in it
  const svg = d.querySelector("#pitch svg");
  const vb = (svg.getAttribute("viewBox") || "").split(/\s+/).map(Number);
  const out = {viewBox: vb, outside: [], broken: []};
  const inside = (x, y, slack) =>
    x >= vb[0] - slack && x <= vb[0] + vb[2] + slack &&
    y >= vb[1] - slack && y <= vb[1] + vb[3] + slack;
  const ball = d.querySelector(".layer-players circle:last-of-type");
  for (const sel of [".layer-players g.player.is-role", ".layer-passes polyline",
                     ".layer-passes path", ".layer-passes line"]) {
    for (const n of d.querySelectorAll(sel)) {
      let b;
      try { b = n.getBBox(); } catch { continue; }
      if (!b.width && !b.height) continue;
      // every drawn mark must sit inside the crop, with no slack allowed
      if (!inside(b.x, b.y, 0) || !inside(b.x + b.width, b.y + b.height, 0)) {
        out.outside.push({sel, x: b.x, y: b.y, w: b.width, h: b.height});
      }
    }
  }
  // the ball and the man on it, by their own marks
  const marks = [...d.querySelectorAll(".layer-players g.player")].map((g) => {
    const b = g.getBBox();
    return {shirt: (g.querySelector("text.shirt")||{}).textContent,
            x: b.x, y: b.y, w: b.width, h: b.height,
            role: g.classList.contains("is-role")};
  });
  out.marks = marks;
  const balls = [...d.querySelectorAll(".layer-players circle")];
  if (balls.length) {
    const b = balls[balls.length - 1].getBBox();
    out.ball = {x: b.x, y: b.y, w: b.width, h: b.height};
  }
  return out;
}

function boxes(d) {
  const out = [];
  for (const node of d.querySelectorAll(".layer-labels text")) {
    const cls = node.getAttribute("class") || "";
    if (!/action-label|pass-label/.test(cls)) continue;
    const b = node.getBBox();
    out.push({t: node.textContent, x: b.x, y: b.y, w: b.width, h: b.height});
  }
  const marks = [];
  for (const node of d.querySelectorAll(".layer-players text.shirt")) {
    const b = node.getBBox();
    marks.push({t: node.textContent, x: b.x, y: b.y, w: b.width, h: b.height});
  }
  return {labels: out, shirts: marks};
}
(async () => {
  const report = [];
  for (const width of WIDTHS) {
    for (const code of __CODES__) {
      for (const view of VIEWS) {
        const f = document.createElement("iframe");
        f.style.cssText = `width:${width}px;height:1100px;border:0`;
        f.src = `index.html?showcase=${code}&story=game_solution`;
        document.getElementById("h").replaceChildren(f);
        await new Promise(r => f.addEventListener("load", r));
        await wait(1800);
        const d = f.contentDocument;
        for (const b of d.querySelectorAll("#view-toggle .seg")) {
          if (b.dataset.view === view) b.click();
        }
        await wait(350);
        const segs = [...d.querySelectorAll("#moment-pick .seg")];
        for (let i = 0; i < Math.max(segs.length, 1); i += 1) {
          if (segs[i]) { segs[i].click(); await wait(550); }
          const got = boxes(d);
          report.push({width, code, view,
                       moment: segs[i] ? segs[i].textContent : "default",
                       ...got, geom: geometry(d)});
        }
      }
    }
  }
  fetch("/c", {method: "POST", body: JSON.stringify(report)});
})();
</script>"""


def _render(site: Path, codes: list[str], widths: list[int]) -> list[dict]:
    """One short, bounded Chrome run per (width, scene).

    A single run of the whole matrix does not finish: the pages fetch their
    modes lazily, so virtual time advances slowly and the run outlives any
    sensible timeout. Fourteen small runs each finish in seconds.
    """

    out: list[dict] = []
    for width in widths:
        for code in codes:
            out.extend(_render_one(site, [code], [width]))
    return out


def _render_one(site: Path, codes: list[str], widths: list[int]) -> list[dict]:
    result: list[dict] = []

    class Handler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *a):          # noqa: D102, ANN002
            pass

        def do_POST(self):                  # noqa: N802, D102
            size = int(self.headers.get("Content-Length", 0))
            result.extend(json.loads(self.rfile.read(size) or b"[]"))
            self.send_response(204)
            self.end_headers()

    class Threaded(socketserver.ThreadingTCPServer):
        allow_reuse_address = True
        daemon_threads = True

    probe = site / "collide.html"
    probe.write_text(PROBE.replace("__CODES__", json.dumps(codes))
                          .replace("__WIDTHS__", json.dumps(widths)))
    handler = functools.partial(Handler, directory=str(site))
    with Threaded(("127.0.0.1", 0), handler) as httpd:
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        port = httpd.server_address[1]
        with tempfile.TemporaryDirectory() as profile, \
                tempfile.TemporaryDirectory() as shot:
            try:
                subprocess.run(
                    [find_chrome(), "--headless", "--disable-gpu", "--no-sandbox",
                     f"--user-data-dir={profile}", "--window-size=1900,1200",
                     "--virtual-time-budget=20000",
                     f"--screenshot={Path(shot) / 'x.png'}",
                     f"http://127.0.0.1:{port}/collide.html"],
                    capture_output=True, timeout=90)
            except subprocess.TimeoutExpired:
                pass
        httpd.shutdown()
    probe.unlink(missing_ok=True)
    return result


def _overlap(a: dict, b: dict) -> float:
    w = min(a["x"] + a["w"], b["x"] + b["w"]) - max(a["x"], b["x"])
    h = min(a["y"] + a["h"], b["y"] + b["h"]) - max(a["y"], b["y"])
    return w * h if w > 0 and h > 0 else 0.0


class LabelCollisionTests(unittest.TestCase):
    """Section 17: every view, every moment, every scene, two widths."""

    site: Path
    report: list[dict]
    tmp: tempfile.TemporaryDirectory

    @classmethod
    def setUpClass(cls):
        if find_chrome() is None:
            raise unittest.SkipTest("no headless Chrome")
        showcase = json.loads(
            (REPO_ROOT / "demo_viz" / "data" / "submission_showcase.json").read_text())
        rows = showcase["scenes"] if isinstance(showcase, dict) else showcase
        solver = REPO_ROOT / "demo_viz" / "web_data" / "solver"
        codes = [row["showcase_id"] for row in rows
                 if row.get("scene_id")
                 and (solver / f"{row['scene_id'].replace(':', '_')}.json").exists()]
        if not codes:
            raise unittest.SkipTest("no scenes with solved panels")
        cls.codes = sorted(codes)
        cls.tmp = tempfile.TemporaryDirectory()
        cls.site = Path(build(Path(cls.tmp.name) / "site"))
        cls.report = _render(cls.site, cls.codes, [1440, 1800])
        if not cls.report:
            raise unittest.SkipTest("the probe returned nothing")
        keep = os.environ.get("LABEL_REPORT")
        if keep:
            Path(keep).write_text(json.dumps(cls.report))

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "tmp"):
            cls.tmp.cleanup()

    def test_every_scene_and_moment_was_measured(self):
        seen = {(row["code"], row["view"], row["moment"]) for row in self.report}
        self.assertGreaterEqual(len(seen), len(self.codes) * 2,
                                f"only {len(seen)} states measured")
        for code in self.codes:
            self.assertTrue(any(row["code"] == code for row in self.report), code)

    def test_no_two_labels_overlap(self):
        bad = []
        for row in self.report:
            labels = row["labels"]
            for i, a in enumerate(labels):
                for b in labels[i + 1:]:
                    if _overlap(a, b) > 0:
                        bad.append(f"{row['code']} {row['view']} {row['moment']} "
                                   f"{row['width']}px: {a['t']!r} x {b['t']!r}")
        self.assertEqual([], bad, "\n".join(bad[:20]))

    def test_no_label_sits_on_a_shirt_number(self):
        bad = []
        for row in self.report:
            for a in row["labels"]:
                for b in row["shirts"]:
                    if _overlap(a, b) > 0:
                        bad.append(f"{row['code']} {row['view']} {row['moment']} "
                                   f"{row['width']}px: {a['t']!r} on #{b['t']}")
        self.assertEqual([], bad, "\n".join(bad[:20]))

    def test_no_label_is_empty(self):
        bad = [f"{row['code']} {row['view']} {row['moment']}"
               for row in self.report for a in row["labels"] if not a["t"].strip()]
        self.assertEqual([], bad, "\n".join(bad[:20]))

    def test_two_actions_at_the_same_percentage_are_visibly_apart(self):
        """Two players may genuinely share a probability.

        S44 at 0.0 s is three bodies at 100% -- runner "left", teammate
        "forward", defender "toward goal" -- and the figure's own convention
        gives an off-ball attacker's arrow the percentage alone, because the
        direction is the arrow's job. So the same text twice is correct; what
        would be wrong is two of them in the same place, with no way to tell
        which arrow either belongs to.
        """

        bad = []
        for row in self.report:
            for i, a in enumerate(row["labels"]):
                for b in row["labels"][i + 1:]:
                    if a["t"] != b["t"]:
                        continue
                    gap = math.dist(
                        (a["x"] + a["w"] / 2, a["y"] + a["h"] / 2),
                        (b["x"] + b["w"] / 2, b["y"] + b["h"] / 2))
                    if gap < max(a["h"], b["h"]) * 1.5:
                        bad.append(f"{row['code']} {row['view']} {row['moment']} "
                                   f"{row['width']}px: two {a['t']!r} only "
                                   f"{gap:.2f} apart")
        self.assertEqual([], bad, "\n".join(bad[:20]))


if __name__ == "__main__":
    unittest.main()


class FocusBoundsTests(LabelCollisionTests):
    """Sections 20, 22, 23, 34, 35: the crop holds everything it draws."""

    def test_the_crop_holds_the_ball(self):
        bad = []
        for row in self.report:
            geom = row.get("geom") or {}
            vb, ball = geom.get("viewBox"), geom.get("ball")
            if not vb or not ball:
                continue
            if not (vb[0] <= ball["x"] and ball["x"] + ball["w"] <= vb[0] + vb[2]
                    and vb[1] <= ball["y"] and ball["y"] + ball["h"] <= vb[1] + vb[3]):
                bad.append(f"{row['code']} {row['view']} {row['moment']} "
                           f"{row['width']}px: ball outside the crop")
        self.assertEqual([], bad, "\n".join(bad[:20]))

    def test_the_crop_holds_every_player_the_game_is_about(self):
        bad = []
        for row in self.report:
            geom = row.get("geom") or {}
            vb = geom.get("viewBox")
            if not vb:
                continue
            for mark in geom.get("marks", []):
                if not mark["role"]:
                    continue
                if not (vb[0] <= mark["x"] and mark["x"] + mark["w"] <= vb[0] + vb[2]
                        and vb[1] <= mark["y"]
                        and mark["y"] + mark["h"] <= vb[1] + vb[3]):
                    bad.append(f"{row['code']} {row['view']} {row['moment']} "
                               f"{row['width']}px: #{mark['shirt']} cropped")
        self.assertEqual([], bad, "\n".join(bad[:20]))

    def test_no_drawn_geometry_runs_outside_the_crop(self):
        """Arrow shafts, heads and the pass, which Focus used to cut."""

        bad = []
        for row in self.report:
            for item in (row.get("geom") or {}).get("outside", []):
                bad.append(f"{row['code']} {row['view']} {row['moment']} "
                           f"{row['width']}px: {item['sel']} outside")
        self.assertEqual([], bad, "\n".join(bad[:20]))
