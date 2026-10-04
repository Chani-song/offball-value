"""Every public scene opens on its first frame, and the solved moments do not move.

A rendering test: it builds the site, opens each curated scene in headless
Chrome, and reads back the playhead, the readout and the solved moments the
panels actually offer.
"""

from __future__ import annotations

import functools
import http.server
import json
import socketserver
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from demo_viz.web.build import build            # noqa: E402
from demo_viz.web.validate import find_chrome    # noqa: E402

PROBE = """<!doctype html><meta charset=utf-8><div id=h></div>
<script>
const wait = (ms) => new Promise(r => setTimeout(r, ms));
(async () => {
  const out = [];
  for (const code of __CODES__) {
    const f = document.createElement("iframe");
    f.style.cssText = "width:1440px;height:1100px;border:0";
    f.src = `index.html?showcase=${code}`;
    document.getElementById("h").replaceChildren(f);
    await new Promise(r => f.addEventListener("load", r));
    await wait(2300);
    const d = f.contentDocument;
    const row = {code,
      frame: Number(d.getElementById("time").value),
      readout: d.getElementById("readout").textContent.trim(),
      tab: [...d.querySelectorAll(".mode.is-on .mode-t")].map(n => n.textContent)[0]};
    // the solved moments Nash offers, and the frames they jump to
    for (const b of d.querySelectorAll("#story-modes .mode")) {
      if (b.dataset.mode === "game_solution") b.click();
    }
    await wait(1600);
    const segs = [...d.querySelectorAll("#moment-pick .seg")];
    row.moments = segs.map(s => s.textContent);
    row.momentFrames = [];
    for (const s of segs) {
      s.click(); await wait(500);
      row.momentFrames.push(Number(d.getElementById("time").value));
    }
    out.push(row);
  }
  fetch("/c", {method: "POST", body: JSON.stringify(out)});
})();
</script>"""


def _render(site: Path, codes: list[str]) -> list[dict]:
    """One short Chrome run per scene.

    A single run of every scene stopped finishing once the demo had twenty of
    them: the pages fetch their modes lazily, so virtual time crawls and the
    probe posted nothing inside the timeout -- which this module then read as
    "no report" and skipped, quietly losing its coverage. One run per scene is
    what `test_label_collisions` already does, for the same reason.
    """

    out: list[dict] = []
    for code in codes:
        out.extend(_render_one(site, [code]))
    return out


def _render_one(site: Path, codes: list[str]) -> list[dict]:
    result: list[dict] = []

    class Handler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *a):           # noqa: D102, ANN002
            pass

        def do_POST(self):                   # noqa: N802, D102
            size = int(self.headers.get("Content-Length", 0))
            result.extend(json.loads(self.rfile.read(size) or b"[]"))
            self.send_response(204)
            self.end_headers()

    class Threaded(socketserver.ThreadingTCPServer):
        allow_reuse_address = True
        daemon_threads = True

    probe = site / "timeline.html"
    probe.write_text(PROBE.replace("__CODES__", json.dumps(codes)))
    handler = functools.partial(Handler, directory=str(site))
    with Threaded(("127.0.0.1", 0), handler) as httpd:
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        port = httpd.server_address[1]
        with tempfile.TemporaryDirectory() as profile, \
                tempfile.TemporaryDirectory() as shot:
            chrome = subprocess.Popen(
                [find_chrome(), "--headless", "--disable-gpu", "--no-sandbox",
                 f"--user-data-dir={profile}", "--window-size=1500,1200",
                 "--virtual-time-budget=90000",
                 f"--screenshot={Path(shot) / 'x.png'}",
                 f"http://127.0.0.1:{port}/timeline.html"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            # the probe's post is the signal: since Chrome 154 the process
            # writes its screenshot and stays up, so waiting for it to exit
            # spent the whole timeout on every run
            deadline = time.monotonic() + 240
            while time.monotonic() < deadline:
                if result:
                    time.sleep(0.3)
                    break
                if chrome.poll() is not None:
                    break
                time.sleep(0.2)
            chrome.kill()
            try:
                chrome.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
        httpd.shutdown()
    probe.unlink(missing_ok=True)
    return result


def _panels_of(code: str) -> list[dict]:
    """The solved panels this showcase code exports."""

    scenes = json.loads((REPO_ROOT / "demo_viz" / "data"
                         / "submission_showcase.json").read_text())["scenes"]
    scene_id = next(s["scene_id"] for s in scenes if s.get("showcase_id") == code)
    path = (REPO_ROOT / "demo_viz" / "web_data" / "solver"
            / f"{scene_id.replace(':', '_')}.json")
    return json.loads(path.read_text())["panels"]


class PublicTimelineRenderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if find_chrome() is None:
            raise unittest.SkipTest("no headless Chrome")
        showcase = json.loads(
            (REPO_ROOT / "demo_viz" / "data" / "submission_showcase.json").read_text())
        rows = showcase["scenes"] if isinstance(showcase, dict) else showcase
        solver = REPO_ROOT / "demo_viz" / "web_data" / "solver"
        cls.codes = sorted(
            row["showcase_id"] for row in rows
            if row.get("scene_id")
            and (solver / f"{row['scene_id'].replace(':', '_')}.json").exists())
        cls.tmp = tempfile.TemporaryDirectory()
        cls.site = Path(build(Path(cls.tmp.name) / "site"))
        cls.report = _render(cls.site, cls.codes)
        if not cls.report:
            raise unittest.SkipTest("the probe returned nothing")

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "tmp"):
            cls.tmp.cleanup()

    def test_every_scene_was_measured(self):
        self.assertEqual(self.codes, [row["code"] for row in self.report])

    def test_every_scene_opens_on_its_first_frame(self):
        bad = [f"{r['code']}: frame {r['frame']}" for r in self.report
               if r["frame"] != 0]
        self.assertEqual([], bad, "\n".join(bad))

    def test_play_opens_at_zero_seconds(self):
        bad = [f"{r['code']}: {r['readout']!r}" for r in self.report
               if r["readout"] != "0.00 s"]
        self.assertEqual([], bad, "\n".join(bad))

    def test_play_is_the_opening_tab(self):
        bad = [f"{r['code']}: {r['tab']!r}" for r in self.report
               if r["tab"] != "1. Play"]
        self.assertEqual([], bad, "\n".join(bad))

    def test_the_solved_moments_are_unchanged(self):
        """Re-zeroing the display may not move a model-evaluation moment.

        Against the scene's own panels, not a fixed 0.0 / 0.6 / 1.2: S48 is
        solved at one moment only, because from 0.6 s the ball is already with
        its runner and a 2v1 game needs the carrier on it. Reading the export
        keeps the question "are these the solved moments?" rather than "are
        there three of them?".
        """

        bad = []
        for row in self.report:
            want = [f"{panel['dt']:.1f} s" for panel in _panels_of(row["code"])]
            if row["moments"] != want:
                bad.append(f"{row['code']}: moments {row['moments']} != {want}")
        self.assertEqual([], bad, "\n".join(bad))

    def test_each_moment_jumps_to_its_own_solved_frame(self):
        """And those frames are the panels' own, read from the export."""

        bad = []
        for row in self.report:
            want = [p["frame"] for p in _panels_of(row["code"])]
            if row["momentFrames"] != want:
                bad.append(f"{row['code']}: {row['momentFrames']} != {want}")
        self.assertEqual([], bad, "\n".join(bad))


if __name__ == "__main__":
    unittest.main()
