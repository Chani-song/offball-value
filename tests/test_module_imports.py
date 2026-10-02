"""Every built module parses and imports in a real browser.

The cheap source-level checks live in `test_paper_story.ModuleHealthTests`.
This one catches what they cannot: a module that is syntactically fine on its
own but fails when the engine links the graph. A failure here is severe --
`boot()` never runs and the page renders its static shell, which looks exactly
like a stale build rather than a broken one.
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
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from demo_viz.web.build import build             # noqa: E402
from demo_viz.web.validate import find_chrome     # noqa: E402

PROBE = """<!doctype html><meta charset=utf-8><script>
(async () => {
  const out = {};
  for (const m of %s) {
    try { await import("./js/" + m); out[m] = "ok"; }
    catch (e) { out[m] = String(e); }
  }
  fetch("/c", {method: "POST", body: JSON.stringify(out)});
})();
</script>"""


class ModuleImportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if find_chrome() is None:
            raise unittest.SkipTest("no headless Chrome")
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(build(Path(cls.tmp.name) / "site"))
        modules = sorted(p.name for p in (root / "js").glob("*.js"))
        result: dict = {}

        class Handler(http.server.SimpleHTTPRequestHandler):
            def log_message(self, *a):          # noqa: D102, ANN002
                pass

            def do_POST(self):                  # noqa: N802, D102
                size = int(self.headers.get("Content-Length", 0))
                result.update(json.loads(self.rfile.read(size) or b"{}"))
                self.send_response(204)
                self.end_headers()

        class Threaded(socketserver.ThreadingTCPServer):
            allow_reuse_address = True
            daemon_threads = True

        probe = root / "modules.html"
        probe.write_text(PROBE % json.dumps(modules))
        handler = functools.partial(Handler, directory=str(root))
        with Threaded(("127.0.0.1", 0), handler) as httpd:
            threading.Thread(target=httpd.serve_forever, daemon=True).start()
            with tempfile.TemporaryDirectory() as profile, \
                    tempfile.TemporaryDirectory() as shot:
                try:
                    subprocess.run(
                        [find_chrome(), "--headless", "--disable-gpu", "--no-sandbox",
                         f"--user-data-dir={profile}", "--virtual-time-budget=20000",
                         f"--screenshot={Path(shot) / 'x.png'}",
                         f"http://127.0.0.1:{httpd.server_address[1]}/modules.html"],
                        capture_output=True, timeout=120)
                except subprocess.TimeoutExpired:
                    pass
            httpd.shutdown()
        probe.unlink(missing_ok=True)
        cls.modules = modules
        cls.result = result

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "tmp"):
            cls.tmp.cleanup()

    def test_the_probe_reached_every_module(self):
        if not self.result:
            self.skipTest("the probe returned nothing")
        self.assertEqual(sorted(self.modules), sorted(self.result))

    def test_every_module_imports(self):
        if not self.result:
            self.skipTest("the probe returned nothing")
        bad = [f"{name}: {why}" for name, why in sorted(self.result.items())
               if why != "ok"]
        self.assertEqual([], bad, "\n".join(bad))


if __name__ == "__main__":
    unittest.main()
