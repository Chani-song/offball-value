"""OBSO threat view and the candidate-pass fan.

Two things are worth testing and one is worth testing hard: the browser
reconstructs the OBSO surface from precomputed pitch control plus two terms it
computes itself, so the interesting question is whether that reconstruction
still equals ``evaluate_reference_obso``. It is checked against Python in
headless Chrome rather than reimplemented here.
"""

from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from demo_viz.config import demo_paths  # noqa: E402
from demo_viz.web.build import SITE, build  # noqa: E402
from demo_viz.web.export_data import WEB_DATA, slug  # noqa: E402
from demo_viz.core.candidates import (  # noqa: E402
    CANDIDATE_ANGLES_DEG,
    CANDIDATE_DISTANCE_M,
    CARRIER_MARGIN_M,
    CARRIER_MAX_M,
)
from demo_viz.web.export_obso import DEFAULT_STRIDE, OBSO_DIR, QUANT_LEVELS  # noqa: E402

HERE = REPO_ROOT / "demo_viz" / "web"


def _exported_scenes() -> list[Path]:
    if not OBSO_DIR.exists():
        return []
    return sorted(p for p in OBSO_DIR.glob("*.json") if p.name != "score_grid.json")


class ObsoExportTests(unittest.TestCase):
    """The exported file says what it is and stays inside its declared range."""

    @unittest.skipIf(not _exported_scenes(), "no exported OBSO data")
    def test_payload_declares_the_product_not_the_bare_grid(self):
        payload = json.loads(_exported_scenes()[0].read_text())
        self.assertEqual("pitch_control", payload["quantity"])
        self.assertIn("pitch control x ball transition x EPV score", payload["method"])
        self.assertEqual(50, payload["grid"]["nx"])
        self.assertEqual(32, payload["grid"]["ny"])

    @unittest.skipIf(not _exported_scenes(), "no exported OBSO data")
    def test_samples_cover_the_clip_at_the_declared_stride(self):
        for path in _exported_scenes()[:5]:
            payload = json.loads(path.read_text())
            indices = payload["indices"]
            self.assertEqual(0, indices[0])
            self.assertEqual(payload["n_frames"] - 1, indices[-1])
            self.assertEqual(sorted(set(indices)), indices)
            # regular samples may be joined by extra ones at offside steps,
            # never replaced by them
            self.assertLessEqual(max(np.diff(indices)), DEFAULT_STRIDE)

    @unittest.skipIf(not _exported_scenes(), "no exported OBSO data")
    def test_control_is_a_probability(self):
        import base64

        payload = json.loads(_exported_scenes()[0].read_text())
        raw = np.frombuffer(base64.b64decode(payload["data"]), dtype=np.uint8)
        self.assertEqual(len(payload["indices"]) * 50 * 32, raw.size)
        self.assertLessEqual(raw.max() / QUANT_LEVELS, 1.0)

    @unittest.skipIf(not (OBSO_DIR / "score_grid.json").exists(), "no exported OBSO data")
    def test_the_score_grid_ships_its_own_caveat(self):
        grid = json.loads((OBSO_DIR / "score_grid.json").read_text())
        self.assertIn("defensive line", grid["caveat"])
        self.assertEqual(50 * 32, len(grid["values"]))


class ObsoParityTests(unittest.TestCase):
    """The browser's OBSO equals the Python reference at the sampled frames."""

    @unittest.skipIf(not _exported_scenes(), "no exported OBSO data")
    def test_browser_surface_matches_evaluate_reference_obso(self):
        from demo_viz.loader import load_scene
        from demo_viz.quantities import _frame_at, _velocities
        from demo_viz.web.validate import find_chrome, serve
        from offball_value.reference_obso import (
            ReferenceOBSOConfig,
            evaluate_reference_obso,
        )

        chrome = find_chrome()
        if chrome is None:
            self.skipTest("no Chrome/Chromium available")

        payload = json.loads(_exported_scenes()[0].read_text())
        scene_id = payload["scene_id"]
        sampled = payload["indices"][:: max(1, len(payload["indices"]) // 6)][:6]
        plan = {"file": f"{slug(scene_id)}.json", "indices": sampled,
                "fan_indices": sampled}

        with tempfile.TemporaryDirectory() as tmp:
            root = build(Path(tmp) / "site", data_dir=WEB_DATA)
            page = (HERE / "obso_harness.html").read_text().replace(
                "<body>",
                '<body><script type="application/json" id="plan">'
                + json.dumps(plan).replace("</", "<\\/") + "</script>",
            )
            (root / "obso_harness.html").write_text(page)
            with serve(root) as port:
                result = subprocess.run(
                    [chrome, "--headless", "--disable-gpu", "--no-sandbox",
                     "--virtual-time-budget=60000", "--dump-dom",
                     f"http://127.0.0.1:{port}/obso_harness.html"],
                    capture_output=True, text=True, timeout=180,
                )
        dom = result.stdout
        start = dom.find('<pre id="out">')
        self.assertGreaterEqual(start, 0, "harness did not render")
        start = dom.index(">", start) + 1
        text = dom[start:dom.index("</pre>", start)].strip()
        text = text.replace("&quot;", '"').replace("&amp;", "&").replace("&lt;", "<")
        self.assertNotEqual("running", text, "harness timed out")
        report = json.loads(text)
        self.assertNotIn("error", report, report.get("error", ""))

        scene = load_scene(scene_id, quantities=False, surfaces=False)
        config = ReferenceOBSOConfig()
        goalkeepers = tuple(p.player_id for p in scene.players.values() if p.is_goalkeeper)
        worst = 0.0
        for entry in report["frames"]:
            index = entry["index"]
            reference = evaluate_reference_obso(
                _frame_at(scene, index),
                attacking_team_id=scene.attacking_team_id,
                attacking_direction=int(scene.attacking_direction),
                velocities=_velocities(scene, index),
                goalkeeper_ids=goalkeepers,
                apply_offside=True,
                config=config,
            )
            mine = np.asarray(entry["values"], dtype=float).reshape(32, 50)
            peak = float(reference.obso.max())
            self.assertGreater(peak, 0.0)
            worst = max(worst, float(np.abs(mine - reference.obso).max()) / peak)
        # quantisation of pitch control to one byte is the only loss
        self.assertLess(worst, 0.02, f"worst relative error {worst:.4f}")

    @unittest.skipIf(not _exported_scenes(), "no exported OBSO data")
    def test_the_fan_is_five_rays_centred_on_the_attacking_direction(self):
        from demo_viz.loader import load_scene
        from demo_viz.web.validate import find_chrome, serve

        chrome = find_chrome()
        if chrome is None:
            self.skipTest("no Chrome/Chromium available")

        payload = json.loads(_exported_scenes()[0].read_text())
        scene_id = payload["scene_id"]
        frames = list(range(0, payload["n_frames"], 12))
        plan = {"file": f"{slug(scene_id)}.json", "indices": [], "fan_indices": frames}

        with tempfile.TemporaryDirectory() as tmp:
            root = build(Path(tmp) / "site", data_dir=WEB_DATA)
            page = (HERE / "obso_harness.html").read_text().replace(
                "<body>",
                '<body><script type="application/json" id="plan">'
                + json.dumps(plan).replace("</", "<\\/") + "</script>",
            )
            (root / "obso_harness.html").write_text(page)
            with serve(root) as port:
                result = subprocess.run(
                    [chrome, "--headless", "--disable-gpu", "--no-sandbox",
                     "--virtual-time-budget=60000", "--dump-dom",
                     f"http://127.0.0.1:{port}/obso_harness.html"],
                    capture_output=True, text=True, timeout=180,
                )
        start = result.stdout.index(">", result.stdout.find('<pre id="out">')) + 1
        text = result.stdout[start:result.stdout.index("</pre>", start)].strip()
        report = json.loads(text.replace("&quot;", '"').replace("&amp;", "&"))
        self.assertEqual([-45, -22.5, 0, 22.5, 45], report["constants"]["angles"])

        scene = load_scene(scene_id, quantities=False, surfaces=False)
        direction = int(scene.attacking_direction)
        length, width = scene.pitch_length, scene.pitch_width
        drawn = 0
        for entry in report["fan"]:
            if entry["origin"] is None:
                self.assertIsNone(entry["carrier"])
                continue
            drawn += 1
            self.assertEqual(5, len(entry["ends"]))
            x0, y0 = entry["origin"]
            for (x1, y1), degrees in zip(entry["ends"], entry["degrees"]):
                # inside the pitch, always
                self.assertLessEqual(abs(x1), length / 2 + 1e-6)
                self.assertLessEqual(abs(y1), width / 2 + 1e-6)
                # and pointing the way the team attacks, within the sector
                if math.hypot(x1 - x0, y1 - y0) < 1e-6:
                    continue
                bearing = math.degrees(math.atan2((y1 - y0) * direction,
                                                  (x1 - x0) * direction))
                self.assertAlmostEqual(degrees, bearing, places=4)
        self.assertGreater(drawn, 0, "no frame had a clear carrier")


class NoPassModelClaimTests(unittest.TestCase):
    """Nothing in the demo may claim a completion probability or a best pass."""

    #: Phrases that would assert a model this repository does not have.
    FORBIDDEN = (
        r"p\(complete", r"\bbest pass\b", r"expected value",
        r"success probability", r"completion probability", r"\bxpass\b",
        r"\btop 5\b", r"\branked by\b", r"\boptimal pass\b",
    )

    def _surfaces(self):
        files = [SITE / "index.html", SITE / "js" / "obso.js", SITE / "js" / "app.js",
                 SITE / "js" / "pitch.js", SITE / "js" / "palette.js"]
        return [(p, p.read_text()) for p in files if p.exists()]

    def test_no_completion_or_ranking_language_in_the_browser(self):
        for path, text in self._surfaces():
            for pattern in self.FORBIDDEN:
                match = re.search(pattern, text, re.IGNORECASE)
                self.assertIsNone(
                    match, f"{path.name} claims '{match.group(0)}'" if match else "")

    def test_the_threat_view_is_never_labelled_xt(self):
        """The page may say it does not show xT. It may not call this xT."""

        markup = (SITE / "index.html").read_text()
        self.assertIn('<option value="obso">OBSO threat</option>', markup)
        self.assertNotIn("xT threat", markup)
        self.assertNotIn("xT surface", markup)
        # the existing disclaimer is the only place the letters may appear
        for match in re.finditer(r"xT", markup):
            window = markup[max(0, match.start() - 60):match.end() + 60]
            self.assertIn("No calibrated", window)

    def test_the_fan_is_described_as_directions_not_options_ranked(self):
        markup = (SITE / "index.html").read_text()
        self.assertIn("Candidate passes", markup)
        self.assertIn("not</b> ranked", markup)

    def test_the_source_panel_carries_the_epv_caveat(self):
        markup = (SITE / "index.html").read_text()
        self.assertIn("pitch control × ball transition × EPV", markup)
        self.assertIn("does not\n          explicitly encode the defensive line", markup)


class ObsoIsRoleBlindTests(unittest.TestCase):
    """The threat surface must not move when the visitor picks other roles."""

    @unittest.skipIf(demo_paths().annotations_xlsx is None, "no shot_annotations.xlsx")
    def test_reference_obso_takes_no_role_argument(self):
        import inspect

        from offball_value.reference_obso import evaluate_reference_obso

        names = set(inspect.signature(evaluate_reference_obso).parameters)
        for role in ("runner", "defender", "beneficiary", "target", "roles"):
            self.assertNotIn(role, names)

    @unittest.skipIf(not _exported_scenes(), "no exported OBSO data")
    def test_the_export_stores_nothing_role_shaped(self):
        payload = json.loads(_exported_scenes()[0].read_text())
        for key in ("roles", "runner", "defender", "beneficiary"):
            self.assertNotIn(key, payload)


class ObsoSurvivesRoleSwitchingTests(unittest.TestCase):
    """End to end: the same frame gives the same threat whoever is picked."""

    @unittest.skipIf(not _exported_scenes(), "no exported OBSO data")
    def test_the_peak_does_not_move_when_the_roles_do(self):
        from demo_viz.web.validate import find_chrome, serve

        chrome = find_chrome()
        if chrome is None:
            self.skipTest("no Chrome/Chromium available")
        scene_id = json.loads(_exported_scenes()[0].read_text())["scene_id"]

        peaks = set()
        with tempfile.TemporaryDirectory() as tmp:
            root = build(Path(tmp) / "site", data_dir=WEB_DATA)
            with serve(root) as port:
                base = (f"http://127.0.0.1:{port}/index.html?scene={scene_id}"
                        "&mode=obso&t=40")
                for roles in ("r=7&d=11&b=34", "r=10&d=23&b=9", "r=18&d=33&b=25"):
                    result = subprocess.run(
                        [chrome, "--headless", "--disable-gpu", "--no-sandbox",
                         "--virtual-time-budget=20000", "--dump-dom", f"{base}&{roles}"],
                        capture_output=True, text=True, timeout=120,
                    )
                    found = re.search(r'id="stat-value"[^>]*>([^<]*)', result.stdout)
                    self.assertIsNotNone(found, "stat panel did not render")
                    peaks.add(found.group(1).strip())
        self.assertEqual(1, len(peaks), f"OBSO moved with the roles: {peaks}")
        self.assertNotIn("\u2014", peaks.pop())     # and it actually rendered


class CandidateConstantTests(unittest.TestCase):
    """The geometric rule is one set of documented constants, shared."""

    def test_python_and_javascript_agree_on_the_rule(self):
        source = (SITE / "js" / "obso.js").read_text()
        self.assertEqual(25.0, CANDIDATE_DISTANCE_M)
        self.assertEqual((-45.0, -22.5, 0.0, 22.5, 45.0), CANDIDATE_ANGLES_DEG)
        self.assertIn(f"CANDIDATE_DISTANCE_M = {CANDIDATE_DISTANCE_M:g}", source)
        self.assertIn("[-45, -22.5, 0, 22.5, 45]", source)
        self.assertIn(f"CARRIER_MAX_M = {CARRIER_MAX_M:g}", source)
        self.assertIn(f"CARRIER_MARGIN_M = {CARRIER_MARGIN_M:g}", source)

    def test_the_layer_is_off_by_default(self):
        markup = (SITE / "index.html").read_text()
        match = re.search(r'<input type="checkbox" data-layer="passes"([^>]*)>', markup)
        self.assertIsNotNone(match, "the Candidate passes layer is missing")
        self.assertNotIn("checked", match.group(1))

    def test_the_threat_view_is_offered_but_not_the_default(self):
        markup = (SITE / "index.html").read_text()
        options = re.findall(r'<option value="([a-z]+)"', markup)
        self.assertEqual(["space", "gain", "obso"], options)


if __name__ == "__main__":
    unittest.main(verbosity=2)
