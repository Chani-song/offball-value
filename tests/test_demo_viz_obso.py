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
            root = build(Path(tmp) / "site", data_dir=WEB_DATA,
                         include_legacy_obso=True)
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
            root = build(Path(tmp) / "site", data_dir=WEB_DATA,
                         include_legacy_obso=True)
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

    #: Phrases that are allowed only inside an explicit denial.
    DENIALS = ("never a pass success probability",
               "not validated pass-completion labels",
               "not generated, proposed or ranked by")

    def test_no_completion_or_ranking_language_in_the_browser(self):
        for path, text in self._surfaces():
            for pattern in self.FORBIDDEN:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    window = text[max(0, match.start() - 90):match.end() + 90].lower()
                    self.assertTrue(
                        any(d in window for d in self.DENIALS),
                        f"{path.name} claims '{match.group(0)}' outside a denial")

    def test_obso_is_only_mentioned_as_a_legacy_diagnostic(self):
        """If the public page names OBSO at all, it must disown it."""

        markup = (SITE / "index.html").read_text()
        for match in re.finditer(r"OBSO", markup):
            window = markup[max(0, match.start() - 400):match.end() + 400]
            self.assertIn("Legacy / reference diagnostic", window)
            self.assertIn("not part of the current solver", window)

    def test_nothing_public_is_labelled_xt(self):
        markup = (SITE / "index.html").read_text()
        self.assertNotIn("xT threat", markup)
        self.assertNotIn("xT surface", markup)
        for match in re.finditer(r"xT", markup):
            window = markup[max(0, match.start() - 60):match.end() + 60]
            self.assertIn("No calibrated", window)

    def test_the_fan_is_described_as_the_demo_own_geometry(self):
        markup = (SITE / "index.html").read_text()
        self.assertIn("Explore pass model", markup)
        self.assertIn("not the solver", markup)
        self.assertIn("never labelled best or recommended", markup)

    def test_the_source_panel_explains_the_current_solver_chain(self):
        markup = (SITE / "index.html").read_text()
        for phrase in ("Completion proxy", "Positional threat", "Release payoff",
                       "experimental_proxy", "--allow-proxy-labels"):
            self.assertIn(phrase, markup, phrase)
        # and the legacy stack is named as legacy, with its EPV description
        self.assertIn("pitch control × ball transition × a static EPV grid", markup)


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
    """OBSO is a property of the frame, not of the picked roles.

    This used to be checked end to end by reading the stat panel with the OBSO
    view selected. That control was removed when OBSO was demoted, so the
    guarantee is checked where it now lives: the browser module takes no
    selection, and neither does the Python it mirrors. The end-to-end version
    still runs on backup/ssac-obso-demo.
    """

    def test_the_browser_surface_takes_no_role_argument(self):
        source = (SITE / "js" / "obso.js").read_text()
        signature = re.search(r"export function surfaceAt\(([^)]*)\)", source)
        self.assertIsNotNone(signature)
        for role in ("runner", "defender", "beneficiary", "selection", "roles"):
            self.assertNotIn(role, signature.group(1).lower())

    def test_the_exported_surface_is_frame_keyed_only(self):
        if not _exported_scenes():
            self.skipTest("no exported OBSO data")
        payload = json.loads(_exported_scenes()[0].read_text())
        self.assertEqual(sorted(payload["indices"]), payload["indices"])
        self.assertNotIn("roles", payload)


class CandidateConstantTests(unittest.TestCase):
    """The geometric rule is one set of documented constants, shared."""

    def test_python_and_javascript_agree_on_the_rule(self):
        # the carrier rule moved to carrier.js when OBSO became lazy; obso.js
        # re-exports it, and this checks the definitions at their new home
        source = (SITE / "js" / "carrier.js").read_text()
        self.assertEqual(25.0, CANDIDATE_DISTANCE_M)
        self.assertEqual((-45.0, -22.5, 0.0, 22.5, 45.0), CANDIDATE_ANGLES_DEG)
        self.assertIn(f"CANDIDATE_DISTANCE_M = {CANDIDATE_DISTANCE_M:g}", source)
        self.assertIn("[-45, -22.5, 0, 22.5, 45]", source)
        self.assertIn(f"CARRIER_MAX_M = {CARRIER_MAX_M:g}", source)
        self.assertIn(f"CARRIER_MARGIN_M = {CARRIER_MARGIN_M:g}", source)

    def test_obso_still_re_exports_them_for_its_own_readers(self):
        source = (SITE / "js" / "obso.js").read_text()
        self.assertIn('from "./carrier.js"', source)
        for name in ("candidatePasses", "carrierAt", "CANDIDATE_DISTANCE_M",
                     "CANDIDATE_ANGLES_DEG", "CARRIER_MAX_M", "CARRIER_MARGIN_M"):
            self.assertIn(name, source, name)

    def test_the_layer_is_off_by_default(self):
        markup = (SITE / "index.html").read_text()
        match = re.search(r'<input type="checkbox" data-layer="passes"([^>]*)>', markup)
        self.assertIsNotNone(match, "the Candidate passes layer is missing")
        self.assertNotIn("checked", match.group(1))

    def test_the_public_space_views_are_the_off_ball_pair(self):
        """OBSO was demoted: it is not part of the current solver's payoff.

        The implementation and its export stay; only the public control went.
        The version of the demo built around it is on backup/ssac-obso-demo.
        """

        markup = (SITE / "index.html").read_text()
        options = re.findall(r'<option value="([a-z]+)"', markup)
        self.assertEqual(["space", "gain"], options)
        self.assertNotIn(">OBSO threat<", markup)


class PublicBuildExcludesLegacyTests(unittest.TestCase):
    """The public site must not ship a payload nothing can request.

    OBSO was demoted when the demo went solver-native. The implementation, the
    exporter and every numerical test stay; only the ~5.5 MB of surfaces leave
    the public build. `--include-legacy-obso` puts them back for internal
    debugging, and is what the parity tests above use.
    """

    def _sizes(self, root: Path) -> tuple[int, int]:
        obso = root / "data" / "obso"
        files = list(obso.glob("*.json")) if obso.exists() else []
        return len(files), sum(f.stat().st_size for f in files)

    def test_the_default_build_has_no_obso_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = build(Path(tmp) / "site", data_dir=WEB_DATA)
            count, _ = self._sizes(root)
            self.assertEqual(0, count, "the public build shipped OBSO surfaces")
            # and the rest of the site is intact
            self.assertTrue((root / "index.html").exists())
            self.assertTrue((root / "data" / "index.json").exists())

    @unittest.skipIf(not _exported_scenes(), "no exported OBSO data")
    def test_the_flag_puts_them_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = build(Path(tmp) / "site", data_dir=WEB_DATA,
                         include_legacy_obso=True)
            count, size = self._sizes(root)
            self.assertGreater(count, 0)
            self.assertGreater(size, 1_000_000)

    @unittest.skipIf(not _exported_scenes(), "no exported OBSO data")
    def test_dropping_it_saves_what_it_claims(self):
        with tempfile.TemporaryDirectory() as tmp:
            public = build(Path(tmp) / "public", data_dir=WEB_DATA)
            legacy = build(Path(tmp) / "legacy", data_dir=WEB_DATA,
                           include_legacy_obso=True)
            total = lambda root: sum(f.stat().st_size for f in root.rglob("*")
                                     if f.is_file())
            self.assertGreater(total(legacy) - total(public), 4_000_000)

    def test_the_solver_native_payloads_survive(self):
        """Dropping the legacy payload must not touch what the demo does use."""

        with tempfile.TemporaryDirectory() as tmp:
            root = build(Path(tmp) / "site", data_dir=WEB_DATA)
            for folder in ("release", "solver"):
                source = WEB_DATA / folder
                if not source.exists():
                    continue
                shipped = list((root / "data" / folder).glob("*.json"))
                self.assertEqual(len(list(source.glob("*.json"))), len(shipped), folder)

    def test_the_public_page_cannot_ask_for_the_missing_payload(self):
        """No reachable path fetches data/obso, including via the URL."""

        app = (SITE / "js" / "app.js").read_text()
        modes = re.search(r'\["space", "gain"[^\]]*\]\.includes\(wanted\.mode\)', app)
        self.assertIsNotNone(modes, "the URL should accept only the public views")
        self.assertNotIn('"obso"', modes.group(0))
        # the two fetching call sites stay behind a mode the UI cannot produce
        for call in ("obsoSurface(index)", "obsoSurface(state.frame)"):
            position = app.index(call)
            window = app[max(0, position - 220):position]
            self.assertIn('mode === "obso"', window, call)


if __name__ == "__main__":
    unittest.main(verbosity=2)
