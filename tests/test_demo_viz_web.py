"""Tests for scene coverage, the role dock rules, and the browser export.

The browser maths itself is checked by ``python -m demo_viz.web.validate``,
which runs the JavaScript in a headless browser and compares against Python.
These tests cover the Python side: which scenes are offered, how role slots
behave, and that the exported payload is complete and free of anything local.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from demo_viz.annotations import select_clips  # noqa: E402
from demo_viz.app.state import DEFAULT_EFFECTS, scene_options  # noqa: E402
from demo_viz.config import demo_paths  # noqa: E402
from demo_viz.core.selection import Selection  # noqa: E402
from demo_viz.sources.synthetic import synthetic_scene  # noqa: E402
from demo_viz.web.build import SITE, build, report  # noqa: E402
from demo_viz.web.export_data import SCENE_FIELDS, scene_payload, slug  # noqa: E402

WEB_DATA = REPO_ROOT / "demo_viz" / "web_data"


class SceneCoverageTests(unittest.TestCase):
    @unittest.skipIf(demo_paths().annotations_xlsx is None, "no shot_annotations.xlsx")
    def test_default_offers_every_renderable_strong_and_medium_scene(self):
        expected = {c.clip_id for c in select_clips(("strong", "medium"))}
        offered = {o["value"] for o in scene_options(DEFAULT_EFFECTS)}
        self.assertEqual(expected, offered - {"synthetic"})
        self.assertGreaterEqual(len(expected), 40)

    @unittest.skipIf(demo_paths().annotations_xlsx is None, "no shot_annotations.xlsx")
    def test_low_and_ignore_never_appear_by_default(self):
        offered = {o["value"] for o in scene_options(DEFAULT_EFFECTS)}
        for clip in select_clips("low"):
            self.assertNotIn(clip.clip_id, offered)

    @unittest.skipIf(demo_paths().annotations_xlsx is None, "no shot_annotations.xlsx")
    def test_filters_narrow_to_one_effect(self):
        strong = {o["value"] for o in scene_options(("strong",))} - {"synthetic"}
        medium = {o["value"] for o in scene_options(("medium",))} - {"synthetic"}
        self.assertTrue(strong)
        self.assertTrue(medium)
        self.assertFalse(strong & medium)

    @unittest.skipIf(demo_paths().annotations_xlsx is None, "no shot_annotations.xlsx")
    def test_labels_show_the_effect_first(self):
        for option in scene_options(DEFAULT_EFFECTS):
            self.assertRegex(option["label"], r"^(STRONG|MEDIUM|SYNTHETIC) · ")

    @unittest.skipIf(demo_paths().annotations_xlsx is None, "no shot_annotations.xlsx")
    def test_coverage_includes_the_shapes_the_demo_must_handle(self):
        shapes = {c.shape for c in select_clips(("strong", "medium"))}
        self.assertIn("1R-1D-1B", shapes)                     # the clean case
        self.assertTrue(any(s.startswith("1R-2D") or s.startswith("2R-2D")
                            for s in shapes))                 # multi-defender
        self.assertTrue(any(s.startswith("2R") for s in shapes))   # multi-runner


class RoleDockTests(unittest.TestCase):
    def setUp(self):
        self.scene = synthetic_scene()
        self.runner = self.scene.by_shirt("7", "attack").player_id
        self.other = self.scene.by_shirt("9", "attack").player_id
        self.beneficiary = self.scene.by_shirt("34", "attack").player_id
        self.defender = self.scene.by_shirt("11", "defend").player_id

    def test_arming_a_slot_sends_the_next_compatible_click_there(self):
        selection = Selection()
        selection.apply_click(self.scene, self.runner)
        selection.arm("beneficiary")
        selection.apply_click(self.scene, self.other)
        self.assertEqual(selection.beneficiaries, [self.other])
        self.assertEqual(selection.runners, [self.runner])
        self.assertFalse(selection.armed)

    def test_arming_replaces_rather_than_appending(self):
        selection = Selection(beneficiaries=[self.beneficiary])
        selection.arm("beneficiary")
        selection.apply_click(self.scene, self.other)
        self.assertEqual(selection.beneficiaries, [self.other])

    def test_an_armed_slot_ignores_the_wrong_team(self):
        selection = Selection()
        selection.arm("runner")
        selection.apply_click(self.scene, self.defender)
        self.assertEqual(selection.runners, [])
        self.assertEqual(selection.defenders, [self.defender])

    def test_swap_exchanges_the_two_attacking_roles(self):
        selection = Selection(runners=[self.runner], defenders=[self.defender],
                              beneficiaries=[self.beneficiary])
        selection.swap_attack_roles()
        self.assertEqual(selection.runners, [self.beneficiary])
        self.assertEqual(selection.beneficiaries, [self.runner])
        self.assertEqual(selection.defenders, [self.defender])

    def test_swap_is_its_own_inverse(self):
        before = Selection(runners=[self.runner], beneficiaries=[self.beneficiary])
        after = Selection(runners=[self.runner], beneficiaries=[self.beneficiary])
        after.swap_attack_roles()
        after.swap_attack_roles()
        self.assertEqual(after.runners, before.runners)
        self.assertEqual(after.beneficiaries, before.beneficiaries)

    def test_swap_works_with_an_empty_slot(self):
        selection = Selection(runners=[self.runner])
        selection.swap_attack_roles()
        self.assertEqual(selection.runners, [])
        self.assertEqual(selection.beneficiaries, [self.runner])

    def test_multiple_defenders_are_kept(self):
        selection = Selection()
        selection.apply_click(self.scene, self.runner)
        for shirt in ("11", "20", "21"):
            player = self.scene.by_shirt(shirt, "defend")
            if player:
                selection.apply_click(self.scene, player.player_id)
        self.assertGreaterEqual(len(selection.defenders), 2)

    def test_armed_state_survives_a_json_round_trip(self):
        selection = Selection().arm("defender")
        restored = Selection.from_dict(selection.to_dict())
        self.assertTrue(restored.armed)
        self.assertEqual(restored.pick, "defender")


class ExportTests(unittest.TestCase):
    def test_payload_has_exactly_the_declared_fields(self):
        scene = synthetic_scene()
        payload = scene_payload(scene, None)
        self.assertEqual(set(payload), set(SCENE_FIELDS))

    def test_payload_carries_no_local_paths_or_usernames(self):
        """A public build must not leak where it was produced."""

        scene = synthetic_scene()
        blob = json.dumps(scene_payload(scene, None))
        for leak in ("/Users/", "/home/", "C:\\\\", "idsse_shots", "web_data",
                     "cache", str(Path.home())):
            self.assertNotIn(leak, blob)

    def test_tracks_are_full_length_and_json_safe(self):
        scene = synthetic_scene()
        payload = scene_payload(scene, None)
        for player in payload["players"]:
            self.assertEqual(len(player["x"]), payload["n_frames"])
            self.assertEqual(len(player["y"]), payload["n_frames"])
        self.assertEqual(len(payload["ball"]["x"]), payload["n_frames"])
        json.dumps(payload)                       # must not raise on NaN

    def test_every_outfield_attacker_gets_a_run_start(self):
        scene = synthetic_scene()
        payload = scene_payload(scene, None)
        for player in payload["players"]:
            if player["side"] == "attack" and not player["gk"]:
                self.assertIn(player["id"], payload["onsets"])

    def test_slug_is_filesystem_safe(self):
        self.assertEqual(slug("J03WOH:shot_006_P1_1054"), "J03WOH_shot_006_P1_1054")


class BuildTests(unittest.TestCase):
    def test_site_has_the_files_the_workflow_checks(self):
        for relative in ("index.html", "style.css", "js/app.js", "js/influence.js",
                         "js/pitch.js", "js/selection.js", "js/ranking.js"):
            self.assertTrue((SITE / relative).exists(), relative)

    def test_build_assembles_site_and_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / "data_in"
            data.mkdir()
            (data / "index.json").write_text('{"scenes":[]}')
            out = build(Path(tmp) / "out", data)
            self.assertTrue((out / "index.html").exists())
            self.assertTrue((out / "js" / "influence.js").exists())
            self.assertTrue((out / "data" / "index.json").exists())
            self.assertIn("shell", report(out))

    def test_the_site_ships_no_build_step_or_dependency(self):
        """The Pages workflow copies files; nothing may need npm at deploy time."""

        html = (SITE / "index.html").read_text()
        self.assertNotIn("node_modules", html)
        self.assertNotIn("http://", html.replace("http://www.w3.org", ""))
        for script in SITE.rglob("*.js"):
            self.assertNotIn("require(", script.read_text())

    @unittest.skipIf(not (WEB_DATA / "index.json").exists(), "no exported web data")
    def test_exported_index_matches_the_scene_files(self):
        index = json.loads((WEB_DATA / "index.json").read_text())["scenes"]
        self.assertTrue(index)
        for row in index:
            self.assertIn(row["effect"], {"strong", "medium"})
            self.assertTrue((WEB_DATA / row["file"]).exists(), row["file"])

    @unittest.skipIf(not (WEB_DATA / "index.json").exists(), "no exported web data")
    def test_exported_scenes_stay_small_enough_to_fetch_lazily(self):
        for path in WEB_DATA.glob("*.json"):
            self.assertLess(path.stat().st_size, 400 * 1024, path.name)


class WorkflowTests(unittest.TestCase):
    def test_pages_workflow_exists_and_uses_the_official_actions(self):
        workflow = REPO_ROOT / ".github" / "workflows" / "pages.yml"
        self.assertTrue(workflow.exists())
        text = workflow.read_text()
        for action in ("actions/checkout", "actions/configure-pages",
                       "actions/upload-pages-artifact", "actions/deploy-pages"):
            self.assertIn(action, text)
        self.assertIn("demo_viz.web.build", text)
        self.assertIn("pages: write", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
