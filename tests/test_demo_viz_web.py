"""Tests for scene coverage, the role dock rules, and the browser export.

The browser maths itself is checked by ``python -m demo_viz.web.validate``,
which runs the JavaScript in a headless browser and compares against Python.
These tests cover the Python side: which scenes are offered, how role slots
behave, and that the exported payload is complete and free of anything local.
"""

from __future__ import annotations

import json
import re
import subprocess
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

    def test_arming_appends_rather_than_replacing(self):
        """A role is a list: a second compatible click adds, it does not evict."""

        selection = Selection(beneficiaries=[self.beneficiary])
        selection.arm("beneficiary")
        selection.apply_click(self.scene, self.other)
        self.assertEqual(selection.beneficiaries, [self.beneficiary, self.other])

    def test_an_armed_slot_stays_armed_for_several_picks(self):
        selection = Selection().arm("beneficiary")
        selection.apply_click(self.scene, self.beneficiary)
        selection.apply_click(self.scene, self.other)
        self.assertEqual(selection.beneficiaries, [self.beneficiary, self.other])

    def test_an_armed_click_on_a_member_removes_only_that_player(self):
        selection = Selection(beneficiaries=[self.beneficiary, self.other]).arm("beneficiary")
        selection.apply_click(self.scene, self.beneficiary)
        self.assertEqual(selection.beneficiaries, [self.other])

    def test_an_armed_slot_ignores_the_wrong_team(self):
        selection = Selection()
        selection.arm("runner")
        selection.apply_click(self.scene, self.defender)
        self.assertEqual(selection.runners, [])
        self.assertEqual(selection.defenders, [self.defender])

    def test_move_relocates_one_player_and_leaves_the_rest(self):
        """Dragging one runner chip to Beneficiary must move only that player."""

        selection = Selection(runners=[self.runner, self.other],
                              beneficiaries=[self.beneficiary])
        selection.move(self.scene, "beneficiary", self.other)
        self.assertEqual(selection.runners, [self.runner])
        self.assertEqual(selection.beneficiaries, [self.beneficiary, self.other])

    def test_move_back_again_returns_only_that_player(self):
        selection = Selection(runners=[self.runner],
                              beneficiaries=[self.beneficiary, self.other])
        selection.move(self.scene, "runner", self.other)
        self.assertEqual(selection.runners, [self.runner, self.other])
        self.assertEqual(selection.beneficiaries, [self.beneficiary])

    def test_move_onto_the_role_a_player_already_holds_is_a_no_op(self):
        selection = Selection(runners=[self.runner, self.other])
        selection.move(self.scene, "runner", self.other)
        self.assertEqual(selection.runners, [self.runner, self.other])

    def test_move_refuses_the_wrong_side(self):
        selection = Selection(defenders=[self.defender])
        selection.move(self.scene, "runner", self.defender)
        self.assertEqual(selection.runners, [])
        self.assertEqual(selection.defenders, [self.defender])

    def test_a_player_cannot_hold_both_attacking_roles(self):
        selection = Selection(runners=[self.runner])
        selection.move(self.scene, "beneficiary", self.runner)
        self.assertEqual(selection.runners, [])
        self.assertEqual(selection.beneficiaries, [self.runner])

    def test_clearing_a_role_empties_only_that_role(self):
        selection = Selection(runners=[self.runner, self.other],
                              defenders=[self.defender],
                              beneficiaries=[self.beneficiary])
        selection.clear("runner")
        self.assertEqual(selection.runners, [])
        self.assertEqual(selection.defenders, [self.defender])
        self.assertEqual(selection.beneficiaries, [self.beneficiary])

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

    def test_every_role_accepts_several_players(self):
        selection = Selection()
        selection.arm("runner")
        for shirt in ("7", "9"):
            selection.apply_click(self.scene, self.scene.by_shirt(shirt, "attack").player_id)
        selection.arm("defender")
        for shirt in ("11", "20", "21"):
            player = self.scene.by_shirt(shirt, "defend")
            if player:
                selection.apply_click(self.scene, player.player_id)
        selection.arm("beneficiary")
        for shirt in ("34", "20"):
            player = self.scene.by_shirt(shirt, "attack")
            if player:
                selection.apply_click(self.scene, player.player_id)
        self.assertEqual(len(selection.runners), 2)
        self.assertGreaterEqual(len(selection.defenders), 2)
        self.assertGreaterEqual(len(selection.beneficiaries), 1)

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


class StrongVideoSelectionTests(unittest.TestCase):
    """The rendered demo videos use exactly the five human-confirmed strong scenes."""

    @unittest.skipIf(demo_paths().annotations_xlsx is None, "no shot_annotations.xlsx")
    def test_exactly_the_five_strong_scenes_are_rendered(self):
        from demo_viz.export_strong import strong_scenes

        chosen = strong_scenes()
        self.assertEqual(5, len(chosen))
        self.assertEqual({c.clip_id for c in select_clips("strong")},
                         {entry.clip_id for entry in chosen})

    @unittest.skipIf(demo_paths().annotations_xlsx is None, "no shot_annotations.xlsx")
    def test_no_medium_or_low_scene_reaches_the_montage(self):
        from demo_viz.export_strong import strong_scenes

        chosen = {entry.clip_id for entry in strong_scenes()}
        for effect in ("medium", "low"):
            for clip in select_clips(effect):
                self.assertNotIn(clip.clip_id, chosen)

    @unittest.skipIf(demo_paths().annotations_xlsx is None, "no shot_annotations.xlsx")
    def test_scenes_are_numbered_simplest_first(self):
        from demo_viz.export_strong import strong_scenes

        chosen = strong_scenes()
        self.assertEqual([1, 2, 3, 4, 5], [entry.position for entry in chosen])
        self.assertEqual("1R-1D-1B", chosen[0].shape)
        complexities = [entry.complexity for entry in chosen]
        self.assertEqual(sorted(complexities), complexities)

    def test_the_passing_lane_is_off_in_the_videos(self):
        from demo_viz.export_strong import DISABLED_LAYERS

        self.assertIn("lane", DISABLED_LAYERS)


class BrowserRoleStateTests(unittest.TestCase):
    """`js/selection.js` is a hand-written mirror of `core/selection.py`.

    The two can drift, so run the dock's own cases through the browser copy.
    """

    def test_the_browser_selection_behaves_like_the_python_one(self):
        import shutil
        import tempfile

        from demo_viz.web.validate import find_chrome, serve

        chrome = find_chrome()
        if chrome is None:
            self.skipTest("no Chrome/Chromium available")
        harness = REPO_ROOT / "demo_viz" / "web" / "selection_harness.html"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shutil.copytree(SITE, root, dirs_exist_ok=True)
            shutil.copy(harness, root / "selection_harness.html")
            with serve(root) as port:
                result = subprocess.run(
                    [chrome, "--headless", "--disable-gpu", "--no-sandbox",
                     "--virtual-time-budget=20000", "--dump-dom",
                     f"http://127.0.0.1:{port}/selection_harness.html"],
                    capture_output=True, text=True, timeout=120,
                )
        dom = result.stdout
        start = dom.find('<pre id="out">')
        self.assertGreaterEqual(start, 0, "harness did not render")
        start = dom.index(">", start) + 1
        payload = dom[start:dom.index("</pre>", start)].strip()
        payload = payload.replace("&quot;", '"').replace("&amp;", "&").replace("&lt;", "<")
        self.assertNotEqual("running", payload, "harness timed out")
        report = json.loads(payload)
        self.assertEqual([], report["failures"])
        self.assertGreaterEqual(report["checks"], 15)


if __name__ == "__main__":
    unittest.main(verbosity=2)


class DefaultViewTests(unittest.TestCase):
    """The defaults a first-time visitor lands on, in both front ends."""

    SITE = REPO_ROOT / "demo_viz" / "web" / "site"

    #: user-facing label -> internal layer key
    LAYER_NAMES = {
        "Runner movement": "trail",
        "Defender response": "tether",
        "Space map": "wake",
        "Passing lane": "lane",
        "Defender if stayed": "ghost",
        "Player numbers": "labels",
        "Suggested players": "candidates",
        "Player movements": "paths",
    }
    ON_BY_DEFAULT = {"trail", "tether", "wake", "ghost", "labels", "candidates"}
    OFF_BY_DEFAULT = {"lane", "paths"}

    # -- Dash ----------------------------------------------------------
    def test_dash_layer_labels_are_human_readable(self):
        from demo_viz.app.components import LAYER_OPTIONS

        labels = {o["label"]: o["value"] for o in LAYER_OPTIONS}
        expected = dict(self.LAYER_NAMES)
        expected["Suggested players"] = "candidates"
        self.assertEqual(labels, expected)

    def test_dash_default_layers(self):
        from demo_viz.app.components import DEFAULT_LAYER_VALUES

        self.assertEqual(set(DEFAULT_LAYER_VALUES), self.ON_BY_DEFAULT)
        for key in self.OFF_BY_DEFAULT:
            self.assertNotIn(key, DEFAULT_LAYER_VALUES)

    def test_dash_space_view_defaults_to_available_space(self):
        from dash import dcc

        from demo_viz.app.components import layout

        found = [c for c in _walk(layout()) if getattr(c, "id", None) == "wake-mode"]
        self.assertEqual(len(found), 1)
        control = found[0]
        self.assertIsInstance(control, dcc.RadioItems)
        self.assertEqual(control.value, "space")
        labels = [o["label"] for o in control.options]
        self.assertEqual(labels, ["Available space", "Space created"])
        self.assertEqual(control.options[0]["value"], "space")

    def test_dash_layer_checklist_starts_with_the_intended_boxes(self):
        from demo_viz.app.components import layout

        found = [c for c in _walk(layout()) if getattr(c, "id", None) == "layers"]
        self.assertEqual(set(found[0].value), self.ON_BY_DEFAULT)

    def test_dash_dock_has_no_global_swap_button(self):
        from demo_viz.app.components import layout

        ids = {getattr(c, "id", None) for c in _walk(layout())}
        self.assertNotIn("btn-swap", ids)

    # -- browser -------------------------------------------------------
    def _html(self):
        return (self.SITE / "index.html").read_text()

    def test_browser_layer_labels_are_human_readable(self):
        html = self._html()
        for label, key in self.LAYER_NAMES.items():
            self.assertRegex(html, rf'data-layer="{key}"[^>]*>\s*{label}<')

    def test_browser_default_layers(self):
        html = self._html()
        for key in self.ON_BY_DEFAULT:
            self.assertRegex(html, rf'data-layer="{key}" checked>',
                             f"{key} should start on")
        for key in self.OFF_BY_DEFAULT:
            self.assertNotRegex(html, rf'data-layer="{key}" checked>',
                                f"{key} should start off")

    def test_browser_space_view_defaults_to_available_space(self):
        html = self._html()
        select = html[html.index('id="wake-mode"'):]
        select = select[: select.index("</select>")]
        self.assertLess(select.index('value="space"'), select.index('value="gain"'),
                        "the first option is the default, and it must be Available space")
        self.assertIn("Available space", select)
        self.assertIn("Space created", select)
        self.assertNotIn("Total space", html)
        self.assertNotIn("Opened space", html)

    def test_browser_initial_layer_state_matches_the_markup(self):
        app = (self.SITE / "js" / "app.js").read_text()
        line = next(l for l in app.splitlines() if "layers: new Set(" in l)
        keys = set(re.findall(r'"(\w+)"', line))
        self.assertEqual(keys, self.ON_BY_DEFAULT)

    def test_browser_dock_has_no_global_swap_button(self):
        self.assertNotIn("btn-swap", self._html())


def _walk(component):
    """Every component in a Dash layout tree."""

    yield component
    children = getattr(component, "children", None)
    if children is None:
        return
    if not isinstance(children, (list, tuple)):
        children = [children]
    for child in children:
        if hasattr(child, "children") or hasattr(child, "id"):
            yield from _walk(child)
