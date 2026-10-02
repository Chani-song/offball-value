"""Tests for the demo_viz renderer.

These run without IDSSE data: everything is exercised on the synthetic
smoke-test scene, so the suite stays green on a machine that only has the
repository checked out.  Tests that need real tracking are skipped with a
message naming what is missing.
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

from demo_viz import palette  # noqa: E402
from demo_viz.annotations import parse_shirt_list  # noqa: E402
from demo_viz.config import demo_paths  # noqa: E402
from demo_viz.loader import load_scene, presentation_for, scene_book  # noqa: E402
from demo_viz.quantities import attach_quantities, detect_moments  # noqa: E402
from demo_viz.render.camera import Camera, CameraConfig  # noqa: E402
from demo_viz.render.figure import FigureConfig, SceneFigure  # noqa: E402
from demo_viz.scene import Scene  # noqa: E402
from demo_viz.sources.pipeline import scene_from_record  # noqa: E402
from demo_viz.sources.synthetic import synthetic_scene  # noqa: E402
from demo_viz.story import LAYERS, build_storyboard  # noqa: E402


class ShirtParsingTests(unittest.TestCase):
    def test_parses_the_spreadsheet_formats(self):
        self.assertEqual(parse_shirt_list("7"), ("7",))
        self.assertEqual(parse_shirt_list("6, 34"), ("6", "34"))
        self.assertEqual(parse_shirt_list("9,25"), ("9", "25"))
        self.assertEqual(parse_shirt_list("23 / 11"), ("23", "11"))

    def test_blank_values_yield_no_roles(self):
        for value in (None, "", "  ", float("nan"), "nan", "-"):
            self.assertEqual(parse_shirt_list(value), ())


class SyntheticSceneTests(unittest.TestCase):
    def setUp(self):
        self.scene = synthetic_scene()

    def test_scene_is_self_consistent(self):
        scene = self.scene
        self.assertEqual(scene.source, "synthetic")
        self.assertEqual(len(scene.ball_xy), scene.n_frames)
        for player in scene.players.values():
            self.assertEqual(len(player.xy), scene.n_frames)
        for player_id in scene.role_ids:
            self.assertIn(player_id, scene.players)

    def test_roles_are_exclusive(self):
        scene = self.scene
        self.assertEqual(scene.role_of(scene.runner_ids[0]), "runner")
        self.assertEqual(scene.role_of(scene.defender_ids[0]), "defender")
        self.assertEqual(scene.role_of(scene.beneficiary_ids[0]), "beneficiary")
        self.assertIsNone(scene.role_of("SYN-A-09"))

    def test_view_transform_mirrors_when_attacking_left(self):
        scene = self.scene
        self.assertFalse(scene.flip)
        scene.attacking_direction = -1
        self.assertTrue(scene.flip)
        moved = scene.view_xy([[10.0, 4.0]])
        self.assertAlmostEqual(moved[0][0], -10.0)
        self.assertAlmostEqual(moved[0][1], -4.0)

    def test_quantities_attach_without_inventing_unavailable_ones(self):
        scene = attach_quantities(self.scene, every=10, grid_resolution_m=3.0)
        self.assertIsNotNone(scene.surfaces)
        self.assertIn("_factual_value", scene.series)
        self.assertIn("_counterfactual_value", scene.series)
        # Hooks that this repository cannot currently fill must stay empty.
        self.assertIsNone(scene.pass_probability)
        self.assertIsNone(scene.dribble_probability)
        self.assertIsNone(scene.threat_surface)

    def test_counterfactual_differs_from_the_factual_field(self):
        scene = attach_quantities(self.scene, every=10, grid_resolution_m=3.0)
        stack = scene.surfaces
        self.assertIsNotNone(stack.counterfactual)
        self.assertIsNotNone(stack.delta)
        self.assertTrue(abs(stack.delta).max() > 0.0)


class StoryboardTests(unittest.TestCase):
    def setUp(self):
        self.scene = attach_quantities(synthetic_scene(), every=10, grid_resolution_m=3.0)
        self.storyboard = build_storyboard(self.scene)

    def test_beats_are_ordered_and_inside_the_window(self):
        starts = [beat.start for beat in self.storyboard.beats]
        self.assertEqual(starts, sorted(starts))
        self.assertGreaterEqual(starts[0], float(self.scene.times[0]) - 1e-6)
        self.assertLessEqual(starts[-1], float(self.scene.times[-1]) + 1e-6)

    def test_final_beat_has_screen_time(self):
        tail = float(self.scene.times[-1]) - self.storyboard.beats[-1].start
        self.assertGreater(tail, 0.5)

    def test_state_covers_every_layer_and_stays_in_range(self):
        for t in self.scene.times[::17]:
            state = self.storyboard.state(float(t))
            self.assertEqual(set(state), set(LAYERS))
            for name, value in state.items():
                self.assertGreaterEqual(value, -1e-9, name)
                self.assertLessEqual(value, 1.0 + 1e-9, name)

    def test_chain_lights_up_in_order(self):
        last = self.storyboard.chain_progress(float(self.scene.times[-1]))
        self.assertTrue(all(value > 0.99 for value in last))
        first = self.storyboard.chain_progress(float(self.scene.times[0]))
        self.assertTrue(all(value < 0.01 for value in first))


class CameraTests(unittest.TestCase):
    def test_viewport_keeps_the_axes_aspect_and_follows_the_cast(self):
        scene = synthetic_scene()
        camera = Camera(scene, aspect=1.6, config=CameraConfig())
        for index in (0, scene.n_frames // 2, scene.n_frames - 1):
            x0, x1, y0, y1 = camera.viewport(index, zoom=1.0)
            self.assertAlmostEqual((x1 - x0) / (y1 - y0), 1.6, places=5)
            self.assertLessEqual(x1 - x0, scene.pitch_length + 1e-6)

    def test_zoom_zero_is_the_full_pitch(self):
        scene = synthetic_scene()
        camera = Camera(scene, aspect=1.6)
        wide = camera.viewport(0, zoom=0.0)
        close = camera.viewport(0, zoom=1.0)
        self.assertGreater(wide[1] - wide[0], close[1] - close[0])

    def test_camera_is_smooth(self):
        scene = synthetic_scene()
        camera = Camera(scene, aspect=1.6)
        centres = [sum(camera.viewport(i, 1.0)[:2]) / 2 for i in range(scene.n_frames)]
        jumps = [abs(b - a) for a, b in zip(centres, centres[1:])]
        self.assertLess(max(jumps), 0.6)      # metres per 1/25 s


class PaletteTests(unittest.TestCase):
    def test_story_roles_separate_in_normal_and_dichromat_vision(self):
        roles = [palette.RUNNER, palette.DEFENDER, palette.BENEFICIARY]
        for index, first in enumerate(roles):
            for second in roles[index + 1:]:
                scores = [palette.delta_e(first, second)] + [
                    palette.delta_e(palette.simulate_cvd(first, kind),
                                    palette.simulate_cvd(second, kind))
                    for kind in ("protan", "deutan", "tritan")
                ]
                self.assertGreaterEqual(min(scores), 14.0, f"{first} vs {second}: {scores}")

    def test_roles_stand_out_against_the_pitch(self):
        for colour in (palette.RUNNER, palette.DEFENDER, palette.BENEFICIARY, palette.BALL):
            self.assertGreater(palette.delta_e(colour, palette.PITCH_DARK), 30.0)


class RenderTests(unittest.TestCase):
    def test_every_beat_renders_without_error(self):
        scene = attach_quantities(synthetic_scene(), every=15, grid_resolution_m=4.0)
        storyboard = build_storyboard(scene)
        figure = SceneFigure(scene, storyboard,
                             FigureConfig(width_px=800, height_px=450, dpi=50))
        for beat in storyboard.beats:
            figure.draw(scene.index_at(beat.start))

    def test_still_export_writes_a_file(self):
        from demo_viz.animate import render_still

        scene = attach_quantities(synthetic_scene(), every=20, grid_resolution_m=4.0)
        storyboard = build_storyboard(scene)
        with tempfile.TemporaryDirectory() as tmp:
            out = render_still(scene, storyboard, Path(tmp) / "frame.png",
                               figure_config=FigureConfig(width_px=640, height_px=360, dpi=40))
            self.assertTrue(out.exists() and out.stat().st_size > 1000)

    def test_renders_without_any_optional_quantity(self):
        scene = synthetic_scene()          # no attach_quantities: surfaces is None
        storyboard = build_storyboard(scene, wake_is_illustrative=True)
        config = FigureConfig(width_px=640, height_px=360, dpi=40, wake_mode="geometric")
        figure = SceneFigure(scene, storyboard, config)
        figure.draw(scene.n_frames - 1)


class SceneBookTests(unittest.TestCase):
    def test_scene_book_is_valid_json_with_the_expected_shape(self):
        book = scene_book()
        self.assertIn("defaults", book)
        self.assertIn("scenes", book)
        for scene_id, settings in book["scenes"].items():
            self.assertIn(":", scene_id)
            self.assertTrue(set(settings) <= {"headline", "before_s", "after_s",
                                              "camera", "wake_mode"})

    def test_presentation_merges_defaults(self):
        merged = presentation_for("J03WOH:shot_006_P1_1054")
        self.assertIn("before_s", merged)


class PipelineAdapterTests(unittest.TestCase):
    def test_missing_keys_raise_a_helpful_error(self):
        with self.assertRaises(ValueError) as caught:
            scene_from_record({"runner": ["7"]})
        self.assertIn("match_id", str(caught.exception))

    def test_record_without_roles_is_rejected(self):
        with self.assertRaises(ValueError) as caught:
            scene_from_record({"match_id": "J03WOH", "period": 1, "period_seconds": 100.0})
        self.assertIn("role", str(caught.exception))


@unittest.skipIf(demo_paths().idsse_dir is None,
                 "IDSSE positions XML not found; set OFFBALL_IDSSE_DIR to enable")
@unittest.skipIf(demo_paths().annotations_xlsx is None,
                 "shot_annotations.xlsx not found; set OFFBALL_ANNOTATIONS to enable")
class RealSceneTests(unittest.TestCase):
    def test_hero_scene_resolves_all_three_roles(self):
        scene = load_scene("J03WOH:shot_006_P1_1054", quantities=False)
        self.assertEqual([scene.players[i].shirt for i in scene.runner_ids], ["7"])
        self.assertEqual([scene.players[i].shirt for i in scene.defender_ids], ["11"])
        self.assertEqual([scene.players[i].shirt for i in scene.beneficiary_ids], ["34"])
        self.assertEqual(len(scene.players), 22)

    def test_second_half_scenes_resolve_too(self):
        scene = load_scene("J03WOY:shot_011_P2_0903", quantities=False)
        self.assertEqual(len(scene.players), 22)
        self.assertTrue(scene.runner_ids and scene.defender_ids and scene.beneficiary_ids)

    def test_moments_come_from_named_detectors(self):
        scene = load_scene("J03WOH:shot_006_P1_1054", quantities=False)
        moments, methods = detect_moments(scene)
        self.assertIn("onset", moments)
        self.assertIn("run_onset", methods["onset"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
