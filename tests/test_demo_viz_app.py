"""Tests for the interactive explorer's core logic.

The Dash front end is not exercised here; what is tested is everything the
callbacks delegate to, which is where the behaviour actually lives: the
factored influence cache, the click-to-role rules, the candidate rankings and
the figure builder. Tests needing IDSSE tracking skip themselves with a message
naming the missing input.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from demo_viz.config import demo_paths  # noqa: E402
from demo_viz.core.figures import ViewOptions, scene_figure, space_chart  # noqa: E402
from demo_viz.core.influence import InfluenceCache  # noqa: E402
from demo_viz.core.role_logic import (  # noqa: E402
    auto_triplet,
    rank_beneficiaries,
    rank_defenders,
    runner_onset_index,
)
from demo_viz.core.selection import Selection  # noqa: E402
from demo_viz.quantities import _frame_at, _velocities  # noqa: E402
from demo_viz.sources.synthetic import synthetic_scene  # noqa: E402

from offball_value.goal_weighted_influence import target_residual_influence  # noqa: E402


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.scene = synthetic_scene()
        self.runner = self.scene.by_shirt("7", "attack").player_id
        self.defender = self.scene.by_shirt("11", "defend").player_id
        self.beneficiary = self.scene.by_shirt("34", "attack").player_id

    def test_runner_defender_beneficiary_flow(self):
        selection = Selection()
        selection.apply_click(self.scene, self.runner)
        self.assertEqual(selection.runners, [self.runner])
        self.assertEqual(selection.pick, "defender")
        selection.apply_click(self.scene, self.defender)
        self.assertEqual(selection.defenders, [self.defender])
        self.assertEqual(selection.pick, "beneficiary")
        selection.apply_click(self.scene, self.beneficiary)
        self.assertEqual(selection.beneficiaries, [self.beneficiary])

    def test_clicking_a_selected_player_drops_the_role(self):
        selection = Selection()
        selection.apply_click(self.scene, self.runner)
        selection.apply_click(self.scene, self.defender)
        selection.apply_click(self.scene, self.defender)
        self.assertEqual(selection.defenders, [])

    def test_the_clicked_side_wins_over_the_pick_target(self):
        selection = Selection(pick="runner")
        selection.apply_click(self.scene, self.defender)
        self.assertEqual(selection.defenders, [self.defender])
        self.assertEqual(selection.runners, [])

    def test_an_attacker_fills_the_runner_slot_then_the_beneficiary_slot(self):
        selection = Selection()
        selection.apply_click(self.scene, self.runner)
        selection.apply_click(self.scene, self.beneficiary)
        self.assertEqual(selection.runners, [self.runner])
        self.assertEqual(selection.beneficiaries, [self.beneficiary])

    def test_a_second_click_always_undoes_whatever_role_the_player_held(self):
        selection = Selection()
        selection.apply_click(self.scene, self.runner)
        selection.apply_click(self.scene, self.beneficiary)
        selection.pick = "runner"
        selection.apply_click(self.scene, self.beneficiary)
        self.assertEqual(selection.beneficiaries, [])
        self.assertEqual(selection.runners, [self.runner])

    def test_a_player_never_holds_two_roles(self):
        selection = Selection()
        for player in self.scene.players.values():
            selection.apply_click(self.scene, player.player_id)
        for player_id in self.scene.players:
            held = sum(player_id in selection.ids(role)
                       for role in ("runner", "defender", "beneficiary"))
            self.assertLessEqual(held, 1)

    def test_only_one_runner_but_several_defenders(self):
        selection = Selection()
        selection.apply_click(self.scene, self.runner)
        other = self.scene.by_shirt("9", "attack").player_id
        selection.pick = "runner"
        selection.apply_click(self.scene, other)
        self.assertEqual(selection.runners, [other])
        for shirt in ("20", "21"):
            player = self.scene.by_shirt(shirt, "defend")
            if player is not None:
                selection.pick = "defender"
                selection.apply_click(self.scene, player.player_id)
        self.assertGreaterEqual(len(selection.defenders), 1)

    def test_round_trips_through_json_shaped_dict(self):
        selection = Selection()
        selection.apply_click(self.scene, self.runner)
        restored = Selection.from_dict(selection.to_dict())
        self.assertEqual(restored.runners, selection.runners)
        self.assertEqual(restored.pick, selection.pick)

    def test_unknown_player_is_ignored(self):
        selection = Selection()
        selection.apply_click(self.scene, "not-a-player")
        self.assertTrue(selection.is_empty)


class InfluenceCacheTests(unittest.TestCase):
    def setUp(self):
        self.scene = synthetic_scene()
        self.cache = InfluenceCache(self.scene, every=20, grid_resolution_m=4.0)

    def test_matches_the_repository_residual_exactly(self):
        """The whole point of the cache: same numbers, far fewer evaluations."""

        scene, cache = self.scene, self.cache
        targets = [p.player_id for p in scene.players_on("attack") if not p.is_goalkeeper][:4]
        for slot, index in list(enumerate(cache.indices))[:3]:
            frame = _frame_at(scene, int(index))
            velocities = _velocities(scene, int(index))
            for target in targets:
                reference = target_residual_influence(
                    frame, velocities, target, scene.attacking_team_id,
                    scene.attacking_direction, goalkeeper_ids=cache.goalkeepers,
                    config=cache.config, xgrid=cache.xgrid, ygrid=cache.ygrid,
                )
                mine = cache.residual(target, slot)
                np.testing.assert_allclose(mine.field, reference.residual_surface,
                                           rtol=0, atol=1e-12)
                self.assertAlmostEqual(mine.value, reference.residual_value, places=9)

    def test_swapping_a_defender_changes_the_field(self):
        defender = self.scene.by_shirt("11", "defend").player_id
        target = self.scene.by_shirt("34", "attack").player_id
        slot = len(self.cache.indices) - 1
        plain = self.cache.residual(target, slot)
        swapped = self.cache.residual(target, slot, ((defender, 0, "hold"),))
        self.assertNotAlmostEqual(plain.value, swapped.value, places=6)

    def test_swap_before_the_freeze_frame_changes_nothing(self):
        defender = self.scene.by_shirt("11", "defend").player_id
        target = self.scene.by_shirt("34", "attack").player_id
        freeze = int(self.cache.indices[-1])
        slot = 0
        plain = self.cache.residual(target, slot)
        swapped = self.cache.residual(target, slot, ((defender, freeze, "hold"),))
        self.assertAlmostEqual(plain.value, swapped.value, places=9)

    def test_unknown_target_returns_an_empty_field(self):
        result = self.cache.residual("nobody", 0)
        self.assertEqual(result.value, 0.0)
        self.assertEqual(float(np.max(result.field)), 0.0)


class RankingTests(unittest.TestCase):
    def setUp(self):
        self.scene = synthetic_scene()
        self.cache = InfluenceCache(self.scene, every=20, grid_resolution_m=4.0)
        self.runner = self.scene.by_shirt("7", "attack").player_id

    def test_defender_ranking_is_ordered_and_excludes_keepers(self):
        rows = rank_defenders(self.scene, self.runner)
        self.assertTrue(rows)
        scores = [row.score for row in rows]
        self.assertEqual(scores, sorted(scores, reverse=True))
        keepers = {p.player_id for p in self.scene.players.values() if p.is_goalkeeper}
        self.assertFalse({row.player_id for row in rows} & keepers)
        for row in rows:
            self.assertGreaterEqual(row.score, 0.0)
            self.assertLessEqual(row.score, 1.0)

    def test_beneficiary_ranking_excludes_the_runner(self):
        defender = self.scene.by_shirt("11", "defend").player_id
        rows = rank_beneficiaries(self.cache, self.runner, (defender,), 0, 0)
        self.assertTrue(rows)
        self.assertNotIn(self.runner, [row.player_id for row in rows])
        gains = [row.gain for row in rows]
        self.assertEqual(gains, sorted(gains, reverse=True))

    def test_beneficiary_ranking_needs_a_defender(self):
        self.assertEqual(rank_beneficiaries(self.cache, self.runner, (), 0, 0), [])

    def test_onset_reports_how_it_was_found(self):
        index, method = runner_onset_index(self.scene, self.runner)
        self.assertIsInstance(index, int)
        self.assertIn(method, {"run onset detector", "peak acceleration", "clip start"})

    def test_auto_triplet_returns_three_distinct_players(self):
        triplet = auto_triplet(self.scene, self.cache)
        self.assertIn("runner", triplet)
        ids = [triplet.get(role) for role in ("runner", "defender", "beneficiary")]
        ids = [i for i in ids if i]
        self.assertEqual(len(ids), len(set(ids)))


class FigureTests(unittest.TestCase):
    def setUp(self):
        self.scene = synthetic_scene()
        self.cache = InfluenceCache(self.scene, every=20, grid_resolution_m=4.0)

    def _selection(self):
        selection = Selection()
        selection.apply_click(self.scene, self.scene.by_shirt("7", "attack").player_id)
        selection.apply_click(self.scene, self.scene.by_shirt("11", "defend").player_id)
        selection.apply_click(self.scene, self.scene.by_shirt("34", "attack").player_id)
        return selection

    def test_every_player_marker_carries_its_id(self):
        figure = scene_figure(self.scene, 10, self._selection(), self.cache, ViewOptions())
        found = set()
        for trace in figure.data:
            for value in (getattr(trace, "customdata", None) or []):
                found.add(value[0] if isinstance(value, (list, tuple)) else value)
        expected = {p.player_id for p in self.scene.players.values()
                    if np.all(np.isfinite(p.xy[10]))}
        self.assertEqual(found, expected)

    def test_renders_with_no_selection_and_no_cache(self):
        scene_figure(self.scene, 0, Selection(), None, ViewOptions())
        space_chart(self.scene, None, Selection(), 0, None)

    def test_layer_toggles_change_the_trace_count(self):
        selection = self._selection()
        full = scene_figure(self.scene, 40, selection, self.cache,
                            ViewOptions(layers=("trail", "tether", "wake", "ghost", "lane")))
        bare = scene_figure(self.scene, 40, selection, self.cache, ViewOptions(layers=()))
        self.assertGreater(len(full.data), len(bare.data))

    def test_fit_crops_the_view(self):
        selection = self._selection()
        wide = scene_figure(self.scene, 40, selection, self.cache, ViewOptions(layers=()))
        close = scene_figure(self.scene, 40, selection, self.cache,
                             ViewOptions(layers=("fit",)))
        wide_x = wide.layout.xaxis.range
        close_x = close.layout.xaxis.range
        self.assertLess(close_x[1] - close_x[0], wide_x[1] - wide_x[0])

    def test_index_out_of_range_is_clamped(self):
        scene_figure(self.scene, 10_000, self._selection(), self.cache, ViewOptions())


@unittest.skipIf(demo_paths().idsse_dir is None,
                 "IDSSE positions XML not found; set OFFBALL_IDSSE_DIR to enable")
@unittest.skipIf(demo_paths().annotations_xlsx is None,
                 "shot_annotations.xlsx not found; set OFFBALL_ANNOTATIONS to enable")
class RealSceneAppTests(unittest.TestCase):
    def test_annotation_mode_preloads_the_annotated_triplet(self):
        from demo_viz.app.state import get_bundle

        bundle = get_bundle("J03WOH:shot_006_P1_1054")
        selection = Selection.from_scene(bundle.scene)
        shirts = {
            role: sorted(bundle.scene.players[p].label for p in selection.ids(role))
            for role in ("runner", "defender", "beneficiary")
        }
        self.assertEqual(shirts["runner"], ["7"])
        self.assertEqual(shirts["defender"], ["11"])
        self.assertEqual(shirts["beneficiary"], ["34"])

    def test_the_annotated_beneficiary_tops_the_gain_ranking(self):
        """The guided ranking should agree with the human on the clean 1-1-1 scene."""

        from demo_viz.app.state import get_bundle, onset_for

        bundle = get_bundle("J03WOH:shot_006_P1_1054")
        scene, cache = bundle.scene, bundle.cache
        runner = scene.by_shirt("7", "attack").player_id
        defender = scene.by_shirt("11", "defend").player_id
        freeze, _ = onset_for("J03WOH:shot_006_P1_1054", runner)
        rows = rank_beneficiaries(cache, runner, (defender,), freeze,
                                  cache.slot_for(freeze))
        self.assertEqual(scene.players[rows[0].player_id].label, "34")

    def test_scene_bundle_is_cached(self):
        from demo_viz.app.state import get_bundle

        first = get_bundle("J03WOH:shot_006_P1_1054")
        self.assertIs(first, get_bundle("J03WOH:shot_006_P1_1054"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
