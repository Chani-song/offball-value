"""The solver's own release chain, checked against the research code.

These are the regression tests for `demo_viz/PASS_MODEL_TRACE.md`. They pin the
numbers the trace reproduced, so a change in the vendored payoff, the model
artifact or the coordinate conversion shows up here rather than on screen.

Everything skips cleanly when the research stack or the fitted model is absent,
which is the same condition the export step degrades on.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from demo_viz.solver_native import (  # noqa: E402
    GROUND,
    PANEL_FEATURES,
    availability,
    release_quantities,
    solver_release_targets,
)
from demo_viz.solver_native.release import _corner, _stack, research_src  # noqa: E402

AVAILABLE = availability().ok
SKIP = f"research stack unavailable: {availability().reason}"

#: The real declared pipeline state the trace reproduced (conditioning_100 #0),
#: in the imported code's corner-origin metres.
STATE_CARRIER = (63.0782029956123, 32.648743934119906)
STATE_RECEIVER = (73.85384588940612, 49.41937142564361)
STATE_DEFENDER = (53.75952274666707, 47.855527919685734)
EXPECTED_LOGIT = 6.498830792478147
EXPECTED_PROXY = 0.998497064162670


def _to_centre(xy):
    """Corner origin -> this repository's centre-spot origin."""

    return (xy[0] - 105.0 / 2.0, xy[1] - 68.0 / 2.0)


class AvailabilityTests(unittest.TestCase):
    """The chain is optional, and says what it is missing."""

    def test_availability_reports_both_dependencies(self):
        state = availability()
        self.assertIsInstance(state.ok, bool)
        self.assertEqual(state.ok, state.research and state.model)
        if not state.ok:
            self.assertTrue(state.reason)

    def test_everything_degrades_to_none_when_unavailable(self):
        """A fresh checkout must not raise; it must return nothing."""

        if AVAILABLE:
            self.skipTest("stack is present here; the absent path is covered below")
        self.assertIsNone(release_quantities((0, 0), (5, 0), (10, 0), [[3, 3]], 1))
        self.assertIsNone(solver_release_targets((5, 0), 1))

    def test_the_model_artifact_is_never_committed(self):
        import subprocess

        out = subprocess.run(["git", "ls-files", "local_inputs"],
                             cwd=REPO_ROOT, capture_output=True, text=True)
        self.assertEqual("", out.stdout.strip())


@unittest.skipUnless(AVAILABLE, SKIP)
class RealStateReproductionTests(unittest.TestCase):
    """§16 A: the trace's numbers, from the same code path."""

    def test_logit_and_completion_proxy_match_the_trace(self):
        stack = _stack()
        model = stack["model"]
        carrier = np.asarray(STATE_CARRIER)
        receiver = np.asarray(STATE_RECEIVER)
        defender = np.asarray([STATE_DEFENDER])
        target = receiver                      # solver pass[0]: ground, zero offset

        features = stack["position_features"](carrier, receiver, defender, target,
                                              GROUND, 1, 105.0, 68.0)
        logit = float(((features - model.mean) / model.scale) @ model.weights + model.bias)
        proxy = float(model.probability(features))

        self.assertAlmostEqual(EXPECTED_LOGIT, logit, places=12)
        self.assertAlmostEqual(EXPECTED_PROXY, proxy, places=12)

    def test_an_independent_calculation_agrees_to_machine_precision(self):
        from demo_viz.solver_native.release import model_path

        raw = json.loads(model_path().read_text())
        stack = _stack()
        features = np.asarray(stack["position_features"](
            np.asarray(STATE_CARRIER), np.asarray(STATE_RECEIVER),
            np.asarray([STATE_DEFENDER]), np.asarray(STATE_RECEIVER),
            GROUND, 1, 105.0, 68.0), dtype=float)
        z = (features - np.asarray(raw["mean"])) / np.asarray(raw["scale"])
        logit = float(z @ np.asarray(raw["weights"]) + raw["bias"])
        proxy = 1.0 / (1.0 + np.exp(-logit))
        self.assertLess(abs(proxy - EXPECTED_PROXY), 1e-15)

    def test_the_demo_entry_point_reproduces_it_from_centre_origin(self):
        """The coordinate conversion must not move the answer."""

        result = release_quantities(
            _to_centre(STATE_CARRIER), _to_centre(STATE_RECEIVER),
            _to_centre(STATE_RECEIVER), [_to_centre(STATE_DEFENDER)], 1, GROUND)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(EXPECTED_PROXY, result["completion_proxy"], places=12)

    def test_the_conversion_is_the_imported_convention(self):
        np.testing.assert_allclose(_corner(_to_centre(STATE_CARRIER)), STATE_CARRIER)


@unittest.skipUnless(AVAILABLE, SKIP)
class BackgroundDefenderTests(unittest.TestCase):
    """§16 B: the whole defending side must reach the model."""

    #: State 3 of conditioning_100, solver pass[6] (ground, offset (8, 0)) --
    #: the example PASS_MODEL_TRACE.md section 3 measured.
    CARRIER = (59.60899540410995, 20.509814893145343)
    RECEIVER = (73.06486636437934, 38.82258411287641)
    CONTROLLED = (74.81123804382483, 27.758996397368044)
    #: one background defender placed nearer the receiver than the controlled one
    BACKGROUND = (RECEIVER[0] - 3.0, RECEIVER[1] + 2.0)
    PROXY_ALONE = 0.992984160123
    PROXY_WITH_BACKGROUND = 0.907019833891

    def _proxy(self, defenders):
        stack = _stack()
        carrier = np.asarray(self.CARRIER)
        receiver = np.asarray(self.RECEIVER)
        target = receiver + np.array([8.0, 0.0])
        features = stack["position_features"](carrier, receiver,
                                              np.asarray(defenders, dtype=float),
                                              target, GROUND, 1, 105.0, 68.0)
        return float(stack["model"].probability(features)), np.asarray(features)

    def test_a_background_defender_moves_the_proxy(self):
        """The number PASS_MODEL_TRACE.md section 3 reports, pinned."""

        alone, _ = self._proxy([self.CONTROLLED])
        both, _ = self._proxy([self.CONTROLLED, self.BACKGROUND])
        self.assertAlmostEqual(self.PROXY_ALONE, alone, places=9)
        self.assertAlmostEqual(self.PROXY_WITH_BACKGROUND, both, places=9)
        self.assertLess(both, alone, "the extra defender should lower the proxy")

    def test_only_the_receiver_block_and_same_defender_move(self):
        """Adding a defender nearer the receiver must not touch the passer block."""

        stack = _stack()
        alone, f_one = self._proxy([self.CONTROLLED])
        both, f_two = self._proxy([self.CONTROLLED, self.BACKGROUND])

        names = list(stack["features"])
        moved = {n for n, a, b in zip(names, f_one, f_two) if abs(a - b) > 1e-12}
        self.assertTrue(moved, "the second defender changed nothing")
        for name in moved:
            self.assertTrue(
                name.startswith("defender_near_receiver_") or name == "same_defender",
                f"{name} moved, but only the receiver block and same_defender should")
        self.assertNotEqual(alone, both)

    def test_a_single_defender_sets_same_defender(self):
        _, features = self._proxy([self.CONTROLLED])
        names = list(_stack()["features"])
        self.assertEqual(1.0, float(features[names.index("same_defender")]))


@unittest.skipUnless(AVAILABLE, SKIP)
class LegalityGateTests(unittest.TestCase):
    """§16 C: legality zeroes the payoff without hiding the proxy."""

    def test_an_offside_target_keeps_its_proxy_but_loses_the_payoff(self):
        # receiver well beyond the whole defending side, attacking to +x
        result = release_quantities(
            carrier_xy=(0.0, 0.0), receiver_xy=(30.0, 4.0), target_xy=(36.0, 4.0),
            defender_xys=[[5.0, 2.0], [8.0, -3.0], [2.0, 6.0]],
            attacking_direction=1)
        self.assertTrue(result["offside"])
        self.assertFalse(result["legal"])
        self.assertGreater(result["completion_proxy"], 0.0)
        self.assertEqual(0.0, result["release_payoff"])

    def test_a_target_off_the_pitch_is_illegal(self):
        result = release_quantities((40.0, 0.0), (48.0, 0.0), (60.0, 0.0),
                                    [[45.0, 2.0], [50.0, 1.0]], 1)
        self.assertFalse(result["inside_pitch"])
        self.assertFalse(result["legal"])
        self.assertEqual(0.0, result["release_payoff"])

    def test_the_multi_defender_offside_rule_is_used(self):
        """Two or more defenders must use the second-last rule, not the line."""

        source = (REPO_ROOT / "demo_viz" / "solver_native"
                  / "_passer2on1_payoff.py").read_text()
        self.assertIn("if defenders.shape[-2] >= 2:", source)
        self.assertIn("return offside(receiver[..., 0]", source)


@unittest.skipUnless(AVAILABLE, SKIP)
class ReleasePayoffTests(unittest.TestCase):
    """§16 D: the payoff is exactly legal x completion x threat."""

    CASES = (
        ((0.0, 0.0), (14.0, 5.0), (20.0, 7.0), [[6.0, 2.0], [16.0, 9.0], [-4.0, 1.0]]),
        ((-20.0, -8.0), (-6.0, 2.0), (2.0, 4.0), [[-10.0, -2.0], [0.0, 3.0]]),
        ((10.0, 10.0), (22.0, 2.0), (30.0, -2.0), [[18.0, 6.0], [26.0, 0.0], [12.0, 12.0]]),
    )

    def test_the_product_holds_exactly(self):
        for carrier, receiver, target, defenders in self.CASES:
            for direction in (1, -1):
                with self.subTest(target=target, direction=direction):
                    r = release_quantities(carrier, receiver, target, defenders, direction)
                    expected = (r["completion_proxy"] * r["positional_threat"]
                                if r["legal"] else 0.0)
                    self.assertEqual(expected, r["release_payoff"])

    def test_threat_stays_inside_its_documented_bounds(self):
        for carrier, receiver, target, defenders in self.CASES:
            r = release_quantities(carrier, receiver, target, defenders, 1)
            self.assertGreaterEqual(r["positional_threat"], 0.2 - 1e-9)
            self.assertLessEqual(r["positional_threat"], 1.0 + 1e-9)

    def test_the_proxy_is_a_number_between_zero_and_one(self):
        for carrier, receiver, target, defenders in self.CASES:
            r = release_quantities(carrier, receiver, target, defenders, 1)
            self.assertGreaterEqual(r["completion_proxy"], 0.0)
            self.assertLessEqual(r["completion_proxy"], 1.0)

    def test_the_panel_features_are_a_subset_of_the_model_features(self):
        names = set(_stack()["features"])
        for key, _, _ in PANEL_FEATURES:
            self.assertIn(key, names, key)

    def test_all_thirty_three_features_are_reported(self):
        r = release_quantities((0.0, 0.0), (14.0, 5.0), (20.0, 7.0), [[6.0, 2.0]], 1)
        self.assertEqual(33, len(r["all_features"]))


@unittest.skipUnless(AVAILABLE, SKIP)
class SolverActionTests(unittest.TestCase):
    """The solver's own 18 releases, read from the imported definition."""

    def test_eighteen_actions_built_from_the_receiver(self):
        targets = solver_release_targets((10.0, 4.0), 1)
        self.assertEqual(18, len(targets))
        offsets = {t["offset"] for t in targets}
        self.assertEqual({(0.0, 0.0), (4.0, 0.0), (8.0, 0.0),
                          (-4.0, 0.0), (0.0, 4.0), (0.0, -4.0)}, offsets)
        self.assertEqual({"ground", "driven", "lofted"}, {t["family"] for t in targets})

    def test_both_axes_rotate_with_the_attacking_direction(self):
        forward = solver_release_targets((10.0, 4.0), 1)
        reverse = solver_release_targets((10.0, 4.0), -1)
        by_key = {(t["offset"], t["family"]): t["target"] for t in reverse}
        for entry in forward:
            mirrored = by_key[(entry["offset"], entry["family"])]
            offset = entry["offset"]
            self.assertAlmostEqual(10.0 - offset[0], mirrored[0])
            self.assertAlmostEqual(4.0 - offset[1], mirrored[1])

    def test_these_are_not_the_demo_exploratory_rays(self):
        """The solver's targets orbit the receiver; the demo's rays leave the carrier."""

        from demo_viz.core.candidates import CANDIDATE_ANGLES_DEG, CANDIDATE_DISTANCE_M

        targets = solver_release_targets((10.0, 4.0), 1)
        radii = {round(((t["target"][0] - 10.0) ** 2
                        + (t["target"][1] - 4.0) ** 2) ** 0.5, 6) for t in targets}
        self.assertNotIn(CANDIDATE_DISTANCE_M, radii)
        self.assertEqual(5, len(CANDIDATE_ANGLES_DEG))


class VendoredPayoffTests(unittest.TestCase):
    """The vendored payoff must stay diffable against the branch it came from."""

    def test_it_is_byte_identical_to_kyuhyeok_dev(self):
        import subprocess

        out = subprocess.run(
            ["git", "show",
             "origin/kyuhyeok-dev:andrew-passer2on1/passer2on1/payoff.py"],
            cwd=REPO_ROOT, capture_output=True, text=True)
        if out.returncode != 0:
            self.skipTest("origin/kyuhyeok-dev not available")
        local = (REPO_ROOT / "demo_viz" / "solver_native"
                 / "_passer2on1_payoff.py").read_text()
        self.assertEqual(out.stdout, local)


class PublicUiTests(unittest.TestCase):
    """What the public interface now exposes, and what it no longer does."""

    def setUp(self):
        from demo_viz.web.build import SITE

        self.site = SITE
        self.markup = (SITE / "index.html").read_text()
        self.app = (SITE / "js" / "app.js").read_text()

    def test_the_legacy_threat_view_is_gone_from_the_public_control(self):
        import re

        self.assertEqual(["space", "gain"],
                         re.findall(r'<option value="([a-z]+)"', self.markup))

    def test_the_analysis_panel_leads_with_the_solver_chain(self):
        order = [self.markup.index(f'id="an-{name}"')
                 for name in ("scene", "offball", "pass", "player")]
        self.assertEqual(sorted(order), order,
                         "Scene, Off-ball effect, Solver pass model, Player")

    def test_the_four_solver_rows_are_rendered(self):
        block = self.app[self.app.index("function renderReleaseRows"):]
        block = block[:block.index("\nfunction ")]
        for label in ("Legal", "Completion proxy", "Positional threat", "Release payoff"):
            self.assertIn(f'"{label}"', block, label)

    def test_the_panel_never_says_probability_for_the_proxy(self):
        block = self.app[self.app.index("function renderReleaseRows"):]
        block = block[:block.index("\nfunction ")]
        for banned in ("success probability", "pass probability", "expected value",
                       "best pass", "optimal pass"):
            self.assertNotIn(banned, block.lower(), banned)

    def test_the_explorer_is_not_called_candidate_passes(self):
        self.assertIn("Explore pass model", self.markup)
        self.assertNotIn("> Candidate passes", self.markup)

    def test_a_missing_receiver_asks_for_one_rather_than_guessing(self):
        release = (self.site / "js" / "release.js").read_text()
        self.assertIn("Select a receiving attacker", release)
        # nothing may pick a receiver on the user's behalf
        for banned in ("nearest attacker", "closest attacker", "autoReceiver"):
            self.assertNotIn(banned, release)

    def test_the_receiver_is_the_selected_beneficiary(self):
        block = self.app[self.app.index("function selectedReceiver"):]
        self.assertIn("state.selection.beneficiaries[0]", block[:200])


class ExportContractTests(unittest.TestCase):
    """The exported payload is derived numbers, and it is optional."""

    RELEASE_DIR = REPO_ROOT / "demo_viz" / "web_data" / "release"

    def test_the_export_skips_cleanly_without_the_research_stack(self):
        source = (REPO_ROOT / "demo_viz" / "web" / "export_release.py").read_text()
        self.assertIn("return 0", source.split("state = availability()")[1][:600],
                      "a missing research stack must not fail the build")

    @unittest.skipIf(not (RELEASE_DIR / "model.json").exists(), "no exported release data")
    def test_the_model_card_states_the_proxy_status(self):
        card = json.loads((self.RELEASE_DIR / "model.json").read_text())
        self.assertEqual("experimental_proxy", card["validation_status"])
        self.assertEqual("annotated_receipt_proxy", card["target_semantics"])
        self.assertEqual(33, card["n_features"])

    @unittest.skipIf(not (RELEASE_DIR / "model.json").exists(), "no exported release data")
    def test_the_payload_carries_no_tracking(self):
        scenes = [p for p in self.RELEASE_DIR.glob("*.json") if p.name != "model.json"]
        self.assertTrue(scenes)
        payload = json.loads(scenes[0].read_text())
        for banned in ("players", "ball", "xy", "t"):
            self.assertNotIn(banned, payload, banned)
        self.assertIn("frames", payload)

    @unittest.skipIf(not (RELEASE_DIR / "model.json").exists(), "no exported release data")
    def test_the_stored_payoff_is_the_product_it_claims(self):
        scenes = [p for p in self.RELEASE_DIR.glob("*.json") if p.name != "model.json"]
        payload = json.loads(scenes[0].read_text())
        columns = payload["columns"]
        legal_i, proxy_i = columns.index("legal"), columns.index("completion_proxy")
        threat_i, payoff_i = (columns.index("positional_threat"),
                              columns.index("release_payoff"))
        checked = 0
        for frame in payload["frames"]:
            for rows in frame["receivers"].values():
                for row in rows:
                    expected = (round(row[proxy_i] * row[threat_i], 5)
                                if row[legal_i] else 0.0)
                    self.assertAlmostEqual(expected, row[payoff_i], places=4)
                    checked += 1
        self.assertGreater(checked, 100)

    @unittest.skipIf(not (RELEASE_DIR / "model.json").exists(), "no exported release data")
    def test_an_illegal_target_keeps_its_proxy(self):
        """The gate must be visible: payoff zero, proxy intact."""

        columns = None
        seen = False
        for path in self.RELEASE_DIR.glob("*.json"):
            if path.name == "model.json":
                continue
            payload = json.loads(path.read_text())
            columns = columns or payload["columns"]
            legal_i = columns.index("legal")
            proxy_i = columns.index("completion_proxy")
            payoff_i = columns.index("release_payoff")
            for frame in payload["frames"]:
                for rows in frame["receivers"].values():
                    for row in rows:
                        if not row[legal_i]:
                            self.assertEqual(0.0, row[payoff_i])
                            self.assertGreater(row[proxy_i], 0.0)
                            seen = True
            if seen:
                break
        self.assertTrue(seen, "no illegal target in the export to check the gate on")

    def test_the_build_serves_the_release_payload(self):
        build_py = (REPO_ROOT / "demo_viz" / "web" / "build.py").read_text()
        self.assertIn('"release"', build_py)


if __name__ == "__main__":
    unittest.main(verbosity=2)
