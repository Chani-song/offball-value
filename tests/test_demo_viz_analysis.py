"""The research inspector: solver adapter, manifest, reachability, threat.

The load-bearing test here is the reachability parity check. `js/reach.js` is a
hand port of `action_space.solve_endpoint_motion`, so the two can drift; the
same cases go through both and the answers are compared, feasibility and
motion alike.
"""

from __future__ import annotations

import json
import math
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from demo_viz.core.candidates import CANDIDATE_DISTANCE_M  # noqa: E402,F401
from demo_viz.core.submission import (  # noqa: E402
    ManifestError,
    SubmissionEntry,
    load,
    validate,
)
from demo_viz.solver import (  # noqa: E402
    UNAVAILABLE,
    SolverArtifactError,
    Unavailable,
    load_for_scene,
    load_state,
)
from demo_viz.web.build import SITE, build  # noqa: E402
from demo_viz.web.export_data import WEB_DATA  # noqa: E402
from offball_value.action_space import (  # noqa: E402
    EndpointActionConfig,
    solve_endpoint_motion,
)

HERE = REPO_ROOT / "demo_viz" / "web"
SOLVER_RUN = (Path.home() / "Research" / "offball_demo" / "mit_ssac2027_ref"
              / "results" / "exact_100")
EXAMPLE_MANIFEST = REPO_ROOT / "demo_viz" / "data" / "submission_scenes.example.json"


# ---------------------------------------------------------------------------
# solver adapter
# ---------------------------------------------------------------------------
class SolverUnavailableTests(unittest.TestCase):
    """The default answer is 'no artifact', and it never becomes a picture."""

    def test_no_artifact_is_unavailable_with_a_reason(self):
        self.assertFalse(UNAVAILABLE.available)
        self.assertEqual("Not computed for this scene", UNAVAILABLE.reason)
        self.assertFalse(UNAVAILABLE.to_payload()["available"])

    def test_missing_path_does_not_fall_back(self):
        result = load_for_scene("/definitely/not/here#0")
        self.assertIsInstance(result, Unavailable)
        self.assertIn("not found", result.reason)

    def test_malformed_reference_is_refused(self):
        self.assertIsInstance(load_for_scene("somewhere#notanumber"), Unavailable)
        self.assertIsInstance(load_for_scene(""), Unavailable)
        self.assertIsInstance(load_for_scene(None), Unavailable)

    def test_a_directory_without_a_manifest_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = load_for_scene(f"{tmp}#0")
            self.assertIsInstance(result, Unavailable)
            self.assertIn("Unreadable", result.reason)

    def test_no_demo_scene_claims_solver_output(self):
        """Every exported solver file must be a declared study state, not a scene."""

        solver_dir = WEB_DATA / "solver"
        if not solver_dir.exists():
            self.skipTest("no exported solver data")
        scenes = {p.stem for p in WEB_DATA.glob("*.json")} - {"index"}
        for path in solver_dir.glob("*.json"):
            payload = json.loads(path.read_text())
            self.assertEqual("solver_reference", payload["kind"], path.name)
            self.assertNotIn(path.stem, scenes, f"{path.name} shadows a tracked scene")


@unittest.skipIf(not SOLVER_RUN.exists(), "no local solver run to read")
class SolverArtifactTests(unittest.TestCase):
    """Parsing a real artifact, with nothing invented."""

    def test_reads_a_solved_state(self):
        state = load_state(SOLVER_RUN, 0, commit="e8b0a95")
        self.assertTrue(state.available)
        self.assertEqual(0, state.index)
        self.assertGreater(state.value, 0.0)
        for body in ("carrier", "receiver", "defender"):
            self.assertIn(body, state.start)
        self.assertEqual(5, len(state.root_defender))
        self.assertEqual(len(state.directions) ** 2 + 1, len(state.root_attack))

    def test_coordinates_are_centred_on_the_pitch(self):
        """The artifact is corner-origin; the demo is centre-origin."""

        state = load_state(SOLVER_RUN, 0)
        length, width = state.pitch
        for body, (x, y) in state.start.items():
            self.assertLessEqual(abs(x), length / 2 + 1e-6, body)
            self.assertLessEqual(abs(y), width / 2 + 1e-6, body)

    def test_root_policy_decodes_to_the_solver_own_action_encoding(self):
        state = load_state(SOLVER_RUN, 0)
        actions = len(state.directions)
        attack = state.most_likely_attack()
        self.assertIn(attack["kind"], ("move", "release"))
        if attack["kind"] == "move":
            index = attack["carrier_index"] * actions + attack["receiver_index"]
            self.assertEqual(max(range(len(state.root_attack)),
                                 key=lambda i: state.root_attack[i]), index)

    def test_probabilities_are_probabilities(self):
        for index in (0, 3, 6):
            state = load_state(SOLVER_RUN, index)
            for policy in (state.root_attack, state.root_defender):
                self.assertAlmostEqual(1.0, sum(policy), places=6)
                self.assertTrue(all(p >= -1e-12 for p in policy))

    def test_mixed_and_pure_states_are_distinguished(self):
        self.assertFalse(load_state(SOLVER_RUN, 0).is_mixed)
        self.assertTrue(load_state(SOLVER_RUN, 6).is_mixed)

    def test_trajectories_carry_events_the_solver_defines(self):
        allowed = {"move", "release", "retain", "tackled_during_previous_interval"}
        state = load_state(SOLVER_RUN, 3)
        self.assertTrue(state.trajectories)
        for trajectory in state.trajectories:
            self.assertLessEqual(set(trajectory.events), allowed)
            for body, path in trajectory.paths.items():
                self.assertEqual(len(trajectory.times), len(path), body)

    def test_provenance_points_back_at_the_run(self):
        state = load_state(SOLVER_RUN, 0, commit="e8b0a95")
        provenance = state.provenance
        self.assertEqual("e8b0a95", provenance.commit)
        self.assertEqual(0, provenance.state_index)
        self.assertTrue(provenance.fingerprint, "policy fingerprint missing")
        self.assertTrue(provenance.created_utc)

    def test_a_pass_rollout_keeps_its_target(self):
        state = load_state(SOLVER_RUN, 3)
        releases = [t for t in state.trajectories if t.terminal_event == "release"]
        self.assertTrue(releases, "state 3 should contain a release")
        self.assertIsNotNone(releases[0].release_target)
        self.assertIn(releases[0].release_family, ("ground", "driven", "lofted"))


# ---------------------------------------------------------------------------
# submission manifest
# ---------------------------------------------------------------------------
class SubmissionManifestTests(unittest.TestCase):
    def test_absent_manifest_falls_back_to_the_explorer(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(load(Path(tmp) / "nothing.json"))

    def test_the_example_validates(self):
        entries = validate(json.loads(EXAMPLE_MANIFEST.read_text()))
        self.assertGreaterEqual(len(entries), 2)
        self.assertEqual([1, 2], [e.order for e in entries][:2])
        self.assertIsInstance(entries[0], SubmissionEntry)

    def test_the_example_does_not_pretend_to_be_the_final_ten(self):
        raw = json.loads(EXAMPLE_MANIFEST.read_text())
        self.assertIn("_comment", raw)
        self.assertNotEqual(10, len(raw["scenes"]),
                            "an example with ten entries reads as the real list")

    def test_the_example_carries_no_local_path(self):
        text = EXAMPLE_MANIFEST.read_text()
        self.assertNotIn("/Users/", text)
        self.assertNotIn("/home/", text)

    def test_entries_are_sorted_by_order(self):
        entries = validate([
            {"scene_id": "b", "order": 2},
            {"scene_id": "a", "order": 1},
        ])
        self.assertEqual(["a", "b"], [e.scene_id for e in entries])

    def test_duplicate_order_is_refused(self):
        with self.assertRaises(ManifestError):
            validate([{"scene_id": "a", "order": 1}, {"scene_id": "b", "order": 1}])

    def test_duplicate_scene_id_is_refused(self):
        with self.assertRaises(ManifestError):
            validate([{"scene_id": "a", "order": 1}, {"scene_id": "a", "order": 2}])

    def test_a_solver_reference_needs_an_artifact(self):
        with self.assertRaises(ManifestError):
            validate([{"scene_id": "a", "order": 1, "kind": "solver_reference"}])

    def test_unknown_kind_is_refused(self):
        with self.assertRaises(ManifestError):
            validate([{"scene_id": "a", "order": 1, "kind": "guess"}])

    def test_bad_role_shape_is_refused(self):
        with self.assertRaises(ManifestError):
            validate([{"scene_id": "a", "order": 1,
                       "annotation": {"runners": [7]}}])
        with self.assertRaises(ManifestError):
            validate([{"scene_id": "a", "order": 1,
                       "annotation": {"strikers": ["7"]}}])

    def test_order_must_be_a_positive_integer(self):
        for bad in (0, -1, "1", True):
            with self.assertRaises(ManifestError):
                validate([{"scene_id": "a", "order": bad}])


# ---------------------------------------------------------------------------
# reachability
# ---------------------------------------------------------------------------
def _reach_cases(seed: int = 7, count: int = 60) -> list[dict]:
    rng = random.Random(seed)
    cases = []
    for i in range(count):
        start = [rng.uniform(-45, 45), rng.uniform(-28, 28)]
        # most of the pitch is unreachable in 2 s, so aim near the player: the
        # point is to exercise the feasible branch, not to rediscover the range
        angle = rng.uniform(0, 2 * math.pi)
        radius = rng.uniform(0, 14)
        cases.append({
            "key": f"case{i}",
            "start": start,
            "velocity": [rng.uniform(-8, 8), rng.uniform(-6, 6)],
            "endpoint": [start[0] + math.cos(angle) * radius,
                         start[1] + math.sin(angle) * radius],
        })
    # deliberate edge cases: outside the pitch, at rest, already at the endpoint
    cases += [
        {"key": "outside", "start": [0, 0], "velocity": [0, 0], "endpoint": [60, 0]},
        {"key": "at_rest", "start": [0, 0], "velocity": [0, 0], "endpoint": [3, 0]},
        {"key": "same_point", "start": [5, 5], "velocity": [0, 0], "endpoint": [5, 5]},
        {"key": "too_far", "start": [0, 0], "velocity": [0, 0], "endpoint": [40, 0]},
    ]
    return cases


class ReachPythonTests(unittest.TestCase):
    def test_the_config_is_the_one_the_browser_states(self):
        config = EndpointActionConfig()
        source = (SITE / "js" / "reach.js").read_text()
        self.assertIn(f"horizonSeconds: {config.horizon_seconds}", source)
        self.assertIn(f"maxSpeed: {config.max_speed_mps}", source)
        self.assertIn(f"maxAcceleration: {config.max_acceleration_mps2}", source)
        self.assertIn(f"gridResolutionM: {config.grid_resolution_m}", source)

    def test_an_endpoint_off_the_pitch_is_never_reachable(self):
        motion = solve_endpoint_motion((0, 0), (0, 0), (60, 0))
        self.assertFalse(motion.feasible)
        self.assertEqual("endpoint_outside_pitch", motion.failure_reason)

    def test_the_layer_is_off_by_default(self):
        markup = (SITE / "index.html").read_text()
        for layer in ("reach", "solver"):
            import re

            match = re.search(rf'data-layer="{layer}"([^>]*)>', markup)
            self.assertIsNotNone(match, f"{layer} layer missing")
            self.assertNotIn("checked", match.group(1))


class ReachParityTests(unittest.TestCase):
    """js/reach.js must answer exactly what action_space does."""

    def test_browser_matches_python_endpoint_by_endpoint(self):
        from demo_viz.web.validate import find_chrome, serve

        chrome = find_chrome()
        if chrome is None:
            self.skipTest("no Chrome/Chromium available")
        cases = _reach_cases()
        plan = {"cases": cases, "masks": [
            {"key": "mask0", "start": [0, 0], "velocity": [0, 0]},
            {"key": "mask1", "start": [10, -5], "velocity": [6, 2]},
        ]}

        with tempfile.TemporaryDirectory() as tmp:
            root = build(Path(tmp) / "site", data_dir=WEB_DATA)
            page = (HERE / "analysis_harness.html").read_text().replace(
                "<body>",
                '<body><script type="application/json" id="plan">'
                + json.dumps(plan).replace("</", "<\\/") + "</script>",
            )
            (root / "analysis_harness.html").write_text(page)
            with serve(root) as port:
                result = subprocess.run(
                    [chrome, "--headless", "--disable-gpu", "--no-sandbox",
                     "--virtual-time-budget=60000", "--dump-dom",
                     f"http://127.0.0.1:{port}/analysis_harness.html"],
                    capture_output=True, text=True, timeout=180,
                )
        start = result.stdout.find('<pre id="out">')
        self.assertGreaterEqual(start, 0, "harness did not render")
        start = result.stdout.index(">", start) + 1
        text = result.stdout[start:result.stdout.index("</pre>", start)].strip()
        text = text.replace("&quot;", '"').replace("&amp;", "&").replace("&lt;", "<")
        self.assertNotEqual("running", text, "harness timed out")
        report = json.loads(text)

        by_key = {case["key"]: case for case in cases}
        disagreements = []
        feasible_count = 0
        for entry in report["cases"]:
            case = by_key[entry["key"]]
            expected = solve_endpoint_motion(
                tuple(case["start"]), tuple(case["velocity"]), tuple(case["endpoint"]))
            if expected.feasible != entry["feasible"]:
                disagreements.append(
                    f"{entry['key']}: python={expected.feasible} browser={entry['feasible']}")
                continue
            if not expected.feasible:
                continue
            feasible_count += 1
            for name, mine, theirs in (
                ("acceleration", entry["acceleration"], expected.acceleration_mps2),
                ("duration", entry["accelerationDuration"], expected.acceleration_duration_s),
                ("terminal speed", entry["terminalSpeed"], expected.terminal_speed_mps),
            ):
                if not math.isclose(mine, theirs, rel_tol=1e-9, abs_tol=1e-9):
                    disagreements.append(f"{entry['key']} {name}: {mine} vs {theirs}")
        self.assertEqual([], disagreements)
        self.assertGreater(feasible_count, 9,
                           "the cases never exercised a feasible path")

    def test_the_mask_area_is_sane(self):
        """A player at rest reaches a disc; one at speed reaches a shifted one."""

        config = EndpointActionConfig()
        reachable = 0
        for ix in range(-52, 53):
            for iy in range(-33, 34):
                if solve_endpoint_motion((0.0, 0.0), (0.0, 0.0),
                                         (float(ix), float(iy)), config).feasible:
                    reachable += 1
        # 0.5 * a * t^2 = 7 m at rest, so a disc of radius 7
        self.assertLess(abs(reachable - math.pi * 7 * 7), 40, reachable)


class TimelineMarkTests(unittest.TestCase):
    """Marks on the scrubber come from data, not from presentation pacing."""

    def test_only_defined_moments_are_marked(self):
        source = (SITE / "js" / "app.js").read_text()
        marks = source[source.index("function renderTicks"):]
        marks = marks[:marks.index("\n}\n")]
        # the run onset and the annotated shot are exported per scene
        self.assertIn("scene.onsets", marks)
        self.assertIn("-scene.t0 * scene.fps", marks)
        # nothing the exported scene does not define may be marked
        for invented in ("reacts", "defender_reacts", "space opens", "beneficiary"):
            self.assertNotIn(invented, marks.lower())

    def test_the_shot_mark_is_where_the_clip_clock_reads_zero(self):
        index_file = WEB_DATA / "index.json"
        if not index_file.exists():
            self.skipTest("no exported scene data")
        scenes = json.loads(index_file.read_text())["scenes"]
        payload = json.loads((WEB_DATA / scenes[0]["file"]).read_text())
        shot = round(-payload["t0"] * payload["fps"])
        self.assertGreaterEqual(shot, 0)
        self.assertLess(shot, payload["n_frames"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
