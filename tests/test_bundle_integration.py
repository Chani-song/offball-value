"""The local bundle, and the line it must not cross.

The bundle carries licensed DFL tracking. These tests hold two things: that
the derived numbers reaching the public build are *exactly* upstream's, and
that nothing licensed reaches it at all.
"""

from __future__ import annotations

import json
import subprocess
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
WEB_DATA = REPO_ROOT / "demo_viz" / "web_data"
BUNDLE_ROOT = REPO_ROOT / "local_inputs" / "offball_demo_data_20260930"
S05 = "J03WOH:shot_010_P1_1759"
S05_KEY = "J03WOH_shot_010_P1_1759"


def _bundle():
    from demo_viz.paper_story.bundle import available

    if not available():
        raise unittest.SkipTest("local bundle not installed")
    from demo_viz.paper_story.bundle import BundleEvaluationSource

    return BundleEvaluationSource()


class LicenceTests(unittest.TestCase):
    """Nothing licensed may enter git or the built site."""

    def test_the_bundle_cannot_be_committed(self):
        for path in ("local_inputs/offball_demo_data_20260930/tracking/S05.csv",
                     "local_inputs/offball_demo_data_20260930/panels/eval-S05.json",
                     "local_inputs/offball_demo_data_20260930/scenes.csv"):
            out = subprocess.run(["git", "check-ignore", "-q", path],
                                 cwd=REPO_ROOT)
            self.assertEqual(0, out.returncode, f"{path} is not gitignored")

    def test_no_bundle_file_is_tracked(self):
        out = subprocess.run(["git", "ls-files", "local_inputs"],
                             cwd=REPO_ROOT, capture_output=True, text=True)
        self.assertEqual("", out.stdout.strip())

    def test_the_built_site_carries_nothing_licensed(self):
        """Section 23: walk the generated site and fail on raw data."""

        import tempfile

        from demo_viz.web.build import build

        banned_names = ("tracking", "bundesliga-integrated", ".csv", ".xml",
                        ".tar.gz", ".npz")
        banned_text = ("bundesliga-integrated", "DFL_02_01_matchinformation",
                       "offball_demo_data_20260930.tar.gz", "x_raw", "local_inputs")
        with tempfile.TemporaryDirectory() as out:
            root = Path(build(Path(out) / "site"))
            for path in root.rglob("*"):
                if not path.is_file():
                    continue
                name = str(path.relative_to(root))
                for bad in banned_names:
                    self.assertNotIn(bad, name, f"{name} looks licensed")
                if path.suffix in (".json", ".js", ".html", ".css"):
                    text = path.read_text(errors="ignore")
                    for bad in banned_text:
                        self.assertNotIn(bad, text, f"{name} mentions {bad}")

    def test_the_exported_panels_carry_no_tracking_series(self):
        """A body's start and the solver's own path only -- never a trajectory
        over the clip."""

        path = WEB_DATA / "solver" / f"{S05_KEY}.json"
        if not path.exists():
            self.skipTest("panels not exported")
        payload = json.loads(path.read_text())
        for panel in payload["panels"]:
            for body in panel["bodies"].values():
                self.assertEqual(2, len(body["pos"]))
                for option in body["options"]:
                    # the solver's 0.6 s path is 25 points; a clip is hundreds
                    self.assertLessEqual(len(option["path"]), 30)


class CommandRenameTests(unittest.TestCase):
    """The Korean -> English map is upstream's own rename, not a translation."""

    def test_it_is_exactly_the_rename_in_e89e78f(self):
        from demo_viz.paper_story.bundle import COMMAND_EN

        out = subprocess.run(
            ["git", "show", "e89e78f", "--", "src/offball_value/stage3_read.py"],
            cwd=REPO_ROOT, capture_output=True, text=True)
        if out.returncode != 0:
            self.skipTest("e89e78f not available")
        diff = out.stdout
        # every pair in the table must appear on the two sides of that diff
        for korean, english in COMMAND_EN.items():
            self.assertIn(korean, diff, f"{korean} is not in the rename commit")
            self.assertIn(english, diff, f"{english} is not in the rename commit")

    def test_every_name_in_the_bundle_is_covered(self):
        from demo_viz.paper_story.bundle import COMMAND_EN, english

        if not BUNDLE_ROOT.exists():
            self.skipTest("local bundle not installed")
        seen = set()
        for path in (BUNDLE_ROOT / "panels").glob("*.json"):
            panel = json.loads(path.read_text())
            for body in panel["bodies"].values():
                for option in body["options"]:
                    seen.add(option["name"])
        korean = {n for n in seen if any(ord(c) > 0x3000 for c in n)}
        self.assertTrue(korean, "expected legacy Korean names in the panels")
        for name in korean:
            self.assertIn(name, COMMAND_EN, f"{name} has no upstream English name")
            self.assertNotEqual(name, english(name))

    def test_the_public_export_carries_english_and_keeps_the_legacy_name(self):
        path = WEB_DATA / "solver" / f"{S05_KEY}.json"
        if not path.exists():
            self.skipTest("panels not exported")
        payload = json.loads(path.read_text())
        for panel in payload["panels"]:
            for body in panel["bodies"].values():
                for option in body["options"]:
                    self.assertFalse(any(ord(c) > 0x3000 for c in option["label"]),
                                     option["label"])
                    self.assertIn("legacy_label", option)


class S05ParityTests(unittest.TestCase):
    """Section 22: the public payload equals the bundle, value for value."""

    @classmethod
    def setUpClass(cls):
        cls.source = _bundle()
        story = WEB_DATA / "story" / f"{S05_KEY}.json"
        panels = WEB_DATA / "solver" / f"{S05_KEY}.json"
        if not (story.exists() and panels.exists()):
            raise unittest.SkipTest("S05 not exported")
        cls.story = json.loads(story.read_text())
        cls.panels = json.loads(panels.read_text())
        cls.moments = cls.source.moments(S05)

    # -- the moment the paper figure uses --------------------------------
    def test_the_three_moments_are_the_real_0_0_6_1_2(self):
        self.assertEqual([0.0, 0.6, 1.2], [m.dt for m in self.moments])
        self.assertEqual([55, 70, 85], [m.frame for m in self.moments])
        self.assertEqual([55, 70, 85], [p["frame"] for p in self.panels["panels"]])

    def test_s05_at_0_6_reproduces_the_figure_1_defender_mix(self):
        """56.8 / 43.2 -- the split render_figure1_dilemma reports as 57/43."""

        panel = next(p for p in self.panels["panels"] if p["dt"] == 0.6)
        probs = sorted((o["prob"] for o in panel["bodies"]["defender"]["options"]
                        if o["prob"] > 1e-9), reverse=True)
        self.assertEqual(2, len(probs))
        self.assertAlmostEqual(0.568, probs[0], places=3)
        self.assertAlmostEqual(0.432, probs[1], places=3)

    def test_s05_at_0_6_reproduces_the_through_pass(self):
        panel = next(p for p in self.panels["panels"] if p["dt"] == 0.6)
        self.assertEqual(1, len(panel["passes"]))
        self.assertAlmostEqual(0.74, panel["passes"][0]["prob"], places=3)
        self.assertEqual("run +8 / side +0 / goal +0", panel["passes"][0]["where"])
        self.assertEqual("runner", panel["passes"][0]["to"])

    # -- every exported number against its source ------------------------
    def test_policy_probabilities_and_paths_are_the_bundle_s(self):
        from demo_viz.paper_story.bundle import STORY_ROLE_OF, to_scene_xy

        for moment, panel in zip(self.moments, self.panels["panels"]):
            self.assertAlmostEqual(moment.panel["value"], panel["value"], places=12)
            for body in moment.panel["bodies"].values():
                role = STORY_ROLE_OF.get(body["role"])
                if role is None:
                    continue
                got = panel["bodies"][role]["options"]
                self.assertEqual(len(body["options"]), len(got))
                for want, have in zip(body["options"], got):
                    self.assertAlmostEqual(want["prob"], have["prob"], places=12)
                    self.assertEqual(to_scene_xy(want["end"]), have["end"])
                    self.assertEqual([to_scene_xy(p) for p in want["path"]],
                                     have["path"])

    def test_probability_mass(self):
        for panel in self.panels["panels"]:
            for role, body in panel["bodies"].items():
                total = sum(o["prob"] for o in body["options"])
                if role == "defender":
                    self.assertAlmostEqual(1.0, total, places=6, msg=role)
                else:
                    # an attacker's marginal plus the pass columns sums to one
                    total += sum(q["prob"] for q in panel["passes"])
                    self.assertLessEqual(total, 1.0 + 1e-6, msg=role)

    def test_dilemma_flags_are_upstream_s(self):
        for moment, panel in zip(self.moments, self.panels["panels"]):
            a = moment.analysis
            d = panel["dilemma"]
            self.assertEqual(bool(a["defender_mixed"]), d["defender_mixed"])
            self.assertEqual(bool(a["no_saddle"]), d["no_saddle"])
            self.assertAlmostEqual(a["saddle_gap"], d["saddle_gap"], places=12)
            self.assertEqual(bool(a["defender_mixed"] and a["no_saddle"]),
                             d["is_dilemma"])

    def test_static_metrics_are_upstream_s(self):
        for moment, panel in zip(self.moments, self.panels["panels"]):
            for key in ("static_attack_appeared", "static_attack_responsive",
                        "static_attack_loss", "static_attack_loss_rel"):
                self.assertAlmostEqual(moment.analysis[key], panel["static"][key],
                                       places=12, msg=key)

    def test_rank_similarity_and_regret_reach_the_story_payload(self):
        series = {s["name"]: s
                  for s in self.story["evaluation"]["runner"]["frame_series"]}
        want = {"relative_rank": [], "similarity_to_optimal": [], "regret": []}
        for moment in self.moments:
            row = next(r for r in moment.analysis["players"] if r["role"] == "runner")
            want["relative_rank"].append(row["rank_frac"])
            want["similarity_to_optimal"].append(row["similarity"])
            want["regret"].append(row["regret"])
        for name, values in want.items():
            self.assertEqual("available", series[name]["availability"], name)
            self.assertEqual([55, 70, 85], series[name]["frames"], name)
            for a, b in zip(values, series[name]["values"]):
                self.assertAlmostEqual(a, b, places=12, msg=name)

    def test_the_fit_distances_reach_the_observed_action(self):
        block = self.story["counterfactual"]["runner"]["observed_action"]
        row = next(r for r in self.moments[0].analysis["players"]
                   if r["role"] == "runner")
        self.assertEqual("available", block["availability"])
        self.assertAlmostEqual(row["fit_m"][0], block["projection_distance_m"],
                               places=12)

    def test_the_static_pair_reaches_the_story_payload(self):
        block = self.story["counterfactual"]["runner"]
        a = self.moments[0].analysis
        self.assertAlmostEqual(a["static_attack_appeared"],
                               block["static"]["value"]["value"], places=12)
        self.assertAlmostEqual(a["static_attack_responsive"],
                               block["responsive"]["value"]["value"], places=12)

    def test_no_vsra_is_claimed(self):
        """The bundle has analyze_eval's static comparison, not
        static_counterfactual's whole-window V / S / R / A."""

        text = json.dumps(self.story) + json.dumps(self.panels)
        for name in ('"V"', '"S"', '"R"', '"A"'):
            self.assertNotIn(name, text, name)

    def test_the_provenance_names_the_bundle_and_the_upstream_sha(self):
        p = self.panels["provenance"]
        self.assertEqual("offball_demo_data_20260930", p["bundle"])
        self.assertEqual("8a5c69d0bb1c42948c3f9d50becd2ffc41e72344", p["upstream"])
        self.assertEqual("offball_demo_data_20260930",
                         self.story["provenance"]["evaluation_source"])


class RoleScopeTests(unittest.TestCase):
    """The beneficiary is visible in Game solution and absent from Evaluation.

    A 3v1 game's beneficiary -- Figure 2's "Teammate" -- has a real
    equilibrium policy, so the pitch draws him rather than two thirds of the
    equilibrium. He is deliberately *not* an Evaluation role: the story schema
    is runner / passer / defender, and adding a fourth is a schema change, not
    a plumbing fix.
    """

    def test_the_story_schema_has_three_roles(self):
        from demo_viz.paper_story import STORY_ROLES

        self.assertEqual(("runner", "passer", "defender"), STORY_ROLES)

    def test_a_3v1_scene_draws_the_beneficiary_but_does_not_evaluate_him(self):
        solver = WEB_DATA / "solver" / "J03WN1_shot_002_P1_0057.json"
        story = WEB_DATA / "story" / "J03WN1_shot_002_P1_0057.json"
        if not (solver.exists() and story.exists()):
            self.skipTest("S20 not exported")
        panels = json.loads(solver.read_text())
        self.assertEqual("3v1", panels["panels"][0]["kind"])
        for panel in panels["panels"]:
            self.assertIn("beneficiary", panel["bodies"])
        evaluation = json.loads(story.read_text())["evaluation"]
        self.assertNotIn("beneficiary", evaluation)
        self.assertEqual({"runner", "passer", "defender"}, set(evaluation))

    def test_every_body_the_game_contains_is_drawn(self):
        """Three bodies per moment, in both game kinds."""

        for path in sorted((WEB_DATA / "solver").glob("*.json")):
            payload = json.loads(path.read_text())
            if payload.get("kind") != "bundle_panels":
                continue
            for panel in payload["panels"]:
                self.assertEqual(3, len(panel["bodies"]),
                                 f"{payload['code']} at {panel['dt']} s")


class LeadSceneTests(unittest.TestCase):
    """S05 opens first, decided at ingest and reproducible."""

    def test_the_lead_scene_is_named_in_the_ingest(self):
        from demo_viz.ingest_showcase import LEAD_SCENE

        self.assertEqual("S05", LEAD_SCENE)

    def test_regenerating_reproduces_the_committed_order(self):
        """The guard this replaces a hand edit with."""

        from demo_viz.ingest_showcase import OUT, build

        committed = json.loads(OUT.read_text())
        rebuilt = build()
        self.assertEqual([s["order"] for s in committed["scenes"]],
                         [s["order"] for s in rebuilt["scenes"]])

    def test_the_lead_scene_is_one_the_bundle_covers(self):
        """Opening on a scene with no solved data would be a poor first look."""

        from demo_viz.ingest_showcase import LEAD_SCENE

        source = _bundle()
        codes = {source.moments(s)[0].code for s in source.scenes()}
        self.assertIn(LEAD_SCENE, codes)
