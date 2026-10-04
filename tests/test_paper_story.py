"""The paper-story contract, and what it refuses to let the demo show.

The contract exists to make one class of mistake impossible: a number on
screen that no research code produced. These tests hold that line from both
ends -- the dataclasses reject a record that would carry a value without a
source, and a sweep over every exported payload checks that none slipped
through by another route.

They also hold the *integration* promise. `test_a_future_source_populates_the`
`_same_payload` builds a scene from a stub evaluation source and asserts the
values arrive with provenance attached, through the same code path the real
one will use. If that test needs the interface changed to pass, the adapter
boundary has leaked.
"""

from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

from demo_viz.paper_story import (
    AVAILABILITY,
    AVAILABILITY_LABEL,
    STORY_ROLES,
    ActionRef,
    ContractError,
    EVALUATION_METRICS,
    Metric,
    NoEvaluationSource,
    Series,
    build_scene,
    contract_payload,
    equilibrium_for,
    measured,
    pending,
    validate_payload,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
WEB_DATA = REPO_ROOT / "demo_viz" / "web_data"
STORY_DIR = WEB_DATA / "story"
SITE = REPO_ROOT / "demo_viz" / "web" / "site"


class MetricInvariantTests(unittest.TestCase):
    """A value implies a source, and no source implies no value."""

    def test_an_available_metric_must_carry_a_value(self):
        with self.assertRaises(ContractError):
            Metric(name="regret", label="Regret", availability="available",
                   source="somewhere")

    def test_an_available_metric_must_name_a_source(self):
        with self.assertRaises(ContractError):
            Metric(name="regret", label="Regret", availability="available", value=0.4)

    def test_an_unavailable_metric_may_not_carry_a_number(self):
        """The rule that stops a placeholder becoming a result."""

        for availability in AVAILABILITY:
            if availability == "available":
                continue
            with self.assertRaises(ContractError, msg=availability):
                Metric(name="relative_rank", label="Relative rank",
                       availability=availability, value=0)

    def test_zero_is_a_value_like_any_other(self):
        metric = measured("regret", "Regret", 0.0, source="stub")
        self.assertEqual(0.0, metric.value)
        self.assertEqual("available", metric.availability)

    def test_an_unknown_availability_is_rejected(self):
        with self.assertRaises(ContractError):
            Metric(name="x", label="X", availability="maybe")

    def test_an_unknown_provenance_is_rejected(self):
        with self.assertRaises(ContractError):
            Metric(name="x", label="X", provenance="vibes")

    def test_every_availability_state_has_its_own_sentence(self):
        """Five reasons stay five reasons; none collapses to a dash."""

        for state in AVAILABILITY:
            self.assertIn(state, AVAILABILITY_LABEL)
        messages = [AVAILABILITY_LABEL[s] for s in AVAILABILITY if s != "available"]
        self.assertEqual(len(messages), len(set(messages)))
        for message in messages:
            self.assertGreater(len(message), 12, message)
            self.assertNotIn("N/A", message)

    def test_a_pending_metric_reports_why(self):
        metric = pending("relative_rank", "Relative rank",
                         detail="No relative-rank definition exists yet.")
        self.assertIsNone(metric.value)
        self.assertIn("definition", metric.message)


class SeriesInvariantTests(unittest.TestCase):

    def test_frames_and_values_must_be_the_same_length(self):
        with self.assertRaises(ContractError):
            Series(name="regret", label="Regret", availability="available",
                   source="stub", frames=(1, 2, 3), values=(0.1, 0.2))

    def test_an_available_series_needs_samples(self):
        with self.assertRaises(ContractError):
            Series(name="regret", label="Regret", availability="available",
                   source="stub")

    def test_an_unavailable_series_carries_no_samples(self):
        with self.assertRaises(ContractError):
            Series(name="regret", label="Regret", frames=(1,), values=(0.5,))

    def test_frames_must_be_sorted_and_unique(self):
        """The timeline maps frames to x directly; an unsorted series would
        draw a sample at the wrong instant rather than fail."""

        with self.assertRaises(ContractError):
            Series(name="regret", label="Regret", availability="available",
                   source="stub", frames=(9, 3), values=(0.1, 0.2))
        with self.assertRaises(ContractError):
            Series(name="regret", label="Regret", availability="available",
                   source="stub", frames=(3, 3), values=(0.1, 0.2))

    def test_joining_samples_is_off_unless_the_producer_allows_it(self):
        series = Series(name="regret", label="Regret", availability="available",
                        source="stub", frames=(0, 40), values=(0.1, 0.9))
        self.assertFalse(series.interpolate,
                         "a line between two computed points asserts values "
                         "nobody computed")


class ActionInvariantTests(unittest.TestCase):

    def test_an_unavailable_action_may_not_carry_a_target(self):
        with self.assertRaises(ContractError):
            ActionRef(name="observed_action", label="Observed action",
                      target=(1.0, 2.0))

    def test_an_available_action_must_describe_something(self):
        with self.assertRaises(ContractError):
            ActionRef(name="observed_action", label="Observed action",
                      availability="available", source="stub")

    def test_an_action_id_survives_a_round_trip(self):
        action = ActionRef(name="observed_action", label="Observed action",
                           availability="available", source="stub",
                           action_id="move_E_12", kind="move",
                           projection_distance_m=0.8, confidence=0.91)
        again = ActionRef.from_payload(action.to_payload())
        self.assertEqual("move_E_12", again.action_id)
        self.assertEqual(0.8, again.projection_distance_m)
        self.assertEqual(0.91, again.confidence)


class PayloadValidationTests(unittest.TestCase):

    def test_todays_scene_validates_with_every_future_field_absent(self):
        payload = build_scene("J03WOY:shot_011_P2_0903").to_payload()
        self.assertEqual(payload, validate_payload(payload))

    def test_a_foreign_schema_version_is_refused(self):
        payload = build_scene("x").to_payload()
        payload["schema"] = "paper-story/99"
        with self.assertRaises(ContractError):
            validate_payload(payload)

    def test_a_smuggled_value_is_caught_on_the_way_in(self):
        """Validation runs over data this process did not build."""

        payload = build_scene("x").to_payload()
        payload["evaluation"]["runner"]["metrics"][1]["value"] = 0.87
        with self.assertRaises(ContractError):
            validate_payload(payload)

    def test_the_story_roles_are_the_paper_s_three(self):
        payload = build_scene("x").to_payload()
        self.assertEqual(list(STORY_ROLES), payload["roles"])
        self.assertEqual(["runner", "passer", "defender"], payload["roles"])
        self.assertNotIn("beneficiary", payload["roles"],
                         "the beneficiary is off-ball context, not a story role")

    def test_the_contract_ships_its_own_vocabulary(self):
        contract = contract_payload()
        self.assertEqual(list(AVAILABILITY), contract["availability"])
        self.assertEqual(dict(AVAILABILITY_LABEL), contract["availability_label"])


class TodayIsHonestTests(unittest.TestCase):
    """What the adapter says while the evaluation pipeline does not exist."""

    def setUp(self):
        self.payload = build_scene("J03WOY:shot_011_P2_0903",
                                   source=NoEvaluationSource()).to_payload()

    def test_every_evaluation_metric_is_an_explicit_gap(self):
        for role in STORY_ROLES:
            for metric in self.payload["evaluation"][role]["metrics"]:
                self.assertNotEqual("available", metric["availability"], metric)
                self.assertIsNone(metric["value"])
                self.assertTrue(metric["detail"] or
                                AVAILABILITY_LABEL[metric["availability"]])

    def test_the_reserved_names_are_the_abstract_s(self):
        names = [m["name"] for m in self.payload["evaluation"]["runner"]["metrics"]]
        self.assertEqual([spec.name for spec in EVALUATION_METRICS], names)
        for wanted in ("relative_rank", "similarity_to_optimal", "regret"):
            self.assertIn(wanted, names)

    def test_the_regret_slot_carries_the_upstream_definition(self):
        """Replaces a test from when no regret existed. analyze_eval defines
        it (2026-09-30), so the slot states the definition and says the
        *output* is what is missing."""

        regret = next(m for m in self.payload["evaluation"]["runner"]["metrics"]
                      if m["name"] == "regret")
        self.assertEqual("artifact_missing", regret["availability"])
        self.assertIn("best option", regret["detail"])
        self.assertIn("observed minus best", regret["detail"])

    def test_the_static_side_carries_the_upstream_semantics(self):
        """Replaces a test from when no static/responsive pair existed on one
        scale. static_counterfactual.py defines it; only the run is missing."""

        static = self.payload["counterfactual"]["runner"]["static"]
        responsive = self.payload["counterfactual"]["runner"]["responsive"]
        self.assertIn("held to his real commands at every turn", static["semantics"])
        self.assertIn("equilibrium", responsive["semantics"])
        self.assertEqual("artifact_missing", static["value"]["availability"])

    def test_the_evaluation_fields_are_awaiting_a_run_not_a_method(self):
        for role in STORY_ROLES:
            for metric in self.payload["evaluation"][role]["metrics"]:
                self.assertEqual("artifact_missing", metric["availability"],
                                 metric["name"])
                self.assertIn("Awaiting an evaluation run", metric["detail"])

    def test_clip_aggregation_is_not_assumed(self):
        for role in STORY_ROLES:
            for metric in self.payload["clip_summary"][role]:
                self.assertIsNone(metric["value"])
                self.assertIsNone(metric["aggregation"])

    def test_the_source_row_names_the_missing_pipeline(self):
        self.assertEqual("none", self.payload["provenance"]["evaluation_source"])


class FutureIntegrationTests(unittest.TestCase):
    """Proving the promise: new outputs arrive without an interface change.

    The stub below is what Kyuhyeok's module will be. It lives here, in a test,
    and a separate test asserts nothing like it can reach a public payload.
    """

    class StubSource:
        name = "stub_pipeline@abc1234"
        definition_version = "v0.3"

        def counterfactual(self, scene_id, role, frame):
            return {
                "observed_action": {"action_id": "move_NE", "kind": "move",
                                    "description": "drive into the half-space",
                                    "projection_distance_m": 0.6},
                "feasible_actions": {"value": 26, "unit": "actions"},
                "static": {"semantics": "defenders held at their onset positions",
                           "best_action": {"action_id": "move_N",
                                           "description": "run in behind"},
                           "value": {"value": 0.71}},
                "responsive": {"semantics": "the marking defender best-responds",
                               "best_action": {"action_id": "move_E",
                                               "description": "check to the ball"},
                               "value": {"value": 0.42}},
                "value_change": {"value": -0.29},
            }

        def evaluation(self, scene_id, role, frame):
            return {"relative_rank": {"value": 0.83},
                    "similarity_to_optimal": {"value": 0.62},
                    "regret": {"value": 0.19, "unit": "value"},
                    "optimal_action": {"action_id": "move_N",
                                       "description": "run in behind"}}

        def frame_series(self, scene_id, role):
            return {"regret": {"frames": [0, 25, 50], "values": [0.1, 0.4, 0.2],
                               "domain": [0.0, 1.0]}}

        def clip_metrics(self, scene_id, role):
            return [{"name": "mean_regret", "label": "Mean regret", "value": 0.23,
                     "aggregation": "mean over evaluated frames"}]

    def setUp(self):
        self.payload = build_scene("J03WOY:shot_011_P2_0903",
                                   source=self.StubSource()).to_payload()
        validate_payload(self.payload)

    def test_values_arrive_with_a_source_and_a_definition_version(self):
        metric = next(m for m in self.payload["evaluation"]["runner"]["metrics"]
                      if m["name"] == "relative_rank")
        self.assertEqual("available", metric["availability"])
        self.assertEqual(0.83, metric["value"])
        self.assertEqual("stub_pipeline@abc1234", metric["source"])
        self.assertEqual("v0.3", metric["definition_version"])
        self.assertEqual("evaluation_pipeline", metric["provenance"])

    def test_the_static_and_responsive_semantics_are_the_producer_s_words(self):
        block = self.payload["counterfactual"]["runner"]
        self.assertEqual("defenders held at their onset positions",
                         block["static"]["semantics"])
        self.assertEqual(0.71, block["static"]["value"]["value"])
        self.assertEqual(0.42, block["responsive"]["value"]["value"])
        self.assertEqual(-0.29, block["value_change"]["value"])

    def test_a_projected_observed_action_carries_how_well_it_matched(self):
        observed = self.payload["counterfactual"]["runner"]["observed_action"]
        self.assertEqual("move_NE", observed["action_id"])
        self.assertEqual(0.6, observed["projection_distance_m"])

    def test_a_frame_series_arrives_with_its_frames(self):
        series = {s["name"]: s for s in
                  self.payload["evaluation"]["runner"]["frame_series"]}
        self.assertEqual("available", series["regret"]["availability"])
        self.assertEqual([0, 25, 50], series["regret"]["frames"])
        # the two series the stub did not produce stay explicit gaps
        self.assertNotEqual("available", series["relative_rank"]["availability"])

    def test_a_clip_metric_carries_its_own_aggregation(self):
        metric = self.payload["clip_summary"]["runner"][0]
        self.assertEqual("mean_regret", metric["name"])
        self.assertEqual("mean over evaluated frames", metric["aggregation"])

    def test_the_interface_needs_no_allow_list_for_a_new_metric(self):
        """`mean_regret` is a name no interface file mentions, and it renders."""

        app = (SITE / "js" / "app.js").read_text()
        story = (SITE / "js" / "story.js").read_text()
        self.assertNotIn("mean_regret", app + story)
        self.assertIn("contractRows", app)


class ExportedPayloadTests(unittest.TestCase):
    """A sweep over what this build would actually publish."""

    @classmethod
    def setUpClass(cls):
        if not STORY_DIR.exists():
            raise unittest.SkipTest("no story payloads exported")
        cls.files = sorted(p for p in STORY_DIR.glob("*.json")
                           if p.name != "contract.json")

    def _records(self, payload):
        for role, block in (payload.get("counterfactual") or {}).items():
            yield block["observed_action"]
            yield block["feasible_actions"]
            for side in ("static", "responsive"):
                part = block.get(side) or {}
                for key in ("best_action", "value"):
                    if part.get(key):
                        yield part[key]
            if block.get("value_change"):
                yield block["value_change"]
        for role, block in (payload.get("evaluation") or {}).items():
            yield from block.get("metrics", ())
            if block.get("optimal_action"):
                yield block["optimal_action"]
            yield from block.get("frame_series", ())
        for metrics in (payload.get("clip_summary") or {}).values():
            yield from metrics

    def test_every_exported_payload_validates(self):
        for path in self.files:
            with self.subTest(path.name):
                validate_payload(json.loads(path.read_text()))

    def test_no_exported_value_exists_without_a_source(self):
        for path in self.files:
            payload = json.loads(path.read_text())
            for record in self._records(payload):
                if record.get("availability") == "available":
                    self.assertTrue(record.get("source"),
                                    f"{path.name}: {record['name']} has no source")

    def test_no_exported_gap_carries_a_number(self):
        for path in self.files:
            payload = json.loads(path.read_text())
            for record in self._records(payload):
                if record.get("availability") != "available":
                    self.assertIsNone(record.get("value"),
                                      f"{path.name}: {record['name']}")
                    self.assertFalse(record.get("frames"),
                                     f"{path.name}: {record['name']}")

    def test_the_evaluation_source_is_the_bundle_or_nothing(self):
        """Replaces a test that required `none`. The local bundle is now a
        real source; what must never appear is a stub or a fixture."""

        from demo_viz.paper_story.bundle import BUNDLE, EXTRA_BUNDLES

        allowed = {"none", BUNDLE, *EXTRA_BUNDLES}
        for path in self.files:
            payload = json.loads(path.read_text())
            source = payload["provenance"]["evaluation_source"]
            self.assertIn(source, allowed, f"{path.name}: {source}")
            for record in self._records(payload):
                text = str(record.get("source") or "").lower()
                self.assertNotIn("stub", text)
                self.assertNotIn("fixture", text)

    def test_the_abstract_s_placeholders_never_reach_the_payload(self):
        """Section 25: no XX, no TBD, no example rank."""

        for path in self.files:
            text = path.read_text()
            for banned in ("XX%", "XX ", "TBD", "placeholder", "example value"):
                self.assertNotIn(banned, text, f"{path.name}: {banned}")

    def test_equilibrium_is_attached_only_where_a_file_exists(self):
        """Nothing is matched by name similarity."""

        for path in self.files:
            payload = json.loads(path.read_text())
            block = payload["equilibrium"]
            if block["availability"] == "available":
                self.assertTrue((WEB_DATA / "solver" / f"{block['key']}.json").exists())
            else:
                self.assertIsNone(block["key"])
                self.assertTrue(block["detail"])

    def test_an_equilibrium_claim_is_backed_by_an_exported_file(self):
        """Replaces "no tracked scene claims an equilibrium", which was true
        until the 2026-09-30 bundle arrived. Twenty scenes claim one now
        (twelve from that bundle, six from the 2026-10-03 batch, two from the
        2026-10-04 one), and each must have the file to show for it."""

        claimed = []
        for path in self.files:
            block = json.loads(path.read_text())["equilibrium"]
            if block["availability"] != "available":
                continue
            claimed.append(path.stem)
            self.assertTrue((WEB_DATA / "solver" / f"{block['key']}.json").exists(),
                            f"{path.stem} claims {block['key']} with no file")
        self.assertGreaterEqual(len(claimed), 1,
                                "the bundle covers seven published scenes")


class EquilibriumArtifactTests(unittest.TestCase):
    """The half of the abstract that is real, checked against real output."""

    @classmethod
    def setUpClass(cls):
        # two shapes live here now: the legacy exact-study states and the
        # bundle's solved moments. This class is about the legacy shape.
        cls.files = [p for p in sorted((WEB_DATA / "solver").glob("*.json"))
                     if json.loads(p.read_text()).get("kind") != "bundle_panels"]
        if not cls.files:
            raise unittest.SkipTest("no legacy solver artifacts exported")

    def test_root_policies_are_probability_distributions(self):
        for path in self.files:
            state = json.loads(path.read_text())
            if not state.get("available"):
                continue
            for key in ("root_attack", "root_defender"):
                total = sum(state[key])
                self.assertAlmostEqual(1.0, total, places=6,
                                       msg=f"{path.name}:{key} sums to {total}")
                self.assertTrue(all(p >= -1e-12 for p in state[key]), key)

    def test_the_attack_policy_is_the_moves_squared_plus_release(self):
        """The index arithmetic policy.js decodes, checked against the data."""

        for path in self.files:
            state = json.loads(path.read_text())
            if not state.get("available"):
                continue
            n = len(state["directions"])
            self.assertEqual(n * n + 1, len(state["root_attack"]), path.name)
            self.assertEqual(n, len(state["root_defender"]), path.name)
            self.assertAlmostEqual(state["root_attack"][-1],
                                   state["release_probability"], places=12)

    def test_a_reference_state_is_never_dressed_as_a_match_scene(self):
        for path in self.files:
            state = json.loads(path.read_text())
            if state.get("kind") == "solver_reference":
                self.assertIn("not a tracked match scene", state["caveat"])

    def test_the_adapter_finds_no_equilibrium_for_a_scene_without_one(self):
        # S30: curated, not solved (S02 until 2026-10-03, then S06 until 2026-10-04)
        block = equilibrium_for("J03WN1:shot_017_P2_1003", WEB_DATA / "solver")
        self.assertEqual("artifact_missing", block.availability)
        self.assertIn("batch job", block.detail)


class BoundaryTests(unittest.TestCase):
    """The browser must not learn the research code's file layout."""

    @staticmethod
    def _code(text):
        """The JavaScript with line comments removed.

        Prose may name an upstream file -- story.js says in so many words that
        it does not read one. What must not appear is a reference in code.
        """

        out = []
        for line in text.splitlines():
            head, marker, _ = line.partition("//")
            out.append(head if marker and "http" not in head else line)
        return "\n".join(out)

    def test_the_site_never_names_an_upstream_artifact_layout(self):
        text = "".join(self._code(p.read_text()) for p in (SITE / "js").glob("*.js"))
        text += (SITE / "index.html").read_text()
        for leaked in ("rows.jsonl", "state_00", "starting_states",
                       "policies/", "exact_100"):
            self.assertNotIn(leaked, text, leaked)

    def test_the_adapter_is_the_only_place_that_reads_a_run_directory(self):
        story = (SITE / "js" / "story.js").read_text()
        self.assertIn("data/story", story)
        self.assertNotIn("manifest", story)


class StoryUiTests(unittest.TestCase):
    """The interface, held to the same rule as the data."""

    @classmethod
    def setUpClass(cls):
        cls.markup = (SITE / "index.html").read_text()
        cls.app = (SITE / "js" / "app.js").read_text()
        cls.story = (SITE / "js" / "story.js").read_text()
        cls.strip = (SITE / "js" / "evalstrip.js").read_text()
        cls.policy = (SITE / "js" / "policy.js").read_text()

    # -- navigation ------------------------------------------------------
    def test_the_four_modes_are_the_primary_control(self):
        for mode in ("observed", "counterfactual", "evaluation", "game_solution"):
            self.assertIn(f'data-mode="{mode}"', self.markup, mode)
        # each one states the question it answers, not just its name
        for question in ("What happened?", "What else could the player have done?",
                         "How good was the observed decision?",
                         "What does the strategic equilibrium recommend?"):
            self.assertIn(question, self.markup, question)

    def test_the_mode_bar_sits_above_the_layer_switches(self):
        self.assertLess(self.markup.index('id="story-modes"'),
                        self.markup.index('class="layers"'),
                        "the paper's four questions come before the checkboxes")

    def _mode_sections(self, mode):
        block = self.app[self.app.index("const MODE_SECTIONS = {"):]
        block = block[:block.index("\n};")]
        line = block[block.index(f"{mode}: ["):]
        return line[:line.index("]")]

    def test_observed_mode_shows_only_what_exists_today(self):
        sections = self._mode_sections("observed")
        for absent in ("an-counterfactual", "an-evaluation", "an-clip"):
            self.assertNotIn(absent, sections, absent)
        for present in ("an-scene", "an-player", "an-offball"):
            self.assertIn(present, sections, present)

    def test_counterfactual_mode_exists_without_evaluation_outputs(self):
        """Section 6: the mode is not hidden because its values are missing."""

        self.assertIn("an-counterfactual", self._mode_sections("counterfactual"))
        self.assertNotIn('id="an-counterfactual"', self.app,
                         "visibility is the mode's decision, not a data check")

    def test_the_release_drill_down_is_secondary_to_the_story(self):
        """Section 16: the pass model is available, never prominent."""

        self.assertLess(self.markup.index('id="an-counterfactual"'),
                        self.markup.index('id="an-pass"'))
        self.assertNotIn("an-pass", self._mode_sections("observed"))
        self.assertNotIn("an-pass", self._mode_sections("evaluation"))

    # -- pending states --------------------------------------------------
    def test_a_missing_value_renders_a_sentence_and_a_pending_style(self):
        block = self.app[self.app.index("function contractRows"):]
        block = block[:block.index("\nfunction ")]
        self.assertIn('dd.className = "pending"', block)
        # the reason comes from the record, never from a local fallback
        for banned in ('"N/A"', '"--"', '"—"', '"0"', "'TBD'"):
            self.assertNotIn(banned, block, banned)

    def test_the_evaluation_empty_state_explains_itself(self):
        collapsed = " ".join(self.app.split())
        self.assertIn("Evaluation outputs are not available for this scene yet",
                      collapsed)
        self.assertIn("populates from the player-evaluation pipeline", collapsed)
        self.assertIn("substituted in the meantime", collapsed)

    def test_the_frame_strip_empty_state_names_what_is_coming(self):
        self.assertIn("Frame-level evaluation will appear here when evaluation",
                      self.app)

    def test_nothing_in_the_strip_can_generate_a_curve(self):
        """Section 9 and 32: no fake example series, ever."""

        for banned in ("Math.random", "sampleValues", "synthetic", "demoSeries"):
            self.assertNotIn(banned, self.strip, banned)
        self.assertNotIn("Math.random", self.app)

    def test_samples_are_not_joined_unless_the_payload_allows_it(self):
        self.assertIn("line.interpolate", self.strip)
        self.assertIn("evalstem", self.strip,
                      "sparse samples draw as stems, not as a line")

    def test_the_browser_never_spells_an_availability_reason_itself(self):
        """Section 23: the five states live in the contract, in one place."""

        self.assertIn("contract.availability_label", self.story)
        for banned in ("method_not_implemented:", "Awaiting evaluation output"):
            self.assertNotIn(banned, self.app, banned)

    # -- the equilibrium half --------------------------------------------
    def test_the_game_solution_renders_only_with_a_real_artifact(self):
        block = self.app[self.app.index("function renderSolverRows"):]
        block = block[:block.index("\n// ---")]
        self.assertIn("if (!data || !data.available)", block)
        self.assertIn("policy.hidden = true", block)

    def test_the_modal_line_is_labelled_an_illustration(self):
        self.assertIn("Modal policy illustration", self.policy)
        self.assertIn("not a sampled trajectory", self.policy)
        for banned in ("predicted trajectory", "optimal trajectory",
                       "simulated player path", "simulated trajectory"):
            self.assertNotIn(banned, self.policy + self.app, banned)

    def test_mixed_policy_never_depends_on_colour_alone(self):
        block = self.policy[self.policy.index("export function drawPolicy"):]
        self.assertIn("polname", block)
        self.assertIn("polvalue", block)
        self.assertIn("toFixed(0)}%", block)

    def test_a_truncated_policy_reports_its_remainder(self):
        """A 26-entry policy is cut for readability, not silently."""

        block = self.policy[self.policy.index("export function drawPolicy"):]
        self.assertIn("further action", block)

    def test_certified_quantities_keep_the_solver_s_own_names(self):
        block = self.app[self.app.index("function renderSolverRows"):]
        block = block[:block.index("\n// ---")]
        self.assertIn('"Certificate gap"', block)
        self.assertIn('"Equilibrium value"', block)

    # -- what must not be there ------------------------------------------
    def test_the_legacy_obso_stack_stays_out_of_the_public_story(self):
        """Section 37: the implementation stays, the public payload does not."""

        import re
        import tempfile

        from demo_viz.web.build import build

        # nothing in the interface can ask for the threat view
        self.assertEqual(["space", "gain"],
                         re.findall(r'<option value="([a-z]+)"', self.markup))
        self.assertNotIn("an-obso", self.markup)
        for mode in ("observed", "counterfactual", "evaluation", "game_solution"):
            self.assertNotIn("obso", self._mode_sections(mode), mode)
        # and the default build ships none of the surfaces
        with tempfile.TemporaryDirectory() as out:
            root = Path(build(Path(out) / "site"))
            self.assertFalse((root / "data" / "obso").exists())
        # while the implementation and its parity tests remain
        self.assertTrue((SITE / "js" / "obso.js").exists())

    def test_no_abstract_placeholder_reaches_the_interface(self):
        for banned in ("XX%", "TBD", "lorem", "placeholder value"):
            self.assertNotIn(banned, self.markup, banned)

    def test_the_development_harness_is_not_part_of_the_site(self):
        """Section 33: a fixture page must be unable to ship."""

        harness = REPO_ROOT / "demo_viz" / "web" / "story_harness.html"
        self.assertTrue(harness.exists())
        self.assertFalse((SITE / "story_harness.html").exists(),
                         "the harness must live outside the copied site folder")
        self.assertIn("NOT PART OF THE PUBLIC SITE", harness.read_text())

    def test_the_built_site_contains_no_harness(self):
        import tempfile

        from demo_viz.web.build import build

        with tempfile.TemporaryDirectory() as out:
            root = Path(build(Path(out) / "site"))
            names = sorted(p.name for p in root.rglob("*.html"))
        # the page, and the method sheet it fetches when Details is opened --
        # no harness, no bench, no debug page
        self.assertEqual(["details.html", "index.html"], names)


class SolverReleaseLibraryTests(unittest.TestCase):
    """Section 17: the real action library, or an honest gap -- not the rays."""

    def setUp(self):
        self.block = build_scene("x").to_payload()["counterfactual"]["passer"]

    def test_the_release_library_is_reserved_and_unavailable(self):
        metric = self.block["release_library"]
        self.assertEqual("artifact_missing", metric["availability"])
        self.assertIsNone(metric["value"])

    def test_it_says_the_exploratory_rays_are_not_a_substitute(self):
        detail = self.block["release_library"]["detail"]
        self.assertIn("not a substitute", detail)
        self.assertIn("solved state", detail)

    def test_the_construction_is_still_implemented(self):
        """`solver_release_targets` reads DEFAULT_PASSES from the solver's own
        module, so the set cannot drift from it. Kept, not deleted."""

        source = (REPO_ROOT / "demo_viz" / "solver_native" / "release.py").read_text()
        self.assertIn("def solver_release_targets", source)
        self.assertIn("DEFAULT_PASSES", source)


class StoryTitleTests(unittest.TestCase):
    """Section 18: curated titles, with a safe fallback and no generation."""

    def test_the_showcase_accepts_a_title_and_a_summary(self):
        from demo_viz.core.showcase import validate_scene

        raw = {"showcase_id": "S01", "scene_id": "a:b", "provenance": "human",
               "mapping": {"status": "verified"}, "solver": {"status": "unavailable"},
               "roles": {"runners": [], "defenders": [], "beneficiaries": []},
               "story_title": "Static advantage disappears after response",
               "story_summary": "The runner's lane closes once the marker reacts."}
        scene = validate_scene(raw, 0)
        self.assertEqual("Static advantage disappears after response",
                         scene.story_title)
        self.assertIn("story_summary", scene.to_payload())

    def test_a_scene_without_a_title_falls_back_rather_than_inventing_one(self):
        from demo_viz.core.showcase import load

        for scene in load() or ():
            self.assertEqual("", scene.story_title,
                             "no title has been curated yet; nothing may "
                             "generate one")
        showcase = (SITE / "js" / "showcase.js").read_text()
        self.assertIn("scene.story_title", showcase)
        self.assertIn("scene.showcase_id", showcase)

    def test_the_final_ten_can_be_chosen_without_a_code_change(self):
        """Section 19: `featured` and `order` stay the only switches."""

        from demo_viz.core.showcase import ShowcaseScene

        fields = ShowcaseScene.__dataclass_fields__
        self.assertIn("featured", fields)
        self.assertIn("order", fields)


class CommandNamingTests(unittest.TestCase):
    """One definition of what a compass move is called."""

    def test_the_policy_view_reuses_the_solver_module_s_naming(self):
        policy = (SITE / "js" / "policy.js").read_text()
        self.assertIn('import { commandName } from "./solver.js"', policy)
        self.assertNotIn("toward own goal", policy,
                         "the naming lives in solver.js; a second copy would "
                         "drift from it")

    def test_the_names_are_the_solver_s_own_semantics(self):
        solver = (SITE / "js" / "solver.js").read_text()
        block = solver[solver.index("export function commandName"):]
        block = block[:block.index("\n/**")]
        for name in ("hold", "toward goal", "toward own goal",
                     "left touchline", "right touchline"):
            self.assertIn(f'"{name}"', block, name)


class LazyComponentTests(unittest.TestCase):
    """Section 36: a component used by one mode is not in the first load."""

    def setUp(self):
        self.app = (SITE / "js" / "app.js").read_text()

    def test_the_policy_and_strip_components_are_imported_on_demand(self):
        """Whatever the build calls deferred must actually be deferred."""

        from demo_viz.web.build import ON_DEMAND

        head = self.app[:self.app.index("const $ =")]
        deferred = {path.rsplit("/", 1)[-1] for path in ON_DEMAND
                    if path.endswith(".js")}
        self.assertIn("details.html", ON_DEMAND)
        self.assertIn('fetch("details.html")', self.app)
        for name in sorted(deferred):
            self.assertNotIn(name, head, name)
            if f'import("./{name}")' in self.app:
                continue
            # or it is reached only through another deferred module, which is
            # just as lazy: labels.js arrives with arrows.js
            importers = {
                other for other in deferred
                if other != name
                and f'"./{name}"' in (SITE / "js" / other).read_text()
            }
            self.assertTrue(importers, f"{name} is deferred but nothing fetches it")

    def test_the_strip_is_not_fetched_when_there_is_nothing_to_draw(self):
        """Today that means it is never fetched at all."""

        block = self.app[self.app.index("function renderEvalStrip"):]
        block = block[:block.index("\n// ---")] if "\n// ---" in block else block
        self.assertLess(block.index("if (!series.length)"), block.index("stripLib()"))

    def test_the_initial_load_stays_within_the_architecture_s_budget(self):
        import tempfile

        from demo_viz.web.build import build

        with tempfile.TemporaryDirectory() as out:
            root = Path(build(Path(out) / "site"))
            shell = sum(p.stat().st_size for p in root.rglob("*")
                        if p.is_file() and "data" not in p.relative_to(root).parts)
            index = (root / "data" / "index.json").stat().st_size
            contract = (root / "data" / "story" / "contract.json").stat().st_size
            extra = (root / "data" / "submission_showcase.json").stat().st_size
            initial = shell + index + contract + extra
            from demo_viz.web.build import ON_DEMAND as _DEFERRED

            for path in _DEFERRED:
                where = SITE / path
                initial -= (where if where.exists()
                            else root / path).stat().st_size
        # the lazy components are part of the shell on disk but are not fetched
        # until their mode is opened, so subtract what a visitor actually loads.
        # The list is the build's, so the two cannot drift apart.

        # 2026-09-30, figure alignment: the role markers, the attack-direction
        # indicator and the annotation->solver role translation run on every
        # render and cannot be deferred (+7 KB); splitting the carrier rule out
        # of obso.js and deferring the OBSO stack, the conventions and the
        # arrow language paid for them and more. Raise this only with a reason.
        # 2026-10-01, source-of-truth rebase: +2 KB for Figure 1's two-option
        # dilemma, which runs in render(). The figure conventions, the arrow
        # language, the policy component, the evaluation strip and the OBSO
        # stack are all deferred (33.9 KB) and excluded above.
        # 2026-10-01, showcase visual pass: the paper theme costs ~6 KB of CSS
        # (a body[data-paper] override per component) and ~2 KB of JS -- the
        # pitch, chart, arrows and stat each read their colours from the active
        # theme instead of the research palette. None of it can be deferred:
        # the theme is chosen before the first paint. Raise this only with a
        # reason, and prefer deferring a mode-specific module over raising it.
        # Which is what paid for it: release.js (the pass-model reader, wanted
        # only by the explorer's exploratory-pass layer) is now on demand too.
        # 2026-10-01, public editorial pass: the showcase ships a second
        # vocabulary (public.js) and a second chrome, and both are chosen
        # before the first paint. ranking.js and chart.js went on demand to
        # pay part of it -- the showcase has no hints card and no chart, so it
        # fetches neither. Raise this only with a reason, and prefer deferring
        # a mode-specific module over raising it.
        # 2026-10-01, the two scientific modes: Dilemma's candidate comparison
        # and Nash's moment picker. The evidence reader (compare.js) and the
        # defender-grid reader (grid.js) are both on demand -- Dilemma fetches
        # the first only when it opens, and the second stays dormant until the
        # three grid files exist. What is left is the wiring, which runs before
        # the first paint. Raise this only with a reason, and prefer deferring
        # a mode-specific module over raising it.
        # The shell now carries two interfaces, and the scientific modes' own
        # readers are all on demand: policy, evaluation strip, figure
        # conventions, arrow language, OBSO, pass model, rankings, chart,
        # tracking evidence and the defender grid. What is left runs before
        # the first paint. Prefer deferring a mode-specific module to raising
        # this; if it has to rise again, say which module could not be moved.
        self.assertLess(initial, 302 * 1024,
                        f"initial load is {initial / 1024:.1f} KB")


class LayoutTests(unittest.TestCase):
    """Section 35: no horizontal scrolling at the reviewer's width."""

    def setUp(self):
        self.css = (SITE / "style.css").read_text()

    def test_the_panel_grids_are_allowed_to_shrink(self):
        """A grid item's default min-width is min-content, which pushes a long
        label past the panel instead of wrapping it."""

        for rule in (".cfpair {", ".analysis dl {"):
            block = self.css[self.css.index(rule):]
            block = block[:block.index("}")]
            self.assertIn("minmax(0,", block, rule)

    def test_the_comparison_cards_cannot_overflow_their_column(self):
        block = self.css[self.css.index(".cfside {"):]
        self.assertIn("min-width: 0", block[:block.index("}")])


class SourceDiscoveryTests(unittest.TestCase):
    """`OFFBALL_EVALUATION_SOURCE` is the whole integration switch."""

    def setUp(self):
        import os

        self._saved = os.environ.get("OFFBALL_EVALUATION_SOURCE")

    def tearDown(self):
        import os

        if self._saved is None:
            os.environ.pop("OFFBALL_EVALUATION_SOURCE", None)
        else:
            os.environ["OFFBALL_EVALUATION_SOURCE"] = self._saved

    def _set(self, value):
        import os

        os.environ["OFFBALL_EVALUATION_SOURCE"] = value

    def test_unset_falls_back_to_the_local_bundle_when_installed(self):
        import os

        from demo_viz.paper_story import NoEvaluationSource, discover_source
        from demo_viz.paper_story.bundle import BUNDLE, available

        os.environ.pop("OFFBALL_EVALUATION_SOURCE", None)
        found = discover_source()
        if available():
            self.assertEqual(BUNDLE, found.name)
        else:
            self.assertIsInstance(found, NoEvaluationSource)

    def test_a_class_is_instantiated(self):
        from demo_viz.paper_story import discover_source

        self._set("tests.test_paper_story:_StubFactory")
        source = discover_source()
        self.assertEqual("stub_pipeline@abc1234", source.name)

    def test_a_malformed_spec_raises_rather_than_falling_back(self):
        from demo_viz.paper_story import ContractError, discover_source

        self._set("not_a_spec")
        with self.assertRaises(ContractError):
            discover_source()

    def test_a_source_missing_a_method_is_refused(self):
        from demo_viz.paper_story import ContractError, discover_source

        self._set("tests.test_paper_story:_Incomplete")
        with self.assertRaises(ContractError):
            discover_source()


#: The stub, addressable by `module:attribute` the way a real one will be.
_StubFactory = FutureIntegrationTests.StubSource


class _Incomplete:
    name = "broken"
    definition_version = "v0"

    def counterfactual(self, scene_id, role, frame):
        return None


class MetricFormattingTests(unittest.TestCase):
    """The browser formats by type, never by an assumed definition."""

    def test_no_metric_name_is_special_cased_for_formatting(self):
        story = (SITE / "js" / "story.js").read_text()
        block = story[story.index("export function metricText"):]
        block = block[:block.index("\n/**")]
        for name in ("relative_rank", "similarity_to_optimal", "regret",
                     "observed_action_rank"):
            self.assertNotIn(name, block,
                             f"{name} is formatted by its definition, which "
                             "this repository does not know")

    def test_an_integer_prints_as_an_integer(self):
        """A rank of 3 must not read as 3.000."""

        story = (SITE / "js" / "story.js").read_text()
        self.assertIn("Number.isInteger(value) ? String(value) : value.toFixed(3)",
                      story)


class MultiPassArtifactTests(unittest.TestCase):
    """Reading a solver run from kyuhyeok-dev@d1bbbb4 onward.

    That commit gives every pass candidate its own attack column
    (``multi_pass.MultiPassGame``: ``attack_columns = actions**2 + n_pass``),
    so a run's attack policy is no longer ``actions**2 + 1`` long and its last
    entry is one candidate among many. No such artifact exists locally yet, so
    the layout is pinned here against the schema the solver writes -- the
    alternative is finding out when the first real run lands.
    """

    #: A 2-direction game with 3 pass candidates: 4 move columns, then 3 passes.
    STATE = {
        "index": 7, "stratum": "eval", "value": 0.61,
        "scenario": {"pitch_length": 105.0, "pitch_width": 68.0, "attack_direction": 1,
                     "carrier": {"position": [60.0, 34.0], "velocity": [1.0, 0.0]},
                     "receiver": {"position": [70.0, 40.0], "velocity": [2.0, 0.0]},
                     "defender": {"position": [72.0, 36.0], "velocity": [0.0, 1.0]}},
        "certificate": {"lower": 0.61, "upper": 0.61, "gap": 0.0},
        "multi_pass": True,
        "pass_candidates": ["runner:ground:along 8 lateral 0 goalward 0",
                            "runner:ground:along 4 lateral -4 goalward 0",
                            "runner:ground:along 0 lateral 0 goalward 4"],
        "root_attack": [0.10, 0.05, 0.05, 0.20, 0.35, 0.15, 0.10],
        "root_defender": [0.7, 0.3],
        "modal_line": [{"step": 0, "time": 0.0, "value": 0.61,
                        "defender_pure_loss": 0.042,
                        "defender_policy": [0.7, 0.3],
                        "defender_values": [0.61, 0.61],
                        "carrier_slot_values": [0.55, 0.61],
                        "receiver_slot_values": [0.58, 0.61],
                        "release_value": None, "release_probability": 0.60,
                        "chosen_defender": 0, "chosen_attack": 4, "event": "move"}],
        "root_game": {"matrix": [[0.6, 0.5, 0.7], [0.4, 0.8, 0.6]],
                      "defender_legal": [True, True],
                      "attack_legal": [True, True, True],
                      "columns": ["move 0 0", "move 0 1", "runner:ground:along 8 lateral 0 goalward 0"]},
        "rollouts": [],
    }

    def _write(self, root):
        import json

        (root / "states").mkdir(parents=True)
        (root / "manifest.json").write_text(json.dumps(
            {"created_utc": "2026-09-30T00:00:00Z",
             "config": {"steps": 3, "step_seconds": 0.6,
                        "directions": [[1.0, 0.0], [0.0, 1.0]]}}))
        (root / "states" / "state_007.json").write_text(json.dumps(self.STATE))

    def _state(self):
        import tempfile

        from demo_viz.solver.adapter import load_state

        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self._write(root)
        return load_state(root, 7)

    def test_the_pass_columns_are_not_read_as_moves(self):
        state = self._state()
        self.assertTrue(state.multi_pass)
        self.assertEqual(4, state.move_columns)
        self.assertEqual(7, len(state.root_attack))

    def test_pass_probability_is_the_sum_over_candidates_not_the_last_entry(self):
        """The bug this guards: 0.10 reported where the truth is 0.60."""

        state = self._state()
        self.assertAlmostEqual(0.15 + 0.10 + 0.35, state.release_probability, places=9)
        self.assertNotAlmostEqual(state.root_attack[-1], state.release_probability)

    def test_the_likeliest_action_is_the_candidate_the_solver_named(self):
        state = self._state()
        best = state.most_likely_attack()
        self.assertEqual("release", best["kind"])
        self.assertAlmostEqual(0.35, best["probability"])
        self.assertEqual("runner:ground:along 8 lateral 0 goalward 0", best["candidate"])
        self.assertEqual(0, best["candidate_index"])

    def test_the_equilibrium_extras_survive_into_the_payload(self):
        payload = self._state().to_payload()
        self.assertEqual(4, payload["move_columns"])
        self.assertEqual(3, len(payload["pass_candidates"]))
        self.assertAlmostEqual(0.042, payload["modal_line"][0]["defender_pure_loss"])
        self.assertEqual(["move 0 0", "move 0 1",
                          "runner:ground:along 8 lateral 0 goalward 0"],
                         payload["root_game"]["columns"])

    def test_a_multi_pass_run_writes_no_rollouts_and_that_is_not_an_error(self):
        state = self._state()
        self.assertEqual((), state.trajectories)
        self.assertTrue(state.to_payload()["available"])

    def test_the_legacy_layout_still_reads_the_old_way(self):
        import json

        from demo_viz.solver.adapter import load_state

        for path in sorted((WEB_DATA / "solver").glob("*.json")):
            raw = json.loads(path.read_text())
            if not raw.get("available") or raw.get("kind") == "bundle_panels":
                continue
            self.assertFalse(raw.get("multi_pass", False))
            n = len(raw["directions"])
            self.assertEqual(n * n + 1, len(raw["root_attack"]))
            self.assertAlmostEqual(raw["root_attack"][-1], raw["release_probability"],
                                   places=9)

    def test_the_browser_reads_the_boundary_from_the_payload(self):
        policy = (SITE / "js" / "policy.js").read_text()
        block = policy[policy.index("export function attackRows"):]
        block = block[:block.index("\n/**")]
        self.assertIn("state.move_columns", block)
        self.assertIn("pass_candidates", block)
        self.assertNotIn("index === n * n", block,
                         "the release boundary is data, not a constant")


class SelectorLabelTests(unittest.TestCase):
    """Section 13: the selector reads as football, not as a database key."""

    def setUp(self):
        self.source = (SITE / "js" / "showcase.js").read_text()
        self.block = self.source[self.source.index("export function selectorLabel"):]
        self.block = self.block[:self.block.index("\n/**")]

    def test_the_fixture_is_the_fallback_when_no_title_is_curated(self):
        self.assertIn("scene.story_title || fixture", self.block)
        self.assertIn("scene.match", self.block)

    def test_the_technical_id_is_always_kept(self):
        self.assertIn("scene.showcase_id", self.block)

    def test_nothing_generates_a_title(self):
        for banned in ("template", "`The ", "join(\" pulls \")"):
            self.assertNotIn(banned, self.block, banned)


class CaseStudyLineTests(unittest.TestCase):
    """Section 6: a case-study sentence, but never invented prose."""

    def setUp(self):
        self.app = (SITE / "js" / "app.js").read_text()
        self.block = self.app[self.app.index("function renderCaseStudy"):]
        self.block = self.block[:self.block.index("\nfunction ")]

    def test_it_prefers_a_curated_summary_and_falls_back_to_the_reviewer(self):
        self.assertIn("curated?.story_summary", self.block)
        self.assertIn("curated?.annotation", self.block)

    def test_a_reviewer_note_is_shown_as_a_quotation(self):
        """Quoted text is visibly someone's words, not the interface's."""

        self.assertIn("\\u201c", self.block)
        self.assertIn("quoted", self.block)
        self.assertIn("verbatim", self.block)

    def test_no_tactical_sentence_is_assembled_anywhere(self):
        """Checked against code only: the comment above renderCaseStudy names
        such a sentence in order to forbid it."""

        import re

        code = re.sub(r"/\*.*?\*/", "", self.app, flags=re.S)
        code = "\n".join(line.split("//")[0] for line in code.splitlines())
        for banned in ("pulls the defender", "drags the", "creates space for",
                       " away from ", "tactical"):
            self.assertNotIn(banned, code.lower(), banned)


class AdvancedControlTests(unittest.TestCase):
    """Section 15: the research toggles are kept, but not in the reviewer's face."""

    def setUp(self):
        self.markup = (SITE / "index.html").read_text()
        self.app = (SITE / "js" / "app.js").read_text()

    def _row(self, marker):
        start = self.markup.index(marker)
        return self.markup[start:self.markup.index("</div>", start)]

    def test_the_first_control_row_is_short(self):
        row = self._row('<div class="layers" id="layers-row">')
        visible = row.count("data-layer=")
        self.assertLessEqual(visible, 6,
                             f"{visible} always-visible toggles is a forest")

    def test_the_research_layers_are_kept_not_deleted(self):
        for layer in ("candidates", "lane", "paths", "reach", "solver", "passes"):
            self.assertIn(f'data-layer="{layer}"', self.markup, layer)

    def test_the_advanced_panel_starts_collapsed(self):
        body = self.markup[self.markup.index('id="adv-body"'):][:40]
        self.assertIn("hidden", body)

    def test_a_layer_a_mode_switched_on_is_named_while_collapsed(self):
        """Otherwise the pitch changes with no visible cause."""

        block = self.app[self.app.index("function renderAdvanced"):]
        block = block[:block.index("\n/**")]
        self.assertIn("adv-hint", block)
        self.assertIn("state.advancedOpen || !on.length", block)

    def test_a_default_on_layer_is_not_announced_as_a_change(self):
        self.assertIn("ADVANCED_DEFAULT_ON", self.app)
        block = self.app[self.app.index("const ADVANCED_DEFAULT_ON"):][:120]
        self.assertIn("candidates", block)


class PublicLabelTests(unittest.TestCase):
    """Labels that would misstate what upstream computes.

    `analyze_eval` defines similarity as the equilibrium's own probability of
    the observed option. Shown as "Similarity to optimal 0.78" that reads as a
    distance from a best action, which it is not -- and under a mixed
    equilibrium there is no single best action to be distant from.
    """

    def setUp(self):
        self.payload = build_scene("x").to_payload()
        self.metrics = {m["name"]: m
                        for m in self.payload["evaluation"]["runner"]["metrics"]}

    def test_similarity_keeps_its_key_for_compatibility(self):
        self.assertIn("similarity_to_optimal", self.metrics)

    def test_but_is_labelled_for_what_it_is(self):
        label = self.metrics["similarity_to_optimal"]["label"]
        self.assertEqual("Equilibrium probability of observed action", label)
        self.assertNotIn("Similarity", label)

    def test_and_says_what_it_is_not(self):
        detail = self.metrics["similarity_to_optimal"]["detail"]
        self.assertIn("Not a distance", detail)
        self.assertIn("not a cosine", detail)

    def test_no_single_optimal_action_is_claimed(self):
        action = self.payload["evaluation"]["runner"]["optimal_action"]
        self.assertEqual("Best-valued option", action["label"])
        self.assertIn("no single optimal action", action["detail"])

    def test_the_browser_shows_the_payload_label_not_its_own(self):
        """So a label fix in the adapter reaches the page without a UI change."""

        app = (SITE / "js" / "app.js").read_text()
        self.assertIn("dt.textContent = record.label", app)
        for banned in ("Similarity to optimal", "Optimal action"):
            self.assertNotIn(banned, app, banned)


class PublicCopyTests(unittest.TestCase):
    """The Submission showcase speaks football; the explorer speaks pipeline."""

    def setUp(self):
        self.app = (SITE / "js" / "app.js").read_text()
        self.public = (SITE / "js" / "public.js").read_text()
        self.markup = (SITE / "index.html").read_text()

    def test_the_four_tabs_are_named_and_numbered_for_a_public_audience(self):
        """The numbers carry the paper's story order, in plain type."""

        for name in ("1. Play", "2. Dilemma", "3. Nash equilibrium",
                     "4. Player evaluation"):
            self.assertIn(f'data-public="{name}"', self.markup, name)

    def test_the_tabs_are_in_that_order_in_the_dom(self):
        order = re.findall(r'data-public="(\d)\.', self.markup)
        self.assertEqual(["1", "2", "3", "4"], order)

    def test_the_public_title_is_the_paper_s(self):
        self.assertIn("<title>The Defender's Dilemma | SSAC 2027</title>", self.markup)
        self.assertIn('content="The Defender\'s Dilemma | SSAC 2027"', self.markup)
        self.assertIn('"The Defender\'s Dilemma"', self.app)

    def test_a_tab_carries_no_subtitle_in_the_showcase(self):
        block = self.app[self.app.index("function applyPublicChrome"):]
        block = block[:block.index("\n/**")]
        self.assertIn("question.hidden = showcase", block)

    def test_the_showcase_hides_the_interface_headings_and_role_controls(self):
        block = self.app[self.app.index("function applyPublicChrome"):]
        block = block[:block.index("\n/**")]
        for node in ("collection-label", "showcase-label", "roles-control",
                     "showcase-filters", "jump-peak", "stat-card", "hints-card",
                     "notes-card", "layers-row"):
            self.assertIn(f'$("{node}")', block, node)

    def test_the_showcase_panel_drops_the_identifier_and_source_sections(self):
        block = self.app[self.app.index("const PUBLIC_SECTIONS"):]
        block = block[:block.index("};")]
        for banned in ("an-scene", "an-source", "an-clip", "an-player"):
            self.assertNotIn(banned, block, banned)

    def test_every_command_the_bundle_names_has_a_football_word(self):
        """A new bundle with a new command name fails here, not on the pitch."""

        solver = WEB_DATA / "solver"
        if not solver.exists():
            self.skipTest("no exported solver panels")
        defender, attack = set(), set()
        for path in sorted(solver.glob("*.json")):
            payload = json.loads(path.read_text())
            if payload.get("kind") != "bundle_panels":
                continue
            for panel in payload.get("panels", []):
                for body in panel["bodies"].values():
                    names = defender if body["solver_role"] == "defender" else attack
                    for option in body["options"]:
                        names.add(option["label"])
        self.assertTrue(defender, "no defender options in the exported panels")
        block = self.public[self.public.index("DEFENDER_FOOTBALL = {"):]
        block = block[:block.index("};")]
        for name in sorted(defender):
            self.assertIn(f'"{name}"', block.replace("sideways:", '"sideways":'), name)
        for name in sorted(attack):
            self.assertTrue(
                f'{name}:' in self.public or f'"{name}"' in self.public, name)

    def test_command_zero_is_never_called_a_hold(self):
        """It is maximum braking along the current heading, not a hold.

        `agile_motion.steer` with a zero desired velocity; the research code
        calls it "slow down" for that reason. See PAPER_FIGURE_ALIGNMENT.md
        section 10.
        """

        source = re.sub(r"/\*.*?\*/", "", self.public, flags=re.S)
        source = re.sub(r"//.*", "", source)
        self.assertNotIn("Hold", source)
        for word in ('"Slow down', '"Stop'):
            self.assertIn(word, source)

    def test_the_showcase_does_not_repoint_the_cast_on_a_click(self):
        """Selection is not editing.

        A click in the showcase may open a comparison; it may never reach
        `state.selection`, because the curated cast is what every solver and
        evaluation number is keyed to.
        """

        block = self.app[self.app.index("function beginDrag"):]
        block = block[:block.index("function markDropTargets")]
        public = block[block.index("!dragged.moved && isPublic()"):]
        public = public[:public.index("} else if")]
        self.assertIn("compareWith(", public)
        self.assertNotIn("state.selection", public)
        self.assertNotIn("state.inspect", public)

    def test_a_comparison_never_writes_to_the_curated_cast(self):
        block = self.app[self.app.index("function compareWith"):]
        block = block[:block.index("\n/** Back to the curated play")]
        self.assertNotIn("state.selection.", block)
        self.assertIn("state.compare =", block)

    def test_leaving_dilemma_returns_to_the_curated_play(self):
        block = self.app[self.app.index("function setMode"):]
        block = block[:block.index("\n/**")]
        self.assertIn('if (mode !== "counterfactual") state.compare = null;', block)

    def test_the_explorer_keeps_its_own_words(self):
        for name in ("Observed", "Counterfactual", "Evaluation", "Game solution"):
            self.assertIn(f'"{name}"', self.app, name)
        self.assertIn("MODE_SECTIONS", self.app)
        self.assertIn('"an-source"', self.app)


class ScientificModeTests(unittest.TestCase):
    """Sections 4-10: the two contributions, without their jargon."""

    def setUp(self):
        self.app = (SITE / "js" / "app.js").read_text()
        self.markup = (SITE / "index.html").read_text()

    def test_the_dilemma_shows_three_quantities_and_not_the_heuristic_score(self):
        panel = (SITE / "js" / "panel.js").read_text()
        block = panel[panel.index("function dilemmaMetrics"):]
        block = block[:block.index("\n/**")]
        for label in ("Marking distance", "Reaction", "Space created"):
            self.assertIn(f'"{label}"', block, label)
        # the 0.55/0.45 blend is the app's own weighting, not a research output
        self.assertNotIn("score", block)

    def test_the_evidence_is_read_not_recomputed(self):
        """Marking distance and reaction come from the export, not the browser."""

        compare = (SITE / "js" / "compare.js").read_text()
        for banned in ("Math.hypot", "Math.cos", "velocity"):
            self.assertNotIn(banned, compare, banned)
        source = (REPO_ROOT / "demo_viz" / "web" / "export_compare.py").read_text()
        self.assertIn("marking_series", source)
        self.assertIn("defender_reaction_index", source)

    def test_the_moment_picker_offers_only_solved_moments(self):
        block = self.app[self.app.index("function renderMoments"):]
        block = block[:block.index("\n/**")]
        self.assertIn("data.panels", block)
        self.assertIn("panel.dt.toFixed(1)", block)

    def test_the_defender_field_control_is_hidden_without_data(self):
        """No dead public control: absent grids mean no checkbox at all."""

        panel = (SITE / "js" / "panel.js").read_text()
        block = panel[panel.index("function renderDefenderField"):]
        self.assertIn("control.hidden = !grids", block)
        self.assertIn("if (!grids || !c.state.field) return;", block)

    def test_a_partly_covered_field_is_not_drawn(self):
        panel = (SITE / "js" / "panel.js").read_text()
        block = panel[panel.index("function gridsFor"):]
        block = block[:block.index("\n/**")]
        self.assertIn("grids.every(Boolean)", block)

    def test_the_grid_reader_uses_the_upstream_definitions(self):
        """The abstract's own flags, not the renderer's defaults."""

        grid = (SITE / "js" / "grid.js").read_text()
        self.assertIn("preferred defender position", grid)   # --value-key
        self.assertIn('"#B8AE9C"', grid)                     # BG_HIGH
        self.assertIn('"#FF8000"', grid)                     # --flow-color
        self.assertIn("alpha: 0.4", grid)                    # --flow-alpha
        # the move field is the export's own probability-weighted 0.6 s
        # displacement -- read, not recomputed from the per-command ends
        block = grid[grid.index("export function movePoints"):]
        block = block[:block.index("\n}")]
        self.assertIn("[p.x, p.y, p.dx, p.dy]", block)
        source = (REPO_ROOT / "demo_viz" / "web" / "export_grid.py").read_text()
        self.assertIn('"dx": round(float(row["dx"]), 4)', source)

    def test_the_actual_move_follows_the_figures_convention(self):
        pitch = (SITE / "js" / "pitch.js").read_text()
        block = pitch[pitch.index("drawRealMoves("):]
        block = block[:block.index("\n  draw")]
        self.assertIn("stroke-dasharray", block)
        self.assertIn('fill: "none"', block)              # a hollow marker

    def test_every_solved_moment_is_in_the_table(self):
        """The equilibrium is three solves, so all three are side by side."""

        panel = (SITE / "js" / "panel.js").read_text()
        block = panel[panel.index("function renderMomentTable"):]
        block = block[:block.index("\n/**")]
        self.assertIn("data.panels", block)            # every solved moment
        self.assertIn("panel.dt.toFixed(1)", block)    # named by its own time
        self.assertIn("is-now", block)                 # the one on screen, marked

    def test_the_static_comparison_is_shown_whole(self):
        """Section 22: both values and the overestimate, not just the ratio."""

        panel = (SITE / "js" / "panel.js").read_text()
        block = panel[panel.index("function renderMomentTable"):]
        block = block[:block.index("\n/**")]
        for label in ("Fixed defender", "Responding defender",
                      "Fixed-defence overestimate"):
            self.assertIn(f'"{label}"', block, label)

    def test_the_nash_panel_reads_only_the_solved_panel(self):
        """Nothing here may recompute or interpolate a solver value."""

        panel = (SITE / "js" / "panel.js").read_text()
        block = panel[panel.index("function renderMomentTable"):]
        block = block[:block.index("\n/**")]
        self.assertIn("c.panelAt(data, c.state.frame)", block)
        for banned in ("Math.", "interpolat", "* 0.5", "+ 1"):
            self.assertNotIn(banned, block, banned)

    def test_no_selection_checks_block_exists(self):
        """S05 fails the isolation filter, so nothing may imply it was extracted."""

        for banned in ("Selection checks", "isolation", "passes the filter",
                       "extraction"):
            self.assertNotIn(banned, self.markup, banned)


class PublicShellTests(unittest.TestCase):
    """Sections 1-6: the public header, the panel, and the type sizes."""

    def setUp(self):
        self.app = (SITE / "js" / "app.js").read_text()
        self.css = (SITE / "style.css").read_text()
        self.markup = (SITE / "index.html").read_text()

    def test_the_collection_switch_is_not_in_the_public_header(self):
        block = self.app[self.app.index("function applyPublicChrome"):]
        block = block[:block.index("\n/**")]
        self.assertIn('$("collection-control").hidden = showcase', block)

    def test_the_explorer_is_still_reachable(self):
        """Removed from the header, not from the build."""

        self.assertIn('params.get("explorer")', self.app)
        self.assertIn('wanted.scene || wanted.explorer', self.app)
        self.assertIn('data-collection="explorer"', self.markup)

    def test_public_type_sizes_meet_the_lower_bounds(self):
        block = self.css[self.css.index("body.is-public {"):]
        block = block[:block.index("}")]
        wanted = {"--t-body": 15, "--t-name": 16, "--t-head": 17,
                  "--t-num": 17, "--t-title": 20}
        for token, floor in wanted.items():
            line = next(l for l in block.splitlines() if token in l)
            size = float(re.search(r"(\d+(?:\.\d+)?)px", line).group(1))
            self.assertGreaterEqual(size, floor, f"{token} is {size}px")

    def test_no_public_rule_sets_text_below_the_floor(self):
        """A 10-12px helper line is what the pass exists to remove."""

        small = []
        for rule in re.findall(r"body\.is-public[^{]*\{[^}]*\}", self.css):
            for size in re.findall(r"font-size:\s*(\d+(?:\.\d+)?)px", rule):
                if float(size) < 15:
                    small.append(rule.split("{")[0].strip() + f" -> {size}px")
        self.assertEqual([], small, "\n".join(small))

    def test_the_panel_is_wide_enough_to_work_in(self):
        rule = next(r for r in re.findall(r"body\.is-public \.dock \{[^}]*\}", self.css))
        self.assertIn("clamp(330px", rule)

    def test_the_player_rows_are_clickable_and_large(self):
        rule = next(r for r in re.findall(r"\.players li \{[^}]*\}", self.css))
        self.assertIn("cursor: pointer", rule)
        self.assertIn("var(--t-name)", rule)


class SceneListTests(unittest.TestCase):
    """Twenty scenes have a solved equilibrium; the list leads with them."""

    def test_the_build_counts_each_scenes_solved_moments(self):
        build = (REPO_ROOT / "demo_viz" / "web" / "build.py").read_text()
        self.assertIn('keep["solved_moments"]', build)
        # read from the exported panels, never declared by hand
        self.assertIn('payload.get("kind") != "bundle_panels"', build)

    def test_the_list_leads_with_the_scenes_that_have_one(self):
        showcase = (SITE / "js" / "showcase.js").read_text()
        block = showcase[showcase.index("export function filtered"):]
        block = block[:block.index("\n}")]
        self.assertIn("solved_moments", block)
        self.assertIn("return bs - as", block)

    def test_every_solver_file_is_a_scene_in_the_list(self):
        """A solved scene the list does not carry is a scene nobody can open."""

        import json
        from demo_viz.web.export_data import slug
        rows = json.loads((REPO_ROOT / "demo_viz" / "data"
                           / "submission_showcase.json").read_text())["scenes"]
        listed = {r.get("scene_id") for r in rows}
        solver = REPO_ROOT / "demo_viz" / "web_data" / "solver"
        missing = []
        for path in sorted(solver.glob("*.json")):
            payload = json.loads(path.read_text())
            if payload.get("kind") != "bundle_panels":
                continue
            if payload.get("scene_id") not in listed:
                missing.append(path.name)
            else:
                self.assertEqual(slug(payload["scene_id"]), path.stem)
        self.assertEqual([], missing, missing)


class EvaluationReadingTests(unittest.TestCase):
    """The public Player evaluation tab: what is drawn, and what is not."""

    def setUp(self):
        self.app = (SITE / "js" / "app.js").read_text()
        self.css = (SITE / "style.css").read_text()
        self.options = (SITE / "js" / "options.js").read_text()
        self.panel = (SITE / "js" / "panel.js").read_text()

    def test_the_frame_by_frame_strip_is_explorer_only(self):
        block = self.app[self.app.index("function renderEvalStrip"):]
        block = block[:block.index("\n/**")]
        self.assertIn('state.mode !== "evaluation" || isPublic()', block)

    def test_every_option_is_drawn_not_only_the_named_ones(self):
        block = self.panel[self.panel.index("export function renderDecisionPaths"):]
        block = block[:block.index("\n}")]
        self.assertIn("all: (entry.options || []).map", block)

    def test_the_label_halo_is_the_figures_own_ratio(self):
        """`figure_style.halo()` is 2.2pt against FS_LABEL 9.0 -- 0.24 em.

        It must be written in `em`: a `px` on an SVG geometry property is a
        user unit, and this pitch's user unit is a metre.
        """

        for rule in re.findall(r"[^}]*\{[^}]*paint-order: stroke[^}]*\}", self.css):
            width = re.search(r"stroke-width:\s*([\d.]+)(em|px)", rule)
            self.assertIsNotNone(width, rule)
            self.assertEqual("em", width.group(2), rule)
            self.assertLessEqual(float(width.group(1)), 0.3, rule)

    def test_the_ball_carrier_is_ranked_like_everyone_else(self):
        """A dash read as missing data; the note says what is missing."""

        self.assertNotIn('entry.rank_partial ? "\\u2014"', self.panel)
        block = self.panel[self.panel.index("export function renderOptions"):]
        self.assertIn("lib.ordinal(option.rank)", block)

    def test_both_solved_moment_tabs_hold_at_a_solved_moment(self):
        block = self.app[self.app.index("function holdOnSolvedMoment"):]
        block = block[:block.index("\n}")]
        self.assertIn('state.mode === "game_solution"', block)
        self.assertIn('state.mode === "evaluation"', block)
        self.assertIn("2000", block)

    def test_the_options_say_they_are_clickable(self):
        markup = (SITE / "index.html").read_text()
        block = markup[markup.index('id="options-card"'):]
        self.assertIn("click to compare", block[:block.index("</div>")])

    def test_the_played_option_is_the_role_colour_in_the_list_too(self):
        block = self.panel[self.panel.index("export function renderOptions"):]
        self.assertIn("is-playedrow", block)
        self.assertIn('row.style.setProperty("--pick-played", played)', block)
        self.assertIn("--pick-played", self.css)

    def test_the_two_accents_are_the_same_in_the_list_and_on_the_pitch(self):
        for colour in ("#00A8D8", "#FF2D8E"):
            self.assertIn(colour, self.options, colour)
            self.assertIn(colour, self.css, colour)

    def test_the_played_option_keeps_the_role_colour(self):
        block = self.options[self.options.index("export function drawDecision"):]
        self.assertIn("accent(observed, colour)", block)

    def test_the_decision_layer_is_the_topmost_one(self):
        pitch = (SITE / "js" / "pitch.js").read_text()
        block = pitch[pitch.index("this.layers = {}"):]
        block = block[:block.index("]) {")]
        order = re.findall(r'"([a-z]+)"', block)
        self.assertEqual("decision", order[-1], order)
        self.assertIn("decision", pitch[pitch.index("clearDynamic"):])


class DilemmaInteractionTests(unittest.TestCase):
    """Sections 9-12, 32: a click changes the pitch, not only a number."""

    def setUp(self):
        self.app = (SITE / "js" / "app.js").read_text()

    def test_the_marking_line_follows_the_compared_defender(self):
        block = self.app[self.app.index("const markers = state.compare?.defender"):]
        block = block[:block.index("renderDilemma")]
        self.assertIn("pitch.drawTether(scene, runnerId, markers", block)
        self.assertIn("pitch.drawGhost(scene, cache, markers", block)

    def test_the_role_sweep_restricts_each_role_to_its_own_side(self):
        block = self.app[self.app.index("function compareWith"):]
        block = block[:block.index("\n/**")]
        self.assertIn('wantsDefender !== (player.side === "defend")', block)
        self.assertIn("player.gk", block)

    def test_clicking_the_compared_player_again_clears_it(self):
        block = self.app[self.app.index("function compareWith"):]
        block = block[:block.index("\n/**")]
        self.assertIn("const already = next[role] === playerId", block)
        self.assertIn("if (curated || already) delete next[role]", block)

    def test_the_cast_has_a_cell_per_role(self):
        """Default beside Selected, and all three roles live at once."""

        panel = (SITE / "js" / "panel.js").read_text()
        block = panel[panel.index("function renderCast"):]
        block = block[:block.index("\n/**")]
        self.assertIn('"Default"', block)
        self.assertIn('"Selected"', block)
        for role in ('"runner"', '"teammate"', '"defender"'):
            self.assertIn(role, block, role)
        # the cell arms its own role and empties only itself
        self.assertIn("c.state.sweep = role", block)
        self.assertIn("c.compareWith(chosenId)", block)

    def test_the_three_compared_roles_are_independent(self):
        """A teammate and a defender can be swapped at the same time."""

        block = self.app[self.app.index("function compareWith"):]
        block = block[:block.index("\n/**")]
        # the other roles' choices are carried forward, not discarded
        self.assertIn("{ ...(state.compare || {}) }", block)
        panel = (SITE / "js" / "panel.js").read_text()
        sweep = panel[panel.index("export function renderSweep"):]
        sweep = sweep[:sweep.index("\n/**")]
        # and changing which role the next click fills clears nothing
        self.assertNotIn("c.state.compare = null", sweep)

    def test_the_marking_numbers_are_read_at_the_playhead(self):
        """Section 3: a window mean does not move while the clip plays."""

        panel = (SITE / "js" / "panel.js").read_text()
        block = panel[panel.index("function dilemmaMetrics"):]
        block = block[:block.index("\n/**")]
        self.assertIn("lib.markingAt(row, frame)", block)
        self.assertIn("lib.secondsToReaction(payload, row, frame, times)", block)
        compare = (SITE / "js" / "compare.js").read_text()
        # the per-frame series is read, never recomputed in the browser
        self.assertIn("row.marking_dm", compare)
        self.assertIn("row.reaction_index", compare)

    def test_the_run_trail_follows_a_compared_runner(self):
        block = self.app[self.app.index("const runnerId = state.compare?.runner"):]
        block = block[:block.index("renderDilemma")]
        self.assertIn("pitch.drawTrail(scene, runnerId", block)

    def test_the_space_field_is_drawn_for_the_teammate_in_question(self):
        block = self.app[self.app.index("const dilemmaSpace"):]
        block = block[:block.index("} else if")]
        self.assertIn("state.compare?.teammate", block)
        self.assertIn("pitch.drawField(cache.grid", block)

    def test_each_tab_has_its_own_view_default(self):
        """The two solved-moment tabs open cropped; the other two do not."""

        block = self.app[self.app.index("const PUBLIC_VIEW = {"):]
        block = block[:block.index("};")]
        self.assertIn('observed: "full"', block)
        self.assertIn('counterfactual: "full"', block)
        self.assertIn('game_solution: "focus"', block)
        self.assertIn('evaluation: "focus"', block)

    def test_a_chosen_view_does_not_leak_between_tabs(self):
        block = self.app[self.app.index("function setMode"):]
        block = block[:block.index("\n/**")]
        self.assertIn("delete state.viewByMode[leaving]", block)
        self.assertIn("state.viewByMode[mode] || PUBLIC_VIEW[mode]", block)

    def test_the_dilemma_overlay_is_total_space_and_the_metric_is_the_delta(self):
        """Section 11: two quantities, two jobs, never the same one twice."""

        block = self.app[self.app.index("if (dilemmaSpace && spaceFor.length)"):]
        block = block[:block.index("} else if")]
        # the overlay is the residual surface itself, with no subtraction
        self.assertIn("cache.combined(spaceFor, slot).field", block)
        self.assertNotIn("Math.max(", block)
        panel = (SITE / "js" / "panel.js").read_text()
        metric = panel[panel.index("function dilemmaMetrics"):]
        metric = metric[:metric.index("\n/**")]
        # the metric keeps the held-defender subtraction
        self.assertIn('mode: "hold"', metric)
        self.assertIn("- cache.combined(", metric.replace("\n", " "))


class PublicTimelineTests(unittest.TestCase):
    """The public clip plays from its own first frame.

    `scene.times` stays the solver's coordinate -- zero at the annotated shot,
    which every solved moment and evaluation series is keyed to. Only the
    public readout is re-zeroed.
    """

    def setUp(self):
        self.app = (SITE / "js" / "app.js").read_text()

    def test_a_public_scene_opens_on_its_first_frame(self):
        block = self.app[self.app.index("  if (isPublic()) {\n    // the clip's first"):]
        block = block[:block.index("} else {")]
        self.assertIn("state.frame = 0;", block)
        for banned in ("onsetFor", "peakGainFrame", "decisionFrame"):
            self.assertNotIn(banned, block, banned)

    def test_nothing_annotation_derived_moves_the_public_playhead(self):
        """The URL path used to jump to the influence peak on any ?param."""

        block = self.app[self.app.index("if (wanted.time != null"):]
        block = block[:block.index("\n}")]
        self.assertIn("} else if (!isPublic()) {", block)
        peak = block[block.index("peakGainFrame"):]
        self.assertNotIn("isPublic", peak.split("} else {")[0])

    def test_the_public_readout_is_clip_local(self):
        block = self.app[self.app.index("const clipTimeSec"):]
        block = block[:block.index("\n\n")]
        self.assertIn("scene.times[index] - scene.times[0]", block)
        self.assertIn("isPublic()", block)
        # and the explorer keeps the solver's own clock
        self.assertIn("const solverTimeSec = scene.times[index];", block)

    def test_the_solver_clock_is_not_rewritten(self):
        """A re-zeroed display may not become a re-zeroed coordinate."""

        for banned in ("scene.times[i] -= ", "times = scene.times.map",
                       "scene.times = "):
            self.assertNotIn(banned, self.app, banned)

    def test_the_solved_moments_still_come_from_the_panels(self):
        block = self.app[self.app.index("function renderMoments"):]
        block = block[:block.index("\n/**")]
        self.assertIn("panel.dt.toFixed(1)", block)
        self.assertIn("frameOfPanel(panel)", block)
        frame_of = self.app[self.app.index("function frameOfPanel"):]
        frame_of = frame_of[:frame_of.index("\n}")]
        # the panel's own frame, clamped -- never shifted by the clip's start
        self.assertIn("panel.frame", frame_of)
        self.assertNotIn("times[0]", frame_of)


class ModuleHealthTests(unittest.TestCase):
    """A duplicate declaration is a SyntaxError that shows as a blank page.

    `boot()` never runs, nothing logs where a probe would see it, and the
    shell renders its static defaults -- which looks exactly like a stale
    build. Source-level checks are cheap; the browser-level one is in
    `tests/test_module_imports.py`.
    """

    def setUp(self):
        self.root = SITE / "js"

    def test_no_module_declares_the_same_top_level_name_twice(self):
        bad = []
        for path in sorted(self.root.glob("*.js")):
            text = path.read_text()
            names = (re.findall(r"^function (\w+)\(", text, re.M)
                     + re.findall(r"^(?:const|let|class) (\w+)\b", text, re.M)
                     + re.findall(r"^export (?:async )?function (\w+)\(", text, re.M)
                     + re.findall(r"^export (?:const|let|class) (\w+)\b", text, re.M))
            seen = set()
            for name in names:
                if name in seen:
                    bad.append(f"{path.name}: {name}")
                seen.add(name)
        self.assertEqual([], bad, "\n".join(bad))

    def test_every_static_import_resolves(self):
        bad = []
        for path in sorted(self.root.glob("*.js")):
            text = path.read_text()
            for spec in (re.findall(r'from\s+"(\./[^"]+)"', text)
                         + re.findall(r'import\("(\./[^"]+)"\)', text)):
                if not (self.root / spec[2:]).exists():
                    bad.append(f"{path.name} -> {spec}")
        self.assertEqual([], bad, "\n".join(bad))

    def test_every_named_import_is_exported(self):
        bad = []
        for path in sorted(self.root.glob("*.js")):
            text = path.read_text()
            for block, module in re.findall(
                    r'import\s+\{([^}]*)\}\s+from\s+"\./([\w.]+\.js)"', text):
                source = (self.root / module).read_text()
                for raw in block.split(","):
                    name = raw.strip().split(" as ")[0].strip()
                    if not name:
                        continue
                    exported = (
                        re.search(rf"export\s+(?:async\s+)?"
                                  rf"(?:function|const|let|class)\s+{re.escape(name)}\b",
                                  source)
                        or re.search(rf"export\s*\{{[^}}]*\b{re.escape(name)}\b",
                                     source))
                    if not exported:
                        bad.append(f"{path.name} imports {name} from {module}")
        self.assertEqual([], bad, "\n".join(bad))


class PublicWordingTests(unittest.TestCase):
    """Public names for quantities whose research names are opaque."""

    def setUp(self):
        self.panel = (SITE / "js" / "panel.js").read_text()

    def test_the_evaluation_rows_are_named_in_words(self):
        for label in ("Observed", "Rank", "Probability under equilibrium",
                      "Expected loss vs best"):
            self.assertIn(f'"{label}"', self.panel, label)

    def test_the_opaque_names_are_gone_from_the_public_panel(self):
        source = re.sub(r"//.*", "", re.sub(r"/\*.*?\*/", "", self.panel, flags=re.S))
        for banned in ('"Regret"', '"Equilibrium probability"', '"Similarity"'):
            self.assertNotIn(banned, source, banned)

    def test_the_space_row_reads_before_to_after(self):
        """Not a bare delta: the two numbers the delta is between."""

        block = self.panel[self.panel.index("const spanOf = (id)"):]
        block = block[:block.index("out.push([\"Space created\"")]
        self.assertIn("cache.combined([id], slot).value", block)
        self.assertIn("cache.combined([id], slot, swap).value", block)
        self.assertIn("\\u2192", block)

    def test_a_change_in_space_is_coloured_by_direction(self):
        css = (SITE / "style.css").read_text()
        self.assertIn(".m-value.is-up { color: #0000FF; }", css)
        self.assertIn(".m-value.is-down { color: #FF0000; }", css)

    def test_the_dilemma_panel_is_two_columns_while_comparing(self):
        block = self.panel[self.panel.index("const comparing ="):]
        block = block[:block.index("for (const [label")]
        self.assertIn('"Default"', block)
        self.assertIn('"Selected"', block)
        self.assertIn("is-two-up", block)


class BundleIntegrationTests(unittest.TestCase):
    """The 2026-10-01 vector-field and ranking bundle, as exported."""

    GRID = WEB_DATA / "grid"
    OPTIONS = WEB_DATA / "options"

    def test_the_defender_grids_are_exported_for_every_solved_moment(self):
        if not self.GRID.exists():
            self.skipTest("no grid export")
        names = sorted(p.name for p in self.GRID.glob("*.json"))
        self.assertEqual(["S05_0.0.json", "S05_0.6.json", "S05_1.2.json"], names)

    def test_a_grid_carries_the_value_and_the_move(self):
        if not self.GRID.exists():
            self.skipTest("no grid export")
        payload = json.loads((self.GRID / "S05_0.0.json").read_text())
        self.assertEqual("grid/1", payload["schema"])
        self.assertEqual(1.0, payload["spacing_m"])
        self.assertGreater(len(payload["points"]), 200)
        for point in payload["points"]:
            for key in ("i", "j", "x", "y", "v", "dx", "dy"):
                self.assertIn(key, point)
        # exactly one start is where the defender really was
        observed = [p for p in payload["points"] if p.get("observed")]
        self.assertEqual(1, len(observed))

    def test_every_exported_rank_matches_the_bundle(self):
        """The one test that would catch a wrong slot mapping."""

        import csv

        from demo_viz.paper_story.bundle import DEFAULT_ROOT, roots

        players = DEFAULT_ROOT / "analysis" / "players.csv"
        if not (self.OPTIONS.exists() and players.exists()):
            self.skipTest("no options export or no bundle")
        stored = {}
        for _, folder in roots():               # every installed bundle, the first first
            for r in csv.DictReader((folder / "analysis" / "players.csv").open()):
                stored.setdefault((r["code"], r["role"]), r)
        bad = []
        checked = partial = 0
        for path in sorted(self.OPTIONS.glob("*.json")):
            payload = json.loads(path.read_text())
            code = payload["code"]
            for moment in payload["moments"]:
                dt = moment["dt"]
                suffix = "" if dt == 0.0 else f"@{dt:.1f}".rstrip("0").rstrip(".")
                for role, entry in moment["roles"].items():
                    row = stored.get((f"{code}{suffix}", role))
                    if not row:
                        continue
                    if entry.get("rank_partial"):
                        partial += 1
                        continue
                    try:
                        index = int(row["observed"])
                    except ValueError:
                        continue
                    mine = next((o for o in entry["options"]
                                 if o["command"] == index), None)
                    if mine is None:
                        continue
                    checked += 1
                    if mine["rank"] != int(row["rank"]):
                        bad.append(f"{code}{suffix} {role}: "
                                   f"{mine['rank']} != {row['rank']}")
        self.assertEqual([], bad, "\n".join(bad))
        self.assertGreater(checked, 40, "too few ranks compared to mean anything")

    def test_the_carrier_s_missing_pass_is_declared_not_guessed(self):
        if not self.OPTIONS.exists():
            self.skipTest("no options export")
        flagged = []
        for path in sorted(self.OPTIONS.glob("*.json")):
            payload = json.loads(path.read_text())
            for moment in payload["moments"]:
                for role, entry in moment["roles"].items():
                    if entry.get("rank_partial"):
                        flagged.append((role, entry.get("missing_option")))
        self.assertTrue(flagged, "no carrier moment was flagged")
        for role, why in flagged:
            self.assertEqual("ball carrier", role)
            self.assertIn("pass", why)
