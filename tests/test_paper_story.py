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

    def test_the_regret_slot_says_which_regret_it_is_not(self):
        """The `regret` that exists belongs to another lineage; the panel must
        not let a reviewer assume it is this one."""

        regret = next(m for m in self.payload["evaluation"]["runner"]["metrics"]
                      if m["name"] == "regret")
        self.assertIn("dynamic_response_game", regret["detail"])

    def test_the_static_side_declares_no_semantics_yet(self):
        static = self.payload["counterfactual"]["runner"]["static"]
        self.assertEqual("", static["semantics"],
                         "three readings of 'static' exist; picking one is a "
                         "research decision, not an interface default")

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

    def test_no_development_fixture_can_reach_the_public_payload(self):
        """Section 33: a layout fixture must never ship."""

        for path in self.files:
            payload = json.loads(path.read_text())
            self.assertEqual("none", payload["provenance"]["evaluation_source"],
                             f"{path.name} was exported from a live source; "
                             "check it is the real pipeline, not a fixture")
            for record in self._records(payload):
                self.assertNotIn("stub", str(record.get("source") or "").lower())
                self.assertNotIn("fixture", str(record.get("source") or "").lower())

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

    def test_no_bundesliga_scene_claims_an_equilibrium_today(self):
        """Stated plainly so the day one does, this test says so."""

        attached = [p.name for p in self.files
                    if json.loads(p.read_text())["equilibrium"]["availability"]
                    == "available"]
        self.assertEqual([], attached,
                         "a solver artifact now covers a tracked scene; update "
                         "PAPER_STORY_TRACE.md and this test together")


class EquilibriumArtifactTests(unittest.TestCase):
    """The half of the abstract that is real, checked against real output."""

    @classmethod
    def setUpClass(cls):
        cls.files = sorted((WEB_DATA / "solver").glob("*.json"))
        if not cls.files:
            raise unittest.SkipTest("no solver artifacts exported")

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
        block = equilibrium_for("J03WOY:shot_011_P2_0903", WEB_DATA / "solver")
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
            names = [p.name for p in root.rglob("*.html")]
        self.assertEqual(["index.html"], names)


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
        head = self.app[:self.app.index("const $ =")]
        for name in ("policy.js", "evalstrip.js", "figure.js", "arrows.js",
                     "obso.js"):
            self.assertNotIn(name, head, name)
            self.assertIn(f'import("./{name}")', self.app, name)

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
        # the two lazy components are part of the shell on disk but are never
        # fetched today, so subtract them for what a visitor actually loads
        for name in ("policy.js", "evalstrip.js", "figure.js", "arrows.js",
                     "obso.js"):
            initial -= (SITE / "js" / name).stat().st_size
        # 2026-09-30, figure alignment: the role markers, the attack-direction
        # indicator and the annotation->solver role translation run on every
        # render and cannot be deferred (+7 KB); splitting the carrier rule out
        # of obso.js and deferring the OBSO stack, the conventions and the
        # arrow language paid for them and more. Raise this only with a reason.
        self.assertLess(initial, 250 * 1024,
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

    def test_unset_means_no_source(self):
        import os

        from demo_viz.paper_story import NoEvaluationSource, discover_source

        os.environ.pop("OFFBALL_EVALUATION_SOURCE", None)
        self.assertIsInstance(discover_source(), NoEvaluationSource)

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
            if not raw.get("available"):
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
        row = self._row('<div class="layers">')
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
