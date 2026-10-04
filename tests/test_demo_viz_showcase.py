"""The curated submission showcase: ingestion, mapping, schema and the UI rules.

The showcase file is curation metadata. Two things matter most here and both
are tested hard: that no trajectory from the local-only source can reach it,
and that a mapping is never guessed -- an entry either names the repository
scene the evidence supports, or it names none.
"""

from __future__ import annotations

import json
import re
import sys
import tempfile
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from demo_viz.core.showcase import (  # noqa: E402
    ShowcaseError,
    default_filter,
    featured,
    load,
    validate,
)
from demo_viz.web.build import SITE  # noqa: E402

SHOWCASE = REPO_ROOT / "demo_viz" / "data" / "submission_showcase.json"
SOURCE = REPO_ROOT / "local_inputs" / "dilemma_showcase.html"
EXPECTED_TOTAL = 20     # 21 curated, S30 left out (ingest_showcase.EXCLUDED, 2026-10-04)
EXPECTED_HUMAN = 15
EXPECTED_SOLVER = 5


def _raw() -> dict:
    return json.loads(SHOWCASE.read_text())


class ShowcaseContentTests(unittest.TestCase):
    """The curated set, as parsed, is the set that was curated."""

    def test_exactly_the_curated_scenes_are_present(self):
        scenes = load(SHOWCASE)
        self.assertEqual(EXPECTED_TOTAL, len(scenes))
        ids = {s.showcase_id for s in scenes}
        self.assertEqual(EXPECTED_TOTAL, len(ids))

    def test_provenance_counts(self):
        scenes = load(SHOWCASE)
        human = [s for s in scenes if s.provenance == "human"]
        solver = [s for s in scenes if s.provenance == "solver"]
        self.assertEqual(EXPECTED_HUMAN, len(human))
        self.assertEqual(EXPECTED_SOLVER, len(solver))

    def test_reviewer_ratings_are_preserved(self):
        scenes = load(SHOWCASE)
        for scene in scenes:
            self.assertTrue(scene.reviewers, scene.showcase_id)
            self.assertEqual(len(scene.ratings), len(scene.reviewers), scene.showcase_id)
            for rating in scene.ratings:
                self.assertIn(rating, (1, 2, 3, 4, 5))
            self.assertEqual("/".join(str(r) for r in scene.ratings),
                             scene.agreement_label, scene.showcase_id)

    def test_rating_groups(self):
        scenes = load(SHOWCASE)
        labels = [s.agreement_label for s in scenes]
        self.assertEqual(7, labels.count("5/5"))
        self.assertEqual(EXPECTED_TOTAL - 7,
                         sum(1 for label in labels if label in ("5/4", "4/5")))

    def test_every_scene_keeps_its_hand_assigned_roles(self):
        for scene in load(SHOWCASE):
            for key in ("runners", "defenders", "beneficiaries"):
                self.assertTrue(scene.roles[key], f"{scene.showcase_id}.{key}")
                for shirt in scene.roles[key]:
                    self.assertRegex(shirt, r"^\d+$")

    def test_timing_convention_is_kept_per_scene(self):
        """Solver entries are run-onset centred; the rest are shot centred."""

        for scene in load(SHOWCASE):
            expected = "run_onset" if scene.provenance == "solver" else "슛"
            self.assertEqual(expected, scene.timing, scene.showcase_id)

    def test_reviewer_notes_survive_but_stay_out_of_the_selector(self):
        scenes = load(SHOWCASE)
        self.assertTrue(any(r.note for s in scenes for r in s.reviewers))
        source = (SITE / "js" / "showcase.js").read_text()
        selector = source[source.index("export function selectorLabel"):]
        selector = selector[:selector.index("\n}")]
        for banned in ("note", "annotation"):
            self.assertNotIn(banned, selector)


class ShowcaseMappingTests(unittest.TestCase):
    def test_mapping_counts(self):
        scenes = load(SHOWCASE)
        verified = [s for s in scenes if s.mapping_status == "verified"]
        unresolved = [s for s in scenes if s.mapping_status == "unresolved"]
        # since 2026-10-03 the five pipeline scenes are published too
        # (data/pipeline_scenes.json), so every curated entry maps
        self.assertEqual(EXPECTED_TOTAL, len(verified))
        self.assertEqual(0, len(unresolved))
        self.assertEqual(0, sum(1 for s in scenes if s.mapping_status == "ambiguous"))

    def test_a_verified_mapping_points_at_a_scene_the_demo_ships(self):
        index = json.loads((REPO_ROOT / "demo_viz" / "web_data" / "index.json").read_text())
        known = {row["scene_id"] for row in index["scenes"]}
        for scene in load(SHOWCASE):
            if scene.mapping_status == "verified":
                self.assertIn(scene.scene_id, known, scene.showcase_id)

    def test_an_unresolved_entry_names_no_scene(self):
        """The rule that stops a guess becoming a mapping."""

        for scene in load(SHOWCASE):
            if scene.mapping_status != "verified":
                self.assertIsNone(scene.scene_id, scene.showcase_id)
                self.assertFalse(scene.playable, scene.showcase_id)

    def test_every_mapping_records_its_evidence(self):
        for scene in load(SHOWCASE):
            self.assertTrue(scene.mapping_evidence.strip(), scene.showcase_id)

    def test_the_verified_roles_exist_in_the_mapped_scene(self):
        """The hand-labelled shirts must actually be in that scene's annotation."""

        web = REPO_ROOT / "demo_viz" / "web_data"
        files = {json.loads((web / row["file"]).read_text())["scene_id"]: row["file"]
                 for row in json.loads((web / "index.json").read_text())["scenes"]}
        for scene in load(SHOWCASE):
            if scene.mapping_status != "verified":
                continue
            payload = json.loads((web / files[scene.scene_id]).read_text())
            by_id = {p["id"]: p for p in payload["players"]}
            for key, role, side in (("runners", "runner", "attack"),
                                    ("defenders", "defender", "defend"),
                                    ("beneficiaries", "beneficiary", "attack")):
                annotated = {by_id[i]["shirt"] for i in payload["roles"][role] if i in by_id}
                for shirt in scene.roles[key]:
                    self.assertIn(shirt, annotated,
                                  f"{scene.showcase_id} {key} #{shirt} "
                                  f"absent from {scene.scene_id}")
                    match = next(p for p in payload["players"]
                                 if p["shirt"] == shirt and p["side"] == side)
                    self.assertEqual(side, match["side"])


class NoRawDataLeakTests(unittest.TestCase):
    """Nothing from the local-only source may be republished."""

    def test_the_local_source_is_not_tracked(self):
        import subprocess

        out = subprocess.run(["git", "ls-files", "local_inputs"],
                             cwd=REPO_ROOT, capture_output=True, text=True)
        self.assertEqual("", out.stdout.strip(), "local_inputs must not be tracked")

    def test_local_inputs_is_ignored(self):
        self.assertIn("local_inputs/", (REPO_ROOT / ".gitignore").read_text())

    def test_the_showcase_file_carries_no_trajectory(self):
        text = SHOWCASE.read_text()
        # a track would show up as a long run of comma-separated numbers
        self.assertEqual([], re.findall(r"\[[-0-9.,\s]{80,}\]", text))
        raw = _raw()
        for scene in raw["scenes"]:
            for banned in ("t", "ball", "xy", "players", "tracks"):
                self.assertNotIn(banned, scene, scene["showcase_id"])

    def test_the_validator_rejects_a_record_carrying_tracks(self):
        with self.assertRaises(ShowcaseError):
            validate({"scenes": [{
                "showcase_id": "S01", "provenance": "human",
                "mapping": {"status": "unresolved"},
                "solver": {"status": "unavailable"},
                "ball": [[0, 0], [1, 1]],
            }]})

    def test_the_showcase_file_is_small(self):
        self.assertLess(SHOWCASE.stat().st_size, 120 * 1024,
                        "the showcase file should hold metadata, not data")


class ShowcaseSchemaTests(unittest.TestCase):
    def _entry(self, **overrides):
        base = {
            "showcase_id": "S01", "scene_id": None, "provenance": "human",
            "featured": False, "order": None,
            "review": {"reviewers": [], "agreement_label": ""},
            "roles": {"runners": [], "defenders": [], "beneficiaries": []},
            "mapping": {"status": "unresolved", "evidence": "x"},
            "solver": {"status": "unavailable", "scenario_type": None},
        }
        base.update(overrides)
        return base

    def test_absent_file_falls_back_to_the_explorer(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(load(Path(tmp) / "missing.json"))

    def test_verified_requires_a_scene_id(self):
        with self.assertRaises(ShowcaseError):
            validate({"scenes": [self._entry(mapping={"status": "verified", "evidence": "x"})]})

    def test_unresolved_may_not_name_a_scene_id(self):
        with self.assertRaises(ShowcaseError):
            validate({"scenes": [self._entry(scene_id="J03WOH:shot_006_P1_1054")]})

    def test_solver_available_requires_an_artifact(self):
        with self.assertRaises(ShowcaseError):
            validate({"scenes": [self._entry(
                solver={"status": "available", "scenario_type": "2v1", "artifact": None})]})

    def test_unknown_status_values_are_refused(self):
        for field, bad in (("mapping", {"status": "probably", "evidence": ""}),
                           ("solver", {"status": "maybe"})):
            with self.assertRaises(ShowcaseError):
                validate({"scenes": [self._entry(**{field: bad})]})

    def test_duplicate_ids_and_orders_are_refused(self):
        with self.assertRaises(ShowcaseError):
            validate({"scenes": [self._entry(), self._entry()]})
        with self.assertRaises(ShowcaseError):
            validate({"scenes": [
                self._entry(showcase_id="S01", featured=True, order=1),
                self._entry(showcase_id="S02", featured=True, order=1)]})


class FeaturedBehaviourTests(unittest.TestCase):
    """Marking the final ten must be a metadata edit, nothing more."""

    def test_default_filter_is_all_while_nothing_is_featured(self):
        scenes = load(SHOWCASE)
        self.assertFalse(any(s.featured for s in scenes),
                         "the final ten are not chosen yet; nothing may be featured")
        self.assertEqual("all", default_filter(scenes))

    def test_default_filter_becomes_featured_once_any_scene_is(self):
        raw = _raw()
        raw["scenes"][3]["featured"] = True
        raw["scenes"][3]["order"] = 1
        scenes = validate(raw)
        self.assertEqual("featured", default_filter(scenes))
        chosen = featured(scenes)
        self.assertEqual(1, len(chosen))
        self.assertEqual(raw["scenes"][3]["showcase_id"], chosen[0].showcase_id)

    def test_featured_scenes_come_back_in_order(self):
        raw = _raw()
        # `order` is unique across the whole file, and the lead scene already
        # holds one (ingest_showcase.LEAD_SCENE), so start from a clean slate:
        # this is about the ordering rule, not about the real curation.
        for scene in raw["scenes"]:
            scene["order"] = None
        for position, index in enumerate((5, 1, 9), start=1):
            raw["scenes"][index]["featured"] = True
            raw["scenes"][index]["order"] = position
        chosen = featured(validate(raw))
        self.assertEqual([1, 2, 3], [s.order for s in chosen])

    def test_order_must_be_unique_across_the_file(self):
        """Which is why the test above clears it first."""

        from demo_viz.core.showcase import ShowcaseError

        raw = _raw()
        raw["scenes"][0]["order"] = 1
        raw["scenes"][1]["order"] = 1
        with self.assertRaises(ShowcaseError):
            validate(raw)

    def test_the_lead_scene_opens_first_and_is_set_at_ingest(self):
        """S05 is the paper's Figure 1 / Figure 2 scene. The selector sorts by
        `order`, the ingest writes it, and nothing is featured -- so the full
        curated list is still shown and no final ten is implied."""

        from demo_viz.ingest_showcase import LEAD_SCENE

        scenes = load(SHOWCASE)
        ordered = sorted(scenes, key=lambda s: (s.order if s.order is not None
                                                else float("inf"), s.showcase_id))
        self.assertEqual(LEAD_SCENE, ordered[0].showcase_id)
        self.assertEqual(1, ordered[0].order)
        self.assertEqual([LEAD_SCENE],
                         [s.showcase_id for s in scenes if s.order is not None])
        self.assertFalse(any(s.featured for s in scenes))

    def test_the_browser_uses_the_same_default_rule(self):
        source = (SITE / "js" / "showcase.js").read_text()
        self.assertIn('scenes.some((s) => s.featured) ? "featured" : "all"', source)


class ShowcaseBrowserTests(unittest.TestCase):
    def test_the_build_serves_the_showcase(self):
        build_py = (REPO_ROOT / "demo_viz" / "web" / "build.py").read_text()
        self.assertIn("submission_showcase.json", build_py)

    def test_public_labels_avoid_internal_vocabulary(self):
        source = (SITE / "js" / "showcase.js").read_text()
        self.assertIn('human: "Human-reviewed"', source)
        self.assertIn('solver: "Solver-derived"', source)
        markup = (SITE / "index.html").read_text()
        self.assertIn("Submission showcase", markup)
        self.assertIn("Full explorer", markup)

    def test_an_unplayable_entry_is_disabled_not_hidden(self):
        """The curated set is 21; hiding five would misrepresent it."""

        source = (SITE / "js" / "app.js").read_text()
        block = source[source.index("function refreshShowcaseList"):]
        block = block[:block.index("\n}\n")]
        self.assertIn("option.disabled = !scene.playable", block)

    def test_ratings_are_never_called_confidence(self):
        for path in (SITE / "index.html", SITE / "js" / "showcase.js", SITE / "js" / "app.js"):
            text = path.read_text().lower()
            for banned in ("confidence score", "model confidence", "probability of dilemma"):
                self.assertNotIn(banned, text, path.name)


@unittest.skipIf(not SOURCE.exists(), "local showcase source not present")
class IngestionTests(unittest.TestCase):
    """The generator reproduces the committed file from the local source."""

    def test_regenerating_matches_what_is_committed(self):
        from demo_viz.ingest_showcase import build

        rebuilt = json.dumps(build(SOURCE), ensure_ascii=False, indent=2) + "\n"
        self.assertEqual(SHOWCASE.read_text(), rebuilt,
                         "submission_showcase.json is stale; re-run "
                         "python -m demo_viz.ingest_showcase")

    def test_the_source_really_holds_the_expected_scene_count(self):
        from demo_viz.core.showcase_ingest import read_showcase
        from demo_viz.ingest_showcase import EXCLUDED

        # the source keeps every curated entry; the demo leaves EXCLUDED out
        self.assertEqual(EXPECTED_TOTAL + len(EXCLUDED), len(read_showcase(SOURCE)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
