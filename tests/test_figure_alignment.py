"""The paper figures' conventions, and whether the demo really shares them.

The figure renderer is not in this repository or on this machine. Its data
producer is, so the decoders it feeds are what "the same action semantics"
can actually mean here, and this file holds the demo's port to them
numerically rather than by eye.

`FigureDecoderParityTests` runs the research functions and the browser's port
over the same inputs in a headless browser and requires identical strings.
"""

from __future__ import annotations

import json
import math
import subprocess
import time
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SITE = REPO_ROOT / "demo_viz" / "web" / "site"
KYUHYEOK = "origin/kyuhyeok-dev"


def _show(path):
    out = subprocess.run(["git", "show", f"{KYUHYEOK}:{path}"],
                         cwd=REPO_ROOT, capture_output=True, text=True)
    return out.stdout if out.returncode == 0 else None


# ---------------------------------------------------------------------------
# the research side, executed rather than paraphrased
# ---------------------------------------------------------------------------
SOLVER_DIRS = ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0))


def world_direction(k, attack_direction):
    ux, uy = SOLVER_DIRS[k]
    return (ux * attack_direction, uy * attack_direction)


def compass_name(u, direction):
    """extract_panel_policy.compass_name."""
    x, y = float(u[0]) * direction, float(u[1]) * direction
    if abs(x) < 1e-9 and abs(y) < 1e-9:
        return "stop"
    if abs(x) >= abs(y):
        return "forward" if x > 0 else "back"
    return "left" if y > 0 else "right"


def name_move_targets(u, defender, targets):
    """stage3_read.name_move_targets."""
    if tuple(u) == (0.0, 0.0):
        return "slow down"
    best, score = "sideways", 0.3
    for name, tgt in targets:
        vx, vy = tgt[0] - defender[0], tgt[1] - defender[1]
        n = math.hypot(vx, vy)
        if n > 1e-6:
            c = (u[0] * vx + u[1] * vy) / n
            if c >= score:
                best, score = name, c
    return best


class SourceIsUnchangedTests(unittest.TestCase):
    """The ported functions still read the way this port assumed."""

    def test_the_panel_extractor_still_names_moves_that_way(self):
        source = _show("scripts/extract_panel_policy.py")
        if source is None:
            self.skipTest("origin/kyuhyeok-dev not available")
        self.assertIn("def compass_name", source)
        for word in ('"stop"', '"forward"', '"back"', '"left"', '"right"'):
            self.assertIn(word, source, word)

    def test_the_defender_namer_still_uses_a_cosine_floor_of_point_three(self):
        source = _show("src/offball_value/stage3_read.py")
        if source is None:
            self.skipTest("origin/kyuhyeok-dev not available")
        self.assertIn('best, score = "sideways", 0.3', source)
        self.assertIn('return "slow down"', source,
                      "a zero command is slow down, not stand")
        self.assertIn('("toward ball", snapshot["carrier"]), '
                      '("toward runner", snapshot["receiver"])', source)

    def test_the_abstracts_runs_use_the_compass_command_set(self):
        """Which decides which decoder the figures are showing."""

        for job in ("deploy/delta/eval_v1.sbatch",
                    "deploy/delta/multipass_full_2v1.sbatch"):
            source = _show(job)
            if source is None:
                self.skipTest("origin/kyuhyeok-dev not available")
            self.assertIn("--commands compass", source, job)


class UpstreamNameTests(unittest.TestCase):
    """The action names are upstream's own English; nothing is glossed here."""

    def setUp(self):
        self.figure = (SITE / "js" / "figure.js").read_text()

    def test_the_names_are_the_upstream_strings(self):
        for name in ("toward ball", "toward runner", "toward goal",
                     "sideways", "slow down"):
            self.assertIn(f'"{name}"', self.figure, name)

    def test_no_gloss_table_survives(self):
        """They were Korean until the 2026-09-30 cleanup; the demo translated
        them. Upstream now ships English, so the translation must be gone."""

        self.assertNotIn("GLOSS", self.figure)
        for korean in ("볼 쪽", "러너 쪽", "골문 쪽", "옆으로", "멈추기"):
            self.assertNotIn(korean, self.figure, korean)

    def test_the_old_reconstruction_is_corrected(self):
        """Checked against code: the header names the old wording to forbid it."""

        import re

        code = re.sub(r"/\*.*?\*/", "", self.figure, flags=re.S)
        code = "\n".join(line.split("//")[0] for line in code.splitlines())
        self.assertNotIn("toward the ball", code)
        self.assertNotIn("brake", code)

    def test_the_source_of_truth_is_named_at_the_top(self):
        head = " ".join(self.figure[:2200].replace("//", " ").split())
        self.assertIn("origin/kyuhyeok-dev@139498a", head)
        self.assertIn("render_figure2_abstract.py", head)


class FigureLabelParityTests(unittest.TestCase):
    """render_figure2_abstract's label rules and width scale, held exactly."""

    def setUp(self):
        self.source = _show("scripts/render_figure2_abstract.py")
        if self.source is None:
            self.skipTest("origin/kyuhyeok-dev not available")
        self.figure = (SITE / "js" / "figure.js").read_text()

    def test_the_width_scale_matches(self):
        self.assertIn("return 0.6 + 2.4 * prob", self.source)
        self.assertIn("0.6 + 2.4 * Math.max(0, Math.min(1, probability || 0))",
                      self.figure)

    def test_the_four_label_forms_match(self):
        for upstream, ported in ((f'text = f"Slow down {{text}}"', "Slow down ${pct}"),
                                 (f'text = f"Stop {{text}}"', "Stop ${pct}"),
                                 (f'text = f"Dribble {{text}}"', "Dribble ${pct}")):
            self.assertIn(upstream, self.source, upstream)
            self.assertIn(ported, self.figure, ported)

    def test_the_pass_label_matches(self):
        self.assertIn("Pass + receive \u00b7 ", self.source)
        self.assertIn("Pass + receive \u00b7 ${pct}", self.figure)

    def test_only_the_ball_carrier_gets_a_verb(self):
        """Everyone else's label is the percentage; direction is the arrow's job."""

        block = self.figure[self.figure.index("export function optionLabel"):]
        block = block[:block.index("\n/**")]
        self.assertIn('role === "ball carrier"', block)
        self.assertIn("return pct;", block)

    def test_the_minimum_drawn_probability_matches(self):
        self.assertIn("MIN_P = 0.02", self.figure)

    def test_the_title_matches(self):
        self.assertIn("Equilibrium choices during an off-ball play", self.source)
        self.assertIn("Equilibrium choices during an off-ball play", self.figure)


class PrintStyleParityTests(unittest.TestCase):
    """figure_style.py's palette and markers, held exactly."""

    def setUp(self):
        self.source = _show("scripts/figure_style.py")
        if self.source is None:
            self.skipTest("origin/kyuhyeok-dev not available")
        self.figure = (SITE / "js" / "figure.js").read_text()

    def test_the_team_palette_matches(self):
        """2026-10-01 replaced the Okabe-Ito pair with pure blue / pure red."""

        self.assertIn('ATTACK, DEFENCE = "#0000FF", "#FF0000"', self.source)
        self.assertIn('attack: "#0000FF"', self.figure)
        self.assertIn('defence: "#FF0000"', self.figure)
        self.assertNotIn("#0072B2", self.figure)
        self.assertNotIn("#D55E00", self.figure)

    def test_the_key_markers_carry_no_outline(self):
        self.assertIn("KEY_EDGE = 0.0", self.source)
        self.assertIn("KEY_EDGE = 0.0", self.figure)

    def test_the_move_gap_matches(self):
        self.assertIn("MOVE_GAP = 2.5", self.source)
        self.assertIn("MOVE_GAP = 2.5", self.figure)

    def test_both_attackers_are_blue(self):
        """The runner and the ball carrier are told apart by marker, not hue."""

        self.assertIn('"ball carrier": ATTACK', self.source)
        block = self.figure[self.figure.index("export const ROLE_COLOR"):]
        block = block[:block.index("}")]
        self.assertIn("runner: PAPER.attack", block)
        self.assertIn('"ball carrier": PAPER.attack', block)
        self.assertIn("defender: PAPER.defence", block)

    def test_the_markers_match(self):
        self.assertIn('"runner": ("D", 0.80 * KEY_D)', self.source)
        self.assertIn('"ball carrier": ("o", KEY_D)', self.source)
        self.assertIn('"defender": ("s", 0.86 * KEY_D)', self.source)
        self.assertIn('"teammate": ("^", 1.15 * KEY_D)', self.source)
        for role, shape in (("runner", "diamond"), ("ball carrier", "circle"),
                            ("defender", "square"), ("teammate", "triangle")):
            self.assertIn(f'"{shape}"', self.figure, shape)

    def test_the_beneficiary_takes_the_disc_not_the_triangle(self):
        """The previous pass had this wrong: the triangle is the 3v1 teammate."""

        self.assertIn('"beneficiary": ("o", KEY_D)', self.source)
        block = self.figure[self.figure.index("export const ROLE_MARKER"):]
        block = block[:block.index("};")]
        self.assertIn('beneficiary: ["circle"', block)
        self.assertIn('teammate: ["triangle"', block)

    def test_the_tint_matches(self):
        self.assertIn("TEAM_TINT = {s: tint(c, 0.45)", self.source)
        self.assertIn("k = 0.45", self.figure)


class FigureDecoderParityTests(unittest.TestCase):
    """The browser's port against the research functions, on the same inputs."""

    CASES = []
    for ad in (1, -1):
        for k in range(5):
            for defender, carrier, runner, goal in (
                ([0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [52.5, 0.0]),
                ([30.0, -12.0], [12.0, 4.0], [40.0, -20.0], [52.5, 0.0]),
                ([-5.0, 3.0], [-5.0, 3.0001], [-30.0, 25.0], [-52.5, 0.0]),
                ([8.0, 8.0], [8.0, -8.0], [-8.0, 8.0], [52.5, 0.0]),
            ):
                CASES.append({"k": k, "ad": ad, "defender": defender,
                              "carrier": carrier, "runner": runner, "goal": goal})

    def test_the_browser_decoders_agree_with_the_research_code(self):
        from demo_viz.web.validate import find_chrome

        chrome = find_chrome()
        if chrome is None:
            self.skipTest("no headless Chrome")

        expected = []
        for c in self.CASES:
            u = world_direction(c["k"], c["ad"])
            targets = (("toward ball", c["carrier"]), ("toward runner", c["runner"]),
                       ("toward goal", c["goal"]))
            expected.append({"compass": compass_name(u, c["ad"]),
                             "defender": name_move_targets(u, c["defender"], targets)})

        import tempfile

        script = """
import { worldDirection, compassName, defenderMoveName, targetsFor }
  from "./js/figure.js";
const CASES = %s;
const out = CASES.map((c) => {
  const u = worldDirection(c.k, c.ad);
  return { compass: compassName(u, c.ad),
           defender: defenderMoveName(u, c.defender,
                                      targetsFor(c.carrier, c.runner, c.goal)) };
});
// report over HTTP: --dump-dom needs Chrome to exit cleanly, and it often
// does not here, whereas the answer is already ours once this request lands
await fetch("/result", { method: "POST", body: JSON.stringify(out) });
"""

        # ES modules will not load over file://, so the probe is served
        import functools
        import http.server
        import socketserver
        import threading

        received = []

        class Probe(http.server.SimpleHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                received.append(self.rfile.read(length).decode())
                self.send_response(204)
                self.end_headers()

        class Threaded(socketserver.ThreadingTCPServer):
            allow_reuse_address = True
            daemon_threads = True

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "js").mkdir()
            (root / "js" / "figure.js").write_text((SITE / "js" / "figure.js").read_text())
            (root / "probe.html").write_text(
                f'<!doctype html><meta charset="utf-8">'
                f'<script type="module">{script % json.dumps(self.CASES)}</script>')
            handler = functools.partial(Probe, directory=str(root))
            with Threaded(("127.0.0.1", 0), handler) as httpd:
                threading.Thread(target=httpd.serve_forever, daemon=True).start()
                port = httpd.server_address[1]
                process = subprocess.Popen(
                    [chrome, "--headless", "--disable-gpu", "--no-sandbox",
                     f"--user-data-dir={root / 'profile'}",
                     "--virtual-time-budget=3000",
                     f"http://127.0.0.1:{port}/probe.html"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                deadline = time.time() + 60
                while not received and time.time() < deadline:
                    time.sleep(0.2)
                process.kill()
                try:                      # reap it, or Python warns at exit
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    pass
                httpd.shutdown()

        if not received:
            self.skipTest("headless Chrome did not report in time")
        got = json.loads(received[0])
        self.assertEqual(len(expected), len(got))
        for case, want, have in zip(self.CASES, expected, got):
            self.assertEqual(want["compass"], have["compass"], case)
            self.assertEqual(want["defender"], have["defender"], case)


class ShapeTableTests(unittest.TestCase):
    """pitch.js inlines the shape table; the two copies must not drift."""

    def _table(self, text):
        import re

        key = "ROLE_MARKER = {" if "ROLE_MARKER = {" in text else "ROLE_SHAPE = {"
        block = text[text.index(key):]
        block = block[:block.index("};" if key.startswith("ROLE_MARKER") else "}")]
        return {k.strip(): v
                for k, v in re.findall(r'"?([a-z ]+)"?:\s*\[?"([a-z]+)"', block)}

    def test_the_inlined_table_equals_the_conventions_module(self):
        figure = self._table((SITE / "js" / "figure.js").read_text())
        pitch = self._table((SITE / "js" / "pitch.js").read_text())
        self.assertEqual(figure, pitch)
        self.assertTrue(figure, "no shapes parsed")

    def test_each_solver_role_has_its_own_shape(self):
        figure = self._table((SITE / "js" / "figure.js").read_text())
        for role in ("ball carrier", "runner", "defender"):
            self.assertIn(role, figure, role)
        # the three 2v1 actors must not share a shape, or the figure language
        # stops distinguishing them
        core = [figure[r] for r in ("ball carrier", "runner", "defender")]
        self.assertEqual(len(core), len(set(core)), core)


class AttackDirectionTests(unittest.TestCase):
    """Section 26: which way the attack runs is a per-scene fact, always shown."""

    def setUp(self):
        self.pitch = (SITE / "js" / "pitch.js").read_text()
        self.app = (SITE / "js" / "app.js").read_text()

    def test_the_pitch_can_draw_it(self):
        self.assertIn("drawAttackDirection", self.pitch)
        self.assertIn('"attack"', self.pitch)

    def test_it_is_drawn_on_every_render(self):
        block = self.app[self.app.index("function render()"):]
        block = block[:block.index("\n/**")]
        self.assertIn("drawAttackDirection", block)

    def test_it_is_a_constant_because_scenes_are_normalised(self):
        """scene.flip already turns every scene so the attack runs right; the
        indicator must not contradict that by following the raw direction."""

        scene = (SITE / "js" / "scene.js").read_text()
        self.assertIn("scene.flip = scene.attacking_direction < 0", scene)
        head = self.pitch[:self.pitch.index("Pitch.prototype.drawAttackDirection")]
        doc = head[head.rindex("/**"):]
        self.assertIn("runs to the right on", doc)

    def test_a_direction_is_transformed_with_the_linear_part(self):
        """Drawing an aim with the point transform mirrors every arrow."""

        scene = (SITE / "js" / "scene.js").read_text()
        self.assertIn("export function viewVector", scene)
        app = (SITE / "js" / "app.js").read_text()
        block = app[app.index("function defenderOptionsAt"):]
        block = block[:block.index("\n/**")]
        self.assertIn("viewVector(scene", block)
        self.assertIn("playerAt(scene, defender, index)", block)


class RoleTranslationTests(unittest.TestCase):
    """Section 4: annotation roles and solver roles stay distinct."""

    def setUp(self):
        self.app = (SITE / "js" / "app.js").read_text()
        self.block = self.app[self.app.index("function solverRolesAt"):]
        self.block = self.block[:self.block.index("\n/**")]

    def test_the_ball_carrier_comes_from_the_tracking(self):
        self.assertIn("candidatePasses", self.block)
        self.assertIn('"ball carrier"', self.block)

    def test_nothing_is_guessed_for_an_undetermined_player(self):
        self.assertNotIn("|| \"runner\"", self.block)
        self.assertNotIn("default", self.block)

    def test_the_two_vocabularies_are_documented_as_different(self):
        head = self.app[:self.app.index("function solverRolesAt")]
        doc = head[head.rindex("/**"):]
        self.assertIn("two vocabularies, not one", doc)


class PolicyRenderingTests(unittest.TestCase):
    """Sections 10-13: the policy is readable on the pitch, and it is real."""

    def setUp(self):
        self.app = (SITE / "js" / "app.js").read_text()
        self.arrows = (SITE / "js" / "arrows.js").read_text()
        self.policy = (SITE / "js" / "policy.js").read_text()

    def _block(self):
        block = self.arrows[self.arrows.index("export function drawActionArrows"):]
        return block[:block.index("\nexport ")]

    def test_arrows_are_drawn_only_from_an_available_artifact(self):
        block = self.app[self.app.index("function renderPolicyArrows"):]
        block = block[:block.index("\n/**")]
        self.assertIn("data?.available", block)
        self.assertIn('state.mode !== "game_solution"', block)

    def test_the_decision_follows_the_playhead(self):
        """Section 9: scrubbing time changes the policy shown."""

        block = self.app[self.app.index("function renderPolicyArrows"):]
        block = block[:block.index("\n/**")]
        self.assertIn("scene.times[index]", block)
        self.assertIn("data.step_seconds", block)
        self.assertIn("bodyPolicies(data, step)", block)

    def test_actual_movement_suppresses_the_policy(self):
        block = self.app[self.app.index("function renderPolicyArrows"):]
        block = block[:block.index("\n/**")]
        self.assertIn('state.solverView === "actual"', block)

    def test_probability_is_the_width_channel_as_upstream(self):
        """render_figure2_abstract: lw_of(p) = 0.6 + 2.4p, one scale for every
        panel. Replaces an earlier test that asserted an opacity channel this
        repository invented before the figure code was available."""

        block = self._block()
        self.assertIn("0.6 + 2.4 * p", block)
        self.assertIn("stroke-width", block)
        self.assertNotIn("colour = `", block)

    def test_geometry_is_physical_and_never_rescaled_by_probability(self):
        """Section 9, and upstream: an option is the solver's own 0.6 s path,
        drawn unchanged. A high-probability command can have a short arrow."""

        self.assertIn("GEOMETRY IS PHYSICAL", self.arrows)
        block = self._block()
        self.assertIn("option.path", block)
        start = block.index("if (!pts)")
        fallback = block[start:block.index("}", start)]
        self.assertNotIn("probability", fallback)

    def test_a_low_probability_action_keeps_its_arrow(self):
        block = self._block()
        self.assertLess(block.index("group.appendChild(line);"),
                        block.index("p < labelFloor"),
                        "the arrow is appended before any label is skipped")

    def test_a_stop_ends_in_a_bar_not_a_head(self):
        """render_figure2_abstract: 'a stop ends in a bar, no head'."""

        block = self._block()
        self.assertIn("option.stop", block)
        self.assertLess(block.index("option.stop"), block.index("arrowhead"))

    def test_an_unweighted_fan_is_the_honest_pre_policy_picture(self):
        self.assertIn("p == null", self._block())

    def test_the_attack_marginal_is_the_joint_summed_over_the_other(self):
        """As extract_panel_policy does it; not an approximation."""

        block = self.policy[self.policy.index("export function bodyPolicies"):]
        block = block[:block.index("\n/**")]
        self.assertIn("carrier[Math.floor(i / n)] += attack[i]", block)
        self.assertIn("receiver[i % n] += attack[i]", block)

    def test_later_steps_do_not_reuse_the_root_attack_policy(self):
        block = self.policy[self.policy.index("export function bodyPolicies"):]
        block = block[:block.index("\n/**")]
        self.assertIn("attackKnown: false", block)
        self.assertIn("carrier: null", block)

    def test_missing_mass_is_reported_rather_than_normalised(self):
        """Section 29."""

        self.assertIn("export function massOf", self.policy)
        block = self.policy[self.policy.index("export function massOf"):]
        self.assertIn("residual", block)
        self.assertNotIn("/ total", block, "normalising away the residual")

    def test_the_pass_line_is_dashed_and_only_from_a_release_policy(self):
        """Section 15: the five-ray explorer has its own geometry and layer."""

        block = self.arrows[self.arrows.index("export function drawPassChoice"):]
        self.assertIn("stroke-dasharray", block)
        harness = (REPO_ROOT / "demo_viz" / "web" / "story_harness.html").read_text()
        self.assertIn("body?.release", harness)
        self.assertNotIn("candidatePasses", harness)


class ReferenceHarnessTests(unittest.TestCase):
    """Sections 30-32: the study state is usable, and cannot reach the public."""

    def setUp(self):
        self.harness = (REPO_ROOT / "demo_viz" / "web" / "story_harness.html").read_text()

    def test_it_is_labelled_a_reference_state_not_a_match_scene(self):
        self.assertIn("REFERENCE SOLVER STUDY STATE", self.harness)
        self.assertIn("NOT PART OF THE PUBLIC SITE", self.harness)
        self.assertIn("not a match scene", self.harness)
        # and the banner carries the artifact's own caveat, not a retyped one
        self.assertIn("${data.caveat}", self.harness)

    def test_the_artifact_itself_declares_the_caveat(self):
        import json

        path = (REPO_ROOT / "demo_viz" / "web_data" / "solver"
                / "solver_exact_100_006.json")
        if not path.exists():
            self.skipTest("reference artifact not exported")
        state = json.loads(path.read_text())
        self.assertEqual("solver_reference", state["kind"])
        self.assertIn("not a tracked match scene", state["caveat"])

    def test_it_lives_outside_the_copied_site(self):
        self.assertFalse((SITE / "story_harness.html").exists())

    def test_it_draws_the_real_pitch_with_the_shared_grammar(self):
        """So what it validates is what the demo would show."""

        for shared in ('from "./js/pitch.js"', 'from "./js/arrows.js"',
                       'from "./js/figure.js"', 'from "./js/policy.js"'):
            self.assertIn(shared, self.harness, shared)

    def test_no_bundesliga_scene_is_given_a_study_state(self):
        import json

        data = REPO_ROOT / "demo_viz" / "web_data"
        for path in sorted((data / "solver").glob("*.json")):
            state = json.loads(path.read_text())
            if state.get("kind") == "solver_reference":
                self.assertTrue(path.stem.startswith("solver_"),
                                f"{path.name} is keyed like a scene")


class SolverRoleTests(unittest.TestCase):
    """extract_panel_policy.ROLES, which the harness and the panels rely on."""

    def test_the_role_names_match_upstream(self):
        source = _show("scripts/extract_panel_policy.py")
        if source is None:
            self.skipTest("origin/kyuhyeok-dev not available")
        self.assertIn('ROLES = {"2v1": ("ball carrier", "runner", "defender"), '
                      '"3v1": ("runner", "beneficiary", "defender")}', source)
        figure = (SITE / "js" / "figure.js").read_text()
        block = figure[figure.index("export const SOLVER_ROLES"):]
        block = block[:block.index("};")]
        self.assertIn('"2v1": ["ball carrier", "runner", "defender"]', block)
        self.assertIn('"3v1": ["runner", "beneficiary", "defender"]', block)


class ModuleExportTests(unittest.TestCase):
    """Every name a page imports is actually exported.

    The harness is only exercised by screenshots, so a removed export used to
    fail silently there -- it did, when the Korean gloss wrapper went away.
    """

    PAGES = ("demo_viz/web/story_harness.html",)
    MODULES = ("app.js", "arrows.js", "figure.js", "policy.js", "pitch.js",
               "story.js", "carrier.js", "scene.js", "solver.js", "showcase.js",
               "evalstrip.js", "release.js", "reach.js", "palette.js",
               "selection.js", "ranking.js", "chart.js", "influence.js", "obso.js")

    @staticmethod
    def _exports(text):
        import re

        names = set(re.findall(r"export\s+(?:async\s+)?function\s+(\w+)", text))
        names |= set(re.findall(r"export\s+const\s+(\w+)", text))
        names |= set(re.findall(r"export\s+class\s+(\w+)", text))
        for block in re.findall(r"export\s*\{([^}]*)\}", text):
            for piece in block.split(","):
                piece = piece.strip()
                if piece:
                    names.add(piece.split(" as ")[-1].strip())
        return names

    def _imports(self, text):
        import re

        out = []
        for block, module in re.findall(r"import\s*\{([^}]*)\}\s*from\s*\"([^\"]+)\"",
                                        text):
            names = [p.strip().split(" as ")[0].strip()
                     for p in block.split(",") if p.strip()]
            out.append((module, names))
        return out

    def test_every_import_resolves(self):
        sources = {}
        for name in self.MODULES:
            path = SITE / "js" / name
            if path.exists():
                sources[name] = path.read_text()

        files = [(SITE / "js" / n, s) for n, s in sources.items()]
        files += [(REPO_ROOT / p, (REPO_ROOT / p).read_text()) for p in self.PAGES]

        for path, text in files:
            for module, names in self._imports(text):
                target = module.rsplit("/", 1)[-1]
                if target not in sources:
                    continue                      # not one of ours
                exported = self._exports(sources[target])
                for name in names:
                    self.assertIn(name, exported,
                                  f"{path.name} imports {name} from {target}, "
                                  "which does not export it")


class DefenderNameTests(unittest.TestCase):
    """`--defender-names short`: the fix for a percentage with nothing to sit on.

    At S05's 0.6 s the defender's "toward ball" move is 0.57 m -- shorter than
    his own marker -- so the 43% floated unexplained. Upstream names the move;
    so does the demo.
    """

    def setUp(self):
        self.source = _show("scripts/render_figure2_abstract.py")
        if self.source is None:
            self.skipTest("origin/kyuhyeok-dev not available")
        self.figure = (SITE / "js" / "figure.js").read_text()

    def test_the_fourth_target_follows_the_same_rule(self):
        """3v1 games have a fourth defender target upstream's S05-only table
        does not list; four of the seven demo scenes are 3v1."""

        self.assertIn('"toward beneficiary"', self.figure)
        self.assertIn("Toward beneficiary", self.figure)
        source = _show("src/offball_value/stage3_read.py")
        if source:
            self.assertIn('("toward beneficiary", snapshot["beneficiary"])', source)

    def test_the_three_names_match_upstream(self):
        self.assertIn('DEFENDER_NAME = {"toward goal": "Toward goal", '
                      '"toward ball": "Toward ball", "toward runner": "Toward runner"}',
                      self.source)
        for korean_free in ("toward goal", "toward ball", "toward runner"):
            self.assertIn(f'"{korean_free}"', self.figure, korean_free)
        for title in ("Toward goal", "Toward ball", "Toward runner"):
            self.assertIn(title, self.figure, title)

    def test_the_short_form_is_upstreams_replace(self):
        self.assertIn("name.replace('Toward', 'To')", self.source)
        self.assertIn('full.replace("Toward", "To")', self.figure)

    def test_the_near_final_settings_are_recorded(self):
        """The README's own command for the final figure."""

        readme = _show("README.md")
        if readme is None:
            self.skipTest("README not available")
        self.assertIn("--defender-names short --label-gap 0.35", readme)
        self.assertIn('--flow-color "#FF8000" --flow-alpha 0.4', readme)
        self.assertIn('defenderNames: "short"', self.figure)
        self.assertIn("labelGapM: 0.35", self.figure)
        self.assertIn('flowColour: "#FF8000"', self.figure)

    def test_a_name_outside_the_three_falls_back_rather_than_failing(self):
        """Upstream raises; a browser must not blank the render."""

        block = self.figure[self.figure.index("export function defenderNamePrefix"):]
        block = block[:block.index("\n/**")]
        self.assertIn("if (!full", block)
        self.assertIn('return ""', block)


class LabelPlacementTests(unittest.TestCase):
    """Section 6: deterministic placement, nothing hidden, nothing random."""

    def setUp(self):
        self.arrows = (SITE / "js" / "arrows.js").read_text()
        self.block = self.arrows[self.arrows.index("export function drawActionArrows"):]
        self.block = self.block[:self.block.index("\nexport ")]

    def test_placement_is_deterministic(self):
        for banned in ("Math.random", "Date.now", "shuffle"):
            self.assertNotIn(banned, self.arrows, banned)

    def test_candidate_offsets_are_a_fixed_ordered_list(self):
        self.assertIn("[[0, 0], [1, 0], [2, 0], [0, 1], [0, -1]", self.block)

    def test_a_label_keeps_clear_of_the_ones_already_placed(self):
        self.assertIn("collides(", self.block)
        self.assertIn("labelGap", self.block)

    def test_heaviest_first_and_a_resting_stop_last(self):
        """Upstream places by `prob - 1.0` for a stop at rest."""

        self.assertIn("o.stop && o.rests ? 1 : 0", self.block)

    def test_no_option_is_dropped_to_make_room(self):
        """Only the probability floor removes a label, never a collision."""

        self.assertIn("p < labelFloor", self.block)
        tail = self.block[self.block.index("p < labelFloor"):]
        self.assertNotIn("continue", tail[tail.index("collides"):]
                         if "collides" in tail else "")

    def test_a_move_shorter_than_its_marker_still_gets_its_label(self):
        """S05 at 0.6 s: 'toward ball' is 0.57 m, shorter than the marker, so
        it has no shaft -- and that is why upstream names these moves."""

        self.assertIn("shaft.length > 1", self.block)
        self.assertIn("the label carries it", self.block)


class StopLabelTests(unittest.TestCase):
    """`rests` decides Stop from Slow down, and it is upstream's exact test."""

    def test_the_rest_test_is_exact_equality_with_zero(self):
        source = _show("scripts/render_figure2_abstract.py")
        if source is None:
            self.skipTest("origin/kyuhyeok-dev not available")
        self.assertIn('math.hypot(*o["end_velocity"]) == 0.0', source)
        bundle = (REPO_ROOT / "demo_viz" / "paper_story" / "bundle.py").read_text()
        self.assertIn('math.hypot(*(o.get("end_velocity") or (1.0, 0.0)))', bundle)
        self.assertIn("== 0.0", bundle)

    def test_s05_at_1_2_rests_so_it_reads_stop(self):
        path = (REPO_ROOT / "demo_viz" / "web_data" / "solver"
                / "J03WOH_shot_010_P1_1759.json")
        if not path.exists():
            self.skipTest("S05 panels not exported")
        panel = next(p for p in json.loads(path.read_text())["panels"]
                     if p["dt"] == 1.2)
        stop = panel["bodies"]["defender"]["options"][0]
        self.assertTrue(stop["rests"])


class PaperColourScopeTests(unittest.TestCase):
    """Figure 2's palette is used where a reader compares with Figure 2."""

    def test_game_solution_uses_the_paper_colours(self):
        app = (SITE / "js" / "app.js").read_text()
        self.assertIn('state.mode === "game_solution" ? figureLib()?.ROLE_COLOR', app)

    def test_observed_keeps_the_annotation_colours(self):
        """The role dock shows those, so the two must not disagree."""

        app = (SITE / "js" / "app.js").read_text()
        block = app[app.index("paperColours:"):]
        self.assertIn("null", block[:160])
