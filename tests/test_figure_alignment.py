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
        return "멈추기(감속)"
    best, score = "옆으로", 0.3
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
        self.assertIn('best, score = "옆으로", 0.3', source)
        self.assertIn('return "멈추기(감속)"', source,
                      "a zero command is brake, not stand")
        self.assertIn('("볼 쪽", snapshot["carrier"]), ("러너 쪽", snapshot["receiver"])',
                      source)

    def test_the_abstracts_runs_use_the_compass_command_set(self):
        """Which decides which decoder the figures are showing."""

        for job in ("deploy/delta/eval_v1.sbatch",
                    "deploy/delta/multipass_full_2v1.sbatch"):
            source = _show(job)
            if source is None:
                self.skipTest("origin/kyuhyeok-dev not available")
            self.assertIn("--commands compass", source, job)


class GlossTests(unittest.TestCase):
    """Every name the research code can emit has exactly one English gloss."""

    def setUp(self):
        self.figure = (SITE / "js" / "figure.js").read_text()

    def test_every_defender_name_is_glossed(self):
        for korean in ("볼 쪽", "러너 쪽", "수혜자 쪽", "골문 쪽",
                       "옆으로", "멈추기(감속)"):
            self.assertIn(f'"{korean}"', self.figure, korean)

    def test_the_gloss_is_marked_as_this_repository_s_own(self):
        block = self.figure[self.figure.index("DEFENDER_MOVE_GLOSS"):]
        head = self.figure[:self.figure.index("DEFENDER_MOVE_GLOSS")]
        self.assertIn("gloss is this file's, not the research code's",
                      head + block[:400])

    def test_the_missing_renderer_is_stated_at_the_top(self):
        """So nobody later mistakes the house style for a recovered one."""

        head = " ".join(self.figure[:2000].replace("//", " ").split())
        self.assertIn("figure *renderer* is not in this repository", head)
        self.assertIn("could NOT be recovered from source", head)


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
            targets = (("볼 쪽", c["carrier"]), ("러너 쪽", c["runner"]),
                       ("골문 쪽", c["goal"]))
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

        block = text[text.index("ROLE_SHAPE = {"):]
        block = block[:block.index("}")]
        return {k.strip(): v
                for k, v in re.findall(r'"?([a-z ]+)"?:\s*"([a-z]+)"', block)}

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

    def test_probability_drives_width_and_opacity_not_colour(self):
        """Section 12: a 3% action must not look like an 83% one, and colour
        must not be the channel that says so."""

        block = self.arrows[self.arrows.index("export function drawActionArrows"):]
        block = block[:block.index("\nexport ")]
        self.assertIn("stroke-width", block)
        self.assertIn("opacity", block)
        # the colour is an argument, fixed per actor, never derived from p
        self.assertNotIn("colour =", block.split("{", 1)[1].split("for (")[0]
                         .replace("colour = P.defender", ""))

    def test_a_low_probability_action_keeps_its_arrow(self):
        block = self.arrows[self.arrows.index("export function drawActionArrows"):]
        block = block[:block.index("\nexport ")]
        # only the *label* is dropped below the floor; the arrow is always drawn
        self.assertIn("if (!labels || (p != null && p < labelFloor)) continue;", block)
        self.assertLess(block.index("group.appendChild(path);"),
                        block.index("p < labelFloor"),
                        "the arrow must be appended before the label is skipped")

    def test_an_unweighted_fan_is_the_honest_pre_policy_picture(self):
        block = self.arrows[self.arrows.index("export function drawActionArrows"):]
        block = block[:block.index("\nexport ")]
        self.assertIn("p == null", block)

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
