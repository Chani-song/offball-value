// The paper figures' own conventions, ported from the code that draws them.
//
// SOURCE OF TRUTH
// ---------------
// origin/kyuhyeok-dev@8a5c69d. The figure renderer is now in the repository --
// the previous pass could not find it and reconstructed parts of this file;
// everything reconstructed has been replaced by the real thing:
//
//   scripts/figure_style.py            the shared print style: face, palette,
//                                      one marker per role, one arrow
//   scripts/render_figure1_dilemma.py  Figure 1 (current state -> follow / stay)
//   scripts/render_figure2_abstract.py Figure 2 (equilibrium choices at the
//                                      real 0.0 / 0.6 / 1.2 s moments)
//   src/offball_value/stage3_read.py   name_move_targets, targets_for,
//                                      world_direction
//
// Two corrections this file used to get wrong, both now upstream-exact:
//   * the action names are English upstream since the 2026-09-30 cleanup, so
//     the Korean gloss table is gone. "slow down", not "brake"; "toward ball",
//     not "toward the ball".
//   * an option's arrow is the solver's own 0.6 s PATH. Probability sets the
//     line WIDTH and nothing else. The old dark figure scaled length by
//     probability; render_figure2_abstract does not, and neither does this.

// ---------------------------------------------------------------------------
// geometry: stage3_read.SOLVER_DIRS / world_direction
// ---------------------------------------------------------------------------
/** The five compass commands, in the order the solver indexes them. */
export const SOLVER_DIRS = [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]];

/**
 * Compass command `k` as a direction on the pitch.
 *
 * `stage3_read.world_direction`: the commands are relative to the attack, so a
 * scene attacking toward decreasing x has every one turned half a circle.
 */
export function worldDirection(k, attackDirection) {
  const [ux, uy] = SOLVER_DIRS[k];
  return [ux * attackDirection, uy * attackDirection];
}

// ---------------------------------------------------------------------------
// action names (stage3_read, English upstream)
// ---------------------------------------------------------------------------
/** The cosine a move must reach before it is named for a target. */
export const NAME_COSINE = 0.3;

/**
 * Name a defender's compass move for whichever target it points along most.
 *
 * `stage3_read.name_move_targets`, ported exactly:
 *   - `(0, 0)` is **"slow down"**, not "stand" -- a running defender keeps
 *     sliding the same way while he slows;
 *   - otherwise the target with the largest cosine that reaches 0.3;
 *   - ties go to the *later* target, "as they always have";
 *   - "sideways" when nothing reaches 0.3.
 */
export function defenderMoveName(u, defender, targets) {
  if (u[0] === 0 && u[1] === 0) return "slow down";
  let best = "sideways";
  let score = NAME_COSINE;
  for (const [name, target] of targets) {
    const vx = target[0] - defender[0];
    const vy = target[1] - defender[1];
    const n = Math.hypot(vx, vy);
    if (n > 1e-6) {
      const c = (u[0] * vx + u[1] * vy) / n;
      if (c >= score) { best = name; score = c; }   // >= : ties to the later
    }
  }
  return best;
}

/**
 * `stage3_read.targets_for` for the 2v1 game, in the order ties resolve. In
 * 2v1 the ball carrier *is* the beneficiary, so "toward ball" covers him.
 */
export function targetsFor(carrier, runner, goal) {
  return [["toward ball", carrier], ["toward runner", runner], ["toward goal", goal]];
}

/**
 * `extract_panel_policy.ROLES`: the solver's own role names per game kind, by
 * slot. These are **solver** roles and are not the demo's annotation roles.
 */
export const SOLVER_ROLES = {
  "2v1": ["ball carrier", "runner", "defender"],
  "3v1": ["runner", "beneficiary", "defender"],
};

/** `extract_panel_policy.compass_name`: an attacker's move, named against the attack. */
export function compassName(u, attackDirection) {
  const x = u[0] * attackDirection;
  const y = u[1] * attackDirection;
  if (Math.abs(x) < 1e-9 && Math.abs(y) < 1e-9) return "stop";
  if (Math.abs(x) >= Math.abs(y)) return x > 0 ? "forward" : "back";
  return y > 0 ? "left" : "right";
}

// ---------------------------------------------------------------------------
// Figure 2: what an option is labelled
// ---------------------------------------------------------------------------
/** `render_figure2_abstract.MIN_P` (from render_panel_figure): options below
 *  2% are not drawn. Their mass stays in the side panel. */
export const MIN_P = 0.02;

/**
 * An option's label, exactly as `render_figure2_abstract` builds it.
 *
 *   base            "NN%"
 *   stop, moving    "Slow down NN%"   braking but still moving at 0.6 s
 *   stop, at rest   "Stop NN%"        reached speed 0 inside the 0.6 s
 *   ball carrier    "Dribble NN%"
 *   everyone else   the percentage only -- the direction is the arrow's job
 */
export function optionLabel(role, probability, { stop = false, rests = false } = {}) {
  const pct = `${Math.round(probability * 100)}%`;
  if (stop && !rests) return `Slow down ${pct}`;
  if (stop && rests) return `Stop ${pct}`;
  if (role === "ball carrier") return `Dribble ${pct}`;
  return pct;
}

/**
 * A pass's label. A pass column is ONE joint attack option -- the passer's
 * release and the receiver's run onto it are the same probability -- so it is
 * labelled once, where the two meet.
 */
export function passLabel(probability, { received = true } = {}) {
  const pct = `${Math.round(probability * 100)}%`;
  return received ? `Pass + receive · ${pct}` : `Pass ${pct}`;
}

/** `render_figure2_abstract.lw_of`: one width scale for every panel. */
export function lwOf(probability) {
  return 0.6 + 2.4 * Math.max(0, Math.min(1, probability || 0));
}

/** The figure's own title. */
export const FIGURE2_TITLE = "Equilibrium choices during an off-ball play";

// ---------------------------------------------------------------------------
// Figure 1: the dilemma
// ---------------------------------------------------------------------------
/**
 * The two option arrows Figure 1 draws from the defender, and their labels.
 *
 * `render_figure1_dilemma`: they point at where the runner and the ball
 * carrier **really were 0.6 s later**, and both are **3 m long** -- the
 * docstring is explicit that this is a "picture choice, not data". Figure 1
 * carries no numbers by design.
 */
export const OPTION_M = 3.0;
export const FIGURE1_OPTIONS = [
  { key: "follow", label: "Follow?", toward: "runner" },
  { key: "stay", label: "Stay?", toward: "carrier" },
];

// ---------------------------------------------------------------------------
// the print style (scripts/figure_style.py)
// ---------------------------------------------------------------------------
/**
 * The Okabe-Ito pair the team chose, told apart under every common
 * colour-vision deficiency. Both attackers are blue: the runner and the ball
 * carrier are distinguished by marker and by a direct label, so no role rests
 * on colour alone.
 */
export const PAPER = {
  attack: "#0072B2",
  defence: "#D55E00",
  ink: "#222222",
  muted: "#666666",
  faint: "#B3B3B3",
  line: "#D0D0D0",
  page: "#FFFFFF",
  pitch: "#F7F7F7",
};

/** `figure_style.tint`: a team colour at 45% on white, for players outside the game. */
export function tint(hex, k = 0.45) {
  const n = parseInt(hex.slice(1), 16);
  const ch = [(n >> 16) & 255, (n >> 8) & 255, n & 255]
    .map((v) => Math.round(255 * (1 - k) + k * v));
  return `#${ch.map((v) => v.toString(16).padStart(2, "0")).join("")}`;
}

export const ROLE_COLOR = {
  runner: PAPER.attack,
  "ball carrier": PAPER.attack,
  beneficiary: PAPER.attack,
  teammate: PAPER.attack,
  defender: PAPER.defence,
};

/** `figure_style.ROLE_NAME`: what a role is called once, in the first panel. */
export const ROLE_NAME = {
  runner: "Runner",
  "ball carrier": "Ball carrier",
  beneficiary: "Ball carrier",
  teammate: "Teammate",
  defender: "Defender",
};

/**
 * `figure_style.ROLE_MARKER`, with its own note: the sizes are chosen so the
 * three shapes cover about the same area. The triangle is the **teammate** --
 * the second attacker in a 3v1 game, when a scripted passer has the ball --
 * and not the beneficiary, who takes the ball carrier's disc.
 */
export const KEY_D = 8.5;
export const ROLE_MARKER = {
  runner: ["diamond", 0.80 * KEY_D],
  "ball carrier": ["circle", KEY_D],
  beneficiary: ["circle", KEY_D],
  defender: ["square", 0.86 * KEY_D],
  teammate: ["triangle", 1.15 * KEY_D],
};

/** Shape alone, for callers that size markers themselves. */
export const ROLE_SHAPE = Object.fromEntries(
  Object.entries(ROLE_MARKER).map(([role, [shape]]) => [role, shape]));

/** `figure_style` line weights, in points at the printed size. */
export const STROKE = {
  move: 1.5,        // a move still to come (Figure 1)
  past: 1.0,        // a path already run: thinner and lighter
  pastAlpha: 0.45,
  ball: 1.3,        // the ball's moves (pass, shot), dashed charcoal
  real: 0.8,        // a real tracked move shown for comparison (Figure 2)
};
