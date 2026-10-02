// The paper figures' own conventions, ported from the code that draws them.
//
// SOURCE OF TRUTH
// ---------------
// origin/kyuhyeok-dev@139498a. The figure renderer is in the repository --
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
export function optionLabel(role, probability,
                            { stop = false, rests = false, name = "",
                              defenderNames = "none" } = {}) {
  const pct = `${Math.round(probability * 100)}%`;
  if (stop && !rests) return `Slow down ${pct}`;
  if (stop && rests) return `Stop ${pct}`;
  if (role === "ball carrier") return `Dribble ${pct}`;
  if (role === "defender" && defenderNames !== "none") {
    const prefix = defenderNamePrefix(name, defenderNames);
    if (prefix) return `${prefix} ${pct}`;
  }
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

/**
 * `figure_style` at kyuhyeok-dev@33378ac, the abstract's own rendering.
 *
 *   HEAD_OPEN     every arrowhead an open chevron in the line's own width,
 *                 not a filled triangle; HEAD_LW_MAX caps that width, so a
 *                 heavy line runs on into the chevron instead of thickening it
 *   OTHER_ALPHA   a player outside the game at 50%: at full colour he drew
 *                 the eye, and the 45% tint before that read as blurred
 *   MIN_ARROW_M   a move showing less than this outside its player's marker is
 *                 scaled about its start until it does. Its SHAPE is kept and
 *                 its LENGTH is then not to scale -- upstream's `--min-arrow`,
 *                 for moves like S05's 0.57 m "toward ball" that otherwise
 *                 vanish under the marker. Probability still never touches it.
 */
export const HEAD_OPEN = true;
export const HEAD_LW_MAX = 1.0;
export const OTHER_ALPHA = 0.5;
export const MIN_ARROW_M = 1.1;
export const FIGURE1_OPTIONS = [
  { key: "follow", label: "Follow?", toward: "runner" },
  { key: "stay", label: "Stay?", toward: "carrier" },
];

// ---------------------------------------------------------------------------
// the print style (scripts/figure_style.py)
// ---------------------------------------------------------------------------
/**
 * `figure_style.ATTACK / DEFENCE`, as of 2026-10-01: **pure blue and pure
 * red**, replacing the Okabe-Ito pair, "which read as a stock palette".
 *
 * Both attackers are blue -- the runner and the ball carrier are told apart by
 * marker and by a direct label, so no role rests on colour alone.
 */
export const PAPER = {
  attack: "#0000FF",
  defence: "#FF0000",
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

/**
 * Since 2026-10-01 the three key players carry **no outline** (`KEY_EDGE` went
 * 0.8 -> 0), players outside the game are drawn in their team's **full**
 * colour rather than a 45% tint ("which read as blurred"), and the ball is
 * **white with a charcoal edge** so it shows on the near-white pitch.
 */
export const KEY_EDGE = 0.0;

/**
 * Clear ground between a marker's edge and a move leaving it, in points.
 * `figure_style.MOVE_GAP`, 2026-10-01: "the user found the arrows stuck to the
 * shapes".
 */
export const MOVE_GAP = 2.5;

/**
 * Points from a key player's centre to his marker's edge along `u`.
 *
 * `figure_style.marker_edge`, ported: a disc is its radius in every
 * direction; a square reaches further along a diagonal; a diamond the
 * opposite way.
 */
export function markerEdge(role, u) {
  const entry = ROLE_MARKER[role];
  if (!entry) return KEY_D / 2;
  const [shape, size] = entry;
  const c = Math.abs(u[0]);
  const s = Math.abs(u[1]);
  if (shape === "circle") return size / 2;
  if (shape === "square") return size / 2 / Math.max(c, s, 1e-9);
  if (shape === "diamond") return (size / 2) * Math.SQRT2 / Math.max(c + s, 1e-9);
  return (size / 2) * Math.SQRT2;
}

/**
 * The solver's own name for a defender command, title-cased for a label.
 *
 * `render_figure2_abstract.DEFENDER_NAME`, with its `--defender-names short`
 * form. This exists because of a real readability problem the team hit: at
 * 0.6 s the defender's "toward ball" move is **0.57 m** long, so it hides
 * under his marker and its percentage floats unexplained. Naming the move
 * attaches the number to something a reader can see.
 *
 * Upstream raises on a name outside the three; a browser falls back to the
 * percentage alone instead of failing the render.
 */
export const DEFENDER_NAME = {
  "toward goal": "Toward goal",
  "toward ball": "Toward ball",
  "toward runner": "Toward runner",
  // Upstream's table stops here because the README's final command renders
  // S05, a 2v1 game. `stage3_read.targets_for` gives a 3v1 defender a fourth
  // target, and four of the demo's seven scenes are 3v1 -- S15 at 0.6 s plays
  // it 26% of the time. Title-cased by the same rule rather than left as a
  // bare percentage, which is the very thing naming these moves fixes.
  "toward beneficiary": "Toward beneficiary",
};

export function defenderNamePrefix(name, style = "short") {
  const full = DEFENDER_NAME[name];
  if (!full || style === "none") return "";
  return style === "short" ? full.replace("Toward", "To") : full;
}

/**
 * The near-final Figure 2's own settings, from the README's command:
 *
 *     --defender-names short --label-gap 0.35
 *     --flow-color "#FF8000" --flow-alpha 0.4 --value-fade 0 --frames
 */
export const FIGURE2 = {
  defenderNames: "short",
  labelGapM: 0.35,
  flowColour: "#FF8000",
  flowAlpha: 0.4,
};

/** `figure_style` line weights, in points at the printed size. */
export const STROKE = {
  move: 1.5,        // a move still to come (Figure 1)
  past: 1.0,        // a path already run: thinner and lighter
  pastAlpha: 0.45,
  ball: 1.3,        // the ball's moves (pass, shot), dashed charcoal
  real: 0.8,        // a real tracked move shown for comparison (Figure 2)
};
