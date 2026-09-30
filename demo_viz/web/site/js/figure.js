// The paper figures' own conventions, ported from the code that produces them.
//
// PROVENANCE, AND ITS LIMIT
// ------------------------
// The figure *renderer* is not in this repository, in any branch, or anywhere
// on this machine -- searched for its own captions ("Pass + receive",
// "Follow the runner", "Equilibrium choices") across every ref and the
// filesystem. So its colours, marker shapes and English action captions could
// NOT be recovered from source.
//
// What *is* authoritative, and is ported exactly here, is the figure's data
// producer and its decoders, at origin/kyuhyeok-dev@e84553a:
//
//   scripts/extract_panel_policy.py   "One game's opening decision, as a
//                                      figure panel needs it" -- the panel
//                                      contract, compass_name, pass_where
//   src/offball_value/stage3_read.py  world_direction, name_move_targets,
//                                      targets_for
//
// Those give the role names, the action names, the probabilities and the
// geometry -- the semantics a reviewer must recognise. The marker/colour
// constants below come from the brief's description of the figures and are
// marked as such; they are a house style, not a recovered one.

// ---------------------------------------------------------------------------
// geometry: stage3_read.SOLVER_DIRS / world_direction
// ---------------------------------------------------------------------------
/** The five compass commands, in the order the solver indexes them. */
export const SOLVER_DIRS = [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]];

/**
 * Compass command `k` as a direction on the pitch.
 *
 * `stage3_read.world_direction`: the solver's commands are relative to the
 * attack, so a scene attacking toward decreasing x has every command turned
 * half a circle. Reading them as absolute gets those scenes backwards -- the
 * source notes the unturned reading agreed with the actual displacement 6% of
 * the time on such scenes.
 */
export function worldDirection(k, attackDirection) {
  const [ux, uy] = SOLVER_DIRS[k];
  return [ux * attackDirection, uy * attackDirection];
}

// ---------------------------------------------------------------------------
// action names
// ---------------------------------------------------------------------------
/**
 * An attacker's compass move, named against the attack.
 *
 * `extract_panel_policy.compass_name`, verbatim: attack runs +x after the
 * turn, the larger component wins, and a zero vector is "stop".
 */
export function compassName(u, attackDirection) {
  const x = u[0] * attackDirection;
  const y = u[1] * attackDirection;
  if (Math.abs(x) < 1e-9 && Math.abs(y) < 1e-9) return "stop";
  if (Math.abs(x) >= Math.abs(y)) return x > 0 ? "forward" : "back";
  return y > 0 ? "left" : "right";
}

/**
 * The Korean names `stage3_read` gives a defender's move, and the English the
 * demo shows.
 *
 * The gloss is this file's, not the research code's -- the research code is
 * Korean throughout. The keys are exact so a reader can grep either side.
 */
export const DEFENDER_MOVE_GLOSS = {
  "볼 쪽": "toward the ball",
  "러너 쪽": "toward the runner",
  "수혜자 쪽": "toward the beneficiary",
  "골문 쪽": "toward goal",
  "옆으로": "sideways",
  "멈추기(감속)": "brake",
};

/** The cosine a move must reach before it is named for a target. */
export const NAME_COSINE = 0.3;

/**
 * Name a defender's compass move for whichever target it points along most.
 *
 * `stage3_read.name_move_targets`, ported exactly:
 *   - `(0, 0)` is **brake**, not "stand" -- a running defender keeps sliding
 *     the same way while he slows;
 *   - otherwise the target with the largest cosine that reaches 0.3;
 *   - ties go to the *later* target, "as they always have";
 *   - "sideways" when nothing reaches 0.3.
 *
 * `targets` is ordered `[name, [x, y]]` pairs. `stage3_read.targets_for` gives
 * 2v1 the ball carrier, the runner and the goal; 3v1 the passer, the runner,
 * the beneficiary and the goal.
 */
export function defenderMoveName(u, defender, targets) {
  if (u[0] === 0 && u[1] === 0) return "멈추기(감속)";
  let best = "옆으로";
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

/** The same, already glossed for the interface. */
export function defenderMoveLabel(u, defender, targets) {
  const name = defenderMoveName(u, defender, targets);
  return DEFENDER_MOVE_GLOSS[name] || name;
}

/**
 * `stage3_read.targets_for` for the 2v1 game: what a defender's move can be
 * toward, in the order ties resolve. In 2v1 the ball carrier *is* the
 * beneficiary, so "toward the ball" covers him.
 */
export function targetsFor(carrier, runner, goal) {
  return [["볼 쪽", carrier], ["러너 쪽", runner], ["골문 쪽", goal]];
}

/**
 * Where a pass is aimed, in the figure's words.
 *
 * `extract_panel_policy.pass_where`: an axis pass reads "4 m ahead", "4 m
 * left", "feet"; a run pass reads "run +8 / side 0 / goal 0".
 */
export function passWhere(choice) {
  if (!choice) return "";
  if (choice.along !== undefined) {
    const g = (v) => (Number.isInteger(v) ? String(v) : String(v));
    return `run ${choice.along >= 0 ? "+" : ""}${g(choice.along)} / `
         + `side ${g(choice.lateral)} / goal ${g(choice.goalward)}`;
  }
  const [along, side] = choice.offset || [0, 0];
  const parts = [];
  if (along) parts.push(`${Math.abs(along)} m ${along > 0 ? "ahead" : "behind"}`);
  if (side) parts.push(`${Math.abs(side)} m ${side > 0 ? "left" : "right"}`);
  return parts.join(" ") || "feet";
}

/**
 * The solver's own role names per game kind (`extract_panel_policy.ROLES`),
 * by slot. These are **solver roles**, and they are not the demo's
 * human-annotation roles -- see PAPER_FIGURE_ALIGNMENT.md.
 */
export const SOLVER_ROLES = {
  "2v1": ["ball carrier", "runner", "defender"],
  "3v1": ["runner", "beneficiary", "defender"],
};

// ---------------------------------------------------------------------------
// house style (from the brief; NOT recovered from figure source)
// ---------------------------------------------------------------------------
/**
 * Marker shape per solver role. Shape carries the role so identity survives
 * greyscale and colour-vision deficiency; colour only reinforces it.
 */
export const ROLE_SHAPE = {
  "ball carrier": "circle",
  runner: "diamond",
  beneficiary: "triangle",
  teammate: "triangle",
  defender: "square",
};

/** Probability -> a visual weight that is never colour alone. */
export function arrowWeight(probability) {
  const p = Math.max(0, Math.min(1, probability || 0));
  return { width: 1.4 + 4.6 * p, opacity: 0.35 + 0.65 * p };
}

/** Actions worth a label on the pitch; the rest keep their mass in the panel. */
export const LABEL_FLOOR = 0.10;
