// Role colours, matching demo_viz/palette.py. Green is never a highlight
// because the pitch is green, and every role also carries a size change and a
// label so identity is never colour alone.
export const P = {
  ink: "#05090A",
  panel: "#0C1315",
  pitch: "#132019",
  pitchBand: "#17271E",
  line: "#9FB5A6",
  grid: "#24352C",
  text: "#F4F6F3",
  text2: "#A9B6AE",
  muted: "#6B7A72",
  runner: "#FF2D87",
  defender: "#FFA62B",
  beneficiary: "#5CE8F5",
  attack: "#D9CFB8",
  defend: "#5F7392",
  ball: "#FFFFFF",
  obso: "#9B8CFF",
  reach: "#8FA6B8",
  solver: "#7BE8A8",
};

/**
 * The paper's own values, for the Submission showcase.
 *
 * `scripts/figure_style.py` @ kyuhyeok-dev 139498a: the pitch is a near-white
 * ground with thin light-grey lines ("a hair off white, so a panel reads as a
 * panel with no frame"), attack pure blue and defence pure red, charcoal text,
 * a white ball with a charcoal edge. The Full explorer keeps the dark research
 * palette above -- two audiences, two themes.
 */
export const PAPER_THEME = {
  ink: "#222222",            // figure_style.INK: text, and the ball's edge
  panel: "#FFFFFF",          // figure_style.PAGE
  pitch: "#F7F7F7",          // figure_style.PITCH
  pitchBand: "#F7F7F7",      // no banding in the figures
  line: "#D0D0D0",           // figure_style.LINE, drawn at lw 0.6
  grid: "#E4E4E4",
  text: "#222222",
  text2: "#666666",          // figure_style.MUTED
  muted: "#B3B3B3",          // figure_style.FAINT
  attack: "#0000FF",         // figure_style.ATTACK
  defend: "#FF0000",         // figure_style.DEFENCE
  ball: "#FFFFFF",
  // the diagnostic washes, restated in print values: a grey reachable set
  // (figure_style.BG_LINE) and a blue threat, so neither reads as neon
  reach: "#8C8C8C",
  obso: "#0000FF",
  solver: "#222222",
};

/**
 * The same three roles in the paper's grammar: both attackers blue, the
 * defender red. Shape carries the role (figure_style.ROLE_MARKER), so colour
 * never has to.
 */
export const ROLE_PAPER = {
  runner: "#0000FF",
  beneficiary: "#0000FF",
  defender: "#FF0000",
};

export const ROLE_COLOUR = {
  runner: P.runner,
  defender: P.defender,
  beneficiary: P.beneficiary,
};

export const ROLE_SIDE = { runner: "attack", beneficiary: "attack", defender: "defend" };

export const ROLE_LABEL = { runner: "Runner", beneficiary: "Beneficiary", defender: "Defender" };

/** User-facing names for the overlay layers; the internal keys stay as they are. */
export const LAYER_LABEL = {
  trail: "Runner movement",
  tether: "Defender response",
  wake: "Space map",
  lane: "Passing lane",
  ghost: "Defender if stayed",
  labels: "Player numbers",
  candidates: "Suggested players",
  paths: "Player movements",
  commands: "Defender command set (5)",
  passes: "Explore action value",
  reach: "Kinematic reachable area",
  solver: "Solver solution",
};
