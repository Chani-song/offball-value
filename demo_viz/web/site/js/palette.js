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
};

export const ROLE_COLOUR = {
  runner: P.runner,
  defender: P.defender,
  beneficiary: P.beneficiary,
};

export const ROLE_SIDE = { runner: "attack", beneficiary: "attack", defender: "defend" };
