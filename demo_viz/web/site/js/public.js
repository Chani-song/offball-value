// The Submission showcase's words.
//
// A display layer and nothing else: every name here is a translation of a
// value the research code produced, made for someone who watches football
// rather than someone who reads the pipeline. The solver's own labels stay
// exactly as they are and keep being what the Full explorer shows, what the
// panel payload carries and what the tests pin -- this module never reaches
// back into a computation.
//
// The rule for adding anything here: if the football word would say something
// the solver did not, it does not go in. A command the model expresses as a
// direction gets a football verb for that direction, not a football intention.

/** The four story modes, named for a public audience. */
export const MODE_NAME = {
  observed: "Play",
  counterfactual: "Dilemma",
  evaluation: "Player evaluation",
  game_solution: "Nash equilibrium",
};

/** The card title above the sidebar's story, in the same voice. */
export const MODE_TITLE = {
  observed: "The play",
  counterfactual: "The dilemma",
  evaluation: "Player evaluation",
  game_solution: "Nash equilibrium",
};

/**
 * A solver role as a football position in this play.
 *
 * `ball carrier` is the body the model gives the ball to, so "On ball" is
 * literally what it is. `runner` is the off-ball attacker whose movement the
 * play is about. `beneficiary` is the 3v1's second strategic attacker --
 * fixedpasser/game.py gives him his own five commands and lets the scripted
 * passer release to him, so he is a player in the play, not a label on it.
 */
export const PUBLIC_ROLE = {
  "ball carrier": "On ball",
  runner: "Runner",
  // the 3v1's second strategic attacker: the teammate standing in the space
  // the run opens, and the other man the passer may pick. The paper figure
  // draws him as the "teammate" (render_figure2_abstract.shown_as), so the
  // showcase says the same word.
  beneficiary: "Teammate",
  defender: "Defender",
};

/** The story-role picker's three options, in the same words. */
export const PUBLIC_STORY_ROLE = {
  passer: "On ball",
  runner: "Runner",
  defender: "Defender",
};

/** An integer rank as a football audience writes it: 1st, 2nd, 3rd, 4th. */
export function ordinal(n) {
  if (!Number.isFinite(n)) return null;
  const i = Math.round(n);
  const tens = i % 100;
  if (tens >= 11 && tens <= 13) return `${i}th`;
  return `${i}${({ 1: "st", 2: "nd", 3: "rd" })[i % 10] || "th"}`;
}

/** A probability as a percentage, for a reader who does not want 3 decimals. */
export function percent(p) {
  if (!Number.isFinite(p)) return null;
  return `${Math.round(p * 100)}%`;
}

/** The curated row's own outcome word, as a match report would print it. */
const OUTCOME = {
  goal: "Goal",
  saved: "Shot saved",
  blocked: "Shot blocked",
  "off target": "Shot off target",
  woodwork: "Hit the woodwork",
};

/**
 * The scene's one-line caption.
 *
 * The exported subtitle is `clock · shot N · shooter · OUTCOME`, then `   ·   `
 * and a reviewer's sentence where one was written. That sentence is the only
 * part worth a public caption; the outcome stands in when there is none, and
 * the shot number and identifiers belong in Details.
 */
export function publicSubtitle(scene, curated) {
  const clock = (scene.match_clock || "").trim();
  const headline = (scene.subtitle || "").split("   ·   ")[1];
  if (headline) return [clock, headline.trim()].filter(Boolean).join(" · ");
  const event = ((curated && curated.event) || "").split("·")[0].trim();
  return [clock, OUTCOME[event]].filter(Boolean).join(" · ");
}

/**
 * The case-study selector, without the identifier and the review badge.
 *
 * The clock comes with it: several curated plays are the same fixture, so the
 * fixture alone would list the same line four times.
 */
export function publicSelectorLabel(scene, clock) {
  const fixture = (scene.match || "").split("·")[0].trim();
  const label = [scene.story_title || fixture, clock].filter(Boolean)
    .join(" · ");
  return scene.playable ? label : `${label} — not available`;
}

// ---------------------------------------------------------------------------
// The football words for the solver's commands
// ---------------------------------------------------------------------------
/*
 * The football words below are grounded in what the commands are, not in what
 * their names suggest. PAPER_FIGURE_ALIGNMENT.md section 10 has the trace,
 * with file:line in the research code; the three facts that decide the words:
 *
 *   - the solved studies run `--commands compass`: one five-element set
 *     shared by every body, stated against the attack direction;
 *   - the defender's names are per-state descriptions, not action names --
 *     `stage3_read.name_move_targets` names each compass move after whichever
 *     line it points along most, so "toward ball" means "along the line to
 *     the man on the ball", which is what "close down" says in football;
 *   - command 0 is maximum braking along the current heading, not a hold, so
 *     it keeps "Slow down" (still moving at 0.6 s) and "Stop" (at rest).
 */
export const DEFENDER_FOOTBALL = {
  "toward ball": "Close down",
  "toward runner": "Track runner",
  "toward goal": "Drop",
  // 3v1 only. The beneficiary is the second strategic attacker: the teammate
  // standing in the space the run opens, and one of the two men the scripted
  // passer may pick (fixedpasser/game.py RECEIVERS). render_figure2_abstract
  // draws him as the "teammate", so that is the word used here too.
  "toward beneficiary": "Cover teammate",
  // no line reached a 0.3 cosine: a lateral move with nothing to name it after
  sideways: "Slide across",
  "slow down": "Slow down",
  stop: "Stop",
};

/**
 * One option's label on the public pitch: the figure's rule about which
 * options carry a word, with the football word in place of the solver's.
 */
export function publicOptionLabel(role, probability,
                                  { stop = false, rests = false, name = "" } = {}) {
  const pct = `${Math.round(probability * 100)}%`;
  if (stop && !rests) return `Slow down ${pct}`;
  if (stop && rests) return `Stop ${pct}`;
  if (role === "ball carrier") return `Dribble ${pct}`;
  if (role === "defender") {
    const word = DEFENDER_FOOTBALL[name];
    if (word) return `${word} ${pct}`;
  }
  return pct;
}

/** A pass option, named the way a commentator would. */
export function publicPassLabel(probability) {
  return `Pass ${Math.round(probability * 100)}%`;
}

/** `extract_panel_policy.compass_name`'s five, against the attack direction. */
const ATTACK_FOOTBALL = {
  forward: "Run forward",
  back: "Check back",
  left: "Cut left",
  right: "Cut right",
  stop: "Stop",
};

const CARRIER_FOOTBALL = {
  forward: "Drive forward",
  back: "Turn back",
  left: "Carry left",
  right: "Carry right",
  stop: "Stop",
};

/** A command name from the payload, as football, for whichever body moved. */
export function publicActionName(solverRole, name) {
  if (!name) return null;
  if (solverRole === "defender") return DEFENDER_FOOTBALL[name] || null;
  if (solverRole === "ball carrier") return CARRIER_FOOTBALL[name] || null;
  return ATTACK_FOOTBALL[name] || null;
}
