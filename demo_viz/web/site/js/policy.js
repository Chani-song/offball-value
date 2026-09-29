// Reading a solved equilibrium policy, and showing that it is mixed.
//
// The numbers are the solver's. This module decodes the index arithmetic
// markov.py uses -- an attack action is one of len(directions)**2 movement
// pairs, with release as the last entry -- and names each command with
// solver.js's own commandName, so the football naming has one definition. It
// computes no probability and renames no certified quantity.
//
// A mixed policy is the interesting case and the one a single arrow destroys,
// so weight is shown as a labelled bar with a printed percentage. Colour is
// never the only carrier: every row has its name and its number.

import { commandName } from "./solver.js";

const EPS = 1e-9;

/** How many actions carry weight. "Pure" means exactly one. */
export function support(policy, eps = EPS) {
  return (policy || []).filter((p) => p > eps).length;
}

export function isMixed(policy, eps = EPS) {
  return support(policy, eps) > 1;
}

/** The defender's policy: one weight per movement command. */
export function defenderRows(state) {
  const directions = state?.directions || [];
  return (state?.root_defender || []).map((probability, index) => ({
    index,
    probability,
    label: commandName(directions[index], state.attack_direction),
    vector: directions[index] || null,
    kind: "move",
  })).filter((row) => row.probability > EPS)
    .sort((a, b) => b.probability - a.probability);
}

/**
 * The attack's policy, decoded.
 *
 * `markov.py` encodes an attacking action over `len(directions) ** 2 + 1`
 * entries: the carrier's command times the receiver's command, then release as
 * the final entry.
 */
export function attackRows(state) {
  const directions = state?.directions || [];
  const n = directions.length;
  return (state?.root_attack || []).map((probability, index) => {
    if (index === n * n) {
      return { index, probability, label: "release the ball", kind: "release",
               vector: null };
    }
    const carry = Math.floor(index / n);
    const run = index % n;
    return {
      index, probability, kind: "move",
      label: `carrier ${commandName(directions[carry], state.attack_direction)}, `
           + `receiver ${commandName(directions[run], state.attack_direction)}`,
      vector: directions[carry] || null,
      receiverVector: directions[run] || null,
    };
  }).filter((row) => row.probability > EPS)
    .sort((a, b) => b.probability - a.probability);
}

/**
 * Render weighted rows.
 *
 * `limit` keeps a 26-entry attack policy readable; the remainder is reported
 * as a residual row rather than dropped, so the weights still sum to one on
 * screen.
 */
export function drawPolicy(node, rows, { limit = 5 } = {}) {
  node.replaceChildren();
  const shown = rows.slice(0, limit);
  const rest = rows.slice(limit);
  const widest = rows.length ? rows[0].probability : 1;

  for (const row of shown) {
    const line = document.createElement("div");
    line.className = "polrow";
    const bar = document.createElement("div");
    bar.className = "polbar";
    bar.style.width = `${Math.max((row.probability / widest) * 100, 2)}%`;
    if (row.kind === "release") bar.classList.add("is-release");
    const name = document.createElement("span");
    name.className = "polname";
    name.textContent = row.label;
    const value = document.createElement("span");
    value.className = "polvalue";
    value.textContent = `${(row.probability * 100).toFixed(0)}%`;
    line.append(bar, name, value);
    node.append(line);
  }

  if (rest.length) {
    const residual = rest.reduce((sum, row) => sum + row.probability, 0);
    const line = document.createElement("div");
    line.className = "polrow is-rest";
    const name = document.createElement("span");
    name.className = "polname";
    name.textContent = `${rest.length} further action${rest.length > 1 ? "s" : ""}`;
    const value = document.createElement("span");
    value.className = "polvalue";
    value.textContent = `${(residual * 100).toFixed(0)}%`;
    line.append(name, value);
    node.append(line);
  }
  return shown.length;
}

/**
 * The modal line: at every decision, the action each side weights most.
 *
 * `stage3_read.modal_path` is explicit that this illustrates the policy and is
 * not a sample from it, so the label travels with the data rather than living
 * in one template. Nothing here simulates a player path.
 */
export const MODAL_LABEL = "Modal policy illustration";
export const MODAL_NOTE =
  "Highest-weight action at each decision. This illustrates the equilibrium "
  + "policy; it is not a sampled trajectory.";
