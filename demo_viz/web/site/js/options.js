// The feasible actions at a solved moment, ranked as the paper ranks them.
//
// Read from `data/options/<scene>.json`, which `export_options.py` builds from
// the 2026-10-01 bundle. Every number here was produced by the solver: the
// value of choosing an option with the others left at equilibrium, its rank
// among that role's options (ties taking the best position), its equilibrium
// probability, and the solver's own 0.6 s path.
//
// That rule reproduces the bundle's own stored rank for 198 of 203 decisions.
// The five it does not are 2v1 ball carriers, whose sixth option is the pass
// and whose value the bundle does not publish; those moments carry
// `rank_partial` and say so rather than showing a rank that would disagree
// with the paper.

import { roleMarker } from "./pitch.js";

const cache = new Map();

export async function loadOptions(sceneId) {
  if (cache.has(sceneId)) return cache.get(sceneId);
  const name = sceneId.replace(/[^A-Za-z0-9_-]+/g, "_");
  try {
    const response = await fetch(`data/options/${name}.json`);
    const payload = response.ok ? await response.json() : null;
    cache.set(sceneId, payload);
    return payload;
  } catch {
    cache.set(sceneId, null);
    return null;
  }
}

/** One role's options at the solved moment nearest `dt`, or null. */
export function at(payload, dt, role) {
  if (!payload) return null;
  let best = null;
  for (const moment of payload.moments || []) {
    if (best === null || Math.abs(moment.dt - dt) < Math.abs(best.dt - dt)) best = moment;
  }
  return best?.roles?.[role] || null;
}

/** The option the player actually took, or null when it was the pass. */
export function observedOption(entry) {
  const index = Number(entry?.observed?.action);
  if (!Number.isInteger(index)) return null;
  return (entry.options || []).find((o) => o.command === index) || null;
}

/** The best-valued option: rank 1. Several may share it. */
export function bestOption(entry) {
  return (entry?.options || []).find((o) => o.rank === 1) || null;
}

/** "1st", "2nd", "3rd"… for a rank a reader is about to click. */
export function ordinal(n) {
  if (!Number.isFinite(n)) return "";
  const i = Math.round(n);
  const tens = i % 100;
  if (tens >= 11 && tens <= 13) return `${i}th`;
  return `${i}${({ 1: "st", 2: "nd", 3: "rd" })[i % 10] || "th"}`;
}

/**
 * Player evaluation's four readings of one decision.
 *
 * Every feasible option is drawn, so a reader sees the whole fan of moves the
 * solver chose between rather than two of them in isolation. Three of the fan
 * are then named in colour:
 *
 *   played    the role's own colour -- blue for an attacker, red for a
 *             defender, the same two the pitch has used throughout
 *   best      sky
 *   selected  pink
 *
 * Sky and pink are the list's own two accents, so the row a reader clicked and
 * the line that appeared are the same colour, and they are drawn in the
 * `decision` layer, above the players and their labels. Each is cased in page
 * colour first: a bright line needs separation from the shading to stay a line.
 */
export const DECISION = { best: "#00A8D8", selected: "#FF2D8E" };

export function drawDecision(pitch, scene, { all, best, observed, selected, colour, solverRole }) {
  const k = Math.max(pitch.k, 0.62);
  const line = (pts, attrs, layer = "decision") => {
    if (!pts || pts.length < 2) return;
    pitch.add(layer, "path", {
      d: pts.map(([x, y], i) => `${i ? "L" : "M"}${x} ${y}`).join(" "),
      fill: "none", "stroke-linecap": "round", "stroke-linejoin": "round",
      ...attrs,
    });
  };
  // a cased line: page colour underneath, the accent on top
  const accent = (pts, stroke, dash) => {
    line(pts, { stroke: pitch.theme.pitch, "stroke-width": 0.62 * k,
                opacity: 0.9, "stroke-dasharray": dash });
    line(pts, { stroke, "stroke-width": 0.36 * k, "stroke-dasharray": dash });
  };
  const dot = (pts, fill) => {
    if (!pts || !pts.length) return;
    const [x, y] = pts[pts.length - 1];
    pitch.add("decision", "circle", { cx: x, cy: y, r: 0.42 * k, fill,
                                      stroke: pitch.theme.pitch,
                                      "stroke-width": 0.16 * k });
  };
  // the whole option set first, quietly: where each move would take him
  for (const pts of all || []) {
    line(pts, { stroke: pitch.theme.text2, "stroke-width": 0.14 * k, opacity: 0.5 },
         "paths");
  }
  if (observed) {
    accent(observed, colour);
    const [x, y] = observed[observed.length - 1];
    pitch.add("decision", "g", {}).appendChild(
      roleMarker(x, y, 1.55 * k, solverRole, {
        fill: "none", stroke: colour, "stroke-width": 0.26,
      }));
  }
  if (best) { accent(best, DECISION.best); dot(best, DECISION.best); }
  if (selected && selected !== best) {
    accent(selected, DECISION.selected, `${1.4 * k} ${0.8 * k}`);
    dot(selected, DECISION.selected);
  }
}
