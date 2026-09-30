// The figures' arrow language, drawn onto a pitch.
//
// Separate from pitch.js because only a solver view draws an arrow: keeping
// these here leaves them out of the first load for every visitor who never
// opens Counterfactual or Game solution. They take the Pitch as an argument
// rather than extending it, so the deferred module owns no prototype state.

import { P } from "./palette.js";

const SVG_NS = "http://www.w3.org/2000/svg";

function circle(cx, cy, r, attrs) {
  const node = document.createElementNS(SVG_NS, "circle");
  node.setAttribute("cx", cx);
  node.setAttribute("cy", cy);
  node.setAttribute("r", r);
  for (const [k, v] of Object.entries(attrs || {})) node.setAttribute(k, String(v));
  return node;
}

/**
 * Action arrows from one actor, in the figures' language.
 *
 * `options` are `{label, aim: [ux, uy], probability}`. When a probability is
 * given the arrow is weighted by it -- width and opacity together, never
 * colour alone, so a 3% action cannot pass for an 83% one. When it is null
 * the arrows are drawn alike, which is the honest rendering of "these are the
 * moves available" before any game says how they should be mixed.
 *
 * Labels go to the arrow tip and are dropped below `labelFloor`; the panel
 * keeps every option's mass either way, so nothing disappears silently.
 */
export function drawActionArrows(pitch, origin, options, { colour = P.defender, length = 4.2, labelFloor = 0.10,
                     labels = true } = {}) {
  const k = Math.max(pitch.k, 0.62);
  const group = pitch.add("passes", "g", { class: "action-arrows" });
  for (const option of options) {
    const [ux, uy] = option.aim || [0, 0];
    const n = Math.hypot(ux, uy);
    if (n < 1e-9) continue;               // a brake has no direction to draw
    const p = option.probability;
    const weight = p == null
      ? { width: 0.30 * k, opacity: 0.85 }
      : { width: (0.16 + 0.44 * Math.max(0, Math.min(1, p))) * k,
          opacity: 0.35 + 0.65 * Math.max(0, Math.min(1, p)) };
    const len = length * k;
    const x1 = origin[0] + (ux / n) * len;
    const y1 = origin[1] + (uy / n) * len;
    const head = 0.9 * k;
    const ax = (ux / n) * head;
    const ay = (uy / n) * head;
    const path = document.createElementNS(SVG_NS, "path");
    path.setAttribute("d",
      `M ${origin[0]} ${origin[1]} L ${x1} ${y1} `
      + `M ${x1} ${y1} l ${-ax - ay * 0.6} ${-ay + ax * 0.6} `
      + `M ${x1} ${y1} l ${-ax + ay * 0.6} ${-ay - ax * 0.6}`);
    path.setAttribute("fill", "none");
    path.setAttribute("stroke", colour);
    path.setAttribute("stroke-width", weight.width);
    path.setAttribute("stroke-linecap", "round");
    path.setAttribute("opacity", weight.opacity);
    group.appendChild(path);

    if (!labels || (p != null && p < labelFloor)) continue;
    const text = pitch.add("labels", "text", {
      x: x1 + (ux / n) * 0.9 * k, y: y1 + (uy / n) * 0.9 * k + 0.4 * k,
      "text-anchor": ux >= 0 ? "start" : "end",
      "font-size": 1.45 * k, fill: colour, class: "action-label",
      opacity: p == null ? 0.95 : Math.max(0.55, weight.opacity),
    }, p == null ? option.label : `${option.label} ${Math.round(p * 100)}%`);
    group.appendChild(text);
  }
  return group;
}

/**
 * The pass the solver's policy plays, as the figures draw it: a dashed line
 * from the ball to the target. Only ever drawn from a real release policy --
 * the five-ray explorer has its own geometry and its own layer.
 */
export function drawPassChoice(pitch, from, to, label) {
  const k = Math.max(pitch.k, 0.62);
  const group = pitch.add("passes", "g", { class: "pass-choice" });
  const line = document.createElementNS(SVG_NS, "line");
  line.setAttribute("x1", from[0]);
  line.setAttribute("y1", from[1]);
  line.setAttribute("x2", to[0]);
  line.setAttribute("y2", to[1]);
  line.setAttribute("stroke", P.text);
  line.setAttribute("stroke-width", 0.26 * k);
  line.setAttribute("stroke-dasharray", `${1.1 * k} ${0.8 * k}`);
  line.setAttribute("stroke-linecap", "round");
  group.appendChild(line);
  group.appendChild(circle(to[0], to[1], 0.55 * k,
                           { fill: "none", stroke: P.text, "stroke-width": 0.22 * k }));
  if (label) {
    group.appendChild(pitch.add("labels", "text", {
      x: to[0], y: to[1] - 1.2 * k, "text-anchor": "middle",
      "font-size": 1.45 * k, fill: P.text, class: "pass-label",
    }, label));
  }
  return group;
}
