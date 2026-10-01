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
 * Action arrows, in the figures' language.
 *
 * GEOMETRY IS PHYSICAL. An option is drawn along `path` -- the solver's own
 * 0.6 s movement under that command, momentum included, exactly as
 * `render_figure2_abstract` draws the panel file's `path` unchanged. A
 * high-probability command can therefore have a *short* arrow: a player
 * running one way who is told to accelerate the other covers little ground in
 * 0.6 s. **Probability never rescales it.**
 *
 * Where no path exists the caller may pass `aim` and a `length`, and must say
 * so: Figure 1's two option arrows are a fixed 3 m by upstream's own account
 * ("picture choices, not data"), and that is the only honest use of a fixed
 * length.
 *
 * PROBABILITY IS WIDTH. `lwOf(p) = 0.6 + 2.4p`, one scale for every panel, as
 * upstream. A stop ends in a bar across the path, not a head.
 */
export function drawActionArrows(pitch, origin, options, {
  colour = P.defender, length = 4.2, labelFloor = 0, labels = true,
  widthOf = null, scale = 1,
} = {}) {
  const k = Math.max(pitch.k, 0.62);
  const group = pitch.add("passes", "g", { class: "action-arrows" });

  for (const option of options) {
    // the path wins; aim + length is the declared fallback
    let pts = option.path && option.path.length > 1 ? option.path : null;
    if (!pts) {
      const [ux, uy] = option.aim || [0, 0];
      const n = Math.hypot(ux, uy);
      if (n < 1e-9) continue;              // a stop has no direction to aim
      pts = [origin, [origin[0] + (ux / n) * length * scale,
                      origin[1] + (uy / n) * length * scale]];
    }
    const p = option.probability;
    const width = (widthOf ? widthOf(p) : (p == null ? 1.3 : 0.6 + 2.4 * p)) * 0.22 * k;

    const [x0, y0] = pts[pts.length - 2];
    const [x1, y1] = pts[pts.length - 1];
    const dx = x1 - x0;
    const dy = y1 - y0;
    const n = Math.hypot(dx, dy) || 1;
    const head = 0.9 * k;
    const ax = (dx / n) * head;
    const ay = (dy / n) * head;

    const shaft = pts.map(([x, y]) => `${x},${y}`).join(" ");
    const line = document.createElementNS(SVG_NS, "polyline");
    line.setAttribute("points", shaft);
    line.setAttribute("fill", "none");
    line.setAttribute("stroke", colour);
    line.setAttribute("stroke-width", width);
    line.setAttribute("stroke-linecap", "butt");
    line.setAttribute("stroke-linejoin", "round");
    group.appendChild(line);

    if (option.stop) {
      // braking: a bar across the end, as wide as a head on this line
      const bar = document.createElementNS(SVG_NS, "line");
      bar.setAttribute("x1", x1 - (dy / n) * head * 0.5);
      bar.setAttribute("y1", y1 + (dx / n) * head * 0.5);
      bar.setAttribute("x2", x1 + (dy / n) * head * 0.5);
      bar.setAttribute("y2", y1 - (dx / n) * head * 0.5);
      bar.setAttribute("stroke", colour);
      bar.setAttribute("stroke-width", Math.max(width * 0.7, 0.12 * k));
      group.appendChild(bar);
    } else {
      const arrowhead = document.createElementNS(SVG_NS, "path");
      arrowhead.setAttribute("d",
        `M ${x1} ${y1} l ${-ax - ay * 0.55} ${-ay + ax * 0.55} `
        + `M ${x1} ${y1} l ${-ax + ay * 0.55} ${-ay - ax * 0.55}`);
      arrowhead.setAttribute("fill", "none");
      arrowhead.setAttribute("stroke", colour);
      arrowhead.setAttribute("stroke-width", width);
      arrowhead.setAttribute("stroke-linecap", "round");
      group.appendChild(arrowhead);
    }

    if (!labels || !option.label) continue;
    if (p != null && p < labelFloor) continue;
    const text = pitch.add("labels", "text", {
      x: x1 + (dx / n) * 1.0 * k, y: y1 + (dy / n) * 1.0 * k + 0.4 * k,
      "text-anchor": dx >= 0 ? "start" : "end",
      "font-size": 1.45 * k, fill: colour, class: "action-label",
    }, option.label);
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
