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
/**
 * Where a path should start so it leaves a marker's edge with clear ground.
 *
 * `render_figure2_abstract.leave` / `clear_of`: walk forward until the first
 * point that is `reach` metres from the centre, then bisect the crossing
 * segment. A path that already starts clear is untouched.
 */
function leaveMarker(pts, reachM) {
  if (!pts.length || reachM <= 0) return pts;
  const centre = pts[0];
  const far = (p) => Math.hypot(p[0] - centre[0], p[1] - centre[1]) >= reachM;
  let i = 0;
  while (i < pts.length - 1 && !far(pts[i + 1])) i += 1;
  if (i === pts.length - 1) return pts.slice(-1);
  const [ax, ay] = pts[i];
  const [bx, by] = pts[i + 1];
  let lo = 0;
  let hi = 1;
  for (let n = 0; n < 40; n += 1) {
    const mid = (lo + hi) / 2;
    if (far([ax + (bx - ax) * mid, ay + (by - ay) * mid])) hi = mid; else lo = mid;
  }
  return [[ax + (bx - ax) * hi, ay + (by - ay) * hi], ...pts.slice(i + 1)];
}

/** Do two boxes overlap, once `gap` is added round the placed one? */
function collides(box, placed, gap) {
  return placed.some(([x0, y0, x1, y1]) =>
    box[0] < x1 + gap && box[2] > x0 - gap && box[1] < y1 + gap && box[3] > y0 - gap);
}

export function drawActionArrows(pitch, origin, options, {
  colour = P.defender, length = 4.2, labelFloor = 0, labels = true,
  widthOf = null, scale = 1, labelGap = 0, clearM = 0, placed = null,
} = {}) {
  const k = Math.max(pitch.k, 0.62);
  const group = pitch.add("passes", "g", { class: "action-arrows" });
  // labels are placed heaviest first, so the probability that matters most
  // keeps the spot it wants and lighter ones step aside -- upstream's order
  // upstream's placement order: heaviest first, and a stop that has come to
  // rest last -- its label sits on the player and can go anywhere round him,
  // so it takes whatever the moves leave free (`o["prob"] - 1.0` there).
  const priority = (o) => (o.probability ?? 0) - (o.stop && o.rests ? 1 : 0);
  const order = [...options].sort((a, b) => priority(b) - priority(a));
  const taken = placed || [];

  for (const option of order) {
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

    // the shaft starts clear of the marker; the *whole* path still sets the
    // direction and the label's anchor. A move shorter than the marker --
    // S05's 0.6 s "toward ball" is 0.57 m -- clips to nothing, so it has no
    // shaft to draw and lives entirely in its named label, which is why
    // upstream names these moves at all.
    const shaft = clearM > 0 ? leaveMarker(pts, clearM) : pts;
    const [x0, y0] = pts[pts.length - 2];
    const [x1, y1] = pts[pts.length - 1];
    const dx = x1 - x0;
    const dy = y1 - y0;
    const n = Math.hypot(dx, dy) || 1;
    const head = 0.9 * k;
    const ax = (dx / n) * head;
    const ay = (dy / n) * head;

    if (shaft.length > 1) {
      const line = document.createElementNS(SVG_NS, "polyline");
      line.setAttribute("points", shaft.map(([x, y]) => `${x},${y}`).join(" "));
      line.setAttribute("fill", "none");
      line.setAttribute("stroke", colour);
      line.setAttribute("stroke-width", width);
      line.setAttribute("stroke-linecap", "butt");
      line.setAttribute("stroke-linejoin", "round");
      group.appendChild(line);
    }

    if (shaft.length <= 1) {
      // nothing to head or bar: the label carries it
    } else if (option.stop) {
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

    // deterministic placement: try the arrow tip, then step outward along the
    // arrow, then to its two sides. The first position clear of every label
    // already placed wins; nothing is hidden and nothing is random.
    const size = 1.45 * k;
    const w = option.label.length * size * 0.56;
    const h = size * 1.15;
    const ux = dx / n;
    const uy = dy / n;
    let bx = 0;
    let by = 0;
    for (const [sx, sy] of [[0, 0], [1, 0], [2, 0], [0, 1], [0, -1],
                            [1, 1], [1, -1], [3, 0], [0, 2], [0, -2]]) {
      const ox = ux * sx * labelGap + -uy * sy * labelGap;
      const oy = uy * sx * labelGap + ux * sy * labelGap;
      const [cx, cy] = option.stop && option.rests ? pts[0] : [x1, y1];
      const left = cx + ux * 1.0 * k + ox - (ux >= 0 ? 0 : w);
      const top = cy + uy * 1.0 * k + oy - h / 2;
      if (!collides([left, top, left + w, top + h], taken, labelGap)) {
        bx = ox; by = oy; break;
      }
    }
    // a resting stop is labelled on the player, not at the path's end
    const [ax0, ay0] = option.stop && option.rests ? pts[0] : [x1, y1];
    const lx = ax0 + ux * 1.0 * k + bx;
    const ly = ay0 + uy * 1.0 * k + by;
    taken.push([lx - (ux >= 0 ? 0 : w), ly - h / 2,
                lx + (ux >= 0 ? w : 0), ly + h / 2]);
    const text = pitch.add("labels", "text", {
      x: lx, y: ly + 0.4 * k,
      "text-anchor": ux >= 0 ? "start" : "end",
      "font-size": size, fill: colour, class: "action-label",
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
  // charcoal on the paper pitch, near-white on the dark one: the pass is ink,
  // not a role, so it takes the theme's text colour rather than a role colour
  const ink = pitch.theme?.text || P.text;
  line.setAttribute("stroke", ink);
  line.setAttribute("stroke-width", 0.26 * k);
  line.setAttribute("stroke-dasharray", `${1.1 * k} ${0.8 * k}`);
  line.setAttribute("stroke-linecap", "round");
  group.appendChild(line);
  group.appendChild(circle(to[0], to[1], 0.55 * k,
                           { fill: "none", stroke: ink, "stroke-width": 0.22 * k }));
  if (label) {
    group.appendChild(pitch.add("labels", "text", {
      x: to[0], y: to[1] - 1.2 * k, "text-anchor": "middle",
      "font-size": 1.45 * k, fill: ink, class: "pass-label",
    }, label));
  }
  return group;
}
