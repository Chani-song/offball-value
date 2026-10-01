// Deterministic, collision-aware placement for every label on the pitch.
//
// The rule it enforces: no two visible labels overlap, and no label sits on a
// player marker, a shirt number or a pass label -- at every solved moment, in
// every scene, in Focus and Full pitch alike, and again after a resize. It is
// not a set of offsets tuned for one panel; nothing here knows which scene it
// is drawing.
//
// How a position is chosen:
//
//   1. the anchor is the action's own point -- an arrow's tip, or the player
//      himself for a command that ends at rest;
//   2. candidates are rings of increasing radius around it, each ring swept
//      from the action's own direction outward in alternating turns, so the
//      first tries are the ones a reader would expect;
//   3. the box is the REAL rendered box: the text is put in the DOM and
//      measured with getBBox, never estimated from character counts;
//   4. a candidate overlapping anything already placed or blocked is
//      rejected;
//   5. the first survivor wins; if a ring is reached far from the anchor the
//      label gets a leader line back to it.
//
// Text size never changes to resolve a collision. A crowded panel moves
// labels further out, or draws a leader, and the type stays readable.

const SVG_NS = "http://www.w3.org/2000/svg";

/** Radii to try, in marker-ish units; multiplied by the pitch's scale. */
const RINGS = [1.15, 1.9, 2.8, 3.9, 5.2, 6.8];

/** Turns from the anchor direction, in degrees, tried in this order. */
const TURNS = [0, -28, 28, -56, 56, -84, 84, -115, 115, -145, 145, 180];

function overlaps(a, b, gap) {
  return a[0] < b[2] + gap && a[2] > b[0] - gap
    && a[1] < b[3] + gap && a[3] > b[1] - gap;
}

function area(a, b) {
  const w = Math.min(a[2], b[2]) - Math.max(a[0], b[0]);
  const h = Math.min(a[3], b[3]) - Math.max(a[1], b[1]);
  return w > 0 && h > 0 ? w * h : 0;
}

/**
 * One pitch's labels for one render.
 *
 * Built fresh every time anything redraws, so a resize, a new moment or a
 * different scene all re-solve from scratch rather than reusing a layout that
 * was only collision-free at the size it was computed at.
 */
export class LabelLayout {
  constructor(pitch) {
    this.pitch = pitch;
    this.k = Math.max(pitch.k, 0.62);
    this.blocked = [];        // markers, shirt numbers: never written over
    this.placed = [];         // labels already given a home
  }

  /** A box nothing may be written over. */
  block(box) {
    if (box) this.blocked.push(box);
  }

  /** A round obstacle -- a player marker and the shirt number inside it. */
  blockDisc(cx, cy, radius) {
    this.blocked.push([cx - radius, cy - radius, cx + radius, cy + radius]);
  }

  /** Every obstacle a candidate must clear. */
  get obstacles() {
    return this.blocked.concat(this.placed);
  }

  /**
   * Place `text` for one action.
   *
   * `anchor` is the action's own point and `dir` its direction; `gap` is the
   * clear ground every label keeps, in metres. Returns the text node, already
   * in the labels layer, with its leader line drawn if it needed one.
   */
  place(text, { anchor, dir = [1, 0], size, colour, className = "action-label",
                gap = 0.25, leaderFrom = null } = {}) {
    const node = this.pitch.add("labels", "text", {
      x: 0, y: 0, "text-anchor": "middle", "font-size": size,
      fill: colour, class: className,
    }, text);
    // the real rendered box, at the real font size: no estimate from the
    // character count, which is wrong for "47%" against "Track runner 31%"
    let measured;
    try {
      measured = node.getBBox();
    } catch {
      measured = null;
    }
    const w = measured && measured.width > 0
      ? measured.width : text.length * size * 0.56;
    const h = measured && measured.height > 0 ? measured.height : size * 1.15;

    const n = Math.hypot(dir[0], dir[1]) || 1;
    const base = Math.atan2(dir[1] / n, dir[0] / n);
    const boxAt = (cx, cy) => [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2];

    let best = null;
    let fallback = null;
    outer:
    for (const ring of RINGS) {
      const radius = ring * this.k + Math.max(w, h) * 0.5;
      for (const turn of TURNS) {
        const angle = base + (turn * Math.PI) / 180;
        const cx = anchor[0] + Math.cos(angle) * radius;
        const cy = anchor[1] + Math.sin(angle) * radius;
        const box = boxAt(cx, cy);
        let worst = 0;
        let clash = false;
        for (const other of this.obstacles) {
          if (overlaps(box, other, gap)) { clash = true; worst += area(box, other); }
        }
        if (!clash) { best = { cx, cy, box, ring }; break outer; }
        if (!fallback || worst < fallback.worst) {
          fallback = { cx, cy, box, ring, worst };
        }
      }
    }
    // every ring was crowded: take the least-overlapping candidate rather than
    // dropping the label, because an action without its percentage is worse
    const chosen = best || fallback || { cx: anchor[0], cy: anchor[1], ring: 0,
                                         box: boxAt(anchor[0], anchor[1]) };
    node.setAttribute("x", chosen.cx);
    // getBBox gives the ink box; shifting by its centre puts the text, not the
    // baseline, where the layout asked for it
    const shift = measured ? (chosen.cy - (measured.y + measured.height / 2)) : 0;
    node.setAttribute("y", measured ? Number(node.getAttribute("y")) + shift
                                    : chosen.cy + size * 0.36);
    this.placed.push(chosen.box);

    // far from its action: say which action it belongs to
    const from = leaderFrom || anchor;
    const distance = Math.hypot(chosen.cx - from[0], chosen.cy - from[1]);
    if (distance > (RINGS[1] * this.k + Math.max(w, h) * 0.5) * 1.02) {
      const toward = Math.atan2(chosen.cy - from[1], chosen.cx - from[0]);
      const edge = Math.min(w / 2, h / 2) + gap;
      const line = document.createElementNS(SVG_NS, "line");
      line.setAttribute("x1", from[0] + Math.cos(toward) * this.k * 0.5);
      line.setAttribute("y1", from[1] + Math.sin(toward) * this.k * 0.5);
      line.setAttribute("x2", chosen.cx - Math.cos(toward) * edge);
      line.setAttribute("y2", chosen.cy - Math.sin(toward) * edge);
      line.setAttribute("stroke", colour);
      line.setAttribute("stroke-width", 0.06 * this.k);
      line.setAttribute("opacity", "0.55");
      line.setAttribute("class", "label-leader");
      this.pitch.layers.labels.insertBefore(line, node);
    }
    return node;
  }
}
