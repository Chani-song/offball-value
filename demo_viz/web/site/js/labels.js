// Every label on the pitch, placed by Figure 2's own algorithm.
//
// This is a port of `Placer` in `scripts/render_figure2_abstract.py`, not a
// second design that aims at the same thing. Its docstring is the spec:
//
//   "Labels next to what they name. Each goes to the free spot nearest its
//    anchor (a path's end, a player), preferring the side the path points to
//    (a cost growing with the angle off it); free = covers no player, no line,
//    no arrowhead, no other label or leader, and stays inside the panel. A
//    label farther than LEADER from its anchor gets a thin leader line, and
//    each bit of leader over something drawn costs extra. Placement is greedy
//    in some order; ORDERS orders are tried (the most likely first, then
//    seeded shuffles, role names always last) and the one with the least total
//    cost is drawn."
//
// What that changes against what was here before: candidates are a lattice,
// not rings, so a label may sit anywhere free rather than on one of 72 spokes;
// the winner is the lowest-cost spot, not the first that happens to fit; the
// obstacles include the arrows themselves and their heads, not only players;
// and the order labels are placed in is searched rather than assumed, because
// greedy placement in the wrong order is what crowds the last label out.
//
// Two things are ours, and both are forced by the browser:
//
//   * the lattice is scaled with the pitch's zoom (`REACH` and `STEP` are
//     fixed metres in the figure, whose panels are always one crop; ours is
//     the same panel at a settable zoom, so the lattice keeps its size in type
//     rather than in metres);
//   * `ORDERS` is 8, not 60. The layout runs at a solved moment, in a render
//     loop, on whatever machine opened the page.
//
// Text size never changes to resolve a collision. A crowded panel moves labels
// further out, or draws a leader, and the type stays readable.

const SVG_NS = "http://www.w3.org/2000/svg";

/** `Placer.STEP / REACH / ORDERS`, and `LEADER`, in metres at the figure's zoom. */
const STEP = 0.15;
const REACH = 5.0;
const LEADER = 0.6;
const ORDERS = 8;

/** Rings beyond `REACH`, tried only when the lattice has no free spot. */
const RINGS_REACH = 3;          // out to three times `REACH`
const RING_STEP = 4;            // at four lattice steps

/** Two labels with the same text keep this many label heights between centres. */
const TWIN_SPACING = 1.5;

/** The zoom the figure's own panels are drawn at, which sets the lattice. */
const FIGURE_K = 0.62;

/** `Placer._samples`: a polyline as boxes, one every 0.12 m. */
function samples(pts, half) {
  const out = [];
  for (let i = 1; i < pts.length; i += 1) {
    const [ax, ay] = pts[i - 1];
    const [bx, by] = pts[i];
    const n = Math.max(1, Math.round(Math.hypot(bx - ax, by - ay) / 0.12));
    for (let s = 0; s <= n; s += 1) {
      const x = ax + ((bx - ax) * s) / n;
      const y = ay + ((by - ay) * s) / n;
      out.push([x - half, y - half, x + half, y + half]);
    }
  }
  return out;
}

/** `Placer._free`, for one box. */
function free(box, walls) {
  for (let i = 0; i < walls.length; i += 1) {
    const w = walls[i];
    if (box[0] < w[2] && w[0] < box[2] && box[1] < w[3] && w[1] < box[3]) return false;
  }
  return true;
}

/** The area two boxes share. */
function overlapArea(a, b) {
  const w = Math.min(a[2], b[2]) - Math.max(a[0], b[0]);
  const h = Math.min(a[3], b[3]) - Math.max(a[1], b[1]);
  return w > 0 && h > 0 ? w * h : 0;
}

function overlapWith(box, walls) {
  let total = 0;
  for (let i = 0; i < walls.length; i += 1) total += overlapArea(box, walls[i]);
  return total;
}

/** `Placer._end`: the point of `box` nearest `anchor`. */
function endOf(anchor, box) {
  return [Math.min(Math.max(anchor[0], box[0]), box[2]),
          Math.min(Math.max(anchor[1], box[1]), box[3])];
}

/** Seeded, so the same panel lays out the same way twice. */
function rng(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6D2B79F5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function shuffled(items, random) {
  const out = items.slice();
  for (let i = out.length - 1; i > 0; i -= 1) {
    const j = Math.floor(random() * (i + 1));
    [out[i], out[j]] = [out[j], out[i]];
  }
  return out;
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
    this.solid = [];          // `Placer.solid`: players, lines, heads, pins
    this.hard = [];           // the part of `solid` no label may cover even as
                              // a last resort: player markers, shirt numbers, the ball
    this.todo = [];           // labels measured but not yet placed
    this.placed = [];         // boxes of labels already given a home
    this.placedTexts = [];    // their text and centre, for the twin spacing
    this.gap = 0;             // `Placer.GAP`
    // the lattice, in the figure's steps scaled to this zoom
    const scale = this.k / FIGURE_K;
    this.step = STEP * scale;
    this.reach = REACH * scale;
    this.leader = LEADER * scale;
    this.offsets = [];
    const n = Math.round(this.reach / this.step);
    for (let i = -n; i <= n; i += 1) {
      for (let j = -n; j <= n; j += 1) {
        this.offsets.push([i * this.step, j * this.step]);
      }
    }
    this.bounds = this.panelBounds();
  }

  /** The panel: the pitch's own crop, with the figure's 0.1 m margin. */
  panelBounds() {
    const view = (this.pitch.svg?.getAttribute("viewBox") || "")
      .split(/[\s,]+/).map(Number);
    if (view.length !== 4 || view.some((v) => !Number.isFinite(v))) {
      return [-52.5, -34, 52.5, 34];
    }
    const m = 0.1 * (this.k / FIGURE_K);
    return [view[0] + m, view[1] + m, view[0] + view[2] - m, view[1] + view[3] - m];
  }

  /** A box nothing may be written over. */
  block(box) {
    if (box) this.solid.push(box);
  }

  /** `Placer.disc`: a player marker and the shirt number inside it. */
  blockDisc(cx, cy, radius, hard = false) {
    const box = [cx - radius, cy - radius, cx + radius, cy + radius];
    this.solid.push(box);
    if (hard) this.hard.push(box);
  }

  /** A drawn text's own box (a shirt number), never to be written over. */
  blockText(node, pad = 0) {
    let b;
    try {
      b = node.getBBox();
    } catch {
      return;
    }
    if (!b || (!b.width && !b.height)) return;
    const box = [b.x - pad, b.y - pad, b.x + b.width + pad, b.y + b.height + pad];
    this.solid.push(box);
    this.hard.push(box);
  }

  /** `Placer.line`: an arrow's shaft, as the ground it covers. */
  blockLine(pts, halfWidth) {
    if (!pts || pts.length < 2) return;
    this.solid.push(...samples(pts, Math.max(halfWidth, 0.08 * this.k) + 0.05 * this.k));
  }

  /** `Placer.head`: an arrowhead is wider than its line. */
  blockHead(tip, radius) {
    this.blockDisc(tip[0], tip[1], radius + 0.05 * this.k);
  }

  /** Every obstacle a candidate must clear. */
  get obstacles() {
    return this.solid.concat(this.placed);
  }

  /**
   * `Placer.add`: measure one label and queue it.
   *
   * `anchor` is the action's own point and `dir` its direction; `rank` orders
   * the greedy pass (heaviest first); `leader` is the colour of the leader
   * line it may need, or null for a label that never gets one.
   */
  add(text, { anchor, dir = [1, 0], size, colour, className = "action-label",
              gap = 0, leader = null, rank = 0 } = {}) {
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
    const w = (measured && measured.width > 0
      ? measured.width : text.length * size * 0.56) + 0.12 * this.k;
    const h = (measured && measured.height > 0
      ? measured.height : size * 1.15) + 0.08 * this.k;
    this.gap = Math.max(this.gap, gap);
    const n = Math.hypot(dir[0], dir[1]) || 1;
    this.todo.push({
      node, measured, size, anchor, w, h, rank, text,
      leader: leader === null ? null : (leader || colour),
      dir: [dir[0] / n, dir[1] / n],
    });
    return node;
  }

  /** Measure, place and draw one label on its own. */
  place(text, options = {}) {
    const node = this.add(text, options);
    this.run();
    return node;
  }

  /**
   * `Placer._search`: the best free spot for one label, or null.
   *
   * A leader may cross a line (at a cost) but never another label or leader.
   */
  search(item, walls, labelBoxes, twins = []) {
    // the lattice and its costs depend only on the label, so they are built
    // once and reused across the orders: only the walls change between them
    if (!item.cands) item.cands = this.candidates(item, this.offsets);
    const found = this.pick(item, item.cands, walls, labelBoxes, twins);
    if (found && !found.fallback) return found;
    // nothing free within REACH: the same rules on rings further out, where a
    // leader carries the label back to its action, before any overlap
    if (!item.ring) item.ring = this.candidates(item, this.ringOffsets());
    const far = this.pick(item, item.ring, walls, labelBoxes, twins);
    if (far && !far.fallback) return far;
    return this.leastOverlap(item, walls, twins) || found || far;
  }

  /** The candidate boxes for one label at `offsets`, cheapest first. */
  candidates(item, offsets) {
    const [ax, ay] = item.anchor;
    const { w, h } = item;
    const [bx0, by0, bx1, by1] = this.bounds;
    const cands = [];
    for (let i = 0; i < offsets.length; i += 1) {
      const [di, dj] = offsets[i];
      const cx = ax + di;
      const cy = ay + dj;
      const x0 = cx - w / 2;
      const x1 = cx + w / 2;
      const y0 = cy - h / 2;
      const y1 = cy + h / 2;
      if (x0 < bx0 || x1 > bx1 || y0 < by0 || y1 > by1) continue;
      // `gap`: how far the box is from the anchor, zero while it covers it
      const gx = Math.max(x0 - ax, 0, ax - x1);
      const gy = Math.max(y0 - ay, 0, ay - y1);
      const gap = Math.hypot(gx, gy);
      const off = Math.hypot(di, dj);
      const cos = off > 0 ? (di * item.dir[0] + dj * item.dir[1]) / off : 1;
      // 0 straight ahead, +1 m right behind -- in the figure's metres
      cands.push({ cx, cy, gap, box: [x0, y0, x1, y1],
                   cost: gap + 0.5 * (1 - cos) * (this.k / FIGURE_K) });
    }
    cands.sort((a, b) => a.cost - b.cost);
    return cands;
  }

  /** Square rings beyond the lattice, out to `RINGS_REACH` times `REACH`. */
  ringOffsets() {
    if (this.rings) return this.rings;
    const step = RING_STEP * this.step;
    const inner = Math.round(this.reach / step);
    const outer = Math.round((RINGS_REACH * this.reach) / step);
    this.rings = [];
    for (let i = -outer; i <= outer; i += 1) {
      for (let j = -outer; j <= outer; j += 1) {
        if (Math.max(Math.abs(i), Math.abs(j)) <= inner) continue;
        this.rings.push([i * step, j * step]);
      }
    }
    return this.rings;
  }

  /**
   * The last resort when no spot anywhere is free: the one covering the least
   * of a player marker, shirt number or the ball, then the least of anything
   * else, then the cheapest. Never simply the cheapest blocked spot.
   */
  leastOverlap(item, walls, twins) {
    let best = null;
    for (const cand of (item.cands || []).concat(item.ring || [])) {
      if (this.twinClash(item, cand, twins)) continue;
      const hard = overlapWith(cand.box, this.hard);
      const soft = overlapWith(cand.box, walls);
      if (!best || hard < best.hard - 1e-9
          || (Math.abs(hard - best.hard) <= 1e-9
              && (soft < best.soft - 1e-9
                  || (Math.abs(soft - best.soft) <= 1e-9 && cand.cost < best.cost)))) {
        best = { ...cand, hard, soft };
      }
    }
    if (!best) return null;
    const { hard, soft, ...spot } = best;
    return { ...spot, fallback: true };
  }

  /**
   * Two labels with the same text (two players at the same percentage) must
   * read as two: centres at least `TWIN_SPACING` label heights apart.
   */
  twinClash(item, cand, twins) {
    for (let i = 0; i < twins.length; i += 1) {
      const t = twins[i];
      if (t.text !== item.text) continue;
      const need = TWIN_SPACING * Math.max(item.h, t.h);
      if (Math.hypot(cand.cx - t.cx, cand.cy - t.cy) < need) return true;
    }
    return false;
  }

  /** The cheapest free candidate, with the leader's own cost. */
  pick(item, cands, walls, labelBoxes, twins = []) {
    const [ax, ay] = item.anchor;
    let best = null;
    let nearest = null;                  // the cheapest blocked spot, if none is free
    for (const cand of cands) {
      if (best && cand.cost >= best.cost) break;
      if (!free(cand.box, walls) || this.twinClash(item, cand, twins)) {
        if (!nearest) nearest = { ...cand, fallback: true };
        continue;
      }
      let cost = cand.cost;
      if (cand.gap > this.leader && item.leader) {
        // each 0.15 m of leader over something drawn costs 0.4 m; a leader
        // that would cross a label or another leader is not allowed at all
        const [ex, ey] = endOf(item.anchor, cand.box);
        const n = Math.max(1, Math.round(cand.gap / this.step));
        const half = 0.03 * this.k;
        let over = 0;
        let blocked = false;
        for (let s = 2; s <= n; s += 1) {
          const px = ax + ((ex - ax) * s) / n;
          const py = ay + ((ey - ay) * s) / n;
          const dot = [px - half, py - half, px + half, py + half];
          if (!free(dot, labelBoxes)) { blocked = true; break; }
          if (!free(dot, walls)) over += 1;
        }
        if (blocked) continue;
        cost += 0.4 * over * (this.k / FIGURE_K);
      }
      if (!best || cost < best.cost) best = { ...cand, cost };
    }
    return best || nearest || null;
  }

  /** `Placer._run`: one greedy pass in one order. */
  runOrder(order) {
    const walls = this.solid.concat(this.placed);
    const labelBoxes = this.placed.slice();
    const twins = this.placedTexts.slice();
    let total = 0;
    const out = [];
    for (const item of order) {
      const found = this.search(item, walls, labelBoxes, twins);
      if (!found) return null;
      total += found.fallback ? found.cost + 1e3 : found.cost;
      out.push([item, found]);
      twins.push({ text: item.text, cx: found.cx, cy: found.cy, h: item.h });
      const g = this.gap;
      walls.push([found.box[0] - g, found.box[1] - g,
                  found.box[2] + g, found.box[3] + g]);
      labelBoxes.push(found.box);
      if (found.gap > this.leader && item.leader) {
        // a leader: nothing may sit on it, and no leader may cross it
        const lead = samples([item.anchor, endOf(item.anchor, found.box)],
                             0.08 * this.k);
        walls.push(...lead);
        labelBoxes.push(...lead.slice(2));
      }
    }
    return { total, out };
  }

  /**
   * `Placer.place`: try several orders, draw the cheapest.
   *
   * The numbers go first, heaviest first, because the probability that matters
   * most should keep the spot it wants; role names have no leader and always
   * go last, taking whatever is left.
   */
  run() {
    if (!this.todo.length) return;
    const base = this.todo.slice().sort((a, b) => b.rank - a.rank);
    const names = base.filter((t) => t.leader === null);
    const movable = base.filter((t) => t.leader !== null);
    const random = rng(0);
    const orders = [base];
    for (let i = 0; i < ORDERS - 1 && movable.length > 1; i += 1) {
      orders.push(shuffled(movable, random).concat(names));
    }
    let best = null;
    for (const order of orders) {
      const result = this.runOrder(order);
      if (result && (!best || result.total < best.total)) best = result;
    }
    if (!best) best = { out: base.map((item) => [item, this.anchorSpot(item)]) };
    for (const [item, found] of best.out) this.draw(item, found);
    this.todo = [];
  }

  /** Nowhere was free: the label still belongs to its action. */
  anchorSpot(item) {
    const [ax, ay] = item.anchor;
    return { cx: ax, cy: ay, gap: 0,
             box: [ax - item.w / 2, ay - item.h / 2,
                   ax + item.w / 2, ay + item.h / 2] };
  }

  draw(item, found) {
    const { node, measured } = item;
    node.setAttribute("x", found.cx);
    // getBBox gives the ink box; shifting by its centre puts the text, not the
    // baseline, where the layout asked for it
    const shift = measured ? (found.cy - (measured.y + measured.height / 2)) : 0;
    node.setAttribute("y", measured ? Number(node.getAttribute("y")) + shift
                                    : found.cy + item.size * 0.36);
    this.placed.push(found.box);
    this.placedTexts.push({ text: item.text, cx: found.cx, cy: found.cy, h: item.h });
    if (found.gap > this.leader && item.leader) {
      const [ex, ey] = endOf(item.anchor, found.box);
      const line = document.createElementNS(SVG_NS, "line");
      line.setAttribute("x1", item.anchor[0]);
      line.setAttribute("y1", item.anchor[1]);
      line.setAttribute("x2", ex);
      line.setAttribute("y2", ey);
      line.setAttribute("stroke", item.leader);
      // `lw=0.5` pt, `alpha=0.9`, butt caps -- the figure's own leader
      line.setAttribute("stroke-width", 0.055 * this.k);
      line.setAttribute("opacity", "0.9");
      line.setAttribute("stroke-linecap", "butt");
      line.setAttribute("class", "label-leader");
      this.pitch.layers.labels.insertBefore(line, node);
    }
  }
}
