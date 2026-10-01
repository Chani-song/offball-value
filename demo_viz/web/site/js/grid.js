// Figure 2's defender-start background and move field.
//
// Both come from ONE file per solved moment, produced upstream by
// `scripts/extract_defender_grid.py` (jobs/solve_defender_grid.sbatch):
//
//     data/processed/showcase_v1/defender_grid_S05/grid_S05_1m_full.json
//     data/processed/showcase_v1/defender_grid_S05/grid_S05@0.6_1m_full.json
//     data/processed/showcase_v1/defender_grid_S05/grid_S05@1.2_1m_full.json
//
// Those files are not in this build; `.gitignore` excludes `data/processed/*`
// upstream and the 2026-09-30 bundle predates them. Everything here is
// therefore dormant: `loadGrid` returns null, the control that would show it
// stays hidden, and nothing is drawn. Drop the three files in at
// `demo_viz/web_data/grid/<code>/<dt>.json` and the same build lights up.
//
// Neither field is ever reconstructed from the figure's PNG.
//
// The schema this reads, from `extract_defender_grid.py`:
//
//   { "points": [ { "status": "solved" | "unsolved" | "dropped",
//                   "observed": bool,
//                   "defender_start": [x, y],       // pitch metres, centre
//                   "value": float,                 // the game's value there
//                   "defender": [ { "prob": float, "end": [x, y] }, ... ] },
//                 ... ] }
//
// and the two derived quantities, exactly as upstream derives them:
//
//   shading  `point.value` at `point.defender_start`
//            (render_figure2_abstract.value_grid), darker = LOWER value =
//            a better start for the defender
//   field    sum over his commands of prob * (end - start)
//            (render_defender_flow_moments.moves), his probability-weighted
//            displacement over the first 0.6 s

import { hexToRgb } from "./pitch.js";
import { view } from "./scene.js";

const SVG_NS = "http://www.w3.org/2000/svg";

/** figure_style's two ends of the background shade (BG_LOW .. BG_HIGH). */
export const SHADE = { low: "#F7F7F7", high: "#B8AE9C" };

/** `--flow-color` / `--flow-alpha` at kyuhyeok-dev@b5f26cb: defence red, 20%. */
export const FLOW = { colour: "#FF0000", alpha: 0.2 };

/** The key the figure prints under the background, word for word. */
export const SHADE_KEY = "darker: better start for the defender";

const cache = new Map();

/**
 * One moment's grid, or null when this build has none.
 *
 * Null is the normal state today and is not an error: the control that would
 * draw it is hidden unless every moment resolves.
 */
export async function loadGrid(code, dt) {
  const key = `${code}@${dt.toFixed(1)}`;
  if (cache.has(key)) return cache.get(key);
  const name = `${code}_${dt.toFixed(1)}`.replace(/[^A-Za-z0-9_@.-]+/g, "_");
  let payload = null;
  try {
    const response = await fetch(`data/grid/${name}.json`);
    payload = response.ok ? await response.json() : null;
  } catch {
    payload = null;
  }
  cache.set(key, payload);
  return payload;
}

/**
 * The solved starts as a regular raster, for the background shade.
 *
 * The lattice is already regular -- `extract_defender_grid` writes `i`/`j` per
 * point at 1 m spacing -- so the raster is built from those indices directly
 * and the browser does the smoothing when it scales the image up. Upstream
 * interpolates with `griddata` onto a 0.1 m mesh instead; the values are the
 * same values, laid out the same way, and nothing is invented between them.
 *
 * Returns { nx, ny, values, x0, x1, y0, y1, min, max }, with NaN where a point
 * was not solved, or null when the file carries no usable lattice.
 */
export function valueRaster(grid) {
  const points = (grid?.points || []).filter(
    (point) => Number.isInteger(point.i) && Number.isInteger(point.j),
  );
  const solved = points.filter((point) => point.status === "solved"
    && Array.isArray(point.defender_start));
  if (!solved.length) return null;
  const is = points.map((p) => p.i);
  const js = points.map((p) => p.j);
  const i0 = Math.min(...is);
  const j0 = Math.min(...js);
  const nx = Math.max(...is) - i0 + 1;
  const ny = Math.max(...js) - j0 + 1;
  const values = new Float64Array(nx * ny).fill(NaN);
  let x0 = Infinity; let x1 = -Infinity; let y0 = Infinity; let y1 = -Infinity;
  let min = Infinity; let max = -Infinity;
  for (const point of solved) {
    values[(point.j - j0) * nx + (point.i - i0)] = point.value;
    const [x, y] = point.defender_start;
    x0 = Math.min(x0, x); x1 = Math.max(x1, x);
    y0 = Math.min(y0, y); y1 = Math.max(y1, y);
    min = Math.min(min, point.value); max = Math.max(max, point.value);
  }
  if (!(max > min)) return null;
  return { nx, ny, values, x0, x1, y0, y1, min, max };
}

/** The solved starts, as [x, y, dx, dy]: his probability-weighted 0.6 s move. */
export function movePoints(grid) {
  if (!grid?.points) return [];
  const out = [];
  for (const point of grid.points) {
    if (point.status !== "solved" || !Array.isArray(point.defender_start)) continue;
    const [x, y] = point.defender_start;
    let dx = 0;
    let dy = 0;
    for (const command of point.defender || []) {
      dx += command.prob * (command.end[0] - x);
      dy += command.prob * (command.end[1] - y);
    }
    out.push([x, y, dx, dy]);
  }
  return out;
}

/**
 * Figure 2's background: the game's value with the defender starting at
 * each lattice point. Darker = lower value = a better start for him, which
 * is `render_figure2_abstract`'s own direction and key.
 */
export function drawValueField(pitch, raster, scene, { low = "#F7F7F7", high = "#B8AE9C" } = {}) {
  if (!raster) return;
  const { nx, ny, values, min, max } = raster;
  pitch.canvas.width = nx;
  pitch.canvas.height = ny;
  const context = pitch.canvas.getContext("2d");
  const image = context.createImageData(nx, ny);
  const lo = hexToRgb(low);
  const hi = hexToRgb(high);
  for (let iy = 0; iy < ny; iy += 1) {
    // image rows run top-down; the lattice runs bottom-up, and the
    // left-to-right mirror flips which end of it is on top
    const sourceRow = scene.flip ? iy : ny - 1 - iy;
    for (let ix = 0; ix < nx; ix += 1) {
      const sourceCol = scene.flip ? nx - 1 - ix : ix;
      const value = values[sourceRow * nx + sourceCol];
      const offset = (iy * nx + ix) * 4;
      if (!Number.isFinite(value)) { image.data[offset + 3] = 0; continue; }
      // darker for the defender: the high colour marks the LOW value
      const t = 1 - (value - min) / (max - min);
      for (let c = 0; c < 3; c += 1) {
        image.data[offset + c] = Math.round(lo[c] + (hi[c] - lo[c]) * t);
      }
      image.data[offset + 3] = 255;
    }
  }
  context.putImageData(image, 0, 0);
  const [vx0, vy0] = view(scene, raster.x0, raster.y0);
  const [vx1, vy1] = view(scene, raster.x1, raster.y1);
  pitch.add("field", "image", {
    href: pitch.canvas.toDataURL(),
    x: Math.min(vx0, vx1) - 0.5, y: Math.min(vy0, vy1) - 0.5,
    width: Math.abs(vx1 - vx0) + 1, height: Math.abs(vy1 - vy0) + 1,
    preserveAspectRatio: "none", style: "image-rendering:auto",
  });
}

/**
 * Figure 2's move field: at each start, the defender's probability-weighted
 * displacement over the first 0.6 s.
 *
 * Upstream integrates these into streamlines (`render_defender_flow_moments`);
 * this draws the vectors themselves, one per lattice point, which is the same
 * field without reimplementing a streamline integrator. Width follows the
 * move's length, as the figure's does.
 */
export function drawMoveField(pitch, points, scene, { colour = "#FF0000", alpha = 0.2, every = 2 } = {}) {
  if (!points?.length) return;
  let top = 0;
  for (const [, , dx, dy] of points) top = Math.max(top, Math.hypot(dx, dy));
  if (!(top > 1e-9)) return;
  const group = pitch.add("field", "g", { opacity: alpha });
  points.forEach(([x, y, dx, dy], index) => {
    if (index % every) return;
    const length = Math.hypot(dx, dy);
    if (length < 0.05) return;
    const from = view(scene, x, y);
    const to = view(scene, x + dx, y + dy);
    const line = document.createElementNS(SVG_NS, "path");
    line.setAttribute("d", `M${from[0]} ${from[1]} L${to[0]} ${to[1]}`);
    line.setAttribute("fill", "none");
    line.setAttribute("stroke", colour);
    line.setAttribute("stroke-width", String(0.1 + 0.26 * (length / top)));
    line.setAttribute("stroke-linecap", "round");
    group.appendChild(line);
  });
}
