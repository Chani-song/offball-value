// Figure 2's defender-start background and move field.
//
// The data arrived in the 2026-10-01 vector-field bundle and is exported by
// `demo_viz/web/export_grid.py` to `data/grid/<code>_<dt>.json`: one row per
// solved 1 m lattice start, for S05 at 0.0, 0.6 and 1.2 s.
//
// Two quantities, both the bundle's own:
//
//   value   `v` -- the equilibrium value with the defender starting there.
//           The background shade; LOWER is better for the defence, which is
//           why the dark end marks the low values.
//   move    `dx, dy` -- the sum over his five commands of p x (end - start),
//           his probability-weighted displacement over the first 0.6 s. This
//           is the vector the figure's streamlines follow.
//
// Coordinates are the demo's own already: centre-spot origin, metres, attack
// to the right. Nothing here is reconstructed from the figure's PNG.
//
// Upstream renders the field by interpolating onto a 0.25 m mesh, smoothing
// with a Gaussian of 0.6 m and calling `streamplot`. This draws the measured
// vectors themselves, one per lattice point -- the same field without
// reimplementing a streamline integrator, and without inventing values
// between the points that were actually solved.

import { hexToRgb } from "./pitch.js";
import { view } from "./scene.js";

const SVG_NS = "http://www.w3.org/2000/svg";

/** figure_style's two ends of the background shade (BG_LOW .. BG_HIGH). */
export const SHADE = { low: "#F7F7F7", high: "#B8AE9C" };

/** The abstract renders with `--flow-color "#FF8000" --flow-alpha 0.4`
 *  (README at kyuhyeok-dev@33378ac), not the renderer's red default. */
export const FLOW = { colour: "#FF8000", alpha: 0.4 };

/** The key the abstract prints: `--value-key "preferred defender position"`. */
export const SHADE_KEY = "preferred defender position";

const cache = new Map();

/**
 * One moment's grid, or null when this build has none.
 *
 * Null is the normal state today and is not an error: the control that would
 * draw it is hidden unless every moment resolves.
 */
let shipped = null;

/**
 * Which grids this build ships (`data/grid/index.json`).
 *
 * The vector field exists for the scenes whose bundle carried one, and for no
 * others. Without this list every other scene asked the server for a file that
 * is not there: harmless on the page, three 404s per scene in the log. A build
 * with no manifest falls back to asking, so an older build still works.
 */
async function gridIndex() {
  if (!shipped) {
    shipped = fetch("data/grid/index.json")
      .then((response) => (response.ok ? response.json() : null))
      .catch(() => null);
  }
  return shipped;
}

export async function loadGrid(code, dt) {
  const key = `${code}@${dt.toFixed(1)}`;
  if (cache.has(key)) return cache.get(key);
  const name = `${code}_${dt.toFixed(1)}`.replace(/[^A-Za-z0-9_@.-]+/g, "_");
  const index = await gridIndex();
  if (index && !(index.grids || []).includes(name)) {
    cache.set(key, null);
    return null;
  }
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
 * The lattice is regular -- the export carries `i`/`j` at 1 m spacing -- so
 * the raster is built from those indices and the browser smooths it when it
 * scales the image up. Unsolved starts are left transparent rather than
 * filled in: seven points per moment sit inside the carrier's tackle radius
 * and were never solved.
 */
export function valueRaster(grid) {
  const points = (grid?.points || []).filter(
    (p) => Number.isInteger(p.i) && Number.isInteger(p.j));
  if (!points.length) return null;
  const is = points.map((p) => p.i);
  const js = points.map((p) => p.j);
  const i0 = Math.min(...is);
  const j0 = Math.min(...js);
  const nx = Math.max(...is) - i0 + 1;
  const ny = Math.max(...js) - j0 + 1;
  const values = new Float64Array(nx * ny).fill(NaN);
  let x0 = Infinity; let x1 = -Infinity; let y0 = Infinity; let y1 = -Infinity;
  for (const p of points) {
    values[(p.j - j0) * nx + (p.i - i0)] = p.v;
    x0 = Math.min(x0, p.x); x1 = Math.max(x1, p.x);
    y0 = Math.min(y0, p.y); y1 = Math.max(y1, p.y);
  }
  const min = grid.value?.min ?? Math.min(...points.map((p) => p.v));
  const max = grid.value?.max ?? Math.max(...points.map((p) => p.v));
  if (!(max > min)) return null;
  return { nx, ny, values, x0, x1, y0, y1, min, max };
}

/** The solved starts, as [x, y, dx, dy]: his probability-weighted 0.6 s move. */
export function movePoints(grid) {
  return (grid?.points || []).map((p) => [p.x, p.y, p.dx, p.dy]);
}

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
