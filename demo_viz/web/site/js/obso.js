// OBSO threat, and the fixed five-direction candidate pass fan.
//
// OBSO = pitch control x ball transition x EPV score.
//
// Only pitch control is precomputed (demo_viz/web/export_obso.py): it is the
// expensive term and the slow-moving one. The other two are exact here at
// every frame -- transition because the ball moves fast enough that
// interpolating it would smear the brightest moment of a clip, and score
// because one static grid serves all 45 scenes.
//
// Nothing in this file depends on which runner, defender or beneficiary the
// visitor has picked. The surface is a property of the frame and the two
// teams, which is the whole reason it can be precomputed.

/** Where the OBSO files live, relative to the page. */
const OBSO_BASE = "data/obso";

let scoreGrid = null;            // shared across scenes, fetched once
const sceneCache = new Map();    // sceneId -> decoded control samples

function sceneFile(sceneId) {
  return sceneId.replace(/[^A-Za-z0-9]/g, "_");
}

/** The static EPV grid, peak-normalised, oriented attacking to the right. */
export async function loadScoreGrid() {
  if (scoreGrid) return scoreGrid;
  const raw = await (await fetch(`${OBSO_BASE}/score_grid.json`)).json();
  scoreGrid = { ...raw.grid, values: Float64Array.from(raw.values) };
  return scoreGrid;
}

/**
 * Decoded pitch-control samples for one scene, fetched on first use.
 * Returns null when the scene has no exported file, so a caller can fall
 * back rather than throw.
 */
export async function loadControl(sceneId) {
  if (sceneCache.has(sceneId)) return sceneCache.get(sceneId);
  let entry = null;
  try {
    const response = await fetch(`${OBSO_BASE}/${sceneFile(sceneId)}.json`);
    if (response.ok) {
      const raw = await response.json();
      const binary = atob(raw.data);
      const bytes = new Uint8Array(binary.length);
      for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i);
      entry = {
        grid: raw.grid,
        indices: raw.indices,
        levels: raw.levels,
        sigma: raw.transition_sigma_m,
        direction: raw.attacking_direction,
        cells: raw.grid.nx * raw.grid.ny,
        bytes,
      };
    }
  } catch (error) {
    entry = null;
  }
  sceneCache.set(sceneId, entry);
  return entry;
}

/**
 * Pitch control at one frame, linearly interpolated between stored samples.
 *
 * The exporter puts a sample on both sides of every offside change, so a step
 * in control is crossed by a one-frame ramp rather than smeared across the
 * whole gap.
 */
function controlAt(entry, index, out) {
  const { indices, bytes, cells, levels } = entry;
  let hi = 0;
  while (hi < indices.length && indices[hi] < index) hi += 1;
  if (hi >= indices.length) hi = indices.length - 1;
  const lo = Math.max(0, hi - 1);
  const a = indices[lo];
  const b = indices[hi];
  const weight = b > a ? (index - a) / (b - a) : 0;
  const offsetLo = lo * cells;
  const offsetHi = hi * cells;
  for (let i = 0; i < cells; i += 1) {
    const low = bytes[offsetLo + i] / levels;
    const high = bytes[offsetHi + i] / levels;
    out[i] = low + (high - low) * weight;
  }
  return out;
}

/**
 * The OBSO surface for one frame, as a flat array over the 50 x 32 grid.
 *
 * `scene.ball` is in pitch coordinates; the grid is centred on the pitch, the
 * same convention the Python `pitch_grid` uses, so no flip is needed here.
 */
export function surfaceAt(entry, score, scene, index, out) {
  const { nx, ny, length, width } = entry.grid;
  const cells = nx * ny;
  const values = out || new Float64Array(cells);
  controlAt(entry, index, values);
  surfaceAt.lastControl = Float64Array.from(values);

  const bx = scene.ball.x[index];
  const by = scene.ball.y[index];
  if (!Number.isFinite(bx) || !Number.isFinite(by)) {
    values.fill(0);
    return values;
  }
  const dx = length / nx;
  const dy = width / ny;
  const twoSigmaSq = 2 * entry.sigma * entry.sigma;
  // mirror the score grid when the attack runs the other way, exactly as
  // reference_obso.score_surface does
  const mirror = entry.direction < 0;

  for (let iy = 0; iy < ny; iy += 1) {
    const y = iy * dy - width / 2 + dy / 2;
    const rowOffset = iy * nx;
    for (let ix = 0; ix < nx; ix += 1) {
      const x = ix * dx - length / 2 + dx / 2;
      const distanceSq = (x - bx) ** 2 + (y - by) ** 2;
      const transition = Math.exp(-distanceSq / twoSigmaSq);
      const column = mirror ? nx - 1 - ix : ix;
      values[rowOffset + ix] *= transition * score.values[rowOffset + column];
    }
  }
  return values;
}

/** Bilinear sample of a grid field at a pitch coordinate. */
export function sampleAt(grid, values, x, y) {
  const { nx, ny, length, width } = grid;
  const dx = length / nx;
  const dy = width / ny;
  const fx = (x + length / 2 - dx / 2) / dx;
  const fy = (y + width / 2 - dy / 2) / dy;
  const x0 = Math.max(0, Math.min(nx - 1, Math.floor(fx)));
  const y0 = Math.max(0, Math.min(ny - 1, Math.floor(fy)));
  const x1 = Math.min(nx - 1, x0 + 1);
  const y1 = Math.min(ny - 1, y0 + 1);
  const tx = Math.max(0, Math.min(1, fx - x0));
  const ty = Math.max(0, Math.min(1, fy - y0));
  const v00 = values[y0 * nx + x0];
  const v10 = values[y0 * nx + x1];
  const v01 = values[y1 * nx + x0];
  const v11 = values[y1 * nx + x1];
  return (v00 * (1 - tx) + v10 * tx) * (1 - ty) + (v01 * (1 - tx) + v11 * tx) * ty;
}

// ---------------------------------------------------------------------------
// candidate passes
// ---------------------------------------------------------------------------

export {
  CANDIDATE_ANGLES_DEG, CANDIDATE_DISTANCE_M, CARRIER_MARGIN_M, CARRIER_MAX_M,
  candidatePasses, carrierAt,
} from "./carrier.js";

/**
 * The three OBSO terms and their product at one pitch coordinate.
 *
 * Reported separately because they answer different questions: control is who
 * would get there, transition is whether the ball goes there at all, and EPV
 * is what the place is worth. The product is the only one of the four that is
 * OBSO; EPV on its own is position-only and knows nothing about the defensive
 * line, which is why the panel never calls it xT.
 */
export function componentsAt(entry, score, scene, index, x, y, cached = null) {
  const { nx, ny, length, width } = entry.grid;
  const control = cached || controlAt(entry, index, new Float64Array(entry.cells));
  const bx = scene.ball.x[index];
  const by = scene.ball.y[index];
  if (!Number.isFinite(bx) || !Number.isFinite(by)) return null;

  const grid = entry.grid;
  const controlValue = sampleAt(grid, control, x, y);
  const transition = Math.exp(
    -(((x - bx) ** 2 + (y - by) ** 2)) / (2 * entry.sigma * entry.sigma),
  );
  // the score grid is stored attacking right; mirror the lookup, not the array
  const epv = sampleAt(grid, score.values, entry.direction < 0 ? -x : x, y);
  return { control: controlValue, transition, epv, obso: controlValue * transition * epv };
}

/** Pitch control alone at one frame, for callers that want to reuse it. */
export function controlFieldAt(entry, index, out) {
  return controlAt(entry, index, out || new Float64Array(entry.cells));
}
