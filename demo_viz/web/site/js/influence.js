// Residual goal-weighted space, ported from the Python implementation.
//
// Mirrors, function for function:
//   offball_value.fernandez_influence.fernandez_influence_surface
//   offball_value.goal_weighted_influence.goal_weighted_space_surface
//   offball_value.goal_weighted_influence.influence_pitch_grid
//   demo_viz.core.influence.InfluenceCache
//
//   intrinsic = target_influence * goal_weighted_space_value(target)
//   uncovered = exp(-k * sum_of_defender_influences)
//   residual  = intrinsic * uncovered
//
// demo_viz/web/validate.py checks this against Python on random frames,
// runners, defenders and beneficiaries.

export const CONFIG = {
  gridResolution: 2.0,
  maxInfluenceSpeed: 13.0,
  goalDistancePower: 1.5,
  goalSideSoftness: 1.5,
  goalSideFloor: 0.2,
  goalConeBaseHalfWidth: 2.5,
  goalConeGrowthPerM: 0.2,
  goalConeFloor: 0.2,
  defenderSuppression: 1.25,
  fieldLength: 105.0,
  fieldWidth: 68.0,
};

export function pitchGrid(config = CONFIG) {
  const nx = Math.ceil(config.fieldLength / config.gridResolution);
  const ny = Math.ceil(config.fieldWidth / config.gridResolution);
  const halfL = config.fieldLength / 2;
  const halfW = config.fieldWidth / 2;
  const x = linspace(-halfL + config.fieldLength / (2 * nx), halfL - config.fieldLength / (2 * nx), nx);
  const y = linspace(-halfW + config.fieldWidth / (2 * ny), halfW - config.fieldWidth / (2 * ny), ny);
  return { x, y, nx, ny, cellArea: (config.fieldLength / nx) * (config.fieldWidth / ny) };
}

function linspace(start, stop, count) {
  const out = new Float64Array(count);
  if (count === 1) { out[0] = start; return out; }
  const step = (stop - start) / (count - 1);
  for (let i = 0; i < count; i += 1) out[i] = start + step * i;
  return out;
}

// ScenePlayer.velocity: central difference over +/- (window // 2) samples.
export function velocityAt(xs, ys, index, fps, windowSamples) {
  const n = xs.length;
  if (n < 2) return [0, 0];
  const half = Math.max(1, Math.floor(windowSamples / 2));
  const lo = Math.max(0, index - half);
  const hi = Math.min(n - 1, index + half);
  if (hi === lo) return [0, 0];
  const dt = (hi - lo) / fps;
  const vx = (xs[hi] - xs[lo]) / dt;
  const vy = (ys[hi] - ys[lo]) / dt;
  if (!Number.isFinite(vx) || !Number.isFinite(vy)) return [0, 0];
  return [vx, vy];
}

export function influenceRadius(distanceToBall) {
  return Math.min(4.0 + (distanceToBall ** 3) / 1120.0, 10.0);
}

/** Fernandez & Bornn (2018) player influence over the whole grid. */
export function fernandezSurface(grid, px, py, vx, vy, bx, by, out, config = CONFIG) {
  const target = out || new Float64Array(grid.nx * grid.ny);
  const speed = Math.hypot(vx, vy);
  const distanceToBall = Math.hypot(px - bx, py - by);
  const radius = influenceRadius(distanceToBall);
  const speedRatio = Math.min(1.0, (speed / config.maxInfluenceSpeed) ** 2);
  // Python computes the angle in degrees and converts back; mirror that.
  const angle = speed > 1e-12 ? (Math.atan2(vy, vx) * 180) / Math.PI * (Math.PI / 180) : 0.0;
  const cx = px + 0.5 * vx;
  const cy = py + 0.5 * vy;
  const major = radius * (1.0 + speedRatio);
  const minor = radius * (1.0 - speedRatio);
  const sigmaAlong = Math.max(major / 2.0, 1e-6);
  const sigmaLateral = Math.max(minor / 2.0, 1e-6);
  const cos = Math.cos(angle);
  const sin = Math.sin(angle);

  const pdx = px - cx;
  const pdy = py - cy;
  const playerAlong = pdx * cos + pdy * sin;
  const playerLateral = -pdx * sin + pdy * cos;
  const playerMahalanobis =
    (playerAlong / sigmaAlong) ** 2 + (playerLateral / sigmaLateral) ** 2;

  for (let iy = 0; iy < grid.ny; iy += 1) {
    const dy = grid.y[iy] - cy;
    const rowOffset = iy * grid.nx;
    for (let ix = 0; ix < grid.nx; ix += 1) {
      const dx = grid.x[ix] - cx;
      const along = dx * cos + dy * sin;
      const lateral = -dx * sin + dy * cos;
      const mahalanobis = (along / sigmaAlong) ** 2 + (lateral / sigmaLateral) ** 2;
      let value = Math.exp(-0.5 * (mahalanobis - playerMahalanobis));
      if (value < 0) value = 0;
      else if (value > 1) value = 1;
      target[rowOffset + ix] = value;
    }
  }
  return target;
}

/** Goal-weighted value of space around one attacker. */
export function goalWeightedSurface(grid, ax, ay, attackingDirection, out, config = CONFIG) {
  const target = out || new Float64Array(grid.nx * grid.ny);
  const gx = attackingDirection * (config.fieldLength / 2);
  const gy = 0;
  const toGoalX = gx - ax;
  const toGoalY = gy - ay;
  const goalDistance = Math.hypot(toGoalX, toGoalY);
  let ux;
  let uy;
  if (goalDistance <= 1e-9) { ux = attackingDirection; uy = 0; }
  else { ux = toGoalX / goalDistance; uy = toGoalY / goalDistance; }
  const farthest = Math.hypot(config.fieldLength, config.fieldWidth / 2);

  for (let iy = 0; iy < grid.ny; iy += 1) {
    const dy = grid.y[iy] - ay;
    const gdy = grid.y[iy] - gy;
    const rowOffset = iy * grid.nx;
    for (let ix = 0; ix < grid.nx; ix += 1) {
      const dx = grid.x[ix] - ax;
      const along = dx * ux + dy * uy;
      const lateral = Math.abs(-dx * uy + dy * ux);
      let side = 1.0 / (1.0 + Math.exp(-along / config.goalSideSoftness));
      side = config.goalSideFloor + (1.0 - config.goalSideFloor) * side;
      const clamped = Math.min(Math.max(along, 0.0), goalDistance);
      const halfWidth = config.goalConeBaseHalfWidth + config.goalConeGrowthPerM * clamped;
      let cone = Math.exp(-0.5 * (lateral / halfWidth) ** 2);
      cone = config.goalConeFloor + (1.0 - config.goalConeFloor) * cone;
      const distanceToGoal = Math.hypot(grid.x[ix] - gx, gdy);
      let proximity = 1.0 - distanceToGoal / farthest;
      if (proximity < 0) proximity = 0;
      else if (proximity > 1) proximity = 1;
      target[rowOffset + ix] = proximity ** config.goalDistancePower * side * cone;
    }
  }
  return target;
}

/**
 * Per-frame influence for every player, computed once and reused.
 *
 * Same trade as the Python cache: the expensive part is one Gaussian per
 * player per frame, so keeping them means swapping a role is a subtract and an
 * add rather than a recomputation.
 */
export class InfluenceCache {
  constructor(scene, { every = 5, config = CONFIG } = {}) {
    this.scene = scene;
    this.config = config;
    this.grid = pitchGrid(config);
    this.cells = this.grid.nx * this.grid.ny;
    this.velocityWindow = Math.max(2, Math.round(0.4 * scene.fps));

    const step = Math.max(1, every);
    const indices = [];
    for (let i = 0; i < scene.n_frames; i += step) indices.push(i);
    if (indices[indices.length - 1] !== scene.n_frames - 1) indices.push(scene.n_frames - 1);
    this.indices = indices;
    this.times = indices.map((i) => scene.t0 + i / scene.fps);

    this.goalkeepers = new Set(scene.players.filter((p) => p.gk).map((p) => p.id));
    this._influence = new Map();      // playerId -> Float64Array[] by slot
    this._defenderSum = new Array(indices.length).fill(null);
    this._spaceValue = new Map();
    this._baseline = new Map();
  }

  slotFor(frameIndex) {
    let best = 0;
    let bestGap = Infinity;
    for (let i = 0; i < this.indices.length; i += 1) {
      const gap = Math.abs(this.indices[i] - frameIndex);
      if (gap < bestGap) { bestGap = gap; best = i; }
    }
    return best;
  }

  ballAt(index) {
    const { ball } = this.scene;
    const bx = ball.x[index];
    const by = ball.y[index];
    if (bx == null || by == null || !Number.isFinite(bx) || !Number.isFinite(by)) return null;
    return [bx, by];
  }

  influenceOf(playerId) {
    let stack = this._influence.get(playerId);
    if (stack) return stack;
    const player = this.scene.byId.get(playerId);
    stack = new Array(this.indices.length);
    for (let slot = 0; slot < this.indices.length; slot += 1) {
      const index = this.indices[slot];
      const buffer = new Float64Array(this.cells);
      const ball = this.ballAt(index);
      const px = player ? player.x[index] : null;
      const py = player ? player.y[index] : null;
      if (ball && px != null && py != null && Number.isFinite(px) && Number.isFinite(py)) {
        const [vx, vy] = velocityAt(player.x, player.y, index, this.scene.fps, this.velocityWindow);
        fernandezSurface(this.grid, px, py, vx, vy, ball[0], ball[1], buffer, this.config);
      }
      stack[slot] = buffer;
    }
    this._influence.set(playerId, stack);
    return stack;
  }

  defenderSum(slot, swap = []) {
    if (this._defenderSum[slot] === null) {
      const total = new Float64Array(this.cells);
      for (const player of this.scene.players) {
        if (player.side === "attack" || this.goalkeepers.has(player.id)) continue;
        const surface = this.influenceOf(player.id)[slot];
        for (let i = 0; i < this.cells; i += 1) total[i] += surface[i];
      }
      this._defenderSum[slot] = total;
    }
    const base = this._defenderSum[slot];
    if (!swap.length) return base;
    const out = Float64Array.from(base);
    for (const { playerId, freezeIndex, mode } of swap) {
      const factual = this.influenceOf(playerId)[slot];
      const baseline = this.baselineInfluence(playerId, freezeIndex, mode)[slot];
      for (let i = 0; i < this.cells; i += 1) {
        const value = out[i] - factual[i] + baseline[i];
        out[i] = value < 0 ? 0 : value;
      }
    }
    return out;
  }

  spaceValueOf(playerId) {
    let stack = this._spaceValue.get(playerId);
    if (stack) return stack;
    const player = this.scene.byId.get(playerId);
    stack = new Array(this.indices.length);
    for (let slot = 0; slot < this.indices.length; slot += 1) {
      const index = this.indices[slot];
      const buffer = new Float64Array(this.cells);
      const px = player ? player.x[index] : null;
      const py = player ? player.y[index] : null;
      if (px != null && py != null && Number.isFinite(px) && Number.isFinite(py)) {
        goalWeightedSurface(this.grid, px, py, this.scene.attacking_direction, buffer, this.config);
      }
      stack[slot] = buffer;
    }
    this._spaceValue.set(playerId, stack);
    return stack;
  }

  /** A defender that stopped responding at `freezeIndex`. */
  baselineInfluence(playerId, freezeIndex, mode = "hold") {
    const key = `${playerId}|${freezeIndex}|${mode}`;
    let stack = this._baseline.get(key);
    if (stack) return stack;
    const player = this.scene.byId.get(playerId);
    const factual = this.influenceOf(playerId);
    stack = new Array(this.indices.length);
    const halfL = this.config.fieldLength / 2;
    const halfW = this.config.fieldWidth / 2;
    const window = Math.max(2, Math.round(0.6 * this.scene.fps));
    let vx = 0;
    let vy = 0;
    if (mode !== "hold" && player) {
      const start = Math.max(0, freezeIndex - window);
      const seconds = Math.max((freezeIndex - start) / this.scene.fps, 1e-6);
      vx = (player.x[freezeIndex] - player.x[start]) / seconds;
      vy = (player.y[freezeIndex] - player.y[start]) / seconds;
    }
    const anchorX = player ? player.x[freezeIndex] : null;
    const anchorY = player ? player.y[freezeIndex] : null;

    for (let slot = 0; slot < this.indices.length; slot += 1) {
      const index = this.indices[slot];
      // Up to the freeze frame the baseline player *is* the real one.
      if (index <= freezeIndex || anchorX == null || !Number.isFinite(anchorX)) {
        stack[slot] = factual[slot];
        continue;
      }
      const buffer = new Float64Array(this.cells);
      const ball = this.ballAt(index);
      if (ball) {
        const steps = index - freezeIndex;
        let bx = anchorX + (vx * steps) / this.scene.fps;
        let by = anchorY + (vy * steps) / this.scene.fps;
        bx = Math.min(Math.max(bx, -halfL), halfL);
        by = Math.min(Math.max(by, -halfW), halfW);
        fernandezSurface(this.grid, bx, by, vx, vy, ball[0], ball[1], buffer, this.config);
      }
      stack[slot] = buffer;
    }
    this._baseline.set(key, stack);
    return stack;
  }

  /** Goal-weighted space the target keeps after defensive coverage. */
  residual(targetId, slot, swap = []) {
    const field = new Float64Array(this.cells);
    const player = this.scene.byId.get(targetId);
    if (!player) return { field, value: 0 };
    const influence = this.influenceOf(targetId)[slot];
    const spaceValue = this.spaceValueOf(targetId)[slot];
    const covered = this.defenderSum(slot, swap);
    const k = this.config.defenderSuppression;
    let total = 0;
    for (let i = 0; i < this.cells; i += 1) {
      const value = influence[i] * spaceValue[i] * Math.exp(-k * covered[i]);
      field[i] = value;
      total += value;
    }
    return { field, value: total * this.grid.cellArea };
  }

  residualSeries(targetId, swap = []) {
    const out = new Float64Array(this.indices.length);
    for (let slot = 0; slot < this.indices.length; slot += 1) {
      out[slot] = this.residual(targetId, slot, swap).value;
    }
    return out;
  }

  /** Several targets: fields combined by maximum, values summed. */
  combined(targetIds, slot, swap = []) {
    const field = new Float64Array(this.cells);
    let value = 0;
    for (const targetId of targetIds) {
      const one = this.residual(targetId, slot, swap);
      value += one.value;
      for (let i = 0; i < this.cells; i += 1) {
        if (one.field[i] > field[i]) field[i] = one.field[i];
      }
    }
    return { field, value };
  }
}
