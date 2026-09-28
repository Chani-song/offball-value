// Kinematic reachable area: a faithful port of action_space.solve_endpoint_motion.
//
// The player uses one constant two-dimensional acceleration for `tau` seconds
// and then continues at the resulting velocity. Among feasible solutions the
// longest acceleration phase is used, because it needs the smallest
// acceleration magnitude for the same endpoint subject to the terminal speed
// cap. Every constant and every branch below matches the Python; a harness
// checks the two agree endpoint by endpoint.
//
// This is MODEL REACHABILITY -- where the motion model says a player could
// arrive within the horizon -- not an observed or estimated probability.
//
// The repository also implements a richer steering model (steering_reachable:
// tangential and normal control, with plant-and-cut), which the defender
// response pipeline uses. It costs about 9.5 s for one player at one frame, so
// it cannot run here and is not what this layer draws. The Source panel says so.

export const REACH = {
  horizonSeconds: 2.0,
  gridResolutionM: 1.0,
  maxSpeed: 9.0,
  maxAcceleration: 3.5,
  maxDeceleration: 3.5,
  velocityHistorySeconds: 0.4,
  fieldLength: 105.0,
  fieldWidth: 68.0,
  tolerance: 1e-8,
};

function inPitch(x, y, config) {
  const t = config.tolerance;
  return (
    x >= -config.fieldLength / 2 - t && x <= config.fieldLength / 2 + t
    && y >= -config.fieldWidth / 2 - t && y <= config.fieldWidth / 2 + t
  );
}

/** Exact coordinate extrema of the parabolic acceleration phase. */
function trajectoryStaysInPitch(start, velocity, acceleration, duration, end, config) {
  if (!inPitch(start[0], start[1], config)) return false;
  if (!inPitch(end[0], end[1], config)) return false;

  const bounds = [];
  for (let axis = 0; axis < 2; axis += 1) {
    const p = start[axis];
    const v = velocity[axis];
    const a = acceleration[axis];
    const values = [p, p + v * duration + 0.5 * a * duration * duration];
    if (Math.abs(a) > config.tolerance) {
      const turning = -v / a;
      if (turning > 0 && turning < duration) {
        values.push(p + v * turning + 0.5 * a * turning * turning);
      }
    }
    bounds.push([Math.min(...values), Math.max(...values)]);
  }
  const xBound = config.fieldLength / 2 + config.tolerance;
  const yBound = config.fieldWidth / 2 + config.tolerance;
  return (
    bounds[0][0] >= -xBound && bounds[0][1] <= xBound
    && bounds[1][0] >= -yBound && bounds[1][1] <= yBound
  );
}

const failed = (reason) => ({ feasible: false, reason });

/**
 * Can this player reach `endpoint` within the horizon?
 *
 * Returns the same fields the Python dataclass carries, so the parity harness
 * can compare them directly.
 */
export function solveEndpointMotion(start, velocity, endpoint, config = REACH) {
  const finite = [...start, ...velocity, ...endpoint].every(Number.isFinite);
  if (!finite) return failed("non_finite");
  if (!inPitch(endpoint[0], endpoint[1], config)) return failed("endpoint_outside_pitch");

  const [vx, vy] = velocity;
  const initialSpeed = Math.hypot(vx, vy);
  if (initialSpeed > config.maxSpeed + config.tolerance) {
    return failed("initial_speed_above_cap");
  }

  const horizon = config.horizonSeconds;
  const deltaX = endpoint[0] - start[0] - vx * horizon;
  const deltaY = endpoint[1] - start[1] - vy * horizon;
  const deltaNorm = Math.hypot(deltaX, deltaY);

  if (deltaNorm <= config.tolerance) {
    if (!trajectoryStaysInPitch(start, velocity, [0, 0], 0, endpoint, config)) {
      return failed("trajectory_leaves_pitch");
    }
    return {
      feasible: true, accelerationX: 0, accelerationY: 0, acceleration: 0,
      accelerationDuration: 0, cruiseDuration: horizon,
      terminalVx: vx, terminalVy: vy, terminalSpeed: initialSpeed,
    };
  }

  const maximumDisplacement = 0.5 * config.maxAcceleration * horizon * horizon;
  if (deltaNorm > maximumDisplacement + config.tolerance) return failed("acceleration_limit");

  const radicand = Math.max(0, horizon * horizon - (2 * deltaNorm) / config.maxAcceleration);
  const minimumTau = horizon - Math.sqrt(radicand);
  const qMin = 1 / (horizon - 0.5 * minimumTau);
  const qMax = 2 / horizon;

  // |v0 + q*delta| <= vmax; the initial speed is already inside the cap, so the
  // positive quadratic root is the binding bound
  const a = deltaNorm * deltaNorm;
  const b = 2 * (vx * deltaX + vy * deltaY);
  const c = initialSpeed * initialSpeed - config.maxSpeed * config.maxSpeed;
  let discriminant = b * b - 4 * a * c;
  if (discriminant < -config.tolerance) return failed("speed_limit");
  discriminant = Math.max(0, discriminant);
  const positiveRoot = (-b + Math.sqrt(discriminant)) / (2 * a);
  let q = Math.min(qMax, positiveRoot);
  if (q < qMin - config.tolerance || q <= 0) return failed("speed_limit");
  q = Math.max(q, qMin);

  let duration = 2 * (horizon - 1 / q);
  duration = Math.min(horizon, Math.max(minimumTau, duration));
  const denominator = duration * (horizon - 0.5 * duration);
  if (denominator <= config.tolerance) return failed("degenerate_trajectory");

  const ax = deltaX / denominator;
  const ay = deltaY / denominator;
  const acceleration = Math.hypot(ax, ay);
  const terminalVx = vx + ax * duration;
  const terminalVy = vy + ay * duration;
  const terminalSpeed = Math.hypot(terminalVx, terminalVy);

  if (acceleration > config.maxAcceleration + 1e-6) return failed("acceleration_limit");
  if (terminalSpeed > config.maxSpeed + 1e-6) return failed("speed_limit");
  if (!trajectoryStaysInPitch(start, velocity, [ax, ay], duration, endpoint, config)) {
    return failed("trajectory_leaves_pitch");
  }
  return {
    feasible: true, accelerationX: ax, accelerationY: ay, acceleration,
    accelerationDuration: duration, cruiseDuration: horizon - duration,
    terminalVx, terminalVy, terminalSpeed,
  };
}

/**
 * The reachable set on the action grid, as a flat mask over a pitch grid.
 *
 * The grid is the action-space grid (1 m), not the OBSO grid: the feasibility
 * question is asked at the resolution the research code asks it at.
 */
export function reachableMask(start, velocity, config = REACH) {
  const nx = Math.ceil(config.fieldLength / config.gridResolutionM);
  const ny = Math.ceil(config.fieldWidth / config.gridResolutionM);
  const grid = { nx, ny, length: config.fieldLength, width: config.fieldWidth };
  const values = new Float64Array(nx * ny);
  const dx = config.fieldLength / nx;
  const dy = config.fieldWidth / ny;
  let reached = 0;

  for (let iy = 0; iy < ny; iy += 1) {
    const y = iy * dy - config.fieldWidth / 2 + dy / 2;
    for (let ix = 0; ix < nx; ix += 1) {
      const x = ix * dx - config.fieldLength / 2 + dx / 2;
      const motion = solveEndpointMotion(start, velocity, [x, y], config);
      if (motion.feasible) {
        values[iy * nx + ix] = 1;
        reached += 1;
      }
    }
  }
  return { grid, values, cells: reached, areaM2: reached * dx * dy };
}
