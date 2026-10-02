// Who is on the ball, and the demo's own five exploratory pass directions.
//
// Split out of obso.js because none of this is OBSO. The carrier rule is read
// off the tracking and is needed on every render -- the story roles, the
// dilemma's action names and the pass explorer all ask who has the ball -- so
// it belongs in the first load, while the OBSO surfaces do not: the public
// interface has no control that can request them (its threat view was removed
// when the demo went solver-native) and they are ~8 KB of parse a visitor
// never uses. obso.js re-exports these so its parity tests are untouched.

/**
 * How far a candidate ray runs, in metres. One number, fixed, documented: a
 * plausible forward pass, not an optimised or predicted distance.
 */
export const CANDIDATE_DISTANCE_M = 25;

/** The five directions, in degrees either side of the attacking direction. */
export const CANDIDATE_ANGLES_DEG = [-45, -22.5, 0, 22.5, 45];

/**
 * A carrier must be this close to the ball to count as holding it.
 *
 * A drawing threshold, looser than the pipeline's 1.5 m gate: tracking records
 * the torso while a dribbler pushes the ball a couple of metres ahead of it.
 * See demo_viz/core/candidates.py, which states the same rule.
 */
export const CARRIER_MAX_M = 2.5;

/** ...and this much closer than the next player, or the ball is contested. */
export const CARRIER_MARGIN_M = 0.5;

/**
 * Who is on the ball, or null when nobody clearly is.
 *
 * The margin is the honest part:
 * during a pass the ball is in flight and nobody holds it, and two players
 * converging on it are contesting it, not carrying it. Both cases return
 * null so the caller can hide the fan rather than invent a passer.
 */
export function carrierAt(scene, index) {
  const bx = scene.ball.x[index];
  const by = scene.ball.y[index];
  if (!Number.isFinite(bx) || !Number.isFinite(by)) return null;

  let best = null;
  let bestDistance = Infinity;
  let runnerUp = Infinity;
  for (const player of scene.players) {
    const x = player.x[index];
    const y = player.y[index];
    if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
    const distance = Math.hypot(x - bx, y - by);
    if (distance < bestDistance) {
      runnerUp = bestDistance;
      bestDistance = distance;
      best = player;
    } else if (distance < runnerUp) {
      runnerUp = distance;
    }
  }
  if (!best || bestDistance > CARRIER_MAX_M) return null;
  if (runnerUp - bestDistance < CARRIER_MARGIN_M) return null;
  if (best.side !== "attack") return null;
  return best;
}

/** Clip a ray to the pitch rectangle, returning the endpoint. */
function clipToPitch(x0, y0, x1, y1, length, width) {
  const halfL = length / 2;
  const halfW = width / 2;
  let t = 1;
  const dx = x1 - x0;
  const dy = y1 - y0;
  if (dx > 1e-9) t = Math.min(t, (halfL - x0) / dx);
  if (dx < -1e-9) t = Math.min(t, (-halfL - x0) / dx);
  if (dy > 1e-9) t = Math.min(t, (halfW - y0) / dy);
  if (dy < -1e-9) t = Math.min(t, (-halfW - y0) / dy);
  t = Math.max(0, Math.min(1, t));
  return [x0 + dx * t, y0 + dy * t];
}

/**
 * Five candidate pass directions from the carrier.
 *
 * A fixed geometric rule: a 90 degree sector centred on the attacking
 * direction, cut into five evenly spaced rays of one fixed length, clipped to
 * the pitch. These are not ranked, not scored for completion, and not
 * optimised -- they are five directions a forward pass could point, drawn so
 * the threat field behind them can be read.
 */
export function candidatePasses(scene, index) {
  const carrier = carrierAt(scene, index);
  if (!carrier) return null;
  const x0 = carrier.x[index];
  const y0 = carrier.y[index];
  const [length, width] = scene.pitch;
  const heading = scene.attacking_direction >= 0 ? 0 : Math.PI;

  return {
    carrier,
    origin: [x0, y0],
    rays: CANDIDATE_ANGLES_DEG.map((degrees) => {
      const angle = heading + (degrees * Math.PI) / 180;
      const [x1, y1] = clipToPitch(
        x0, y0,
        x0 + Math.cos(angle) * CANDIDATE_DISTANCE_M,
        y0 + Math.sin(angle) * CANDIDATE_DISTANCE_M,
        length, width,
      );
      return { degrees, end: [x1, y1] };
    }),
  };
}
