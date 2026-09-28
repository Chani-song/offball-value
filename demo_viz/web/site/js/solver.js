// Loading real solver output, and saying plainly when there is none.
//
// The solver is a separate batch job (Andrew's exact 2v1 game, mit_ssac2027).
// Nothing here computes anything: it fetches an artifact that the solver
// actually produced, exported by demo_viz/web/export_solver.py with the run's
// provenance attached.
//
// A scene with no artifact returns {available: false}. There is deliberately no
// fallback path that draws something else -- a generated trajectory is not a
// solver result, and the two must never be confusable on screen.

const SOLVER_BASE = "data/solver";

const cache = new Map();

const key = (sceneId) => sceneId.replace(/[^A-Za-z0-9]/g, "_");

/** Real solver output for a scene, or an unavailable state with a reason. */
export async function loadSolver(sceneId) {
  if (cache.has(sceneId)) return cache.get(sceneId);
  let result = { available: false, reason: "Not computed for this scene" };
  try {
    const response = await fetch(`${SOLVER_BASE}/${key(sceneId)}.json`);
    if (response.ok) {
      const raw = await response.json();
      result = raw && raw.available ? raw
        : { available: false, reason: raw?.reason || "Not computed for this scene" };
    }
  } catch (error) {
    result = { available: false, reason: "Not computed for this scene" };
  }
  cache.set(sceneId, result);
  return result;
}

/** The five movement commands, named for what they do on the pitch. */
export function commandName(direction, attackDirection) {
  if (!direction) return "—";
  const [dx, dy] = direction;
  if (dx === 0 && dy === 0) return "hold";
  if (dx !== 0) {
    const forward = (dx > 0) === (attackDirection >= 0);
    return forward ? "toward goal" : "toward own goal";
  }
  return dy > 0 ? "left touchline" : "right touchline";
}

/**
 * The rollout a reviewer should see first.
 *
 * The solver writes several sampled rollouts of the same solved policy. Where
 * they differ it is because the policy is mixed, so the longest one is shown
 * and the panel reports how many there are rather than averaging them into a
 * path that no rollout actually took.
 */
export function primaryTrajectory(state) {
  if (!state?.trajectories?.length) return null;
  return state.trajectories.reduce(
    (best, current) => (current.times.length > best.times.length ? current : best),
    state.trajectories[0],
  );
}

/** How the solved value decomposes, for the analysis panel. */
export function summary(state) {
  if (!state?.available) return null;
  const certificate = state.certificate || {};
  return {
    value: state.value,
    gap: certificate.gap,
    mixed: state.is_mixed,
    releaseProbability: state.release_probability,
    attack: state.most_likely_attack,
    defence: state.most_likely_defence,
    steps: state.steps,
    stepSeconds: state.step_seconds,
    rollouts: state.trajectories?.length ?? 0,
    provenance: state.provenance,
  };
}
