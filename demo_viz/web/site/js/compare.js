// What the tracking says about a play, for "Compare players".
//
// Read, never computed here. The marking distance and the reaction time come
// from `demo_viz/web/export_compare.py`, which runs the repository's own
// `dynamic_marking.marking_sample` and `role_logic.defender_reaction_index`;
// the space a teammate has is the influence cache's own number, which the
// browser already computes exactly. Nothing in this module is an extraction
// criterion: the showcase's triplets were picked by hand, so these are
// observations about a play, not a record of how it was chosen.

const cache = new Map();

/** The scene's evidence file, or null while it is on its way / absent. */
export async function loadCompare(sceneId) {
  if (cache.has(sceneId)) return cache.get(sceneId);
  const name = sceneId.replace(/[^A-Za-z0-9_-]+/g, "_");
  try {
    const response = await fetch(`data/compare/${name}.json`);
    const payload = response.ok ? await response.json() : null;
    cache.set(sceneId, payload);
    return payload;
  } catch {
    cache.set(sceneId, null);
    return null;
  }
}

/**
 * One defender's row against one runner, or null.
 *
 * `runnerId` defaults to the curated runner. Every outfield attacker has his
 * own set of rows, so asking about someone else's run gets numbers measured
 * for that pairing rather than the curated one's borrowed.
 */
export function defenderRow(payload, playerId, runnerId = null) {
  if (!payload) return null;
  // `by_runner` is the one that carries the per-frame series, so the curated
  // runner is looked up there too and only falls back to the flat list
  const id = runnerId || payload.runner;
  const rows = (payload.by_runner && payload.by_runner[id])
    || payload.defenders || [];
  return rows.find((row) => row.id === playerId) || null;
}

/**
 * The marking cost at one frame, not the window's mean.
 *
 * `marking_dm` is `weighted_error_m` in decimetres, frame by frame, exactly as
 * `offball_value.dynamic_marking` produced it. Where a frame has no sample --
 * a player off the tracking -- there is no number to show, and the window mean
 * stands in for the rows that predate the series.
 */
export function markingAt(row, frame) {
  if (!row) return null;
  const series = row.marking_dm;
  if (!Array.isArray(series)) return row.marking_distance_m ?? null;
  const value = series[Math.max(0, Math.min(frame, series.length - 1))];
  return value == null ? null : value / 10;
}

/**
 * The reaction, in words.
 *
 * Zero under the pursuit rule does not mean he reacted instantly: it means the
 * rule's condition -- moving at 1.5 m/s with his motion already pointing at
 * the runner's goal side -- was true at the moment the run began. Saying "0.0 s"
 * there would claim a reaction the tracking never shows starting.
 */
export function reactionText(row, secondsUntil = null) {
  if (!row || row.reaction_s == null) return "never commits";
  if (row.reaction_s <= 0) return "already pursuing";
  // the commit is one event in the tracking, so the honest per-frame reading
  // is the time left until it: a countdown while the playhead is short of it,
  // and the measured reaction once it has happened. Nothing is re-measured.
  if (secondsUntil != null && secondsUntil > 0.05) {
    return `in ${secondsUntil.toFixed(1)} s`;
  }
  return `${row.reaction_s.toFixed(1)} s`;
}

/** Seconds from `frame` to the frame where he commits, or null. */
export function secondsToReaction(payload, row, frame, times) {
  if (!row || row.reaction_index == null || !times?.length) return null;
  const at = times[Math.max(0, Math.min(frame, times.length - 1))];
  const when = times[Math.max(0, Math.min(row.reaction_index, times.length - 1))];
  return Number.isFinite(at) && Number.isFinite(when) ? when - at : null;
}

/** A metre value, at the precision the export carries. */
export function metres(value) {
  return Number.isFinite(value) ? `${value.toFixed(1)} m` : "—";
}
