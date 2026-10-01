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

/** One defender's row, or null when this scene's evidence does not cover him. */
export function defenderRow(payload, playerId) {
  if (!payload) return null;
  return (payload.defenders || []).find((row) => row.id === playerId) || null;
}

/**
 * The reaction, in words.
 *
 * Zero under the pursuit rule does not mean he reacted instantly: it means the
 * rule's condition -- moving at 1.5 m/s with his motion already pointing at
 * the runner's goal side -- was true at the moment the run began. Saying "0.0 s"
 * there would claim a reaction the tracking never shows starting.
 */
export function reactionText(row) {
  if (!row || row.reaction_s == null) return "never commits";
  if (row.reaction_s <= 0) return "already pursuing";
  return `${row.reaction_s.toFixed(1)} s`;
}

/** A metre value, at the precision the export carries. */
export function metres(value) {
  return Number.isFinite(value) ? `${value.toFixed(1)} m` : "—";
}
