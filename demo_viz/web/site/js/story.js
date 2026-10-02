// The paper-story contract, as the browser sees it.
//
// Everything the story panel shows arrives through here in one shape, defined
// by demo_viz/paper_story/schema.py and produced by its adapter. This module
// knows nothing about solver run directories, rows.jsonl or any upstream
// variable name -- that is the whole point of the boundary.
//
// The invariant the Python side enforces at construction, restated because the
// rendering depends on it:
//
//     availability === "available"  =>  value !== null  and  source is set
//     availability !== "available"  =>  value === null
//
// So there is no code path here that can print a number that nothing computed.
// A field whose method does not exist yet renders its reason.

const STORY_BASE = "data/story";

const cache = new Map();
let contract = null;

const key = (sceneId) => sceneId.replace(/[^A-Za-z0-9]/g, "_");

/** The vocabulary: availability states and their wording. Fetched once. */
export async function loadContract() {
  if (contract !== null) return contract || null;
  try {
    const response = await fetch(`${STORY_BASE}/contract.json`);
    contract = response.ok ? await response.json() : false;
  } catch (error) {
    contract = false;
  }
  return contract || null;
}

/**
 * The story payload for a scene, or null when this build has none.
 *
 * Null is a normal state, not an error: a checkout that has not run
 * export_paper_story simply shows the observed story.
 */
export async function loadStory(sceneId) {
  if (cache.has(sceneId)) return cache.get(sceneId);
  let payload = null;
  try {
    const response = await fetch(`${STORY_BASE}/${key(sceneId)}.json`);
    if (response.ok) payload = await response.json();
  } catch (error) {
    payload = null;
  }
  cache.set(sceneId, payload);
  return payload;
}

/**
 * What to say when a field has no value.
 *
 * The wording is the contract's, shipped with it, so the five reasons stay
 * five reasons. Collapsing them into one dash is what this exists to prevent.
 */
export function availabilityLabel(availability) {
  const labels = (contract && contract.availability_label) || {};
  return labels[availability] || "Not available";
}

/**
 * A metric's value as text, or its reason. Never both, never neither.
 *
 * Formatting follows the value's own type: an integer prints as an integer, a
 * float to three places. Nothing here decides that `relative_rank` is a
 * fraction or an ordinal -- that is the producer's definition, and guessing it
 * would print "3.000" for a rank of 3.
 */
export function metricText(metric) {
  if (!metric) return { available: false, text: "" };
  if (metric.availability !== "available") {
    return { available: false,
             text: metric.detail || availabilityLabel(metric.availability) };
  }
  const value = metric.value;
  let text;
  if (typeof value === "number") {
    text = Number.isInteger(value) ? String(value) : value.toFixed(3);
  } else if (typeof value === "boolean") {
    text = value ? "Yes" : "No";
  } else {
    text = String(value);
  }
  return { available: true, text: metric.unit ? `${text} ${metric.unit}` : text };
}

/** An action's description, or its reason. */
export function actionText(action) {
  if (!action) return { available: false, text: "" };
  if (action.availability !== "available") {
    return { available: false,
             text: action.detail || availabilityLabel(action.availability) };
  }
  const parts = [];
  if (action.description) parts.push(action.description);
  else if (action.action_id) parts.push(action.action_id);
  if (action.kind && !action.description) parts.push(`(${action.kind})`);
  if (action.projection_distance_m != null) {
    parts.push(`· matched within ${action.projection_distance_m.toFixed(1)} m`);
  }
  return { available: true, text: parts.join(" ") || "—" };
}

/**
 * The hover text for a value: who produced it, and under which definition.
 *
 * Shown for real values only. A definition version is carried per metric so a
 * changed upstream definition is visible rather than silently reusing a label.
 */
export function sourceTitle(record) {
  if (!record || record.availability !== "available") return "";
  const bits = [record.source];
  if (record.definition_version) bits.push(`definition ${record.definition_version}`);
  if (record.aggregation) bits.push(record.aggregation);
  if (record.provenance) bits.push(record.provenance.replace(/_/g, " "));
  return bits.filter(Boolean).join(" · ");
}

/** Only the series a payload actually carries values for. */
export function presentSeries(block) {
  return (block?.frame_series || []).filter((s) => s.availability === "available");
}

/** Every reserved series, so an empty state can say what is coming. */
export function reservedSeries(block) {
  return block?.frame_series || [];
}
