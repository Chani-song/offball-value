// The solver's own release quantities, as the research code computed them.
//
//     legal = inside pitch AND not offside
//     release payoff = legal ? completion proxy x positional threat : 0
//
// Nothing here computes any of that. demo_viz/web/export_release.py runs the
// research implementation -- Andrew's 33-feature logistic, Kyuhyeok's threat
// and offside rule -- and this module reads the numbers it produced.
//
// The targets are the demo's own five rays, not the solver's action space. The
// values are real; the geometry is ours, and the interface says so.

const RELEASE_BASE = "data/release";

const cache = new Map();
let modelCard = null;

const key = (sceneId) => sceneId.replace(/[^A-Za-z0-9]/g, "_");

/** What the model artifact says about itself, for the Source panel. */
export async function loadModelCard() {
  if (modelCard !== null) return modelCard || null;
  try {
    const response = await fetch(`${RELEASE_BASE}/model.json`);
    modelCard = response.ok ? await response.json() : false;
  } catch (error) {
    modelCard = false;
  }
  return modelCard || null;
}

/**
 * Precomputed release quantities for a scene, or null when the export was
 * skipped -- which is the normal state of a checkout without the research
 * stack, not an error.
 */
export async function loadRelease(sceneId) {
  if (cache.has(sceneId)) return cache.get(sceneId);
  let payload = null;
  try {
    const response = await fetch(`${RELEASE_BASE}/${key(sceneId)}.json`);
    if (response.ok) {
      const raw = await response.json();
      const byFrame = new Map();
      for (const frame of raw.frames || []) byFrame.set(frame.index, frame);
      payload = { ...raw, byFrame, indices: (raw.frames || []).map((f) => f.index) };
    }
  } catch (error) {
    payload = null;
  }
  cache.set(sceneId, payload);
  return payload;
}

/** The nearest exported frame, which is where the numbers were computed. */
function nearestFrame(payload, index) {
  if (!payload?.indices?.length) return null;
  let best = payload.indices[0];
  let distance = Math.abs(best - index);
  for (const candidate of payload.indices) {
    const gap = Math.abs(candidate - index);
    if (gap < distance) { best = candidate; distance = gap; }
  }
  return { frame: payload.byFrame.get(best), sampled: best, requested: index };
}

/**
 * The four quantities for one exploratory ray.
 *
 * Returns a reason instead of a number whenever one is missing, so the panel
 * can say why rather than showing a blank or a zero that looks computed.
 */
export function releaseFor(payload, index, receiverId, rayIndex) {
  if (!payload) return { available: false, reason: "Pass model not exported for this build" };
  if (!receiverId) {
    return { available: false,
             reason: "Select a receiving attacker to inspect the solver pass model." };
  }
  const found = nearestFrame(payload, index);
  if (!found?.frame) return { available: false, reason: "No carrier at this frame" };
  const rows = found.frame.receivers?.[receiverId];
  if (!rows) {
    return { available: false,
             reason: "The selected receiver is not on the pitch at this frame" };
  }
  const row = rows[rayIndex];
  if (!row) return { available: false, reason: "No such target" };

  const columns = payload.columns;
  const value = (name) => row[columns.indexOf(name)];
  const features = {};
  for (const name of columns.slice(5)) features[name] = value(name);
  return {
    available: true,
    carrier: found.frame.carrier,
    sampledFrame: found.sampled,
    exact: found.sampled === found.requested,
    legal: Boolean(value("legal")),
    offside: Boolean(value("offside")),
    completionProxy: value("completion_proxy"),
    positionalThreat: value("positional_threat"),
    releasePayoff: value("release_payoff"),
    features,
    labels: payload.feature_labels || {},
  };
}

/** All five rays at once, for drawing. Entries may be unavailable. */
export function releaseRow(payload, index, receiverId) {
  return [0, 1, 2, 3, 4].map((ray) => releaseFor(payload, index, receiverId, ray));
}
