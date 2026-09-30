// The curated submission showcase: which scenes lead, and how they are labelled.
//
// Curation metadata only. A showcase entry is played from the repository scene
// it maps to; entries without a verified mapping carry no trajectory and are
// listed with the reason instead of being shown. Reviewer ratings are human
// ratings -- never a model score, never a probability.

const SHOWCASE_FILE = "data/submission_showcase.json";

/** Public wording. The source's own strong/medium/solver stay in metadata. */
export const PROVENANCE_LABEL = {
  human: "Human-reviewed",
  solver: "Solver-derived",
};

let loaded = null;

/** The showcase, or null when the build does not carry one. */
export async function loadShowcase() {
  if (loaded !== null) return loaded;
  try {
    const response = await fetch(SHOWCASE_FILE);
    if (!response.ok) { loaded = false; return null; }
    const raw = await response.json();
    const scenes = (raw.scenes || []).map((scene) => ({
      ...scene,
      playable: scene.mapping?.status === "verified" && Boolean(scene.scene_id),
      ratings: (scene.review?.reviewers || [])
        .map((r) => r.rating).filter((r) => typeof r === "number"),
    }));
    loaded = scenes.length ? scenes : false;
    return loaded || null;
  } catch (error) {
    loaded = false;
    return null;
  }
}

/** `featured` once anything is marked, otherwise `all`. */
export function defaultFilter(scenes) {
  return scenes.some((s) => s.featured) ? "featured" : "all";
}

/**
 * The filters worth offering, given what the metadata actually contains.
 * A filter that would match nothing is not shown.
 */
export function availableFilters(scenes) {
  const filters = [];
  const count = (predicate) => scenes.filter(predicate).length;
  if (count((s) => s.featured)) filters.push({ key: "featured", label: "Featured" });
  filters.push({ key: "all", label: "All" });
  if (count((s) => topRating(s) === "5/5")) filters.push({ key: "5/5", label: "5/5" });
  for (const kind of ["human", "solver"]) {
    if (count((s) => s.provenance === kind)) {
      filters.push({ key: kind, label: PROVENANCE_LABEL[kind] });
    }
  }
  for (const scenario of ["2v1", "3v1"]) {
    if (count((s) => s.solver?.scenario_type === scenario)) {
      filters.push({ key: scenario, label: scenario });
    }
  }
  return filters;
}

function topRating(scene) {
  return scene.review?.agreement_label || "";
}

export function matches(scene, filter) {
  switch (filter) {
    case "featured": return Boolean(scene.featured);
    case "all": return true;
    case "5/5": return topRating(scene) === "5/5";
    case "human":
    case "solver": return scene.provenance === filter;
    case "2v1":
    case "3v1": return scene.solver?.scenario_type === filter;
    default: return true;
  }
}

export function filtered(scenes, filter) {
  const kept = scenes.filter((s) => matches(s, filter));
  return kept.sort((a, b) => {
    if (a.featured !== b.featured) return a.featured ? -1 : 1;
    const ao = a.order ?? Infinity;
    const bo = b.order ?? Infinity;
    if (ao !== bo) return ao - bo;
    return a.showcase_id.localeCompare(b.showcase_id);
  });
}

/** One concise line for the selector: "S05 · 5/5 · Human-reviewed". */
export function selectorLabel(scene) {
  // A curated story title names the case study when one has been written.
  // Until then the fixture does: "S20 · Leverkusen vs Bochum" reads as a
  // football match, where "S20" alone reads as a database key. Nothing
  // generates a title, so an uncurated scene never acquires a fabricated one --
  // it falls back to metadata it already has.
  const fixture = (scene.match || "").split("·")[0].trim();
  const head = scene.story_title || fixture;
  const parts = [head ? `${scene.showcase_id} · ${head}` : scene.showcase_id];
  const rating = topRating(scene);
  if (rating) parts.push(rating);
  parts.push(PROVENANCE_LABEL[scene.provenance] || scene.provenance);
  if (scene.solver?.scenario_type) parts.push(scene.solver.scenario_type);
  if (!scene.playable) parts.push("scene data not published");
  return parts.join(" · ");
}

/** The fixture, without the Korean half marker the source carries. */
export function matchLabel(scene) {
  const head = (scene.match || "").split("·")[0].trim();
  const half = scene.half === "second" ? "2nd half" : "1st half";
  return head ? `${head} · ${half}` : half;
}

/**
 * Turn the hand-labelled shirt numbers into player ids for a loaded scene.
 *
 * These are the reviewer's own roles and are used as the scene's initial
 * state; nothing heuristic replaces them. A shirt that is not on the expected
 * side is dropped rather than forced, which keeps a bad mapping visible as a
 * missing chip instead of a wrong one.
 */
export function resolveRoles(scene, showcase) {
  const pick = (shirts, side) => (shirts || [])
    .map((shirt) => scene.players.find((p) => p.shirt === String(shirt) && p.side === side))
    .filter(Boolean)
    .map((p) => p.id);
  return {
    runners: pick(showcase.roles?.runners, "attack"),
    defenders: pick(showcase.roles?.defenders, "defend"),
    beneficiaries: pick(showcase.roles?.beneficiaries, "attack"),
  };
}
