// Wiring: scene loading, the role dock, drag and drop, overlays, playback.

import { InfluenceCache, velocityAt } from "./influence.js";
import {
  LAYER_LABEL, P, PAPER_THEME, ROLE_COLOUR, ROLE_LABEL, ROLE_SIDE,
} from "./palette.js";
import { Pitch } from "./pitch.js";
import {
  MODE_TITLE as PUBLIC_MODE_TITLE, PUBLIC_ROLE, PUBLIC_STORY_ROLE,
  ordinal, percent, publicActionName, publicOptionLabel, publicPassLabel,
  publicSelectorLabel, publicSubtitle,
} from "./public.js";
import { Selection } from "./selection.js";
import { ballAt, loadIndex, loadScene, onsetFor, playerAt, viewVector } from "./scene.js";
import { candidatePasses } from "./carrier.js";
import { REACH, reachableMask } from "./reach.js";
import { loadSolver, primaryTrajectory, summary as solverSummary } from "./solver.js";
import {
  PROVENANCE_LABEL, availableFilters, defaultFilter, filtered, loadShowcase,
  loadShowcaseDetail, matchLabel, resolveRoles, selectorLabel,
} from "./showcase.js";
import {
  actionText, loadContract, loadStory, metricText, presentSeries, reservedSeries,
  sourceTitle,
} from "./story.js";

const $ = (id) => document.getElementById(id);

// ---------------------------------------------------------------------------
// mode-specific components, fetched when they are first needed
// ---------------------------------------------------------------------------
// Two components belong to exactly one mode: the policy bars to Game solution,
// the evaluation strip to Evaluation. Neither is reachable in the public demo
// today -- no scene has a solved policy, and no evaluation series exists -- so
// keeping them out of the first load costs nothing now and one re-render later.
let policyModule = null;
let policyLoading = false;

/** The equilibrium policy component, or null until it arrives. */
function policyLib() {
  if (policyModule) return policyModule;
  if (!policyLoading) {
    policyLoading = true;
    import("./policy.js").then((module) => { policyModule = module; render(); });
  }
  return null;
}

let figureModule = null;
let figureLoading = false;

/**
 * The paper figures' decoders, or null until they arrive.
 *
 * Only a solver view names an action, so the conventions module stays out of
 * the first load; the shape table pitch.js needs is inlined there.
 */
function figureLib() {
  if (figureModule) return figureModule;
  if (!figureLoading) {
    figureLoading = true;
    import("./figure.js").then((module) => { figureModule = module; render(); });
  }
  return null;
}

let obsoModule = null;
let obsoLoading = false;

/**
 * The legacy OBSO stack, or null until it arrives.
 *
 * The public wake-mode control offers Available space and Space created only,
 * so nothing in the submission interface can reach this. It is kept, and kept
 * lazy: the implementation and its parity tests stay (see the Source panel's
 * "Legacy / reference diagnostic"), and a visitor never parses it.
 */
function obsoLib() {
  if (obsoModule) return obsoModule;
  if (!obsoLoading) {
    obsoLoading = true;
    import("./obso.js").then((module) => { obsoModule = module; render(); });
  }
  return null;
}

let arrowsModule = null;
let arrowsLoading = false;

/** The figures' arrow language, or null until it arrives. */
function arrowsLib() {
  if (arrowsModule) return arrowsModule;
  if (!arrowsLoading) {
    arrowsLoading = true;
    import("./arrows.js").then((module) => { arrowsModule = module; render(); });
  }
  return null;
}

let panelModule = null;
let panelLoading = false;

/** The public panel, fetched on the showcase's first render. */
function panelLib() {
  if (panelModule) return panelModule;
  if (!panelLoading) {
    panelLoading = true;
    import("./panel.js").then((module) => { panelModule = module; render(); });
  }
  return null;
}

/**
 * Everything the public panel reads, in one object.
 *
 * It holds no state of its own, so this is the whole seam: change what the
 * panel shows by changing these, never by reaching back into app.js.
 */
function panelContext() {
  return {
    state, el: $, NUMBER, surnameOf, SHAPE_CLASS, PUBLIC_ROLE, ROLE_COLOUR,
    compareLib, compareFor, swapSpec, solverFor, panelAt, storyFor,
    metricsAtFrame, presentSeries, publicObservedRow, ordinal, percent,
    compareWith, isPublic, solverRolesAt, freezeIndex, comparedPlayer,
    optionsLib, optionsFor, momentSeconds, solverRoleOfStory,
    publicActionName, gridLib, figureLib, viewVector0,
    setStoryRole, STORY_ROLE_LABEL, PUBLIC_STORY_ROLE, render,
  };
}

let optionsModule = null;
let optionsLoading = false;

/** The ranked-actions reader, fetched when Player evaluation opens. */
function optionsLib() {
  if (optionsModule) return optionsModule;
  if (!optionsLoading) {
    optionsLoading = true;
    import("./options.js").then((module) => { optionsModule = module; render(); });
  }
  return null;
}

/** This scene's ranked actions, fetched once. */
function optionsFor() {
  const { scene } = state;
  if (!scene) return null;
  if (state.optionData?.sceneId === scene.scene_id) return state.optionData.payload;
  const lib = optionsLib();
  if (!lib) return null;
  if (state.optionData?.pending === scene.scene_id) return null;
  state.optionData = { pending: scene.scene_id };
  lib.loadOptions(scene.scene_id).then((payload) => {
    if (state.scene?.scene_id !== scene.scene_id) return;
    state.optionData = { sceneId: scene.scene_id, payload };
    render();
  });
  return null;
}

/** The solved moment the playhead is on, in seconds. */
function momentSeconds() {
  const data = solverFor();
  if (data?.kind !== "bundle_panels") return 0;
  return panelAt(data, state.frame)?.panel?.dt ?? 0;
}

/** The story role, named as the solved panel names it. */
function solverRoleOfStory() {
  const data = solverFor();
  const kind = data?.kind === "bundle_panels" ? data.panels?.[0]?.kind : null;
  if (state.storyRole === "defender") return "defender";
  if (state.storyRole === "runner") return "runner";
  // `passer` is the 2v1's ball carrier; a 3v1's passer is scripted and has no
  // decision to evaluate, so there is nothing to show for him there
  return kind === "3v1" ? "beneficiary" : "ball carrier";
}

let compareModule = null;
let compareLoading = false;

/** The tracking-evidence reader, fetched when Dilemma first needs it. */
function compareLib() {
  if (compareModule) return compareModule;
  if (!compareLoading) {
    compareLoading = true;
    import("./compare.js").then((module) => { compareModule = module; render(); });
  }
  return null;
}

let gridModule = null;
let gridLoading = false;

/** The defender-grid reader. Dormant until the three moment files exist. */
function gridLib() {
  if (gridModule) return gridModule;
  if (!gridLoading) {
    gridLoading = true;
    import("./grid.js").then((module) => { gridModule = module; render(); });
  }
  return null;
}

let rankingModule = null;
let rankingLoading = false;

/**
 * The hint rankings, or null until they arrive.
 *
 * Only the explorer asks: the showcase's cast is curated, so nothing there
 * suggests a different one.
 */
function rankingLib() {
  if (rankingModule) return rankingModule;
  if (!rankingLoading) {
    rankingLoading = true;
    import("./ranking.js").then((module) => { rankingModule = module; render(); });
  }
  return null;
}

let chartModule = null;
let chartLoading = false;

/** The sidebar's time series, which only the explorer draws. */
function chartLib() {
  if (chartModule) return chartModule;
  if (!chartLoading) {
    chartLoading = true;
    import("./chart.js").then((module) => { chartModule = module; render(); });
  }
  return null;
}

let releaseModule = null;
let releaseLoading = false;

/**
 * The pass-model reader, or null until it arrives.
 *
 * Only the exploratory-pass layer asks for it, and that layer lives in the
 * Full explorer's Advanced section, so the showcase never fetches it.
 */
function releaseLib() {
  if (releaseModule) return releaseModule;
  if (!releaseLoading) {
    releaseLoading = true;
    import("./release.js").then((module) => { releaseModule = module; render(); });
  }
  return null;
}

let stripModule = null;
let stripLoading = false;

/** The frame-evaluation strip, or null until it arrives. */
function stripLib() {
  if (stripModule) return stripModule;
  if (!stripLoading) {
    stripLoading = true;
    import("./evalstrip.js").then((module) => { stripModule = module; render(); });
  }
  return null;
}

/** Layers whose only readout lives in the Analysis panel. */
const ANALYSIS_LAYERS = new Set(["solver", "reach", "passes"]);

const state = {
  index: [],
  effects: new Set(["strong", "medium"]),
  scene: null,
  cache: null,
  selection: new Selection(),
  frame: 0,
  playing: false,
  timer: null,
  // the Full explorer's defaults. The showcase narrows these in applyTheme's
  // sibling below: the space map and the held defender are diagnostics, and
  // they are one click away under Advanced rather than on by default.
  layers: new Set(["trail", "tether", "wake", "ghost", "labels", "candidates"]),
  view: "full",              // "full" | "focus"
  hintsOpen: null,           // null = follow the mode, true/false = user's choice
  drag: null,
  obso: null,                // { entry, score } once the threat view is used
  obsoValues: null,          // scratch buffer for the current frame's surface
  inspect: null,             // player id under the last click, for the panel
  endpoint: null,            // index of the selected candidate ray
  reach: null,               // { playerId, frame, mask } memo for one frame
  // Dilemma's "Compare players": a temporary overlay on the curated cast.
  // It never writes to state.selection, so the curated triplet -- and every
  // solver and evaluation number keyed to it -- cannot move.
  compare: null,             // { runner } | { teammate } | { defender } | null
  sweep: "defender",         // which role a click in Dilemma assigns
  compareData: null,         // { sceneId, payload } from data/compare
  space: true,               // Dilemma draws the space the run opened
  viewByMode: {},            // a view the reader picked, while they stay there
  pausedAt: null,            // the solved frame playback already paused on
  option: null,              // a ranked action the reader picked to compare
  field: false,              // Figure 2's defender-start shading and move field
  grids: null,               // { code, moments: Map<dt, grid|null> } once asked
  solver: null,              // { sceneId, state } once the solver layer is used
  showcase: null,            // curated scenes, or null when the build has none
  collection: "showcase",    // "showcase" | "explorer"
  showcaseFilter: "all",
  showcaseId: null,          // the curated entry currently open
  release: null,             // solver release quantities, once the explorer is used
  detailsOpen: false,        // the Model details disclosure
  valueOpen: false,          // the Action value details drill-down
  mode: "observed",          // the story mode: which question is being asked
  storyRole: "runner",       // runner | passer | defender
  story: null,               // the paper-story contract payload for this scene
  solverView: "policy",      // policy | actual | overlay
  advancedOpen: false,       // the Advanced layer disclosure
  showcaseLayersApplied: false,
};

// ---------------------------------------------------------------------------
// boot
// ---------------------------------------------------------------------------
/**
 * Read `?scene=&r=&d=&b=&t=` so a particular pick is linkable, and so the
 * screenshots in the docs show real states rather than mock-ups.
 * Roles accept shirt numbers or player ids; `d` and `b` take comma lists.
 */
function urlState() {
  const params = new URLSearchParams(window.location.search);
  if (![...params.keys()].length) return null;
  return {
    scene: params.get("scene"),
    runner: params.get("r"),
    defender: params.get("d"),
    beneficiary: params.get("b"),
    time: params.get("t"),
    swap: params.get("swap") === "1",
    showcase: params.get("showcase"),   // open a curated case study by its id
    explorer: params.get("explorer") === "1",  // the research interface
    fit: params.get("fit"),
    about: params.get("about") === "1",
    mode: params.get("mode"),          // space | gain | obso
    passes: params.get("passes") === "1",
    on: params.get("on"),              // comma list of layers to switch on
    inspect: params.get("inspect"),    // shirt number to open the panel on
    target: params.get("target"),      // exploratory ray index, 0-4
    story: params.get("story"),        // story mode, for linkable screenshots
    role: params.get("role"),          // runner | passer | defender
  };
}

function resolveShirts(scene, value, side) {
  if (!value) return [];
  return value.split(",").map((token) => token.trim()).filter(Boolean)
    .map((token) => {
      if (scene.byId.has(token)) return token;
      const player = scene.players.find((p) => p.side === side && p.shirt === token);
      return player ? player.id : null;
    })
    .filter(Boolean);
}

function applyUrlState(wanted) {
  const { scene } = state;
  const runners = resolveShirts(scene, wanted.runner, "attack");
  const defenders = resolveShirts(scene, wanted.defender, "defend");
  const beneficiaries = resolveShirts(scene, wanted.beneficiary, "attack");
  if (runners.length || defenders.length || beneficiaries.length) {
    state.selection = new Selection({ runners, defenders, beneficiaries, source: "manual" });
  }
  if (wanted.swap) state.selection.swapAttackRoles();
  if (wanted.fit != null) setView(wanted.fit === "1" ? "focus" : "full");
  if (wanted.role && ["runner", "passer", "defender"].includes(wanted.role)) {
    state.storyRole = wanted.role;
  }
  if (wanted.story) setMode(wanted.story, true);
  if (wanted.mode && ["space", "gain"].includes(wanted.mode)) {
    $("wake-mode").value = wanted.mode;
  }
  const wantedLayers = [
    ...(wanted.passes ? ["passes"] : []),
    ...(wanted.on ? wanted.on.split(",").map((s) => s.trim()).filter(Boolean) : []),
  ];
  for (const layer of wantedLayers) {
    state.layers.add(layer);
    const box = document.querySelector(`[data-layer="${layer}"]`);
    if (box) box.checked = true;
  }
  if (wanted.target != null && wanted.target !== "") {
    const ray = Number(wanted.target);
    if (Number.isInteger(ray) && ray >= 0 && ray < 5) state.endpoint = ray;
  }
  if (wanted.inspect) {
    const found = scene.players.find((p) => p.shirt === String(wanted.inspect));
    if (found) state.inspect = found.id;
  }
  if (wanted.time != null && wanted.time !== "") {
    const frame = Number(wanted.time);
    if (Number.isFinite(frame)) {
      state.frame = Math.min(Math.max(Math.round(frame), 0), scene.n_frames - 1);
    }
  } else if (!isPublic()) {
    const frame = peakGainFrame();
    if (frame != null) state.frame = frame;
    // an explicit ?t= is the visitor's; anything else defers to the mode
    ensureDecisionFrame();
  } else {
    // the public clip keeps the first frame `openScene` put it on; only the
    // mode may move it from there, and Play never does
    ensureDecisionFrame();
  }
}

async function boot() {
  state.pitch = new Pitch($("pitch"), {
    onPlayerDown: beginDrag,
    onBackground: () => {
      if (state.selection.armed) { state.selection.armed = false; render(); }
    },
  });
  state.index = await loadIndex();
  state.showcase = await loadShowcase();
  bindControls();
  refreshSceneList();

  const wanted = urlState();
  if (state.showcase) {
    state.showcaseFilter = defaultFilter(state.showcase);
    // a collection only makes sense when there are two of them
    $("collection-control").hidden = false;
    // ?scene=... or ?explorer=1 opens the research interface; nothing in the
    // public header does, which is what keeps the public header public
    if (wanted && (wanted.scene || wanted.explorer)) state.collection = "explorer";
    if (wanted && wanted.showcase) state.showcaseId = wanted.showcase;
  } else {
    state.collection = "explorer";
  }
  applyCollection();

  let start = null;
  if (state.collection === "showcase") {
    const curated = playableShowcase();
    const chosen = curated.find((s) => s.showcase_id === state.showcaseId) || curated[0];
    if (chosen) {
      state.showcaseId = chosen.showcase_id;
      start = state.index.find((row) => row.scene_id === chosen.scene_id);
    }
  }
  if (!start) {
    start = visibleScenes()[0];
    if (wanted && wanted.scene) {
      const match = state.index.find(
        (row) => row.file === wanted.scene || row.scene_id === wanted.scene,
      );
      if (match) start = match;
    }
  }
  if (start) await openScene(start.file, wanted);
  if (state.endpoint != null) $("an-pass")?.scrollIntoView({ block: "nearest" });
  if (wanted && wanted.about) openSheet(true);
  window.addEventListener("resize", () => render());
}

function visibleScenes() {
  return state.index.filter((row) => state.effects.has(row.effect));
}

function refreshSceneList() {
  const select = $("scene");
  const current = select.value;
  select.replaceChildren();
  for (const row of visibleScenes()) {
    const option = document.createElement("option");
    option.value = row.file;
    option.textContent = `${row.effect.toUpperCase()} · ${row.scene_id} · ${row.shape}`;
    select.appendChild(option);
  }
  if ([...select.options].some((o) => o.value === current)) select.value = current;
  $("scene-count").textContent = `${visibleScenes().length} scenes`;
}

async function openScene(file, wanted = null) {
  $("status").textContent = "loading\u2026";
  const scene = await loadScene(file);
  state.scene = scene;
  state.cache = new InfluenceCache(scene, { every: 5 });
  state.selection = Selection.fromRoles(scene.roles, "annotation");
  state.showcaseRolesApplied = state.collection === "showcase"
    ? applyShowcaseRoles(scene) : null;
  if (isPublic()) {
    // the clip's first provided frame. Not a run onset, not the shot's zero,
    // not the solver's, not release - 1.8 s, and not a peak the influence
    // model picked: a public clip plays from its own beginning, and a scene
    // with no shot opens exactly like one with a shot.
    state.frame = 0;
  } else {
    const onset = onsetFor(scene, state.selection.runner);
    state.frame = peakGainFrame() ?? Math.min(onset.index + Math.round(1.5 * scene.fps),
                                              scene.n_frames - 1);
  }
  // a new scene invalidates everything keyed to the old one; clear before the
  // URL state is applied, or it would wipe what the URL just asked for
  state.endpoint = null;
  state.option = null;
  state.optionData = null;
  state.reach = null;
  state.solver = null;
  state.release = null;
  state.story = null;
  if (wanted) applyUrlState(wanted);
  // the panel is more use open than empty, so start on the runner
  state.inspect = state.selection.runner
    || state.selection.defenders[0] || state.selection.beneficiaries[0] || null;
  $("time").max = String(scene.n_frames - 1);
  $("time").value = String(state.frame);
  $("scene").value = file;
  $("title").textContent = scene.title;
  $("subtitle").textContent = isPublic()
    ? publicSubtitle(scene) : scene.subtitle;
  $("notes").textContent = scene.notes || "—";
  $("team-attack").textContent = scene.attacking_team;
  $("team-defend").textContent = scene.defending_team;
  render();
  $("status").textContent = "";
}

// ---------------------------------------------------------------------------
// quantities
// ---------------------------------------------------------------------------
function freezeIndex() {
  return onsetFor(state.scene, state.selection.runner).index;
}

function swapSpec() {
  const { selection } = state;
  if (!selection.defenders.length) return [];
  const freeze = freezeIndex();
  return selection.defenders.map((playerId) => ({
    playerId, freezeIndex: freeze, mode: "hold",
  }));
}

function peakGainFrame() {
  const { cache, selection } = state;
  if (!cache || !selection.defenders.length || !selection.beneficiaries.length) return null;
  const swap = swapSpec();
  const freeze = freezeIndex();
  let best = null;
  let bestGain = -Infinity;
  for (let slot = 0; slot < cache.indices.length; slot += 1) {
    if (cache.indices[slot] < freeze) continue;
    const factual = cache.combined(selection.beneficiaries, slot).value;
    const counter = cache.combined(selection.beneficiaries, slot, swap).value;
    if (factual - counter > bestGain) { bestGain = factual - counter; best = cache.indices[slot]; }
  }
  return best;
}

// ---------------------------------------------------------------------------
// render
// ---------------------------------------------------------------------------
function render() {
  const { scene, cache, selection, pitch } = state;
  if (!scene) return;
  const index = state.frame;
  const slot = cache.slotFor(index);
  const swap = swapSpec();
  const freeze = freezeIndex();
  // the side that can take the pick comes forward: while a slot is armed, and
  // while a compatible player is being dragged
  const activeSide = state.drag ? state.drag.side
    : (selection.armed ? ROLE_SIDE[selection.pick] : null);

  // a solved moment carries arrowheads, labels and leader lines beyond the
  // bodies, so Nash's crop asks for more clear ground than a plain focus
  pitch.setView(state.view === "focus"
    ? fitBox(isPublic() && state.mode === "game_solution" ? 9 : 12) : null);
  pitch.clearDynamic();

  let hints = [];
  // all three roles can be compared at once, and each one the reader picked
  // keeps its ring: the alternative cast is the thing being looked at
  const comparedAll = comparedPlayers();
  const compared = comparedAll.length ? comparedAll[0] : null;
  if (comparedAll.length) hints = comparedAll;
  const ranking = isPublic() ? null : rankingLib();
  if (compared) {
    // the comparison owns the hint ring while it is open
  } else if (ranking && selection.runner && !selection.defenders.length) {
    hints = ranking.rankDefenders(scene, selection.runner, freeze)
      .slice(0, 4).map((r) => r.id);
  } else if (ranking && selection.runner && selection.defenders.length
             && !selection.beneficiaries.length) {
    hints = ranking.rankBeneficiaries(cache, scene, selection.runner,
                                      selection.defenders, freeze, slot)
      .filter((r) => r.gain > 0.05).slice(0, 3).map((r) => r.id);
  }

  // Dilemma's whole point, drawn: the space the run opened for the teammate
  // in question -- the curated one, or whoever is being compared. Same
  // quantity as "Space created", same influence cache, shown on the pitch.
  const dilemmaSpace = isPublic() && state.mode === "counterfactual"
    && state.space && selection.defenders.length;
  const spaceFor = state.compare?.teammate
    ? [state.compare.teammate] : selection.beneficiaries;
  const mode = $("wake-mode").value;
  const threat = mode === "obso" ? obsoSurface(index) : null;
  if (dilemmaSpace && spaceFor.length) {
    // THE OVERLAY IS TOTAL SPACE, not the delta: `residual_surface` from
    // `goal_weighted_influence.target_residual_influence` -- where this
    // teammate's useful space actually is. The panel's "Space created" stays
    // the delta against the held defender, so the picture answers "where is
    // the space" and the number answers "how much of it the run opened".
    // Two quantities, two jobs; a test keeps them from collapsing into one.
    pitch.drawField(cache.grid, cache.combined(spaceFor, slot).field, scene);
  } else if (state.layers.has("wake") && threat) {
    // frame-level threat: it does not move when the roles change, so it is
    // drawn in its own colour rather than the beneficiary's
    pitch.drawField(threat.grid, threat.values, scene, { colour: tint("obso") });
  } else if (state.layers.has("wake") && mode !== "obso"
             && selection.beneficiaries.length) {
    const factual = cache.combined(selection.beneficiaries, slot);
    let values = factual.field;
    if (mode === "gain" && swap.length) {
      const counter = cache.combined(selection.beneficiaries, slot, swap);
      values = factual.field.map((v, i) => Math.max(v - counter.field[i], 0));
    }
    pitch.drawField(cache.grid, values, scene);
  } else {
    pitch.drawField(cache.grid, null, scene);
  }

  // the defender field is Figure 2's background: under the players, over the
  // pitch lines, and only where the grids exist
  const realMoves = state.mode === "game_solution" && state.solverView !== "policy";
  if (realMoves && isPublic()) {
    // the paper's convention: the three bodies' real next 0.6 s, grey dotted
    // to a hollow marker -- not every player's whole-clip path
    pitch.drawRealMoves(scene, solverRolesAt(index), index, 0.6);
  } else if (state.layers.has("paths") || realMoves) {
    pitch.drawPaths(scene);
  }
  renderReach(index);
  const fan = renderPasses(index, threat);
  renderSolverLayer();
  if (state.layers.has("lane")) pitch.drawLane(scene, selection.beneficiaries, index);
  // comparing a runner moves the trail and the marking relation onto him
  const runnerId = state.compare?.runner || selection.runner;
  if (state.layers.has("trail") && runnerId) {
    pitch.drawTrail(scene, runnerId, index);
  }
  // comparing a defender moves the marking line and the held-defender ghost
  // onto him, so the click changes the pitch and not only a number
  const markers = state.compare?.defender ? [state.compare.defender]
                                          : selection.defenders;
  if (state.layers.has("tether") && runnerId) {
    pitch.drawTether(scene, runnerId, markers, index,
                     { labels: state.layers.has("labels") });
  }
  if (state.layers.has("ghost")) {
    pitch.drawGhost(scene, cache, markers, index, freeze,
                    { labels: state.layers.has("labels") });
  }
  renderDilemma(index);
  renderPolicyArrows(index);
  pitch.drawPlayers(scene, index, selection, {
    labels: state.layers.has("labels"),
    activeSide,
    hints: state.layers.has("candidates") ? hints : [],
    dragging: state.drag ? state.drag.playerId : null,
    solverRoles: solverRolesAt(index),
    // the figures mute everyone but the central actors; the numbers keep them
    muted: state.collection === "showcase"
      || state.mode === "counterfactual" || state.mode === "game_solution",
    // the whole Submission showcase speaks the paper's grammar: attack blue,
    // defence red, no outline, shape for the role. The Full explorer keeps the
    // annotation colours its role dock shows.
    paperColours: state.collection === "showcase" ? figureLib()?.ROLE_COLOR : null,
  });
  pitch.drawAttackDirection();

  renderDock();
  // the headline stat is the explorer's: the showcase drops it entirely, and
  // outside Observed its space belongs to the mode's own panel
  const stat = document.querySelector(".card.stat");
  if (stat) stat.hidden = isPublic() || state.mode !== "observed";
  renderStat(slot, swap);
  renderChart(swap, freeze);
  renderCandidates(freeze, slot);
  renderAnalysis(index, slot, threat, fan);
  renderAdvanced();
  renderMoments();
  const panelUi = isPublic() ? panelLib() : null;
  if (panelUi) {
    panelUi.renderDefenderField(panelContext(), index);
    panelUi.renderDecisionPaths(panelContext(), index);
    const context = panelContext();
    panelUi.renderLegend(context);
    panelUi.renderMetrics(context, slot);
    panelUi.renderSweep(context);
    panelUi.renderOptions(context);
  }
  renderTicks();
  renderEvalStrip(index);
  // Two clocks, and only one of them is public. `scene.times` is the solver's
  // own coordinate (zero at the annotated shot), which every solved moment,
  // every panel frame and every evaluation series is keyed to -- untouched.
  // The public timeline is clip-local: zero at the clip's first frame,
  // counting forward at the real frame rate.
  const clipTimeSec = scene.times[index] - scene.times[0];
  const solverTimeSec = scene.times[index];
  $("readout").textContent = isPublic()
    ? `${clipTimeSec.toFixed(2)} s`
    : `${solverTimeSec >= 0 ? "+" : ""}${solverTimeSec.toFixed(2)} s`;
  $("time").value = String(index);
}

/**
 * Crop to the picked players, plus the ball only while it is near them.
 * A ball cleared to the far corner would otherwise pull the box back out to a
 * full-pitch view for no gain.
 */
/**
 * Everything the Focus view may not crop.
 *
 * The ball and the man on it are in unconditionally -- a solved moment with
 * the ball off-screen is not a picture of that moment -- and so is every
 * point of every option path that will be drawn, the pass target, and the
 * real next 0.6 s. The bounds are computed from the geometry that is about to
 * be drawn, not from the players alone, which is why an arrow can no longer
 * run off the edge.
 */
function focusPoints(index) {
  const { scene, selection } = state;
  const points = [];
  const add = (xy) => { if (xy && Number.isFinite(xy[0])) points.push(xy); };
  const addPlayer = (id) => {
    const player = id && scene.byId.get(id);
    if (player) add(playerAt(scene, player, index));
  };
  for (const role of ["runner", "beneficiary", "defender"]) {
    for (const id of selection.ids(role)) addPlayer(id);
  }
  addPlayer(state.compare?.runner || state.compare?.teammate || state.compare?.defender);
  // the ball, always, and whoever is on it
  add(ballAt(scene, index));
  addPlayer(candidatePasses(scene, index)?.carrier?.id);

  const data = solverFor();
  const found = data?.kind === "bundle_panels" ? panelAt(data, index) : null;
  if (found) {
    for (const body of Object.values(found.panel.bodies || {})) {
      addPlayer(null);
      add(playerPoint(scene, body.pos));
      for (const option of body.options || []) {
        if ((option.prob ?? 1) < 0.02) continue;      // not drawn, so not framed
        for (const point of option.path || []) add(viewVector0(scene, point[0], point[1]));
      }
    }
    for (const pass of found.panel.passes || []) {
      if (pass.target) add(playerPoint(scene, pass.target));
    }
    // the real next 0.6 s, which the Actual and Both views draw
    const span = Math.round(0.6 * scene.fps);
    for (const id of Object.keys(solverRolesAt(index) || {})) {
      const player = scene.byId.get(id);
      if (player) add(playerAt(scene, player, Math.min(index + span, scene.n_frames - 1)));
    }
  }
  return points;
}

function fitBox(margin = 12, minimumWidth = 46, ballAttach = 18) {
  const { scene, selection } = state;
  let points = [];
  if (isPublic()) {
    points = focusPoints(state.frame);
  } else {
    for (const role of ["runner", "beneficiary", "defender"]) {
      for (const id of selection.ids(role)) {
        const player = scene.byId.get(id);
        if (!player) continue;
        const position = playerAt(scene, player, state.frame);
        if (position) points.push(position);
      }
    }
    if (points.length) {
      const centreX = points.reduce((a, p) => a + p[0], 0) / points.length;
      const centreY = points.reduce((a, p) => a + p[1], 0) / points.length;
      const ball = ballAt(scene, state.frame);
      if (ball && Math.hypot(ball[0] - centreX, ball[1] - centreY) <= ballAttach) {
        points.push(ball);
      }
    }
  }
  if (!points.length) return null;

  const xs = points.map((p) => p[0]);
  const ys = points.map((p) => p[1]);
  const fullW = 109.4;
  const fullH = 72;
  const width = Math.min(Math.max(Math.max(...xs) - Math.min(...xs) + 2 * margin,
                                  minimumWidth), fullW);
  const height = Math.min(Math.max(Math.max(...ys) - Math.min(...ys) + 2 * margin,
                                   minimumWidth / 1.7), fullH);
  const cx = (Math.min(...xs) + Math.max(...xs)) / 2;
  const cy = (Math.min(...ys) + Math.max(...ys)) / 2;
  let x0 = Math.min(Math.max(cx - width / 2, -fullW / 2), fullW / 2 - width);
  let y0 = Math.min(Math.max(cy - height / 2, -fullH / 2), fullH / 2 - height);
  let w = width;
  let h = height;
  // clamping the box to the pitch can push it off something it had to hold --
  // a path that runs past the touchline, say -- so the box is grown back over
  // every point it is responsible for, with the margin kept
  const pad = Math.min(margin, 6);
  const needX0 = Math.min(...xs) - pad;
  const needX1 = Math.max(...xs) + pad;
  const needY0 = Math.min(...ys) - pad;
  const needY1 = Math.max(...ys) + pad;
  if (needX0 < x0) { w += x0 - needX0; x0 = needX0; }
  if (needX1 > x0 + w) w = needX1 - x0;
  if (needY0 < y0) { h += y0 - needY0; y0 = needY0; }
  if (needY1 > y0 + h) h = needY1 - y0;
  return [x0, y0, w, h];
}

/**
 * The OBSO surface for one frame, or null when it is not loaded yet.
 *
 * Loading is lazy and asynchronous: the first frame drawn after the visitor
 * picks the threat view has nothing to show, so the load schedules one more
 * render and returns null. Every later frame is synchronous.
 */
function obsoSurface(index) {
  const { scene } = state;
  const obso = obsoLib();
  if (!scene || !obso) return null;
  const { loadControl, loadScoreGrid, surfaceAt } = obso;
  if (!state.obso || state.obso.sceneId !== scene.scene_id) {
    if (state.obso && state.obso.pending) return null;
    state.obso = { sceneId: scene.scene_id, pending: true, entry: null, score: null };
    Promise.all([loadControl(scene.scene_id), loadScoreGrid()])
      .then(([entry, score]) => {
        if (!state.scene || state.scene.scene_id !== scene.scene_id) return;
        state.obso = { sceneId: scene.scene_id, pending: false, entry, score };
        state.obsoValues = null;
        render();
      })
      .catch(() => { state.obso = { sceneId: scene.scene_id, pending: false,
                                    entry: null, score: null }; render(); });
    return null;
  }
  const { entry, score } = state.obso;
  if (!entry || !score) return null;
  if (!state.obsoValues || state.obsoValues.length !== entry.cells) {
    state.obsoValues = new Float64Array(entry.cells);
  }
  surfaceAt(entry, score, scene, index, state.obsoValues);
  return { grid: entry.grid, values: state.obsoValues };
}

/**
 * The five candidate directions, when the layer is on and somebody is clearly
 * on the ball. Nothing here is ranked or scored for completion; when the
 * threat view is on, each endpoint carries the OBSO value at that point so the
 * field behind the fan can be read.
 */
function renderPasses(index, threat) {
  const { scene, pitch } = state;
  const note = $("passes-note");
  if (!state.layers.has("passes")) {
    if (note) note.hidden = true;
    return null;
  }
  const fan = candidatePasses(scene, index);
  if (!fan) {
    if (note) { note.hidden = false; note.textContent = "No clear carrier this frame"; }
    return null;
  }
  if (note) note.hidden = true;
  // the endpoint caption is the solver's release payoff when the research
  // export exists, and nothing at all when it does not
  const payload = releaseFor_scene();
  const receiver = selectedReceiver();
  let labelled = false;
  if (payload && receiver) {
    const row = releaseLib().releaseRow(payload, index, receiver);
    for (let i = 0; i < fan.rays.length; i += 1) {
      const entry = row[i];
      if (entry.available) { fan.rays[i].obso = entry.releasePayoff; labelled = true; }
    }
  }
  pitch.drawCandidates(scene, fan, { values: labelled, selected: state.endpoint });
  return fan;
}

/** Velocity and acceleration from the tracks, the same way influence.js does. */
function motionOf(player, index) {
  const { scene } = state;
  const window = Math.max(2, Math.round(REACH.velocityHistorySeconds * scene.fps));
  const [vx, vy] = velocityAt(player.x, player.y, index, scene.fps, window);
  const step = Math.max(1, Math.round(scene.fps * 0.2));
  const before = Math.max(0, index - step);
  const after = Math.min(scene.n_frames - 1, index + step);
  const [ax0, ay0] = velocityAt(player.x, player.y, before, scene.fps, window);
  const [ax1, ay1] = velocityAt(player.x, player.y, after, scene.fps, window);
  const dt = (after - before) / scene.fps;
  const ax = dt > 0 ? (ax1 - ax0) / dt : 0;
  const ay = dt > 0 ? (ay1 - ay0) / dt : 0;
  return { vx, vy, speed: Math.hypot(vx, vy), acceleration: Math.hypot(ax, ay) };
}

/** The reachable set for the inspected player, memoised per player and frame. */
function reachFor(index, playerId = state.inspect) {
  const { scene } = state;
  if (!playerId) return null;
  const player = scene.byId.get(playerId);
  if (!player) return null;
  const x = player.x[index];
  const y = player.y[index];
  if (!Number.isFinite(x) || !Number.isFinite(y)) return null;
  if (state.reach && state.reach.playerId === playerId && state.reach.frame === index) {
    return state.reach.mask;
  }
  const { vx, vy } = motionOf(player, index);
  const mask = reachableMask([x, y], [vx, vy]);
  state.reach = { playerId, frame: index, mask };
  return mask;
}

function renderReach(index) {
  if (!state.layers.has("reach")) return;
  // in Counterfactual mode the question is about the story role, so the
  // reachable set follows it rather than whoever was last clicked
  const target = state.mode === "counterfactual"
    ? resolveStoryRole(index)?.id : state.inspect;
  const mask = reachFor(index, target);
  if (!mask) return;
  state.pitch.drawReach(mask.grid, mask.values, state.scene);
}

/** Real solver output for this scene, fetched once, or the unavailable state. */
function solverFor() {
  const { scene } = state;
  if (!scene) return null;
  if (state.solver && state.solver.sceneId === scene.scene_id) return state.solver.data;
  if (state.solver && state.solver.pending) return null;
  state.solver = { sceneId: scene.scene_id, pending: true, data: null };
  loadSolver(scene.scene_id).then((data) => {
    if (!state.scene || state.scene.scene_id !== scene.scene_id) return;
    state.solver = { sceneId: scene.scene_id, pending: false, data };
    render();
  });
  return null;
}

function renderSolverLayer() {
  if (!state.layers.has("solver")) return;
  const data = solverFor();
  // nothing is drawn when there is no artifact: the panel says so instead
  if (!data || !data.available) return;
  // "Actual movement" shows the tracking alone, as the old meeting page did
  if (state.mode === "game_solution" && state.solverView === "actual") return;
  const trajectory = primaryTrajectory(data);
  if (trajectory) state.pitch.drawSolverTrajectory(state.scene, trajectory);
}

// ---------------------------------------------------------------------------
// analysis panel
// ---------------------------------------------------------------------------
const NUMBER = (value, digits = 2) =>
  (Number.isFinite(value) ? value.toFixed(digits) : "—");

function rows(node, entries) {
  node.replaceChildren();
  for (const [label, value, dim] of entries) {
    const dt = document.createElement("dt");
    dt.textContent = label;
    const dd = document.createElement("dd");
    dd.textContent = value;
    if (dim) dd.className = "dim";
    node.append(dt, dd);
  }
}

/**
 * Show a panel section, if the current mode asks that question.
 *
 * The mode decides what the reviewer is being asked; the panel follows. A
 * section outside the current mode's list is not rendered at all -- an empty
 * card answering a question nobody asked is noise, and section 31 gives the
 * reviewer thirty seconds.
 */
function section(id, visible) {
  const sections = isPublic() ? PUBLIC_SECTIONS : MODE_SECTIONS;
  const on = Boolean(visible) && (sections[state.mode] || []).includes(id);
  const node = $(id);
  if (node) node.hidden = !on;
  return on;
}

/**
 * The one renderer every paper-facing field goes through.
 *
 * A metric that arrives from an updated research pipeline needs no code here:
 * it carries its own label, unit, source and definition version, and lands in
 * this list. A field with no value renders its reason -- never a dash, never a
 * zero, never an example number.
 */
function contractRows(node, records, textOf) {
  node.replaceChildren();
  for (const record of records) {
    if (!record) continue;
    const { available, text } = textOf(record);
    const dt = document.createElement("dt");
    dt.textContent = record.label;
    const dd = document.createElement("dd");
    dd.textContent = text;
    if (available) dd.title = sourceTitle(record);
    else dd.className = "pending";
    node.append(dt, dd);
  }
}

function renderAnalysis(index, slot, threat, fan) {
  const { scene, cache, selection } = state;
  const card = $("analysis-card");
  if (!card) return;

  // ---- scene ----
  const curated = currentShowcase();
  // outside Observed the scene is context for a question asked below it, so
  // it shrinks to a header and gives the room to the mode's own content
  const brief = state.mode !== "observed";
  if (section("an-scene", Boolean(curated))) {
    const entries = brief ? [
      ["Case study",
       `${curated.story_title || curated.showcase_id} · `
       + `${curated.review?.agreement_label || "—"} human-reviewed`],
    ] : [
      ["Case study", curated.showcase_id],
      ...(curated.story_title ? [["Story", curated.story_title]] : []),
      ["Review", `${curated.review?.agreement_label || "—"} · human reviewer ratings`],
      ["Provenance", PROVENANCE_LABEL[curated.provenance] || curated.provenance],
      ["Match", matchLabel(curated)],
    ];
    if (curated.event) entries.push(["Event", curated.event]);
    entries.push(["Timeline zero",
      curated.timing === "run_onset" ? "Run starts" : "Shot"]);
    if (curated.solver?.scenario_type) {
      entries.push(["Solver scenario", curated.solver.scenario_type]);
    }
    if (brief) entries.length = 1;
    renderCaseStudy(curated, brief);
    rows($("an-scene-rows"), entries);
    const note = $("an-scene-note");
    if (state.showcaseRolesApplied === false) {
      note.hidden = false;
      note.className = "an-note warn";
      note.textContent = "The reviewer's roles could not be resolved in this scene; "
        + "the stored annotation is shown instead.";
    } else {
      note.hidden = true;
    }
  }

  const player = state.inspect ? scene.byId.get(state.inspect) : null;
  const wantsPass = state.layers.has("passes") && Boolean(fan);
  const wantsSolver = state.mode === "game_solution" || state.layers.has("solver");

  // ---- player ----
  if (section("an-player", Boolean(player))) {
    const motion = motionOf(player, index);
    const entries = [
      ["Player", `#${player.shirt} ${player.name}`],
      ["Team", player.side === "attack" ? scene.attacking_team : scene.defending_team],
      ["Role", ROLE_LABEL[selection.roleOf(player.id)] || "—"],
      ["Speed", `${NUMBER(motion.speed, 1)} m/s`],
      ["Acceleration", `${NUMBER(motion.acceleration, 1)} m/s²`],
    ];
    const ball = brief ? null : ballRaw(index);
    if (ball) {
      entries.push(["Distance to ball",
        `${NUMBER(Math.hypot(player.x[index] - ball[0], player.y[index] - ball[1]), 1)} m`]);
    }
    const counterpart = brief ? null : nearestCounterpart(player, index);
    if (counterpart) {
      entries.push([`Distance to #${counterpart.player.shirt}`,
        `${NUMBER(counterpart.distance, 1)} m`]);
    }
    const onset = brief ? null : scene.onsets?.[player.id];
    if (onset && player.side === "attack") {
      const delta = (index - onset.index) / scene.fps;
      entries.push(["Run start",
        delta >= 0 ? `${NUMBER(delta, 1)} s ago` : `in ${NUMBER(-delta, 1)} s`]);
    }
    if (!brief && player.side === "attack" && cache) {
      // the beneficiary's own figure lives in Off-ball effect; this is the
      // residual space of whoever is being inspected, which is not the same row
      entries.push(["Space at this player",
        `${NUMBER(cache.residual(player.id, slot).value, 1)}`]);
    }
    if (!brief && state.layers.has("reach")) {
      const mask = reachFor(index);
      entries.push(["Kinematic reach, 2.0 s",
        mask ? `${Math.round(mask.areaM2)} m²` : "—", !mask]);
    }
    rows($("an-player-rows"), entries);
  }

  // ---- off-ball effect: what the run did to the beneficiary's space ----
  if (section("an-offball", selection.beneficiaries.length > 0 && Boolean(cache))) {
    const factual = cache.combined(selection.beneficiaries, slot).value;
    const entries = [["Available space", NUMBER(factual, 1)]];
    const swap = swapSpec();
    if (swap.length) {
      const counter = cache.combined(selection.beneficiaries, slot, swap).value;
      const gain = factual - counter;
      entries.push(["Space created", `${gain >= 0 ? "+" : ""}${NUMBER(gain, 1)}`]);
    } else {
      entries.push(["Space created", "Pick a defender to hold", true]);
    }
    rows($("an-offball-rows"), entries);
  }

  // ---- the solver's own release chain, for the selected exploratory target ----
  if (section("an-pass", wantsPass)) {
    renderReleaseRows(index, fan);
  }

  // ---- solver ----
  if (section("an-solver", wantsSolver)) {
    renderSolverRows();
  }

  // ---- the rest of the paper story ----
  renderDecision(index);
  renderCounterfactual();
  renderEvaluation();
  renderClipSummary();
  renderSourceRows();

  card.hidden = ![...card.querySelectorAll(".an-section")].some((s) => !s.hidden);
  if (isPublic()) {
    $("analysis-label").textContent = PUBLIC_MODE_TITLE[state.mode] || "Story";
    $("analysis-hint").textContent = "";
  } else {
    $("analysis-label").textContent = MODE_TITLE[state.mode] || "Analysis";
    $("analysis-hint").textContent = player ? `#${player.shirt}` : "";
  }
}

// ---------------------------------------------------------------------------
// the paper story: four modes over one scene
// ---------------------------------------------------------------------------
/**
 * Which panel sections each mode shows.
 *
 * Each list is a subset of the panel's own order, which is the paper's
 * argument: scene, player, decision, counterfactual, evaluation, clip, game
 * solution, off-ball context, action value details, source.
 */
const MODE_SECTIONS = {
  observed: ["an-scene", "an-player", "an-offball", "an-source"],
  counterfactual: ["an-scene", "an-player", "an-decision", "an-counterfactual",
                   "an-pass", "an-source"],
  evaluation: ["an-scene", "an-player", "an-decision", "an-evaluation", "an-clip",
               "an-source"],
  game_solution: ["an-scene", "an-solver", "an-source"],
};

/**
 * The same four modes for a public reader. Scene, review, provenance and
 * source repeat in every mode, so they live in Details. The clicked-player
 * block goes too: the showcase's cast is curated and nothing repoints it.
 */
const PUBLIC_SECTIONS = {
  observed: [], counterfactual: [], evaluation: [], game_solution: [],
};

const MODE_TITLE = {
  observed: "What happened",
  counterfactual: "What else was possible",
  evaluation: "How good the observed decision was",
  game_solution: "What the strategic equilibrium recommends",
};

const STORY_ROLE_LABEL = { runner: "Runner", passer: "Passer", defender: "Defender" };

/**
 * The paper-story payload for this scene, fetched once.
 *
 * Null means this build has none -- a checkout that has not run
 * export_paper_story still shows the observed story, which is the honest
 * subset rather than a broken page.
 */
function storyFor() {
  const { scene } = state;
  if (!scene) return null;
  if (state.story && state.story.sceneId === scene.scene_id) return state.story.data;
  if (state.story && state.story.pending) return null;
  state.story = { sceneId: scene.scene_id, pending: true, data: null };
  Promise.all([loadStory(scene.scene_id), loadContract()]).then(([data]) => {
    if (!state.scene || state.scene.scene_id !== scene.scene_id) return;
    state.story = { sceneId: scene.scene_id, pending: false, data };
    render();
  });
  return null;
}

/**
 * The first frame at or after `from` where the tracking gives a ball carrier.
 *
 * The dilemma and the equilibrium are both asked *of a decision*, and a
 * decision needs someone on the ball: the research code's own defender action
 * names are "toward the ball", "toward the runner", "toward goal", and the
 * first of those has no meaning once the ball is loose. The shot frame, which
 * is where the timeline starts, usually has no carrier at all.
 */
function decisionFrame(from) {
  const { scene, selection } = state;
  if (!scene) return from;
  const onset = scene.onsets?.[selection.runner]?.index;
  const start = Number.isInteger(onset) ? onset : from;
  for (let i = start; i < scene.n_frames; i += 1) {
    if (candidatePasses(scene, i)?.carrier) return i;
  }
  for (let i = from; i >= 0; i -= 1) {
    if (candidatePasses(scene, i)?.carrier) return i;
  }
  return from;                              // nothing on the ball anywhere
}

/**
 * Put the playhead on a frame the current mode can actually ask about.
 *
 * Called from `setMode` and again at the end of `applyUrlState`, because the
 * URL's own time handling runs after the mode is applied and would otherwise
 * put the playhead back on the shot.
 */
function ensureDecisionFrame() {
  const asksADecision = state.mode === "counterfactual"
    || state.mode === "game_solution";
  if (!asksADecision) return;
  if (candidatePasses(state.scene, state.frame)?.carrier) return;
  state.frame = decisionFrame(state.frame);
  const slider = $("time");
  if (slider) slider.value = String(state.frame);
}

function setMode(mode, quiet = false) {
  if (!MODE_SECTIONS[mode]) return;
  const leaving = state.mode;
  state.mode = mode;
  ensureDecisionFrame();
  for (const button of document.querySelectorAll("#story-modes .mode")) {
    const on = button.dataset.mode === mode;
    button.classList.toggle("is-on", on);
    button.setAttribute("aria-selected", String(on));
  }
  // each mode brings on the layer that answers its question; the reviewer can
  // still switch it off, and switching back to Observed puts it away
  // the reachable set is a diagnostic: it belongs to the Full explorer's
  // Counterfactual. The showcase's is Figure 1 -- three bodies and two
  // options -- and the layer stays available under Advanced.
  setLayer("reach", mode === "counterfactual" && state.collection !== "showcase");
  setLayer("solver", mode === "game_solution");
  $("solver-view").hidden = mode !== "game_solution";
  // a comparison belongs to Dilemma: leaving it returns to the curated play,
  // because nothing outside Dilemma has numbers for anyone else
  if (mode !== "counterfactual") state.compare = null;
  if (isPublic() && leaving !== mode) delete state.viewByMode[leaving];
  // Each tab has its own view, and its own memory of it. Nash opens in Focus
  // because the equilibrium options are too dense to read across a whole
  // pitch; everything else opens on the full pitch. A view the reader picks
  // lasts while they stay in that tab and is forgotten when they leave, so
  // coming back to Play always shows the play.
  if (isPublic()) setView(state.viewByMode[mode] || PUBLIC_VIEW[mode] || "full");
  // Nash and Player evaluation both read a solved state, so entering either
  // puts the playhead on one rather than on whatever frame Play left behind
  if (isPublic() && (mode === "game_solution" || mode === "evaluation")) {
    const data = solverFor();
    if (data?.kind === "bundle_panels") {
      const found = panelAt(data, state.frame);
      const frame = found ? frameOfPanel(found.panel) : null;
      if (frame != null) state.frame = frame;
    }
  }
  const space = $("space-control");
  if (space) space.hidden = !(isPublic() && mode === "counterfactual");
  renderMoments();
  if (!quiet) render();
}

/**
 * The solved moments, as a picker -- 0.0 s, 0.6 s, 1.2 s for most scenes.
 *
 * They are the moments the bundle actually solved, read off the panels. There
 * is nothing between them to show: the evaluation solves a fresh game at each
 * real moment and none in between.
 */
function renderMoments() {
  const node = $("moment-pick");
  if (!node) return;
  const data = solverFor();
  const panels = data?.kind === "bundle_panels" ? data.panels || [] : [];
  const on = isPublic() && panels.length > 0
    && (state.mode === "game_solution" || state.mode === "evaluation");
  node.hidden = !on;
  node.replaceChildren();
  if (!on) return;
  // which one is on follows the playhead, so the picker and the arrows can
  // never disagree about which moment is being shown
  const current = panelAt(data, state.frame)?.panel;
  for (const panel of panels) {
    const button = document.createElement("button");
    button.className = `seg${panel === current ? " is-on" : ""}`;
    button.type = "button";
    button.textContent = `${panel.dt.toFixed(1)} s`;
    button.addEventListener("click", () => {
      // the playhead goes to that solved frame: the arrows and the bodies are
      // one moment, never a mixture of two
      const frame = frameOfPanel(panel);
      if (frame != null) state.frame = frame;
      render();
    });
    node.appendChild(button);
  }
}

/**
 * A panel's frame.
 *
 * The bundle integration already converted it into this scene's own indexing
 * (`start_frame - clip_first_frame`), which is why `panelAt` compares it with
 * the playhead directly.
 */
function frameOfPanel(panel) {
  const { scene } = state;
  if (!scene || panel?.frame == null) return null;
  return Math.max(0, Math.min(scene.n_frames - 1, panel.frame));
}

function setLayer(name, on) {
  if (on) state.layers.add(name); else state.layers.delete(name);
  // one layer, two checkboxes: the explorer's row and the showcase's Layers
  for (const selector of [`[data-layer="${name}"]`, `[data-showlayer="${name}"]`]) {
    const box = document.querySelector(selector);
    if (box) box.checked = on;
  }
}

/**
 * The player the story question is being asked about.
 *
 * `passer` is the ball carrier at this frame, resolved by the same possession
 * rule the pass explorer uses -- read off the tracking, not assigned. It is
 * *not* the demo's `beneficiary`, which is the attacker whose space the run
 * opens and which stays off-ball context.
 */
function resolveStoryRole(index) {
  const { scene, selection } = state;
  if (!scene) return null;
  if (state.storyRole === "runner") return scene.byId.get(selection.runner) || null;
  if (state.storyRole === "defender") {
    return scene.byId.get(selection.defenders[0]) || null;
  }
  return candidatePasses(scene, index)?.carrier || null;
}

function setStoryRole(role) {
  if (!STORY_ROLE_LABEL[role]) return;
  state.storyRole = role;
  // the role picker is a player picker: the panel above and the reachable set
  // follow it, so the three readouts cannot disagree about who is meant
  const player = resolveStoryRole(state.frame);
  if (player) { state.inspect = player.id; state.reach = null; }
  render();
}

/**
 * The demo's annotation roles translated to the paper's solver roles.
 *
 * These are two vocabularies, not one. The annotation names a **runner**, a
 * **defender** and a **beneficiary** -- a person's reading of the play. The
 * solver names a **ball carrier**, a **runner**, a **beneficiary** and a
 * **defender** -- slots in a game. They overlap but do not coincide: the ball
 * carrier is read from the tracking and has no annotation role at all, and in
 * the solver's 2v1 game the carrier *is* the beneficiary, which is not true of
 * the annotation.
 *
 * So this maps only what is warranted, and returns nothing for a player whose
 * solver role is not determined. Nothing is guessed to fill a shape.
 */
function solverRolesAt(index) {
  const { scene, selection } = state;
  if (!scene) return {};
  const out = {};
  for (const id of selection.runners) out[id] = "runner";
  for (const id of selection.beneficiaries) out[id] = "beneficiary";
  for (const id of selection.defenders) out[id] = "defender";
  const carrier = candidatePasses(scene, index)?.carrier;
  // read from the tracking, and it wins: "who has the ball" is not an opinion
  if (carrier) out[carrier.id] = "ball carrier";
  return out;
}

/**
 * The football name for a defender's compass move, by the research code's own
 * rule (`stage3_read.name_move_targets`, ported in figure.js).
 *
 * Returns null unless the scene actually gives the three targets the rule
 * needs -- the ball carrier, the runner and the goal.
 */
function defenderMoveNameAt(index, commandIndex) {
  const { scene, selection } = state;
  const defender = scene?.byId.get(selection.defenders[0]);
  const runner = scene?.byId.get(selection.runner);
  const carrier = candidatePasses(scene, index)?.carrier;
  if (!defender || !runner || !carrier) return null;
  const at = (p) => [p.x[index], p.y[index]];
  if (![...at(defender), ...at(runner), ...at(carrier)].every(Number.isFinite)) {
    return null;
  }
  const lib = figureLib();
  if (!lib) return null;                    // a re-render follows when it lands
  const direction = scene.attacking_direction >= 0 ? 1 : -1;
  const goal = [direction * (scene.pitch[0] / 2), 0];
  return lib.defenderMoveName(lib.worldDirection(commandIndex, direction),
                               at(defender),
                               lib.targetsFor(at(carrier), at(runner), goal));
}

/**
 * Figure 1's two options: **Follow?** the runner, or **Stay?** with the ball
 * carrier.
 *
 * `render_figure1_dilemma` draws exactly these two arrows from the defender,
 * pointing at where the runner and the ball carrier **really were 0.6 s
 * later**, both a fixed `OPTION_M` = 3 m. Its docstring calls that a "picture
 * choice, not data", and Figure 1 "carries no number by design" -- so neither
 * does this. The question is the whole content: which one should he cover?
 *
 * Returns null unless the scene gives all three bodies and a frame 0.6 s on.
 */
function figureOneOptionsAt(index) {
  const { scene, selection } = state;
  const lib = figureLib();
  const defender = scene?.byId.get(selection.defenders[0]);
  const runner = scene?.byId.get(selection.runner);
  const carrier = candidatePasses(scene, index)?.carrier;
  if (!lib || !defender || !runner || !carrier) return null;
  const later = index + Math.round(0.6 * scene.fps);
  if (later >= scene.n_frames) return null;
  const origin = playerAt(scene, defender, index);
  if (!origin) return null;
  const toward = { runner, carrier };
  const options = [];
  for (const spec of lib.FIGURE1_OPTIONS) {
    const target = playerAt(scene, toward[spec.toward], later);
    if (!target) return null;
    const dx = target[0] - origin[0];
    const dy = target[1] - origin[1];
    const n = Math.hypot(dx, dy);
    if (n < 1e-6) return null;
    options.push({ label: spec.label, aim: [dx / n, dy / n], probability: null });
  }
  return { origin, options, length: lib.OPTION_M };
}

/**
 * The defender's five compass commands, named by the research rule.
 *
 * Kept as a diagnostic: Figure 1 shows two options, not five, and the five-way
 * fan was this repository's own reading before the figure code was available.
 * It stays available under Advanced because it is still the solver's real
 * command set.
 */
function defenderOptionsAt(index) {
  const { scene, selection } = state;
  const lib = figureLib();
  const defender = scene?.byId.get(selection.defenders[0]);
  if (!lib || !defender) return null;
  // the naming rule works in pitch coordinates; the drawing is on screen, and
  // `view` reflects one into the other -- mixing them mirrors every arrow
  const origin = playerAt(scene, defender, index);
  if (!origin) return null;
  const direction = scene.attacking_direction >= 0 ? 1 : -1;
  const options = [];
  for (let k = 0; k < lib.SOLVER_DIRS.length; k += 1) {
    const label = defenderMoveNameAt(index, k);
    if (!label) return null;
    const [ux, uy] = lib.worldDirection(k, direction);
    options.push({ index: k, label, aim: viewVector(scene, ux, uy),
                   probability: null });
  }
  return { origin, defender, options };
}

/**
 * The equilibrium policy on the pitch, for a scene with a real artifact.
 *
 * The arrows are the same ones the dilemma draws, weighted by the solved
 * policy instead of drawn alike, and the decision follows the playhead: the
 * step whose time bracket contains the current frame. Nothing here runs
 * without `solverFor()` returning an available state, and no Bundesliga scene
 * has one -- this is exercised by demo_viz/web/story_harness.html against a
 * genuine solver study state.
 */
/**
 * The solved moment nearest the playhead, and whether we are standing on it.
 *
 * The bundle solves a fresh game at three real moments (0.0 / 0.6 / 1.2 s) and
 * nothing in between. Rather than interpolate -- there is no solved game to
 * interpolate -- the view holds the nearest solved moment and says which one
 * it is, so persistence never reads as a new solve.
 */
function panelAt(data, index) {
  const panels = data?.panels;
  if (!panels?.length) return null;
  let best = panels[0];
  for (const panel of panels) {
    if (Math.abs(panel.frame - index) < Math.abs(best.frame - index)) best = panel;
  }
  return { panel: best, exact: best.frame === index };
}

/** The bundle's equilibrium policy, drawn at the solved moment. */
function renderBundlePolicy(index, data) {
  const lib = arrowsLib();
  const figure = figureLib();
  if (!lib || !figure) return;
  const found = panelAt(data, index);
  if (!found) return;
  // an equilibrium is a property of a solved state, so its arrows and its
  // percentages only exist at the moments that were solved. Between them the
  // pitch shows the play and nothing is drawn that would imply a solution.
  if (isPublic() && !found.exact) return;
  const { scene } = state;
  // one layout for the whole panel, so a defender's label steps aside from an
  // attacker's and from every player marker, not only from its own body's
  const layout = new lib.LabelLayout(state.pitch);
  blockPlayers(layout, index);
  const gap = figure.FIGURE2.labelGapM;
  for (const [role, body] of Object.entries(found.panel.bodies || {})) {
    const colour = figure.ROLE_COLOR[body.solver_role] || figure.PAPER.attack;
    const options = (body.options || [])
      .filter((o) => o.prob >= figure.MIN_P)
      .map((o) => ({
        // geometry is the solver's own 0.6 s path, converted to screen space;
        // probability never touches it
        path: (o.path || []).map(([x, y]) => viewVector0(scene, x, y)),
        probability: o.prob,
        stop: o.command === 0,
        rests: Boolean(o.rests),
        // the figure's wording in the explorer, football in the showcase:
        // the same option, the same probability, a different reader
        label: isPublic()
          ? publicOptionLabel(body.solver_role, o.prob, {
            stop: o.command === 0, rests: Boolean(o.rests), name: o.label,
          })
          : figure.optionLabel(body.solver_role, o.prob, {
            stop: o.command === 0, rests: Boolean(o.rests),
            // the solver's own name for this command, so a 0.57 m move's
            // percentage has something visible to belong to
            name: o.label, defenderNames: figure.FIGURE2.defenderNames,
          }),
      }));
    if (options.length) {
      lib.drawActionArrows(state.pitch, playerPoint(scene, body.pos), options, {
        colour, labelFloor: figure.MIN_P, labelGap: gap, layout,
        // a move leaves the marker with clear ground, as the figure does
        clearM: markerClearance(figure, body.solver_role),
        // and one that would show almost nothing is stretched until it reads
        minArrowM: figure.MIN_ARROW_M,
      });
    }
  }
  // the pass: dashed charcoal from the ball to its target, labelled once
  for (const pass of found.panel.passes || []) {
    if (!pass.target || pass.prob < figure.MIN_P) continue;
    const ball = ballAt(scene, index);
    if (!ball) continue;
    lib.drawPassChoice(state.pitch, ball, playerPoint(scene, pass.target),
                       isPublic() ? publicPassLabel(pass.prob)
                                  : figure.passLabel(pass.prob,
                                                     { received: Boolean(pass.to) }),
                       layout);
  }
  // every label in the panel is now measured and every mark is an obstacle,
  // so the whole panel is laid out at once -- the figure places a panel, not
  // a body, and a body at a time is what crowded the last label out
  layout.run();
}

/**
 * Every drawn player is ground a label may not be written over.
 *
 * The marker and the shirt number inside it, for all 22, at the radii
 * `drawPlayers` actually uses -- so a percentage never lands on a number, in
 * any scene, at any moment, at any width.
 */
function blockPlayers(layout, index) {
  const { scene, selection, pitch } = state;
  if (!scene) return;
  const k = Math.max(pitch.k, 0.62);
  for (const player of scene.players) {
    const position = playerAt(scene, player, index);
    if (!position) continue;
    const role = selection.roleOf(player.id);
    layout.blockDisc(position[0], position[1], (role ? 1.55 : 1.2) * k * 1.1, true);
  }
  const ball = ballAt(scene, index);
  if (ball) layout.blockDisc(ball[0], ball[1], 0.8 * k, true);
  // and every shirt number, background players included, by its measured
  // box: the players are drawn after the labels, so each number is set once
  // here with drawPlayers' own attributes, measured, and taken away again
  if (!state.layers.has("labels")) return;
  for (const player of scene.players) {
    const position = playerAt(scene, player, index);
    if (!position) continue;
    const role = selection.roleOf(player.id);
    const probe = pitch.add("players", "text", {
      x: position[0], y: position[1] + 0.45 * pitch.k, "text-anchor": "middle",
      class: "shirt", "font-size": (role ? 1.6 : 1.35) * k,
    }, player.shirt);
    layout.blockText(probe, 0.1 * k);
    probe.remove();
  }
}

/**
 * How far a move should start from a body's centre, in pitch metres.
 *
 * `figure_style` works in points at the printed size; the pitch works in
 * metres. The marker's drawn radius here is `1.55 * max(k, 0.62)` metres, and
 * the figure's gap is `MOVE_GAP / KEY_D` of a marker diameter, so the same
 * proportion is applied to the radius this pitch actually draws.
 */
function markerClearance(figure, solverRole) {
  const radius = 1.55 * Math.max(state.pitch.k, 0.62);
  return radius * (1 + figure.MOVE_GAP / figure.KEY_D);
}

/** A scene-space point as a screen point (the same reflection `view` applies). */
function playerPoint(scene, xy) {
  return scene.flip ? [-xy[0], xy[1]] : [xy[0], -xy[1]];
}

/** The same for a point that is part of a path. */
function viewVector0(scene, x, y) {
  return scene.flip ? [-x, y] : [x, -y];
}

function renderPolicyArrows(index) {
  if (state.mode !== "game_solution") return;
  if (state.solverView === "actual") return;
  const data = solverFor();
  if (!data?.available) return;
  if (data.kind === "bundle_panels") { renderBundlePolicy(index, data); return; }

  const lib = arrowsLib();
  const figure = figureLib();
  const policies = policyLib();
  if (!lib || !figure || !policies) return;

  const { scene, selection } = state;
  const seconds = scene.times[index] - scene.times[0];
  const step = Math.max(0, Math.min(data.steps - 1,
                                    Math.floor(seconds / (data.step_seconds || 1))));
  const body = policies.bodyPolicies(data, step);
  if (!body) return;

  const direction = scene.attacking_direction >= 0 ? 1 : -1;
  const draw = (playerId, probabilities, colour, isDefender) => {
    const player = scene.byId.get(playerId);
    if (!player || !probabilities) return;
    const origin = playerAt(scene, player, index);
    if (!origin) return;
    const options = probabilities.map((probability, k) => {
      const [ux, uy] = figure.worldDirection(k, direction);
      const label = isDefender ? defenderMoveNameAt(index, k)
                               : figure.compassName([ux, uy], direction);
      return { label: label || "", probability, aim: viewVector(scene, ux, uy) };
    }).filter((o) => o.probability > 1e-9 && o.label);
    lib.drawActionArrows(state.pitch, origin, options, { colour, length: 5.0 });
  };

  // the paper's palette here: attack blue, defence vermillion (figure_style).
  // Observed keeps the demo's annotation colours -- two vocabularies, and
  // PAPER_FIGURE_ALIGNMENT.md says which is which.
  draw(selection.defenders[0], body.defender, figure.ROLE_COLOR.defender, true);
  draw(selection.runner, body.receiver, figure.ROLE_COLOR.runner, false);
  const carrier = candidatePasses(scene, index)?.carrier;
  if (carrier) draw(carrier.id, body.carrier, figure.ROLE_COLOR["ball carrier"], false);
}

/**
 * The dilemma, drawn where the mode asks the dilemma question.
 *
 * Figure 1's two options by default. The five-command fan is a diagnostic and
 * appears only when the Advanced layer asks for it.
 */
function renderDilemma(index) {
  if (state.mode !== "counterfactual") return;
  const lib = arrowsLib();
  const figure = figureLib();
  if (!lib || !figure) return;
  const colour = figure.ROLE_COLOR.defender;
  if (state.layers.has("commands")) {
    const fan = defenderOptionsAt(index);
    if (fan) {
      lib.drawActionArrows(state.pitch, fan.origin, fan.options,
                           { colour, length: 4.6 });
    }
    return;
  }
  const dilemma = figureOneOptionsAt(index);
  if (!dilemma) return;
  lib.drawActionArrows(state.pitch, dilemma.origin, dilemma.options,
                       { colour, length: dilemma.length });
}

function renderStoryRoles() {
  const node = $("story-roles");
  node.replaceChildren();
  for (const role of Object.keys(STORY_ROLE_LABEL)) {
    const button = document.createElement("button");
    button.className = `roleb${state.storyRole === role ? " is-on" : ""}`;
    button.type = "button";
    button.dataset.storyRole = role;
    button.textContent = isPublic()
      ? (PUBLIC_STORY_ROLE[role] || STORY_ROLE_LABEL[role])
      : STORY_ROLE_LABEL[role];
    button.setAttribute("aria-pressed", String(state.storyRole === role));
    node.append(button);
  }
}

/**
 * The one-line case study, in Observed mode.
 *
 * A curated `story_summary` when one has been written; otherwise the
 * reviewer's own note, verbatim and attributed. Nothing here writes tactical
 * prose: a sentence like "the runner pulls the defender away" is a claim, and
 * the only warrant for it in this repository is a person having said so.
 */
function renderCaseStudy(curated, brief) {
  const node = $("an-casestudy");
  if (!node) return;
  const summary = curated?.story_summary?.trim();
  const note = curated?.annotation?.trim();
  const text = summary || note;
  node.hidden = brief || !text;
  if (node.hidden) return;
  node.textContent = summary ? summary : `\u201c${note}\u201d`;
  node.className = summary ? "casestudy" : "casestudy quoted";
  node.title = summary ? "Curated case-study summary"
    : "The reviewer's own note on this scene, shown verbatim";
}

function renderDecision(index) {
  if (!section("an-decision", true)) return;
  const { scene } = state;
  renderStoryRoles();
  const player = resolveStoryRole(index);
  const time = scene.times[index];
  if (isPublic()) {
    // the role buttons above say which role and the legend says who, so the
    // public card adds no row of its own here
    rows($("an-decision-rows"), player ? [] : [
      [PUBLIC_STORY_ROLE[state.storyRole] || STORY_ROLE_LABEL[state.storyRole],
       "Not in this play", true],
    ]);
    return;
  }
  const entries = [
    ["Frame", `${index} · ${time >= 0 ? "+" : ""}${time.toFixed(2)} s`],
  ];
  if (player) {
    entries.push([STORY_ROLE_LABEL[state.storyRole], `#${player.shirt} ${player.name}`]);
    if (state.storyRole === "passer") {
      entries.push(["Resolved by", "in possession at this frame · observed tracking"]);
    }
  } else {
    entries.push([STORY_ROLE_LABEL[state.storyRole],
                  state.storyRole === "passer"
                    ? "No clear carrier at this frame"
                    : "No player in this role for this scene", true]);
  }
  if (state.layers.has("reach")) {
    const mask = player ? reachFor(index, player.id) : null;
    entries.push(["Kinematic reach, 2.0 s",
                  mask ? `${Math.round(mask.areaM2)} m² · model reachability` : "—",
                  !mask]);
  }
  rows($("an-decision-rows"), entries);
}

/**
 * Observed action, feasible set, and the static/responsive pair.
 *
 * Every one of these is a reserved slot today. The research code has no
 * projection of observed tracking onto an action set, no feasible-alternative
 * ranking, and no paired static/responsive evaluation on one scale -- see
 * PAPER_STORY_TRACE.md section 2. The structure is here so that when those
 * land, the adapter fills them and this function does not change.
 */
function renderCounterfactual() {
  const story = storyFor();
  if (!section("an-counterfactual", true)) return;
  const block = story?.counterfactual?.[state.storyRole];
  const pair = $("cf-pair");
  if (!block) {
    rows($("an-cf-rows"), [["Counterfactual",
      story ? "Not defined for this role" : "Paper-story payload not in this build",
      true]]);
    pair.replaceChildren();
    $("an-cf-change").replaceChildren();
    return;
  }
  if (isPublic()) {
    // the public card carries the one thing this mode knows -- what he did --
    // and leaves the reserved slots, their reasons and the paired baselines
    // to the explorer, which is where an unfilled slot is information
    rows($("an-cf-rows"), publicObservedRow(block));
    pair.replaceChildren();
    $("an-cf-change").replaceChildren();
    return;
  }
  contractRows($("an-cf-rows"),
               [block.observed_action, block.feasible_actions, block.release_library],
               (r) => (r.name === "observed_action" ? actionText(r) : metricText(r)));
  pair.replaceChildren();
  for (const side of [block.static, block.responsive]) {
    if (!side) continue;
    const card = document.createElement("div");
    card.className = "cfside";
    const head = document.createElement("div");
    head.className = "cfhead";
    head.textContent = side.label;
    const semantics = document.createElement("div");
    semantics.className = side.semantics ? "cfsem" : "cfsem pending";
    semantics.textContent = side.semantics
      || "Baseline semantics pending updated research implementation";
    const list = document.createElement("dl");
    contractRows(list, [side.best_action, side.value],
                 (r) => (r.name.endsWith("_value") ? metricText(r) : actionText(r)));
    card.append(head, semantics, list);
    pair.append(card);
  }
  contractRows($("an-cf-change"), [block.value_change], metricText);
}

/**
 * "Observed: Run forward". `observed_action.description` is the pipeline's own
 * name for the command nearest where the player really was 0.6 s later, so the
 * football word translates a measurement. No word, no row.
 */
function publicObservedRow(block) {
  const action = block?.observed_action;
  if (!action || action.availability !== "available") return [];
  const solverRole = STORY_TO_SOLVER[state.storyRole] || state.storyRole;
  const word = publicActionName(solverRole, action.description)
    || (action.description ? sentenceCase(action.description) : null);
  return word ? [["Observed", word]] : [];
}

const STORY_TO_SOLVER = { passer: "ball carrier", runner: "runner", defender: "defender" };

function sentenceCase(text) {
  const s = String(text || "");
  return s ? s[0].toUpperCase() + s.slice(1) : s;
}

function renderEvaluation() {
  const story = storyFor();
  if (!section("an-evaluation", true)) return;
  const block = story?.evaluation?.[state.storyRole];
  const note = $("an-eval-note");
  if (!block) {
    rows($("an-eval-rows"), [["Evaluation",
      story ? "Not defined for this role" : "Paper-story payload not in this build",
      true]]);
    note.hidden = true;
    return;
  }
  // the numbers belong to a solved moment, so they follow the playhead: the
  // series carry one value per solve and the rows show the nearest one,
  // saying which. Nothing is interpolated -- between solves there is no
  // solved game to interpolate.
  const atMoment = metricsAtFrame(block, state.frame);
  const shown = atMoment ? atMoment.metrics : block.metrics;
  if (isPublic()) {
    renderPublicEvaluation(shown, atMoment);
    note.hidden = true;
    return;
  }
  contractRows($("an-eval-rows"), [...shown, block.optimal_action],
               (r) => (r.name === "optimal_action" ? actionText(r) : metricText(r)));
  const none = shown.every((m) => m.availability !== "available");
  note.hidden = !(none || (atMoment && !atMoment.exact));
  if (none) {
    note.textContent = "Evaluation outputs are not available for this scene yet. "
      + "This view populates from the player-evaluation pipeline; nothing is "
      + "substituted in the meantime.";
  } else if (atMoment && !atMoment.exact) {
    note.textContent = `Solved at frame ${atMoment.frame}; the playhead is `
      + "between solved moments, so these are that moment's numbers.";
  }
}

/**
 * How good the decision was, in four lines: the same records the explorer
 * reads, as an ordinal, a percentage and a number. Relative rank is the rank
 * again on a 0-1 scale, so only one of the two is here.
 *
 * The observed action joins them only at the moment it belongs to -- the
 * payload carries the first solved moment's, while these numbers follow the
 * playhead, so pairing them elsewhere would caption one moment with another's.
 */
function renderPublicEvaluation(metrics, atMoment) {
  const byName = new Map(metrics.map((m) => [m.name, m]));
  const value = (name) => {
    const record = byName.get(name);
    return record && record.availability === "available" ? record.value : null;
  };
  const entries = [];
  const story = storyFor();
  const first = presentSeries(story?.evaluation?.[state.storyRole] || {})[0];
  const firstFrame = first?.frames?.[0];
  if (atMoment && firstFrame != null && atMoment.frame === firstFrame) {
    entries.push(...publicObservedRow(story?.counterfactual?.[state.storyRole]));
  }
  const rank = value("observed_action_rank");
  if (rank != null) entries.push(["Rank", ordinal(rank)]);
  const probability = value("similarity_to_optimal");
  if (probability != null) {
    entries.push(["Equilibrium probability", percent(probability)]);
  }
  const regret = value("regret");
  if (regret != null) entries.push(["Regret", NUMBER(regret, 2)]);
  if (!entries.length) {
    entries.push(["Player evaluation", "Not available for this play", true]);
  }
  rows($("an-eval-rows"), entries);
}

/**
 * A role's metrics at the solved moment nearest `frame`.
 *
 * The evaluation solves a fresh game at each real moment and nothing between,
 * so the rows show one solved moment's numbers and say which. Returns null
 * when no series carries a value, leaving the static metrics in place.
 */
function metricsAtFrame(block, frame) {
  const series = presentSeries(block);
  if (!series.length) return null;
  let best = null;
  for (const line of series) {
    for (const f of line.frames) {
      if (best === null || Math.abs(f - frame) < Math.abs(best - frame)) best = f;
    }
  }
  if (best === null) return null;
  const byName = new Map(series.map((s) => [s.name, s]));
  const metrics = block.metrics.map((metric) => {
    const line = byName.get(metric.name);
    if (!line) return metric;
    const i = line.frames.indexOf(best);
    if (i < 0) return metric;
    return { ...metric, availability: "available", value: line.values[i],
             source: line.source, definition_version: line.definition_version,
             provenance: line.provenance };
  });
  return { metrics, frame: best, exact: best === frame };
}

function renderClipSummary() {
  const story = storyFor();
  if (!section("an-clip", true)) return;
  const node = $("clip-cards");
  node.replaceChildren();
  const summary = story?.clip_summary || {};
  for (const role of story?.roles || Object.keys(STORY_ROLE_LABEL)) {
    const card = document.createElement("div");
    card.className = "clipcard";
    const head = document.createElement("div");
    head.className = "cliphead";
    head.textContent = STORY_ROLE_LABEL[role] || role;
    const list = document.createElement("dl");
    // the aggregation is the producer's, carried per metric: nothing here
    // means, medians or percentiles anything
    contractRows(list, summary[role] || [], metricText);
    card.append(head, list);
    node.append(card);
  }
}

function renderSourceRows() {
  const story = storyFor();
  if (!section("an-source", true)) return;
  const entries = [];
  if (story) {
    entries.push(["Story contract", story.schema]);
    const p = story.provenance || {};
    const none = p.evaluation_source === "none";
    entries.push(["Evaluation source",
                  none ? "none · not implemented in the research code yet"
                       : `${p.evaluation_source} · definition ${p.definition_version}`,
                  none]);
  } else {
    entries.push(["Story contract", "not exported in this build", true]);
  }
  entries.push(["Tracking", "IDSSE Bundesliga, 25 Hz · observed"]);
  rows($("an-source-rows"), entries);
}

/**
 * The frame-by-frame strip.
 *
 * Only series the payload carries are drawn. With none, the strip says what
 * would appear here and names the reserved series -- an empty state that looks
 * intentional, because it is. No example curve is ever drawn.
 */
/** The three evaluation lines, named for the public strip and card. */
const PUBLIC_METRIC = {
  observed_action_rank: "Rank",
  similarity_to_optimal: "Equilibrium probability",
  regret: "Regret",
};

function renderEvalStrip(index) {
  const strip = $("eval-strip");
  // the public tab reads one solved moment at a time and says its numbers in
  // words; a frame-by-frame rank/regret plot under it answered a question the
  // public reader was not asking. The explorer keeps it.
  strip.hidden = state.mode !== "evaluation" || isPublic();
  if (strip.hidden) return;
  const block = storyFor()?.evaluation?.[state.storyRole];
  const series = presentSeries(block);
  const plot = $("eval-plot");
  const empty = $("eval-empty");
  const readout = $("eval-readout");
  if (!series.length) {
    plot.replaceChildren();
    empty.hidden = false;
    const reserved = reservedSeries(block).map((s) => s.label);
    empty.textContent = "Frame-level evaluation will appear here when evaluation "
      + "outputs are loaded"
      + (reserved.length ? `: ${reserved.join(", ")}.` : ".");
    readout.textContent = "";
    return;
  }
  empty.hidden = true;
  const lib = stripLib();
  if (!lib) return;                       // a re-render follows when it lands
  // relative rank is the rank again on a 0-1 scale, so the public strip draws
  // one of the two and the explorer keeps both
  const lines = isPublic()
    ? series.filter((s) => s.name !== "relative_rank")
      .map((s) => ({ ...s, label: PUBLIC_METRIC[s.name] || s.label }))
    : series;
  const nearest = lib.drawEvalStrip(plot, {
    series: lines, nFrames: state.scene.n_frames, frame: index,
    width: plot.clientWidth,
    onSeek: (frame) => { state.frame = frame; render(); },
  });
  // the nearest computed sample, named as such: between two solved frames
  // there is no value, and saying which frame it came from is the honest read
  readout.textContent = (nearest || [])
    .map((near, i) => (near
      // an ordinal keeps its integer form; a ratio gets three places
      ? `${lines[i].label} ${Number.isInteger(near.value)
            ? near.value : near.value.toFixed(3)}`
        + (near.exact ? "" : ` (frame ${near.frame})`)
      : null))
    .filter(Boolean).join(" · ");
}

/**
 * The OBSO convention's own ball speed, 15 m/s, from ReferenceOBSOConfig.
 * It is the only ball-travel number this repository defines without a fitted
 * model, and it is what the pitch-control term already assumes.
 */
const OBSO_BALL_SPEED = 15.0;

function ballRaw(index) {
  const bx = state.scene.ball.x[index];
  const by = state.scene.ball.y[index];
  return Number.isFinite(bx) && Number.isFinite(by) ? [bx, by] : null;
}

function nearestCounterpart(player, index) {
  const { selection, scene } = state;
  const ids = player.side === "attack" ? selection.defenders
    : [...selection.runners, ...selection.beneficiaries];
  let best = null;
  for (const id of ids) {
    const other = scene.byId.get(id);
    if (!other || other.id === player.id) continue;
    const distance = Math.hypot(player.x[index] - other.x[index],
                                player.y[index] - other.y[index]);
    if (!best || distance < best.distance) best = { player: other, distance };
  }
  return best;
}

function threatComponents(index, point) {
  const obso = obsoLib();
  if (!obso || !point || !state.obso || state.obso.pending) return null;
  const { entry, score } = state.obso;
  if (!entry || !score) return null;
  return obso.componentsAt(entry, score, state.scene, index, point[0], point[1],
                           obso.surfaceAt.lastControl);
}

/**
 * Precomputed release quantities for this scene, fetched on first use.
 *
 * Absent means the export was skipped because the research stack or the fitted
 * model was not present -- the normal state of a checkout without them, not an
 * error, so the panel says so rather than showing nothing.
 */
function releaseFor_scene() {
  const { scene } = state;
  if (!scene) return null;
  if (state.release && state.release.sceneId === scene.scene_id) return state.release.data;
  if (state.release && state.release.pending) return null;
  const lib = releaseLib();
  if (!lib) return null;
  state.release = { sceneId: scene.scene_id, pending: true, data: null };
  Promise.all([lib.loadRelease(scene.scene_id), lib.loadModelCard()]).then(([data]) => {
    if (!state.scene || state.scene.scene_id !== scene.scene_id) return;
    state.release = { sceneId: scene.scene_id, pending: false, data };
    render();
  });
  return null;
}

/** The receiver the pass model is asked about: the selected beneficiary. */
function selectedReceiver() {
  return state.selection.beneficiaries[0] || null;
}

/**
 * Legal -> completion proxy -> positional threat -> release payoff.
 *
 * Read, not computed: export_release.py ran the research implementation. The
 * multiplication is shown as a chain so the gate is visible -- an illegal
 * target keeps its completion proxy and takes a payoff of zero.
 */
function renderReleaseRows(index, fan) {
  const node = $("an-pass-rows");
  const body = $("an-pass-body");
  const valueToggle = $("value-toggle");
  if (body) body.hidden = !state.valueOpen;
  if (valueToggle) {
    $("value-chev").textContent = state.valueOpen ? "\u25be" : "\u25b8";
    valueToggle.setAttribute("aria-expanded", String(state.valueOpen));
  }
  const note = $("an-pass-note");
  const toggle = $("details-toggle");
  const details = $("an-detail-rows");
  const hideDetails = () => {
    if (toggle) toggle.hidden = true;
    if (details) details.hidden = true;
  };

  const payload = releaseFor_scene();
  if (state.endpoint == null) {
    rows(node, [["Target", "Click one to inspect", true]]);
    note.hidden = true;
    if ($("value-hint")) $("value-hint").textContent = "";
    hideDetails();
    return;
  }

  const receiver = selectedReceiver();
  const result = releaseLib().releaseFor(payload, index, receiver, state.endpoint);
  if (!result.available) {
    rows(node, [["Solver pass model", "unavailable", true]]);
    note.hidden = false;
    note.className = "an-note";
    note.textContent = result.reason;
    hideDetails();
    return;
  }

  const ray = fan?.rays?.[state.endpoint];
  const receiverPlayer = state.scene.byId.get(receiver);
  const entries = [
    ["Receiver", receiverPlayer ? `#${receiverPlayer.shirt} ${receiverPlayer.name}` : "—"],
    ["Direction", ray ? `${ray.degrees > 0 ? "+" : ""}${ray.degrees}° from attack` : "—"],
    ["Legal", result.legal ? "Yes" : (result.offside ? "No · offside" : "No · off pitch")],
    ["Completion proxy", NUMBER(result.completionProxy, 3)],
    ["Positional threat", NUMBER(result.positionalThreat, 3)],
    ["Release payoff", NUMBER(result.releasePayoff, 3)],
  ];
  rows(node, entries);
  const hint = $("value-hint");
  if (hint) {
    hint.textContent = result.legal
      ? `payoff ${NUMBER(result.releasePayoff, 3)}`
      : (result.offside ? "offside" : "off pitch");
  }
  // make the gate legible without a paragraph about it
  node.lastElementChild?.classList.add("strong");
  if (!result.legal) {
    node.querySelectorAll("dd")[2]?.classList.add("gate");
  }

  note.hidden = result.exact;
  if (!result.exact) {
    note.className = "an-note";
    note.textContent = `Computed at frame ${result.sampledFrame}, the nearest exported sample.`;
  }

  if (toggle && details) {
    toggle.hidden = false;
    details.hidden = !state.detailsOpen;
    $("details-chev").textContent = state.detailsOpen ? "\u25be" : "\u25b8";
    toggle.setAttribute("aria-expanded", String(state.detailsOpen));
    if (state.detailsOpen) {
      const labels = result.labels || {};
      const metres = new Set(["pass_length", "forward_distance", "receiver_target_gap"]);
      rows(details, Object.entries(result.features).map(([name, value]) => {
        if (name === "same_defender") return [labels[name] || name, value ? "Yes" : "No"];
        return [labels[name] || name,
                metres.has(name) ? `${NUMBER(value, 1)} m` : NUMBER(value, 3)];
      }));
    }
  }
}

/**
 * The game solution: the equilibrium half of the abstract, which is real.
 *
 * Every quantity here is the solver's own, under the solver's own name. The
 * certificate is the root best-response gap recomputed through the whole
 * policy tree, not a residual from the solved value arrays, so it is reported
 * as "Certificate gap" and not renamed to anything friendlier.
 */
/**
 * The bundle's solved moment, summarised.
 *
 * The pitch carries the policy; this says which moment is being shown, what
 * the game is worth there, and whether upstream called it a dilemma. The
 * criterion is upstream's boolean -- the defender mixes **and** there is no
 * saddle point -- and is never recomputed here.
 */
function renderBundleRows(data, node, policy, provenance) {
  const found = panelAt(data, state.frame);
  if (!found) { rows(node, [["Game solution", "No solved moment", true]]); return; }
  const panel = found.panel;
  const d = panel.dilemma || {};
  const s = panel.static || {};
  const entries = [
    ["Decision", `${panel.dt.toFixed(1)} s`
      + (found.exact ? "" : " \u00b7 nearest solved moment")],
    ["Equilibrium value", NUMBER(panel.value, 4)],
    ["Dilemma", d.is_dilemma ? "yes" : "no"],
    ["Defender policy", d.defender_mixed ? "mixed" : "pure"],
    ["Attack policy", d.attack_mixed ? "mixed" : "pure"],
    ["Saddle-point gap", d.saddle_gap != null ? d.saddle_gap.toExponential(1) : "\u2014"],
  ];
  if (s.static_attack_appeared != null) {
    entries.push(["Held defender, attack's best", NUMBER(s.static_attack_appeared, 4)]);
    entries.push(["Once he may answer", NUMBER(s.static_attack_responsive, 4)]);
    entries.push(["Overstated by", `${NUMBER(100 * (s.static_attack_loss_rel ?? 0), 1)}%`]);
  }
  rows(node, entries);
  policy.hidden = true;                 // the arrows on the pitch are the policy
  provenance.hidden = false;
  const p = data.provenance || {};
  provenance.textContent = `${p.bundle} \u00b7 ${p.repository}@${String(p.upstream).slice(0, 7)}`
    + ` \u00b7 ${p.script}`;
}

function renderSolverRows() {
  const node = $("an-solver-rows");
  const policy = $("an-policy");
  const provenance = $("an-solver-prov");
  const data = solverFor();

  if (!data || !data.available) {
    const detail = storyFor()?.equilibrium?.detail;
    rows(node, [
      ["Game solution",
       data ? (data.reason || "Not computed for this scene") : "Loading\u2026", true],
      ...(data && detail ? [["Why", detail, true]] : []),
    ]);
    policy.hidden = true;
    provenance.hidden = true;
    return;
  }

  if (data.kind === "bundle_panels") { renderBundleRows(data, node, policy, provenance); return; }

  const lib = policyLib();
  if (!lib) {
    rows(node, [["Game solution", "Loading\u2026", true]]);
    policy.hidden = true;
    provenance.hidden = true;
    return;
  }

  const info = solverSummary(data);
  const attack = lib.attackRows(data);
  const defence = lib.defenderRows(data);
  rows(node, [
    ["Equilibrium value", NUMBER(info.value, 4)],
    ["Certificate gap", info.gap != null ? info.gap.toExponential(1) : "—"],
    ["Root policy", info.mixed ? "mixed" : "pure"],
    ["Attack support",
     `${lib.support(data.root_attack)} of ${data.root_attack.length}`],
    ["Defender support",
     `${lib.support(data.root_defender)} of ${data.root_defender.length}`],
    ["Release probability", NUMBER(info.releaseProbability, 3)],
    ["Horizon", `${info.steps} × ${NUMBER(info.stepSeconds, 1)} s`],
  ]);

  policy.hidden = false;
  lib.drawPolicy($("pol-attack"), attack);
  lib.drawPolicy($("pol-defence"), defence);
  $("pol-attack-kind").textContent = lib.isMixed(data.root_attack) ? "mixed" : "pure";
  $("pol-defence-kind").textContent =
    lib.isMixed(data.root_defender) ? "mixed" : "pure";
  $("pol-note").textContent = `${lib.MODAL_LABEL}. ${lib.MODAL_NOTE}`;

  // one element, rewritten -- not a fresh sibling per render
  const p = info.provenance || {};
  provenance.hidden = false;
  provenance.textContent =
    `${p.repository}@${p.commit} · ${p.artifact}#${p.state_index}`;
}


/**
 * Marks on the scrubber, from data rather than from pacing.
 *
 * Two things are actually defined for every scene: the run onset the detector
 * found for each selected runner (`scene.onsets`, with its own method label),
 * and the annotated shot, which is where the clip's own clock reads zero. No
 * mark is drawn for anything the data does not define -- there is no
 * "defender reacts" frame in the exported scene, so none is invented.
 */
// ---------------------------------------------------------------------------
// submission showcase
// ---------------------------------------------------------------------------
/** Curated entries that map to a scene the demo actually ships. */
function playableShowcase() {
  if (!state.showcase) return [];
  return filtered(state.showcase, state.showcaseFilter).filter((s) => s.playable);
}

function currentShowcase() {
  if (!state.showcase || !state.showcaseId) return null;
  return state.showcase.find((s) => s.showcase_id === state.showcaseId) || null;
}

/** Show the controls the active collection needs, and hide the other's. */
/**
 * The Submission showcase is a paper-themed view; the Full explorer keeps the
 * dark research theme.
 *
 * One attribute on <body> drives the CSS, and the pitch takes the matching
 * palette. The split is deliberate: the showcase is read next to the paper
 * figures, the explorer is a research tool, and one theme cannot serve both.
 */
function applyTheme() {
  const paper = state.collection === "showcase";
  document.body.dataset.paper = paper ? "1" : "";
  if (!paper) document.body.removeAttribute("data-paper");
  state.pitch?.setTheme(paper ? PAPER_THEME : P);
  if (paper && !state.showcaseLayersApplied) {
    // once, on the first showcase render: the space map and the held defender
    // stay available, they simply do not open the public view
    state.showcaseLayersApplied = true;
    for (const layer of ["wake", "ghost"]) setLayer(layer, false);
  }
}

/** Layers the showcase opens with: the play, not the diagnostics. */
const SHOWCASE_LAYERS = ["trail", "tether", "labels", "candidates"];

function applyCollection() {
  const showcase = state.collection === "showcase" && Boolean(state.showcase);
  $("showcase-control").hidden = !showcase;
  $("explorer-control").hidden = showcase;
  $("effect-control").hidden = showcase;
  for (const button of document.querySelectorAll("#collection .seg-btn")) {
    button.classList.toggle("is-on", button.dataset.collection === state.collection);
  }
  if (showcase) refreshShowcaseList();
  // the explorer shows the reviewers' scores and notes, so it fetches them;
  // the public list never does
  if (!showcase && state.showcase && !state.showcaseDetail) {
    state.showcaseDetail = true;
    loadShowcaseDetail(state.showcase).then(() => {
      refreshShowcaseList();
      render();
    });
  }
  applyPublicChrome(showcase);
  applyTheme();
}

/**
 * Which of the two interfaces this is: the showcase drops the headings, the
 * filters, the role controls, the diagnostic layers and the identifiers that
 * the explorer exists to show. Nothing here changes a value.
 */
/** True while the Submission showcase is open: the public interface. */
function isPublic() {
  return state.collection === "showcase" && Boolean(state.showcase);
}

function applyPublicChrome(showcase) {
  document.body.classList.toggle("is-public", showcase);

  $("brand-name").textContent = showcase ? "The Defender's Dilemma"
                                       : "Off-the-ball value";
  $("brand-sub").hidden = showcase;
  $("btn-source").textContent = showcase ? "Details" : "Source";
  // the public header is the match and Details. The explorer is not a mode a
  // reader chooses between -- it is the research interface, reached with
  // ?explorer=1 (or any ?scene=), and the switch back is there, not here.
  $("collection-control").hidden = showcase;
  // the micro-headings describe the interface, not the football
  $("collection-label").hidden = showcase;
  $("showcase-label").hidden = showcase;
  $("roles-control").hidden = showcase;
  $("showcase-filters").hidden = showcase;
  $("jump-peak").hidden = showcase;
  $("jump-run").hidden = showcase;

  // "Solver policy" is what the model recommends; in public it says so
  for (const button of document.querySelectorAll("#solver-view .seg")) {
    const names = showcase ? PUBLIC_SOLVER_VIEW : RESEARCH_SOLVER_VIEW;
    button.textContent = names[button.dataset.solverview] || button.textContent;
  }
  for (const button of document.querySelectorAll("#story-modes .mode")) {
    const name = button.querySelector(".mode-t");
    const question = button.querySelector(".mode-q");
    name.textContent = showcase
      ? button.dataset.public
      : MODE_RESEARCH_NAME[button.dataset.mode];
    question.hidden = showcase;
  }

  $("layers-row").hidden = showcase;
  // two public layers, each where it means something -- no checkbox wall
  $("show-layers").hidden = true;
  $("space-control").hidden = !(showcase && state.mode === "counterfactual");
  $("roles-card").hidden = showcase;
  $("legend-card").hidden = !showcase;
  $("stat-card").hidden = showcase;
  $("hints-card").hidden = showcase;
  $("notes-card").hidden = showcase;
  const advanced = document.querySelector(".advanced");
  if (advanced) advanced.hidden = showcase;
  const evalTitle = document.querySelector(".evalstrip-head");
  if (evalTitle) evalTitle.hidden = showcase;
}

const PUBLIC_SOLVER_VIEW = { policy: "Equilibrium", actual: "Actual", overlay: "Both" };
const RESEARCH_SOLVER_VIEW = {
  policy: "Solver policy", actual: "Actual movement", overlay: "Overlay",
};

/** Which view each public tab opens on. */
const PUBLIC_VIEW = {
  observed: "full", counterfactual: "full",
  // both solved-moment tabs open cropped: the options fan, its labels and the
  // marker a reader is asked to compare are centimetres apart on a full pitch
  game_solution: "focus", evaluation: "focus",
};

/** The research names for the four modes, which the explorer keeps. */
const MODE_RESEARCH_NAME = {
  observed: "Observed",
  counterfactual: "Counterfactual",
  evaluation: "Evaluation",
  game_solution: "Game solution",
};

function refreshShowcaseList() {
  if (!state.showcase) return;

  const chips = $("showcase-filters");
  chips.replaceChildren();
  for (const filter of availableFilters(state.showcase)) {
    const chip = document.createElement("button");
    chip.className = `fchip${filter.key === state.showcaseFilter ? " is-on" : ""}`;
    chip.textContent = filter.label;
    chip.addEventListener("click", () => {
      state.showcaseFilter = filter.key;
      refreshShowcaseList();
      const first = playableShowcase()[0];
      if (first && first.showcase_id !== state.showcaseId) openShowcase(first.showcase_id);
    });
    chips.appendChild(chip);
  }

  const select = $("showcase-scene");
  select.replaceChildren();
  for (const scene of filtered(state.showcase, state.showcaseFilter)) {
    const option = document.createElement("option");
    option.value = scene.showcase_id;
    const clock = state.index?.find((r) => r.scene_id === scene.scene_id)?.match_clock;
    option.textContent = isPublic() ? publicSelectorLabel(scene, clock)
                                    : selectorLabel(scene);
    // an entry with no published trajectory is listed, with the reason, but
    // cannot be opened; hiding it would misrepresent the curated set
    option.disabled = !scene.playable;
    select.appendChild(option);
  }
  if (state.showcaseId) select.value = state.showcaseId;
}

async function openShowcase(showcaseId) {
  const scene = (state.showcase || []).find((s) => s.showcase_id === showcaseId);
  if (!scene || !scene.playable) return;
  state.showcaseId = showcaseId;
  const row = state.index.find((r) => r.scene_id === scene.scene_id);
  if (!row) return;
  await openScene(row.file);
  refreshShowcaseList();
}

/**
 * The reviewer's own roles become the scene's opening state.
 *
 * Only when they resolve completely: a partial resolution would silently show
 * a different cast from the one that was reviewed, so the annotation stands
 * instead and the Analysis panel says the roles could not be applied.
 */
function applyShowcaseRoles(scene) {
  const curated = currentShowcase();
  if (!curated || !curated.playable) return true;
  const ids = resolveRoles(scene, curated);
  const wanted = curated.roles || {};
  const complete = ["runners", "defenders", "beneficiaries"].every(
    (key) => ids[key].length === (wanted[key] || []).length,
  );
  if (!complete) return false;
  state.selection = new Selection({ ...ids, source: "annotation" });
  return true;
}

function renderTicks() {
  const node = $("time-ticks");
  if (!node) return;
  const { scene, selection } = state;
  node.replaceChildren();
  if (!scene) return;
  const last = Math.max(1, scene.n_frames - 1);

  // The public scrubber carries no marks. A run onset is an inference the
  // demo is not ready to publish, and a shot is metadata some scenes do not
  // have -- a public story may not depend on either. Both stay in the
  // explorer, and the detector itself is untouched.
  if (isPublic()) return;
  const theme = state.pitch?.theme || P;
  const marks = [];
  for (const runnerId of selection.runners) {
    const onset = scene.onsets?.[runnerId];
    const player = scene.byId.get(runnerId);
    if (!onset || !player) continue;
    // the run belongs to an attacker, so on the paper theme the tick is the
    // attack colour rather than the research palette's runner pink
    marks.push({ index: onset.index, colour: theme === P ? P.runner : theme.attack,
                 label: isPublic() ? "run starts" : `run #${player.shirt}`,
                 title: onset.method });
  }
  // the clip is cut around the annotated shot, so t = 0 is that shot
  const shot = Math.round(-scene.t0 * scene.fps);
  if (shot >= 0 && shot <= last) {
    marks.push({ index: shot, colour: theme.text, label: "shot",
                 title: "annotated shot" });
  }

  for (const mark of marks) {
    const tick = document.createElement("i");
    tick.style.left = `${(mark.index / last) * 100}%`;
    tick.style.background = mark.colour;
    tick.dataset.label = mark.label;
    tick.title = mark.title;
    node.appendChild(tick);
  }
}

/**
 * An inline colour for the sidebar's stat, in whichever theme is active.
 *
 * The research palette gives space a cyan and a loss a pink; the paper theme
 * has no third and fourth hue to spare, so a gain is attack blue and a loss
 * defence red -- the same two colours the pitch is already using.
 */
function tint(key) {
  const theme = state.pitch?.theme;
  if (!theme || theme === P) return P[key];
  const paper = { beneficiary: theme.attack, obso: theme.attack,
                  runner: theme.defend, muted: theme.muted };
  return paper[key] || theme[key] || P[key];
}

function renderStat(slot, swap) {
  const { cache, selection } = state;
  const node = $("stat-value");
  const mode = $("wake-mode").value;
  if (mode === "obso") {
    // the panel has to name what the pitch is showing, and in this view that
    // is a frame-level quantity with no beneficiary in it
    const threat = obsoSurface(state.frame);
    $("stat-label").textContent = "OBSO threat · peak";
    if (!threat) { node.textContent = "…"; node.style.color = tint("muted"); return; }
    let peak = 0;
    for (let i = 0; i < threat.values.length; i += 1) {
      if (threat.values[i] > peak) peak = threat.values[i];
    }
    node.textContent = peak.toFixed(3);
    node.style.color = tint("obso");
    return;
  }
  const created = mode === "gain";
  if (!selection.beneficiaries.length) {
    $("stat-label").textContent = created ? "Space created" : "Available space";
    node.textContent = "—";
    node.style.color = tint("muted");
    return;
  }
  const factual = cache.combined(selection.beneficiaries, slot).value;
  if (!created || !swap.length) {
    $("stat-label").textContent = "Available space";
    node.textContent = factual.toFixed(1);
    node.style.color = tint("beneficiary");
    return;
  }
  const counter = cache.combined(selection.beneficiaries, slot, swap).value;
  const gain = factual - counter;
  $("stat-label").textContent = "Space created";
  node.textContent = `${gain >= 0 ? "+" : ""}${gain.toFixed(1)}`;
  node.style.color = gain >= 0 ? tint("beneficiary") : tint("runner");
}

function renderChart(swap, freeze) {
  const { cache, scene, selection } = state;
  if (isPublic()) return;           // the showcase has no chart to fetch one for
  const lib = chartLib();
  if (!lib) return;
  if (!selection.beneficiaries.length) {
    lib.drawChart($("chart"), { theme: state.pitch?.theme || null });
    return;
  }
  const sum = (ids, sw) => {
    const out = new Float64Array(cache.indices.length);
    for (const id of ids) {
      const series = cache.residualSeries(id, sw);
      for (let i = 0; i < out.length; i += 1) out[i] += series[i];
    }
    return Array.from(out);
  };
  lib.drawChart($("chart"), {
    theme: state.pitch?.theme || null,
    times: cache.times,
    factual: sum(selection.beneficiaries, []),
    counter: swap.length ? sum(selection.beneficiaries, swap) : null,
    now: scene.times[state.frame],
    freezeTime: selection.runner ? scene.times[freeze] : null,
  });
}

function renderCandidates(freeze, slot) {
  const { scene, cache, selection } = state;
  if (isPublic()) return;           // the hints card is the explorer's
  const list = $("candidates");
  const card = $("hints-card");
  const toggle = $("hints-toggle");
  const expanded = hintsExpanded();
  const secondary = selection.source === "annotation";
  list.replaceChildren();
  card.classList.toggle("is-secondary", secondary);
  card.classList.toggle("is-collapsed", !expanded);
  toggle.setAttribute("aria-expanded", String(expanded));
  $("hints-chev").textContent = expanded ? "▾" : "▸";
  list.hidden = !expanded;

  if (!selection.runner) {
    $("candidates-label").textContent = "Hints";
    $("hints-count").textContent = "";
    if (expanded) list.innerHTML = '<div class="muted">Pick a runner first.</div>';
    return;
  }
  const wantBeneficiaries = selection.pick === "beneficiary"
    || (selection.defenders.length && !selection.beneficiaries.length);
  let rows;
  let role;
  const ranking = rankingLib();
  if (!ranking) return;
  if (wantBeneficiaries && selection.defenders.length) {
    rows = ranking.rankBeneficiaries(cache, scene, selection.runner,
                                     selection.defenders, freeze, slot).slice(0, 8);
    role = "beneficiary";
    $("candidates-label").textContent = "Gains most";
  } else if (wantBeneficiaries) {
    $("candidates-label").textContent = "Gains most";
    $("hints-count").textContent = "";
    if (expanded) list.innerHTML = '<div class="muted">Pick a defender first.</div>';
    return;
  } else {
    rows = ranking.rankDefenders(scene, selection.runner, freeze).slice(0, 8);
    role = "defender";
    $("candidates-label").textContent = "Reacts most";
  }
  $("hints-count").textContent = expanded ? "" : `${rows.length}`;
  if (!expanded) return;

  for (const row of rows) {
    const item = document.createElement("div");
    item.className = "cand";
    if (selection.roleOf(row.id)) item.classList.add("is-picked");
    item.innerHTML = '<span class="cand-shirt"></span><span class="cand-name"></span>'
      + '<span class="cand-value"></span>';
    item.querySelector(".cand-shirt").textContent = `#${row.shirt}`;
    item.querySelector(".cand-name").textContent = row.name;
    item.querySelector(".cand-value").textContent = row.caption;
    item.addEventListener("click", () => {
      state.selection.arm(role).applyClick(scene, row.id);
      render();
    });
    list.appendChild(item);
  }
}

/**
 * Who is in this play, in three lines: a shape, a name and the football
 * position the solver role already is. It replaces the explorer's role dock,
 * which exists to let someone pick a different cast.
 */
/**
 * Put one other player beside the curated one.
 *
 * A defender becomes the marking comparison, an attacker the space one, and
 * the curated players themselves clear it. Nothing here touches
 * `state.selection`: the cast the solver was run on stays exactly as curated,
 * which is why no equilibrium or evaluation number moves with a comparison.
 */
function compareWith(playerId) {
  const { scene, selection } = state;
  const player = scene?.byId.get(playerId);
  if (!player || player.gk) return;
  // a side cannot take the other side's role: only a defender may be the
  // Defender, only an attacker the Runner or the Teammate
  const role = state.sweep;
  const wantsDefender = role === "defender";
  if (wantsDefender !== (player.side === "defend")) return;
  // clicking the curated player, or the one already being compared, clears it
  const curated = selection.runner === playerId
    || selection.defenders.includes(playerId)
    || selection.beneficiaries.includes(playerId);
  // the three roles are independent: picking a teammate leaves a defender
  // already in the comparison alone, so a reader can build a whole alternative
  // cast rather than one substitution at a time
  const next = { ...(state.compare || {}) };
  const already = next[role] === playerId;
  if (curated || already) delete next[role];
  else next[role] = playerId;
  state.compare = Object.keys(next).length ? next : null;
  render();
}

/** Whoever is being compared, in whichever role, or null. */
function comparedPlayer() {
  const c = state.compare;
  return c ? (c.runner || c.teammate || c.defender || null) : null;
}

/** Every player the reader has put into the comparison, in role order. */
function comparedPlayers() {
  const c = state.compare;
  if (!c) return [];
  return [c.runner, c.teammate, c.defender].filter(Boolean);
}

/** Back to the curated play. */
function clearCompare() {
  if (!state.compare) return;
  state.compare = null;
  render();
}

/** The evidence file for this scene, fetched once, or null until it lands. */
function compareFor() {
  const { scene } = state;
  if (!scene) return null;
  if (state.compareData?.sceneId === scene.scene_id) return state.compareData.payload;
  const lib = compareLib();
  if (!lib) return null;
  if (state.compareData?.pending === scene.scene_id) return null;
  state.compareData = { pending: scene.scene_id };
  lib.loadCompare(scene.scene_id).then((payload) => {
    if (state.scene?.scene_id !== scene.scene_id) return;
    state.compareData = { sceneId: scene.scene_id, payload };
    render();
  });
  return null;
}

/** The marker shape each solver role is drawn with, as a CSS class. */
const SHAPE_CLASS = {
  "ball carrier": "circle", runner: "diamond",
  beneficiary: "triangle", defender: "square",
};

/** "D. Ginczek" -> "Ginczek": a legend has room for the name people use. */
function surnameOf(player) {
  const parts = String(player.name || "").trim().split(/\s+/);
  return parts.length > 1 ? parts[parts.length - 1] : (parts[0] || `#${player.shirt}`);
}

function renderDock() {
  const { scene, selection } = state;
  for (const role of ["runner", "beneficiary", "defender"]) {
    const slot = $(`slot-${role}`);
    const body = $(`body-${role}`);
    body.replaceChildren();
    slot.classList.toggle("is-armed", selection.armed && selection.pick === role);
    slot.classList.toggle("is-filled", selection.ids(role).length > 0);
    const ids = selection.ids(role).filter((id) => scene.byId.has(id));
    if (!ids.length) {
      const empty = document.createElement("span");
      empty.className = "slot-empty";
      empty.textContent = (selection.armed && selection.pick === role)
        ? "pick on pitch" : "drop a player";
      body.appendChild(empty);
      continue;
    }
    for (const playerId of ids) {
      body.appendChild(chip(scene, role, playerId));
    }
  }
  const meta = $("dock-meta");
  meta.replaceChildren();
  if (selection.runner) {
    const onset = onsetFor(scene, selection.runner);
    const line = document.createElement("div");
    line.textContent = `Run starts ${scene.times[onset.index] >= 0 ? "+" : ""}`
      + `${scene.times[onset.index].toFixed(1)} s`;
    line.title = `detected by: ${onset.method}`;
    meta.appendChild(line);
  }
  if (selection.source === "suggested") {
    const line = document.createElement("div");
    line.textContent = "Auto triplet";
    meta.appendChild(line);
  }
}

function chip(scene, role, playerId) {
  const player = scene.byId.get(playerId);
  const node = document.createElement("span");
  node.className = "chip";
  // the dock's chips follow whichever palette the pitch is drawn in, so a
  // reader is never told the runner is pink and shown a blue diamond
  node.style.background = state.pitch?.roles?.[role] || ROLE_COLOUR[role];
  node.draggable = false;
  node.dataset.player = playerId;
  node.dataset.role = role;
  const text = document.createElement("span");
  text.textContent = `#${player.shirt} ${player.name}`;
  const close = document.createElement("span");
  close.className = "chip-x";
  close.textContent = "×";
  close.addEventListener("pointerdown", (event) => {
    event.stopPropagation();
    const bucket = state.selection.ids(role);
    const at = bucket.indexOf(playerId);
    if (at >= 0) bucket.splice(at, 1);
    state.selection.source = "manual";
    render();
  });
  node.append(text, close);
  node.addEventListener("pointerdown", (event) => beginDrag(event, player, role));
  return node;
}

// ---------------------------------------------------------------------------
// drag and drop
// ---------------------------------------------------------------------------
function beginDrag(event, player, fromRole = null) {
  if (event.button !== undefined && event.button !== 0) return;
  event.preventDefault();
  const start = { x: event.clientX, y: event.clientY };
  state.drag = { playerId: player.id, side: player.side, fromRole, moved: false, start };

  const ghost = document.createElement("div");
  ghost.className = "drag-ghost";
  ghost.textContent = `#${player.shirt} ${player.name}`;
  ghost.style.display = "none";
  document.body.appendChild(ghost);
  state.drag.ghost = ghost;

  document.body.classList.add("dragging");
  markDropTargets(player.side);

  const move = (moveEvent) => {
    const dx = moveEvent.clientX - start.x;
    const dy = moveEvent.clientY - start.y;
    if (!state.drag.moved && Math.hypot(dx, dy) < 4) return;
    const first = !state.drag.moved;
    state.drag.moved = true;
    ghost.style.display = "block";
    ghost.style.left = `${moveEvent.clientX}px`;
    ghost.style.top = `${moveEvent.clientY}px`;
    const slot = slotUnder(moveEvent.clientX, moveEvent.clientY);
    const role = slot ? slot.dataset.role : null;
    const droppable = role && ROLE_SIDE[role] === state.drag.side;
    // dragging a chip from one role to the other moves that player only
    const moving = droppable && fromRole && fromRole !== role;
    for (const node of document.querySelectorAll(".slot")) {
      node.classList.toggle("is-over", droppable && node === slot);
      node.classList.toggle("is-swap", Boolean(moving) && node === slot);
    }
    ghost.classList.toggle("is-swap", Boolean(moving));
    ghost.textContent = droppable
      ? `#${player.shirt} \u2192 ${ROLE_LABEL[role]}`
      : `#${player.shirt} ${player.name}`;
    if (first) render();             // bring the compatible side forward
  };

  const up = (upEvent) => {
    document.removeEventListener("pointermove", move);
    document.removeEventListener("pointerup", up);
    ghost.remove();
    document.body.classList.remove("dragging");
    for (const node of document.querySelectorAll(".slot")) {
      node.classList.remove("is-over", "is-droppable", "is-blocked", "is-swap");
    }
    const slot = state.drag.moved ? slotUnder(upEvent.clientX, upEvent.clientY) : null;
    const dragged = state.drag;
    state.drag = null;

    if (slot) {
      const role = slot.dataset.role;
      if (ROLE_SIDE[role] === dragged.side) {
        // one chip, one player: the rest of both roles is left alone
        state.selection.move(state.scene, role, dragged.playerId);
      }
    } else if (!dragged.moved && isPublic()) {
      // selection is not editing: Dilemma compares, every other mode ignores
      // the click, and the curated cast is never written to
      if (state.mode === "counterfactual") compareWith(dragged.playerId);
    } else if (!dragged.moved) {
      if (dragged.fromRole) state.selection.arm(dragged.fromRole);
      else state.selection.applyClick(state.scene, dragged.playerId);
      // clicking always inspects, whatever the click did to the roles
      state.inspect = dragged.playerId;
    }
    render();
  };

  document.addEventListener("pointermove", move);
  document.addEventListener("pointerup", up);
}

function markDropTargets(side) {
  for (const node of document.querySelectorAll(".slot")) {
    const ok = ROLE_SIDE[node.dataset.role] === side;
    node.classList.toggle("is-droppable", ok);
    node.classList.toggle("is-blocked", !ok);   // stays visibly inactive
  }
}

function slotUnder(x, y) {
  for (const node of document.querySelectorAll(".slot")) {
    const box = node.getBoundingClientRect();
    if (x >= box.left && x <= box.right && y >= box.top && y <= box.bottom) return node;
  }
  return null;
}

// ---------------------------------------------------------------------------
// controls
// ---------------------------------------------------------------------------
function setView(view) {
  state.view = view;
  for (const button of document.querySelectorAll("#view-toggle .seg")) {
    button.classList.toggle("is-on", button.dataset.view === view);
  }
}

/** Hints stay collapsed while the roles are the annotated ones. */
function hintsExpanded() {
  if (state.hintsOpen !== null) return state.hintsOpen;
  return state.selection.source !== "annotation";
}

function bindControls() {
  for (const button of document.querySelectorAll("#view-toggle .seg")) {
    button.addEventListener("click", () => {
      // remembered for this tab only: leaving it drops the choice
      state.viewByMode[state.mode] = button.dataset.view;
      setView(button.dataset.view);
      render();
    });
  }
  $("hints-toggle").addEventListener("click", () => {
    state.hintsOpen = !hintsExpanded();
    render();
  });
  $("btn-source").addEventListener("click", () => openSheet(true));
  $("source-close").addEventListener("click", () => openSheet(false));
  $("source-sheet").addEventListener("click", (event) => {
    if (event.target.id === "source-sheet") openSheet(false);
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") openSheet(false);
  });

  $("scene").addEventListener("change", (event) => openScene(event.target.value));
  for (const effect of ["strong", "medium"]) {
    $(`effect-${effect}`).addEventListener("change", (event) => {
      if (event.target.checked) state.effects.add(effect);
      else state.effects.delete(effect);
      if (!state.effects.size) {
        state.effects.add(effect);
        event.target.checked = true;
        return;
      }
      refreshSceneList();
      const rows = visibleScenes();
      if (rows.length && !rows.some((r) => r.file === $("scene").value)) {
        openScene(rows[0].file);
      }
    });
  }
  $("time").addEventListener("input", (event) => {
    state.frame = Number(event.target.value);
    render();
  });
  $("play").addEventListener("click", togglePlay);
  $("jump-run").addEventListener("click", () => { state.frame = freezeIndex(); render(); });
  $("jump-peak").addEventListener("click", () => {
    const frame = peakGainFrame();
    if (frame != null) { state.frame = frame; render(); }
  });
  $("btn-reset").addEventListener("click", () => {
    state.selection = new Selection();
    render();
  });
  $("btn-auto").addEventListener("click", () => {
    $("status").textContent = "searching…";
    setTimeout(() => {
      const best = rankingLib()?.autoTriplet(state.cache, state.scene,
                                            state.selection.runner);
      if (best) {
        state.selection = new Selection({
          runners: [best.runner], defenders: [best.defender],
          beneficiaries: [best.beneficiary], source: "suggested",
        });
        const frame = peakGainFrame();
        if (frame != null) state.frame = frame;
      }
      $("status").textContent = "";
      render();
    }, 16);
  });
  $("btn-annotation").addEventListener("click", () => {
    state.selection = Selection.fromRoles(state.scene.roles, "annotation");
    render();
  });
  for (const role of ["runner", "beneficiary", "defender"]) {
    $(`slot-${role}`).addEventListener("pointerdown", (event) => {
      if (event.target.closest(".chip")) return;
      state.selection.arm(role);
      render();
    });
  }
  for (const node of document.querySelectorAll("[data-layer]")) {
    node.addEventListener("change", () => {
      const name = node.dataset.layer;
      if (node.checked) state.layers.add(name);
      else state.layers.delete(name);
      render();
      // a layer that opens an Analysis section is no use below the fold
      if (node.checked && ANALYSIS_LAYERS.has(name)) {
        $("analysis-card")?.scrollIntoView({ block: "nearest", behavior: "smooth" });
      }
    });
  }
  for (const node of document.querySelectorAll("[data-showlayer]")) {
    node.addEventListener("change", () => setLayer(node.dataset.showlayer,
                                                   node.checked) || render());
  }
  $("compare-reset").addEventListener("click", clearCompare);
  $("field-on").addEventListener("change", (event) => {
    state.field = event.target.checked;
    render();
  });
  $("space-on").addEventListener("change", (event) => {
    state.space = event.target.checked;
    render();
  });

  const showLayers = $("show-layers-toggle");
  if (showLayers) {
    showLayers.addEventListener("click", () => {
      const body = $("show-layers-body");
      const open = body.hidden;
      body.hidden = !open;
      $("show-layers-chev").textContent = open ? "\u25be" : "\u25b8";
      showLayers.setAttribute("aria-expanded", String(open));
    });
  }
  $("wake-mode").addEventListener("change", render);

  for (const button of document.querySelectorAll("#collection .seg-btn")) {
    button.addEventListener("click", async () => {
      if (state.collection === button.dataset.collection) return;
      state.collection = button.dataset.collection;
      applyCollection();
      if (state.collection === "showcase") {
        const first = playableShowcase()[0];
        if (first) await openShowcase(first.showcase_id);
      }
      render();
    });
  }
  $("showcase-scene").addEventListener("change", (event) => {
    openShowcase(event.target.value);
  });

  // clicking a candidate endpoint selects it for the Analysis panel; clicking
  // the pitch background clears the selection
  $("pitch").addEventListener("click", (event) => {
    const ray = event.target?.getAttribute?.("data-ray");
    if (ray == null) return;
    event.stopPropagation();
    const index = Number(ray);
    state.endpoint = state.endpoint === index ? null : index;
    render();
    if (state.endpoint != null) {
      $("an-pass")?.scrollIntoView({ block: "nearest", behavior: "smooth" });
    }
  });

  const valueToggle = $("value-toggle");
  if (valueToggle) {
    valueToggle.addEventListener("click", () => {
      state.valueOpen = !state.valueOpen;
      render();
    });
  }

  const detailsToggle = $("details-toggle");
  if (detailsToggle) {
    detailsToggle.addEventListener("click", () => {
      state.detailsOpen = !state.detailsOpen;
      render();
    });
  }

  const analysisToggle = $("analysis-toggle");
  if (analysisToggle) {
    analysisToggle.addEventListener("click", () => {
      const body = $("analysis-body");
      const open = body.hidden;
      body.hidden = !open;
      $("analysis-chev").textContent = open ? "\u25be" : "\u25b8";
      analysisToggle.setAttribute("aria-expanded", String(open));
    });
  }

  bindStory();
}

/** Advanced layers: collapsed, but never silently active. */
const ADVANCED_LAYERS = ["candidates", "lane", "paths", "reach", "solver",
                         "commands", "passes"];

//: Advanced layers that start on. The header hint exists so a layer a *mode*
//: switched on is not invisible; naming one that has been on since load would
//: make the default state look like something had been changed.
const ADVANCED_DEFAULT_ON = new Set(["candidates"]);

function renderAdvanced() {
  const body = $("adv-body");
  const toggle = $("adv-toggle");
  if (!body || !toggle) return;
  body.hidden = !state.advancedOpen;
  $("adv-chev").textContent = state.advancedOpen ? "\u25be" : "\u25b8";
  toggle.setAttribute("aria-expanded", String(state.advancedOpen));
  // a mode can switch one of these on; the header says so while it is closed,
  // otherwise the pitch would change with no visible cause
  const on = ADVANCED_LAYERS
    .filter((name) => state.layers.has(name) && !ADVANCED_DEFAULT_ON.has(name))
    .map((name) => LAYER_LABEL[name] || name);
  $("adv-hint").textContent = state.advancedOpen || !on.length ? "" : on.join(" · ");
}

/** The four story modes, the story-role picker, and the solver view. */
function bindStory() {
  const advanced = $("adv-toggle");
  if (advanced) {
    advanced.addEventListener("click", () => {
      state.advancedOpen = !state.advancedOpen;
      renderAdvanced();
    });
  }
  for (const button of document.querySelectorAll("#story-modes .mode")) {
    button.addEventListener("click", () => setMode(button.dataset.mode));
  }
  const roles = $("story-roles");
  if (roles) {
    // delegated, because the buttons are rebuilt whenever the panel renders
    roles.addEventListener("click", (event) => {
      const button = event.target.closest("[data-story-role]");
      if (button) setStoryRole(button.dataset.storyRole);
    });
  }
  for (const button of document.querySelectorAll("#solver-view .seg")) {
    button.addEventListener("click", () => {
      state.solverView = button.dataset.solverview;
      for (const other of document.querySelectorAll("#solver-view .seg")) {
        other.classList.toggle("is-on", other === button);
      }
      render();
    });
  }
}

let sheetLoaded = false;

/**
 * The method sheet, fetched the first time it is opened.
 *
 * It is a page of prose that no first paint needs, so it is not in the shell.
 * A failed fetch leaves the sheet's own links and footer, which is a usable
 * fallback rather than an empty dialog.
 */
function openSheet(open) {
  $("source-sheet").hidden = !open;
  if (!open || sheetLoaded) return;
  sheetLoaded = true;
  fetch("details.html")
    .then((response) => (response.ok ? response.text() : null))
    .then((html) => { if (html) $("sheet-body").innerHTML = html; })
    .catch(() => { sheetLoaded = false; });
}

/**
 * Playback pauses on a solved moment so it can be read.
 *
 * Two seconds, once per visit to that frame: `pausedAt` remembers which one
 * was already honoured, so scrubbing back and forth does not re-trigger it and
 * the pause never fights the reader.
 */
function holdOnSolvedMoment() {
  // both solved-moment tabs hold: Player evaluation's options and ranks are
  // read at the same three moments the equilibrium is, and they go past in
  // two frames otherwise
  const holds = state.mode === "game_solution" || state.mode === "evaluation";
  if (!state.playing || !holds || !isPublic()) return;
  const data = solverFor();
  if (data?.kind !== "bundle_panels") return;
  const found = panelAt(data, state.frame);
  if (!found?.exact || state.pausedAt === state.frame) return;
  state.pausedAt = state.frame;
  if (state.timer) clearInterval(state.timer);
  state.timer = null;
  setTimeout(() => {
    if (!state.playing) return;             // they pressed pause meanwhile
    startPlayTimer();
  }, 2000);
}

function startPlayTimer() {
  if (state.timer) clearInterval(state.timer);
  state.timer = setInterval(() => {
    const from = state.frame;
    let next = from + 2;
    if (next > state.scene.n_frames - 1) {
      next = 0;
      state.pausedAt = null;                // a new pass may pause again
    } else {
      // Playback moves two frames at a time, so a solved frame with an odd
      // index -- S05's 55 and 85 -- would be stepped straight over and only
      // 70 would ever be landed on. Land on any solved frame the step would
      // cross, so every moment is shown and every moment pauses.
      const crossed = solvedFrameBetween(from, next);
      if (crossed != null) next = crossed;
    }
    state.frame = next;
    render();
    holdOnSolvedMoment();
  }, 80);
}

/** The first solved frame strictly after `from` and at or before `to`. */
function solvedFrameBetween(from, to) {
  const data = solverFor();
  if (data?.kind !== "bundle_panels") return null;
  const frames = (data.panels || [])
    .map((panel) => frameOfPanel(panel))
    .filter((frame) => frame != null && frame > from && frame <= to)
    .sort((a, b) => a - b);
  return frames.length ? frames[0] : null;
}

function togglePlay() {
  state.playing = !state.playing;
  $("play").textContent = state.playing ? "❚❚" : "▶";
  if (state.timer) clearInterval(state.timer);
  state.timer = null;
  if (!state.playing) return;
  state.pausedAt = null;
  startPlayTimer();
}

boot().catch((error) => {
  $("status").textContent = `failed: ${error.message}`;
  console.error(error);
});
