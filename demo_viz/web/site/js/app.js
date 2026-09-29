// Wiring: scene loading, the role dock, drag and drop, overlays, playback.

import { InfluenceCache, velocityAt } from "./influence.js";
import { P, ROLE_COLOUR, ROLE_LABEL, ROLE_SIDE } from "./palette.js";
import { Pitch } from "./pitch.js";
import { autoTriplet, rankBeneficiaries, rankDefenders } from "./ranking.js";
import { drawChart } from "./chart.js";
import { Selection } from "./selection.js";
import { ballAt, loadIndex, loadScene, onsetFor, playerAt } from "./scene.js";
import {
  candidatePasses, componentsAt, loadControl, loadScoreGrid, sampleAt, surfaceAt,
} from "./obso.js";
import { REACH, reachableMask } from "./reach.js";
import { loadSolver, primaryTrajectory, summary as solverSummary } from "./solver.js";
import {
  PROVENANCE_LABEL, availableFilters, defaultFilter, filtered, loadShowcase,
  matchLabel, resolveRoles, selectorLabel,
} from "./showcase.js";
import { loadModelCard, loadRelease, releaseFor, releaseRow } from "./release.js";
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
  layers: new Set(["trail", "tether", "wake", "ghost", "labels", "candidates"]),
  view: "full",              // "full" | "focus"
  hintsOpen: null,           // null = follow the mode, true/false = user's choice
  drag: null,
  obso: null,                // { entry, score } once the threat view is used
  obsoValues: null,          // scratch buffer for the current frame's surface
  inspect: null,             // player id under the last click, for the panel
  endpoint: null,            // index of the selected candidate ray
  reach: null,               // { playerId, frame, mask } memo for one frame
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
  } else {
    const frame = peakGainFrame();
    if (frame != null) state.frame = frame;
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
    if (wanted && wanted.scene) state.collection = "explorer";
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
  const onset = onsetFor(scene, state.selection.runner);
  state.frame = peakGainFrame() ?? Math.min(onset.index + Math.round(1.5 * scene.fps),
                                            scene.n_frames - 1);
  // a new scene invalidates everything keyed to the old one; clear before the
  // URL state is applied, or it would wipe what the URL just asked for
  state.endpoint = null;
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
  $("subtitle").textContent = scene.subtitle;
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

  pitch.setView(state.view === "focus" ? fitBox() : null);
  pitch.clearDynamic();

  let hints = [];
  if (selection.runner && !selection.defenders.length) {
    hints = rankDefenders(scene, selection.runner, freeze).slice(0, 4).map((r) => r.id);
  } else if (selection.runner && selection.defenders.length && !selection.beneficiaries.length) {
    hints = rankBeneficiaries(cache, scene, selection.runner, selection.defenders,
                              freeze, slot).filter((r) => r.gain > 0.05).slice(0, 3)
      .map((r) => r.id);
  }

  const mode = $("wake-mode").value;
  const threat = mode === "obso" ? obsoSurface(index) : null;
  if (state.layers.has("wake") && threat) {
    // frame-level threat: it does not move when the roles change, so it is
    // drawn in its own colour rather than the beneficiary's
    pitch.drawField(threat.grid, threat.values, scene, { colour: P.obso });
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

  if (state.layers.has("paths")
      || (state.mode === "game_solution" && state.solverView !== "policy")) {
    pitch.drawPaths(scene);
  }
  renderReach(index);
  const fan = renderPasses(index, threat);
  renderSolverLayer();
  if (state.layers.has("lane")) pitch.drawLane(scene, selection.beneficiaries, index);
  if (state.layers.has("trail") && selection.runner) {
    pitch.drawTrail(scene, selection.runner, index);
  }
  if (state.layers.has("tether") && selection.runner) {
    pitch.drawTether(scene, selection.runner, selection.defenders, index,
                     { labels: state.layers.has("labels") });
  }
  if (state.layers.has("ghost")) {
    pitch.drawGhost(scene, cache, selection.defenders, index, freeze,
                    { labels: state.layers.has("labels") });
  }
  pitch.drawPlayers(scene, index, selection, {
    labels: state.layers.has("labels"),
    activeSide,
    hints: state.layers.has("candidates") ? hints : [],
    dragging: state.drag ? state.drag.playerId : null,
  });

  renderDock();
  // the headline stat is off-ball context, which the paper puts below the
  // story; outside Observed its space belongs to the mode's own panel
  const stat = document.querySelector(".card.stat");
  if (stat) stat.hidden = state.mode !== "observed";
  renderStat(slot, swap);
  renderChart(swap, freeze);
  renderCandidates(freeze, slot);
  renderAnalysis(index, slot, threat, fan);
  renderTicks();
  renderEvalStrip(index);
  $("readout").textContent = `${scene.times[index] >= 0 ? "+" : ""}${scene.times[index].toFixed(2)} s`;
  $("time").value = String(index);
}

/**
 * Crop to the picked players, plus the ball only while it is near them.
 * A ball cleared to the far corner would otherwise pull the box back out to a
 * full-pitch view for no gain.
 */
function fitBox(margin = 12, minimumWidth = 46, ballAttach = 18) {
  const { scene, selection } = state;
  const points = [];
  for (const role of ["runner", "beneficiary", "defender"]) {
    for (const id of selection.ids(role)) {
      const player = scene.byId.get(id);
      if (!player) continue;
      const position = playerAt(scene, player, state.frame);
      if (position) points.push(position);
    }
  }
  if (!points.length) return null;
  const centreX = points.reduce((a, p) => a + p[0], 0) / points.length;
  const centreY = points.reduce((a, p) => a + p[1], 0) / points.length;
  const ball = ballAt(scene, state.frame);
  if (ball && Math.hypot(ball[0] - centreX, ball[1] - centreY) <= ballAttach) {
    points.push(ball);
  }

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
  const x0 = Math.min(Math.max(cx - width / 2, -fullW / 2), fullW / 2 - width);
  const y0 = Math.min(Math.max(cy - height / 2, -fullH / 2), fullH / 2 - height);
  return [x0, y0, width, height];
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
  if (!scene) return null;
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
    const row = releaseRow(payload, index, receiver);
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
  const on = Boolean(visible) && (MODE_SECTIONS[state.mode] || []).includes(id);
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
  $("analysis-label").textContent = MODE_TITLE[state.mode] || "Analysis";
  $("analysis-hint").textContent = player ? `#${player.shirt}` : "";
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

const MODE_TITLE = {
  observed: "What happened",
  counterfactual: "What else was possible",
  evaluation: "How good the observed action was",
  game_solution: "What the equilibrium recommends",
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

function setMode(mode, quiet = false) {
  if (!MODE_SECTIONS[mode]) return;
  state.mode = mode;
  for (const button of document.querySelectorAll("#story-modes .mode")) {
    const on = button.dataset.mode === mode;
    button.classList.toggle("is-on", on);
    button.setAttribute("aria-selected", String(on));
  }
  // each mode brings on the layer that answers its question; the reviewer can
  // still switch it off, and switching back to Observed puts it away
  setLayer("reach", mode === "counterfactual");
  setLayer("solver", mode === "game_solution");
  $("solver-view").hidden = mode !== "game_solution";
  if (!quiet) render();
}

function setLayer(name, on) {
  if (on) state.layers.add(name); else state.layers.delete(name);
  const box = document.querySelector(`[data-layer="${name}"]`);
  if (box) box.checked = on;
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

function renderStoryRoles() {
  const node = $("story-roles");
  node.replaceChildren();
  for (const role of Object.keys(STORY_ROLE_LABEL)) {
    const button = document.createElement("button");
    button.className = `roleb${state.storyRole === role ? " is-on" : ""}`;
    button.type = "button";
    button.dataset.storyRole = role;
    button.textContent = STORY_ROLE_LABEL[role];
    button.setAttribute("aria-pressed", String(state.storyRole === role));
    node.append(button);
  }
}

function renderDecision(index) {
  if (!section("an-decision", true)) return;
  const { scene } = state;
  renderStoryRoles();
  const player = resolveStoryRole(index);
  const time = scene.times[index];
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
  contractRows($("an-eval-rows"), [...block.metrics, block.optimal_action],
               (r) => (r.name === "optimal_action" ? actionText(r) : metricText(r)));
  const none = block.metrics.every((m) => m.availability !== "available");
  note.hidden = !none;
  if (none) {
    note.textContent = "Evaluation outputs are not available for this scene yet. "
      + "This view populates from the player-evaluation pipeline; nothing is "
      + "substituted in the meantime.";
  }
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
function renderEvalStrip(index) {
  const strip = $("eval-strip");
  strip.hidden = state.mode !== "evaluation";
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
  const nearest = lib.drawEvalStrip(plot, {
    series, nFrames: state.scene.n_frames, frame: index,
    width: plot.clientWidth,
    onSeek: (frame) => { state.frame = frame; render(); },
  });
  // the nearest computed sample, named as such: between two solved frames
  // there is no value, and saying which frame it came from is the honest read
  readout.textContent = (nearest || [])
    .map((near, i) => (near
      ? `${series[i].label} ${near.value.toFixed(3)}`
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
  if (!point || !state.obso || state.obso.pending) return null;
  const { entry, score } = state.obso;
  if (!entry || !score) return null;
  return componentsAt(entry, score, state.scene, index, point[0], point[1],
                      surfaceAt.lastControl);
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
  state.release = { sceneId: scene.scene_id, pending: true, data: null };
  Promise.all([loadRelease(scene.scene_id), loadModelCard()]).then(([data]) => {
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
  const result = releaseFor(payload, index, receiver, state.endpoint);
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
function applyCollection() {
  const showcase = state.collection === "showcase" && Boolean(state.showcase);
  $("showcase-control").hidden = !showcase;
  $("explorer-control").hidden = showcase;
  $("effect-control").hidden = showcase;
  for (const button of document.querySelectorAll("#collection .seg-btn")) {
    button.classList.toggle("is-on", button.dataset.collection === state.collection);
  }
  if (showcase) refreshShowcaseList();
}

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
    option.textContent = selectorLabel(scene);
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

  const marks = [];
  for (const runnerId of selection.runners) {
    const onset = scene.onsets?.[runnerId];
    const player = scene.byId.get(runnerId);
    if (!onset || !player) continue;
    marks.push({ index: onset.index, colour: P.runner,
                 label: `run #${player.shirt}`, title: onset.method });
  }
  // the clip is cut around the annotated shot, so t = 0 is that shot
  const shot = Math.round(-scene.t0 * scene.fps);
  if (shot >= 0 && shot <= last) {
    marks.push({ index: shot, colour: P.text, label: "shot",
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

function renderStat(slot, swap) {
  const { cache, selection } = state;
  const node = $("stat-value");
  const mode = $("wake-mode").value;
  if (mode === "obso") {
    // the panel has to name what the pitch is showing, and in this view that
    // is a frame-level quantity with no beneficiary in it
    const threat = obsoSurface(state.frame);
    $("stat-label").textContent = "OBSO threat · peak";
    if (!threat) { node.textContent = "…"; node.style.color = P.muted; return; }
    let peak = 0;
    for (let i = 0; i < threat.values.length; i += 1) {
      if (threat.values[i] > peak) peak = threat.values[i];
    }
    node.textContent = peak.toFixed(3);
    node.style.color = P.obso;
    return;
  }
  const created = mode === "gain";
  if (!selection.beneficiaries.length) {
    $("stat-label").textContent = created ? "Space created" : "Available space";
    node.textContent = "—";
    node.style.color = P.muted;
    return;
  }
  const factual = cache.combined(selection.beneficiaries, slot).value;
  if (!created || !swap.length) {
    $("stat-label").textContent = "Available space";
    node.textContent = factual.toFixed(1);
    node.style.color = P.beneficiary;
    return;
  }
  const counter = cache.combined(selection.beneficiaries, slot, swap).value;
  const gain = factual - counter;
  $("stat-label").textContent = "Space created";
  node.textContent = `${gain >= 0 ? "+" : ""}${gain.toFixed(1)}`;
  node.style.color = gain >= 0 ? P.beneficiary : P.runner;
}

function renderChart(swap, freeze) {
  const { cache, scene, selection } = state;
  if (!selection.beneficiaries.length) { drawChart($("chart"), {}); return; }
  const sum = (ids, sw) => {
    const out = new Float64Array(cache.indices.length);
    for (const id of ids) {
      const series = cache.residualSeries(id, sw);
      for (let i = 0; i < out.length; i += 1) out[i] += series[i];
    }
    return Array.from(out);
  };
  drawChart($("chart"), {
    times: cache.times,
    factual: sum(selection.beneficiaries, []),
    counter: swap.length ? sum(selection.beneficiaries, swap) : null,
    now: scene.times[state.frame],
    freezeTime: selection.runner ? scene.times[freeze] : null,
  });
}

function renderCandidates(freeze, slot) {
  const { scene, cache, selection } = state;
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
  if (wantBeneficiaries && selection.defenders.length) {
    rows = rankBeneficiaries(cache, scene, selection.runner, selection.defenders,
                             freeze, slot).slice(0, 8);
    role = "beneficiary";
    $("candidates-label").textContent = "Gains most";
  } else if (wantBeneficiaries) {
    $("candidates-label").textContent = "Gains most";
    $("hints-count").textContent = "";
    if (expanded) list.innerHTML = '<div class="muted">Pick a defender first.</div>';
    return;
  } else {
    rows = rankDefenders(scene, selection.runner, freeze).slice(0, 8);
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
  node.style.background = ROLE_COLOUR[role];
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
    button.addEventListener("click", () => { setView(button.dataset.view); render(); });
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
      const best = autoTriplet(state.cache, state.scene, state.selection.runner);
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

/** The four story modes, the story-role picker, and the solver view. */
function bindStory() {
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

function openSheet(open) {
  $("source-sheet").hidden = !open;
}

function togglePlay() {
  state.playing = !state.playing;
  $("play").textContent = state.playing ? "❚❚" : "▶";
  if (state.timer) clearInterval(state.timer);
  if (!state.playing) return;
  state.timer = setInterval(() => {
    state.frame += 2;
    if (state.frame > state.scene.n_frames - 1) state.frame = 0;
    render();
  }, 80);
}

boot().catch((error) => {
  $("status").textContent = `failed: ${error.message}`;
  console.error(error);
});
