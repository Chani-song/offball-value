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
import { commandName, loadSolver, primaryTrajectory, summary as solverSummary } from "./solver.js";

const $ = (id) => document.getElementById(id);

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
    fit: params.get("fit"),
    about: params.get("about") === "1",
    mode: params.get("mode"),          // space | gain | obso
    passes: params.get("passes") === "1",
    on: params.get("on"),              // comma list of layers to switch on
    inspect: params.get("inspect"),    // shirt number to open the panel on
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
  if (wanted.mode && ["space", "gain", "obso"].includes(wanted.mode)) {
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
  bindControls();
  refreshSceneList();
  const wanted = urlState();
  let start = visibleScenes()[0];
  if (wanted && wanted.scene) {
    const match = state.index.find(
      (row) => row.file === wanted.scene || row.scene_id === wanted.scene,
    );
    if (match) start = match;
  }
  if (start) await openScene(start.file, wanted);
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
  const onset = onsetFor(scene, state.selection.runner);
  state.frame = peakGainFrame() ?? Math.min(onset.index + Math.round(1.5 * scene.fps),
                                            scene.n_frames - 1);
  if (wanted) applyUrlState(wanted);
  // a new scene invalidates everything keyed to the old one
  state.endpoint = null;
  state.reach = null;
  state.solver = null;
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

  if (state.layers.has("paths")) pitch.drawPaths(scene);
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
  renderStat(slot, swap);
  renderChart(swap, freeze);
  renderCandidates(freeze, slot);
  renderAnalysis(index, slot, threat, fan);
  renderTicks();
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
  if (threat) {
    for (const ray of fan.rays) {
      ray.obso = sampleAt(threat.grid, threat.values, ray.end[0], ray.end[1]);
    }
  }
  pitch.drawCandidates(scene, fan, {
    values: Boolean(threat), selected: state.endpoint,
  });
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
function reachFor(index) {
  const { scene } = state;
  if (!state.inspect) return null;
  const player = scene.byId.get(state.inspect);
  if (!player) return null;
  const x = player.x[index];
  const y = player.y[index];
  if (!Number.isFinite(x) || !Number.isFinite(y)) return null;
  if (state.reach && state.reach.playerId === state.inspect && state.reach.frame === index) {
    return state.reach.mask;
  }
  const { vx, vy } = motionOf(player, index);
  const mask = reachableMask([x, y], [vx, vy]);
  state.reach = { playerId: state.inspect, frame: index, mask };
  return mask;
}

function renderReach(index) {
  if (!state.layers.has("reach")) return;
  const mask = reachFor(index);
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

function section(id, visible) {
  const node = $(id);
  if (node) node.hidden = !visible;
  return visible;
}

function renderAnalysis(index, slot, threat, fan) {
  const { scene, cache, selection } = state;
  const card = $("analysis-card");
  if (!card) return;

  const player = state.inspect ? scene.byId.get(state.inspect) : null;
  const wantsThreat = Boolean(player) || state.endpoint != null;
  const wantsPass = state.layers.has("passes") && Boolean(fan);
  const wantsSolver = state.layers.has("solver");

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
    const ball = ballRaw(index);
    if (ball) {
      entries.push(["Distance to ball",
        `${NUMBER(Math.hypot(player.x[index] - ball[0], player.y[index] - ball[1]), 1)} m`]);
    }
    const counterpart = nearestCounterpart(player, index);
    if (counterpart) {
      entries.push([`Distance to #${counterpart.player.shirt}`,
        `${NUMBER(counterpart.distance, 1)} m`]);
    }
    const onset = scene.onsets?.[player.id];
    if (onset && player.side === "attack") {
      const delta = (index - onset.index) / scene.fps;
      entries.push(["Run start",
        delta >= 0 ? `${NUMBER(delta, 1)} s ago` : `in ${NUMBER(-delta, 1)} s`]);
    }
    if (player.side === "attack" && cache) {
      entries.push(["Available space",
        `${NUMBER(cache.residual(player.id, slot).value, 1)}`]);
    }
    if (state.layers.has("reach")) {
      const mask = reachFor(index);
      entries.push(["Reachable in 2.0 s",
        mask ? `${Math.round(mask.areaM2)} m²` : "—", !mask]);
    }
    rows($("an-player-rows"), entries);
  }

  // ---- space & threat ----
  const point = state.endpoint != null && fan
    ? fan.rays[state.endpoint].end
    : (player ? [player.x[index], player.y[index]] : null);
  if (section("an-threat", wantsThreat && Boolean(point))) {
    const parts = threatComponents(index, point);
    rows($("an-threat-rows"), parts
      ? [
        ["Where", state.endpoint != null ? "Selected endpoint" : `#${player.shirt}`],
        ["Pitch control", NUMBER(parts.control, 2)],
        ["Ball transition", NUMBER(parts.transition, 3)],
        ["EPV", NUMBER(parts.epv, 3)],
        ["OBSO", NUMBER(parts.obso, 4)],
      ]
      : [["Threat components", "Select OBSO threat to compute", true]]);
  }

  // ---- passing ----
  if (section("an-pass", wantsPass)) {
    const node = $("an-pass-rows");
    if (state.endpoint == null) {
      rows(node, [["Endpoint", "Click one to inspect", true]]);
    } else {
      const ray = fan.rays[state.endpoint];
      const [x0, y0] = fan.origin;
      const distance = Math.hypot(ray.end[0] - x0, ray.end[1] - y0);
      rows(node, [
        ["Carrier", `#${fan.carrier.shirt} ${fan.carrier.name}`],
        ["Direction", `${ray.degrees > 0 ? "+" : ""}${ray.degrees}° from attack`],
        ["Distance", `${NUMBER(distance, 1)} m`],
        ["Ball travel", `${NUMBER(distance / OBSO_BALL_SPEED, 2)} s`],
      ]);
    }
  }

  // ---- solver ----
  if (section("an-solver", wantsSolver)) {
    renderSolverRows();
  }

  card.hidden = !(Boolean(player) || wantsPass || wantsSolver);
  $("analysis-hint").textContent = player ? `#${player.shirt}` : "";
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

function renderSolverRows() {
  const node = $("an-solver-rows");
  const data = solverFor();
  if (!data) { rows(node, [["Solver", "Loading…", true]]); return; }
  if (!data.available) {
    rows(node, [["Solver", data.reason || "Not computed for this scene", true]]);
    return;
  }
  const info = solverSummary(data);
  const attack = info.attack?.kind === "release"
    ? "release the ball"
    : `carrier ${commandName(info.attack?.carrier, data.attack_direction)}, `
      + `receiver ${commandName(info.attack?.receiver, data.attack_direction)}`;
  rows(node, [
    ["Equilibrium value", NUMBER(info.value, 4)],
    ["Certificate gap", info.gap != null ? info.gap.toExponential(1) : "—"],
    ["Root policy", info.mixed ? "mixed" : "pure"],
    ["Attack action", attack],
    ["Attack probability", NUMBER(info.attack?.probability, 3)],
    ["Defender action", commandName(info.defence?.defender, data.attack_direction)],
    ["Defender probability", NUMBER(info.defence?.probability, 3)],
    ["Release probability", NUMBER(info.releaseProbability, 3)],
    ["Horizon", `${info.steps} × ${NUMBER(info.stepSeconds, 1)} s`],
  ]);
  const note = document.createElement("div");
  note.className = "an-provenance";
  const p = info.provenance || {};
  note.textContent = `${p.repository}@${p.commit} · ${p.artifact}#${p.state_index}`;
  node.after(note);
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
    });
  }
  $("wake-mode").addEventListener("change", render);

  // clicking a candidate endpoint selects it for the Analysis panel; clicking
  // the pitch background clears the selection
  $("pitch").addEventListener("click", (event) => {
    const ray = event.target?.getAttribute?.("data-ray");
    if (ray == null) return;
    event.stopPropagation();
    const index = Number(ray);
    state.endpoint = state.endpoint === index ? null : index;
    render();
  });

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
