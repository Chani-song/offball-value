// Wiring: scene loading, the role dock, drag and drop, overlays, playback.

import { InfluenceCache } from "./influence.js";
import { P, ROLE_COLOUR, ROLE_SIDE } from "./palette.js";
import { Pitch } from "./pitch.js";
import { autoTriplet, rankBeneficiaries, rankDefenders } from "./ranking.js";
import { drawChart } from "./chart.js";
import { Selection } from "./selection.js";
import { ballAt, loadIndex, loadScene, onsetFor, playerAt } from "./scene.js";

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
  layers: new Set(["trail", "tether", "wake", "lane", "ghost", "labels", "hints"]),
  drag: null,
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
  if (wanted.fit != null) {
    const on = wanted.fit === "1";
    if (on) state.layers.add("fit"); else state.layers.delete("fit");
    const box = document.querySelector('[data-layer="fit"]');
    if (box) box.checked = on;
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
  $("status").textContent = "loading…";
  const started = performance.now();
  const scene = await loadScene(file);
  state.scene = scene;
  state.cache = new InfluenceCache(scene, { every: 5 });
  state.selection = Selection.fromRoles(scene.roles, "annotation");
  const onset = onsetFor(scene, state.selection.runner);
  state.frame = peakGainFrame() ?? Math.min(onset.index + Math.round(1.5 * scene.fps),
                                            scene.n_frames - 1);
  if (wanted) applyUrlState(wanted);
  $("time").max = String(scene.n_frames - 1);
  $("time").value = String(state.frame);
  $("scene").value = file;
  $("title").textContent = scene.title;
  $("subtitle").textContent = scene.subtitle;
  $("notes").textContent = scene.notes || "—";
  $("team-attack").textContent = scene.attacking_team;
  $("team-defend").textContent = scene.defending_team;
  render();
  $("status").textContent = `${Math.round(performance.now() - started)} ms`;
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
  const activeSide = selection.armed ? ROLE_SIDE[selection.pick] : null;

  pitch.setView(state.layers.has("fit") ? fitBox() : null);
  pitch.clearDynamic();

  let hints = [];
  if (selection.runner && !selection.defenders.length) {
    hints = rankDefenders(scene, selection.runner, freeze).slice(0, 4).map((r) => r.id);
  } else if (selection.runner && selection.defenders.length && !selection.beneficiaries.length) {
    hints = rankBeneficiaries(cache, scene, selection.runner, selection.defenders,
                              freeze, slot).filter((r) => r.gain > 0.05).slice(0, 3)
      .map((r) => r.id);
  }

  if (state.layers.has("wake") && selection.beneficiaries.length) {
    const factual = cache.combined(selection.beneficiaries, slot);
    let values = factual.field;
    if ($("wake-mode").value === "gain" && swap.length) {
      const counter = cache.combined(selection.beneficiaries, slot, swap);
      values = factual.field.map((v, i) => Math.max(v - counter.field[i], 0));
    }
    pitch.drawField(cache.grid, values, scene);
  } else {
    pitch.drawField(cache.grid, null, scene);
  }

  if (state.layers.has("paths")) pitch.drawPaths(scene);
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
    hints: state.layers.has("hints") ? hints : [],
    dragging: state.drag ? state.drag.playerId : null,
  });

  renderDock();
  renderStat(slot, swap);
  renderChart(swap, freeze);
  renderCandidates(freeze, slot);
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

function renderStat(slot, swap) {
  const { cache, selection } = state;
  const node = $("stat-value");
  if (!selection.beneficiaries.length) {
    $("stat-label").textContent = "Opened space";
    node.textContent = "—";
    node.style.color = P.muted;
    return;
  }
  const factual = cache.combined(selection.beneficiaries, slot).value;
  if (!swap.length) {
    $("stat-label").textContent = "Space held";
    node.textContent = factual.toFixed(1);
    node.style.color = P.beneficiary;
    return;
  }
  const counter = cache.combined(selection.beneficiaries, slot, swap).value;
  const gain = factual - counter;
  $("stat-label").textContent = "Opened space";
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
  list.replaceChildren();
  if (!selection.runner) {
    $("candidates-label").textContent = "Hints";
    list.innerHTML = '<div class="muted">Pick a runner first.</div>';
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
    list.innerHTML = '<div class="muted">Pick a defender first.</div>';
    return;
  } else {
    rows = rankDefenders(scene, selection.runner, freeze).slice(0, 8);
    role = "defender";
    $("candidates-label").textContent = "Reacts most";
  }
  for (const row of rows) {
    const item = document.createElement("div");
    item.className = "cand";
    if (selection.roleOf(row.id)) item.classList.add("is-picked");
    item.innerHTML = `<span class="cand-shirt">#${row.shirt}</span>`
      + `<span class="cand-name"></span>`
      + `<span class="cand-value">${row.caption}</span>`;
    item.querySelector(".cand-name").textContent = row.name;
    item.addEventListener("click", () => {
      state.selection.arm(role).applyClick(scene, row.id);
      render();
    });
    list.appendChild(item);
  }
}

// ---------------------------------------------------------------------------
// role dock
// ---------------------------------------------------------------------------
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
    line.textContent =
      `run starts ${scene.times[onset.index] >= 0 ? "+" : ""}`
      + `${scene.times[onset.index].toFixed(1)}s · ${onset.method}`;
    meta.appendChild(line);
  }
  if (selection.source !== "manual") {
    const line = document.createElement("div");
    line.textContent = selection.source;
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
    state.drag.moved = true;
    ghost.style.display = "block";
    ghost.style.left = `${moveEvent.clientX}px`;
    ghost.style.top = `${moveEvent.clientY}px`;
    const slot = slotUnder(moveEvent.clientX, moveEvent.clientY);
    for (const node of document.querySelectorAll(".slot")) {
      node.classList.toggle("is-over", node === slot);
    }
  };

  const up = (upEvent) => {
    document.removeEventListener("pointermove", move);
    document.removeEventListener("pointerup", up);
    ghost.remove();
    document.body.classList.remove("dragging");
    for (const node of document.querySelectorAll(".slot")) {
      node.classList.remove("is-over", "is-droppable");
    }
    const slot = state.drag.moved ? slotUnder(upEvent.clientX, upEvent.clientY) : null;
    const dragged = state.drag;
    state.drag = null;

    if (slot) {
      const role = slot.dataset.role;
      if (ROLE_SIDE[role] === dragged.side) {
        // dragging one attack role onto the other swaps them, which is the
        // gesture people reach for first
        if (dragged.fromRole && dragged.fromRole !== role
            && ROLE_SIDE[dragged.fromRole] === ROLE_SIDE[role]) {
          state.selection.swapAttackRoles();
        } else {
          state.selection.assign(state.scene, role, dragged.playerId);
        }
      }
    } else if (!dragged.moved) {
      if (dragged.fromRole) state.selection.arm(dragged.fromRole);
      else state.selection.applyClick(state.scene, dragged.playerId);
    }
    render();
  };

  document.addEventListener("pointermove", move);
  document.addEventListener("pointerup", up);
}

function markDropTargets(side) {
  for (const node of document.querySelectorAll(".slot")) {
    node.classList.toggle("is-droppable", ROLE_SIDE[node.dataset.role] === side);
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
function bindControls() {
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
  $("btn-swap").addEventListener("click", () => {
    state.selection.swapAttackRoles();
    render();
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
