let allShots = [];
let visibleShots = [];
let currentIndex = 0;
let currentEffect = "";

const video = document.getElementById("video");
const timeline = document.getElementById("timeline");
const currentTimeEl = document.getElementById("currentTime");
const durationEl = document.getElementById("duration");
const progressEl = document.getElementById("progress");
const overlayEl = document.getElementById("overlay");
const statusEl = document.getElementById("saveStatus");
const trackingSvg = document.getElementById("trackingOverlay");
const trackingStatusEl = document.getElementById("trackingStatus");
let trackingData = null;
let trackingRequestToken = 0;
let trackingRAF = null;

class NumberEditor {
  constructor(rootId, overlayClass, overlayLabel) {
    this.root = document.getElementById(rootId);
    this.values = [];
    this.overlayClass = overlayClass;
    this.overlayLabel = overlayLabel;
    this.render();
  }
  set(values) {
    if (typeof values === "string") values = values.split(/[,;\s]+/).filter(Boolean);
    this.values = [...new Set((values || []).map(v => String(v).replace(/^#/, "").trim()).filter(Boolean))];
    this.render();
    updateOverlay();
  }
  get() {
    // The number currently typed in the primary box is live immediately.
    // "+ Add" is only needed when the user wants to enter another player.
    const input = this.root.querySelector("input");
    const live = input ? String(input.value || "").replace(/^#/, "").trim() : "";
    const out = [...this.values];
    if (/^\d{1,3}$/.test(live)) {
      const n = String(parseInt(live, 10));
      if (!out.includes(n)) out.push(n);
    }
    return out;
  }
  add(v) {
    v = String(v || "").replace(/^#/, "").trim();
    if (!/^\d{1,3}$/.test(v)) return;
    v = String(parseInt(v, 10));
    if (!this.values.includes(v)) this.values.push(v);
    this.render();
    updateOverlay();
  }
  remove(v) {
    this.values = this.values.filter(x => x !== v);
    this.render();
    updateOverlay();
  }
  render() {
    this.root.innerHTML = "";
    this.values.forEach(v => {
      const chip = document.createElement("span");
      chip.className = "chip";
      chip.innerHTML = `<span>#${v}</span>`;
      const remove = document.createElement("button");
      remove.type = "button";
      remove.textContent = "×";
      remove.addEventListener("click", () => this.remove(v));
      chip.appendChild(remove);
      this.root.appendChild(chip);
    });
    const input = document.createElement("input");
    input.type = "number";
    input.min = "0";
    input.max = "999";
    input.inputMode = "numeric";
    input.placeholder = "#";
    const add = document.createElement("button");
    add.type = "button";
    add.className = "add-number";
    add.textContent = "+ Add";
    const commit = () => { this.add(input.value); input.value = ""; input.focus(); };
    add.addEventListener("click", commit);
    input.addEventListener("input", () => updateOverlay());
    input.addEventListener("keydown", e => { if (e.key === "Enter") { e.preventDefault(); commit(); } });
    this.root.appendChild(input);
    this.root.appendChild(add);
  }
}

const offballEditor = new NumberEditor("offballInput", "overlay-offball", "OTB");
const defenderEditor = new NumberEditor("defenderInput", "overlay-defender", "DEF");
const beneficiaryEditor = new NumberEditor("beneficiaryInput", "overlay-benefit", "SPACE");

function currentShot() { return visibleShots[currentIndex]; }
function esc(s) { return String(s ?? "").replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}[c])); }
function fmt(t) { return Number.isFinite(t) ? t.toFixed(1) : "0.0"; }

function updateOverlay() {
  // Keep the video clean: moving circles identify players; a fixed legend explains colors.
  const chips = [];
  if (currentEffect) chips.push(`<span class="overlay-chip overlay-effect">Effect: ${esc(currentEffect.toUpperCase())}</span>`);
  overlayEl.innerHTML = chips.join("");
  renderTrackingRings();
}


function dataToCanvas(x, y) {
  // Exact Matplotlib axes geometry used by make_shot_reels.py (1200x800 render).
  const xMin = -0.025, xMax = 1.025;
  const yMin = -0.025, yMax = 1.025;
  const axLeft = 150.0;
  const axRight = 1080.0;
  const axBottomFromBottom = 94.8571428571;
  const axTopFromBottom = 697.1428571429;
  const px = axLeft + ((x - xMin) / (xMax - xMin)) * (axRight - axLeft);
  const pyFromBottom = axBottomFromBottom + ((y - yMin) / (yMax - yMin)) * (axTopFromBottom - axBottomFromBottom);
  return [px, 800.0 - pyFromBottom];
}

function playerAt(side, jersey, rawFrame) {
  if (!trackingData?.frames?.length) return null;
  const maxIdx = trackingData.frames.length - 1;
  const i0 = Math.max(0, Math.min(Math.floor(rawFrame), maxIdx));
  const i1 = Math.max(0, Math.min(i0 + 1, maxIdx));
  const f = Math.max(0, Math.min(rawFrame - i0, 1));
  const p0 = trackingData.frames[i0]?.[side]?.[String(jersey)] || null;
  const p1 = trackingData.frames[i1]?.[side]?.[String(jersey)] || null;
  if (p0 && p1) return [p0[0] + (p1[0] - p0[0]) * f, p0[1] + (p1[1] - p0[1]) * f];
  return p0 || p1;
}

function ringMarkup(side, jersey, role, radius, rawFrame) {
  const p = playerAt(side, jersey, rawFrame);
  if (!p) return "";
  const [cx, cy] = dataToCanvas(p[0], p[1]);
  return `<circle class="track-ring track-ring-${role}" cx="${cx.toFixed(1)}" cy="${cy.toFixed(1)}" r="${radius}" />`;
}

function renderTrackingRings() {
  if (!trackingSvg) return;
  if (!trackingData?.frames?.length || !Number.isFinite(video.currentTime)) {
    trackingSvg.innerHTML = "";
    return;
  }
  const rawFrame = video.currentTime * Number(trackingData.fps || 5);
  const parts = [];
  offballEditor.get().forEach(n => parts.push(ringMarkup("a", n, "offball", 23, rawFrame)));
  defenderEditor.get().forEach(n => parts.push(ringMarkup("d", n, "defender", 23, rawFrame)));
  beneficiaryEditor.get().forEach(n => parts.push(ringMarkup("a", n, "benefit", 29, rawFrame)));
  trackingSvg.innerHTML = parts.join("");
}

async function loadTrackingForShot(shot) {
  const token = ++trackingRequestToken;
  trackingData = null;
  trackingSvg.innerHTML = "";
  trackingStatusEl.textContent = "Tracking rings: loading…";
  try {
    const r = await fetch(`/api/tracking/${encodeURIComponent(shot.clip_id)}`);
    const out = await r.json();
    if (token !== trackingRequestToken) return;
    if (!r.ok || !out.ok) throw new Error(out.error || "tracking load failed");
    trackingData = out;
    trackingStatusEl.textContent = `Tracking rings: ready · ${out.attacking_team} vs ${out.defending_team}`;
    renderTrackingRings();
  } catch (e) {
    if (token !== trackingRequestToken) return;
    trackingData = null;
    trackingSvg.innerHTML = "";
    trackingStatusEl.textContent = `Tracking rings unavailable: ${e.message}`;
  }
}

function startTrackingAnimation() {
  if (trackingRAF) cancelAnimationFrame(trackingRAF);
  const tick = () => {
    renderTrackingRings();
    if (!video.paused && !video.ended) trackingRAF = requestAnimationFrame(tick);
    else trackingRAF = null;
  };
  trackingRAF = requestAnimationFrame(tick);
}

function setEffect(effect) {
  currentEffect = effect || "";
  document.querySelectorAll("[data-effect]").forEach(b => b.classList.toggle("selected", b.dataset.effect === currentEffect));
  updateOverlay();
}

function loadShot(index) {
  if (!visibleShots.length) {
    document.getElementById("shotTitle").textContent = "No clips found";
    return;
  }
  currentIndex = Math.max(0, Math.min(index, visibleShots.length - 1));
  const s = currentShot();
  const a = s.annotation || {};
  document.getElementById("shotTitle").textContent = `${s.match_id} · Shot ${s.shot_number || '?'} · ${s.match_clock || ''}`;
  document.getElementById("shotSub").textContent = `${s.team || ''} · ${s.shooter || ''} · ${s.shot_result || ''}`;
  video.src = `/video/${encodeURIComponent(s.clip_id)}`;
  video.load();
  loadTrackingForShot(s);
  offballEditor.set(a.offball_attackers || "");
  defenderEditor.set(a.drawn_defenders || "");
  beneficiaryEditor.set(a.space_beneficiaries || "");
  setEffect(a.effect || "");
  document.getElementById("notes").value = a.notes || "";
  statusEl.textContent = s.reviewed ? "Previously saved — editing will update the same Excel row." : "";
  statusEl.className = "save-status";
  updateProgress();
  window.scrollTo({top: 0, behavior: "smooth"});
}

function updateProgress() {
  const reviewed = allShots.filter(s => s.reviewed).length;
  progressEl.textContent = `${reviewed}/${allShots.length} reviewed · showing ${visibleShots.length} · current ${visibleShots.length ? currentIndex + 1 : 0}/${visibleShots.length}`;
}

function applyFilter(keepClipId = null) {
  const unreviewedOnly = document.getElementById("unreviewedOnly").checked;
  visibleShots = allShots.filter(s => !unreviewedOnly || !s.reviewed);
  let idx = 0;
  if (keepClipId) {
    const found = visibleShots.findIndex(s => s.clip_id === keepClipId);
    if (found >= 0) idx = found;
  }
  currentIndex = Math.min(idx, Math.max(visibleShots.length - 1, 0));
  loadShot(currentIndex);
}

async function save(goNext) {
  const s = currentShot();
  if (!s) return;
  const payload = {
    clip_id: s.clip_id,
    offball_attackers: offballEditor.get(),
    drawn_defenders: defenderEditor.get(),
    space_beneficiaries: beneficiaryEditor.get(),
    effect: currentEffect,
    notes: document.getElementById("notes").value,
  };
  statusEl.textContent = "Saving…";
  statusEl.className = "save-status";
  try {
    const r = await fetch("/api/save", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(payload)});
    const out = await r.json();
    if (!r.ok || !out.ok) throw new Error(out.error || "Save failed");
    s.reviewed = true;
    s.annotation = out.row;
    const original = allShots.find(x => x.clip_id === s.clip_id);
    if (original) { original.reviewed = true; original.annotation = out.row; }
    statusEl.textContent = "Saved to shot_annotations.xlsx";
    statusEl.className = "save-status ok";

    if (goNext) {
      const unreviewedOnly = document.getElementById("unreviewedOnly").checked;
      if (unreviewedOnly) {
        const oldId = s.clip_id;
        visibleShots = allShots.filter(x => !x.reviewed);
        currentIndex = Math.min(currentIndex, Math.max(visibleShots.length - 1, 0));
        loadShot(currentIndex);
      } else {
        loadShot(Math.min(currentIndex + 1, visibleShots.length - 1));
      }
    } else {
      updateProgress();
    }
  } catch (e) {
    statusEl.textContent = e.message;
    statusEl.className = "save-status err";
  }
}

function jump(seconds) {
  video.currentTime = Math.max(0, Math.min(video.duration || 0, video.currentTime + seconds));
}

video.addEventListener("loadedmetadata", () => {
  durationEl.textContent = fmt(video.duration);
  timeline.value = 0;
});
video.addEventListener("timeupdate", () => {
  currentTimeEl.textContent = fmt(video.currentTime);
  if (Number.isFinite(video.duration) && video.duration > 0) timeline.value = Math.round((video.currentTime / video.duration) * 1000);
  document.getElementById("playPause").textContent = video.paused ? "Play" : "Pause";
  renderTrackingRings();
});
video.addEventListener("play", () => {
  document.getElementById("playPause").textContent = "Pause";
  startTrackingAnimation();
});
video.addEventListener("pause", () => {
  document.getElementById("playPause").textContent = "Play";
  renderTrackingRings();
});
video.addEventListener("seeked", renderTrackingRings);
timeline.addEventListener("input", () => { if (video.duration) video.currentTime = (Number(timeline.value) / 1000) * video.duration; });
document.getElementById("playPause").addEventListener("click", () => video.paused ? video.play() : video.pause());
document.querySelectorAll("[data-jump]").forEach(b => b.addEventListener("click", () => jump(Number(b.dataset.jump))));
document.querySelectorAll("[data-speed]").forEach(b => b.addEventListener("click", () => {
  video.playbackRate = Number(b.dataset.speed);
  document.querySelectorAll("[data-speed]").forEach(x => x.classList.toggle("active-speed", x === b));
}));
document.querySelectorAll("[data-effect]").forEach(b => b.addEventListener("click", () => setEffect(b.dataset.effect)));
document.getElementById("prevShot").addEventListener("click", () => loadShot(currentIndex - 1));
document.getElementById("nextShot").addEventListener("click", () => loadShot(currentIndex + 1));
document.getElementById("saveOnly").addEventListener("click", () => save(false));
document.getElementById("saveNext").addEventListener("click", () => save(true));
document.getElementById("unreviewedOnly").addEventListener("change", () => applyFilter(currentShot()?.clip_id));

document.addEventListener("keydown", e => {
  const tag = document.activeElement?.tagName;
  if (["INPUT", "TEXTAREA"].includes(tag)) {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") { e.preventDefault(); save(true); }
    return;
  }
  if (e.code === "Space") { e.preventDefault(); video.paused ? video.play() : video.pause(); }
  else if (e.key === "ArrowLeft") { e.preventDefault(); jump(e.shiftKey ? -5 : -1); }
  else if (e.key === "ArrowRight") { e.preventDefault(); jump(e.shiftKey ? 5 : 1); }
  else if (e.key === "1") setEffect("strong");
  else if (e.key === "2") setEffect("medium");
  else if (e.key === "3") setEffect("low");
  else if (e.key === "4") setEffect("ignore");
  else if ((e.ctrlKey || e.metaKey) && e.key === "Enter") { e.preventDefault(); save(true); }
});

async function init() {
  const r = await fetch("/api/shots");
  const data = await r.json();
  allShots = data.shots || [];
  visibleShots = [...allShots];
  loadShot(0);
}
init().catch(e => {
  progressEl.textContent = `Failed to load: ${e.message}`;
});
