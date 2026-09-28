// The pitch: SVG for markings and players (clickable, draggable), a canvas
// underneath for the opened-space field.
//
// Same visual language as the rendered videos and the Dash app.

import { P, ROLE_COLOUR } from "./palette.js";
import { ballAt, playerAt, view } from "./scene.js";

const SVG_NS = "http://www.w3.org/2000/svg";
const PITCH_L = 105;
const PITCH_W = 68;
const MARGIN = 2;

export class Pitch {
  constructor(root, { onPlayerDown, onBackground } = {}) {
    this.root = root;
    this.onPlayerDown = onPlayerDown;
    this.onBackground = onBackground;

    // The field is rasterised offscreen and then placed *inside* the SVG, so
    // it sits above the pitch markings and below the players.
    this.canvas = document.createElement("canvas");

    this.svg = document.createElementNS(SVG_NS, "svg");
    this.svg.setAttribute("class", "pitch-svg");
    this.svg.setAttribute("viewBox",
      `${-PITCH_L / 2 - MARGIN} ${-PITCH_W / 2 - MARGIN} ${PITCH_L + 2 * MARGIN} ${PITCH_W + 2 * MARGIN}`);
    this.svg.setAttribute("preserveAspectRatio", "xMidYMid meet");
    root.appendChild(this.svg);

    // 1 at full pitch, smaller when cropped; text and strokes divide by the
    // zoom so they keep a constant size on screen while players grow.
    this.k = 1;
    this.layers = {};
    for (const name of ["markings", "field", "reach", "paths", "passes", "trail",
                        "lane", "tether", "ghost", "players", "labels"]) {
      const group = document.createElementNS(SVG_NS, "g");
      group.setAttribute("class", `layer-${name}`);
      this.svg.appendChild(group);
      this.layers[name] = group;
    }
    this._drawMarkings();
    this.svg.addEventListener("pointerdown", (event) => {
      if (event.target.closest("[data-player]")) return;
      if (this.onBackground) this.onBackground(event);
    });
  }

  _drawMarkings() {
    const g = this.layers.markings;
    const halfL = PITCH_L / 2;
    const halfW = PITCH_W / 2;
    const add = (tag, attrs) => {
      const node = document.createElementNS(SVG_NS, tag);
      for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
      g.appendChild(node);
      return node;
    };
    add("rect", { x: -halfL, y: -halfW, width: PITCH_L, height: PITCH_W, fill: P.pitch });
    // mow bands, the same cue the rendered frames use
    const bands = 6;
    for (let i = 0; i < bands; i += 2) {
      add("rect", {
        x: -halfL + (i * PITCH_L) / bands, y: -halfW,
        width: PITCH_L / bands, height: PITCH_W, fill: P.pitchBand, opacity: 0.75,
      });
    }
    const line = { fill: "none", stroke: P.line, "stroke-width": 0.28, opacity: 0.5 };
    add("rect", { x: -halfL, y: -halfW, width: PITCH_L, height: PITCH_W, ...line });
    add("line", { x1: 0, y1: -halfW, x2: 0, y2: halfW, ...line });
    add("circle", { cx: 0, cy: 0, r: 9.15, ...line });
    add("circle", { cx: 0, cy: 0, r: 0.4, fill: P.line, opacity: 0.5, stroke: "none" });
    for (const sign of [-1, 1]) {
      const base = sign * halfL;
      add("rect", {
        x: Math.min(base, base - sign * 16.5), y: -20.16,
        width: 16.5, height: 40.32, ...line,
      });
      add("rect", {
        x: Math.min(base, base - sign * 5.5), y: -9.16,
        width: 5.5, height: 18.32, ...line,
      });
      add("rect", {
        x: Math.min(base, base + sign * 2), y: -3.66,
        width: 2, height: 7.32, ...line, fill: P.ink, opacity: 0.6,
      });
      add("circle", { cx: base - sign * 11, cy: 0, r: 0.35,
                      fill: P.line, opacity: 0.5, stroke: "none" });
      const span = (Math.acos((16.5 - 11) / 9.15) * 180) / Math.PI;
      const cx = base - sign * 11;
      const a0 = sign > 0 ? 180 - span : -span;
      const a1 = sign > 0 ? 180 + span : span;
      const rad = (deg) => (deg * Math.PI) / 180;
      add("path", {
        d: `M ${cx + 9.15 * Math.cos(rad(a0))} ${9.15 * Math.sin(rad(a0))} `
           + `A 9.15 9.15 0 0 ${sign > 0 ? 1 : 1} `
           + `${cx + 9.15 * Math.cos(rad(a1))} ${9.15 * Math.sin(rad(a1))}`,
        ...line,
      });
    }
  }

  /** Crop to a box in view coordinates, or to the whole pitch when null. */
  setView(box) {
    const full = [-PITCH_L / 2 - MARGIN, -PITCH_W / 2 - MARGIN,
                  PITCH_L + 2 * MARGIN, PITCH_W + 2 * MARGIN];
    const [x, y, w, h] = box || full;
    this.svg.setAttribute("viewBox", `${x} ${y} ${w} ${h}`);
    // preserveAspectRatio fits whichever axis binds, so take the looser ratio
    this.k = Math.max(w / full[2], h / full[3]);
  }

  clearDynamic() {
    for (const name of ["field", "reach", "paths", "passes", "trail", "lane",
                        "tether", "ghost", "players", "labels"]) {
      this.layers[name].replaceChildren();
    }
  }

  add(layer, tag, attrs, text) {
    const node = document.createElementNS(SVG_NS, tag);
    for (const [key, value] of Object.entries(attrs)) {
      if (value !== null && value !== undefined) node.setAttribute(key, value);
    }
    if (text !== undefined) node.textContent = text;
    this.layers[layer].appendChild(node);
    return node;
  }

  /** Opened-space field, rasterised to an image inside the SVG. */
  drawField(grid, values, scene, { colour = P.beneficiary, peak = null } = {}) {
    if (!values) return;
    const maximum = peak ?? values.reduce((a, b) => Math.max(a, b), 0);
    if (!(maximum > 1e-9)) return;

    const rgb = hexToRgb(colour);
    this.canvas.width = grid.nx;
    this.canvas.height = grid.ny;
    const context = this.canvas.getContext("2d");
    const image = context.createImageData(grid.nx, grid.ny);
    for (let iy = 0; iy < grid.ny; iy += 1) {
      // image rows run top-down; the pitch grid runs bottom-up, and the
      // left-to-right mirror flips which end of it is on top
      const sourceRow = scene.flip ? iy : grid.ny - 1 - iy;
      for (let ix = 0; ix < grid.nx; ix += 1) {
        const sourceCol = scene.flip ? grid.nx - 1 - ix : ix;
        const value = Math.min(Math.max(values[sourceRow * grid.nx + sourceCol] / maximum, 0), 1);
        const offset = (iy * grid.nx + ix) * 4;
        image.data[offset] = rgb[0];
        image.data[offset + 1] = rgb[1];
        image.data[offset + 2] = rgb[2];
        image.data[offset + 3] = Math.round(220 * value ** 1.15);
      }
    }
    context.putImageData(image, 0, 0);
    this.add("field", "image", {
      href: this.canvas.toDataURL(),
      x: -PITCH_L / 2, y: -PITCH_W / 2, width: PITCH_L, height: PITCH_W,
      preserveAspectRatio: "none", style: "image-rendering:auto",
    });
  }

  /**
   * The reachable set, as a flat wash with an edge.
   *
   * Deliberately not a heat map: the model answers yes or no per cell, so a
   * gradient would imply a confidence the computation does not produce.
   */
  drawReach(grid, values, scene) {
    if (!values) return;
    this.canvas.width = grid.nx;
    this.canvas.height = grid.ny;
    const context = this.canvas.getContext("2d");
    const image = context.createImageData(grid.nx, grid.ny);
    const rgb = hexToRgb(P.reach);
    for (let iy = 0; iy < grid.ny; iy += 1) {
      const sourceRow = scene.flip ? iy : grid.ny - 1 - iy;
      for (let ix = 0; ix < grid.nx; ix += 1) {
        const sourceCol = scene.flip ? grid.nx - 1 - ix : ix;
        const on = values[sourceRow * grid.nx + sourceCol] > 0.5;
        const offset = (iy * grid.nx + ix) * 4;
        image.data[offset] = rgb[0];
        image.data[offset + 1] = rgb[1];
        image.data[offset + 2] = rgb[2];
        image.data[offset + 3] = on ? 46 : 0;
      }
    }
    context.putImageData(image, 0, 0);
    this.add("reach", "image", {
      href: this.canvas.toDataURL(),
      x: -PITCH_L / 2, y: -PITCH_W / 2, width: PITCH_L, height: PITCH_W,
      preserveAspectRatio: "none", style: "image-rendering:pixelated",
    });
  }

  /**
   * A solver rollout: one dashed path per body, with the decision instants
   * marked. Dashed and in the solver colour so it cannot be read as tracking,
   * which is always solid.
   */
  drawSolverTrajectory(scene, trajectory) {
    for (const [body, points] of Object.entries(trajectory.paths || {})) {
      const viewed = points.map(([x, y]) => view(scene, x, y)).filter(Boolean);
      if (viewed.length < 2) continue;
      this.add("passes", "path", {
        d: viewed.map((p, i) => `${i ? "L" : "M"}${p[0]} ${p[1]}`).join(" "),
        fill: "none", stroke: P.solver, "stroke-width": 0.42,
        "stroke-dasharray": "2.2 1.4", "stroke-linecap": "round", opacity: 0.95,
      });
      for (const point of viewed) {
        this.add("passes", "circle", {
          cx: point[0], cy: point[1], r: 0.42,
          fill: P.ink, stroke: P.solver, "stroke-width": 0.2, opacity: 0.95,
        });
      }
      const last = viewed[viewed.length - 1];
      this.add("passes", "text", {
        x: last[0], y: last[1] - 1.4, "text-anchor": "middle",
        fill: P.solver, "font-size": 1.25, opacity: 0.9,
        style: "paint-order:stroke; stroke:#05090A; stroke-width:0.6px",
      }, body);
    }
    if (trajectory.release_target) {
      const target = view(scene, trajectory.release_target[0], trajectory.release_target[1]);
      if (target) {
        this.add("passes", "circle", {
          cx: target[0], cy: target[1], r: 0.9,
          fill: "none", stroke: P.solver, "stroke-width": 0.28, opacity: 0.95,
        });
      }
    }
  }

  /** Players, with the side that can be clicked brought forward. */
  drawPlayers(scene, index, selection, { labels = true, activeSide = null,
                                          hints = [], dragging = null } = {}) {
    const hintSet = new Set(hints);
    for (const player of scene.players) {
      const position = playerAt(scene, player, index);
      if (!position) continue;
      const [x, y] = position;
      const role = selection.roleOf(player.id);
      const colour = role ? ROLE_COLOUR[role] : (player.side === "attack" ? P.attack : P.defend);
      let opacity = role ? 1 : 0.55;
      if (activeSide && !role) opacity = player.side === activeSide ? 0.95 : 0.2;
      const radius = (role ? 1.55 : 1.2) * Math.max(this.k, 0.62);

      const group = this.add("players", "g", {
        "data-player": player.id,
        "data-side": player.side,
        class: `player${role ? " is-role" : ""}${dragging === player.id ? " is-dragging" : ""}`,
        opacity,
      });
      if (activeSide && !role && player.side === activeSide) {
        group.appendChild(circle(x, y, radius + 0.85, {
          fill: "none", stroke: P.text2, "stroke-width": 0.16, opacity: 0.55,
        }));
      }
      if (hintSet.has(player.id)) {
        group.appendChild(circle(x, y, radius + 1.1, {
          fill: "none", stroke: role ? colour : P.defender,
          "stroke-width": 0.26, opacity: 0.75,
        }));
      }
      if (role) {
        group.appendChild(circle(x, y, radius + 1.5, { fill: colour, opacity: 0.16 }));
      }
      group.appendChild(circle(x, y, radius, {
        fill: colour, stroke: role ? colour : P.ink,
        "stroke-width": role ? 0.4 : 0.18,
      }));
      if (labels) {
        const text = this.add("players", "text", {
          x, y: y + 0.45 * this.k, "text-anchor": "middle", class: "shirt",
          "font-size": (role ? 1.6 : 1.35) * Math.max(this.k, 0.62), fill: P.ink,
        }, player.shirt);
        group.appendChild(text);
      }
      // a generous transparent hit area, so clicking is never fiddly
      group.appendChild(circle(x, y, Math.max(radius + 1.2, 2.4), {
        fill: "transparent", class: "hit",
      }));
      group.addEventListener("pointerdown", (event) => {
        if (this.onPlayerDown) this.onPlayerDown(event, player);
      });
    }

    const ball = ballAt(scene, index);
    if (ball) {
      this.add("players", "circle", {
        cx: ball[0], cy: ball[1], r: 0.62, fill: P.ball,
        stroke: P.ink, "stroke-width": 0.22,
      });
    }
  }

  drawTrail(scene, playerId, index, seconds = 1.8) {
    const player = scene.byId.get(playerId);
    if (!player) return;
    const span = Math.max(2, Math.round(seconds * scene.fps));
    const points = [];
    for (let i = Math.max(0, index - span); i <= index; i += 1) {
      const position = playerAt(scene, player, i);
      if (position) points.push(position);
    }
    if (points.length < 2) return;
    const d = points.map(([x, y], i) => `${i ? "L" : "M"}${x} ${y}`).join(" ");
    this.add("trail", "path", { d, fill: "none", stroke: P.runner,
                                "stroke-width": 1.4, opacity: 0.16,
                                "stroke-linecap": "round" });
    this.add("trail", "path", { d, fill: "none", stroke: P.runner,
                                "stroke-width": 0.5, opacity: 0.9,
                                "stroke-linecap": "round" });
  }

  drawTether(scene, runnerId, defenderIds, index, { labels = true } = {}) {
    const runner = scene.byId.get(runnerId);
    if (!runner) return;
    const from = playerAt(scene, runner, index);
    if (!from) return;
    for (const defenderId of defenderIds) {
      const defender = scene.byId.get(defenderId);
      if (!defender) continue;
      const to = playerAt(scene, defender, index);
      if (!to) continue;
      this.add("tether", "line", {
        x1: from[0], y1: from[1], x2: to[0], y2: to[1],
        stroke: P.defender, "stroke-width": 0.32, "stroke-dasharray": "1.4 1.1",
        opacity: 0.85,
      });
      if (labels) {
        const distance = Math.hypot(to[0] - from[0], to[1] - from[1]);
        this.add("labels", "text", {
          x: (from[0] + to[0]) / 2, y: (from[1] + to[1]) / 2 - 0.8 * this.k,
          "text-anchor": "middle", "font-size": 1.5 * this.k, fill: P.defender,
          class: "pitch-label",
        }, `${distance.toFixed(0)} m`);
      }
    }
  }

  drawGhost(scene, cache, defenderIds, index, freezeIndex, { labels = true } = {}) {
    for (const defenderId of defenderIds) {
      const defender = scene.byId.get(defenderId);
      if (!defender || index <= freezeIndex) continue;
      const held = view(scene, defender.x[freezeIndex], defender.y[freezeIndex]);
      const now = playerAt(scene, defender, index);
      if (!held || !now) continue;
      this.add("ghost", "line", {
        x1: held[0], y1: held[1], x2: now[0], y2: now[1],
        stroke: P.defender, "stroke-width": 0.2, "stroke-dasharray": "0.6 0.9",
        opacity: 0.6,
      });
      this.add("ghost", "circle", {
        cx: held[0], cy: held[1], r: 1.55, fill: P.defender, opacity: 0.14,
        stroke: P.defender, "stroke-width": 0.28, "stroke-dasharray": "0.9 0.8",
      });
      if (labels) {
        const pulled = Math.hypot(now[0] - held[0], now[1] - held[1]);
        this.add("labels", "text", {
          x: held[0], y: held[1] - 2.6 * this.k, "text-anchor": "middle",
          "font-size": 1.4 * this.k, fill: P.defender, class: "pitch-label",
        }, `held · ${pulled.toFixed(0)} m`);
      }
    }
  }

  drawLane(scene, beneficiaryIds, index, maximum = 34) {
    const ball = ballAt(scene, index);
    if (!ball) return;
    for (const beneficiaryId of beneficiaryIds) {
      const player = scene.byId.get(beneficiaryId);
      if (!player) continue;
      const to = playerAt(scene, player, index);
      if (!to) continue;
      if (Math.hypot(to[0] - ball[0], to[1] - ball[1]) > maximum) continue;
      this.add("lane", "line", {
        x1: ball[0], y1: ball[1], x2: to[0], y2: to[1],
        stroke: P.beneficiary, "stroke-width": 1.2, opacity: 0.22,
        "stroke-linecap": "round",
      });
      this.add("lane", "line", {
        x1: ball[0], y1: ball[1], x2: to[0], y2: to[1],
        stroke: P.beneficiary, "stroke-width": 0.32, opacity: 0.85,
      });
    }
  }

  /**
   * The five candidate pass directions from the carrier.
   *
   * Drawn identically: no ranking, no thickness or colour encoding a score,
   * no arrowhead singled out. The only per-ray text is the OBSO value at the
   * endpoint when the threat view is on, and it is deliberately small.
   */
  drawCandidates(scene, fan, { values = null, selected = null } = {}) {
    if (!fan) return;
    const origin = view(scene, fan.origin[0], fan.origin[1]);
    if (!origin) return;
    // two rays clipped to the same stretch of touchline end in the same place;
    // labelling both just prints one number on top of another
    const labelled = [];
    for (const ray of fan.rays) {
      const end = view(scene, ray.end[0], ray.end[1]);
      if (!end) continue;
      this.add("passes", "line", {
        x1: origin[0], y1: origin[1], x2: end[0], y2: end[1],
        stroke: P.obso, "stroke-width": 1.1, opacity: 0.16,
        "stroke-linecap": "round",
      });
      this.add("passes", "line", {
        x1: origin[0], y1: origin[1], x2: end[0], y2: end[1],
        stroke: P.obso, "stroke-width": 0.26, opacity: 0.8,
        "stroke-dasharray": "1.6 1.1", "stroke-linecap": "round",
      });
      const chosen = selected === fan.rays.indexOf(ray);
      this.add("passes", "circle", {
        cx: end[0], cy: end[1], r: chosen ? 0.95 : 0.62,
        fill: chosen ? P.obso : "none", "fill-opacity": chosen ? 0.35 : 0,
        stroke: P.obso, "stroke-width": chosen ? 0.34 : 0.22, opacity: 0.95,
        "data-ray": String(fan.rays.indexOf(ray)),
        style: "cursor:pointer", "pointer-events": "all",
      });
      if (values && ray.obso != null) {
        // flip the caption below the endpoint near the touchline, so it stays
        // on the pitch rather than floating over the card
        const nearTop = end[1] < -PITCH_W / 2 + 3.2;
        const at = [end[0], end[1] + (nearTop ? 2.5 : -1.6)];
        // crowding is judged on where the text lands, not where the ray ends:
        // the flip above can push two captions together even when their
        // endpoints are well apart. The box is the size of "0.000" set at 1.35.
        const clash = labelled.some(
          ([lx, ly]) => Math.abs(at[0] - lx) < 5.2 && Math.abs(at[1] - ly) < 2.0);
        if (!clash) {
          labelled.push(at);
          this.add("passes", "text", {
            x: at[0], y: at[1], "text-anchor": "middle",
            fill: P.obso, "font-size": 1.35, opacity: 0.9,
            style: "paint-order:stroke; stroke:#05090A; stroke-width:0.6px",
          }, ray.obso.toFixed(3));
        }
      }
    }
    this.add("passes", "circle", {
      cx: origin[0], cy: origin[1], r: 1.15,
      fill: "none", stroke: P.obso, "stroke-width": 0.3, opacity: 0.75,
    });
  }

  drawPaths(scene) {
    for (const player of scene.players) {
      if (player.gk) continue;
      const points = [];
      for (let i = 0; i < scene.n_frames; i += 3) {
        const position = playerAt(scene, player, i);
        if (position) points.push(position);
      }
      if (points.length < 2) continue;
      this.add("paths", "path", {
        d: points.map(([x, y], i) => `${i ? "L" : "M"}${x} ${y}`).join(" "),
        fill: "none", stroke: P.muted, "stroke-width": 0.14, opacity: 0.3,
      });
    }
  }
}

function circle(cx, cy, r, attrs) {
  const node = document.createElementNS(SVG_NS, "circle");
  node.setAttribute("cx", cx);
  node.setAttribute("cy", cy);
  node.setAttribute("r", r);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
  return node;
}

function hexToRgb(hex) {
  const value = hex.replace("#", "");
  return [
    parseInt(value.slice(0, 2), 16),
    parseInt(value.slice(2, 4), 16),
    parseInt(value.slice(4, 6), 16),
  ];
}
