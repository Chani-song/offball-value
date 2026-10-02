// A small inline-SVG time series: observed space against the no-reaction
// baseline, with the difference shaded.

import { P } from "./palette.js";

const NS = "http://www.w3.org/2000/svg";

/**
 * `theme` is the pitch's active theme. The chart is the sidebar's only drawn
 * panel, so it follows the page it sits on: the research palette in the Full
 * explorer, figure_style's attack/defence in the Submission showcase.
 */
export function drawChart(root, { times, factual, counter, now, freezeTime,
                                  theme = null }) {
  const paper = theme && theme !== P;
  const C = paper
    ? { grid: theme.grid, muted: theme.text2, series: theme.attack,
        counter: theme.defend, freeze: theme.muted, now: theme.text }
    : { grid: P.grid, muted: P.muted, series: P.beneficiary,
        counter: P.defender, freeze: P.runner, now: P.text };
  root.replaceChildren();
  if (!times || !times.length) {
    const empty = document.createElement("div");
    empty.className = "chart-empty";
    empty.textContent = "pick a beneficiary";
    root.appendChild(empty);
    return;
  }
  const width = 300;
  const height = 116;
  const pad = { l: 26, r: 8, t: 8, b: 16 };
  const svg = document.createElementNS(NS, "svg");
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.setAttribute("class", "chart-svg");
  svg.setAttribute("preserveAspectRatio", "none");

  const series = counter ? [...factual, ...counter] : [...factual];
  const lo = Math.min(...series);
  const hi = Math.max(...series);
  const span = Math.max(hi - lo, 1e-6);
  const y0 = lo - span * 0.12;
  const y1 = hi + span * 0.12;
  const sx = (t) => pad.l + ((t - times[0]) / Math.max(times[times.length - 1] - times[0], 1e-6))
    * (width - pad.l - pad.r);
  const sy = (v) => height - pad.b - ((v - y0) / (y1 - y0)) * (height - pad.t - pad.b);

  const add = (tag, attrs, text) => {
    const node = document.createElementNS(NS, tag);
    for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
    if (text !== undefined) node.textContent = text;
    svg.appendChild(node);
    return node;
  };

  for (let i = 0; i <= 2; i += 1) {
    const value = y0 + ((y1 - y0) * i) / 2;
    add("line", { x1: pad.l, x2: width - pad.r, y1: sy(value), y2: sy(value),
                  stroke: C.grid, "stroke-width": 0.7 });
    add("text", { x: pad.l - 4, y: sy(value) + 3, "text-anchor": "end",
                  "font-size": 7, fill: C.muted }, value.toFixed(0));
  }

  if (counter) {
    const area = [
      ...factual.map((v, i) => `${i ? "L" : "M"}${sx(times[i])} ${sy(v)}`),
      ...[...counter].reverse().map((v, i) =>
        `L${sx(times[times.length - 1 - i])} ${sy(v)}`),
      "Z",
    ].join(" ");
    add("path", { d: area, fill: C.series, opacity: 0.15, stroke: "none" });
    add("path", {
      d: counter.map((v, i) => `${i ? "L" : "M"}${sx(times[i])} ${sy(v)}`).join(" "),
      fill: "none", stroke: C.counter, "stroke-width": 1.2, "stroke-dasharray": "3 2",
    });
  }
  add("path", {
    d: factual.map((v, i) => `${i ? "L" : "M"}${sx(times[i])} ${sy(v)}`).join(" "),
    fill: "none", stroke: C.series, "stroke-width": 1.7,
    "stroke-linejoin": "round",
  });
  if (freezeTime != null) {
    add("line", { x1: sx(freezeTime), x2: sx(freezeTime), y1: pad.t, y2: height - pad.b,
                  stroke: C.freeze, "stroke-width": 0.8, "stroke-dasharray": "2 2",
                  opacity: 0.8 });
  }
  add("line", { x1: sx(now), x2: sx(now), y1: pad.t, y2: height - pad.b,
                stroke: C.now, "stroke-width": 0.9, opacity: 0.85 });
  add("text", { x: pad.l, y: height - 4, "font-size": 7, fill: C.muted },
      `${times[0].toFixed(0)}s`);
  add("text", { x: width - pad.r, y: height - 4, "text-anchor": "end",
                "font-size": 7, fill: C.muted },
      `${times[times.length - 1].toFixed(0)}s`);
  root.appendChild(svg);
}
