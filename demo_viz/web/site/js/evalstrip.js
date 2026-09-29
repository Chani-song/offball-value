// The frame-by-frame evaluation strip, under the scrubber.
//
// One lane per series the payload actually carries. It draws nothing it was
// not given: a series with no computed samples is not a flat line at zero, and
// two computed samples are not joined unless the producer said joining them is
// meaningful (`interpolate`). Evaluation is expected to be sparse -- a handful
// of solved frames in a 250-frame clip is a normal output -- and a line drawn
// through the gaps would assert values nobody computed.
//
// Shares the scrubber's x mapping exactly, so the marker here and the playhead
// above are the same instant.

const NS = "http://www.w3.org/2000/svg";
const LANE = 34;          // px per series
const PAD = 8;            // left/right gutter, matched to the scrubber's thumb
const LABEL = 118;        // px reserved for the series name

const el = (name, attrs = {}) => {
  const node = document.createElementNS(NS, name);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, String(v));
  return node;
};

/** Value -> lane y, using the producer's domain when it supplied one. */
function scaleFor(series) {
  const numbers = series.values.filter((v) => typeof v === "number" && isFinite(v));
  let low = series.domain ? series.domain[0] : Math.min(...numbers);
  let high = series.domain ? series.domain[1] : Math.max(...numbers);
  if (!isFinite(low) || !isFinite(high) || high === low) { low = 0; high = 1; }
  return (value) => (value - low) / (high - low);
}

/**
 * Draw the strip.
 *
 * `onSeek(frame)` is called when a reviewer clicks or drags, so the strip is an
 * input as well as a readout and the two clocks cannot drift.
 */
export function drawEvalStrip(node, { series, nFrames, frame, width, onSeek }) {
  node.replaceChildren();
  if (!series.length) return null;

  const w = Math.max(width || node.clientWidth || 640, 320);
  const h = series.length * LANE + 6;
  const plotLeft = LABEL + PAD;
  const plotWidth = Math.max(w - plotLeft - PAD, 40);
  const xOf = (index) => plotLeft + (plotWidth * index) / Math.max(nFrames - 1, 1);
  const frameOf = (x) =>
    Math.round(((x - plotLeft) / plotWidth) * Math.max(nFrames - 1, 1));

  const svg = el("svg", { width: "100%", height: h, viewBox: `0 0 ${w} ${h}`,
                          class: "evalsvg", preserveAspectRatio: "none" });

  series.forEach((line, lane) => {
    const top = lane * LANE + 4;
    const bottom = top + LANE - 12;
    const scale = scaleFor(line);
    const yOf = (value) => bottom - scale(value) * (bottom - top);

    svg.append(el("line", { x1: plotLeft, x2: plotLeft + plotWidth,
                            y1: bottom + 0.5, y2: bottom + 0.5, class: "evalbase" }));

    const label = el("text", { x: PAD, y: (top + bottom) / 2 + 4, class: "evallabel" });
    label.textContent = line.label;
    svg.append(label);

    const points = line.frames
      .map((f, i) => [f, line.values[i]])
      .filter(([, v]) => typeof v === "number" && isFinite(v));

    if (line.interpolate && points.length > 1) {
      svg.append(el("polyline", {
        class: "evalline",
        points: points.map(([f, v]) => `${xOf(f)},${yOf(v)}`).join(" "),
      }));
    } else if (points.length > 1) {
      // sampled, not continuous: a tick per sample down to the baseline, so
      // the gaps read as gaps rather than as a curve
      for (const [f, v] of points) {
        svg.append(el("line", { class: "evalstem", x1: xOf(f), x2: xOf(f),
                                y1: bottom, y2: yOf(v) }));
      }
    }
    for (const [f, v] of points) {
      svg.append(el("circle", { class: "evaldot", cx: xOf(f), cy: yOf(v), r: 3.2 }));
    }
    if (!points.length) {
      const none = el("text", { x: plotLeft, y: (top + bottom) / 2 + 4,
                                class: "evalnone" });
      none.textContent = "no samples";
      svg.append(none);
    }
  });

  const marker = el("line", { class: "evalmark", x1: xOf(frame), x2: xOf(frame),
                              y1: 2, y2: h - 4 });
  svg.append(marker);

  // the nearest computed sample to the playhead, highlighted rather than
  // interpolated -- the panel reports which frame it came from
  const nearest = [];
  series.forEach((line, lane) => {
    if (!line.frames.length) { nearest.push(null); return; }
    let best = line.frames[0], gap = Math.abs(best - frame);
    for (const f of line.frames) {
      if (Math.abs(f - frame) < gap) { best = f; gap = Math.abs(f - frame); }
    }
    const value = line.values[line.frames.indexOf(best)];
    if (typeof value === "number" && isFinite(value)) {
      const scale = scaleFor(line);
      const top = lane * LANE + 4, bottom = top + LANE - 12;
      svg.append(el("circle", { class: "evalnear", cx: xOf(best),
                                cy: bottom - scale(value) * (bottom - top), r: 5.4 }));
    }
    nearest.push({ frame: best, value, exact: best === frame });
  });

  if (onSeek) {
    const seek = (event) => {
      const box = svg.getBoundingClientRect();
      const x = ((event.clientX - box.left) / box.width) * w;
      onSeek(Math.min(Math.max(frameOf(x), 0), nFrames - 1));
    };
    svg.addEventListener("pointerdown", (event) => {
      svg.setPointerCapture(event.pointerId);
      seek(event);
    });
    svg.addEventListener("pointermove", (event) => {
      if (event.buttons) seek(event);
    });
    svg.style.cursor = "crosshair";
  }

  node.append(svg);
  return nearest;
}
