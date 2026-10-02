#!/usr/bin/env python3
"""The defender's move field at Figure 2's three moments (2026-10-01, the team's feedback on
S05_05_defender_flow_moves.png): S05 at 0.0 / 0.6 / 1.2 s, each moment re-solved with the defender's start moved
over a 1 m lattice (the *_1m_full.json grids behind Figure 2's value background), everything else real. A line
follows the probability-weighted 0.6 s displacement of his equilibrium first moves (each command's end on the
solver's own path, minus the start) -- the "moves" field of render_defender_flow.py, which stays as it was.

The team's four points:
  1. Figure 1 / Figure 2's colours and marks (scripts/figure_style.py: pure blue / pure red, no outline on the
     three, the others in full team colour, white ball); the lines in the defender's red;
  2. each panel is Figure 2's own window at that moment (render_figure2_abstract's layout: 180 mm, one scale, the
     same windows) and the lines fill it -- the lattice covers every window, so the field is interpolated between
     computed starts everywhere in it (checked below: no mesh point of a window lies outside them);
  3. thinner lines (LW_MIN..LW_MAX pt; width = the average move's length, ONE scale for the three panels) and no
     circles at the starts where the equilibrium mixes -- the line there is still an average of different moves
     (render_defender_fields.py draws the most likely move instead);
  4. the 0.6 s and 1.2 s moments beside 0.0 s.

Picture choices (mine): streamline spacing SPACING m (streamplot's density from it, so the same in metres in every
panel), heads of streamplot arrowsize ARROW, a pitch-coloured margin of MOVE_GAP pt (Figure 1's gap between a mark
and a move) round every player and the ball so the lines stop short of the marks, Figure 2 v4's panel frames.
Interpolation: linear (scipy griddata, as render_defender_flow) on a MESH m mesh, then a Gaussian of --smooth m
(default 0.6 m, Figure 2's value background's own; without it a line along a boundary between two moves zigzags
along the 1 m lattice's steps); the starts with no game (within 1 m of an attacker) and the one uncertified start
(0.0 s) are left out and filled from their neighbours.

Usage:
    python scripts/render_defender_flow_moments.py \\
        --panels data/processed/showcase_v1/figure_multi/panels --label S05-full \\
        --tracking data/processed/showcase_v1/tracking/S05.csv \\
        --grids 0.0=<dir>/grid_S05_1m_full.json 0.6=<dir>/grid_S05@0.6_1m_full.json 1.2=<dir>/grid_S05@1.2_1m_full.json \\
        --output out/showcase_v1/figure_multi/S05_defender_flow.png
    (writes the .png at 600 dpi, a vector .pdf and a .json of what was drawn beside it)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figure_style as st  # noqa: E402
import render_figure2_abstract as f2  # noqa: E402  (Figure 2's scenes, windows and scale)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.interpolate import griddata  # noqa: E402
from scipy.ndimage import gaussian_filter  # noqa: E402
from matplotlib.patches import FancyArrowPatch  # noqa: E402

MM = 1 / 25.4
MESH = 0.25                     # m, the interpolation mesh
SPACING = 0.8                   # m, streamplot's cell: about the closest two lines come
LW_MIN, LW_MAX = 0.35, 1.1      # pt, a line where the average move is ~0 .. the longest of the three panels
ARROW = 0.9                     # streamplot arrowsize: an open chevron ~3 pt long and wide (2026-10-01; was 0.65 for
                                # the filled head) -- smaller than Figure 1's on these thin lines
COLOR = st.DEFENCE
FRAME, FRAME_LW = "#8C8C8C", 0.8    # Figure 2 v4's panel frames


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--panels", type=Path, required=True, help="Figure 2's panel files")
    p.add_argument("--label", required=True, help="their prefix: <label>_<dt>.json")
    p.add_argument("--tracking", type=Path, required=True)
    p.add_argument("--grids", nargs="+", required=True, metavar="DT=GRID", help="a defender grid per moment")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--title", default="How the defender moves from each starting spot")
    p.add_argument("--key", default="defender's equilibrium move in the next 0.6 s",
                   help="the one-line key next to the line sample (the caption says the rest)")
    p.add_argument("--width-mm", type=float, default=180.0)
    p.add_argument("--smooth", type=float, default=f2.BG_SMOOTH,
                   help="m, Gaussian sigma on the interpolated field (default: Figure 2's value background's, so the "
                        "lines do not follow the 1 m lattice's steps); 0 = none")
    return p.parse_args()


def moves(grid: dict):
    """Per solved start: (x, y) and the probability-weighted 0.6 s displacement of his first moves."""
    xy, d = [], []
    for p in grid["points"]:
        if p["status"] != "solved":
            continue
        x, y = p["defender_start"]
        xy.append((x, y))
        d.append((sum(c["prob"] * (c["end"][0] - x) for c in p["defender"]),
                  sum(c["prob"] * (c["end"][1] - y) for c in p["defender"])))
    return np.array(xy), np.array(d)


def same_game(grid: dict, s: dict, dt: float):
    """The grid's observed start is Figure 2's panel: the three where Figure 2 draws them."""
    o = next(p for p in grid["points"] if p["observed"])
    for role, where in (("defender", o["defender_start"]), ("runner", o["runner_start"]),
                        ("ball carrier", o["carrier_start"])):
        if math.dist(where, s["key"][role]) > 1e-6:
            raise SystemExit(f"{dt:.1f} s: the grid's {role} {where} is not Figure 2's {s['key'][role]}")
    if o["start_frame"] != s["frame"]:
        raise SystemExit(f"{dt:.1f} s: grid frame {o['start_frame']} vs Figure 2's {s['frame']}")
    return o


def field(xy, d, xlim, ylim):
    nx = int(round((xlim[1] - xlim[0]) / MESH)) + 1
    ny = int(round((ylim[1] - ylim[0]) / MESH)) + 1
    xs, ys = np.linspace(*xlim, nx), np.linspace(*ylim, ny)
    X, Y = np.meshgrid(xs, ys)
    U = griddata(xy, d[:, 0], (X, Y), method="linear")
    V = griddata(xy, d[:, 1], (X, Y), method="linear")
    return xs, ys, U, V


def window_field(grid, xlim, ylim, smooth, dt):
    """The field on a mesh over one window: linear between the computed starts, then the Gaussian. Raises if any
    mesh point is outside the starts. Returns xs, ys, U, V and the starts' (xy, d)."""
    xy, d = moves(grid)
    xs, ys, U, V = field(xy, d, xlim, ylim)
    if not (np.isfinite(U).all() and np.isfinite(V).all()):
        bad = ~(np.isfinite(U) & np.isfinite(V))
        raise SystemExit(f"{dt:.1f} s: {bad.sum()} of {bad.size} mesh points outside the computed starts")
    if smooth > 0:                                # sigma in mesh steps along each axis
        sig = (smooth / (ys[1] - ys[0]), smooth / (xs[1] - xs[0]))
        U, V = gaussian_filter(U, sig, mode="nearest"), gaussian_filter(V, sig, mode="nearest")
    return xs, ys, U, V, xy, d


def streamlines(ax, xs, ys, U, V, top, color=COLOR, alpha=1.0, zorder=4):
    """The streamlines of one window: width = the average move's length on the scale `top` (m), SPACING apart.
    With alpha < 1 (an overlay, render_figure2_abstract --flow-grids) the line segments get butt caps, so their
    joints do not overlap into darker dots, and the heads the same alpha."""
    lw = LW_MIN + (LW_MAX - LW_MIN) * np.clip(np.hypot(U, V) / top, 0.0, 1.0)
    density = ((xs[-1] - xs[0]) / SPACING / 30, (ys[-1] - ys[0]) / SPACING / 30)
    before = set(ax.patches)
    sp = ax.streamplot(xs, ys, U, V, color=color, linewidth=lw, density=density, arrowstyle=st.HEAD,
                       arrowsize=ARROW, minlength=0.1, zorder=zorder)
    if alpha < 1.0:
        sp.lines.set_alpha(alpha); sp.lines.set_capstyle("butt")
        for head in (q for q in ax.patches if q not in before):
            head.set_alpha(alpha)
    return sp, density


def margin_mark(ax, xy, marker, ms):
    """A pitch-coloured margin round a mark: over the lines, under every mark (a neighbour's margin never cuts
    into a mark, e.g. the ball's into the ball carrier's disc)."""
    ax.plot(*xy, marker, ms=ms + 2 * st.MOVE_GAP, color=st.PITCH, mew=0, zorder=5.5)


def players(ax, s):
    for x, y, side in s["others"]:
        margin_mark(ax, (x, y), "o", st.OTHER_D)
        st.other_player(ax, (x, y), side, z=6)
    for role, xy in s["key"].items():
        margin_mark(ax, xy, *st.ROLE_MARKER[role])
        st.key_player(ax, xy, role, z=7)
    if s["ball"]:
        margin_mark(ax, s["ball"], "o", st.BALL_D)
        st.ball(ax, s["ball"], z=7.5)


def names(ax, s, renderer, mpp):
    """The three named once, as in Figure 2's first panel: next to the marker, away from the other two."""
    lab = f2.Placer(ax, renderer, mpp)
    for x, y, _ in s["others"]:
        lab.disc((x, y), (st.OTHER_D / 2 + st.MOVE_GAP) * mpp)
    if s["ball"]:
        lab.disc(s["ball"], (st.BALL_D / 2 + st.MOVE_GAP) * mpp)
    for role, xy in s["key"].items():
        lab.disc(xy, (st.marker_radius(role) + st.MOVE_GAP) * mpp)
    for role, (x, y) in s["key"].items():
        rest = [v for r, v in s["key"].items() if r != role]
        away = f2.unit(x - sum(v[0] for v in rest) / len(rest), y - sum(v[1] for v in rest) / len(rest))
        lab.add(-2.0, (x, y), away, st.ROLE_NAME[role],
                dict(fontsize=st.FS_LABEL, color=st.ROLE_COLOR[role], zorder=9, path_effects=st.halo()), None)
    lab.place()


def main() -> None:
    args = parse_args()
    fonts = st.use_font()
    panels = sorted((float(p.stem.split("_")[-1]), json.loads(p.read_text()))
                    for p in args.panels.glob(f"{args.label}_*.json"))
    track = defaultdict(dict)
    for r in csv.DictReader(args.tracking.open(encoding="utf-8")):
        track[int(r["frame_id"])][r["object_id"]] = r
    scenes = [(dt, f2.scene(p, track)) for dt, p in panels]
    grids = {float(k): json.loads(Path(v).read_text()) for k, v in (a.split("=", 1) for a in args.grids)}
    if sorted(grids) != [dt for dt, _ in scenes]:
        raise SystemExit(f"grids for {sorted(grids)}, Figure 2 panels at {[dt for dt, _ in scenes]}")

    # Figure 2's layout (render_figure2_abstract.main, --layout fit): one scale, each panel centred on what Figure 2
    # draws there and as wide as that needs, all as tall as the tallest
    boxes = [f2.extent(s) for _, s in scenes]
    widths = [x1 - x0 + 2 * f2.PAD for x0, x1, _, _ in boxes]
    span_y = max(y1 - y0 for _, _, y0, y1 in boxes) + 2 * f2.PAD
    W = args.width_mm * MM
    margin, gap, title_in, head_in, foot_in = 0.05, 0.08, 0.30, 0.20, 0.22   # foot: Figure 2's strip
    n = len(scenes)
    m_per_in = sum(widths) / (W - 2 * margin - (n - 1) * gap)
    m_per_pt = m_per_in / 72
    ph = span_y / m_per_in
    H = title_in + head_in + ph + foot_in
    fig = plt.figure(figsize=(W, H), facecolor=st.PAGE)
    renderer = fig.canvas.get_renderer()

    # the fields first: one width scale for the three panels
    work = []
    for (dt, s), (x0, x1, y0, y1), span_x in zip(scenes, boxes, widths):
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        xlim, ylim = (cx - span_x / 2, cx + span_x / 2), (cy - span_y / 2, cy + span_y / 2)
        o = same_game(grids[dt], s, dt)
        xs, ys, U, V, xy, d = window_field(grids[dt], xlim, ylim, args.smooth, dt)
        work.append((dt, s, xlim, ylim, o, xy, d, xs, ys, U, V))
    top = max(float(np.hypot(*w[6].T).max()) for w in work)        # the longest average move, any start
    dump = {"width_mm": args.width_mm, "metres_per_inch": m_per_in, "longest_average_move_m": top,
            "lw_pt": [LW_MIN, LW_MAX], "spacing_m": SPACING, "mesh_m": MESH, "smooth_m": args.smooth,
            "arrowsize": ARROW, "panels": []}

    left = margin
    for k, ((dt, s, xlim, ylim, o, xy, d, xs, ys, U, V), span_x) in enumerate(zip(work, widths)):
        pw = span_x / m_per_in
        ax = fig.add_axes((left / W, foot_in / H, pw / W, ph / H))
        st.pitch(ax)
        ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect("equal")
        sp, density = streamlines(ax, xs, ys, U, V, top)
        players(ax, s)
        if k == 0:
            names(ax, s, renderer, m_per_pt)
        for sp_ in ax.spines.values():          # over the marks: a player at the edge never cuts the frame
            sp_.set_visible(True); sp_.set_color(FRAME); sp_.set_linewidth(FRAME_LW); sp_.set_zorder(8.9)
        fig.text(left / W, (foot_in + ph + 0.05) / H, f"{dt:.1f} s", color=st.INK, fontsize=st.FS_PANEL,
                 fontweight="bold", ha="left", va="bottom")
        od = next(dd for (px, py), dd in zip(xy, d) if math.dist((px, py), o["defender_start"]) < 1e-9)
        dump["panels"].append({
            "dt": dt, "frame": s["frame"], "xlim": list(xlim), "ylim": list(ylim), "starts_used": len(xy),
            "mesh": [len(xs), len(ys)], "mesh_covered": 1.0, "density": list(density),
            "longest_average_move_m": float(np.hypot(*d.T).max()),
            "streamline_segments": len(sp.lines.get_segments()),
            "real_start": {"xy": o["defender_start"], "average_move_m": list(map(float, od)),
                           "probabilities": {c["id"]: c["prob"] for c in o["defender"]}}})
        left += pw + gap

    y_title = (H - title_in / 2) / H
    fig.text(margin / W, y_title, args.title, color=st.INK, fontsize=st.FS_TITLE, fontweight="bold",
             ha="left", va="center")
    arrow_in = 0.28                                 # the attack's direction, as Figure 2
    fig.text((W - margin - arrow_in - 0.05) / W, y_title, "attack", color=st.MUTED, fontsize=st.FS_SMALL,
             ha="right", va="center")
    st.fig_arrow(fig, ((W - margin - arrow_in) / W, y_title), ((W - margin) / W, y_title), color=st.MUTED,
                 lw=st.MOVE_LW * 0.7)
    # the key: one short line bottom left, the match line bottom right, on one row -- as Figure 2's value key
    # (2026-10-01, the user: the two-line key was too long and sat on top of the match line; the details went to
    # the caption, scripts/add_caption.py)
    y_key = 0.07
    kax = fig.add_axes((margin / W, (y_key - 0.035) / H, 0.30 / W, 0.07 / H)); kax.axis("off")
    kax.set_xlim(0, 0.30); kax.set_ylim(-0.035, 0.035)
    lw_key = (LW_MIN + LW_MAX) / 2                  # a streamline: its head halfway along, as streamplot draws it
    kax.plot([0.0, 0.28], [0.0, 0.0], color=COLOR, lw=lw_key, solid_capstyle="butt")
    kax.add_patch(FancyArrowPatch((0.13, 0.0), (0.15, 0.0), arrowstyle=st.HEAD, mutation_scale=10 * ARROW,
                                  lw=lw_key, color=COLOR, shrinkA=0, shrinkB=0))
    key = fig.text((margin + 0.36) / W, y_key / H, args.key, fontsize=st.FS_SMALL - 0.5, color=st.MUTED,
                   ha="left", va="center")
    note = st.match_line(scenes[0][1]["code"], scenes[0][1]["frame"])
    st.match_note(fig, note, W - margin, y_key)
    fig.canvas.draw()                               # the two must leave clear ground between them
    match = next(t for t in fig.texts if t.get_text() == note)
    room = (match.get_window_extent().x0 - key.get_window_extent().x1) / fig.dpi
    if room < 0.5:
        raise SystemExit(f"key and match line only {room:.2f} in apart")
    dump["key"] = {"text": args.key, "clear_of_match_line_in": room}
    st.check_text(fig)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=600, facecolor=fig.get_facecolor())
    fig.savefig(args.output.with_suffix(".pdf"), facecolor=fig.get_facecolor())
    args.output.with_suffix(".json").write_text(json.dumps(dump, indent=1))
    print(f"font: {fonts}")
    print(f"→ {args.output} (+ .pdf, .json) · {W / MM:.0f} x {H / MM:.0f} mm · {m_per_in:.2f} m per inch · "
          f"longest average move {top:.2f} m")
    for p in dump["panels"]:
        print(f"  {p['dt']:.1f} s: {p['starts_used']} starts, window {p['xlim'][1] - p['xlim'][0]:.1f} x "
              f"{p['ylim'][1] - p['ylim'][0]:.1f} m covered, {p['streamline_segments']} segments, real start "
              f"{p['real_start']['probabilities']}")


if __name__ == "__main__":
    main()
