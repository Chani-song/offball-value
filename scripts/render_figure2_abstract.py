#!/usr/bin/env python3
"""Figure 2 of the abstract: the equilibrium choices at three real moments of one off-ball play (S05).

The same panels as `render_panel_figure.py --paths` (out/showcase_v1/figure_multi/S05_full_paths.png),
redrawn for print at 180 mm, in the print style shared with Figure 1 (scripts/figure_style.py, 2026-09-30:
Nimbus Sans, white ground, attack blue / defence vermillion, one marker per role, thin arrows with small
heads, the match line under the figure; the earlier dark version is in scripts/archive_v1_dark/). Everything drawn is read the way that script reads it -- its own helpers
(who_is, real_path, played_passes, MIN_P) are imported -- so the data and the probability aggregation
are unchanged; only the drawing is:

  - each panel is the game solved from that REAL moment (0.0 / 0.6 / 1.2 s), one metres-per-inch
    scale for all three, attack to the right, each panel centred on what it draws and (--layout fit,
    the default) only as wide as that needs -- the 0.6 s pass target sets the widest; fitting the
    other two instead of giving them its width makes the scale ~20% larger at the same 180 mm;
  - the three of the game are Figure 1's markers -- runner a blue diamond, ball carrier a blue disc,
    defender a vermillion square, outlined in charcoal -- named once in the first panel, their paths in
    their team's colour; everyone else a smaller disc in a light tint of his team's colour;
  - an option = the solver's own 0.6 s path under that command (the panel file's `path`, unchanged),
    a solid line in the player's colour, as wide as its equilibrium probability on ONE width scale
    for every panel (lw_of), a small head at its end; a stop (command 0) ends in a bar, no head
    (the two stops drawn, the defender's at 0.0 and 1.2 s, reach speed 0 inside the 0.6 s);
  - a pass = dashed charcoal line from the ball to its target; the receiver's run onto it = a solid
    line in his colour from where he is to the same target (the straight connector the old figure drew,
    not a 0.6 s path). A pass column is ONE joint attack option -- the passer's release and the
    receiver's run are the same probability (extract_panel_policy: release = pass column; the receiver's
    "runs onto the pass" = the same column), so it is labelled once where the two meet,
    "Pass + receive · NN%";
  - 3v1 games (2026-09-30, the other showcase scenes): the game's beneficiary is not on the ball there, so he
    is drawn as the "Teammate" (a blue triangle); the scripted passer -- the attacker standing on the game's
    ball, found as render_panel_figure finds him -- gets the ball carrier's disc and name, no options (his
    only decision is the pass); none while the ball rolls between passers (S36 at 0.6 / 1.2 s); the passer
    and the ball are kept in the panel, so a long pass makes a tall figure (S15, S36);
  - a stop (command 0) that has not reached speed 0 by 0.6 s (S20, S36) is labelled "Slow down NN%" at its
    path's end, like a move; one at rest is "Stop NN%" next to the player, as in S05;
  - the ball carrier's options say "Dribble NN%", a stop "Stop NN%", the others their % only; every
    option drawn with the old figure's MIN_P filter (2%), small ones included; labels placed by Placer
    (next to what they name, never on a line, a player, another label or leader);
  - the real next 0.6 s of the three (tracking): thin grey dotted line to his marker, hollow;
  - removed: shirt numbers and speeds, velocity arrows, the faint last-second trails, the three rows
    of words under each panel, the panel frames, and (2026-09-29, the user's call) the legend -- the
    caption explains the marks; --legend puts a one-row key back.

Probabilities are the equilibrium probabilities of choosing an option (the defender's mix; for the two
attackers the joint attack policy summed over the other attacker), not pass-completion chances.

--dump writes every drawn coordinate, number and label box as JSON, for checking against the panel files,
the solver's policy npz and the tracking.

Usage:
    python scripts/render_figure2_abstract.py \\
        --panels data/processed/showcase_v1/figure_multi/panels \\
        --label S05-full --tracking data/processed/showcase_v1/tracking/S05.csv \\
        --output out/showcase_v1/figure_multi/S05_figure2_abstract.png
    (writes the .png at 600 dpi and a vector .pdf beside it, the face embedded)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figure_style as st  # noqa: E402  (the print style shared with Figure 1)
import render_panel_figure as src  # noqa: E402  (the original figure: its readers and filter)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import numpy as np  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

MM = 1 / 25.4
COLOR = {role: st.ROLE_COLOR[role] for role in ("runner", "ball carrier", "teammate", "defender")}
ROLE_NAME = {role: st.ROLE_NAME[role] for role in COLOR}
PAD = 1.4                                        # m of grass around what a panel draws
FS_PCT, FS_ROLE, FS_PANEL, FS_TITLE, FS_KEY = st.FS_NOTE, st.FS_LABEL, st.FS_PANEL, st.FS_TITLE, st.FS_SMALL
REAL_LW = st.REAL_LW
LEADER = 0.6                                     # m: a label farther than this from its anchor gets a leader
BOX_PAD = 0.15                                   # label box padding, in font sizes (--legend key only)


def lw_of(prob: float) -> float:
    """Line width (pt) of an option played with this probability -- one scale for every panel."""
    return 0.6 + 2.4 * prob


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--panels", type=Path, required=True)
    p.add_argument("--label", required=True, help="file prefix: <label>_<dt>.json")
    p.add_argument("--tracking", type=Path, required=True, help="showcase tracking CSV (attack to the right)")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--title", default="Equilibrium choices during an off-ball play")
    p.add_argument("--width-mm", type=float, default=180.0)
    p.add_argument("--legend", action="store_true",
                   help="draw the one-row key under the panels (off by default since 2026-09-29: the caption "
                        "explains the marks)")
    p.add_argument("--layout", choices=("fit", "equal"), default="fit",
                   help="fit: each panel as wide as it needs, one scale (default); equal: three equal panels")
    p.add_argument("--no-match-line", action="store_true", help="leave out the date / teams / clock line")
    p.add_argument("--scenes", type=Path, default=None,
                   help="scenes.csv the match line reads (default figure_style.SCENES, showcase_v1)")
    p.add_argument("--dump", type=Path, default=None, help="write every drawn coordinate / number as JSON")
    p.add_argument("--value-scale", choices=("panel", "common"), default="panel",
                   help="background shade range: each panel's own min..max (default, 2026-10-01: the common one "
                        "left the early moments nearly blank) or one range for all panels")
    p.add_argument("--value-dark", choices=("defender", "attack"), default="defender",
                   help="which side the background's dark end favours (default defender: dark = a better start for "
                        "him; attack: the first version, lighter = better for the defender)")
    p.add_argument("--frames", action="store_true", help="a thin frame round each moment's panel")
    p.add_argument("--value-fade", type=float, default=0.0,
                   help="m over which the background fades out inside its lattice's edge (0: none -- a lattice "
                        "covering the whole panel needs none, and a fade reads as 'better for the defender')")
    p.add_argument("--value-grids", nargs="*", default=[], metavar="DT=GRID",
                   help="background: the game's value with the defender starting at each spot, from a defender-grid "
                        "file (scripts/extract_defender_grid.py / merge_defender_grids.py) per moment, e.g. "
                        "0.0=grid_S05_1m.json; one shade scale for every panel (2026-10-01)")
    p.add_argument("--flow-grids", nargs="*", default=[], metavar="DT=GRID",
                   help="overlay: the defender's move field from a defender-grid file per moment, drawn as "
                        "render_defender_flow_moments.py draws it (same mesh, smoothing, widths, spacing), under "
                        "every player and option (2026-10-01, the user's call)")
    p.add_argument("--defender-names", choices=("none", "long", "short"), default="none",
                   help="label the defender's moves with the solver's own name for that command in that state "
                        "(Toward goal / Toward ball / Toward runner), not the %% alone (2026-10-01: a short move, e.g. "
                        "0.6 s toward ball 43%% = 0.57 m, hides under his marker and its %% floats unexplained); "
                        "short: To goal / To ball / To runner")
    p.add_argument("--label-gap", type=float, default=0.0,
                   help="m of clear ground every label keeps from the labels placed before it (0: may touch, the "
                        "first version; 1.2 s 'Stop 22%%' and '47%%' then read as one phrase)")
    p.add_argument("--value-key", default="better start for the defender",
                   help="the background key's words after 'darker:' / 'lighter:'")
    p.add_argument("--min-arrow", type=float, default=0.0,
                   help="pt: a move showing less than this outside its player's marker is stretched until it shows "
                        "this much (shape kept, length not to scale); 0 = every path to scale")
    p.add_argument("--no-leader", nargs="*", default=[], metavar="DT|TEXT",
                   help="that label goes only where it needs no leader line (within LEADER of what it names)")
    p.add_argument("--two-line", nargs="*", default=[], metavar="DT|TEXT",
                   help="that label on two lines, its last word (the %%) under the rest")
    p.add_argument("--pass-under-moves", action="store_true",
                   help="draw the ball's line under the players' moves (a dribble crossing it stays in front)")
    p.add_argument("--straight", nargs="*", default=[], metavar="DT:ROLE:COMMAND",
                   help="draw that move as a straight arrow along its --exit-angle (its path's curl dropped)")
    p.add_argument("--stretch-to", nargs="*", default=[], metavar="DT:ROLE:COMMAND=PT",
                   help="as --min-arrow for one move, with its own length, e.g. 0.0:runner:3=20")
    p.add_argument("--exit-angle", nargs="*", default=[], metavar="DT:ROLE:COMMAND=DEG",
                   help="turn that move about its player so it leaves his marker at DEG (0 toward goal, 90 up, -90 "
                        "down), e.g. 1.2:defender:3=-90; ROLE with _ for spaces (ball_carrier)")
    p.add_argument("--label-at", nargs="*", default=[], metavar="DT|TEXT=X,Y",
                   help="pin the label with exactly this text in that moment's panel, centred at (X, Y) pitch metres "
                        "(the others are placed around it), e.g. \"0.0|Defender=30.25,-7.55\"")
    p.add_argument("--attack-size", type=float, default=FS_KEY,
                   help="the 'attack' word, pt; its arrow scales with it")
    p.add_argument("--head-size", type=float, default=FS_PANEL,
                   help="the moments' labels (0.0 s ...), pt (2026-10-01: 12 = Figure 1's panel titles)")
    p.add_argument("--value-key-size", type=float, default=FS_KEY - 0.5,
                   help="its font size, pt (2026-10-01: 7 pt read small once the figure was scaled to an A4 page)")
    p.add_argument("--flow-color", default=st.DEFENCE, help="its colour")
    p.add_argument("--flow-alpha", type=float, default=0.2, help="its opacity (0.2 = 80%% transparent)")
    return p.parse_args()


def scene(p: dict, track: dict) -> dict:
    """What one panel draws, in pitch metres (attack to the right) -- read exactly as the old figure does."""
    sign = p["attack_direction"]
    norm = lambda c: (sign * (c[0] - 52.5), sign * (c[1] - 34.0))
    f = p["start_frame"]
    fr = track[f]
    step = round(0.6 * src.FPS)
    three = p.get("kind") == "3v1"
    # 3v1: the ball is with a scripted passer, so the game's "beneficiary" is not the ball carrier -- drawn as
    # the "teammate"; the passer (the attacker on the ball, as the old figure finds him) is the ball carrier
    shown_as = {"beneficiary": "teammate"} if three else {}
    roles = {shown_as.get(b["role"], b["role"]): b for b in p["bodies"].values()}
    ids = {role: src.who_is(fr, norm(b["pos"])) for role, b in roles.items()}
    if None in ids.values():
        raise SystemExit(f"{p['code']}: a player of the game is not on the tracking frame {f}")
    passer = src.who_is(fr, norm(p["ball"])) if three else None      # none while the ball rolls
    if passer in ids.values():
        passer = None
    key, side, options, real = {}, {}, [], {}
    for role, b in roles.items():
        key[role] = norm(b["pos"])
        side[role] = fr[ids[role]]["side"]
        for c, o in enumerate(b["options"]):
            if o["prob"] < src.MIN_P:
                continue
            if not o.get("path"):
                raise SystemExit(f"{p['code']}: option {o['name']!r} has no solver path")
            options.append({"role": role, "command": c, "name": o["name"], "prob": o["prob"],
                            "stop": c == 0,      # command 0 is the stop in every command set
                            "rests": math.hypot(*o["end_velocity"]) == 0.0,   # at speed 0 by the 0.6 s
                            "path": [norm(pt) for pt in o["path"]]})
        real[role] = src.real_path(track, f, ids[role], range(f, f + step + 1))
    passes = []
    for q in src.played_passes(p):
        to = shown_as.get(q["to"], q["to"])
        receiver = roles.get(to)
        passes.append({"to": to, "prob": q["prob"], "where": q.get("where"), "ball": norm(p["ball"]),
                       "target": norm(q["target"]),
                       "run_from": norm(receiver["pos"]) if receiver is not None else None})
    others = [(float(r["x"]), float(r["y"]), r["side"]) for oid, r in fr.items()
              if oid != "ball" and oid not in ids.values() and oid != passer]
    ball = (float(fr["ball"]["x"]), float(fr["ball"]["y"])) if "ball" in fr else None
    return {"code": p["code"], "frame": f, "key": key, "side": side, "options": options, "real": real,
            "passes": passes, "others": others, "ball": ball, "three": three,
            "passer": (float(fr[passer]["x"]), float(fr[passer]["y"])) if passer else None}


def extent(s: dict):
    pts = list(s["key"].values()) + [pt for o in s["options"] for pt in o["path"]]
    pts += [pt for r in s["real"].values() for pt in r]
    pts += [q["target"] for q in s["passes"]] + [q["ball"] for q in s["passes"]]
    if s["three"]:                  # 3v1: the ball is away from the three, keep it (and its passer) in view
        pts += [pt for pt in (s["passer"], s["ball"]) if pt]
    xs, ys = [x for x, _ in pts], [y for _, y in pts]
    return min(xs), max(xs), min(ys), max(ys)


class Placer:
    """Labels next to what they name. Each goes to the free spot nearest its anchor (a path's end, a player),
    preferring the side the path points to (a cost growing with the angle off it); free = covers no player,
    no line, no arrowhead, no other label or leader, and stays inside the panel. A label farther than LEADER
    from its anchor gets a thin leader line, and each bit of leader over something drawn costs extra.
    Placement is greedy in some order; ORDERS orders are tried (the most likely first, then seeded shuffles,
    role names always last) and the one with the least total cost is drawn."""

    STEP, REACH, ORDERS = 0.15, 5.0, 60
    dt = ""                                            # this panel's moment, "0.0" (set in draw_panel)
    pins = {}                                          # label text -> centre (m), this panel's --label-at
    GAP = 0.0                                          # m round a placed label that later labels keep clear of

    def __init__(self, ax, renderer, m_per_pt):
        self.ax, self.r, self.mpp = ax, renderer, m_per_pt
        (self.x0, self.x1), (self.y0, self.y1) = ax.get_xlim(), ax.get_ylim()
        self.solid, self.todo = [], []
        k = np.arange(-round(self.REACH / self.STEP), round(self.REACH / self.STEP) + 1) * self.STEP
        self.di, self.dj = (a.ravel() for a in np.meshgrid(k, k))

    def disc(self, c, r):
        self.solid.append((c[0] - r, c[1] - r, c[0] + r, c[1] + r))

    @staticmethod
    def _samples(pts, half):
        out = []
        for a, b in zip(pts[:-1], pts[1:]):
            n = max(1, int(math.dist(a, b) / 0.12))
            for k in range(n + 1):
                x, y = a[0] + (b[0] - a[0]) * k / n, a[1] + (b[1] - a[1]) * k / n
                out.append((x - half, y - half, x + half, y + half))
        return out

    def line(self, pts, lw_pt):
        self.solid += self._samples(pts, max(lw_pt / 2 * self.mpp, 0.08) + 0.05)

    def head(self, tip, lw_pt):
        """An arrowhead is wider than its line: keep labels off it."""
        self.disc(tip, st.head_width(lw_pt) / 2 * self.mpp + 0.05)

    def add(self, rank, anchor, direction, text, style, leader):
        key = text                                     # --label-at / --no-leader / --two-line name a label by its text
        if (self.dt, key) in TWO_LINE:                 # "To goal 31%" -> "To goal" over "31%"
            text = "\n".join(text.rsplit(" ", 1))
        t = self.ax.text(*anchor, text, ha="center", va="center", **style)
        ext = t.get_window_extent(self.r)
        pad = 2 * BOX_PAD * style["fontsize"] if "bbox" in style else 0.0
        px_to_m = self.mpp * 72 / self.ax.figure.dpi
        w, h = ext.width * px_to_m + pad * self.mpp + 0.12, ext.height * px_to_m + pad * self.mpp + 0.08
        self.todo.append(dict(rank=rank, anchor=anchor, direction=direction, w=w, h=h, artist=t, leader=leader,
                              key=key, near=(self.dt, key) in NO_LEADER))

    @staticmethod
    def _free(x0, y0, x1, y1, walls):
        if not len(walls):
            return np.ones(np.shape(x0), bool)
        return ~np.any((x0[:, None] < walls[None, :, 2]) & (walls[None, :, 0] < x1[:, None]) &
                       (y0[:, None] < walls[None, :, 3]) & (walls[None, :, 1] < y1[:, None]), axis=1)

    @staticmethod
    def _end(anchor, box):
        return min(max(anchor[0], box[0]), box[2]), min(max(anchor[1], box[1]), box[3])

    def _search(self, item, walls, labels):
        """(cost, centre, box, gap) of the best free spot for one label, or None. A leader may cross a line
        (at a cost) but never another label or leader."""
        (ax_, ay_), (dx, dy), w, h = item["anchor"], item["direction"], item["w"], item["h"]
        cx, cy = ax_ + self.di, ay_ + self.dj
        x0, x1, y0, y1 = cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2
        inside = (x0 >= self.x0 + 0.1) & (x1 <= self.x1 - 0.1) & (y0 >= self.y0 + 0.1) & (y1 <= self.y1 - 0.1)
        gap = np.hypot(np.maximum.reduce([x0 - ax_, np.zeros_like(x0), ax_ - x1]),
                       np.maximum.reduce([y0 - ay_, np.zeros_like(y0), ay_ - y1]))
        if item.get("near"):                           # --no-leader: only spots close enough to need no leader
            inside = inside & (gap <= LEADER)
        off = np.hypot(self.di, self.dj)
        cos = np.where(off > 0, (self.di * dx + self.dj * dy) / np.where(off > 0, off, 1.0), 1.0)
        cost = gap + 0.5 * (1.0 - cos)                     # 0 straight ahead, +1 m right behind
        idx = np.flatnonzero(inside)
        idx = idx[self._free(x0[idx], y0[idx], x1[idx], y1[idx], walls)]
        best = None
        for k in idx[np.argsort(cost[idx], kind="stable")]:
            c = cost[k]
            if best is not None and c >= best[0]:
                break
            box = (x0[k], y0[k], x1[k], y1[k])
            if gap[k] > LEADER:          # each 0.15 m of leader over something drawn costs 0.4 m
                ex, ey = self._end((ax_, ay_), box)
                n = max(1, int(gap[k] / 0.15))
                px = np.array([ax_ + (ex - ax_) * s / n for s in range(2, n + 1)])
                py = np.array([ay_ + (ey - ay_) * s / n for s in range(2, n + 1)])
                if len(px):
                    if not self._free(px - 0.03, py - 0.03, px + 0.03, py + 0.03, labels).all():
                        continue
                    c += 0.4 * np.count_nonzero(~self._free(px - 0.03, py - 0.03, px + 0.03, py + 0.03, walls))
            if best is None or c < best[0]:
                best = (c, (cx[k], cy[k]), box, gap[k])
        return best

    def _run(self, order):
        walls, labels, total, out = list(self.solid), [], 0.0, []
        for item in order:
            res = self._search(item, np.array(walls, dtype=float).reshape(-1, 4),
                               np.array(labels, dtype=float).reshape(-1, 4))
            if res is None:
                return math.inf, None
            total += res[0]
            out.append((item, res))
            g = self.GAP
            walls.append((res[2][0] - g, res[2][1] - g, res[2][2] + g, res[2][3] + g))
            labels.append(res[2])
            if res[3] > LEADER and item["leader"]:     # a leader: nothing may sit on it, no leader may cross it
                lead = self._samples([item["anchor"], self._end(item["anchor"], res[2])], 0.08)
                walls += lead
                labels += lead[2:]                     # (its first samples sit on the anchor, shared by no one)
        return total, out

    def place(self):
        for item in [t for t in self.todo if t["key"] in self.pins]:
            # a label pinned by --label-at: drawn where asked, a fixed obstacle for the others
            cx, cy = self.pins[item["key"]]
            box = (cx - item["w"] / 2, cy - item["h"] / 2, cx + item["w"] / 2, cy + item["h"] / 2)
            item["artist"].set_position((cx, cy))
            g = self.GAP
            self.solid.append((box[0] - g, box[1] - g, box[2] + g, box[3] + g))
            ex, ey = self._end(item["anchor"], box)
            if item["leader"] and not item["near"] and math.dist(item["anchor"], (ex, ey)) > LEADER:
                self.ax.plot([item["anchor"][0], ex], [item["anchor"][1], ey], color=item["leader"], lw=0.5,
                             alpha=0.9, zorder=8, solid_capstyle="butt")
            self.todo.remove(item)
        base = sorted(self.todo, key=lambda t: -t["rank"])
        names = [t for t in base if t["leader"] is None]            # role names: always after the numbers
        movable = [t for t in base if t["leader"] is not None]
        rng = random.Random(0)
        orders = [base] + [rng.sample(movable, len(movable)) + names for _ in range(self.ORDERS)]
        total, out = min((self._run(o) for o in orders), key=lambda r: r[0])
        if out is None:
            walls, stuck = np.array(self.solid, dtype=float).reshape(-1, 4), []
            for item in base:                          # which label finds no spot even on its own
                if self._search(item, walls, np.zeros((0, 4))) is None:
                    stuck.append(item["key"])
            raise SystemExit(f"no room for every label ({self.dt} s; no spot at all for: {stuck or 'none alone'})")
        for item, (_, centre, box, gap) in out:
            item["artist"].set_position(centre)
            if gap > LEADER and item["leader"]:
                ex, ey = self._end(item["anchor"], box)
                self.ax.plot([item["anchor"][0], ex], [item["anchor"][1], ey], color=item["leader"], lw=0.5,
                             alpha=0.9, zorder=8, solid_capstyle="butt")


def pct_style(color, fs=FS_PCT):
    """An option's number: Regular, in the option's colour, a thin ground-coloured halo, no box."""
    return dict(fontsize=fs, color=color, zorder=9, path_effects=st.halo())


def unit(dx, dy):
    n = math.hypot(dx, dy)
    return (dx / n, dy / n) if n > 1e-9 else (0.0, 0.0)


def leave(pts, centre, reach):
    """The path from where it first gets clear of a mark at `centre` (render_figure1_dilemma.leave): `reach(u)`
    metres along the unit direction u from the centre. A path starting clear is kept; only its start is cut."""
    def clear(p):
        r = math.dist(p, centre)
        return r > 1e-9 and r >= reach(((p[0] - centre[0]) / r, (p[1] - centre[1]) / r))
    pts = [tuple(p) for p in pts]
    if clear(pts[0]):
        return pts
    i = 0
    while i < len(pts) - 1 and not clear(pts[i + 1]):
        i += 1
    if i == len(pts) - 1:
        return pts[-1:]
    (ax_, ay), (bx, by) = pts[i], pts[i + 1]
    lo, hi = 0.0, 1.0
    for _ in range(40):
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if clear((ax_ + (bx - ax_) * mid, ay + (by - ay) * mid)) else (mid, hi)
    return [(ax_ + (bx - ax_) * hi, ay + (by - ay) * hi)] + pts[i + 1:]


def clear_of(pts, role, mpp):
    """A move leaving a key player starts MOVE_GAP clear of his marker's edge (2026-10-01, as Figure 1)."""
    return leave(pts, tuple(pts[0]), lambda u: (st.marker_edge(role, u) + st.MOVE_GAP) * mpp)


# --min-arrow / --exit-angle (2026-10-01, the team's call): a move so short that it barely leaves its player's marker
# (0.0 s runner back 3%: 0.33 m; 0.6 s defender toward ball 43%: 0.57 m) is stretched about its start until MIN_ARROW pt
# of it show -- its shape kept, its length then not to scale -- and a move named in EXIT_ANGLE is turned about its
# player so it leaves his marker at that angle (0 = toward goal, 90 = up, -90 = down: the 1.2 s defender's toward
# ball 47% leaves at 4:30 under his momentum and is drawn leaving at 6 o'clock).
MIN_ARROW = 0.0                         # pt, set in main
LABEL_AT = {}                           # ("0.0", label text) -> (x, y) m, set in main from --label-at
STRETCH_TO = {}                         # ("0.0", role, command) -> pt shown, overriding MIN_ARROW (--stretch-to)
STRAIGHT = set()                        # ("0.6", role, command): drawn straight at its --exit-angle (--straight)
NO_LEADER = set()                       # ("0.0", label text): placed close enough to its anchor to need no leader
TWO_LINE = set()                        # ("0.0", label text): name over the % (--two-line)
PASS_UNDER = False                      # --pass-under-moves: the ball's line under the players' moves
EXIT_ANGLE = {}                         # ("1.2", role, command) -> degrees, set in main


def visible_pt(pts, role, mpp):
    shown = clear_of(pts, role, mpp)
    return sum(math.dist(a, b) for a, b in zip(shown, shown[1:])) / mpp


def drawn_path(s, o, mpp):
    """(path drawn, stretch factor, turn in degrees) for a move -- the solver's own path unless MIN_ARROW or
    EXIT_ANGLE apply to it."""
    pts = [tuple(p) for p in o["path"]]
    c, k, turn = pts[0], 1.0, 0.0
    scale = lambda f: [(c[0] + (x - c[0]) * f, c[1] + (y - c[1]) * f) for x, y in pts]
    least = STRETCH_TO.get((f"{s['dt']:.1f}", o["role"], o["command"]), MIN_ARROW)
    if least > 0 and visible_pt(pts, o["role"], mpp) < least:
        lo, hi = 1.0, 2.0
        while visible_pt(scale(hi), o["role"], mpp) < least:
            hi *= 2
        for _ in range(40):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if visible_pt(scale(mid), o["role"], mpp) < least else (lo, mid)
        k, pts = hi, scale(hi)
    want = EXIT_ANGLE.get((f"{s['dt']:.1f}", o["role"], o["command"]))
    if want is not None and (f"{s['dt']:.1f}", o["role"], o["command"]) in STRAIGHT:
        # --straight: a straight arrow leaving at that angle, as long as the path drawn above (its shape not kept)
        L = sum(math.dist(a, b) for a, b in zip(pts, pts[1:]))
        t = math.radians(want)
        pts = [(c[0] + L * f * math.cos(t), c[1] + L * f * math.sin(t)) for f in np.linspace(0.0, 1.0, 25)]
        return pts, k, want - math.degrees(math.atan2(o["path"][-1][1] - c[1], o["path"][-1][0] - c[0]))
    if want is not None:
        e = clear_of(pts, o["role"], mpp)[0]
        t = math.radians(want) - math.atan2(e[1] - c[1], e[0] - c[0])
        cs, sn = math.cos(t), math.sin(t)
        pts = [(c[0] + (x - c[0]) * cs - (y - c[1]) * sn, c[1] + (x - c[0]) * sn + (y - c[1]) * cs) for x, y in pts]
        turn = math.degrees(t)
    return pts, k, turn


# --defender-names: the solver's per-state names of the defender's commands (extract_panel_policy), in English
DEFENDER_NAME = {"toward goal": "Toward goal", "toward ball": "Toward ball", "toward runner": "Toward runner"}
DEFENDER_NAMES = "none"                 # set in main


def draw_panel(ax, s: dict, first: bool, m_per_pt: float, renderer, drawn: dict):
    st.pitch(ax, *((BG_LINE, BG_LINE_LW) if PITCH_OVER_VALUE else ()))
    lab = Placer(ax, renderer, m_per_pt)
    lab.pins = {text: xy for (t, text), xy in LABEL_AT.items() if t == f"{s['dt']:.1f}"}
    lab.dt = f"{s['dt']:.1f}"

    for x, y, side in s["others"]:                 # everyone else: a smaller disc, his team's tint
        st.other_player(ax, (x, y), side, z=2)
        lab.disc((x, y), st.OTHER_D / 2 * m_per_pt + 0.05)
    if s["ball"]:
        st.ball(ax, s["ball"], z=7.5)
        lab.disc(s["ball"], st.BALL_D / 2 * m_per_pt + 0.05)
    for role, (x, y) in s["key"].items():         # the three of the game: his role's marker, outlined
        st.key_player(ax, (x, y), role, z=7)
        lab.disc((x, y), st.marker_radius(role) * m_per_pt + 0.08)
    if s["passer"]:                                # 3v1: the scripted passer, the ball carrier's marker
        st.key_player(ax, s["passer"], "ball carrier", z=7)
        lab.disc(s["passer"], st.marker_radius("ball carrier") * m_per_pt + 0.08)

    # what really happened in the next 0.6 s: grey dots to his marker, hollow
    for role, pts in s["real"].items():
        if len(pts) > 1:
            ax.plot(*zip(*pts), color=st.MUTED, lw=REAL_LW, ls=(0, st.REAL_DOTS), dash_capstyle="round",
                    zorder=6.8)
            st.key_hollow(ax, pts[-1], role, z=6.8)
            lab.line(pts, REAL_LW)
            lab.disc(pts[-1], st.marker_radius(role) * m_per_pt + 0.05)

    # the options: the solver's own 0.6 s paths, width = probability; the less likely drawn on top
    for o in s["options"]:
        pts, lw, color = o["path"], lw_of(o["prob"]), COLOR[o["role"]]
        z = 5.0 + 0.5 * (1.0 - o["prob"])
        if o["stop"]:                              # braking to a stop: the path, then a bar across its end
            ax.plot(*zip(*pts), color=color, lw=lw, solid_capstyle="butt", solid_joinstyle="round", zorder=z)
            d = unit(pts[-1][0] - pts[-4][0], pts[-1][1] - pts[-4][1])
            if d == (0.0, 0.0):
                d = unit(pts[-1][0] - pts[0][0], pts[-1][1] - pts[0][1])
            half = st.head_width(lw) / 2 * m_per_pt   # the bar as wide as a head on this line
            ex, ey = pts[-1]
            ax.plot([ex - d[1] * half, ex + d[1] * half], [ey + d[0] * half, ey - d[0] * half], color=color,
                    lw=max(1.0, 0.7 * lw), solid_capstyle="butt", zorder=z)
            lab.disc((ex, ey), half)
        else:
            pts, stretch, turn = drawn_path(s, o, m_per_pt)
            shown = clear_of(pts, o["role"], m_per_pt)
            st.arrow(ax, shown, color, lw, z=z, mpp=m_per_pt)
            lab.head(pts[-1], lw)
        lab.line(pts if o["stop"] else shown, lw)
        d = unit(pts[-1][0] - pts[-4][0], pts[-1][1] - pts[-4][1])
        text = f"{o['prob']:.0%}"
        at_rest = o["stop"] and o["rests"]
        if o["stop"] and not o["rests"]:   # braking, still moving at 0.6 s (S20, S36): labelled like a move
            text = f"Slow down {text}"
        elif at_rest:            # at rest inside the 0.6 s (S05: speed 0 at 0.38 s / 0.48 s)
            text = f"Stop {text}"
            # its bar sits on the player: label him, on the side his other options leave free
            rest = [q for q in s["options"] if q["role"] == o["role"] and q is not o]
            d = unit(-sum((q["path"][-1][0] - pts[0][0]) * q["prob"] for q in rest),
                     -sum((q["path"][-1][1] - pts[0][1]) * q["prob"] for q in rest)) if rest else d
        elif o["role"] == "ball carrier":
            text = f"Dribble {text}"
        elif o["role"] == "defender" and DEFENDER_NAMES != "none":
            if o["name"] not in DEFENDER_NAME:
                raise SystemExit(f"{s['code']}: no English name for the defender's {o['name']!r}")
            name = DEFENDER_NAME[o["name"]]
            text = f"{name.replace('Toward', 'To') if DEFENDER_NAMES == 'short' else name} {text}"
        # a stop's label is anchored on the player, so it can go anywhere round him: placed after the moves'
        lab.add(o["prob"] - (1.0 if at_rest else 0.0), pts[0] if at_rest else pts[-1], d, text,
                pct_style(color), color)
        drawn["options"].append({"role": o["role"], "command": o["command"], "prob": o["prob"], "label": text,
                                 "lw": lw, "path": o["path"], "stop": o["stop"], "rests": o["rests"],
                                 "drawn_from": None if o["stop"] else list(shown[0])})
        if not o["stop"] and (stretch != 1.0 or turn):
            drawn["options"][-1].update(drawn_path=pts, stretched=stretch, turned_deg=turn)

    # passes (2026-10-01, the user's call): the ball's line solid charcoal (Figure 1's), leaving the passer clear of
    # his marker, labelled "Pass NN%"; the receiver's run onto it in his colour, labelled "receive NN%" -- one joint
    # attacking option, so the same probability on both
    for q in s["passes"]:
        lw = lw_of(q["prob"])
        (bx, by), (tx, ty) = q["ball"], q["target"]
        pct = f"{q['prob']:.0%}"
        labels = []
        if q["run_from"] is not None:
            run = clear_of([q["run_from"], (tx, ty)], q["to"], m_per_pt)
            st.arrow(ax, run, COLOR[q["to"]], lw, z=5, mpp=m_per_pt)
            lab.head((tx, ty), lw)
            lab.line(run, lw)
            (rx, ry) = run[0]
            mid = (rx + (tx - rx) * 0.55, ry + (ty - ry) * 0.55)
            n = unit(-(ty - ry), tx - rx)                      # its left side, away from the ball's line below
            lab.add(q["prob"] + 1.0, mid, n, f"receive {pct}", pct_style(COLOR[q["to"]]), COLOR[q["to"]])
            labels.append(f"receive {pct}")
        passer = "ball carrier" if s["passer"] is None else "ball carrier"
        line = clear_of([(bx, by), (tx, ty)], passer, m_per_pt)
        st.arrow(ax, line, st.BALL_MOVE, lw, z=4.9 if PASS_UNDER else 6, mpp=m_per_pt)
        lab.head((tx, ty), lw)
        lab.line(line, lw)
        (px, py) = line[0]
        mid = (px + (tx - px) * 0.55, py + (ty - py) * 0.55)
        n = unit(ty - py, -(tx - px))                          # its right side, away from the run above
        lab.add(q["prob"] + 1.0, mid, n, f"Pass {pct}", pct_style(st.BALL_MOVE), st.BALL_MOVE)
        labels.insert(0, f"Pass {pct}")
        drawn["passes"].append({"to": q["to"], "prob": q["prob"], "label": " / ".join(labels), "lw": lw,
                                "ball": (bx, by), "target": (tx, ty), "run_from": q["run_from"]})

    if first:                                      # the roles, named once
        for role, (x, y) in s["key"].items():
            mine = [o for o in s["options"] if o["role"] == role]
            mx = sum((o["path"][-1][0] - x) * o["prob"] for o in mine)
            my = sum((o["path"][-1][1] - y) * o["prob"] for o in mine)
            away = unit(-mx, -my) if (mx or my) else (0.0, -1.0)
            lab.add(-2.0, (x, y), away, ROLE_NAME[role],     # after the numbers, next to his marker
                    dict(fontsize=FS_ROLE, color=COLOR[role], zorder=9, path_effects=st.halo()), None)
        if s["passer"]:                            # 3v1: on the side away from the three
            (x, y), n = s["passer"], len(s["key"])
            away = unit(x - sum(k[0] for k in s["key"].values()) / n, y - sum(k[1] for k in s["key"].values()) / n)
            lab.add(-2.0, (x, y), away, ROLE_NAME["ball carrier"],
                    dict(fontsize=FS_ROLE, color=COLOR["ball carrier"], zorder=9, path_effects=st.halo()), None)
    lab.place()
    drawn.update(key=s["key"], real=s["real"], others=s["others"], ball=s["ball"], passer=s["passer"])


def legend(fig, W, y_in, m_per_pt, renderer):
    """One centred row: coloured path, pass, real movement, probability (inches across the figure)."""
    ax = fig.add_axes((0, (y_in - 0.13) / fig.get_figheight(), 1, 0.26 / fig.get_figheight()))
    ax.set_xlim(0, W); ax.set_ylim(-0.13, 0.13); ax.axis("off")
    kw = dict(fontsize=FS_KEY, color=st.INK, va="center", ha="left")
    r_in = st.KEY_D / 2 / 72
    lw = lw_of(0.5)
    mpp = 1 / 72                                   # this axes is in inches

    def possible(x):
        for k, role in enumerate(("runner", "ball carrier", "defender")):
            y = 0.075 - 0.075 * k
            st.arrow(ax, [(x, y), (x + 0.30, y)], COLOR[role], lw, mpp=mpp)
        return 0.30

    def pass_(x):
        st.arrow(ax, [(x, 0), (x + 0.42, 0)], st.INK, lw, dashed=True, mpp=mpp)
        return 0.42

    def actual(x):
        ax.plot([x, x + 0.30], [0, 0], color=st.MUTED, lw=REAL_LW, ls=(0, st.REAL_DOTS), dash_capstyle="round")
        ax.plot(x + 0.30 + r_in, 0, "o", ms=st.KEY_D, mfc="none", mec=st.MUTED, mew=REAL_LW)
        return 0.30 + 2 * r_in

    def width(x):
        for k, p in enumerate((0.1, 0.5, 1.0)):
            y = 0.075 - 0.075 * k
            ax.plot([x, x + 0.30], [y, y], color=st.INK, lw=lw_of(p), solid_capstyle="butt")
            ax.text(x + 0.35, y, f"{p:.0%}", fontsize=FS_KEY - 1.0, color=st.INK, va="center", ha="left")
        return 0.35 + 0.26

    items = [(possible, "Possible move in\nthe next 0.6 s"), (pass_, "Pass"),
             (actual, "Actual move in\nthe next 0.6 s"), (width, "Width and % = equilibrium\nprobability of choosing it")]
    icon_w = {possible: 0.30, pass_: 0.42, actual: 0.30 + 2 * r_in, width: 0.61}
    text_w = []
    for _, text in items:
        t = ax.text(0, 0, text, **kw)
        text_w.append(t.get_window_extent(renderer).width / fig.dpi)
        t.remove()
    icon_gap, item_gap = 0.07, 0.32
    total = sum(icon_w[f] + icon_gap + w for (f, _), w in zip(items, text_w)) + item_gap * (len(items) - 1)
    x = (W - total) / 2
    for (f, text), w in zip(items, text_w):
        f(x)
        x += icon_w[f] + icon_gap
        ax.text(x, 0, text, **kw)
        x += w + item_gap


# --value-grids: a quiet background (2026-10-01, the user's call): the panels' options stay the subject, so a warm
# grey that no player, arrow or label uses, between the pitch's own ground and a light stone; linear between the
# computed starts. Darker = a better start for the defender (the user's call, 2026-10-01: the dark spots mark where
# he should stand; --value-dark attack restores the first version, lighter = better for the defender).
BG_LOW, BG_HIGH = st.PITCH, "#B8AE9C"
BG_DARK_FOR = "defender"                # set in main from --value-dark
BG_LINE, BG_LINE_LW = "#8C8C8C", 0.8   # pitch lines must stay visible over the darkest shade
PITCH_OVER_VALUE = False                # set in main when a value background is drawn
BG_STEP, BG_FADE, BG_SMOOTH = 0.1, 3.0, 0.6     # m: mesh, edge feather, smoothing (Gaussian sigma)


def value_grid(path):
    g = json.loads(Path(path).read_text())
    pts = [(p["defender_start"][0], p["defender_start"][1], p["value"]) for p in g["points"] if p["status"] == "solved"]
    return np.array(pts, dtype=float)


def value_background(ax, pts, norm, fade=0.0):
    from scipy.interpolate import griddata
    from matplotlib.colors import LinearSegmentedColormap
    xs, ys, vs = pts[:, 0], pts[:, 1], pts[:, 2]
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    X, Y = np.meshgrid(np.arange(x0, x1 + 1e-9, BG_STEP), np.arange(y0, y1 + 1e-9, BG_STEP))
    from scipy.ndimage import gaussian_filter
    Z = griddata((xs, ys), vs, (X, Y), method="linear")
    hole = ~np.isfinite(Z)                          # outside the starts' hull (a dropped corner): nearest start
    if hole.any():
        Z[hole] = griddata((xs, ys), vs, (X[hole], Y[hole]), method="nearest")
    Z = gaussian_filter(Z, BG_SMOOTH / BG_STEP, mode="nearest")   # no facets of the triangulation
    ends = [BG_HIGH, BG_LOW] if BG_DARK_FOR == "defender" else [BG_LOW, BG_HIGH]   # low value = good for him
    rgba = LinearSegmentedColormap.from_list("bg", ends)(norm(Z))
    if fade > 0:
        sm = lambda d: (lambda t: t * t * (3 - 2 * t))(np.clip(d / fade, 0.0, 1.0))
        rgba[..., 3] = sm(np.minimum(X - x0, x1 - X)) * sm(np.minimum(Y - y0, y1 - Y))  # feathered corners
    ax.imshow(rgba, extent=(x0 - BG_STEP / 2, x1 + BG_STEP / 2, y0 - BG_STEP / 2, y1 + BG_STEP / 2), origin="lower",
              interpolation="bilinear", zorder=0.3)


def value_key(fig, W, y_in, margin, words="better start for the defender", size=FS_KEY - 0.5):
    """A small gradient and one line, bottom left of the white strip."""
    from matplotlib.colors import LinearSegmentedColormap
    H = fig.get_size_inches()[1]
    k = size / (FS_KEY - 0.5)                      # the swatch grows with the words
    kax = fig.add_axes((margin / W, (y_in - 0.035 * k) / H, 0.42 * k / W, 0.07 * k / H))
    kax.imshow(np.linspace(0, 1, 64)[None, :], aspect="auto", cmap=LinearSegmentedColormap.from_list("bg", [BG_LOW, BG_HIGH]))
    kax.set_xticks([]); kax.set_yticks([])
    for sp in kax.spines.values():
        sp.set_linewidth(0.4); sp.set_color(st.FAINT)
    shade = "darker" if BG_DARK_FOR == "defender" else "lighter"
    fig.text((margin + 0.48 * k) / W, y_in / H, f"{shade}: {words}", fontsize=size,
             color=st.MUTED, ha="left", va="center")


def main() -> None:
    args = parse_args()
    if args.scenes:
        st.SCENES = args.scenes
    fonts = st.use_font()
    panels = sorted((float(p.stem.split("_")[-1]), json.loads(p.read_text()))
                    for p in args.panels.glob(f"{args.label}_*.json"))
    if not panels:
        raise SystemExit(f"no panels {args.label}_*.json in {args.panels}")
    track = defaultdict(dict)
    for r in csv.DictReader(args.tracking.open(encoding="utf-8")):
        track[int(r["frame_id"])][r["object_id"]] = r
    scenes = [(dt, scene(p, track)) for dt, p in panels]
    for dt, s in scenes:
        s["dt"] = dt
    global MIN_ARROW
    MIN_ARROW = args.min_arrow
    global PASS_UNDER
    PASS_UNDER = args.pass_under_moves
    for spec in args.no_leader:                    # DT|TEXT
        t, text = spec.split("|", 1)
        NO_LEADER.add((f"{float(t):.1f}", text))
    for spec in args.two_line:                     # DT|TEXT
        t, text = spec.split("|", 1)
        TWO_LINE.add((f"{float(t):.1f}", text))
    for spec in args.straight:                     # DT:ROLE:COMMAND
        t, role, cmd = spec.split(":")
        STRAIGHT.add((f"{float(t):.1f}", role.replace("_", " "), int(cmd)))
    for spec in args.stretch_to:                   # DT:ROLE:COMMAND=PT, e.g. 0.0:runner:3=20
        where, pt = spec.split("=")
        t, role, cmd = where.split(":")
        STRETCH_TO[(f"{float(t):.1f}", role.replace("_", " "), int(cmd))] = float(pt)
    for spec in args.label_at:                     # DT|TEXT=X,Y, e.g. "0.0|Defender=30.25,-7.55"
        where, xy = spec.rsplit("=", 1)
        t, text = where.split("|", 1)
        LABEL_AT[(f"{float(t):.1f}", text)] = tuple(float(v) for v in xy.split(","))
    for spec in args.exit_angle:                   # DT:ROLE:COMMAND=DEGREES, e.g. 1.2:defender:3=-90
        where, deg = spec.split("=")
        t, role, cmd = where.split(":")
        EXIT_ANGLE[(f"{float(t):.1f}", role.replace("_", " "), int(cmd))] = float(deg)
    grids = {k: value_grid(v) for k, v in (a.split("=", 1) for a in args.value_grids)}
    global PITCH_OVER_VALUE, BG_DARK_FOR, DEFENDER_NAMES
    DEFENDER_NAMES = args.defender_names
    Placer.GAP = args.label_gap
    PITCH_OVER_VALUE, BG_DARK_FOR = bool(grids), args.value_dark
    from matplotlib.colors import Normalize
    if grids:
        allv = np.concatenate([g[:, 2] for g in grids.values()])
        bg_norm = Normalize(float(allv.min()), float(allv.max()))       # one shade scale for every panel
    boxes = [extent(s) for _, s in scenes]
    # one scale (metres per inch) for every panel; each panel as wide as what it draws needs ("fit",
    # the default) or all as wide as the widest ("equal"); all as tall as the tallest
    widths = [x1 - x0 + 2 * PAD for x0, x1, _, _ in boxes]
    if args.layout == "equal":
        widths = [max(widths)] * len(widths)
    span_y = max(y1 - y0 for _, _, y0, y1 in boxes) + 2 * PAD

    # layout (inches): title, panel titles, the panels, the legend if asked for
    W = args.width_mm * MM
    margin, gap, title_in, head_in, key_in = 0.05, 0.08, 0.30, 0.20, 0.36 if args.legend else 0.0
    if not args.title:                             # --title "": no title row (the caption names the figure)
        title_in = 0.0
    head_in += (args.head_size - FS_PANEL) / 72    # a larger moment label gets its own room
    foot_in = 0.05 if args.no_match_line else 0.22   # the white strip under the panels for the match line
    key_grow = max(0.0, (args.value_key_size - (FS_KEY - 0.5)) / 72) if args.value_grids else 0.0
    foot_in += key_grow                            # a larger key gets its own room; the row moves up by half
    n = len(scenes)
    m_per_in = sum(widths) / (W - 2 * margin - (n - 1) * gap)
    ph = span_y / m_per_in
    H = title_in + head_in + ph + key_in + foot_in
    fig = plt.figure(figsize=(W, H), facecolor=st.PAGE)
    renderer = fig.canvas.get_renderer()
    m_per_pt = m_per_in / 72
    dump = {"width_mm": args.width_mm, "metres_per_inch": m_per_in, "windows_m": [[w, span_y] for w in widths],
            "panels": []}
    flows = {}
    if args.flow_grids:                            # the move fields first: one width scale for every panel
        import render_defender_flow_moments as flow
        fgrids = {float(k): json.loads(Path(v).read_text()) for k, v in (a.split("=", 1) for a in args.flow_grids)}
        for (dt, s), (x0, x1, y0, y1), span_x in zip(scenes, boxes, widths):
            if dt not in fgrids:
                continue
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            flow.same_game(fgrids[dt], s, dt)
            flows[dt] = flow.window_field(fgrids[dt], (cx - span_x / 2, cx + span_x / 2),
                                          (cy - span_y / 2, cy + span_y / 2), BG_SMOOTH, dt)
        flow_top = max(float(np.hypot(*f[5].T).max()) for f in flows.values())
    left, axes = margin, []
    for i, ((dt, s), (x0, x1, y0, y1), span_x) in enumerate(zip(scenes, boxes, widths)):
        pw = span_x / m_per_in
        ax = fig.add_axes((left / W, (foot_in + key_in) / H, pw / W, ph / H))
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        ax.set_xlim(cx - span_x / 2, cx + span_x / 2); ax.set_ylim(cy - span_y / 2, cy + span_y / 2)
        ax.set_aspect("equal")
        drawn = {"dt": dt, "code": s["code"], "frame": s["frame"], "xlim": ax.get_xlim(), "ylim": ax.get_ylim(),
                 "options": [], "passes": []}
        draw_panel(ax, s, i == 0, m_per_pt, renderer, drawn)
        if f"{dt:.1f}" in grids:
            g_dt = grids[f"{dt:.1f}"]
            norm_dt = bg_norm if args.value_scale == "common" else Normalize(float(g_dt[:, 2].min()), float(g_dt[:, 2].max()))
            value_background(ax, g_dt, norm_dt, args.value_fade)
            drawn["value_background"] = {"points": len(g_dt), "vmin": norm_dt.vmin, "vmax": norm_dt.vmax,
                                         "scale": args.value_scale, "dark_for": BG_DARK_FOR}
        if dt in flows:                            # over the background and pitch lines, under every player
            xs, ys, U, V, xy, d = flows[dt]
            sp_, density = flow.streamlines(ax, xs, ys, U, V, flow_top, color=args.flow_color,
                                            alpha=args.flow_alpha, zorder=1.0)
            drawn["flow"] = {"starts": len(xy), "longest_average_move_m": flow_top, "color": args.flow_color,
                             "alpha": args.flow_alpha, "segments": len(sp_.lines.get_segments())}
        if args.frames:
            for sp in ax.spines.values():
                sp.set_visible(True); sp.set_color("#8C8C8C"); sp.set_linewidth(0.8)
        dump["panels"].append(drawn)
        axes.append(ax)
        fig.text(left / W, (foot_in + key_in + ph + 0.05) / H, f"{dt:.1f} s", color=st.INK,
                 fontsize=args.head_size, fontweight="bold", ha="left", va="bottom")
        left += pw + gap
    # the attack's direction: a word and the figures' own arrow, right-aligned on the title row -- or, with no
    # title (2026-10-01, the user's call), on the moments' row, sitting on the same line as "0.0 s" ...
    ka = args.attack_size / FS_KEY                 # the arrow grows with the word
    arrow_in = 0.28 * ka
    if args.title:
        y_title = (H - title_in / 2) / H
        fig.text(margin / W, y_title, args.title, color=st.INK, fontsize=FS_TITLE, fontweight="bold",
                 ha="left", va="center")
        fig.text((W - margin - arrow_in - 0.05 * ka) / W, y_title, "attack", color=st.MUTED,
                 fontsize=args.attack_size, ha="right", va="center")
        y_arrow = y_title
    else:
        word = fig.text((W - margin - arrow_in - 0.05 * ka) / W, (foot_in + key_in + ph + 0.05) / H, "attack",
                        color=st.MUTED, fontsize=args.attack_size, ha="right", va="bottom")
        ext = word.get_window_extent(renderer)
        y_arrow = (ext.y0 + ext.y1) / 2 / fig.dpi / H
    st.fig_arrow(fig, ((W - margin - arrow_in) / W, y_arrow), ((W - margin) / W, y_arrow), color=st.MUTED,
                 lw=st.MOVE_LW * 0.7 * ka)
    if args.legend:
        legend(fig, W, foot_in + key_in / 2, m_per_pt, renderer)
    if grids:
        value_key(fig, W, 0.07 + key_grow / 2, margin, args.value_key, args.value_key_size)
    if not args.no_match_line:                     # the scene's start (the first panel, the solver's t = 0)
        note = st.match_line(scenes[0][1]["code"], scenes[0][1]["frame"])
        st.match_note(fig, note, W - margin, 0.07 + key_grow / 2)
        print(f"match line: {note}")
    st.check_text(fig)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=600, facecolor=fig.get_facecolor())
    fig.savefig(args.output.with_suffix(".pdf"), facecolor=fig.get_facecolor())
    print(f"font: {fonts}")
    if args.dump:                                   # every label as drawn: its text and box in pitch metres
        fig.canvas.draw()                           # (redraw at the figure's own dpi: the last save was the pdf's)
        r = fig.canvas.get_renderer()
        for ax, drawn in zip(axes, dump["panels"]):
            inv = ax.transData.inverted()
            drawn["labels"] = []
            for t in ax.texts:
                patch = t.get_bbox_patch()
                ext = (patch or t).get_window_extent(r)
                (bx0, by0), (bx1, by1) = inv.transform([(ext.x0, ext.y0), (ext.x1, ext.y1)])
                drawn["labels"].append({"text": t.get_text(), "box": [bx0, by0, bx1, by1]})
        args.dump.write_text(json.dumps(dump, indent=1))
    print(f"→ {args.output} (+ .pdf) · {W / MM:.0f} x {H / MM:.0f} mm · windows "
          + " / ".join(f"{w:.1f}" for w in widths) + f" x {span_y:.1f} m · {m_per_in:.2f} m per inch")


if __name__ == "__main__":
    main()
