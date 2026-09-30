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
    p.add_argument("--dump", type=Path, default=None, help="write every drawn coordinate / number as JSON")
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
        t = self.ax.text(*anchor, text, ha="center", va="center", **style)
        ext = t.get_window_extent(self.r)
        pad = 2 * BOX_PAD * style["fontsize"] if "bbox" in style else 0.0
        px_to_m = self.mpp * 72 / self.ax.figure.dpi
        w, h = ext.width * px_to_m + pad * self.mpp + 0.12, ext.height * px_to_m + pad * self.mpp + 0.08
        self.todo.append(dict(rank=rank, anchor=anchor, direction=direction, w=w, h=h, artist=t, leader=leader))

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
            walls.append(res[2])
            labels.append(res[2])
            if res[3] > LEADER and item["leader"]:     # a leader: nothing may sit on it, no leader may cross it
                lead = self._samples([item["anchor"], self._end(item["anchor"], res[2])], 0.08)
                walls += lead
                labels += lead[2:]                     # (its first samples sit on the anchor, shared by no one)
        return total, out

    def place(self):
        base = sorted(self.todo, key=lambda t: -t["rank"])
        names = [t for t in base if t["leader"] is None]            # role names: always after the numbers
        movable = [t for t in base if t["leader"] is not None]
        rng = random.Random(0)
        orders = [base] + [rng.sample(movable, len(movable)) + names for _ in range(self.ORDERS)]
        total, out = min((self._run(o) for o in orders), key=lambda r: r[0])
        if out is None:
            raise SystemExit("no room for every label")
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


def draw_panel(ax, s: dict, first: bool, m_per_pt: float, renderer, drawn: dict):
    st.pitch(ax)
    lab = Placer(ax, renderer, m_per_pt)

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
            st.arrow(ax, pts, color, lw, z=z, mpp=m_per_pt)
            lab.head(pts[-1], lw)
        lab.line(pts, lw)
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
        # a stop's label is anchored on the player, so it can go anywhere round him: placed after the moves'
        lab.add(o["prob"] - (1.0 if at_rest else 0.0), pts[0] if at_rest else pts[-1], d, text,
                pct_style(color), color)
        drawn["options"].append({"role": o["role"], "command": o["command"], "prob": o["prob"], "label": text,
                                 "lw": lw, "path": pts, "stop": o["stop"], "rests": o["rests"]})

    # passes: the ball's line (dashed, charcoal) and the receiver's run onto it (solid, his colour)
    for q in s["passes"]:
        lw = lw_of(q["prob"])
        (bx, by), (tx, ty) = q["ball"], q["target"]
        if q["run_from"] is not None:
            st.arrow(ax, [q["run_from"], (tx, ty)], COLOR[q["to"]], lw, z=5, mpp=m_per_pt)
            lab.head((tx, ty), lw)
            lab.line([q["run_from"], (tx, ty)], lw)
        st.arrow(ax, [(bx, by), (tx, ty)], st.INK, lw, dashed=True, z=6, mpp=m_per_pt)
        lab.head((tx, ty), lw)
        lab.line([(bx, by), (tx, ty)], lw)
        text = f"Pass + receive · {q['prob']:.0%}" if q["run_from"] is not None else f"Pass {q['prob']:.0%}"
        lab.add(q["prob"] + 1.0, (tx, ty), unit(tx - bx, ty - by), text, pct_style(st.INK), st.INK)
        drawn["passes"].append({"to": q["to"], "prob": q["prob"], "label": text, "lw": lw, "ball": (bx, by),
                                "target": (tx, ty), "run_from": q["run_from"]})

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


def main() -> None:
    args = parse_args()
    fonts = st.use_font()
    panels = sorted((float(p.stem.split("_")[-1]), json.loads(p.read_text()))
                    for p in args.panels.glob(f"{args.label}_*.json"))
    if not panels:
        raise SystemExit(f"no panels {args.label}_*.json in {args.panels}")
    track = defaultdict(dict)
    for r in csv.DictReader(args.tracking.open(encoding="utf-8")):
        track[int(r["frame_id"])][r["object_id"]] = r
    scenes = [(dt, scene(p, track)) for dt, p in panels]
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
    foot_in = 0.05 if args.no_match_line else 0.22   # the white strip under the panels for the match line
    n = len(scenes)
    m_per_in = sum(widths) / (W - 2 * margin - (n - 1) * gap)
    ph = span_y / m_per_in
    H = title_in + head_in + ph + key_in + foot_in
    fig = plt.figure(figsize=(W, H), facecolor=st.PAGE)
    renderer = fig.canvas.get_renderer()
    m_per_pt = m_per_in / 72
    dump = {"width_mm": args.width_mm, "metres_per_inch": m_per_in, "windows_m": [[w, span_y] for w in widths],
            "panels": []}
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
        dump["panels"].append(drawn)
        axes.append(ax)
        fig.text(left / W, (foot_in + key_in + ph + 0.05) / H, f"{dt:.1f} s", color=st.INK,
                 fontsize=FS_PANEL, fontweight="bold", ha="left", va="bottom")
        left += pw + gap
    y_title = (H - title_in / 2) / H
    fig.text(margin / W, y_title, args.title, color=st.INK, fontsize=FS_TITLE, fontweight="bold",
             ha="left", va="center")
    # the attack's direction: a word and the figures' own arrow, right-aligned on the title row
    arrow_in = 0.28
    fig.text((W - margin - arrow_in - 0.05) / W, y_title, "attack", color=st.MUTED, fontsize=FS_KEY,
                 ha="right", va="center")
    st.fig_arrow(fig, ((W - margin - arrow_in) / W, y_title), ((W - margin) / W, y_title), color=st.MUTED,
                 lw=st.MOVE_LW * 0.7)
    if args.legend:
        legend(fig, W, foot_in + key_in / 2, m_per_pt, renderer)
    if not args.no_match_line:                     # the scene's start (the first panel, the solver's t = 0)
        note = st.match_line(scenes[0][1]["code"], scenes[0][1]["frame"])
        st.match_note(fig, note, W - margin, 0.07)
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
