#!/usr/bin/env python3
"""The abstract figure: real frames with each moment's minimax options drawn on them.

One row of panels per solver (2026-09-28, 준현's suggestion). Each panel is a real
moment: every player where the tracking has him, the three players of the game
ringed, their last second as a thin trail and their REAL next 0.6 s dotted to a
dashed circle (where he really was). From each of the three, one arrow per
command the solver's equilibrium plays at that moment, pointing the way that
command has him running 0.6 s later (the solver's own movement, momentum
included -- so a reversal points back once he has turned), as thick and long as
its probability and labelled with it; the length is a picture choice, not a
distance (the user's choices of 2026-09-28: momentum in the direction, arrows
long enough to see). A stop is a shorter stem along his run ending in a bar. The
ball carrier's arrows are dribbles; a pass is a dashed white arrow from the ball
to its target. Under each panel, every player's choice in words. Each panel's
game was started from that real moment and looks 1.8 s ahead
(scripts/extract_panel_policy.py gives the numbers).
Labels are English, for the abstract.

Usage:
    python scripts/render_panel_figure.py --panels <dir> --label v3sym --tracking <S05.csv> --output <png>
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Arc, Circle, FancyArrowPatch, Rectangle
from matplotlib.path import Path as MPath

FPS = 25
ROLE_COLOR = {"runner": "#ff3b5c", "defender": "#22d3ee", "ball carrier": "#facc15", "beneficiary": "#facc15"}
NAMES = {"멈추기": "stop", "계속": "keep going", "골문 쪽": "toward goal", "왼쪽 옆": "left", "오른쪽 옆": "right",
         "볼 쪽": "toward ball", "옆으로": "sideways", "러너 차단": "cut off runner", "사이 지키기": "hold between",
         "수혜자 쪽": "toward beneficiary", "러너 쪽": "toward runner", "멈추기(감속)": "slow down",
         "forward": "forward", "back": "back", "left": "left", "right": "right", "stop": "stop"}
MIN_P = 0.02
RING = 0.95         # m, the ring around each player of the game
DOT = 0.55         # m, a player
PAD = 3.0          # m around everything a panel draws
FONT = 8.5
NOW = "#d6dbd4"
# 2026-09-29: a muted, lighter pitch so the players and options carry the picture, not the grass
PAGE, BOX, GRASS, LINES = "#252c28", "#2f3833", "#56645b", "#c3ccc6"
OTHERS_ALPHA = 0.7   # players outside the game, a little clearer than before (was 0.45)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--panels", type=Path, required=True)
    p.add_argument("--label", required=True, help="file prefix: <label>_<dt>.json")
    p.add_argument("--title", default=None)
    p.add_argument("--tracking", type=Path, required=True, help="showcase tracking CSV (attack to the right)")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--paths", action="store_true",
                   help="draw each option as the solver's own 0.6 s path (momentum, braking, turning), "
                        "not as a heading arrow sized by probability (2026-09-29)")
    p.add_argument("--velocity", action="store_true",
                   help="draw each player's velocity at the moment as a thin grey arrow (off by default since "
                        "2026-09-29: it cluttered the panels; the speed stays on the name tag)")
    p.add_argument("--option-names", action="store_true",
                   help="label each arrow with its option's name as well as its % (default: % only, "
                        "the user's call 2026-09-29 -- the names are in the rows under each panel)")
    return p.parse_args()


def english(name: str) -> str:
    if name not in NAMES:
        raise SystemExit(f"no English name for command {name!r}")
    return NAMES[name]


def option_names(shown, aim):
    """English names; two compass moves that both head for one target (the meeting solver has no
    diagonals) also say which way they go."""
    names = [english(o["name"]) for o in shown]
    return [f"{n} ({compass(aim(o))})" if names.count(n) > 1 and any(o["aim"]) else n
            for o, n in zip(shown, names)]


def played_passes(p):
    """The passes drawn: every candidate played with some probability in a multi-pass study (2026-09-29),
    else the one release of the imported game."""
    if p.get("passes"):
        return [q for q in p["passes"] if q["prob"] >= MIN_P]
    rel = p["release"]
    return [dict(rel, where=None)] if rel["target"] and rel["prob"] >= MIN_P else []


def pass_breakdown(p):
    """' (to runner 4 m ahead 20% · ...)' for a multi-pass panel, '' otherwise."""
    shown = [q for q in (p.get("passes") or []) if q["prob"] >= MIN_P]
    if len(shown) < 2:
        return ""
    return " (" + " · ".join(f"{q['to']} {q['where']} {q['prob']:.0%}" for q in shown) + ")"


def choice_rows(p, aim):
    """Every player's choice in words, as (who, text, colour) rows under the panel; a line too long
    for the box breaks at a " · " and its continuation rows have no who."""
    roles = {b["role"]: b for b in p["bodies"].values()}
    rel = p["release"]
    lines = []
    for role in ("defender", "ball carrier", "beneficiary", "runner"):
        if role not in roles:
            continue
        shown = sorted((o for o in roles[role]["options"] if o["prob"] >= MIN_P), key=lambda o: -o["prob"])
        ch = " · ".join(f"{k} {o['prob']:.0%}" for o, k in zip(shown, option_names(shown, aim)))
        if role == "ball carrier":
            ch = f"dribble {1.0 - rel['prob']:.0%}" + (f" ({ch})" if ch else "") + f" · pass {rel['prob']:.0%}{pass_breakdown(p)}"
        elif role in ("runner", "beneficiary") and rel["prob"] >= MIN_P:   # their moves share what the pass leaves
            # only the one a pass goes to lists it: a pass to his teammate is the passer's row, not his
            # (2026-09-29: "pass to the other X%" read as a second pass)
            mine = sum(q["prob"] for q in played_passes(p) if q["to"] == role)
            ch = " · ".join(([ch] if ch else []) + ([f"runs onto the pass {mine:.0%}"] if mine >= MIN_P else []))
        lines.append((role.upper(), ch, ROLE_COLOR[role]))
    if "ball carrier" not in roles:                   # 3v1: the passer is scripted, only his pass decides
        if (p.get("release_steps") or [True])[0]:
            what = f"pass now {rel['prob']:.0%}" + (pass_breakdown(p) or (f" (to {rel['to']})" if rel["prob"] >= MIN_P else ""))
        else:                                          # the ball is rolling between passers: no pass exists
            what = "no pass possible now (ball not with a passer)"
        lines.append(("PASSER", what, "#ffffff"))
    rows = []
    for who, what, color in lines:
        wrapped = []
        for part in what.split(" · "):
            if wrapped and len(wrapped[-1]) + 3 + len(part) <= 52:
                wrapped[-1] += " · " + part
            else:
                wrapped.append(part)
        rows += [(who if k == 0 else "", piece, color) for k, piece in enumerate(wrapped)]
    return rows


def pitch(ax):
    kw = dict(color=LINES, lw=1.0, zorder=0)
    ax.add_patch(Rectangle((-52.5, -34), 105, 68, fill=False, **kw))
    ax.plot([0, 0], [-34, 34], **kw)
    ax.add_patch(Circle((0, 0), 9.15, fill=False, **kw))
    for s in (-1, 1):
        ax.add_patch(Rectangle((s * 52.5 - (16.5 if s > 0 else 0), -20.16), 16.5, 40.32, fill=False, **kw))
        ax.add_patch(Rectangle((s * 52.5 - (5.5 if s > 0 else 0), -9.16), 5.5, 18.32, fill=False, **kw))
        ax.add_patch(Rectangle((s * 52.5 + (0 if s > 0 else -2), -3.66), 2, 7.32, fill=False, **kw))
        ax.plot([s * 41.5], [0], "o", ms=2, color=LINES, zorder=0)
        ax.add_patch(Arc((s * 41.5, 0), 18.3, 18.3, theta1=127 if s > 0 else -53, theta2=233 if s > 0 else 53, **kw))


class Labels:
    """Greedy placement, name tags first, then options from the most likely: each goes to the free spot
    nearest its anchor (an arrow's tip, a player) -- free meaning it covers no player, no arrow, no earlier
    label and stays on the panel -- preferring the side the arrow points to (tags: the side away from his
    arrows). A label that ends up away from its tip gets a thin leader line."""

    def __init__(self, ax, scale, bounds):
        self.ax, self.scale, self.bounds, self.boxes, self.todo = ax, scale, bounds, [], []

    def block(self, x, y, r):
        self.boxes.append((x - r, y - r, x + r, y + r))

    def block_arrow(self, a, b):
        n = max(2, int(math.hypot(b[0] - a[0], b[1] - a[1]) / 0.3))
        for i in range(n + 1):
            self.block(a[0] + (b[0] - a[0]) * i / n, a[1] + (b[1] - a[1]) * i / n, 0.22)

    def add(self, prob, anchor, direction, text, color, tag=False):
        self.todo.append((2.0 if tag else prob, anchor, direction, text, color, tag))

    def place(self):
        for _, anchor, direction, text, color, tag in sorted(self.todo, key=lambda t: -t[0]):
            self._put(anchor, direction, text, color, tag)

    def _size(self, text, size):
        return (0.6 * size / 72) * len(text) * self.scale + 0.4, (1.3 * size / 72) * self.scale + 0.25

    def _put(self, anchor, direction, text, color, tag):
        size = 8 if tag else FONT
        w, h = self._size(text, size)
        (x0, x1), (y0, y1) = self.bounds
        spots = []
        for i in range(-24, 25):
            for j in range(-24, 25):
                cx, cy = anchor[0] + 0.35 * i, anchor[1] + 0.35 * j
                box = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
                if box[0] < x0 + 0.2 or box[2] > x1 - 0.2 or box[1] < y0 + 0.2 or box[3] > y1 - 0.2:
                    continue
                near = (max(box[0] - anchor[0], 0, anchor[0] - box[2]), max(box[1] - anchor[1], 0, anchor[1] - box[3]))
                gap = math.hypot(*near)
                ahead = (cx - anchor[0]) * direction[0] + (cy - anchor[1]) * direction[1]
                spots.append((gap + (0.0 if ahead > 0 else 1.2), cx, cy, box, gap))
        spots.sort(key=lambda t: t[0])
        hit = lambda box: any(box[0] < b[2] and b[0] < box[2] and box[1] < b[3] and b[1] < box[3] for b in self.boxes)
        best = None
        for cost, cx, cy, box, gap in spots:
            if best is not None and cost >= best[0]:
                break
            if hit(box):
                continue
            if gap > 0.6:        # a leader line: each 0.25 m of it over an arrow, a player or a label costs 0.5 m
                ex, ey = min(max(anchor[0], box[0]), box[2]), min(max(anchor[1], box[1]), box[3])
                k = int(gap / 0.25)
                cost += 0.5 * sum(hit((px - 0.05, py - 0.05, px + 0.05, py + 0.05)) for px, py in
                                  ((anchor[0] + (ex - anchor[0]) * t / k, anchor[1] + (ey - anchor[1]) * t / k)
                                   for t in range(2, k + 1)))
            if best is None or cost < best[0]:
                best = (cost, cx, cy, box, gap)
        if best is None:
            raise SystemExit(f"no room for label {text!r}")
        _, cx, cy, box, gap = best
        self.boxes.append(box)
        if gap > 0.6:
            self.ax.plot([anchor[0], cx], [anchor[1], cy], color=color, lw=0.7, alpha=0.8, zorder=8)
        if tag:
            self.ax.text(cx, cy, text, color=color, fontsize=size, fontweight="bold", ha="center", va="center",
                         zorder=9, path_effects=[pe.withStroke(linewidth=2.5, foreground="#11251a")])
        else:
            self.ax.text(cx, cy, text, color="#111", fontsize=size, fontweight="bold", ha="center", va="center",
                         zorder=9, bbox=dict(boxstyle="round,pad=0.2", fc=color, ec="none", alpha=0.95))


def who_is(frame: dict, point) -> str | None:
    """The tracked player standing on a game body's start (attack to the right). The bodies are read
    off this very frame, so the match is exact; the export's role column is not used -- for S13 it
    names the runner and the beneficiary the other way round from the solver's start pick."""
    best = min(((abs(float(r["x"]) - point[0]) + abs(float(r["y"]) - point[1]), o)
                for o, r in frame.items() if o != "ball"), default=(None, None))
    return best[1] if best[0] is not None and best[0] < 0.1 else None


def compass(u) -> str:
    """A drawn direction named against the attack (attack runs to the right)."""
    if abs(u[0]) >= abs(u[1]):
        return "forward" if u[0] > 0 else "back"
    return "left" if u[1] > 0 else "right"


def real_path(track, f, oid, frames):
    return [(float(track[g][oid]["x"]), float(track[g][oid]["y"])) for g in frames if oid in track.get(g, {})]


def main() -> None:
    global RING, DOT
    args = parse_args()
    if args.paths:                  # the paths are true distances: smaller players let the short ones show
        RING, DOT = 0.6, 0.4
    panels = sorted((float(p.stem.split("_")[-1]), json.loads(p.read_text()))
                    for p in args.panels.glob(f"{args.label}_*.json"))
    if not panels:
        raise SystemExit(f"no panels {args.label}_*.json in {args.panels}")
    track = defaultdict(dict)
    for r in csv.DictReader(args.tracking.open(encoding="utf-8")):
        track[int(r["frame_id"])][r["object_id"]] = r
    sign = panels[0][1]["attack_direction"]
    norm = lambda c: (sign * (c[0] - 52.5), sign * (c[1] - 34.0))       # solver corner -> attack to the right
    aim = lambda o: (sign * o["aim"][0], sign * o["aim"][1])


    def unit(vx, vy):
        n = math.hypot(vx, vy)
        return (vx / n, vy / n) if n > 0.1 else None

    def is_stop(b, o):
        return o is b["options"][0]                     # command 0 is the stop in every command set

    def heading(b, o):
        """Which way that option has him moving 0.6 s later (the solver's movement, momentum included);
        for a stop, the way he brakes (his run); None for a standing player told to stop."""
        if is_stop(b, o):
            return unit(sign * b["velocity"][0], sign * b["velocity"][1])
        return (unit(sign * o["end_velocity"][0], sign * o["end_velocity"][1])
                or unit(*aim(o)) or unit(sign * b["velocity"][0], sign * b["velocity"][1]))

    def path_of(o):
        return [norm(pt) for pt in o["path"]] if args.paths and o.get("path") else None

    def tip(b, o):
        """Drawn end: along his heading, longer the likelier (a picture choice, not a distance); a stop's
        stem is shorter so the bar reads as a stop, not a move."""
        x, y = norm(b["pos"])
        u = heading(b, o)
        if path_of(o):
            return path_of(o)[-1]
        if u is None:
            return x, y
        n = (1.2 + 1.5 * o["prob"]) if is_stop(b, o) else (2.0 + 3.0 * o["prob"])
        return x + u[0] * n, y + u[1] * n
    step = round(0.6 * FPS)

    # what each panel has to show; one scale for all panels, as tight as the widest needs
    views = []
    for dt, p in panels:
        f = p["start_frame"]
        fr = track.get(f, {})
        roles = {b["role"]: b for b in p["bodies"].values()}
        ids = {role: who_is(fr, norm(b["pos"])) for role, b in roles.items()}
        pts = [norm(b["pos"]) for b in roles.values()]
        pts += [tip(b, o) for b in roles.values() for o in b["options"] if o["prob"] >= MIN_P]
        pts += [real_path(track, f, oid, [f + step])[0] for oid in ids.values()
                if oid and real_path(track, f, oid, [f + step])]
        rel = p["release"]
        for q in played_passes(p):
            pts.append(norm(q["target"]))
        if p["kind"] == "3v1":                          # the scripted passer (or the rolling ball) is in the play
            pts.append(norm(p["ball"]))
        xs, ys = [x for x, _ in pts], [y for _, y in pts]
        views.append((dt, p, f, fr, roles, ids, ((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2),
                      max(max(xs) - min(xs), max(ys) - min(ys))))
    side_m = max(v[-1] for v in views) + 2 * PAD

    # layout in inches: a framed box per panel -- title, pitch, the choices in words
    n = len(views)
    words = [choice_rows(p, aim) for _, p, *_ in views]
    box_w, pitch_in, margin, gap, title_in, foot_in, sup_in = 5.9, 5.5, 0.15, 0.22, 0.45, 0.55, 0.6
    words_in = 0.35 + 0.25 * max(len(w) for w in words)             # every panel's rows fit
    box_h = words_in + pitch_in + title_in + 0.1
    W, H = 2 * margin + n * box_w + (n - 1) * gap, foot_in + box_h + sup_in
    fig = plt.figure(figsize=(W, H), facecolor=PAGE)
    scale = side_m / pitch_in                                           # metres per inch
    for i, (dt, p, f, fr, roles, ids, (cx, cy), _) in enumerate(views):
        bx0 = margin + i * (box_w + gap)
        fig.add_artist(Rectangle((bx0 / W, foot_in / H), box_w / W, box_h / H, transform=fig.transFigure,
                                 fc=BOX, ec="none", zorder=-1))
        fig.add_artist(Rectangle((bx0 / W, foot_in / H), box_w / W, box_h / H, transform=fig.transFigure,
                                 fill=False, ec="#c9d6cf", lw=1.6, zorder=10))
        ax = fig.add_axes(((bx0 + (box_w - pitch_in) / 2) / W, (foot_in + words_in) / H, pitch_in / W, pitch_in / H))
        ax.set_facecolor(GRASS); pitch(ax)
        ax.set_xlim(cx - side_m / 2, cx + side_m / 2); ax.set_ylim(cy - side_m / 2, cy + side_m / 2)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#8fa89a"); sp.set_linewidth(0.8)
        labels = Labels(ax, scale, ((cx - side_m / 2, cx + side_m / 2), (cy - side_m / 2, cy + side_m / 2)))
        # the 3v1 passer is scripted but in the game: drawn solid (none while the ball rolls between passers)
        passer_id = who_is(fr, norm(p["ball"])) if p["kind"] == "3v1" else None
        for oid, r in fr.items():                     # everyone at this real moment
            if oid == "ball":
                continue
            x, y = float(r["x"]), float(r["y"])
            game = oid in ids.values() or oid == passer_id
            ax.add_patch(Circle((x, y), DOT, color="#f08c3a" if r["side"] == "att" else "#4b86e8",
                                alpha=1.0 if game else OTHERS_ALPHA, zorder=3, lw=0))
            labels.block(x, y, (RING if game else DOT) + 0.15)
        if passer_id:                                   # who has the ball in a 3v1 game
            pr = fr[passer_id]
            labels.add(0, (float(pr["x"]), float(pr["y"])), (0.0, -1.0), f"#{pr['shirt']} passer", "#f0f0f0", tag=True)
        if "ball" in fr:
            ball = (float(fr["ball"]["x"]), float(fr["ball"]["y"]))
            ax.add_patch(Circle(ball, 0.42, color="white", ec="black", lw=0.7, zorder=6))
            labels.block(*ball, 0.5)
        rel = p["release"]
        # the three players: ring, shirt and role, last second, where he really was 0.6 s later, the options
        for role, b in roles.items():
            color, oid = ROLE_COLOR[role], ids[role]
            x, y = norm(b["pos"])
            ax.add_patch(Circle((x, y), RING, fill=False, ec=color, lw=2.2, zorder=5))
            vx, vy = sign * b["velocity"][0], sign * b["velocity"][1]
            now = math.hypot(vx, vy)
            if now > 0.1 and args.velocity:             # his velocity now, out to where 0.6 s of it would take him
                tx, ty = x + vx * 0.6, y + vy * 0.6
                ax.add_patch(FancyArrowPatch((x, y), (tx, ty), arrowstyle="->", mutation_scale=11, lw=1.6, color=NOW,
                                             shrinkA=0, shrinkB=0, zorder=6))
                labels.block_arrow((x, y), (tx, ty))
            shown = [o for o in b["options"] if o["prob"] >= MIN_P]
            if oid:
                past = real_path(track, f, oid, range(f - FPS, f + 1))
                nxt = real_path(track, f, oid, range(f, f + step + 1))
                if len(past) > 1:
                    ax.plot(*zip(*past), color=color, lw=1.3, alpha=0.45, zorder=2)
                if len(nxt) > 1:
                    ax.plot(*zip(*nxt), color="white", lw=1.5, ls=(0, (1.0, 1.6)), zorder=4)
                    ax.add_patch(Circle(nxt[-1], DOT, fill=False, ec="white", lw=1.3, ls=(0, (2, 1.5)), zorder=4))
                    labels.block(*nxt[-1], DOT + 0.1)
                mx = sum((heading(b, o) or (0.0, 0.0))[0] * o["prob"] for o in shown)
                my = sum((heading(b, o) or (0.0, 0.0))[1] * o["prob"] for o in shown)
                m = math.hypot(mx, my) or 1.0
                labels.add(0, (x, y), (-mx / m, -my / m), f"#{fr[oid]['shirt']} {role} · {now:.1f} m/s", color, tag=True)
            for o, name in zip(shown, option_names(shown, aim)):
                text = f"{name} {o['prob']:.0%}" if args.option_names else f"{o['prob']:.0%}"
                u = heading(b, o)
                if u is None:                               # standing and told to stop: a square, no arrow
                    ax.add_patch(Rectangle((x - 0.35, y - 0.35), 0.7, 0.7, fc=color, ec="#0d1f14", lw=1, zorder=8))
                    labels.add(o["prob"], (x, y), (0.0, -1.0), text, color)
                    continue
                ex, ey = tip(b, o)
                lw = 2.5 + 5.0 * o["prob"]
                edge = [pe.Stroke(linewidth=lw + 1.8, foreground="#0d1f14", alpha=0.7), pe.Normal()]
                pts = path_of(o)
                if pts:                                     # the solver's own 0.6 s path under this command
                    ex, ey = pts[-1]
                    end_dir = unit(pts[-1][0] - pts[-4][0], pts[-1][1] - pts[-4][1]) or u
                    if is_stop(b, o):
                        ax.plot(*zip(*pts), color=color, lw=lw, solid_capstyle="butt", zorder=7, path_effects=edge)
                        half = 0.8 + 0.4 * o["prob"]
                        ax.plot([ex - end_dir[1] * half, ex + end_dir[1] * half],
                                [ey + end_dir[0] * half, ey - end_dir[0] * half], color=color, lw=lw + 1.5,
                                solid_capstyle="round", zorder=7, path_effects=edge)
                    else:
                        if role == "ball carrier":
                            text = f"dribble {text}" if args.option_names else text
                        arrow = FancyArrowPatch(path=MPath(pts), arrowstyle="-|>", mutation_scale=10 + 8 * o["prob"],
                                                lw=lw, color=color, alpha=0.6 + 0.4 * o["prob"], zorder=7)
                        arrow.set_path_effects(edge)
                        ax.add_patch(arrow)
                    for a_, b_ in zip(pts[:-1], pts[1:]):
                        labels.block_arrow(a_, b_)
                    labels.add(o["prob"], (ex, ey), end_dir, text, color)
                    continue
                if is_stop(b, o):                           # a stem along his run ending in a bar across it
                    half = 0.8 + 0.4 * o["prob"]
                    ax.plot([x, ex], [y, ey], color=color, lw=lw, solid_capstyle="butt", zorder=7, path_effects=edge)
                    ax.plot([ex - u[1] * half, ex + u[1] * half], [ey + u[0] * half, ey - u[0] * half], color=color,
                            lw=lw + 1.5, solid_capstyle="round", zorder=7, path_effects=edge)
                    labels.block(ex, ey, half)
                else:
                    if role == "ball carrier":
                        text = f"dribble {text}" if args.option_names else text
                    arrow = FancyArrowPatch((x, y), (ex, ey), arrowstyle="-|>", mutation_scale=12 + 10 * o["prob"],
                                            lw=lw, color=color, alpha=0.6 + 0.4 * o["prob"], shrinkA=0, shrinkB=0,
                                            zorder=7)
                    arrow.set_path_effects(edge)
                    ax.add_patch(arrow)
                labels.block_arrow((x, y), (ex, ey))
                labels.add(o["prob"], (ex, ey), u, text, color)
        for q in played_passes(p):                      # every pass played with some probability
            bx, by = norm(p["ball"]); tx, ty = norm(q["target"])
            receiver = roles.get(q["to"])
            if receiver is not None:                    # and the receiver running onto it, in his colour
                rx, ry = norm(receiver["pos"])
                run = FancyArrowPatch((rx, ry), (tx, ty), arrowstyle="-|>", mutation_scale=12, color=ROLE_COLOR[q["to"]],
                                      lw=1.2 + 2.5 * q["prob"], ls=(0, (4, 2)), zorder=6, shrinkA=0, shrinkB=0)
                run.set_path_effects([pe.Stroke(linewidth=2.6 + 2.5 * q["prob"], foreground="#0d1f14", alpha=0.6),
                                      pe.Normal()])
                ax.add_patch(run)
                labels.block_arrow((rx, ry), (tx, ty))
            ax.add_patch(FancyArrowPatch((bx, by), (tx, ty), arrowstyle="-|>", mutation_scale=16, color="white",
                                         lw=1.5 + 3 * q["prob"], ls="--", zorder=7))
            labels.block_arrow((bx, by), (tx, ty))
            d = math.hypot(tx - bx, ty - by) or 1.0
            where = f" ({q['where']})" if q.get("where") else ""
            pct = f"{q['prob']:.0%}"
            # the pass read twice: by the one who plays it, on the ball's line in his colour (the 3v1 passer is
            # scripted, so white), and by the one who takes it, where his run meets the ball, in his colour
            passer_colour = ROLE_COLOR["ball carrier"] if p["kind"] == "2v1" else "#f0f0f0"
            labels.add(q["prob"], ((bx + tx) / 2, (by + ty) / 2), (-(ty - by) / d, (tx - bx) / d),
                       f"pass to {q['to']}{where} {pct}" if args.option_names else pct, passer_colour)
            labels.add(q["prob"], (tx, ty), ((tx - bx) / d, (ty - by) / d),
                       f"runs onto it {pct}" if args.option_names else pct,
                       ROLE_COLOR.get(q["to"], "#ffffff"))
        labels.place()
        fig.text((bx0 + box_w / 2) / W, (foot_in + box_h - 0.08) / H, f"{dt:.1f} s after the start", color="white",
                 fontsize=13, fontweight="bold", ha="center", va="top")
        for k, (who, piece, color) in enumerate(words[i]):
            y = (foot_in + words_in - 0.2 - 0.25 * k) / H
            if who:
                fig.text((bx0 + 0.2) / W, y, who, color=color, fontsize=9.5, fontweight="bold", ha="left", va="top")
            fig.text((bx0 + 1.45) / W, y, piece, color="white", fontsize=9.5, ha="left", va="top")
    fig.text(0.5, (foot_in + box_h + sup_in / 2) / H, args.title or panels[0][1]["code"].split("@")[0],
             color="white", fontsize=15, fontweight="bold", ha="center", va="center")
    fig.text(0.5, 0.36 / H, ("Curve = an option: the path it takes the player along in the next 0.6 s (the solver's "
              "physics: momentum, braking, turning); width and % = its probability.   Bar = stop.   "
              if args.paths else
              "Arrow = an option: the way it has the player running 0.6 s later (momentum in); "
              "width and % = its probability.   Bar = stop.   ")
             + "Solid = dribble, dashed white = pass, dashed in a player's colour = his run onto it.",
             color="#e8ecef", fontsize=9.5, ha="center",
             va="center")
    fig.text(0.5, 0.14 / H, ("Thin grey arrow = his velocity at this moment (0.6 s of it).   " if args.velocity else "")
             + "Speed on the name tag.   White dotted line ending in a dashed circle = where the player really was 0.6 s later.   "
             "Faint trail = his last second.   Attack →", color="#e8ecef", fontsize=9.5, ha="center", va="center")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200, facecolor=fig.get_facecolor())
    fig.savefig(args.output.with_suffix(".svg"), facecolor=fig.get_facecolor())
    print(f"→ {args.output} (+ .svg) · panels {side_m:.1f} m wide")


if __name__ == "__main__":
    main()
