#!/usr/bin/env python3
"""Figure 1 of the paper: an off-ball run forces the defender to choose whom to protect (S05).

The intuition figure; the solver, its payoffs and the equilibrium belong to Figure 2
(scripts/render_figure2_abstract.py), with which it shares one print style (scripts/figure_style.py:
face, colours, markers, arrows, the match line). A tree of three panels, every position read from the
S05 tracking (attack to the right):

  Current state -- Figure 2's "0.6 s": each of the three's last 0.6 s as a line, the runner's run to
      come (his real run, to the shot), the ball carrier's next 0.6 s, and the defender's two possible
      answers, "Follow?" and "Stay?". At this moment the solver's defender is indifferent between
      dropping with the runner and stepping to the ball (equal values, 57/43 mix), and the real
      defender slows to his slowest speed of the play;
  Future 1, follow the runner (what really happened, drawn at the shot): the defender dropped with
      the runner, the ball carrier drove into the space he left and shot from 18 m;
  Future 2, stay with the ball carrier (counterfactual, drawn when the through ball arrives -- 1.06 s
      at the pass model's fitted ball speed): the through ball the solver plays at the dilemma (74%:
      along the run, 8 m ahead of the runner) has reached the space in behind and the runner runs onto
      it (onside when it was played: #33 Elvedi holds the line); the defender has stepped toward the ball
      carrier as far as the solver's own motion takes him in that time (1.8 m: he was moving toward the
      runner and has to turn); everyone else where he really was then.

Why the pairing is the model's and not only football sense (checked 2026-09-29 against the full
multi-pass study): the through ball is worth 0.674 against "toward ball" and 0.661 against
"toward goal"; by the defender's indifference the ball carrier's drive is then worth more against
"toward goal" (0.683 vs 0.645). No number is drawn -- Figure 1 carries none by design.

Encoding (the team's simplification of 2026-09-29, restyled for print 2026-09-30): attackers blue,
defenders vermillion; the runner / ball carrier / defender are a diamond / disc / square with a
charcoal outline, named once in the current state; everyone else a smaller disc in a light tint of his
team's colour. Every player movement one style -- a line in his team's colour with a small head on
what is still to come, thinner and lighter for a path already run (it ends at the player); the ball's
moves (pass, shot) dashed charcoal. No captions inside the panels (times go in the caption); the match
line (date, teams, clock at the solver start) once, under the figure.

Picture choices, not data: the defender's two option arrows point at where the runner / the ball
carrier really were 0.6 s later, both OPTION_M = 3 m long; tracking paths are drawn through a centred
5-frame mean (smooth, as before). No solver output is changed or re-run: the through ball's target is
read from Figure 2's panel file, its flight time from the pass model's ball speed, and Future 2's
defender is moved with the solver's own motion function.

Usage:
    python scripts/render_figure1_dilemma.py --output out/showcase_v1/figure1_S05/S05_figure1_ssac_v2.png
    (--layout tree|row; writes the .png at 600 dpi and a vector .pdf with the face embedded)
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
import figure_style as st  # noqa: E402  (the print style shared with Figure 2)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402

FPS = 25
P = Path("data/processed/showcase_v1")
POST = 3.66                      # m, half the goal's width
GOAL_X = 52.5
ASPECT = 1.68                                    # panel width / height
OFFSIDE_LINE = None                              # set from --offside-line
LABEL_FS = st.FS_LABEL                           # set smaller for --layout row


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--code", default="S05")
    p.add_argument("--tracking", type=Path, default=None, help="default: showcase tracking/<code>.csv")
    p.add_argument("--starts", type=Path, default=P / "solver_starts_202609281928.csv",
                   help="the user's solver starts: start frame and runner / defender / beneficiary ids")
    p.add_argument("--panel", type=Path, default=None,
                   help="Figure 2's panel at the dilemma moment (the played through ball); "
                        "default figure_multi/panels/<code>-full_<dilemma-dt>.json")
    p.add_argument("--before-dt", type=float, default=0.0,
                   help="where the current state's movement lines start: seconds after the solver start")
    p.add_argument("--dilemma-dt", type=float, default=0.6,
                   help="the current state (and Future 2): seconds after the solver start")
    p.add_argument("--shot-frame", type=int, default=53904,
                   help="the beneficiary's shot: last frame at his feet (0.43 m); the ball goes from "
                        "5 to 24 m/s the next frame (read off the S05 tracking)")
    p.add_argument("--pass-model", type=Path, default=Path("data/processed/pass_models_sym/Asym_crossfit.json"),
                   help="the study's pass model (or per-match router): its fitted ball speed times the through ball")
    p.add_argument("--states", type=Path, default=P / "figure_2v1" / "states_2v1_all.json",
                   help="the solved records: the defender's maximum speed at the dilemma")
    p.add_argument("--physics", type=Path, default=Path("data/processed/physics_limits/agile_p999_nodelay.json"),
                   help="the study's motion limits (multipass_full_2v1: P99.9 accelerations, no delay)")
    p.add_argument("--physics-step", type=float, default=0.025, help="the study's physics step (its manifest)")
    p.add_argument("--offside-line", choices=("stay", "both", "none"), default="none",
                   help="the offside line at the pass moment: in Future 2, in the current state too, or "
                        "nowhere (default since 2026-09-29, the team's simplification)")
    p.add_argument("--layout", choices=("tree", "row"), default="tree")
    p.add_argument("--no-match-line", action="store_true", help="leave out the date / teams / clock line")
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


# ---------------------------------------------------------------------------------------------- data

def load(args):
    code = args.code
    tracking = args.tracking or P / "tracking" / f"{code}.csv"
    track = defaultdict(dict)
    for r in csv.DictReader(tracking.open(encoding="utf-8-sig")):
        track[int(r["frame_id"])][r["object_id"]] = r
    start = next(r for r in csv.DictReader(args.starts.open(encoding="utf-8-sig")) if r["code"] == code)
    ids = {"runner": start["runner_id"], "defender": start["defender_id"], "beneficiary": start["beneficiary_id"]}
    if start["carrier_id"] != start["beneficiary_id"]:
        raise SystemExit(f"{code}: this figure assumes the beneficiary is the ball carrier (a 2v1 scene)")
    f0 = int(start["start_frame"])
    before, dilemma = f0 + round(args.before_dt * FPS), f0 + round(args.dilemma_dt * FPS)
    panel_path = args.panel or P / "figure_multi" / "panels" / f"{code}-full_{args.dilemma_dt:.1f}.json"
    panel = json.loads(panel_path.read_text())
    if panel["start_frame"] != dilemma:
        raise SystemExit(f"{panel_path} starts at frame {panel['start_frame']}, not the dilemma frame {dilemma}")
    sign = panel["attack_direction"]
    norm = lambda c: (sign * (c[0] - 52.5), sign * (c[1] - 34.0))     # solver corner -> attack to the right
    to_runner = [q for q in panel["passes"] if q["to"] == "runner"]
    if not to_runner:
        raise SystemExit(f"{panel_path}: the solver plays no pass to the runner at the dilemma moment")
    through = max(to_runner, key=lambda q: q["prob"])
    shot = args.shot_frame
    at = lambda f, oid: (float(track[f][oid]["x"]), float(track[f][oid]["y"]))
    ball_gap = math.dist(at(shot, "ball"), at(shot, ids["beneficiary"]))
    ball_speed = math.dist(at(shot, "ball"), at(shot + 1, "ball")) * FPS
    # the shot, read off the ball: from the shot frame until it stops going toward goal (the keeper)
    flight = [shot]
    while flight[-1] + 1 in track and at(flight[-1] + 1, "ball")[0] > at(flight[-1], "ball")[0]:
        flight.append(flight[-1] + 1)
    # the offside line when the through ball is played: the second-last defender (Law 11; attack to the right)
    deepest = sorted(((float(r["x"]), oid) for oid, r in track[dilemma].items() if r["side"] == "def"), reverse=True)
    line_x, line_id = deepest[1]
    margin = line_x - at(dilemma, ids["runner"])[0]
    print(f"{code}: offside line at the pass = #{track[dilemma][line_id]['shirt']} {track[dilemma][line_id]['name']} "
          f"x {line_x:.2f}; the runner is {'onside' if margin >= 0 else 'OFFSIDE'} by {abs(margin):.2f} m")
    print(f"{code}: before {before}, dilemma {dilemma}, shot {shot} (ball {ball_gap:.2f} m from the beneficiary, "
          f"{ball_speed:.0f} m/s next frame; {(shot - f0) / FPS:.2f} s after the start) · through ball "
          f"{through['where']} {through['prob']:.0%} to {tuple(round(v, 1) for v in norm(through['target']))}")
    out = dict(track=track, ids=ids, before=before, dilemma=dilemma, shot=shot, flight=flight, at=at,
               through_target=norm(through["target"]), passer=norm(panel["ball"]), start=f0, line_id=line_id)
    # Future 2 ends when the through ball reaches the target: its flight at the pass model's fitted speed
    router = json.loads(args.pass_model.read_text())
    model = (args.pass_model.parent / router["by_match"].get(start["match_id"], router["default"])
             if router.get("kind") == "per_match" else args.pass_model)
    fit = json.loads(model.read_text())["ball_speed"]
    length = math.dist(out["passer"], out["through_target"])
    air = length / min(max(fit["intercept"] + fit["slope"] * length, fit["min"]), fit["max"])
    out["arrival"] = dilemma + round(air * FPS)
    rec = next(r for r in json.loads(args.states.read_text())["states"]
               if r["provenance"].get("code") == f"{code}@{args.dilemma_dt:g}")
    dfn = panel["bodies"]["defender"]
    out["stay_path"] = step_to_ball(norm(dfn["pos"]), (sign * dfn["velocity"][0], sign * dfn["velocity"][1]),
                                    at(dilemma + round(0.6 * FPS), ids["beneficiary"]), air,
                                    float(rec["scenario"]["defender"]["maximum_speed"]), args)
    print(f"{code}: through ball {length:.1f} m in the air {air:.2f} s (arrives frame {out['arrival']}); the "
          f"defender stepping to the ball carrier gets {math.dist(out['stay_path'][0], out['stay_path'][-1]):.2f} m "
          f"in that time (his real FOLLOW move: "
          f"{math.dist(at(dilemma, ids['defender']), at(out['arrival'], ids['defender'])):.2f} m)")
    return out


def step_to_ball(position, velocity, aim, seconds, max_speed, args):
    """The defender's STAY run with the solver's own motion (agile_motion.steer and its plant-and-cut, the
    study's limits and physics step): from his state at the dilemma, full speed toward where the ball
    carrier really was 0.6 s later, for as long as the through ball is in the air; of the curve and the
    cut, the one ending nearer the wanted velocity, as agile_motion.advance chooses. Positions [(x, y)]."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "andrew-passer2on1"))
    import numpy as np
    from passer2on1.agile_motion import Physics, _plant_applies, steer
    physics = Physics(**json.loads(args.physics.read_text()))
    dt, aim = args.physics_step, np.asarray(aim, dtype=float)
    stand_steps = round(physics.plant_seconds / dt)

    def run(plant):
        x, v = np.asarray(position, dtype=float), np.asarray(velocity, dtype=float)
        pts, phase, standing = [tuple(x)], None, 0
        for _ in range(round(seconds / dt)):
            desired = (aim - x) / np.linalg.norm(aim - x) * max_speed
            if plant and phase is None:
                if not _plant_applies(v, desired, physics):
                    return None
                phase = "brake"
            if phase == "brake":
                speed, drop = float(np.linalg.norm(v)), physics.plant_braking * dt
                v, phase = (np.zeros(2), "stand") if speed <= drop else (v * (1.0 - drop / speed), phase)
            elif phase == "stand":
                standing += 1
                phase = "go" if standing >= stand_steps else phase
            else:
                v = steer(v, desired, physics, dt)
            x = x + v * dt
            pts.append(tuple(x))
        return pts, np.linalg.norm(v - desired)

    curve, cut = run(False), run(True)
    return cut[0] if cut is not None and cut[1] < curve[1] - 1e-12 else curve[0]


# ------------------------------------------------------------------------------------------ drawing

ROLE = {"runner": "runner", "defender": "defender", "beneficiary": "ball carrier"}   # the style's role names


def unit(v):
    n = math.hypot(*v)
    return (v[0] / n, v[1] / n)


def trimmed(pts, cut):
    """The path without its last `cut` metres (so a line ends at the edge of the marker it reaches)."""
    out, left = list(pts), cut
    while len(out) > 2 and math.dist(out[-1], out[-2]) <= left:
        left -= math.dist(out[-1], out[-2])
        out.pop()
    (ax_, ay), (bx, by) = out[-2], out[-1]
    k = 1 - left / max(math.dist(out[-2], out[-1]), 1e-9)
    out[-1] = (ax_ + (bx - ax_) * k, ay + (by - ay) * k)
    return out


def smooth(pts, half=2):
    """Centred moving average over 2*half+1 frames (0.2 s at 25 fps), ends kept -- for drawing only, so a
    tracking wobble does not kink the line (unchanged from the first version of this figure)."""
    pts = list(pts)
    if len(pts) <= 2:
        return pts
    out = [pts[0]]
    for i in range(1, len(pts) - 1):
        k = min(half, i, len(pts) - 1 - i)
        seg = pts[i - k:i + k + 1]
        out.append((sum(p[0] for p in seg) / len(seg), sum(p[1] for p in seg) / len(seg)))
    return out + [pts[-1]]


def ball_move(ax, pts, *, shrink_start=0.0, shrink_end=0.0):
    """The ball's move (a shot, a pass): dashed charcoal, one small head; starts clear of the kicker and
    (shrink_end, pt) stops at the edge of the ball / player drawn at its end, so the head covers neither."""
    mpp = st.metres_per_point(ax)
    pts = smooth(pts)
    if shrink_start:
        u = unit((pts[1][0] - pts[0][0], pts[1][1] - pts[0][1]))
        pts[0] = (pts[0][0] + u[0] * shrink_start * mpp, pts[0][1] + u[1] * shrink_start * mpp)
    if shrink_end:
        pts = trimmed(pts, shrink_end * mpp)
    st.arrow(ax, pts, st.INK, st.BALL_LW, dashed=True, z=6)


def move(ax, pts, role, *, ran):
    """Every player movement, one style: a line in his team's colour, lightly smoothed. `ran` -- a path
    already run: thinner and lighter, it ends at the player (at his marker's edge), no head; otherwise a
    move still to come: it starts at his marker's edge and ends in a small head."""
    mpp = st.metres_per_point(ax)
    pts, color = smooth(pts), st.ROLE_COLOR[ROLE[role]]
    if ran:
        ax.plot(*zip(*trimmed(pts, (st.key_radius() + 0.5) * mpp)), color=color, lw=st.PAST_LW,
                alpha=st.PAST_ALPHA, zorder=4.9, solid_capstyle="round", solid_joinstyle="round")
        return
    u = unit((pts[1][0] - pts[0][0], pts[1][1] - pts[0][1]))
    gap = (st.key_radius() + 0.8) * mpp
    pts[0] = (pts[0][0] + u[0] * gap, pts[0][1] + u[1] * gap)
    st.arrow(ax, pts, color, st.MOVE_LW, z=5)


def label(ax, xy, text, *, size=None, weight="normal", color=st.INK, ha="center", va="center", z=9):
    ax.text(*xy, text, fontsize=size or LABEL_FS, fontweight=weight, color=color, ha=ha, va=va, zorder=z,
            path_effects=st.halo())


def base(ax, d, frame, window):
    """The pitch and everyone but the three where they were at `frame`."""
    st.pitch(ax)
    (x0, x1), (y0, y1) = window
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    for oid, r in d["track"][frame].items():
        if oid == "ball" or oid in d["ids"].values():
            continue
        st.other_player(ax, (float(r["x"]), float(r["y"])), r["side"])


def player(ax, xy, role):
    """One of the three: his role's marker in his team's colour, outlined."""
    st.key_player(ax, xy, ROLE[role])


def ball(ax, xy, carrier=None):
    """The ball; if it lies under the carrier's marker (S05's shot: 0.43 m from his centre) it is drawn at
    the marker's edge on its real bearing from him -- a drawing convention, so both stay visible."""
    if carrier is not None:
        r = (st.marker_radius("ball carrier") + st.BALL_D / 2 - 0.6) * st.metres_per_point(ax)
        dx, dy = xy[0] - carrier[0], xy[1] - carrier[1]
        if 0 < math.hypot(dx, dy) < r:
            u = unit((dx, dy))
            xy = (carrier[0] + u[0] * r, carrier[1] + u[1] * r)
    st.ball(ax, xy)


def path(d, oid, f_from, f_to):
    return [(float(d["track"][g][oid]["x"]), float(d["track"][g][oid]["y"]))
            for g in range(f_from, f_to + 1) if oid in d["track"].get(g, {})]


def key_positions(d, frame):
    return {role: d["at"](frame, oid) for role, oid in d["ids"].items()}


OPTION_M = 3.0          # m, the defender's two possible answers in the current state: a picture choice


def option_tips(d, length=None):
    """The defender's two answers at the dilemma moment, as drawn: toward where the runner / the ball
    carrier really were 0.6 s later. `length` None = as far as he really moved from there to the shot
    (the FOLLOW he played; STAY is drawn the same distance the other way)."""
    f, ids, at = d["dilemma"], d["ids"], d["at"]
    dx, dy = at(f, ids["defender"])
    later = f + round(0.6 * FPS)
    n = length or math.dist(at(f, ids["defender"]), at(d["shot"], ids["defender"]))
    tips = {}
    for name, role in (("follow", "runner"), ("stay", "beneficiary")):
        u = unit((at(later, ids[role])[0] - dx, at(later, ids[role])[1] - dy))
        tips[name] = (dx + u[0] * n, dy + u[1] * n)
    return tips


def zone(ax, xy, role):
    """A space that opens (the shooting lane): a faint fill in the attack's colour with a thin edge."""
    color = st.ROLE_COLOR[ROLE[role]]
    ax.add_patch(Polygon(xy, closed=True, fc=color, alpha=0.07, ec="none", zorder=2))
    ax.add_patch(Polygon(xy, closed=True, fc="none", ec=color, lw=0.5, alpha=0.4, zorder=2))


def offside_line(ax, d, frame):
    """A thin dashed line through the second-last defender at the pass moment: the runner is behind it."""
    x = d["at"](frame, d["line_id"])[0]
    (y0, y1) = ax.get_ylim()
    ax.plot([x, x], [y0, y1], color=st.MUTED, lw=0.7, ls=(0, (4, 3)), zorder=2.5)
    label(ax, (x - 0.4, y1 - 0.9), "offside line", size=st.FS_SMALL, ha="right", color=st.MUTED)


def past(ax, d, f_from, f_to):
    """Each of the three's past movement, in every panel (the team's call, 2026-09-29)."""
    for role, oid in d["ids"].items():
        move(ax, path(d, oid, f_from, f_to), role, ran=True)


def panel_current(ax, d, window):
    """The current state: where the three came from in the last 0.6 s, the runner's run to come, the ball
    carrier's next 0.6 s, and the defender's two possible answers."""
    f, ids = d["dilemma"], d["ids"]
    k = key_positions(d, f)
    base(ax, d, f, window)
    if OFFSIDE_LINE == "both":
        offside_line(ax, d, f)
    past(ax, d, d["before"], f)
    move(ax, path(d, ids["runner"], f, d["shot"]), "runner", ran=False)
    move(ax, path(d, ids["beneficiary"], f, f + round(0.6 * FPS)), "beneficiary", ran=False)
    tips = option_tips(d, OPTION_M)
    for name in ("follow", "stay"):
        move(ax, [k["defender"], tips[name]], "defender", ran=False)
    for role, xy in k.items():
        player(ax, xy, role)
    ball(ax, d["at"](f, "ball"), carrier=k["beneficiary"])
    fu, sy = tips["follow"], tips["stay"]
    label(ax, (fu[0] + 0.8, fu[1] - 1.5), "Follow?", ha="left")
    label(ax, (sy[0] - 0.5, sy[1] + 0.9), "Stay?", ha="right")
    # who is whom, once (the markers' key), in his team's colour
    label(ax, (k["runner"][0] - 0.3, k["runner"][1] + 1.6), st.ROLE_NAME["runner"], color=st.ATTACK)
    label(ax, (k["defender"][0] + 1.1, k["defender"][1] - 2.0), st.ROLE_NAME["defender"], color=st.DEFENCE)
    label(ax, (k["beneficiary"][0] + 1.2, k["beneficiary"][1] - 1.3), st.ROLE_NAME["ball carrier"],
          ha="left", color=st.ATTACK)


def panel_follow(ax, d, window):
    """Future 1, the real play, drawn at the shot: the paths the three really ran since the dilemma -- the
    defender dropping with the runner, the ball carrier driving into the space he left."""
    f, s = d["dilemma"], d["shot"]
    k = key_positions(d, s)
    base(ax, d, s, window)                            # everyone where he really was at the shot
    shot_from = d["at"](s, "ball")
    zone(ax, [shot_from, (GOAL_X, -POST), (GOAL_X, POST)], "beneficiary")
    past(ax, d, f, s)
    flight = [d["at"](g, "ball") for g in d["flight"]]
    ball_move(ax, [flight[0], flight[-1]], shrink_start=st.key_radius(), shrink_end=st.OTHER_D / 2 + 0.8)
    for role, xy in k.items():
        player(ax, xy, role)
    ball(ax, shot_from, carrier=k["beneficiary"])
    label(ax, ((flight[0][0] + flight[-1][0]) / 2, (flight[0][1] + flight[-1][1]) / 2 + 3.0), "Shoot")


def panel_stay(ax, d, window):
    """Future 2, counterfactual, drawn when the through ball reaches the runner (its flight at the pass
    model's fitted speed): the ball carrier played it at the dilemma moment and is drawn where he played
    it; the defender stepped toward him (the solver's own motion, see step_to_ball); the runner ran on along
    his real run and now runs onto the ball; everyone else where he really was then (the solver's
    background players walk their real tracks too)."""
    f, a, ids = d["dilemma"], d["arrival"], d["ids"]
    base(ax, d, a, window)
    if OFFSIDE_LINE:
        offside_line(ax, d, f)
    target, stay = d["through_target"], d["stay_path"]
    k = {"beneficiary": d["at"](f, ids["beneficiary"]), "defender": stay[-1], "runner": d["at"](a, ids["runner"])}
    move(ax, path(d, ids["beneficiary"], d["before"], f), "beneficiary", ran=True)
    move(ax, stay, "defender", ran=True)
    move(ax, path(d, ids["runner"], f, a), "runner", ran=True)
    ball_move(ax, [d["passer"], target], shrink_start=st.key_radius(), shrink_end=st.BALL_D / 2 + 0.8)
    u = unit((target[0] - k["runner"][0], target[1] - k["runner"][1]))
    move(ax, [k["runner"], (target[0] - 0.8 * u[0], target[1] - 0.8 * u[1])], "runner", ran=False)
    for role, xy in k.items():
        player(ax, xy, role)
    ball(ax, target)
    mid = ((d["passer"][0] + target[0]) / 2, (d["passer"][1] + target[1]) / 2)
    label(ax, (mid[0] + 1.2, mid[1] - 1.8), "Pass")


# ------------------------------------------------------------------------------------------- layout

def window_around(points, pad_x, pad_y, aspect, goal=True):
    """A metres window around `points` (and the goal mouth) of the given width/height."""
    xs = [p[0] for p in points] + ([GOAL_X + 1.0] if goal else [])
    ys = [p[1] for p in points]
    x0, x1 = min(xs) - pad_x, max(xs) + (0.0 if goal else pad_x)
    y0, y1 = min(ys) - pad_y, max(ys) + pad_y
    w, h = x1 - x0, y1 - y0
    if w / h < aspect and goal:                        # widen to the left (the goal stays at the edge)
        x0 = x1 - h * aspect
    elif w / h < aspect:                               # widen about the middle
        c, w = (x0 + x1) / 2, h * aspect
        x0, x1 = c - w / 2, c + w / 2
    else:                                             # heighten about the middle
        c, h = (y0 + y1) / 2, w / aspect
        y0, y1 = c - h / 2, c + h / 2
    return (x0, x1), (y0, y1)


def titles(fig, W, H, box, title):
    x, y, w, h = box                                   # inches, the panel's axes
    fig.text(x / W, (y + h + 0.07) / H, title, fontsize=st.FS_PANEL, fontweight="bold", color=st.INK,
             ha="left", va="bottom")


def fig_line(fig, W, H, a, b, head=False):
    """The tree's connectors, in figure inches."""
    if head:
        st.fig_arrow(fig, (a[0] / W, a[1] / H), (b[0] / W, b[1] / H))
    else:
        fig.add_artist(matplotlib.lines.Line2D([a[0] / W, b[0] / W], [a[1] / H, b[1] / H], color=st.FAINT,
                                               lw=0.9, transform=fig.transFigure, solid_capstyle="butt"))


TEXT = {
    "now": "Current state",
    "f1": "Future 1 — Follow the runner",
    "f2": "Future 2 — Stay with the ball carrier",
}


def main() -> None:
    global OFFSIDE_LINE, LABEL_FS
    args = parse_args()
    OFFSIDE_LINE = None if args.offside_line == "none" else args.offside_line
    fonts = st.use_font()
    d = load(args)
    ids, f = d["ids"], d["dilemma"]
    # one window for every panel (same place, same scale), holding the offside-line defender at the pass
    shown = (path(d, ids["runner"], d["before"], d["shot"]) + path(d, ids["beneficiary"], d["before"], d["shot"])
             + [d["through_target"], d["passer"], *key_positions(d, f).values(), (GOAL_X, -POST), (GOAL_X, POST),
                d["at"](f, d["line_id"])])
    win = window_around(shown, 2.0, 1.8, ASPECT)
    W, m = 7.2, 0.08
    foot = 0.0 if args.no_match_line else 0.22       # the white strip under the panels for the match line

    def axes_at(fig, H, boxes):
        return {n: fig.add_axes((b[0] / W, b[1] / H, b[2] / W, b[3] / H)) for n, b in boxes.items()}

    if args.layout == "tree":                          # the current state on top, forking to the two futures
        wn = 4.3
        hn = wn / ASPECT
        wf = (W - 2 * m - 0.25) / 2
        hf = wf / ASPECT
        top_now, top_f, fork = 0.3, 0.3, 0.5
        H = foot + m + hf + top_f + fork + hn + top_now + 0.02
        yn = H - top_now - hn
        yf = foot + m
        boxes = {"now": ((W - wn) / 2, yn, wn, hn), "f1": (m, yf, wf, hf), "f2": (W - m - wf, yf, wf, hf)}
        fig = plt.figure(figsize=(W, H), facecolor=st.PAGE)
        axes = axes_at(fig, H, boxes)
        bar = yn - fork * 0.45
        fig_line(fig, W, H, (W / 2, yn - 0.05), (W / 2, bar))
        cx1, cx2 = m + wf / 2, W - m - wf / 2
        fig_line(fig, W, H, (cx1, bar), (cx2, bar))
        for cx in (cx1, cx2):
            fig_line(fig, W, H, (cx, bar), (cx, yf + hf + top_f + 0.02), head=True)
    else:                                              # one row: the current state -> {Future 1 over Future 2}
        LABEL_FS = st.FS_NOTE
        wf, gap = 2.3, 0.45
        wn = W - 2 * m - gap - wf
        hn, hf = wn / ASPECT, wf / ASPECT
        top_now, top_f = 0.3, 0.3
        H = foot + m + max(hn + top_now, 2 * (hf + top_f) + 0.12) + 0.02
        xf = W - m - wf
        yn = foot + (H - foot - top_now - hn + m) / 2
        boxes = {"now": (m, yn, wn, hn), "f1": (xf, H - top_f - hf - 0.02, wf, hf), "f2": (xf, foot + m, wf, hf)}
        fig = plt.figure(figsize=(W, H), facecolor=st.PAGE)
        axes = axes_at(fig, H, boxes)
        for n in ("f1", "f2"):
            fig_line(fig, W, H, (m + wn + 0.05, yn + hn / 2), (xf - 0.05, boxes[n][1] + hf / 2), head=True)
    for n, box in boxes.items():
        title = TEXT[n]
        if args.layout == "row" and n != "now":       # narrow futures: the answer in a word, as in the tree
            title = title.split(" — ")[0] + " — " + ("Follow" if n == "f1" else "Stay")
        titles(fig, W, H, box, title)
    panel_current(axes["now"], d, win)
    panel_follow(axes["f1"], d, win)
    panel_stay(axes["f2"], d, win)
    if not args.no_match_line:                         # the scene's start (the solver's t = 0), once
        note = st.match_line(args.code, d["start"])
        st.match_note(fig, note, W - m, 0.07)
        print(f"match line: {note}")
    st.check_text(fig)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=600, facecolor=fig.get_facecolor())
    fig.savefig(args.output.with_suffix(".pdf"), facecolor=fig.get_facecolor())
    print(f"font: {fonts}")
    print(f"→ {args.output} (+ .pdf) · {W:.1f} x {H:.2f} in")


if __name__ == "__main__":
    main()
