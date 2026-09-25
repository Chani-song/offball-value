#!/usr/bin/env python3
"""The solved play, told the way a video explainer does: move, stop, explain, move.

Each scene has two views behind one button, never drawn over each other:

  솔버의 답   the solver's own play of the game, on the real pitch. It plays like
             the real clip, but at each moment the defender decides -- 0, 1 and
             2 s -- it stops, shows that moment's choice as arrows named in
             football terms with their probabilities, and a caption saying
             whether he is torn; then it moves on. Torn moments hold longer.
             At the end the ball travels with the pass and the caption says to
             whom. The play shown is the MOST LIKELY line -- at every decision
             both sides take the action their policy weights most -- not a
             random draw: a draw once landed on a 23% move under a caption
             naming a 36% one, and the page seemed to contradict itself.
             While moving, a small arrow on the defender shows the direction he
             chose: acceleration is capped in the model (3.8, or agile), so a
             defender already running one way takes over a second to turn, and
             without the arrow that drift reads as ignoring his own choice.
  실제 움직임 the real tracking clip, untouched.

Why stop at every decision and not only the first: the dilemma often arises
after the start. Over the 167 2v1 scenes the defender is torn at the start in
60, and at some decision along the play in 84; for 31 of those the moment he is
most torn is 1 s in, for 7 it is 2 s in.

Works for both studies: 2v1 (andrew-passer2on1 -- the carrier is the
beneficiary, so "toward the ball" covers him) and 3v1 (andrew-fixedpasser --
a scripted passer, a runner and a separate beneficiary). Background defenders
move on their real tracks; attackers not in the game stay faint and still.
"""

from __future__ import annotations

import argparse
import json
import math
from html import escape
from pathlib import Path

import pandas as pd

from defensive_positioning.models import GameConfig

from offball_value.stage3_read import modal_path, world_direction
from render_triple_overview import (
    CSS, NAV_BAR, PLAY_JS, R_BALL, R_PLAYER, W, H,
    pitch_markings, player_bar, trail, xy,
)

GLOW = {"runner": "var(--glow-runner)", "defender": "var(--glow-defender)",
        "beneficiary": "var(--glow-benef)"}
TAG = {"runner": "러너", "defender": "반응 수비수", "beneficiary": "수혜자"}
FAMILY_KR = {"ground": "땅볼 패스", "driven": "강한 패스", "lofted": "띄운 패스"}
TO_KR = {"runner": "러너", "beneficiary": "수혜자"}
SPLIT = 0.2          # torn when the likeliest target has under 80%
NEAR = 0.05          # "almost decided" between 95% and 80%


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--solved", type=Path, required=True)
    p.add_argument("--build", type=Path,
                   default=Path("/work/hdd/bbmr/kseo1/offball-out/v7_r9_ssac"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--title", default="솔버의 답")
    p.add_argument("--compare", type=Path, default=None)
    p.add_argument("--compare-label", default="배경 수비수 없을 때")
    p.add_argument("--top", type=int, default=30)
    p.add_argument("--flat", type=int, default=8)
    p.add_argument("--trail-stride", type=int, default=4)
    return p.parse_args()


def c2p(pt, flip):
    return xy(float(pt[0]) - 52.5, float(pt[1]) - 34.0, flip)


def tag_for(split: float) -> tuple[str, str]:
    if split >= SPLIT:
        return "갈림 · 딜레마", "split"
    if split >= NEAR:
        return "거의 확정", "near"
    return "확정", "sure"


# --------------------------------------------------------------------------

def real_svg(game, roles, stride):
    """The tracking clip; roles maps player id -> runner / defender / beneficiary."""
    frames = game["background_frames"]
    onset = game["onset_frame_id"]
    flip = int(game.get("attacking_direction", 1)) < 0
    at = min(frames, key=lambda f: abs(f["frame_id"] - onset))
    att = str(game["attacking_team_id"])
    out = [f"<svg viewBox='0 0 {W:.0f} {H:.0f}' class=pitch role=img aria-label='실제 움직임'>",
           f"<rect width='{W:.0f}' height='{H:.0f}' rx='8' fill='var(--pitch-fill)'/>",
           pitch_markings(flip), "<g class=trails>"]
    for pid, role in roles.items():
        line = trail(frames, pid, flip, stride)
        if line:
            out.append(f"<g style='color:{GLOW[role]}'>{line}</g>")
    out.append("</g>")
    order = [str(p[0]) for p in at["players"]]
    for slot, p in enumerate(at["players"]):
        pid, team, x, y, name = str(p[0]), str(p[1]), p[2], p[3], p[4]
        px, py = xy(x, y, flip)
        fill = "var(--team-att)" if team == att else "var(--team-def)"
        role = roles.get(pid)
        title = f"{name} · {'공격' if team == att else '수비'}" + (f" · {TAG[role]}" if role else "")
        glow = (f"<circle class='gl' data-i='{slot}' cx='{px:.1f}' cy='{py:.1f}' "
                f"r='{R_PLAYER + 2.5:.1f}' fill='none' stroke='{GLOW[role]}' stroke-width='3' "
                f"style='filter:drop-shadow(0 0 5px {GLOW[role]})'/>") if role else ""
        out.append(glow + f"<circle class='pl' data-i='{slot}' data-n=\'{escape(title)}\' "
                   f"cx='{px:.1f}' cy='{py:.1f}' r='{R_PLAYER}' fill='{fill}' "
                   f"stroke='var(--dot-line)' stroke-width='1'/>")
    bx, by = xy(at["ball"][0], at["ball"][1], flip)
    out.append(f"<circle class=ball cx='{bx:.1f}' cy='{by:.1f}' r='{R_BALL}' fill='#fff' "
               f"stroke='#111' stroke-width='1.4' pointer-events='none'/></svg>")
    positions, ball, clock = [], [], []
    for f in frames:
        by_id = {str(q[0]): (q[2], q[3]) for q in f["players"]}
        positions.append([[round(v, 1) for v in xy(*by_id[pid], flip)] if pid in by_id else None
                          for pid in order])
        ball.append([round(v, 1) for v in xy(f["ball"][0], f["ball"][1], flip)])
        clock.append(round(float(f["relative_time_s"]), 2))
    oi = min(range(len(frames)), key=lambda i: abs(frames[i]["frame_id"] - onset))
    return "".join(out), {"p": positions, "b": ball, "t": clock, "onset": oi}


def solver_svg(game, state, path, background):
    """The solved play; its payload drives the stop-and-explain playback."""
    flip = int(game.get("attacking_direction", 1)) < 0
    ad = int(state["scenario"]["attack_direction"])
    frames = game["background_frames"]
    at = min(frames, key=lambda f: abs(f["frame_id"] - game["onset_frame_id"]))
    att = str(game["attacking_team_id"])
    roll, kind = path["rollout"], path["kind"]
    if kind == "3v1":
        bodies = (("p", "passer", None, "볼"), ("r", "runner", "runner", "러너"),
                  ("b", "beneficiary", "beneficiary", "수혜자"), ("d", "defender", "defender", "수비"))
        ball_key = "p"
    else:
        bodies = (("c", "carrier", "beneficiary", "볼·수혜자"), ("r", "receiver", "runner", "러너"),
                  ("d", "defender", "defender", "수비"))
        ball_key = "c"
    steps = [{"t": round(float(s["time"]), 3),
              **{k: [round(v, 1) for v in c2p(s[src], flip)] for k, src, _, _ in bodies}}
             for s in roll]

    out = [f"<svg viewBox='0 0 {W:.0f} {H:.0f}' class=spitch role=img aria-label='솔버의 답'>",
           f"<rect width='{W:.0f}' height='{H:.0f}' rx='8' fill='var(--pitch-fill)'/>",
           pitch_markings(flip)]
    game_ids = set()
    prov = state.get("_provenance", {})
    for key in ("runner_id", "defender_id", "carrier_id", "beneficiary_id"):
        if prov.get(key):
            game_ids.add(str(prov[key]))
    bg_ids = [str(i) for i in (background or {}).get("ids", [])]
    for p in at["players"]:
        if str(p[0]) in game_ids or str(p[0]) in set(bg_ids):
            continue
        px, py = xy(p[2], p[3], flip)
        out.append(f"<circle cx='{px:.1f}' cy='{py:.1f}' r='{R_PLAYER - 1.5}' "
                   f"fill='{'var(--team-att)' if str(p[1]) == att else 'var(--team-def)'}' opacity='.16'/>")
    bg_payload = None
    if bg_ids:
        bg_px = [[[round(v, 1) for v in c2p(pt, flip)] for pt in row] for row in background["positions"]]
        names = background.get("names") or [""] * len(bg_ids)
        for j, (x0, y0) in enumerate(bg_px[0]):
            out.append(f"<circle class=bgd data-j='{j}' cx='{x0}' cy='{y0}' r='{R_PLAYER - 1}' "
                       f"fill='var(--team-def)' opacity='.55' stroke='var(--dot-line)'>"
                       f"<title>{escape(str(names[j]))} · 배경 수비수 (실제 경로)</title></circle>")
        bg_payload = {"t": [float(t) for t in background["times_requested"]], "p": bg_px}

    out.append(f"<rect class=freeze width='{W:.0f}' height='{H:.0f}' rx='8' fill='#000' "
               f"opacity='0' pointer-events='none'/>")
    # one overlay per decision: an arrow per football target, drawn from where
    # the defender stands AT that moment, in the probability-weighted direction
    # of the compass moves that lead there
    points = []
    for i, q in enumerate(path["points"]):
        snap = roll[q["step"]]
        dm = snap["defender"]
        d0 = c2p(dm, flip)
        by_name: dict[str, list] = {}
        for k, name, prob in q["choices"]:
            by_name.setdefault(name, []).append((k, prob))
        g = [f"<g class=dp data-i='{i}' style='display:none'>"]
        for name, prob in q["merged"]:
            vx = sum(world_direction(k, ad)[0] * pr for k, pr in by_name[name])
            vy = sum(world_direction(k, ad)[1] * pr for k, pr in by_name[name])
            norm = math.hypot(vx, vy)
            if norm < 1e-9:
                g.append(f"<g class=opt data-name='{escape(name)}'>"
                         f"<circle cx='{d0[0]:.1f}' cy='{d0[1]:.1f}' r='{R_PLAYER + 6 + 10 * prob:.1f}' "
                         f"fill='none' stroke='var(--solver)' stroke-width='3' stroke-dasharray='5 3'/>"
                         f"<text x='{d0[0]:.1f}' y='{d0[1] + R_PLAYER + 26 + 10 * prob:.1f}' class=cl "
                         f"text-anchor='middle'>{escape(name)} {prob:.0%}</text></g>")
                continue
            ux, uy = vx / norm, vy / norm
            length = 4.0 + 10.0 * prob
            tip = c2p((dm[0] + ux * length, dm[1] + uy * length), flip)
            lab = c2p((dm[0] + ux * (length + 3.5), dm[1] + uy * (length + 3.5)), flip)
            g.append(f"<g class=opt data-name='{escape(name)}'>"
                     f"<line x1='{d0[0]:.1f}' y1='{d0[1]:.1f}' x2='{tip[0]:.1f}' y2='{tip[1]:.1f}' "
                     f"stroke='var(--solver)' stroke-width='{3 + 4 * prob:.1f}' stroke-linecap='round' "
                     f"marker-end='url(#arrow)'/><text x='{lab[0]:.1f}' y='{lab[1] + 4:.1f}' class=cl "
                     f"text-anchor='middle'>{escape(name)} {prob:.0%}</text></g>")
        g.append("</g>")
        out.append("".join(g))
        # the move he picks, and whether momentum will carry him the other way
        # for a while: acceleration is capped in the model (3.8 m/s^2, or the
        # agile limits), so a defender running away from his choice keeps
        # drifting before he turns, and one told to stop keeps sliding
        cu = world_direction(q["chosen"], ad)
        vel = snap.get("defender_velocity") or [0.0, 0.0]
        speed = math.hypot(vel[0], vel[1])
        against = (cu != (0.0, 0.0) and speed > 2.0
                   and (cu[0] * vel[0] + cu[1] * vel[1]) / speed < -0.3)
        tag, cls = tag_for(q["split"])
        when = f"{round(q['t'], 1):g}초"
        points.append({"t": q["t"], "split": q["split"], "tag": tag, "cls": cls,
                       "chosen": q["chosen_name"],
                       "caption": f"⏸ {when} · 수비수가 결정하는 순간 — "
                                  + " · ".join(f"{n} {p:.0%}" for n, p in q["merged"]),
                       "decided": (f"{when} · 결정: {q['chosen_name']}"
                                   + (f"  (지금 {speed:.1f} m/s로 뛰던 중이라, 같은 방향으로 "
                                      "미끄러지면서 속도를 줄입니다)" if speed > 2.0 else "")
                                   if cu == (0.0, 0.0) else
                                   f"{when} · 결정: {q['chosen_name']} 쪽으로".replace("쪽 쪽", "쪽")
                                   + (f"  (지금 {speed:.1f} m/s로 반대쪽으로 뛰던 중이라, "
                                      "도는 데 시간이 걸려 한동안 반대로 미끄러집니다)" if against else ""))})

    end = path["end"]
    ev = end["event"]
    last = steps[-1]
    end_payload = {"event": ev, "t": steps[-1]["t"]}
    if ev == "release" and end.get("target"):
        tgt = [round(v, 1) for v in c2p(end["target"], flip)]
        to = TO_KR.get(end.get("to") or "runner", "러너")
        fam = FAMILY_KR.get(end.get("family"), end.get("family") or "")
        end_payload.update(target=tgt, caption=f"{round(last['t'], 1):g}초 · 패스 → {to} ({fam})")
        src = last[ball_key]
        out.append(f"<g class=spass style='opacity:0'><line x1='{src[0]}' y1='{src[1]}' "
                   f"x2='{tgt[0]}' y2='{tgt[1]}' stroke='#fff' stroke-width='2.6' "
                   f"stroke-dasharray='6 4' marker-end='url(#arrow-w)'/></g>")
    elif ev == "retain":
        end_payload["caption"] = f"{round(last['t'], 1):g}초 · 볼 소유자가 계속 몰고 감 (패스 없음)"
    elif ev == "no_pass":
        end_payload["caption"] = f"{round(last['t'], 1):g}초 · 줄 곳이 없어 패스 없음"
    else:
        end_payload["caption"] = f"{round(last['t'], 1):g}초 · 수비수가 태클로 공을 뺏음"

    for key, _, role, label in bodies:
        x0, y0 = steps[0][key]
        fill = "var(--team-def)" if role == "defender" else "var(--team-att)"
        ring = (f"<circle r='{R_PLAYER + 2.5:.1f}' fill='none' stroke='{GLOW[role]}' stroke-width='3' "
                f"style='filter:drop-shadow(0 0 5px {GLOW[role]})'/>") if role else ""
        out.append(f"<g class='sb sb-{key}' transform='translate({x0} {y0})'>{ring}"
                   f"<circle r='{R_PLAYER}' fill='{fill}' stroke='var(--dot-line)'/>"
                   f"<text y='{-R_PLAYER - 7:.1f}' class=bl text-anchor='middle'>{escape(label)}</text></g>")
    bx, by = steps[0][ball_key]
    out.append(f"<circle class=sball cx='{bx + 6}' cy='{by + 5}' r='{R_BALL}' fill='#fff' "
               f"stroke='#111' stroke-width='1.4' pointer-events='none'/>")
    out.append("</svg>")
    return "".join(out), {"kind": kind, "steps": steps, "ballKey": ball_key, "points": points,
                          "end": end_payload, "bg": bg_payload}


# --------------------------------------------------------------------------

EXTRA_CSS = """
:root{--solver:#c77dff}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme=light])){--solver:#d9a3ff}}
:root[data-theme=dark]{--solver:#d9a3ff}
.head{margin:4px 0 10px;padding:12px 14px;border-radius:8px;background:var(--page);
 border:1px solid var(--border)}
.head .who{font-size:12.5px;color:var(--text-secondary);margin-bottom:8px}
.head .who b{color:var(--text-primary)}
.head .lead{font-size:13px;color:var(--text-secondary);margin-bottom:6px}
.chips{display:flex;flex-wrap:wrap;gap:6px;align-items:center}
.chip{display:inline-flex;flex-direction:column;gap:1px;padding:6px 10px;border-radius:8px;
 border:1px solid var(--border);background:var(--surface);font-size:13px;line-height:1.35}
.chip .when{font-size:11px;color:var(--muted);font-weight:600}
.chip.split{border-color:var(--solver);background:color-mix(in srgb,var(--solver) 16%,transparent)}
.chip.split .when{color:var(--solver)}
.chip.end{border-style:dashed}
.arrow-sep{color:var(--muted);font-size:13px}
.head .more{margin-top:8px;font-size:12px;color:var(--muted)}
.modes{display:flex;gap:0;margin:0 0 8px;border:1px solid var(--border);border-radius:8px;
 overflow:hidden;width:max-content}
.modes button{font:inherit;font-size:13.5px;padding:7px 16px;border:0;background:var(--surface);
 color:var(--text-secondary);cursor:pointer}
.modes button[aria-pressed=true]{background:var(--solver);color:#1a0b26;font-weight:600}
.view{position:relative}
.card .sview,.card .sbar,.card .shelp{display:none}
.card[data-mode=solver] .sview,.card[data-mode=solver] .shelp{display:block}
.card[data-mode=solver] .sbar{display:flex}
.card[data-mode=solver] .rview,.card[data-mode=solver] .play,.card[data-mode=solver] .rhelp{display:none}
.spitch{display:block;width:100%;height:auto}
/* the explainer caption, over the pitch like a subtitle */
.scap{position:absolute;left:10px;right:10px;top:10px;display:flex;gap:10px;align-items:center;
 padding:9px 13px;border-radius:9px;background:rgba(18,8,28,.84);color:#fff;font-size:15px;
 font-weight:600;line-height:1.35;opacity:0;transition:opacity .2s;pointer-events:none}
.scap.on{opacity:1}
.scap-g{flex:none;padding:2px 9px;border-radius:6px;font-size:12.5px;font-weight:700}
.scap-g.split{background:var(--solver);color:#1a0b26}
.scap-g.near{background:#ffd166;color:#2b2100}
.scap-g.sure{background:rgba(255,255,255,.18);color:#fff}
.cl{font:700 12.5px system-ui,sans-serif;fill:#fff;paint-order:stroke;stroke:rgba(40,10,60,.9);stroke-width:4px}
.bl{font:700 11px system-ui,sans-serif;fill:#fff;paint-order:stroke;stroke:rgba(0,0,0,.7);stroke-width:3px}
.sbar{align-items:center;gap:9px;margin:7px 2px 0}
.sp-btn{flex:none;width:32px;height:32px;padding:0;font-size:13px;border:1px solid var(--border);
 border-radius:6px;background:var(--page);color:var(--text-primary);cursor:pointer;line-height:1}
.sp-scrub{flex:1;min-width:0;accent-color:var(--solver);height:24px}
.sp-clock{flex:none;width:52px;text-align:right;font-size:12px;color:var(--text-secondary);
 font-variant-numeric:tabular-nums}
.help{font-size:12.5px;color:var(--text-secondary);margin:6px 2px 0;line-height:1.55}
.qrow{display:flex;align-items:center;gap:10px;padding:10px 2px 4px;border-top:1px solid var(--border);
 margin-top:10px;flex-wrap:wrap}
.qrow .ask{flex:1 1 210px;font-size:13.5px;font-weight:600}
"""

SOLVER_JS = """
<script>
(function () {
  var S = __SOLVER__, rate = 0.5, TICK = 40;
  var HOLD_SPLIT = 3600, HOLD_OTHER = 1600, DECIDE_MS = 1300, FLY_MS = 800, END_MS = 3200;
  var stops = [];
  function stopAll() { stops.forEach(function (f) { f(); }); }
  function lerp(a, b, f) { return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f]; }
  function setup(card) {
    var d = S[card.dataset.card]; if (!d) return;
    var svg = card.querySelector('svg.spitch');
    var keys = Object.keys(d.steps[0]).filter(function (k) { return k !== 't'; });
    var body = {}; keys.forEach(function (k) { body[k] = svg.querySelector('.sb-' + k); });
    var ball = svg.querySelector('.sball'), pass = svg.querySelector('.spass');
    var dps = svg.querySelectorAll('.dp'), bgd = svg.querySelectorAll('circle.bgd');
    var freeze = svg.querySelector('.freeze');
    var cap = card.querySelector('.scap'), capt = cap.querySelector('.scap-t'), capg = cap.querySelector('.scap-g');
    var scrub = card.querySelector('.sp-scrub'), btn = card.querySelector('.sp-btn'), clk = card.querySelector('.sp-clock');
    var st = d.steps, T = st[st.length - 1].t;
    scrub.max = Math.max(1, Math.round(T * 100));
    function pos(t) {
      var i = 0; for (var j = 0; j < st.length; j++) if (st[j].t <= t + 1e-6) i = j;
      var k = Math.min(i + 1, st.length - 1);
      var f = st[k].t > st[i].t ? Math.max(0, Math.min(1, (t - st[i].t) / (st[k].t - st[i].t))) : 0;
      var o = {}; keys.forEach(function (key) { o[key] = lerp(st[i][key], st[k][key], f); }); return o;
    }
    function drawBg(t) {
      if (!d.bg || !bgd.length) return;
      var bt = d.bg.t, i0 = 0; for (var j = 0; j < bt.length; j++) if (bt[j] <= t + 1e-6) i0 = j;
      var i1 = Math.min(i0 + 1, bt.length - 1);
      var g = bt[i1] > bt[i0] ? Math.max(0, Math.min(1, (t - bt[i0]) / (bt[i1] - bt[i0]))) : 0;
      bgd.forEach(function (c) { var p = lerp(d.bg.p[i0][+c.dataset.j], d.bg.p[i1][+c.dataset.j], g);
        c.setAttribute('cx', p[0]); c.setAttribute('cy', p[1]); });
    }
    function draw(t, fly) {
      var p = pos(t);
      keys.forEach(function (k) { body[k].setAttribute('transform', 'translate(' + p[k][0] + ' ' + p[k][1] + ')'); });
      var b = p[d.ballKey]; b = [b[0] + 6, b[1] + 5];
      if (fly != null && d.end.target) b = lerp(b, d.end.target, fly);
      ball.setAttribute('cx', b[0]); ball.setAttribute('cy', b[1]);
      drawBg(t);
      clk.textContent = t.toFixed(2) + 's'; scrub.value = Math.round(t * 100);
    }
    function caption(text, tag, cls) {
      capt.textContent = text; capg.textContent = tag || ''; capg.className = 'scap-g ' + (cls || '');
      capg.style.display = tag ? '' : 'none'; cap.classList.add('on');
    }
    function frozen(on) { if (freeze) freeze.setAttribute('opacity', on ? '0.22' : '0'); }
    function showPoint(i, decided) {
      dps.forEach(function (g) { g.style.display = (+g.dataset.i === i) ? '' : 'none'; });
      if (pass) pass.style.opacity = 0;
      var q = d.points[i], g = dps[i];
      if (g) g.querySelectorAll('.opt').forEach(function (o) {
        o.style.transition = 'opacity .35s';
        o.style.opacity = (!decided || o.dataset.name === q.chosen) ? 1 : 0.1;
      });
      frozen(true);
      if (decided) caption(q.decided); else caption(q.caption, q.tag, q.cls);
    }
    function showEnd() {
      dps.forEach(function (g) { g.style.display = 'none'; });
      if (pass) pass.style.opacity = 1;
      frozen(true); caption(d.end.caption);
    }
    function clear() { cap.classList.remove('on'); frozen(false);
      dps.forEach(function (g) { g.style.display = 'none'; }); if (pass) pass.style.opacity = 0; }
    var timer = null, mode = 'idle', t = 0, next = 0, until = 0, flyAt = 0, cur = -1;
    function stop() { if (timer) { clearInterval(timer); timer = null; } btn.textContent = '▶'; }
    stops.push(stop);
    // a turn: freeze and show the choice; then only the chosen move stays;
    // then everything clears and play resumes with that decision
    function holdAt(i) {
      mode = 'hold'; cur = i; showPoint(i, false); next = i + 1;
      until = performance.now() + (d.points[i].cls === 'split' ? HOLD_SPLIT : HOLD_OTHER);
    }
    function reachEnd(now) {
      if (d.end.event === 'release' && d.end.target) { mode = 'fly'; flyAt = now; clear(); }
      else { mode = 'end'; showEnd(); until = now + END_MS; }
    }
    function tick() {
      var now = performance.now();
      if (mode === 'hold') { if (now >= until) { mode = 'decide'; showPoint(cur, true); until = now + DECIDE_MS; } return; }
      if (mode === 'decide') { if (now >= until) { mode = 'move'; clear(); } return; }
      if (mode === 'move') {
        var goal = next < d.points.length ? d.points[next].t : T;
        t = Math.min(goal, t + TICK / 1000 * rate); draw(t);
        if (t >= goal - 1e-9) { if (next < d.points.length) holdAt(next); else reachEnd(now); }
        return;
      }
      if (mode === 'fly') {
        var f = Math.min(1, (now - flyAt) / FLY_MS); draw(T, f);
        if (f >= 1) { mode = 'end'; showEnd(); until = now + END_MS; }
        return;
      }
      if (mode === 'end' && now >= until) { mode = 'done'; stop(); }
    }
    btn.addEventListener('click', function () {
      if (timer) { stop(); return; }
      if (mode === 'idle' || mode === 'done' || mode === 'end') {
        t = 0; next = 0; cur = -1; draw(0);
        if (d.points.length) holdAt(0); else { mode = 'move'; clear(); }
      }
      btn.textContent = '❚❚'; timer = setInterval(tick, TICK);
    });
    scrub.addEventListener('input', function () {
      stop(); mode = 'idle'; t = +scrub.value / 100; draw(t);
      var hit = -1; d.points.forEach(function (q, i) { if (Math.abs(q.t - t) < 0.03) hit = i; });
      if (hit >= 0) showPoint(hit, false); else if (t >= T - 1e-6) { draw(T, 1); showEnd(); } else clear();
    });
    draw(0); if (d.points.length) showPoint(0, false);
  }
  window.addEventListener('DOMContentLoaded', function () {
    document.querySelectorAll('.stage .card[data-card]').forEach(setup);
    document.querySelectorAll('.modes button').forEach(function (b) {
      b.addEventListener('click', function () {
        var card = b.closest('.card'); stopAll();
        var pr = card.querySelector('.play .pp'); if (pr && pr.textContent !== '▶') pr.click();
        card.dataset.mode = b.dataset.m;
        card.querySelectorAll('.modes button').forEach(function (x) { x.setAttribute('aria-pressed', x === b ? 'true' : 'false'); });
      });
    });
    document.addEventListener('click', function (ev) { if (ev.target.closest && ev.target.closest('.prev, .next')) stopAll(); });
    document.addEventListener('keydown', function (ev) { if (ev.key === 'ArrowLeft' || ev.key === 'ArrowRight') stopAll(); });
    var sel = document.querySelector('.rate');
    if (sel) { rate = parseFloat(sel.value) || 0.5; sel.addEventListener('change', function () { rate = parseFloat(sel.value) || 0.5; }); }
  });
})();
</script>
"""

REVIEW_JS = """
<script>
(function () {
  var KEY = '__KEY__', data = {};
  function load(){ try{ data=JSON.parse(localStorage.getItem(KEY)||'{}')||{}; }catch(e){ data={}; } }
  function save(){ try{ localStorage.setItem(KEY, JSON.stringify(data)); }catch(e){} }
  function cards(){ return document.querySelectorAll('.stage .card'); }
  function paint(){
    var done=0;
    cards().forEach(function(c){ var e=data[c.dataset.card]||{}; if(e.v) done++;
      c.querySelectorAll('button.v').forEach(function(b){ b.setAttribute('aria-pressed', e.v===b.dataset.v?'true':'false'); });
      var n=c.querySelector('input.note'); if(n && e.note!==undefined) n.value=e.note; });
    var el=document.querySelector('.prog'); if(el) el.textContent=done+' / '+cards().length+' 판정함';
  }
  document.addEventListener('click', function(ev){ var b=ev.target.closest && ev.target.closest('button.v'); if(!b) return;
    var e=data[b.dataset.k]=data[b.dataset.k]||{}; e.v=(e.v===b.dataset.v)?'':b.dataset.v; save(); paint(); });
  document.addEventListener('input', function(ev){ if(!ev.target.matches||!ev.target.matches('input.note')) return;
    var e=data[ev.target.dataset.k]=data[ev.target.dataset.k]||{}; e.note=ev.target.value; save(); });
  function cell(v){ v=(v===undefined||v===null)?'':String(v); return /[",\\n]/.test(v)?'"'+v.replace(/"/g,'""')+'"':v; }
  window.addEventListener('DOMContentLoaded', function(){
    load(); paint();
    var ex=document.querySelector('.export');
    if(ex) ex.addEventListener('click', function(){
      var L=['state_index,match_id,onset_frame_id,runner_id,defender_id,path_max_split,verdict,note'];
      cards().forEach(function(c){ var e=data[c.dataset.card]||{}; if(!e.v&&!e.note) return;
        L.push([c.dataset.idx,c.dataset.match,c.dataset.onset,c.dataset.runner,c.dataset.defender,
                c.dataset.split,e.v||'',e.note||''].map(cell).join(',')); });
      var blob=new Blob([L.join('\\n')+'\\n'],{type:'text/csv;charset=utf-8'});
      var a=document.createElement('a'); a.href=URL.createObjectURL(blob); a.download='__KEY__.csv'; a.click(); URL.revokeObjectURL(a.href);
    });
    var cl=document.querySelector('.clear');
    if(cl) cl.addEventListener('click', function(){ if(!confirm('판정을 모두 지웁니다. 계속할까요?')) return;
      data={}; save(); paint(); document.querySelectorAll('input.note').forEach(function(i){ i.value=''; }); });
  });
})();
</script>
"""


def chip(q_t, merged_list, split):
    tag, cls = tag_for(split)
    body = " · ".join(f"{escape(n)} {p:.0%}" for n, p in merged_list)
    return (f"<span class='chip {cls}'><span class=when>{round(q_t, 1):g}초 · {tag}</span>"
            f"<span>{body}</span></span>")


def main() -> None:
    args = parse_args()
    manifest = json.loads((args.solved / "manifest.json").read_text())
    config = GameConfig(**{k: v for k, v in manifest["config"].items()
                           if k in ("steps", "step_seconds", "physics_step")})
    f = pd.read_csv(args.results)
    f = f.sort_values(["path_max_split", "fsplit0"], ascending=False)
    top = f.head(args.top)
    flat = f[f["path_max_split"] == 0].head(args.flat)
    picked = pd.concat([top, flat.loc[~flat.index.isin(top.index)]])
    starting = {int(r["index"]): r for r in
                json.loads((args.solved / "starting_states.json").read_text())["states"]}
    other = pd.read_csv(args.compare).set_index("index") if args.compare else None
    kind = str(f["kind"].iloc[0]) if "kind" in f else "2v1"
    phys = manifest.get("physics") or {"name": "andrew"}
    if phys.get("name") == "andrew":
        motion_note = "모델에서 선수의 가속은 최대 3.8 m/s²입니다(준현님 모델)."
    else:
        motion_note = (f"모델에서 선수는 속도를 올릴 때 최대 {phys['speed_up']:g}, 줄일 때 {phys['braking']:g}, "
                       f"방향을 틀 때 {phys['turning']:g} m/s²이고, {phys['plant_min_speed']:g} m/s 이상으로 뛰다 "
                       f"{phys['plant_min_angle_deg']:g}° 이상 꺾을 땐 {phys['plant_braking']:g} m/s²로 급제동해 "
                       f"멈췄다가 다시 출발할 수 있습니다(예전 가속). 수비수는 런 시작 {phys['defender_delay_s']:g}초 "
                       "뒤부터 반응합니다.")

    cards, clips, solver = [], {}, {}
    for n, r in enumerate(picked.itertuples(), 1):
        i = int(r.index)
        game = json.loads((args.build / r.scene_dir / "local_game_payoff_audits.json").read_text())[0]
        state = json.loads((args.solved / "states" / f"state_{i:03d}.json").read_text())
        record = starting[i]
        state["_provenance"] = record.get("provenance", {})
        path = modal_path(state, args.solved / "policies" / f"state_{i:03d}.npz", config,
                          passer_track=(record.get("passer") or {}).get("positions"),
                          physics=manifest.get("physics"))
        card_id = f"s3:{i}"
        prov = record.get("provenance", {})
        runner, defender = str(prov.get("runner_id")), str(prov.get("defender_id"))
        benef = str(prov.get("beneficiary_id") or prov.get("carrier_id"))
        rs, clip = real_svg(game, {runner: "runner", defender: "defender", benef: "beneficiary"},
                            args.trail_stride)
        clips[card_id] = clip
        ss, sd = solver_svg(game, state, path, record.get("background"))
        solver[card_id] = sd

        chips = [chip(q["t"], q["merged"], q["split"]) for q in path["points"]]
        end = sd["end"]["caption"].split("·", 1)[-1].strip()
        # a release before the last instant ends the game there: say so, or
        # a short solver clip reads as a truncated one
        end_t = float(path["end"]["t"])
        early = end_t < config.steps * config.step_seconds - 1e-9
        when = (f"{round(end_t, 1):g}초에 패스 · 게임 끝" if early and path["end"]["event"] == "release"
                else f"{round(end_t, 1):g}초 · 게임 끝" if early else "끝")
        if early:
            sd["end"]["caption"] += " — 패스가 나가 게임이 여기서 끝납니다"
        chips.append(f"<span class='chip end'><span class=when>{when}</span><span>{escape(end)}</span></span>")
        chips_html = "<span class=arrow-sep>→</span>".join(chips)
        who = (f"러너 <b>{escape(str(r.runner))}</b> · 수비수 <b>{escape(str(r.defender))}</b> · "
               + (f"수혜자 <b>{escape(str(r.beneficiary))}</b> · 볼 <b>{escape(str(r.carrier))}</b>"
                  if path["kind"] == "3v1" else f"볼(=수혜자) <b>{escape(str(r.carrier))}</b>"))
        more = f"공격 기댓값 {r.value:.2f}"
        if other is not None and i in other.index:
            o = other.loc[i]
            more += (f" · {escape(args.compare_label)}(시작 순간): {escape(str(o.root_top))} "
                     f"{1 - o.fsplit0:.0%}" + (f" · {escape(str(o.root_second))}" if o.fsplit0 > 0 else ""))
        head = (f"<div class=head><div class=who>{who}</div>"
                f"<div class=lead>솔버가 푼 경기의 가장 가능성 높은 흐름에서 수비수의 선택</div>"
                f"<div class=chips>{chips_html}</div><div class=more>{more}</div></div>")
        modes = ("<div class=modes><button type=button data-m=solver aria-pressed=true>솔버의 답</button>"
                 "<button type=button data-m=real aria-pressed=false>실제 움직임</button></div>")
        sview = (f"<div class='view sview'>{ss}<div class=scap><span class=scap-t></span>"
                 f"<span class=scap-g></span></div></div>")
        sbar = ("<div class=sbar><button type=button class=sp-btn aria-label='솔버 경기 재생'>▶</button>"
                "<input type=range class=sp-scrub min=0 max=0 value=0 step=1 aria-label='솔버 시간'>"
                "<span class=sp-clock>0.00s</span></div>")
        shelp = ("<div class='help shelp'>▶를 누르면 솔버가 푼 경기 중 <b>가장 가능성 높은 흐름</b>"
                 "(매 순간 양쪽이 확률이 가장 큰 선택을 함)이 재생됩니다. 수비수가 결정하는 순간"
                 "마다 화면이 <b>멈추고</b>(어두워짐) 그 순간 수비수가 어느 쪽으로 몇 %인지 보라 화살표와 자막으로 "
                 "보여줍니다. 이어서 <b>고른 쪽만 남고</b>, 표시가 사라진 뒤 그 결정대로 움직입니다. "
                 "갈리는 순간은 더 오래 멈춥니다. 마지막에 흰 공이 패스로 날아갑니다. "
                 "<b>화살표는 그 턴에 수비수가 목표로 하는 방향</b>(그쪽으로 전력 질주 속도를 목표로 함)이고, "
                 "고를 수 있는 건 동·서·남·북·멈추기 다섯 가지입니다. '러너 쪽' 같은 이름은 고른 방향이 러너·"
                 "수혜자·볼·골문 중 누구를 가장 가리키는지 보고 붙인 해석입니다. '멈추기(감속)'는 속도를 0으로 "
                 "줄이려는 선택입니다. " + motion_note +
                 " 반대로 뛰던 수비수는 고른 쪽으로 도는 데 시간이 걸리고, 멈추기를 골라도 한동안 미끄러집니다 — "
                 "그런 경우 자막에 적어 둡니다. 반투명 파란 점은 배경 수비수(결정 없이 실제 경로), 아주 흐린 점은 "
                 "게임에 없는 선수입니다.</div>")
        rview = f"<div class='view rview'>{rs}</div>"
        rhelp = ("<div class='help rhelp'>실제 경기 영상(런 시작 2초 전 ~ 3초 후). 빨강 = 러너 · "
                 "파랑 테두리 = 반응 수비수 · 노랑 = 수혜자.</div>")
        buttons = "".join(
            f"<button type=button class=v data-k='{card_id}' data-v='{v}'>{t}</button>"
            for v, t in (("ok", "딜레마 맞음"), ("bad", "아님"), ("meh", "애매")))
        cards.append(
            f"<div class=card data-clip='{card_id}' data-card='{card_id}' data-mode=solver "
            f"data-idx='{i}' data-match='{escape(str(r.match_id))}' data-onset='{int(r.onset)}' "
            f"data-runner='{escape(runner)}' data-defender='{escape(defender)}' "
            f"data-split='{r.path_max_split:.3f}'>"
            f"<div class=hd><span class=rank>#{n}</span>"
            f"<span class=match>{escape(str(r.match))} · f{int(r.onset)}</span></div>"
            f"{head}{modes}{sview}{sbar}{shelp}{rview}{player_bar()}{rhelp}"
            f"<div class=qrow><span class=ask>이 수비수, 정말 고민되는 상황인가?</span>{buttons}</div>"
            f"<input class=note data-k='{card_id}' placeholder='메모 (선택)'></div>")

    torn = int((f["path_max_split"] >= SPLIT).sum())
    at_start = int((f["fsplit0"] >= SPLIT).sum())
    later = f[f["path_max_split"] >= SPLIT]["path_split_at"].value_counts().sort_index()
    summary = (f"<div class=legend style='display:block;line-height:1.7'>"
               f"<b>{len(f)}장면</b> · 솔버가 푼 경기 흐름 중 수비수가 한 번이라도 갈린 장면 <b>{torn}</b> "
               f"(시작 순간에만 보면 {at_start}) · 가장 크게 갈린 순간: "
               + " · ".join(f"{round(t, 1):g}초 {c}" for t, c in later.items()) + "</div>")
    bar = ("<div class=bar><span class=prog>0 / 0 판정함</span>"
           "<button type=button class=export>CSV 내려받기</button>"
           "<button type=button class=clear>판정 지우기</button></div>")
    key = f"offball-story-{args.output.parent.name}"
    html = (
        "<!doctype html><html lang=ko><meta charset=utf-8>"
        "<meta name=viewport content='width=device-width,initial-scale=1'>"
        f"<title>{escape(args.title)}</title><style>{CSS}{EXTRA_CSS}</style><body>"
        "<svg width=0 height=0 style='position:absolute' aria-hidden=true><defs>"
        "<marker id='arrow' viewBox='0 0 10 10' refX='7' refY='5' markerWidth='4' markerHeight='4' "
        "orient='auto-start-reverse'><path d='M0,0 L10,5 L0,10 z' fill='var(--solver)'/></marker>"
        "<marker id='arrow-w' viewBox='0 0 10 10' refX='7' refY='5' markerWidth='5' markerHeight='5' "
        "orient='auto-start-reverse'><path d='M0,0 L10,5 L0,10 z' fill='#fff'/></marker>"
        "</defs></svg><div class=wrap>"
        f"<h1>{escape(args.title)}</h1>"
        "<p class=sub>장면마다 <b>솔버의 답</b>과 <b>실제 움직임</b>을 버튼으로 바꿔 봅니다. 위 상자의 칸들이 "
        "솔버가 푼 경기 흐름에서 수비수가 결정한 순간들입니다 — <b>보라색 칸이 갈린 순간(딜레마)</b>. "
        "순서는 경기 흐름 중 수비수가 가장 크게 갈린 정도, 맨 뒤는 비교용으로 끝까지 확정된 장면.</p>"
        + bar + summary + NAV_BAR
        + f"<div class=stage>{''.join(cards)}</div></div>"
        + PLAY_JS.replace("__CLIPS__", json.dumps(clips, separators=(",", ":")))
        + SOLVER_JS.replace("__SOLVER__", json.dumps(solver, separators=(",", ":")))
        + REVIEW_JS.replace("__KEY__", key) + "</body></html>")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html, encoding="utf-8")
    print(f"{args.output}  ({args.output.stat().st_size/1024:.0f} KB) · 카드 {len(cards)} · "
          f"갈린 장면 {torn}/{len(f)}")


if __name__ == "__main__":
    main()
