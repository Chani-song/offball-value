#!/usr/bin/env python3
"""The stage-3 answer for each triple, next to what actually happened.

Two views per scene, switched by a button, never drawn on top of each other:

  솔버의 답   only the three bodies the solver plays -- carrier, runner,
             defender -- solid and labelled, everyone else faint. At the start
             the defender carries one arrow per move the equilibrium puts
             weight on, sized by probability and named in football terms
             (toward the ball, toward the runner, back toward goal, hold).
             Two arrows of similar length is the dilemma. Play runs one solved
             play through the solved game's horizon, choosing the sample whose first
             defender move is the likeliest one, so what moves is the typical
             answer rather than a coin toss that went the rare way.
  실제 움직임 the real tracking clip, untouched.

The directions are named by comparing each compass move with the lines from
the defender to the ball carrier, to the runner and to his own goal, and
taking whichever it points along most (cosine at least 0.3, else "sideways").
The solver works in compass moves; the reader thinks in "does he go to the
ball or the man", and the label is only a translation of the former into the
latter.

A split counts when even the likeliest TARGET is under 80%, after merging
compass moves that lead to the same place. Two reasons the merge matters.
Exact equilibria can put 99.5% on one move and 0.5% on another, which is a
decision, not a dilemma. And a defender split 55/45 between east and south
may be heading for the ball either way -- choosing the angle of approach,
not choosing between the ball and the man. Counted by compass, 11 of 167
starts were split; counted by target, 10, and only one of those is the
textbook ball-or-runner case. The page is ordered by the target split at
the start, then by the (compass) split one turn later. Turn length is
read from the results table, so the page follows whichever study it is given.
"""

from __future__ import annotations

import argparse
import json
import math
from html import escape
from pathlib import Path

import pandas as pd

from offball_value.stage3_read import merged, root_choices, world_direction

from render_triple_overview import (
    CSS, NAV_BAR, PLAY_JS, R_BALL, R_PLAYER, W, H,
    pitch_markings, player_bar, trail, xy,
)

SPLIT_BELOW = 0.8
FAMILY_KR = {"ground": "땅볼 패스", "driven": "강한 패스", "lofted": "띄운 패스"}
GLOW = {"runner": "var(--glow-runner)", "defender": "var(--glow-defender)",
        "carrier": "var(--glow-benef)"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path,
                   default=Path("data/processed/stage3/carrier_results.csv"))
    p.add_argument("--build", type=Path,
                   default=Path("/work/hdd/bbmr/kseo1/offball-out/v7_r9_ssac"))
    p.add_argument("--solved", type=Path,
                   default=Path("/work/hdd/bbmr/kseo1/offball-out/stage3_carrier"))
    p.add_argument("--output", type=Path,
                   default=Path("out/dilemma_review/index.html"))
    p.add_argument("--compare", type=Path, default=None,
                   help="results table of the same triples solved another way; "
                        "its answer is shown under each headline")
    p.add_argument("--compare-label", default="배경 수비수 없을 때")
    p.add_argument("--top", type=int, default=30)
    p.add_argument("--flat", type=int, default=8)
    p.add_argument("--trail-stride", type=int, default=4)
    return p.parse_args()


# --------------------------------------------------------------------------
# geometry

def c2p(pt, flip):
    """Solver corner-origin metres -> card pixels (via our centre origin)."""
    return xy(float(pt[0]) - 52.5, float(pt[1]) - 34.0, flip)


# --------------------------------------------------------------------------
# the two pitches

def real_svg(game, runner, defender, carrier, stride):
    frames = game["background_frames"]
    onset = game["onset_frame_id"]
    flip = int(game.get("attacking_direction", 1)) < 0
    at = min(frames, key=lambda f: abs(f["frame_id"] - onset))
    att = str(game["attacking_team_id"])
    roles = {runner: "runner", defender: "defender", carrier: "carrier"}
    tag = {"runner": "러너", "defender": "반응 수비수", "carrier": "볼 소유자"}
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
        title = f"{name} · {'공격' if team == att else '수비'}" + (f" · {tag[role]}" if role else "")
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


def solver_svg(game, state, runner, defender, carrier, choices, background=None):
    flip = int(game.get("attacking_direction", 1)) < 0
    frames = game["background_frames"]
    at = min(frames, key=lambda f: abs(f["frame_id"] - game["onset_frame_id"]))
    att = str(game["attacking_team_id"])
    rolls = state["rollouts"]
    # the play to animate: first sample whose defender opens with the likeliest move
    want = choices[0][0] if choices else None
    roll = next((r for r in rolls if r and r[0].get("commands", {}).get("defender") == want),
                rolls[0])
    steps = [{"t": round(float(s["time"]), 3),
              "c": [round(v, 1) for v in c2p(s["carrier"], flip)],
              "r": [round(v, 1) for v in c2p(s["receiver"], flip)],
              "d": [round(v, 1) for v in c2p(s["defender"], flip)]} for s in roll]
    last = roll[-1]
    end = {"event": last.get("event"), "t": round(float(last["time"]), 3)}
    if end["event"] == "release":
        end["target"] = [round(v, 1) for v in c2p(last["target"], flip)]
        end["family"] = FAMILY_KR.get(last.get("family"), last.get("family"))

    out = [f"<svg viewBox='0 0 {W:.0f} {H:.0f}' class=spitch role=img aria-label='솔버의 답'>",
           f"<rect width='{W:.0f}' height='{H:.0f}' rx='8' fill='var(--pitch-fill)'/>",
           pitch_markings(flip)]
    # Background defenders are IN the game when the study has them: they make
    # no decision but take up space, mark and hold the line, so they are drawn
    # moving along the real track the game used. Everyone else is not in the
    # game at all and stays a faint, still reference.
    bg_ids = [str(i) for i in (background or {}).get("ids", [])]
    in_game_bg = set(bg_ids)
    for p in at["players"]:
        if str(p[0]) in (runner, defender, carrier) or str(p[0]) in in_game_bg:
            continue
        px, py = xy(p[2], p[3], flip)
        out.append(f"<circle cx='{px:.1f}' cy='{py:.1f}' r='{R_PLAYER - 1.5}' "
                   f"fill='{'var(--team-att)' if str(p[1]) == att else 'var(--team-def)'}' "
                   f"opacity='.18'/>")
    bg_payload = None
    if bg_ids:
        track = background["positions"]            # [steps + 1][m][2], corner origin
        times = [float(t) for t in background["times_requested"]]
        bg_px = [[[round(v, 1) for v in c2p(pt, flip)] for pt in row] for row in track]
        names = background.get("names") or [""] * len(bg_ids)
        for j, (x0, y0) in enumerate(bg_px[0]):
            out.append(f"<circle class=bgd data-j='{j}' cx='{x0}' cy='{y0}' r='{R_PLAYER - 1}' "
                       f"fill='var(--team-def)' opacity='.55' stroke='var(--dot-line)'>"
                       f"<title>{escape(str(names[j]))} · 배경 수비수 (실제 경로)</title></circle>")
        bg_payload = {"t": times, "p": bg_px}
    # the defender's choice at the start
    d0m = rolls[0][0]["defender"]
    d0 = c2p(d0m, flip)
    out.append("<g class=choice>")
    ad = int(state["scenario"]["attack_direction"])
    for k, name, prob in choices:
        ux, uy = world_direction(k, ad)   # commands are relative to the attack
        if (ux, uy) == (0.0, 0.0):
            out.append(f"<circle cx='{d0[0]:.1f}' cy='{d0[1]:.1f}' r='{R_PLAYER + 5 + 9 * prob:.1f}' "
                       f"fill='none' stroke='var(--solver)' stroke-width='2.5' stroke-dasharray='4 3'/>"
                       f"<text x='{d0[0]:.1f}' y='{d0[1] + R_PLAYER + 24 + 9 * prob:.1f}' "
                       f"class=cl text-anchor='middle'>{name} {prob:.0%}</text>")
            continue
        L = 4.0 + 10.0 * prob
        tip = c2p((d0m[0] + ux * L, d0m[1] + uy * L), flip)
        lab = c2p((d0m[0] + ux * (L + 3.2), d0m[1] + uy * (L + 3.2)), flip)
        out.append(f"<line x1='{d0[0]:.1f}' y1='{d0[1]:.1f}' x2='{tip[0]:.1f}' y2='{tip[1]:.1f}' "
                   f"stroke='var(--solver)' stroke-width='{2.5 + 3.5 * prob:.1f}' "
                   f"stroke-linecap='round' marker-end='url(#arrow)'/>"
                   f"<text x='{lab[0]:.1f}' y='{lab[1] + 4:.1f}' class=cl "
                   f"text-anchor='middle'>{name} {prob:.0%}</text>")
    out.append("</g>")
    # the pass, shown at release
    if end["event"] == "release":
        c_end = steps[-1]["c"]
        out.append(f"<g class=spass style='opacity:0'>"
                   f"<line x1='{c_end[0]}' y1='{c_end[1]}' x2='{end['target'][0]}' "
                   f"y2='{end['target'][1]}' stroke='#fff' stroke-width='2.6' "
                   f"stroke-dasharray='6 4' marker-end='url(#arrow-w)'/>"
                   f"<text x='{end['target'][0]}' y='{end['target'][1] - 12}' class=pl2 "
                   f"text-anchor='middle'>{escape(str(end['family']))}</text></g>")
    # the three bodies, solid, labelled
    for key, role, lab in (("c", "carrier", "볼"), ("r", "runner", "러너"), ("d", "defender", "수비")):
        x0, y0 = steps[0][key]
        fill = "var(--team-def)" if role == "defender" else "var(--team-att)"
        out.append(f"<g class='sb sb-{key}' transform='translate({x0} {y0})'>"
                   f"<circle r='{R_PLAYER + 2.5:.1f}' fill='none' stroke='{GLOW[role]}' "
                   f"stroke-width='3' style='filter:drop-shadow(0 0 5px {GLOW[role]})'/>"
                   f"<circle r='{R_PLAYER}' fill='{fill}' stroke='var(--dot-line)'/>"
                   f"<text y='{-R_PLAYER - 7:.1f}' class=bl text-anchor='middle'>{lab}</text></g>")
    out.append("<text class=sclockbig x='14' y='26'>0.0초</text></svg>")
    return "".join(out), {"steps": steps, "end": end, "bg": bg_payload}


# --------------------------------------------------------------------------

EXTRA_CSS = """
:root{--solver:#c77dff}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme=light])){--solver:#d9a3ff}}
:root[data-theme=dark]{--solver:#d9a3ff}
.head{margin:4px 0 10px;padding:12px 14px;border-radius:8px;
 background:var(--page);border:1px solid var(--border)}
.head .who{font-size:12.5px;color:var(--text-secondary);margin-bottom:6px}
.head .who b{color:var(--text-primary)}
.head .ans{font-size:16px;line-height:1.5}
.head .ans .opt{display:inline-block;padding:1px 9px;margin:1px 2px;border-radius:99px;
 background:color-mix(in srgb,var(--solver) 22%,transparent);font-weight:600}
.head .tag{display:inline-block;margin-left:6px;padding:1px 8px;border-radius:5px;
 font-size:12.5px;font-weight:700}
.tag.split{background:#c77dff;color:#1a0b26}
.tag.sure{background:var(--border);color:var(--text-secondary)}
.head .more{margin-top:5px;font-size:12px;color:var(--muted)}
.modes{display:flex;gap:0;margin:0 0 8px;border:1px solid var(--border);
 border-radius:8px;overflow:hidden;width:max-content}
.modes button{font:inherit;font-size:13.5px;padding:7px 16px;border:0;
 background:var(--surface);color:var(--text-secondary);cursor:pointer}
.modes button[aria-pressed=true]{background:var(--solver);color:#1a0b26;font-weight:600}
.card .spitch,.card .sbar,.card .shelp{display:none}
.card[data-mode=solver] .spitch,.card[data-mode=solver] .shelp{display:block}
.card[data-mode=solver] .sbar{display:flex}
.card[data-mode=solver] svg.pitch,.card[data-mode=solver] .play,.card[data-mode=solver] .rhelp{display:none}
.sbar{align-items:center;gap:9px;margin:7px 2px 0}
.sp-btn{flex:none;width:32px;height:32px;padding:0;font-size:13px;border:1px solid var(--border);
 border-radius:6px;background:var(--page);color:var(--text-primary);cursor:pointer;line-height:1}
.sp-btn:hover{border-color:var(--muted)}
.sp-scrub{flex:1;min-width:0;accent-color:var(--solver);height:24px}
.sp-clock{flex:none;width:52px;text-align:right;font-size:12px;color:var(--text-secondary);
 font-variant-numeric:tabular-nums}
.spitch{display:block;width:100%;height:auto}
.cl{font:700 12px system-ui,sans-serif;fill:#fff;paint-order:stroke;
 stroke:rgba(40,10,60,.85);stroke-width:4px}
.bl{font:700 11px system-ui,sans-serif;fill:#fff;paint-order:stroke;
 stroke:rgba(0,0,0,.7);stroke-width:3px}
.pl2{font:700 12px system-ui,sans-serif;fill:#fff;paint-order:stroke;
 stroke:rgba(0,0,0,.75);stroke-width:3.5px}
.sclockbig{font:700 15px system-ui,sans-serif;fill:#fff;opacity:.9;paint-order:stroke;
 stroke:rgba(0,0,0,.5);stroke-width:3px}
.choice{transition:opacity .25s}
.card.playing .choice{opacity:.25}
.help{font-size:12.5px;color:var(--text-secondary);margin:6px 2px 0;line-height:1.55}
.qrow{display:flex;align-items:center;gap:10px;padding:10px 2px 4px;
 border-top:1px solid var(--border);margin-top:10px;flex-wrap:wrap}
.qrow .ask{flex:1 1 210px;font-size:13.5px;font-weight:600}
"""

SOLVER_JS = """
<script>
(function () {
  var S = __SOLVER__, FPS = 25, rate = 0.5;
  var timers = [];
  function stopAll() { timers.forEach(function (f) { f(); }); }
  function setup(card) {
    var d = S[card.dataset.card]; if (!d) return;
    var svg = card.querySelector('svg.spitch');
    var body = {c: svg.querySelector('.sb-c'), r: svg.querySelector('.sb-r'), d: svg.querySelector('.sb-d')};
    var pass = svg.querySelector('.spass'), big = svg.querySelector('.sclockbig');
    var bgd = svg.querySelectorAll('circle.bgd');
    var scrub = card.querySelector('.sp-scrub'), pp = card.querySelector('.sp-btn'), clk = card.querySelector('.sp-clock');
    var st = d.steps, T = st[st.length - 1].t, N = Math.max(1, Math.round(T * 100));
    scrub.max = N; scrub.value = 0;
    function at(t) {
      var a = st[0], b = st[0];
      for (var j = 0; j < st.length; j++) if (st[j].t <= t + 1e-6) { a = st[j]; b = st[Math.min(j + 1, st.length - 1)]; }
      var f = b.t > a.t ? Math.max(0, Math.min(1, (t - a.t) / (b.t - a.t))) : 0;
      return function (k) { return [a[k][0] + (b[k][0] - a[k][0]) * f, a[k][1] + (b[k][1] - a[k][1]) * f]; };
    }
    function draw(v) {
      var t = v / 100, p = at(t);
      ['c', 'r', 'd'].forEach(function (k) { var q = p(k); body[k].setAttribute('transform', 'translate(' + q[0] + ' ' + q[1] + ')'); });
      if (d.bg && bgd.length) {
        var bt = d.bg.t, i0 = 0;
        for (var j = 0; j < bt.length; j++) if (bt[j] <= t + 1e-6) i0 = j;
        var i1 = Math.min(i0 + 1, bt.length - 1);
        var g = bt[i1] > bt[i0] ? Math.max(0, Math.min(1, (t - bt[i0]) / (bt[i1] - bt[i0]))) : 0;
        bgd.forEach(function (c) {
          var k = +c.dataset.j, A = d.bg.p[i0][k], B = d.bg.p[i1][k];
          c.setAttribute('cx', A[0] + (B[0] - A[0]) * g); c.setAttribute('cy', A[1] + (B[1] - A[1]) * g);
        });
      }
      var released = d.end && d.end.event === 'release' && t >= d.end.t - 1e-6;
      if (pass) pass.style.opacity = released ? 1 : 0;
      var txt = t.toFixed(1) + '초';
      if (t >= T - 1e-6) txt += d.end.event === 'release' ? ' · 패스' : d.end.event === 'retain' ? ' · 공 보유' : ' · 태클';
      big.textContent = txt; clk.textContent = t.toFixed(2) + 's';
      card.classList.toggle('playing', v > 0);
    }
    var timer = null;
    function stop() { if (timer) { clearInterval(timer); timer = null; } pp.textContent = '▶'; }
    timers.push(stop);
    pp.addEventListener('click', function () {
      if (timer) { stop(); return; }
      if (+scrub.value >= N) { scrub.value = 0; draw(0); }
      pp.textContent = '❚❚';
      timer = setInterval(function () {
        var v = +scrub.value + 4; if (v > N) { v = N; }
        scrub.value = v; draw(v); if (v >= N) stop();
      }, 1000 / (FPS * rate));   // 4 units of 0.01 s = one 25 fps frame per tick
    });
    scrub.addEventListener('input', function () { stop(); draw(+scrub.value); });
    draw(0);
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
    document.addEventListener('click', function (ev) {
      if (ev.target.closest && ev.target.closest('.prev, .next')) stopAll();
    });
    document.addEventListener('keydown', function (ev) { if (ev.key === 'ArrowLeft' || ev.key === 'ArrowRight') stopAll(); });
    var sel = document.querySelector('.rate');
    if (sel) { rate = parseFloat(sel.value) || 0.5; sel.addEventListener('change', function () { rate = parseFloat(sel.value) || 0.5; stopAll(); }); }
  });
})();
</script>
"""

REVIEW_JS = """
<script>
(function () {
  var KEY = 'offball-dilemma-review-v2', data = {};
  function load(){ try{ data=JSON.parse(localStorage.getItem(KEY)||'{}')||{}; }catch(e){ data={}; } }
  function save(){ try{ localStorage.setItem(KEY, JSON.stringify(data)); }catch(e){} }
  function cards(){ return document.querySelectorAll('.stage .card'); }
  function paint(){
    var done=0;
    cards().forEach(function(c){
      var e=data[c.dataset.card]||{}; if(e.v) done++;
      c.querySelectorAll('button.v').forEach(function(b){ b.setAttribute('aria-pressed', e.v===b.dataset.v?'true':'false'); });
      var n=c.querySelector('input.note'); if(n && e.note!==undefined) n.value=e.note;
    });
    var el=document.querySelector('.prog'); if(el) el.textContent=done+' / '+cards().length+' 판정함';
  }
  document.addEventListener('click', function(ev){
    var b=ev.target.closest && ev.target.closest('button.v'); if(!b) return;
    var e=data[b.dataset.k]=data[b.dataset.k]||{}; e.v=(e.v===b.dataset.v)?'':b.dataset.v; save(); paint();
  });
  document.addEventListener('input', function(ev){
    if(!ev.target.matches||!ev.target.matches('input.note')) return;
    var e=data[ev.target.dataset.k]=data[ev.target.dataset.k]||{}; e.note=ev.target.value; save();
  });
  function cell(v){ v=(v===undefined||v===null)?'':String(v); return /[",\\n]/.test(v)?'"'+v.replace(/"/g,'""')+'"':v; }
  window.addEventListener('DOMContentLoaded', function(){
    load(); paint();
    var ex=document.querySelector('.export');
    if(ex) ex.addEventListener('click', function(){
      var L=['state_index,match_id,onset_frame_id,runner_id,defender_id,split_t0,split_t1,value,verdict,note'];
      cards().forEach(function(c){ var e=data[c.dataset.card]||{}; if(!e.v&&!e.note) return;
        L.push([c.dataset.idx,c.dataset.match,c.dataset.onset,c.dataset.runner,c.dataset.defender,
                c.dataset.s0,c.dataset.s1,c.dataset.value,e.v||'',e.note||''].map(cell).join(',')); });
      var blob=new Blob([L.join('\\n')+'\\n'],{type:'text/csv;charset=utf-8'});
      var a=document.createElement('a'); a.href=URL.createObjectURL(blob); a.download='dilemma_review.csv'; a.click(); URL.revokeObjectURL(a.href);
    });
    var cl=document.querySelector('.clear');
    if(cl) cl.addEventListener('click', function(){ if(!confirm('판정을 모두 지웁니다. 계속할까요?')) return;
      data={}; save(); paint(); document.querySelectorAll('input.note').forEach(function(i){ i.value=''; }); });
  });
})();
</script>
"""


def main() -> None:
    args = parse_args()
    f = pd.read_csv(args.results)
    step = float(f["step_seconds"].iloc[0]) if "step_seconds" in f else 0.4
    horizon = step * (int(f["steps"].iloc[0]) if "steps" in f else 3)
    f = f.sort_values(["fsplit0", "split_t1"], ascending=False)
    top = f.head(args.top)
    # The most certain scenes, for contrast: no split at the start, least split
    # a turn later. Over a 3 s game every scene splits somewhere, so "never"
    # is not available as a control and "least" is.
    flat = f[f["fsplit0"] == 0].nsmallest(args.flat, "split_t1")
    picked = pd.concat([top, flat.loc[~flat.index.isin(top.index)]])

    starting = json.loads((args.solved / "starting_states.json").read_text())["states"]
    backgrounds = {int(r["index"]): r.get("background") for r in starting}
    has_bg = any(backgrounds.values())
    other = (pd.read_csv(args.compare).set_index("index") if args.compare else None)
    cards, clips, solver = [], {}, {}
    for n, r in enumerate(picked.itertuples(), 1):
        game = json.loads((args.build / r.scene_dir / "local_game_payoff_audits.json").read_text())[0]
        state = json.loads((args.solved / "states" / f"state_{int(r.index):03d}.json").read_text())
        runner, defender, carrier = str(r.runner_id), str(r.defender_id), str(game["carrier_id"])
        choices = root_choices(state)
        agg = merged(choices)
        split = bool(agg) and agg[0][1] < SPLIT_BELOW
        card_id = f"s3:{int(r.index)}"

        rs, clip = real_svg(game, runner, defender, carrier, args.trail_stride)
        clips[card_id] = clip
        ss, sd = solver_svg(game, state, runner, defender, carrier, choices,
                            backgrounds.get(int(r.index)))
        solver[card_id] = sd

        opts = " ".join(f"<span class=opt>{escape(k)} {v:.0%}</span>" for k, v in agg)
        tag = ("<span class='tag split'>갈림 · 딜레마</span>" if split
               else "<span class='tag sure'>한쪽으로 확정</span>")
        later = (f"{step:g}초 뒤에도 {r.split_t1:.0%}의 상황에서 갈림" if r.split_t1 > 0
                 else f"{step:g}초 뒤에는 갈리는 상황 없음")
        end = sd["end"]
        if end["event"] == "release":
            when = "바로" if end["t"] < 0.05 else f"{end['t']:.1f}초에"
            how = f"{when} {end['family']}"
        elif end["event"] == "retain":
            how = f"{horizon:g}초 동안 공 보유"
        else:
            how = "태클로 뺏김"
        head = (
            f"<div class=head>"
            f"<div class=who>러너 <b>{escape(str(r.runner))}</b> · "
            f"수비수 <b>{escape(str(r.defender))}</b> · 볼 <b>{escape(str(r.carrier))}</b></div>"
            f"<div class=ans>런이 시작되는 순간, 솔버가 본 수비수의 최선: {opts}{tag}</div>"
            f"<div class=more>{later} · 솔버 경기에서 공격은 {how} · "
            f"공격 기댓값 {r.value:.2f}</div>"
            + (lambda o: "" if o is None else
               f"<div class=more style='margin-top:6px;color:var(--text-secondary)'>"
               f"{escape(args.compare_label)}: {escape(str(o.root_top))} {1 - o.fsplit0:.0%}"
               + (f" · {escape(str(o.root_second))}" if o.fsplit0 > 0 else "")
               + f" · 기댓값 {o.value:.2f}</div>")(
                other.loc[int(r.index)] if other is not None and int(r.index) in other.index else None)
            + "</div>")
        modes = ("<div class=modes>"
                 "<button type=button data-m=solver aria-pressed=true>솔버의 답</button>"
                 "<button type=button data-m=real aria-pressed=false>실제 움직임</button></div>")
        # Own class names throughout: the real clip's player finds its controls
        # with card.querySelector('.pp' / '.scrub' / '.clock') and would take these.
        splay = ("<div class=sbar><button type=button class=sp-btn aria-label='솔버 경기 재생'>▶</button>"
                 "<input type=range class=sp-scrub min=0 max=0 value=0 step=1 aria-label='솔버 시간'>"
                 "<span class=sp-clock>0.00s</span></div>")
        shelp = ("<div class='help shelp'>보라 화살표 = 런이 시작되는 순간 수비수가 각 방향으로 갈 확률. "
                 "비슷한 길이의 화살표가 둘이면 어느 쪽도 정답이 아닌 <b>딜레마</b>입니다. "
                 f"▶를 누르면 솔버가 푼 경기 한 판(수비수가 더 자주 고르는 쪽)이 {horizon:g}초 동안 진행되고, "
                 "패스가 나가면 흰 점선으로 보입니다. "
                 + ("<b>반투명 파란 점</b>은 배경 수비수 — 결정은 안 하지만 게임 안에서 공간을 막고 "
                    "마크하고 오프사이드 라인을 잡으며, 실제로 간 길을 따라 움직입니다. "
                    "아주 흐린 점은 게임에 없는 선수들." if has_bg else
                    "흐린 점은 솔버가 계산에 넣지 않은 나머지 선수들.")
                 + "</div>")
        rhelp = ("<div class='help rhelp'>실제 경기 영상(런 시작 2초 전 ~ 3초 후). "
                 "빨강 = 러너 · 파랑 테두리 = 이 수비수 · 노랑 = 볼 소유자.</div>")
        buttons = "".join(
            f"<button type=button class=v data-k='{card_id}' data-v='{v}'>{t}</button>"
            for v, t in (("ok", "딜레마 맞음"), ("bad", "아님"), ("meh", "애매")))
        cards.append(
            f"<div class=card data-clip='{card_id}' data-card='{card_id}' data-mode=solver "
            f"data-idx='{int(r.index)}' data-match='{escape(str(r.match_id))}' "
            f"data-onset='{int(r.onset)}' data-runner='{escape(runner)}' "
            f"data-defender='{escape(defender)}' data-s0='{r.split_t0:.3f}' "
            f"data-s1='{r.split_t1:.3f}' data-value='{r.value:.4f}'>"
            f"<div class=hd><span class=rank>#{n}</span>"
            f"<span class=match>{escape(str(r.match))} · f{int(r.onset)}</span></div>"
            f"{head}{modes}{ss}{splay}{shelp}{rs}{player_bar()}{rhelp}"
            f"<div class=qrow><span class=ask>이 수비수, 정말 고민되는 상황인가?</span>{buttons}</div>"
            f"<input class=note data-k='{card_id}' placeholder='메모 (선택)'></div>")

    rows = []
    for n, r in enumerate(f.itertuples(), 1):
        rows.append(f"<tr><td class=num>{n}</td><td>{escape(str(r.match))}</td>"
                    f"<td class=num>{int(r.onset)}</td><td>{escape(str(r.runner))}</td>"
                    f"<td>{escape(str(r.defender))}</td><td>{escape(str(r.carrier))}</td>"
                    f"<td>{escape(str(r.root_top))} {1 - r.fsplit0:.0%}"
                    f"{(' · ' + escape(str(r.root_second))) if r.fsplit0 > 0 else ''}</td>"
                    f"<td class=num>{r.split_t1:.0%}</td>"
                    f"<td class=num>{r.split_t2:.0%}</td><td class=num>{r.value:.2f}</td></tr>")
    table = ("<details><summary>솔버가 푼 장면 전부 ({}개) — 딜레마 강한 순</summary>"
             "<table><tr><th>#</th><th>경기</th><th class=num>프레임</th><th>러너</th><th>수비수</th>"
             f"<th>볼 소유자</th><th>시작 순간 최선</th><th class=num>{step:g}s 갈림</th><th class=num>{2*step:g}s</th>"
             "<th class=num>기댓값</th></tr>{}</table></details>").format(len(f), "".join(rows))

    bar = ("<div class=bar><span class=prog>0 / 0 판정함</span>"
           "<button type=button class=export>CSV 내려받기</button>"
           "<button type=button class=clear>판정 지우기</button></div>")
    html = (
        "<!doctype html><html lang=ko><meta charset=utf-8>"
        "<meta name=viewport content='width=device-width,initial-scale=1'>"
        "<title>솔버의 답</title>"
        f"<style>{CSS}{EXTRA_CSS}</style><body>"
        "<svg width=0 height=0 style='position:absolute' aria-hidden=true><defs>"
        "<marker id='arrow' viewBox='0 0 10 10' refX='7' refY='5' markerWidth='4' markerHeight='4' "
        "orient='auto-start-reverse'><path d='M0,0 L10,5 L0,10 z' fill='var(--solver)'/></marker>"
        "<marker id='arrow-w' viewBox='0 0 10 10' refX='7' refY='5' markerWidth='5' markerHeight='5' "
        "orient='auto-start-reverse'><path d='M0,0 L10,5 L0,10 z' fill='#fff'/></marker>"
        "</defs></svg><div class=wrap>"
        "<h1>솔버의 답</h1>"
        "<p class=sub>장면마다 <b>솔버의 답</b>과 <b>실제 움직임</b>을 버튼으로 바꿔 볼 수 있습니다. "
        "위 상자 한 줄이 핵심입니다 — 런이 시작되는 순간 수비수가 어느 쪽으로 가는 게 최선인지를 "
        "솔버가 확률로 답한 것이고, 두 쪽이 비슷하면 딜레마입니다. "
        "순서는 <b>시작 순간 수비수가 서로 다른 대상(볼 / 러너 / 골문 / 제자리) 사이에서 얼마나 고르게 갈리는가</b>, "
        f"그다음 {step:g}초 뒤에도 갈리는가입니다. 솔버는 {horizon:g}초 경기를 {step:g}초 턴으로 풉니다. "
        f"위 {int((f['fsplit0'] > 0.2).sum())}장면이 시작부터 갈린 것이고, 그 아래는 나중에 갈리거나 거의 확정인 것, "
        "맨 뒤는 비교용으로, 시작 순간 한쪽으로 확정되고 그 뒤에도 가장 덜 갈리는 장면입니다.</p>"
        + bar + NAV_BAR
        + f"<div class=stage>{''.join(cards)}</div>{table}</div>"
        + PLAY_JS.replace("__CLIPS__", json.dumps(clips, separators=(",", ":")))
        + SOLVER_JS.replace("__SOLVER__", json.dumps(solver, separators=(",", ":")))
        + REVIEW_JS + "</body></html>")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html, encoding="utf-8")
    print(f"{args.output}  ({args.output.stat().st_size/1024:.0f} KB) · 카드 {len(cards)}")


if __name__ == "__main__":
    main()
