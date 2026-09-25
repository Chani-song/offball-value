#!/usr/bin/env python3
"""Review the other two thirds of the triple: the run and the beneficiary.

The defender review answered "should this man be in the game?" and the gate
was built from the answers. Two questions from the original brief are still
unanswered: is what stage 1 found actually an off-ball run, and is the
player R9 names as the beneficiary the one who actually gains if the
defender follows it.

Same scenes as the defender review by default, so the three verdicts land on
the same clips and a whole triple can be scored right or wrong. Only gated
survivors get a beneficiary row -- the dropped candidates were already
judged and their beneficiaries do not matter.

One card per scene:
  the run          one verdict, on the runner
  each survivor    one verdict, on that defender's beneficiary

The beneficiary is drawn in yellow. When two surviving defenders name
different beneficiaries both are drawn, and the table says which belongs to
whom; hovering a dot gives the name.
"""

from __future__ import annotations

import argparse
import json
import random
from html import escape
from pathlib import Path

import pandas as pd

from render_triple_overview import (
    CSS, NAV_BAR, PLAY_JS, R_BALL, R_PLAYER, W, H,
    pitch_markings, player_bar, trail, xy,
)

GLOW = {"runner": "var(--glow-runner)", "defender": "var(--glow-defender)",
        "beneficiary": "var(--glow-benef)"}
TAG = {"runner": "러너", "defender": "반응 수비수", "beneficiary": "수혜자"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--gate", type=Path,
                   default=Path("data/processed/pair_gate_v7.csv"))
    p.add_argument("--build", type=Path,
                   default=Path("/work/hdd/bbmr/kseo1/offball-out/v7_r9_ssac"))
    p.add_argument("--output", type=Path,
                   default=Path("out/triple_review/index.html"))
    p.add_argument("--same-as", type=Path,
                   default=Path("out/gate_review/cards.csv"),
                   help="reuse these scenes; omit to draw a fresh sample")
    p.add_argument("--sample-n", type=int, default=40)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--trail-stride", type=int, default=4)
    return p.parse_args()


def scene_svg(game: dict, kept: pd.DataFrame, stride: int) -> tuple[str, dict]:
    frames = game["background_frames"]
    onset = game["onset_frame_id"]
    flip = int(game.get("attacking_direction", 1)) < 0
    at = min(frames, key=lambda f: abs(f["frame_id"] - onset))
    carrier = str(game["carrier_id"])
    attacking_team = str(game["attacking_team_id"])
    runner = str(game["runner_id"])
    defenders = {str(d) for d in kept["defender_id"]}
    bens = {str(b) for b in kept["beneficiary_id"].dropna()} - {runner}

    def role_of(pid: str) -> str | None:
        if pid == runner:
            return "runner"
        if pid in defenders:
            return "defender"
        if pid in bens:
            return "beneficiary"
        return None

    out = [f"<svg viewBox='0 0 {W:.0f} {H:.0f}' class=pitch role=img "
           f"aria-label='피치 다이어그램'>",
           f"<rect width='{W:.0f}' height='{H:.0f}' rx='8' fill='var(--pitch-fill)'/>",
           pitch_markings(flip), "<g class=trails>"]
    for pid in [runner, *sorted(defenders), *sorted(bens)]:
        line = trail(frames, pid, flip, stride)
        if line:
            out.append(f"<g style='color:{GLOW[role_of(pid)]}'>{line}</g>")
    out.append("</g>")

    order = [str(p[0]) for p in at["players"]]
    for slot, p in enumerate(at["players"]):
        pid, team, x, y, name = str(p[0]), str(p[1]), p[2], p[3], p[4]
        px, py = xy(x, y, flip)
        fill = "var(--team-att)" if team == attacking_team else "var(--team-def)"
        role = role_of(pid)
        side = "공격" if team == attacking_team else "수비"
        title = f"{name} · {side}" + (f" · {TAG[role]}" if role else "")
        glow = ""
        if role:
            glow = (f"<circle class='gl' data-i='{slot}' cx='{px:.1f}' cy='{py:.1f}' "
                    f"r='{R_PLAYER + 2.5:.1f}' fill='none' stroke='{GLOW[role]}' "
                    f"stroke-width='3' style='filter:drop-shadow(0 0 5px {GLOW[role]})'/>")
        dot = (f"<circle class='pl' data-i='{slot}' data-n=\'{escape(title)}\' "
               f"cx='{px:.1f}' cy='{py:.1f}' r='{R_PLAYER}' fill='{fill}' "
               f"stroke='var(--dot-line)' stroke-width='1'/>")
        pip = (f"<circle class='pip' data-i='{slot}' cx='{px:.1f}' cy='{py:.1f}' "
               f"r='2' fill='#fff' opacity='.9' pointer-events='none'/>") \
            if pid == carrier else ""
        out.append(glow + dot + pip)

    bx, by = xy(at["ball"][0], at["ball"][1], flip)
    out.append(f"<circle class=ball cx='{bx:.1f}' cy='{by:.1f}' r='{R_BALL}' "
               f"fill='#fff' stroke='#111' stroke-width='1.4' pointer-events='none'/>")
    out.append("</svg>")

    positions, ball, clock = [], [], []
    for f in frames:
        by_id = {str(p[0]): (p[2], p[3]) for p in f["players"]}
        positions.append([[round(v, 1) for v in xy(*by_id[pid], flip)]
                          if pid in by_id else None for pid in order])
        ball.append([round(v, 1) for v in xy(f["ball"][0], f["ball"][1], flip)])
        clock.append(round(float(f["relative_time_s"]), 2))
    onset_index = min(range(len(frames)),
                      key=lambda i: abs(frames[i]["frame_id"] - onset))
    return "".join(out), {"p": positions, "b": ball, "t": clock, "onset": onset_index}


def verdict_buttons(key: str, labels: tuple[tuple[str, str], ...]) -> str:
    return "".join(
        f"<button type=button class=v data-k=\'{escape(key)}\' data-v=\'{v}\'>{t}</button>"
        for v, t in labels)


RUN_LABELS = (("ok", "오프더볼 런 맞음"), ("bad", "아님"), ("meh", "애매"))
BEN_LABELS = (("ok", "수혜자 맞음"), ("bad", "아님"), ("meh", "애매"))

EXTRA_CSS = """
.qrow{display:flex;align-items:center;gap:10px;padding:7px 2px;
 border-top:1px solid var(--border);flex-wrap:wrap}
.qrow .who{flex:1 1 230px;min-width:0;font-size:13px}
.qrow .who b{font-weight:600}
.qrow .who span{color:var(--muted);font-size:12px}
.qhead{font-size:11.5px;text-transform:uppercase;letter-spacing:.04em;
 color:var(--text-secondary);margin:12px 2px 0}
.chip{display:inline-block;padding:1px 7px;border-radius:99px;font-size:11px;
 border:1px solid var(--border);color:var(--text-secondary);margin-left:6px}
.runner-name{font-size:13px;color:var(--text-secondary);margin-bottom:2px}
"""

REVIEW_JS = """
<script>
(function () {
  var KEY = 'offball-triple-review-v1';
  var data = {};
  function load(){ try{ data = JSON.parse(localStorage.getItem(KEY)||'{}')||{}; }catch(e){ data={}; } }
  function save(){ try{ localStorage.setItem(KEY, JSON.stringify(data)); }catch(e){} }
  function cards(){ return document.querySelectorAll('.stage .card'); }
  function progress(){
    var done=0, all=cards();
    all.forEach(function(c){
      var ks={}; c.querySelectorAll('button.v').forEach(function(b){ ks[b.dataset.k]=1; });
      var keys=Object.keys(ks);
      if (keys.length && keys.every(function(k){ return data[k] && data[k].v; })) done++;
    });
    var el=document.querySelector('.prog');
    if (el) el.textContent = done + ' / ' + all.length + ' 장면 판정함';
  }
  function paint(){
    document.querySelectorAll('button.v').forEach(function(b){
      var e=data[b.dataset.k]||{};
      b.setAttribute('aria-pressed', e.v===b.dataset.v ? 'true':'false');
    });
    document.querySelectorAll('input.note').forEach(function(i){
      var e=data[i.dataset.k]||{}; if(e.note!==undefined) i.value=e.note;
    });
    progress();
  }
  document.addEventListener('click', function(ev){
    var b=ev.target.closest && ev.target.closest('button.v'); if(!b) return;
    var e=data[b.dataset.k]=data[b.dataset.k]||{};
    e.v = (e.v===b.dataset.v) ? '' : b.dataset.v;
    save(); paint();
  });
  document.addEventListener('input', function(ev){
    if(!ev.target.matches||!ev.target.matches('input.note')) return;
    var e=data[ev.target.dataset.k]=data[ev.target.dataset.k]||{};
    e.note=ev.target.value; save(); progress();
  });
  function cell(v){ v=(v===undefined||v===null)?'':String(v);
    return /[",\\n]/.test(v)?'"'+v.replace(/"/g,'""')+'"':v; }
  function csv(){
    var lines=['match_id,onset_frame_id,runner_id,question,defender_id,verdict,note'];
    cards().forEach(function(c){
      var cid=c.dataset.card, note=(data[cid]||{}).note||'';
      c.querySelectorAll('button.v[data-v=ok]').forEach(function(b){
        var k=b.dataset.k, v=(data[k]||{}).v||'';
        if(!v && !note) return;
        /* key: match:onset:runner:question[:defender] */
        var p=k.split(':');
        lines.push([p[0],p[1],p[2],p[3],p[4]||'',v,note].map(cell).join(','));
      });
    });
    return lines.join('\\n')+'\\n';
  }
  window.addEventListener('DOMContentLoaded', function(){
    load(); paint();
    var ex=document.querySelector('.export');
    if(ex) ex.addEventListener('click', function(){
      var blob=new Blob([csv()],{type:'text/csv;charset=utf-8'});
      var a=document.createElement('a'); a.href=URL.createObjectURL(blob);
      a.download='triple_review.csv'; a.click(); URL.revokeObjectURL(a.href);
    });
    var cl=document.querySelector('.clear');
    if(cl) cl.addEventListener('click', function(){
      if(!confirm('이 브라우저에 저장된 판정을 모두 지웁니다. 계속할까요?')) return;
      data={}; save(); paint();
      document.querySelectorAll('input.note').forEach(function(i){ i.value=''; });
    });
  });
})();
</script>
"""


def main() -> None:
    args = parse_args()
    gate = pd.read_csv(args.gate)
    key = ["match_id", "onset_frame_id", "runner_id"]
    kept_all = gate[gate["kept"]]

    if args.same_as and args.same_as.exists():
        want = pd.read_csv(args.same_as)[key].drop_duplicates()
        source = "수비수 검토와 같은 장면"
    else:
        scenes = kept_all[key].drop_duplicates()
        rng = random.Random(args.seed)
        want = scenes.iloc[sorted(rng.sample(range(len(scenes)),
                                             min(args.sample_n, len(scenes))))]
        source = f"무작위 {len(want)}장면 (seed {args.seed})"

    cards, clips, listing = [], {}, []
    for n, s in enumerate(want.itertuples(index=False), 1):
        mid, onset, rid = s.match_id, int(s.onset_frame_id), s.runner_id
        kept = kept_all[(kept_all.match_id == mid)
                        & (kept_all.onset_frame_id == onset)
                        & (kept_all.runner_id == rid)].sort_values("rank")
        if kept.empty:
            continue
        path = args.build / kept["scene_dir"].iloc[0] / "local_game_payoff_audits.json"
        payload = json.loads(path.read_text())
        if not payload:
            continue
        game = payload[0]
        card_id = f"{mid}:{onset}:{rid}"
        svg, clip = scene_svg(game, kept, args.trail_stride)
        clips[card_id] = clip
        match = str(game.get("match_label", mid)).split(" · ")[0]

        rows = [
            f"<div class=qhead>① 이게 오프더볼 런인가</div>"
            f"<div class=qrow><span class=who><b>{escape(str(game.get('runner_name')))}</b>"
            f"<span> — 공은 {escape(str(game.get('carrier_name')))}가 갖고 있음</span></span>"
            f"{verdict_buttons(f'{card_id}:run', RUN_LABELS)}</div>",
            f"<div class=qhead>② 이 수비수가 러너를 따라가면, 이득 보는 게 이 선수인가</div>",
        ]
        for r in kept.itertuples():
            ben = r.beneficiary_name if isinstance(r.beneficiary_name, str) and r.beneficiary_name else "—"
            same = " <span class=chip>볼 소유자</span>" if ben == game.get("carrier_name") else ""
            rows.append(
                f"<div class=qrow><span class=who>"
                f"<b>{escape(str(r.defender_name))}</b> 가 따라가면 → "
                f"<b>{escape(str(ben))}</b>{same}"
                f"<span class=chip>{escape(str(r.reason))}</span></span>"
                f"{verdict_buttons(f'{card_id}:ben:{r.defender_id}', BEN_LABELS)}</div>")
        rows.append(
            f"<input class=note data-k=\'{escape(card_id)}\' "
            f"placeholder=\'메모 — 진짜 수혜자가 따로 있으면 이름을 적어 주세요\'>")

        cards.append(
            f"<div class=card data-clip='{escape(card_id)}' data-card='{escape(card_id)}'>"
            f"<div class=hd><span class=rank>#{n}</span>"
            f"<span class=match title='{escape(match)}'>{escape(match)} · f{onset}</span></div>"
            f"<div class=runner-name>유지된 수비수 {len(kept)}명</div>"
            f"{svg}{player_bar()}{''.join(rows)}</div>")
        listing.append({"n": n, "match_id": mid, "onset_frame_id": onset,
                        "runner_id": rid, "kept_n": len(kept)})

    legend = (
        "<div class=legend>"
        "<span class=key><span class=tdot style='background:var(--team-att)'></span>공격팀</span>"
        "<span class=key><span class=tdot style='background:var(--team-def)'></span>수비팀</span>"
        "<span class=key><span class=gdot style='color:var(--glow-runner)'></span>러너</span>"
        "<span class=key><span class=gdot style='color:var(--glow-defender)'></span>반응 수비수</span>"
        "<span class=key><span class=gdot style='color:var(--glow-benef)'></span>수혜자</span>"
        "<span class=key>흰 점 = 볼 소유자 · 점에 마우스를 올리면 이름</span>"
        "<label class=key><input type=checkbox class=trailtog checked> 궤적 보기</label></div>")
    bar = ("<div class=bar><span class=prog>0 / 0 장면 판정함</span>"
           "<button type=button class=export>CSV 내려받기</button>"
           "<button type=button class=clear>판정 지우기</button>"
           "<span style='font-size:12.5px;color:var(--muted)'>판정은 이 브라우저에만 저장됩니다.</span></div>")
    html = (
        "<!doctype html><html lang=ko><meta charset=utf-8>"
        "<meta name=viewport content='width=device-width,initial-scale=1'>"
        "<title>러너·수혜자 검토</title>"
        f"<style>{CSS}{EXTRA_CSS}</style><body><div class=wrap>"
        "<h1>러너와 수혜자 검토</h1>"
        f"<p class=sub>{source}. 수비수는 이미 판정하셨으니 여기서는 두 가지만 봅니다 — "
        "<b>①</b> 빨간 테두리가 정말 오프더볼 런인가, "
        "<b>②</b> 그 수비수가 러너를 따라갔을 때 이득을 보는 게 노란 테두리 선수가 맞는가. "
        "게이트를 통과한 수비수만 나옵니다.</p>"
        + bar + legend + NAV_BAR
        + f"<div class=stage>{''.join(cards)}</div></div>"
        + PLAY_JS.replace("__CLIPS__", json.dumps(clips, separators=(",", ":")))
        + REVIEW_JS + "</body></html>")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html, encoding="utf-8")
    pd.DataFrame(listing).to_csv(args.output.with_name("cards.csv"), index=False)
    qs = sum(1 + r["kept_n"] for r in listing)
    print(f"{args.output}  ({args.output.stat().st_size/1024:.0f} KB) · "
          f"카드 {len(cards)} · 판정 항목 {qs}개")


if __name__ == "__main__":
    main()
