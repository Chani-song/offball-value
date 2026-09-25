"""Visualise the vacated-space rule's misses for football adjudication.

Each miss shows: the tracking animation, the region the reacting defender
vacates (heat cells), his reaction path, and the candidate table with the
arrival times that produced the pick. The point is for the reviewer to
decide, scene by scene, whether the rule is wrong or the scene is simply
ambiguous — a contested label is a different problem from a broken rule.

Usage:
    python scripts/render_r3_miss_review.py MISSES_JSON --out OUT_HTML
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

TEMPLATE = """<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>비워진 공간 규칙 — 오답 검토</title>
<style>
:root{color-scheme:dark}
body{margin:0;background:#0d1117;color:#e6edf3;font:15px/1.55 -apple-system,BlinkMacSystemFont,"Apple SD Gothic Neo",sans-serif}
header{padding:16px 22px;border-bottom:1px solid #30363d}
h1{margin:0 0 4px;font-size:19px}.sub{color:#8b949e;font-size:13px}
main{display:grid;grid-template-columns:minmax(0,1.4fr) minmax(300px,1fr);gap:20px;padding:20px;align-items:start}
@media(max-width:980px){main{grid-template-columns:1fr}}
select,button{font:inherit;background:#161b22;color:#e6edf3;border:1px solid #30363d;border-radius:7px;padding:7px 9px;cursor:pointer}
.pitch{width:100%;background:#123d1e;border-radius:10px;border:1px solid #30363d}
.row{display:flex;gap:9px;align-items:center;flex-wrap:wrap;margin:9px 0}
.card{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:14px;margin-bottom:14px}
.card h2{margin:0 0 8px;font-size:15px}
table{width:100%;border-collapse:collapse;font-size:13px}
th,td{text-align:left;padding:6px 5px;border-bottom:1px solid #21262d}
th{color:#8b949e;font-weight:600}
tr.answer{background:#10301b}tr.pick{background:#3a1d1d}
.tag{display:inline-block;padding:1px 6px;border-radius:99px;font-size:11px;font-weight:700;margin-left:5px}
.t-ans{background:#2ea043;color:#fff}.t-pick{background:#da3633;color:#fff}.t-car{background:#ffd052;color:#000}
.legend{color:#8b949e;font-size:12px;line-height:1.7;margin-top:8px}
.key{display:inline-block;width:11px;height:11px;border-radius:3px;vertical-align:-1px;margin-right:5px}
</style></head><body>
<header>
  <h1>비워진 공간 규칙 — 오답 검토</h1>
  <div class="sub">규칙이 틀린 것인지, 장면 자체가 애매한 것인지 판단해 주세요.</div>
</header>
<main>
  <section>
    <div class="row"><select class="pick-scene"></select></div>
    <svg class="pitch" viewBox="0 0 105 68" role="img">
      <rect x="0" y="0" width="105" height="68" fill="#123d1e"/>
      <rect x="0" y="13.85" width="16.5" height="40.3" fill="none" stroke="white" stroke-width=".25"/>
      <rect x="88.5" y="13.85" width="16.5" height="40.3" fill="none" stroke="white" stroke-width=".25"/>
      <line x1="52.5" y1="0" x2="52.5" y2="68" stroke="white" stroke-width=".25"/>
      <circle cx="52.5" cy="34" r="9.15" fill="none" stroke="white" stroke-width=".25"/>
      <g class="cells"></g><g class="paths"></g><g class="moving"></g>
    </svg>
    <div class="row">
      <button class="play">재생</button>
      <input class="scrub" type="range" min="0" value="0" style="flex:1">
      <span class="time"></span>
    </div>
    <div class="legend">
      <span class="key" style="background:#ff8c42"></span>수비수가 1.5초간 비우고 지나간 누적 공간
      &nbsp; <span class="key" style="background:#45e0d0"></span>수비수의 러너 추적 경로
      &nbsp; <span class="key" style="background:#ff51c7"></span>러너
      &nbsp; <span class="key" style="background:#ffd052"></span>볼 소유자
      &nbsp; <span class="key" style="background:#2ea043"></span>형님 답
      &nbsp; <span class="key" style="background:#da3633"></span>규칙의 답
    </div>
  </section>
  <section>
    <div class="card">
      <h2 class="ctx"></h2>
      <div class="sub story"></div>
    </div>
    <div class="card">
      <h2>후보별 도달 경쟁</h2>
      <table><thead><tr><th>선수</th><th>공간점유</th><th>P</th><th>G</th><th>최종</th></tr></thead>
      <tbody class="cands"></tbody></table>
      <div class="legend">공간점유 = 수비수가 1.5초간 비운 공간을 얼마나 덮는가(공격수도 시간에 따라 이동). P = 패스가 그에게 도달·소유될 확률. G = 그 지점의 골 위험. 최종 = 공간점유 × Q. P/G가 —인 선수는 이 산출물의 옵션 목록에 없어 위협이 계산되지 않은 경우입니다.</div>
    </div>
  </section>
</main>
<script id="d" type="application/json">__DATA__</script>
<script>
const D=JSON.parse(document.getElementById('d').textContent);
const q=s=>document.querySelector(s);let si=0,fi=0,timer=null;
const S=()=>D[si];
const esc=t=>String(t??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const px=(s,x)=>52.5+(s.direction<0?-1:1)*Number(x), py=y=>34+Number(y);
function fill(){
  q('.pick-scene').innerHTML=D.map((s,i)=>`<option value="${i}">${esc(s.scene)} · 수비수 ${esc(s.defender)} — 형님 ${esc(s.answer)} / 규칙 ${esc(s.pick)}</option>`).join('');
  q('.pick-scene').value=si;
  const s=S();
  q('.ctx').textContent=`${s.scene} · 러너 ${s.runner} · 볼 소유자 ${s.carrier} · 수비수 ${s.defender}`;
  q('.story').innerHTML=`이 수비수가 <b>${esc(s.runner)}</b>를 따라가며 비운 공간을, 형님은 <b style="color:#2ea043">${esc(s.answer)}</b>가, 규칙은 <b style="color:#da3633">${esc(s.pick)}</b>가 공략한다고 봅니다.`;
  q('.cands').innerHTML=s.candidates.map(c=>{
    const cls=c.is_answer?'answer':(c.is_pick?'pick':'');
    const tags=(c.is_answer?'<span class="tag t-ans">형님</span>':'')+(c.is_pick?'<span class="tag t-pick">규칙</span>':'')+(c.is_carrier?'<span class="tag t-car">볼</span>':'');
    return `<tr class="${cls}"><td>${esc(c.name)}${tags}</td><td>${c.occ}</td><td>${c.P??'—'}</td><td>${c.G??'—'}</td><td><b>${c.score}</b></td></tr>`}).join('');
  const cmax=Math.max(...s.cells.map(c=>c.v),0.0001);
  q('.cells').innerHTML=s.cells.map(c=>`<rect x="${px(s,c.x)-0.5}" y="${py(c.y)-0.5}" width="1" height="1" fill="#ff8c42" opacity="${(c.v/cmax*0.62).toFixed(3)}"/>`).join('');
  q('.paths').innerHTML=`<polyline points="${s.defender_path.map(p=>px(s,p[1])+','+py(p[2])).join(' ')}" fill="none" stroke="#45e0d0" stroke-width=".45" opacity=".9"/>`;
  q('.scrub').max=s.frames.length-1;
}
function draw(){
  const s=S(),f=s.frames[Math.min(fi,s.frames.length-1)];
  let html=(f.players||[]).map(p=>{
    const atk=String(p[1])===String(s.attacking_team_id),
      isR=String(p[0])===s.runner_id,isC=String(p[0])===s.carrier_id,
      isD=String(p[0])===s.defender_id,isA=String(p[0])===s.answer_id,isP=String(p[0])===s.pick_id;
    const stroke=isR?'#ff51c7':isA?'#2ea043':isP?'#da3633':isC?'#ffd052':isD?'#45e0d0':'white';
    const r=(isR||isC||isD||isA||isP)?1.0:0.66;
    return `<circle cx="${px(s,p[2])}" cy="${py(p[3])}" r="${r}" fill="${atk?'#ff6d59':'#438dff'}" stroke="${stroke}" stroke-width="${r>0.9?'.45':'.13'}"><title>${esc(p[4])}</title></circle>`}).join('');
  if(f.ball)html+=`<circle cx="${px(s,f.ball[0])}" cy="${py(f.ball[1])}" r=".4" fill="#111" stroke="white" stroke-width=".16"/>`;
  q('.moving').innerHTML=html;q('.scrub').value=fi;
  q('.time').textContent=`t=${f.t>=0?'+':''}${f.t.toFixed(2)}초`;
}
q('.pick-scene').addEventListener('change',e=>{si=+e.target.value;fi=0;fill();draw()});
q('.scrub').addEventListener('input',e=>{fi=+e.target.value;draw()});
q('.play').addEventListener('click',()=>{if(timer){clearInterval(timer);timer=null;q('.play').textContent='재생';return}
  q('.play').textContent='정지';timer=setInterval(()=>{fi=(fi+1)%S().frames.length;draw()},90)});
fill();draw();
</script></body></html>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("misses_json", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    payload = args.misses_json.read_text(encoding="utf-8").replace("</", "<\\/")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(TEMPLATE.replace("__DATA__", payload), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
