"""Blind screen for re-judging which scenes contain a real defender dilemma.

Today's measurement separated two things the current labels conflate. On
every structural column - what committing to the runner costs on the
beneficiary and vice versa, and whether a compromise exists at all - `clear`
scenes are the highest and `possible` scenes the LOWEST, below `none`. So
the two verdicts denote different situations: `clear` is "this defender is
genuinely torn", `possible` is "this is a dangerous situation". Pooling them
halves the structural signal (0.685 to 0.592).

The reviewer proposed re-labelling conservatively before that was measured.
This builds the screen for it.

BLIND BY CONSTRUCTION. The page carries the tracking animation, the runner,
the carrier and the candidate defenders - nothing else. No dilemma score, no
beneficiary pick, no Q, and deliberately NOT the previous verdict, which
would anchor the re-judgment it is meant to replace. A token scan refuses to
write a page that leaks any of it.

Scene order is shuffled with a fixed seed so it is reproducible but carries
no information about the old labels.

Usage:
    python scripts/render_dilemma_relabel_screen.py <audit_dir> [...] \
        --out data/processed/dilemma_relabel/index.html
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Anything here would tell the reviewer what the model thinks.
FORBIDDEN_TOKENS = (
    "knee",
    "무릎값",
    "cross_cost",
    "exact_minimax_worst_q",
    "derived_option_id",
    "candidate_grid",
    "interaction_review",
    "beneficiary",
    "delivery",
    "accessibility",
    "trade",
)

TEMPLATE = """<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>딜레마 재판정</title>
<style>
:root{color-scheme:dark}
body{margin:0;background:#0d1117;color:#e6edf3;font:15px/1.55 -apple-system,BlinkMacSystemFont,"Apple SD Gothic Neo",sans-serif}
header{padding:16px 22px;border-bottom:1px solid #30363d}
h1{margin:0 0 4px;font-size:19px}.sub{color:#8b949e;font-size:13px}
main{display:grid;grid-template-columns:minmax(0,1.4fr) minmax(320px,1fr);gap:20px;padding:20px;align-items:start}
@media(max-width:980px){main{grid-template-columns:1fr}}
select,button,textarea{font:inherit;background:#161b22;color:#e6edf3;border:1px solid #30363d;border-radius:7px;padding:8px 10px}
button{cursor:pointer}button.primary{background:#1f6feb;border-color:#1f6feb;font-weight:700}
.pitch{width:100%;background:#123d1e;border-radius:10px;border:1px solid #30363d}
.row{display:flex;gap:9px;align-items:center;flex-wrap:wrap;margin:9px 0}
.card{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:14px;margin-bottom:14px}
.card h2{margin:0 0 8px;font-size:15px}
.verdict{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.verdict button{padding:12px;font-weight:600}
.verdict button.on{background:#1f6feb;border-color:#1f6feb}
.legend{color:#8b949e;font-size:12px;line-height:1.7;margin-top:8px}
.key{display:inline-block;width:11px;height:11px;border-radius:3px;vertical-align:-1px;margin-right:5px}
.prog{color:#8b949e;font-size:13px}
table{width:100%;border-collapse:collapse;font-size:13px}
th,td{text-align:left;padding:6px 5px;border-bottom:1px solid #21262d}
th{color:#8b949e;font-weight:600}
</style></head><body>
<header>
  <h1>딜레마 재판정</h1>
  <div class="sub">이 장면에서 <b>수비수 한 명이 두 공격수 사이에서 실제로 찢어지는가</b>를 판정해 주세요. 확신이 서는 것만 <b>확실</b>로 답해 주시면 됩니다.</div>
</header>
<main>
  <section>
    <div class="row">
      <select class="pick"></select>
      <span class="prog"></span>
    </div>
    <svg class="pitch" viewBox="0 0 105 68" role="img">
      <rect x="0" y="0" width="105" height="68" fill="#123d1e"/>
      <rect x="0" y="13.85" width="16.5" height="40.3" fill="none" stroke="white" stroke-width=".25"/>
      <rect x="88.5" y="13.85" width="16.5" height="40.3" fill="none" stroke="white" stroke-width=".25"/>
      <line x1="52.5" y1="0" x2="52.5" y2="68" stroke="white" stroke-width=".25"/>
      <circle cx="52.5" cy="34" r="9.15" fill="none" stroke="white" stroke-width=".25"/>
      <g class="moving"></g>
    </svg>
    <div class="row">
      <button class="play">재생</button>
      <input class="scrub" type="range" min="0" value="0" style="flex:1">
      <span class="time"></span>
    </div>
    <div class="legend">
      <span class="key" style="background:#ff51c7"></span>오프더볼 러너
      &nbsp; <span class="key" style="background:#ffd052"></span>볼 소유자
      &nbsp; <span class="key" style="background:#45e0d0"></span>후보 수비수
      &nbsp; <span class="key" style="background:#ff6d59"></span>공격팀
      &nbsp; <span class="key" style="background:#438dff"></span>수비팀
    </div>
  </section>
  <section>
    <div class="card">
      <h2 class="ctx"></h2>
      <table><tbody class="who"></tbody></table>
    </div>
    <div class="card">
      <h2>이 장면에 딜레마가 있나요?</h2>
      <div class="verdict">
        <button data-v="clear">확실 (clear)</button>
        <button data-v="possible">있을 수도 (possible)</button>
        <button data-v="unclear">모르겠음 (unclear)</button>
        <button data-v="none">없음 (none)</button>
      </div>
      <div class="legend">
        <b>확실</b> = 수비수 한 명이 두 공격수 사이에서 찢어지는 게 분명히 보임<br>
        <b>있을 수도</b> = 위험한 상황이지만 찢어짐이 뚜렷하진 않음<br>
        <b>없음</b> = 그런 구조가 아님
      </div>
    </div>
    <div class="card">
      <h2>찢어지는 수비수 (선택)</h2>
      <select class="torn"><option value="">— 고르지 않음 —</option></select>
      <div class="legend">확실이라고 답하셨다면 누가 찢어지는지도 골라주시면 큰 도움이 됩니다.</div>
    </div>
    <div class="card">
      <h2>메모 (선택)</h2>
      <textarea class="note" rows="3" style="width:100%"></textarea>
    </div>
    <div class="row">
      <button class="prev">◀ 이전</button>
      <button class="next">다음 ▶</button>
      <button class="primary save">CSV 내보내기</button>
    </div>
  </section>
</main>
<script id="d" type="application/json">__DATA__</script>
<script>
const D=JSON.parse(document.getElementById('d').textContent);
const q=s=>document.querySelector(s);let si=0,fi=0,timer=null;
const ans={};
const S=()=>D[si];
const esc=t=>String(t??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const px=(s,x)=>52.5+(s.direction<0?-1:1)*Number(x), py=y=>34+Number(y);
function fill(){
  const s=S();
  q('.pick').innerHTML=D.map((x,i)=>`<option value="${i}">${i+1}. ${esc(x.label)}${ans[x.id]?' ✓':''}</option>`).join('');
  q('.pick').value=si;
  q('.prog').textContent=`${Object.keys(ans).length} / ${D.length} 완료`;
  q('.ctx').textContent=s.label;
  q('.who').innerHTML=
    `<tr><td style="color:#ff51c7">러너</td><td>${esc(s.runner)}</td></tr>`+
    `<tr><td style="color:#ffd052">볼 소유자</td><td>${esc(s.carrier)}</td></tr>`+
    s.defenders.map(d=>`<tr><td style="color:#45e0d0">후보 수비수</td><td>${esc(d.name)}</td></tr>`).join('');
  q('.torn').innerHTML='<option value="">— 고르지 않음 —</option>'+
    s.defenders.map(d=>`<option value="${esc(d.id)}">${esc(d.name)}</option>`).join('');
  const a=ans[s.id]||{};
  q('.torn').value=a.torn||'';
  q('.note').value=a.note||'';
  document.querySelectorAll('.verdict button').forEach(b=>b.classList.toggle('on',a.verdict===b.dataset.v));
  q('.scrub').max=s.frames.length-1;
}
function draw(){
  const s=S(),f=s.frames[Math.min(fi,s.frames.length-1)];
  const ids=new Set(s.defenders.map(d=>d.id));
  q('.moving').innerHTML=(f.players||[]).map(p=>{
    const atk=String(p[1])===String(s.attacking_team_id),
      isR=String(p[0])===s.runner_id,isC=String(p[0])===s.carrier_id,isD=ids.has(String(p[0]));
    const stroke=isR?'#ff51c7':isC?'#ffd052':isD?'#45e0d0':'white';
    const r=(isR||isC||isD)?1.0:0.66;
    return `<circle cx="${px(s,p[2])}" cy="${py(p[3])}" r="${r}" fill="${atk?'#ff6d59':'#438dff'}" stroke="${stroke}" stroke-width="${r>0.9?'.45':'.13'}"><title>${esc(p[4])}</title></circle>`}).join('')
    +(f.ball?`<circle cx="${px(s,f.ball[0])}" cy="${py(f.ball[1])}" r=".4" fill="#111" stroke="white" stroke-width=".16"/>`:'');
  q('.scrub').value=fi;
  q('.time').textContent=`t=${f.t>=0?'+':''}${f.t.toFixed(2)}초`;
}
function go(i){si=(i+D.length)%D.length;fi=0;fill();draw();}
q('.pick').addEventListener('change',e=>go(+e.target.value));
q('.scrub').addEventListener('input',e=>{fi=+e.target.value;draw()});
q('.play').addEventListener('click',()=>{if(timer){clearInterval(timer);timer=null;q('.play').textContent='재생';return}
  q('.play').textContent='정지';timer=setInterval(()=>{fi=(fi+1)%S().frames.length;draw()},90)});
document.querySelectorAll('.verdict button').forEach(b=>b.addEventListener('click',()=>{
  const s=S();ans[s.id]=Object.assign(ans[s.id]||{},{verdict:b.dataset.v});fill();}));
q('.torn').addEventListener('change',e=>{const s=S();ans[s.id]=Object.assign(ans[s.id]||{},{torn:e.target.value});});
q('.note').addEventListener('input',e=>{const s=S();ans[s.id]=Object.assign(ans[s.id]||{},{note:e.target.value});});
q('.prev').addEventListener('click',()=>go(si-1));
q('.next').addEventListener('click',()=>go(si+1));
q('.save').addEventListener('click',()=>{
  const head='match_id,onset_frame_id,runner_name,carrier_name,dilemma_verdict,torn_defender_id,note';
  const body=D.map(s=>{const a=ans[s.id]||{};
    return [s.match_id,s.onset_frame_id,s.runner,s.carrier,a.verdict||'',a.torn||'',
            '"'+String(a.note||'').replace(/"/g,'""')+'"'].join(',')}).join('\\n');
  const blob=new Blob([head+'\\n'+body],{type:'text/csv;charset=utf-8'});
  const url=URL.createObjectURL(blob);const a=document.createElement('a');
  a.href=url;a.download='dilemma_relabel.csv';a.click();URL.revokeObjectURL(url);});
go(0);
</script></body></html>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audit_dirs", nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260912)
    options = parser.parse_args()

    scenes = {}
    for directory in options.audit_dirs:
        for scene in json.loads(
            (Path(directory) / "local_game_payoff_audits.json").read_text()
        ):
            scenes[(str(scene["match_id"]), str(scene["onset_frame_id"]))] = scene

    payload = []
    for key, scene in sorted(scenes.items()):
        frames = sorted(
            scene["background_frames"], key=lambda f: float(f["relative_time_s"])
        )
        if not frames:
            continue
        payload.append(
            {
                "id": f"{key[0]}:{key[1]}",
                "match_id": key[0],
                "onset_frame_id": key[1],
                "label": f"{scene['match_label'].split('·')[0].strip()} · {key[1]}",
                "direction": int(scene.get("attacking_direction") or 1),
                "attacking_team_id": str(scene["attacking_team_id"]),
                "runner": str(scene["runner_name"]),
                "runner_id": str(scene["runner_id"]),
                "carrier": str(scene["carrier_name"]),
                "carrier_id": str(scene["carrier_id"]),
                "defenders": [
                    {
                        "id": str(d["defender_id"]),
                        "name": str(d["defender_name"]),
                    }
                    for d in scene["candidate_defenders"]
                ],
                "frames": [
                    {
                        "t": round(float(f["relative_time_s"]), 3),
                        "players": f["players"],
                        "ball": f.get("ball"),
                    }
                    for f in frames
                ],
            }
        )

    random.Random(options.seed).shuffle(payload)
    html = TEMPLATE.replace(
        "__DATA__", json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    )
    leaked = [token for token in FORBIDDEN_TOKENS if token in html]
    if leaked:
        raise SystemExit(f"blind screen leaks model output: {leaked}")

    options.out.parent.mkdir(parents=True, exist_ok=True)
    options.out.write_text(html, encoding="utf-8")
    print(f"장면 {len(payload)}개 -> {options.out}")
    print(f"누출 검사 통과 (금지 토큰 {len(FORBIDDEN_TOKENS)}개 모두 없음)")


if __name__ == "__main__":
    main()
