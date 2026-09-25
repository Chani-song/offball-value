#!/usr/bin/env python3
"""Render a meeting-ready animation for the influence-based dilemma scene."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from offball_value.bundesliga import (
    find_bundesliga_files,
    load_bundesliga_match_metadata,
    normalize_bundesliga_match_id,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--match-id", default="J03WOH")
    parser.add_argument("--frame-id", type=int, default=68836)
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data/raw/bundesliga-integrated")
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/processed/attacker_maximin_v0_1"),
    )
    return parser.parse_args()


TEMPLATE = r"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Off-ball defensive choice · meeting scene</title>
<style>
:root{--bg:#081321;--panel:#111e30;--line:#30445f;--text:#f2f6ff;--muted:#aebbd0;--pink:#ff55c8;--cyan:#4ae3d0;--orange:#ff9255;--violet:#c98cff;--gold:#ffd452;--blue:#438cff;--red:#ff705b;--green:#78e8b2}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font-family:Inter,Pretendard,system-ui,sans-serif}main{max-width:1580px;margin:auto;padding:24px}h1{margin:0 0 5px;font-size:2rem}h2,h3{margin:0 0 9px}p{color:var(--muted);line-height:1.5}.tag{display:inline-block;padding:5px 10px;border-radius:99px;background:#273b5a;color:#8ee5ff;font-weight:750}.notice{padding:11px 14px;border-left:4px solid var(--gold);background:#192840;border-radius:7px}.layout{display:grid;grid-template-columns:minmax(760px,1fr) 520px;gap:18px}.panel{background:var(--panel);border:1px solid var(--line);border-radius:13px;padding:15px}.controls{display:flex;align-items:center;gap:9px;flex-wrap:wrap;margin:9px 0}select,button{font:inherit;color:var(--text);background:#1b2b43;border:1px solid #49617f;border-radius:8px;padding:8px 11px}svg{display:block;width:100%;background:#17733a;border-radius:8px}input[type=range]{width:100%}.time{margin-left:auto}.cards{display:grid;grid-template-columns:1fr 1fr;gap:8px}.card{padding:10px;background:#17263b;border:1px solid #344a66;border-radius:9px;cursor:pointer}.card.active{border:2px solid var(--cyan);padding:9px}.card h3{font-size:.98rem}.metric{display:grid;grid-template-columns:72px 1fr 48px;gap:7px;align-items:center;margin-top:8px;font-size:.84rem}.bar{height:8px;background:#273950;border-radius:5px;overflow:hidden}.bar i{display:block;height:100%}.explain,.box{background:#0e1928;border:1px solid #30445e;border-radius:9px;padding:11px;margin-top:10px;line-height:1.5}.explain{background:#192840}.legend{display:grid;grid-template-columns:1fr 1fr;gap:7px;margin-top:10px;color:#c9d4e4;font-size:.88rem}.swatch{display:inline-block;width:20px;border-top:4px solid;margin-right:7px}.eq{font-family:ui-monospace,monospace;color:#dbe8ff}.good{color:var(--green)}.warn{color:#ffd27a}.small{font-size:.83rem}.step{display:grid;grid-template-columns:26px 1fr;gap:8px;margin:8px 0;color:#cad5e5}.num{width:24px;height:24px;border-radius:50%;display:grid;place-items:center;background:#2b4568;color:#8ee5ff;font-weight:750}@media(max-width:1120px){.layout{grid-template-columns:1fr}}@media(max-width:650px){.cards,.legend{grid-template-columns:1fr}}
</style></head><body><main><h1>오프더볼이 수비수에게 어떤 선택을 강요하는가?</h1>
<p><span class="tag">meeting diagnostic · frame __FRAME__</span> __RUNNER__의 직접 위협과 __BENEFICIARY__의 파생 위협을, __DEFENDER__ 한 명의 실행 가능한 연속 수비 궤적 위에서 비교한다.</p>
<p class="notice"><b>오늘 보여줄 핵심:</b> 청록색 수비는 더 이상 “최고 OBSO 점으로 가는 길”이 아니다. 각 공격수의 <b>영향력 영역 전체</b>와 공격수–골대 사이의 공간을 얼마나 지우는지를 기준으로 759개 수비 궤적 중 선택했다. 아직 최종 threat model이 아니라 <b>수비 목적함수 검증용 prototype</b>이다.</p>
<div class="layout"><section class="panel"><div class="controls"><label>수비 선택 <select class="response"><option value="direct_cover">① runner 직접 억제</option><option value="tradeoff_compromise" selected>② 중간 절충</option><option value="beneficiary_cover">③ Kownacki 쪽 전환</option><option value="observed_reference">④ 실제 장면</option></select></label><button class="play">▶ 재생</button><span class="time">t=+0.00 s</span></div>
<svg viewBox="0 0 105 68"><rect width="105" height="68" fill="#17733a" stroke="white" stroke-width=".3"/><line x1="52.5" x2="52.5" y1="0" y2="68" stroke="white" stroke-width=".25"/><circle cx="52.5" cy="34" r="9.15" fill="none" stroke="white" stroke-width=".25"/><rect x="0" y="13.84" width="16.5" height="40.32" fill="none" stroke="white" stroke-width=".25"/><rect x="88.5" y="13.84" width="16.5" height="40.32" fill="none" stroke="white" stroke-width=".25"/><g class="corridors"></g><g class="paths"></g><g class="moving"></g></svg><input class="scrub" type="range" min="0" max="50" value="0">
<div class="legend"><span><i class="swatch" style="border-color:var(--pink)"></i><b class="runnerLegend">가상 runner 궤적</b></span><span><i class="swatch" style="border-color:var(--cyan)"></i><b class="defenderLegend">선택한 수비 궤적</b></span><span><i class="swatch" style="border-color:white;border-top-style:dashed"></i><b class="referenceLegend">실제 수비 reference</b></span><span><i class="swatch" style="border-color:var(--violet)"></i>Kownacki 실제 궤적</span><span><i class="swatch" style="border-color:var(--orange)"></i>runner→골대 위험 corridor</span><span><i class="swatch" style="border-color:var(--violet)"></i>Kownacki→골대 corridor</span></div></section>
<aside class="panel"><h2>세 수비 선택의 trade-off</h2><p class="small">수치는 각 공격수의 본래 goal-weighted influence 중 <b>수비 후에도 남은 비율</b>이다. 낮을수록 수비에 유리하다.</p><div class="cards"></div><div class="explain"></div>
<div class="box"><h3>현재 시점의 해석</h3><div class="now"></div></div>
<div class="box"><h3>목적함수</h3><p class="eq">Zᵢ(d) = ∫ Iᵢ(r) · Gᵢ(r) · exp(−λΣ I_def(r)) dr</p><p class="small"><b>I</b>: Fernández 선수 영향력 · <b>G</b>: 골대 거리와 goal-side corridor · <b>exp(…)</b>: 여러 수비수 영향력이 겹칠수록 잔여 공간 감소. G와 suppression은 이번 연구의 명시적 확장이다.</p></div>
<div class="box"><h3>미팅에서 확인할 세 가지</h3><div class="step"><i class="num">1</i><span>이 공간 적분이 “선수에게 붙기”뿐 아니라 <b>진로·골대 사이를 막기</b>도 타당하게 보상하는가?</span></div><div class="step"><i class="num">2</i><span>최종 threat는 이 지표인가, 아니면 영향력을 <b>OBSO의 선수별 attribution mask</b>로 쓸 것인가?</span></div><div class="step"><i class="num">3</i><span>이 장면은 trade-off는 보이지만 양쪽 희생이 대칭적이지 않다. 강한 dilemma 판정 문턱을 어떻게 둘 것인가?</span></div></div></aside></div></main>
<script id="data" type="application/json">__DATA__</script><script>
const s=JSON.parse(document.getElementById('data').textContent),R=s.result.responses,names=s.names,px=x=>x+52.5,py=y=>34-y;let key='tradeoff_compromise',idx=0,playing=false,timer;const interp=(p,t)=>{if(t<=p[0][0])return p[0].slice(1);if(t>=p.at(-1)[0])return p.at(-1).slice(1);for(let i=1;i<p.length;i++)if(t<=p[i][0]){let u=p[i-1],v=p[i],q=(t-u[0])/(v[0]-u[0]);return[u[1]+q*(v[1]-u[1]),u[2]+q*(v[2]-u[2])]}};const poly=(p,c,d='',w=.55)=>`<polyline points="${p.map(z=>`${px(z[1])},${py(z[2])}`).join(' ')}" fill="none" stroke="${c}" stroke-width="${w}" stroke-dasharray="${d}"/>`;const virtualAttack=()=>s.result.attack_path_times_s.map((t,i)=>[t,...s.result.attack_path_xy[i]]);const defense=k=>R[k].defender_path_times_s.map((t,i)=>[t,...R[k].defender_path_xy[i]]);const observedPlayer=id=>s.background.map(f=>[f.time_s,...f.players.find(p=>p[0]===id).slice(2)]);const observedAttack=()=>observedPlayer(s.result.runner_id),observedDef=()=>defense('observed_reference'),kw=()=>observedPlayer(s.result.beneficiary_id),shownAttack=()=>key==='observed_reference'?observedAttack():virtualAttack();const nearPoint=(r,t)=>r.points.reduce((u,v)=>Math.abs(v.time_s-t)<Math.abs(u.time_s-t)?v:u);
function corridor(a,c){let goal=[s.result.attacking_direction*52.5,0],dx=goal[0]-a[0],dy=goal[1]-a[1],L=Math.hypot(dx,dy),nx=-dy/L,ny=dx/L,w=2.6;return `<polygon points="${px(a[0]+nx*w)},${py(a[1]+ny*w)} ${px(goal[0]+nx*6)},${py(goal[1]+ny*6)} ${px(goal[0]-nx*6)},${py(goal[1]-ny*6)} ${px(a[0]-nx*w)},${py(a[1]-ny*w)}" fill="${c}" opacity=".12"/><line x1="${px(a[0])}" y1="${py(a[1])}" x2="${px(goal[0])}" y2="${py(goal[1])}" stroke="${c}" stroke-width=".25" stroke-dasharray=".8 .6"/>`}
const labels={direct_cover:'① runner 직접 억제',tradeoff_compromise:'② 중간 절충',beneficiary_cover:'③ Kownacki 쪽 전환',observed_reference:'④ 실제 장면'};function cards(){document.querySelector('.cards').innerHTML=Object.entries(labels).map(([k,l])=>{let r=R[k],actual=k==='observed_reference';return `<div class="card ${k===key?'active':''}" data-k="${k}"><h3>${l}</h3><div class="metric"><span>G 잔여</span><div class="bar"><i style="width:${r.runner_horizon_mean*100}%;background:var(--orange)"></i></div><b>${(r.runner_horizon_mean*100).toFixed(1)}%</b></div><div class="metric"><span>K 잔여</span><div class="bar"><i style="width:${r.beneficiary_horizon_mean*100}%;background:var(--violet)"></i></div><b>${(r.beneficiary_horizon_mean*100).toFixed(1)}%</b></div><p class="small">t=2 거리 · G ${r.runner_terminal_distance_m.toFixed(1)}m / K ${r.beneficiary_terminal_distance_m.toFixed(1)}m${actual?'<br><span class="warn">실제 Ginczek + 실제 Elvedi 세계</span>':''}</p></div>`}).join('');document.querySelectorAll('.card').forEach(c=>c.onclick=()=>setKey(c.dataset.k))}
function explain(){let d=R.direct_cover,m=R.tradeoff_compromise,k=R.beneficiary_cover;document.querySelector('.explain').innerHTML=`runner에 집중하면 G는 <b>${(d.runner_horizon_mean*100).toFixed(1)}%</b>까지 줄지만 K가 <b>${(d.beneficiary_horizon_mean*100).toFixed(1)}%</b> 남는다. Kownacki 영향영역으로 전환하면 K는 <b>${(k.beneficiary_horizon_mean*100).toFixed(1)}%</b>로 줄고 G는 <b>${(k.runner_horizon_mean*100).toFixed(1)}%</b>로 커진다. 중간 궤적은 <b>G ${(m.runner_horizon_mean*100).toFixed(1)}% / K ${(m.beneficiary_horizon_mean*100).toFixed(1)}%</b>다.<br><span class="good">K 전환 궤적은 759개 중 Kownacki와의 2초 평균거리가 가장 작은 1위 궤적이다.</span> 0.2초 반응 지연과 기존 운동상태 때문에 초반에는 Ginczek 근처를 벗어나지 못하며, t=2에도 K와 ${k.beneficiary_terminal_distance_m.toFixed(1)}m 떨어져 있다.<br><span class="warn">trade-off는 존재하지만 K 개선폭이 작아, 이 한 장면을 “강한 dilemma”라고 결론내리지는 않는다.</span>`}
function setKey(k){key=k;document.querySelector('.response').value=k;cards();drawPaths();render()}
function drawPaths(){let actual=key==='observed_reference';document.querySelector('.paths').innerHTML=poly(shownAttack(),'#ff55c8')+poly(defense(key),'#4ae3d0')+(actual?'':poly(observedDef(),'white','1 .65',.38))+poly(kw(),'#c98cff','1 .5',.4);document.querySelector('.runnerLegend').textContent=actual?'실제 runner 궤적':'가상 runner 궤적';document.querySelector('.defenderLegend').textContent=actual?'실제 수비 궤적':'선택한 수비 궤적';document.querySelector('.referenceLegend').textContent=actual?'(실제 장면에서는 미표시)':'실제 수비 reference'}
function render(){let t=idx/25,f=s.background[idx],ap=interp(shownAttack(),t),dp=interp(defense(key),t),kp=interp(kw(),t),did=s.result.defender_id,actual=key==='observed_reference';document.querySelector('.corridors').innerHTML=corridor(ap,'#ff9255')+corridor(kp,'#c98cff');let players=f.players.filter(p=>p[0]!==s.result.runner_id&&p[0]!==did&&p[0]!==s.result.beneficiary_id).map(p=>`<circle cx="${px(p[2])}" cy="${py(p[3])}" r=".7" fill="${p[1]===s.result.attacking_team_id?'#ff705b':'#438cff'}" stroke="white" stroke-width=".15"><title>${names[p[0]]||p[0]}</title></circle>`).join('');players+=`<circle cx="${px(f.ball[0])}" cy="${py(f.ball[1])}" r=".42" fill="#111" stroke="white" stroke-width=".15"/><circle cx="${px(ap[0])}" cy="${py(ap[1])}" r="1.06" fill="#ff55c8" stroke="white" stroke-width=".22"><title>${s.result.runner_name}${actual?' · actual':' · virtual'}</title></circle><circle cx="${px(kp[0])}" cy="${py(kp[1])}" r=".95" fill="#ff705b" stroke="#c98cff" stroke-width=".55"><title>${s.result.beneficiary_name}</title></circle><circle cx="${px(dp[0])}" cy="${py(dp[1])}" r="1" fill="#438cff" stroke="#4ae3d0" stroke-width=".52"><title>${s.result.defender_name}${actual?' · actual':' · virtual'}</title></circle>`;document.querySelector('.moving').innerHTML=players;let p=nearPoint(R[key],Math.max(t,.2)),world=actual?'실제 Ginczek과 실제 Elvedi를 함께 사용한 관찰 세계다. 다른 세 counterfactual 카드와 수치를 직접 성능 비교하지 않는다.':`가상 Ginczek에 대한 실행 가능한 Elvedi response다. 현재 거리 G ${p.runner_distance_m.toFixed(1)}m / K ${p.beneficiary_distance_m.toFixed(1)}m.`;document.querySelector('.now').innerHTML=`선택: <b>${labels[key]}</b><br>${world}<br>가까운 평가시점 ${p.time_s.toFixed(1)}초에서 G 잔여 <b style="color:var(--orange)">${(p.runner_residual_fraction*100).toFixed(1)}%</b>, K 잔여 <b style="color:var(--violet)">${(p.beneficiary_residual_fraction*100).toFixed(1)}%</b>.`;document.querySelector('.scrub').value=idx;document.querySelector('.time').textContent=`t=+${t.toFixed(2)} s`}
document.querySelector('.response').onchange=e=>setKey(e.target.value);document.querySelector('.scrub').oninput=e=>{idx=Number(e.target.value);render()};document.querySelector('.play').onclick=e=>{playing=!playing;e.target.textContent=playing?'❚❚ 일시정지':'▶ 재생';if(playing)timer=setInterval(()=>{idx=idx>=50?0:idx+1;render()},80);else clearInterval(timer)};cards();explain();drawPaths();render();
</script></body></html>"""


def main() -> None:
    args = parse_args()
    match_id = normalize_bundesliga_match_id(args.match_id)
    directory = args.input_dir / match_id
    result = json.loads(
        (directory / f"frame_{args.frame_id}_influence_dilemma.json").read_text(
            encoding="utf-8"
        )
    )
    original = json.loads(
        (directory / f"frame_{args.frame_id}_attacker_maximin.json").read_text(
            encoding="utf-8"
        )
    )
    metadata = load_bundesliga_match_metadata(
        find_bundesliga_files(args.data_dir, match_id)["matchinfo"]
    )
    data = {
        "result": result,
        "background": original["background_frames"],
        "names": {
            player_id: player.short_name
            for player_id, player in metadata.players.items()
        },
    }
    serialized = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace(
        "</", "<\\/"
    )
    html = (
        TEMPLATE.replace("__DATA__", serialized)
        .replace("__FRAME__", str(args.frame_id))
        .replace("__RUNNER__", result["runner_name"])
        .replace("__BENEFICIARY__", result["beneficiary_name"])
        .replace("__DEFENDER__", result["defender_name"])
    )
    output = directory / f"frame_{args.frame_id}_influence_dilemma_meeting.html"
    output.write_text(html, encoding="utf-8")
    print(f"meeting audit: {output.resolve()}")


if __name__ == "__main__":
    main()
