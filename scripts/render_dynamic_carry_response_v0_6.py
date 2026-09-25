#!/usr/bin/env python3
"""Render the moving-target carry-defense audit as a self-contained HTML."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-json",
        type=Path,
        default=Path(
            "data/processed/dynamic_carry_response_v0_6/"
            "dynamic_carry_response_v0_6.json"
        ),
    )
    parser.add_argument(
        "--output-html",
        type=Path,
        default=Path(
            "data/processed/dynamic_carry_response_v0_6/"
            "dynamic_carry_response_v0_6_audit.html"
        ),
    )
    return parser.parse_args()


HTML = r"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Dynamic carry response · v0.6</title><style>
:root{--bg:#071321;--panel:#101f32;--panel2:#14263c;--line:#304862;--text:#f3f7ff;--muted:#aebcd0;--cyan:#45e0d0;--pink:#ff51c7;--violet:#c58cff;--gold:#ffd052;--red:#ff6d5b;--blue:#438dff;--orange:#ff9c43;--green:#65dda0}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font-family:Inter,Pretendard,system-ui,sans-serif}main{max-width:1800px;margin:auto;padding:18px}h1,h2,h3{margin:0 0 8px}p{color:var(--muted);line-height:1.45}.notice{border-left:4px solid var(--orange);background:#192a42;padding:11px 14px;border-radius:7px;margin:10px 0}.controls{display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin:9px 0}.grow{flex:1}button{font:inherit;color:var(--text);background:#192b43;border:1px solid #4a6483;border-radius:8px;padding:8px 11px}button.active{border-color:var(--gold);color:var(--gold);background:#2a2940}.defenders{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px}.defenders button{text-align:left}.defenders small{display:block;color:var(--muted);margin-top:3px}.layout{display:grid;grid-template-columns:minmax(780px,1fr) 650px;gap:15px}.panel{background:var(--panel);border:1px solid var(--line);border-radius:13px;padding:14px}.pitch{display:block;width:100%;background:#17743a;border-radius:9px}.scrub{width:100%}.time{font-variant-numeric:tabular-nums}.legend{display:flex;gap:12px;flex-wrap:wrap;color:#cbd6e6;font-size:.8rem;margin-top:8px}.sw{display:inline-block;width:19px;border-top:4px solid;margin-right:5px}.metrics{display:grid;grid-template-columns:repeat(3,1fr);gap:7px}.metric{background:#152a40;border-radius:8px;padding:8px;min-height:67px}.metric span{font-size:.7rem;color:var(--muted)}.metric strong{display:block;font-size:1.05rem;margin-top:4px}.box{background:var(--panel2);border:1px solid #2d445f;border-radius:9px;padding:10px;margin-top:9px}.small{font-size:.79rem;color:var(--muted)}.warn{color:#ffd082}.mode-table{width:100%;border-collapse:collapse;font-size:.78rem}.mode-table th,.mode-table td{padding:6px;border-bottom:1px solid #304862;text-align:right}.mode-table th:first-child,.mode-table td:first-child{text-align:left}.mode-table tr{cursor:pointer}.mode-table tr.active{background:#203b55;outline:1px solid var(--gold)}.chart{width:100%;height:auto;background:#0c1a2b;border-radius:7px}.formula{font-family:ui-monospace,SFMono-Regular,monospace;color:#d6e6ff}@media(max-width:1280px){.layout{grid-template-columns:1fr}}
</style></head><body><main>
<h1>Dynamic carry response · v0.6</h1>
<p class="notice"><b>이번 검수의 핵심:</b> 수비수가 Kownacki의 과거 위치가 아니라, <b>0.4초 앞의 예상 위치에서 골대 쪽으로 1.5m 앞선 moving target</b>을 계속 차단하는가? 이 페이지는 수비 경로의 시공간 geometry만 검수하며 아직 최종 threat 모델이 아니다.</p>
<div class="defenders"></div>
<div class="controls modes"></div>
<div class="layout"><section class="panel">
<div class="controls"><button class="play">▶ 재생</button><span class="time"></span><span class="grow"></span><span class="current-error"></span></div>
<svg class="pitch" viewBox="0 0 105 68"><rect width="105" height="68" fill="#17743a" stroke="white" stroke-width=".3"/><line x1="52.5" x2="52.5" y1="0" y2="68" stroke="white" stroke-width=".25"/><circle cx="52.5" cy="34" r="9.15" fill="none" stroke="white" stroke-width=".25"/><rect x="0" y="13.84" width="16.5" height="40.32" fill="none" stroke="white" stroke-width=".25"/><rect x="88.5" y="13.84" width="16.5" height="40.32" fill="none" stroke="white" stroke-width=".25"/><g class="static"></g><g class="moving"></g></svg>
<input class="scrub" type="range" min="0" value="0">
<div class="legend"><span><i class="sw" style="border-color:var(--pink)"></i>Kownacki 가상 carry</span><span><i class="sw" style="border-color:white;border-top-style:dashed"></i>실제 수비</span><span><i class="sw" style="border-color:var(--cyan)"></i>선택 수비</span><span><i class="sw" style="border-color:var(--orange);border-top-style:dashed"></i>moving goal-side target</span><span><i class="sw" style="border-color:var(--gold);border-top-style:dotted"></i>0.4초 뒤 Kownacki</span></div>
<p class="small">주황 표적은 매 시점 갱신된다. 청록 수비수가 주황 표적의 궤적을 선점하는지, 이미 지나간 분홍 위치로 뒤늦게 가는지 본다. 빈 흰 원은 같은 시점의 실제 수비 위치다.</p>
</section><aside class="panel"><h2 class="title"></h2><p class="subtitle"></p>
<div class="metrics"><div class="metric"><span>Dynamic 순위</span><strong class="rank"></strong></div><div class="metric"><span>평균 가중 오차</span><strong class="mean"></strong></div><div class="metric"><span>최대 가중 오차</span><strong class="peak"></strong></div><div class="metric"><span>평균 표적 거리</span><strong class="target"></strong></div><div class="metric"><span>Wrong-side 비율</span><strong class="wrong"></strong></div><div class="metric"><span>평균 선수 간 거리</span><strong class="distance"></strong></div></div>
<div class="box explanation"></div><div class="box"><h3>시간별 moving-target 오차</h3><svg class="chart" viewBox="0 0 600 205"></svg><p class="small">낮을수록 좋다. 표적까지 거리와 수비수가 공격수보다 골 반대편에 선 거리를 합친 geometry cost다.</p></div>
<div class="box"><h3>경로 비교</h3><table class="mode-table"><thead><tr><th>수비 경로</th><th>순위</th><th>평균</th><th>Wrong</th></tr></thead><tbody></tbody></table></div>
<p class="small warn">Dynamic rank는 같은 선수의 feasible 경로들 사이 순위다. 실제 수비는 생성 후보가 아니어서 순위가 없다. 이 cost는 최종 defensive value가 아니라 오래된 위치를 쫓는 오류를 잡기 위한 anchor다.</p>
</aside></div></main><script id="data" type="application/json">__DATA__</script><script>
const S=JSON.parse(document.getElementById('data').textContent);let bi=0,mi=4,fi=0,playing=false,timer=null;const $=q=>document.querySelector(q),B=()=>S.branches[bi],M=()=>B().modes[mi],px=x=>52.5+S.attacking_direction*x,py=y=>34-y,fmt=v=>Number(v).toFixed(2);
function interp(path,t){let p=[...path].sort((a,b)=>a[0]-b[0]);if(t<=p[0][0])return p[0].slice(1);if(t>=p[p.length-1][0])return p[p.length-1].slice(1);let j=p.findIndex(x=>x[0]>=t),a=p[j-1],b=p[j],f=(t-a[0])/(b[0]-a[0]);return[a[1]+f*(b[1]-a[1]),a[2]+f*(b[2]-a[2])]}function sampleAt(samples,t){if(t<=samples[0].time_s)return samples[0];if(t>=samples[samples.length-1].time_s)return samples[samples.length-1];let j=samples.findIndex(x=>x.time_s>=t),a=samples[j-1],b=samples[j],f=(t-a.time_s)/(b.time_s-a.time_s),o={};for(let k of Object.keys(a))o[k]=typeof a[k]==='number'?a[k]+f*(b[k]-a[k]):a[k];return o}function poly(path,color,dash='',width=.5,opacity=.95){return`<polyline points="${path.map(p=>`${px(p[1])},${py(p[2])}`).join(' ')}" fill="none" stroke="${color}" stroke-width="${width}" stroke-dasharray="${dash}" opacity="${opacity}"/>`}
function defenders(){document.querySelector('.defenders').innerHTML=S.branches.map((b,i)=>`<button data-bi="${i}" class="${i===bi?'active':''}"><b>${b.defender_name}</b><small>${b.feasible_response_count}개 feasible response</small></button>`).join('');document.querySelectorAll('.defenders button').forEach(x=>x.onclick=()=>{bi=Number(x.dataset.bi);mi=Math.min(4,B().modes.length-1);renderAll()})}
function modes(){document.querySelector('.modes').innerHTML='<b>수비 경로</b>'+B().modes.map((m,i)=>`<button data-mi="${i}" class="${i===mi?'active':''}">${m.label}</button>`).join('');document.querySelectorAll('.modes button').forEach(x=>x.onclick=()=>{mi=Number(x.dataset.mi);renderAll()})}
function drawStatic(){let b=B(),m=M(),target=m.summary.samples.map(x=>[x.time_s,x.target_x,x.target_y]),future=m.summary.samples.map(x=>[x.time_s,x.anticipated_actor_x,x.anticipated_actor_y]);document.querySelector('.static').innerHTML=poly(b.carrier_option.path_txy,'#ff51c7','',.7)+poly(m.response.path_txy,'#45e0d0','',.65)+poly(b.modes[0].response.path_txy,'white','1 .55',.4,.8)+poly(target,'#ff9c43','1 .55',.48,.92)+poly(future,'#ffd052','.25 .55',.32,.65)}
function frameAt(t){return S.background_frames.reduce((a,b)=>Math.abs(b.time_s-t)<Math.abs(a.time_s-t)?b:a)}function player(frame,id){let p=frame.players.find(x=>x[0]===id);return p?[p[2],p[3]]:null}
function drawFrame(){let f=S.background_frames[fi],t=f.time_s,b=B(),m=M(),carrier=interp(b.carrier_option.path_txy,Math.max(0,t)),defender=interp(m.response.path_txy,Math.max(0,t)),actual=interp(b.modes[0].response.path_txy,Math.max(0,t)),s=sampleAt(m.summary.samples,Math.max(S.config.response_delay_seconds,t)),hidden=[S.ball_owner_id,b.defender_id],html=f.players.filter(p=>!hidden.includes(p[0])).map(p=>`<circle cx="${px(p[2])}" cy="${py(p[3])}" r=".67" fill="${p[1]===S.attacking_team_id?'#ff6d5b':'#438dff'}" stroke="white" stroke-width=".12"><title>${S.names[p[0]]||p[0]}</title></circle>`).join('');html+=`<circle cx="${px(carrier[0])}" cy="${py(carrier[1])}" r="1.04" fill="#ff51c7" stroke="white" stroke-width=".2"><title>${S.ball_owner_name}</title></circle><circle cx="${px(carrier[0])}" cy="${py(carrier[1])}" r=".34" fill="#111" stroke="white" stroke-width=".12"/>`;html+=`<circle cx="${px(actual[0])}" cy="${py(actual[1])}" r="1.12" fill="none" stroke="white" stroke-width=".3"/><circle cx="${px(defender[0])}" cy="${py(defender[1])}" r=".9" fill="#438dff" stroke="#45e0d0" stroke-width=".42"/>`;html+=`<circle cx="${px(s.anticipated_actor_x)}" cy="${py(s.anticipated_actor_y)}" r=".75" fill="none" stroke="#ffd052" stroke-width=".28" stroke-dasharray=".35 .25"/><line x1="${px(s.target_x)-.7}" y1="${py(s.target_y)-.7}" x2="${px(s.target_x)+.7}" y2="${py(s.target_y)+.7}" stroke="#ff9c43" stroke-width=".32"/><line x1="${px(s.target_x)-.7}" y1="${py(s.target_y)+.7}" x2="${px(s.target_x)+.7}" y2="${py(s.target_y)-.7}" stroke="#ff9c43" stroke-width=".32"/><line x1="${px(defender[0])}" y1="${py(defender[1])}" x2="${px(s.target_x)}" y2="${py(s.target_y)}" stroke="#ff9c43" stroke-width=".18" stroke-dasharray=".45 .35"/>`;document.querySelector('.moving').innerHTML=html;document.querySelector('.scrub').value=fi;document.querySelector('.time').textContent=`t=${t>=0?'+':''}${t.toFixed(2)}초`;document.querySelector('.current-error').innerHTML=`현재 표적오차 <b>${fmt(s.weighted_error_m)}m</b>`;chart()}
const colors=['white','#c58cff','#65dda0','#ffd052','#45e0d0'];function chart(){let modes=B().modes,max=Math.max(...modes.flatMap(m=>m.summary.samples.map(s=>s.weighted_error_m)),1),left=38,right=585,top=15,bottom=172,t0=S.config.response_delay_seconds,t1=S.horizon_seconds,x=t=>left+(t-t0)/(t1-t0)*(right-left),y=v=>bottom-v/max*(bottom-top),html=`<line x1="${left}" y1="${bottom}" x2="${right}" y2="${bottom}" stroke="#6f8198"/><line x1="${left}" y1="${top}" x2="${left}" y2="${bottom}" stroke="#6f8198"/><text x="4" y="${top+5}" fill="#aebcd0" font-size="11">${max.toFixed(1)}m</text><text x="10" y="${bottom+4}" fill="#aebcd0" font-size="11">0</text>`;modes.forEach((m,i)=>{let pts=m.summary.samples.map(s=>`${x(s.time_s)},${y(s.weighted_error_m)}`).join(' ');html+=`<polyline points="${pts}" fill="none" stroke="${colors[i]}" stroke-width="${i===mi?3:1.2}" opacity="${i===mi?1:.45}"/>`});let t=Math.max(t0,S.background_frames[fi].time_s);html+=`<line x1="${x(t)}" y1="${top}" x2="${x(t)}" y2="${bottom}" stroke="#ff9c43" stroke-width="1.5"/><text x="${left}" y="196" fill="#aebcd0" font-size="11">${t0.toFixed(1)}s</text><text x="${right-24}" y="196" fill="#aebcd0" font-size="11">${t1.toFixed(1)}s</text>`;document.querySelector('.chart').innerHTML=html}
function details(){let b=B(),m=M(),s=m.summary;document.querySelector('.title').textContent=`${b.defender_name} → ${S.ball_owner_name}`;document.querySelector('.subtitle').innerHTML=`선택 경로 <b>${m.label}</b> · carry endpoint <b>(${b.carrier_option.event_x.toFixed(1)}, ${b.carrier_option.event_y.toFixed(1)}) @ ${b.carrier_option.event_time_s.toFixed(1)}초</b>`;document.querySelector('.rank').textContent=m.dynamic_rank?`${m.dynamic_rank}/${b.feasible_response_count}`:'관측 경로';document.querySelector('.mean').textContent=fmt(s.mean_weighted_error_m)+'m';document.querySelector('.peak').textContent=fmt(s.peak_weighted_error_m)+'m';document.querySelector('.target').textContent=fmt(s.mean_target_error_m)+'m';document.querySelector('.wrong').textContent=(100*s.wrong_side_fraction).toFixed(0)+'%';document.querySelector('.distance').textContent=fmt(s.mean_actor_defender_distance_m)+'m';document.querySelector('.explanation').innerHTML=m.key==='dynamic_goal_side'?'<h3>새 경로가 의미하는 것</h3>모든 feasible 경로를 Kownacki의 <b>전체 carry 궤적</b>과 같은 시간축에서 비교했다. 매 시점 0.4초 앞의 위치를 예상하고 그보다 골대 쪽인 주황 표적을 가장 안정적으로 선점하는 경로다.':'<h3>비교 경로</h3>이 경로는 v0.5의 기존 선택이다. 시간별 주황 표적과 청록 수비수의 간격을 보고, Kownacki가 이미 지나간 공간으로 뒤늦게 이동하는 구간이 있는지 확인한다.';document.querySelector('.mode-table tbody').innerHTML=b.modes.map((x,i)=>`<tr data-mi="${i}" class="${i===mi?'active':''}"><td>${x.label}</td><td>${x.dynamic_rank||'—'}</td><td>${fmt(x.summary.mean_weighted_error_m)}m</td><td>${(100*x.summary.wrong_side_fraction).toFixed(0)}%</td></tr>`).join('');document.querySelectorAll('.mode-table tbody tr').forEach(x=>x.onclick=()=>{mi=Number(x.dataset.mi);renderAll()})}
function renderAll(){defenders();modes();drawStatic();details();drawFrame()}function init(){fi=Math.max(0,S.background_frames.findIndex(f=>f.time_s>=0));document.querySelector('.scrub').max=S.background_frames.length-1;document.querySelector('.scrub').oninput=e=>{fi=Number(e.target.value);drawFrame()};document.querySelector('.play').onclick=e=>{playing=!playing;e.target.textContent=playing?'❚❚ 일시정지':'▶ 재생';if(playing)timer=setInterval(()=>{fi=fi>=S.background_frames.length-1?Math.max(0,S.background_frames.findIndex(f=>f.time_s>=0)):fi+1;drawFrame()},70);else clearInterval(timer)};renderAll()}init();
</script></body></html>"""


def main() -> None:
    args = parse_args()
    data = json.loads(args.input_json.read_text(encoding="utf-8"))
    html = HTML.replace(
        "__DATA__",
        json.dumps(data, ensure_ascii=False).replace("</", "<\/"),
    )
    args.output_html.parent.mkdir(parents=True, exist_ok=True)
    args.output_html.write_text(html, encoding="utf-8")
    print(f"output: {args.output_html.resolve()}")


if __name__ == "__main__":
    main()
