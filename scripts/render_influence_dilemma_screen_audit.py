#!/usr/bin/env python3
"""Render full-resolution influence-dilemma candidates for visual review."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from offball_value.bundesliga import find_bundesliga_files, load_bundesliga_match_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-json",
        type=Path,
        default=Path(
            "data/processed/influence_dilemma_refined_v0_1/refined_scenes.json"
        ),
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data/raw/bundesliga-integrated")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/processed/influence_dilemma_refined_v0_1/"
            "influence_dilemma_audit.html"
        ),
    )
    return parser.parse_args()


TEMPLATE = r"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Influence dilemma candidates</title><style>
:root{--bg:#081321;--panel:#111e30;--line:#30445f;--text:#f2f6ff;--muted:#afbdd1;--pink:#ff55c8;--cyan:#4ae3d0;--orange:#ff9255;--violet:#c98cff;--blue:#438cff;--red:#ff705b;--gold:#ffd452;--green:#7cebb7}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font-family:Inter,Pretendard,system-ui,sans-serif}main{max-width:1600px;margin:auto;padding:22px}h1{margin:0 0 5px}p{color:var(--muted);line-height:1.45}.notice{padding:10px 14px;background:#192840;border-left:4px solid var(--gold);border-radius:7px}.layout{display:grid;grid-template-columns:minmax(760px,1fr) 520px;gap:16px}.panel{padding:14px;background:var(--panel);border:1px solid var(--line);border-radius:12px}.controls{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin:8px 0}select,button,textarea{font:inherit;color:var(--text);background:#1b2b43;border:1px solid #49617f;border-radius:8px;padding:8px 10px}.time{margin-left:auto}svg{display:block;width:100%;background:#17733a;border-radius:8px}input{width:100%}.cards{display:grid;grid-template-columns:1fr 1fr;gap:8px}.card{padding:9px;background:#17263b;border:1px solid #344a66;border-radius:8px;cursor:pointer}.card.active{border:2px solid var(--cyan);padding:8px}.card h3{margin:0 0 6px;font-size:.95rem}.metric{display:grid;grid-template-columns:64px 1fr 45px;gap:6px;align-items:center;margin:5px 0;font-size:.81rem}.bar{height:7px;background:#283b54;border-radius:5px;overflow:hidden}.bar i{height:100%;display:block}.box{background:#0e1928;border:1px solid #30445e;border-radius:9px;padding:10px;margin-top:9px;line-height:1.45}.big{font-size:1.14rem}.warn{color:#ffd27a}.good{color:var(--green)}.legend{display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-top:8px;font-size:.84rem;color:#ccd7e6}.sw{display:inline-block;width:20px;border-top:4px solid;margin-right:6px}.review textarea{display:block;width:100%;min-height:62px;resize:vertical;margin-top:7px}.review-row{display:flex;gap:7px;align-items:center;flex-wrap:wrap}.saved{color:var(--green);font-size:.84rem}@media(max-width:1120px){.layout{grid-template-columns:1fr}}
</style></head><body><main><h1>강한 defensive trade-off 후보 · full-resolution audit</h1><p class="notice">검수된 run onset 48개를 coarse screening한 뒤 상위 후보를 <b>1 m influence grid와 전체 feasible defender actions</b>로 재평가했다. runner와 beneficiary는 실제 움직임이며, 청록색 defender만 counterfactual이다. 이 페이지는 장면 타당성 검수용이며 최종 prevalence 결과가 아니다.</p><div class="layout"><section class="panel"><div class="controls"><label>장면 <select class="scene"></select></label><label>수비 선택 <select class="response"><option value="direct_cover">① runner 방어</option><option value="compromise" selected>② compromise</option><option value="beneficiary_cover">③ teammate 방어</option><option value="observed_reference">④ 실제 수비</option></select></label><button class="play">▶ 재생</button><span class="time"></span></div><svg viewBox="0 0 105 68"><rect width="105" height="68" fill="#17733a" stroke="white" stroke-width=".3"/><line x1="52.5" x2="52.5" y1="0" y2="68" stroke="white" stroke-width=".25"/><circle cx="52.5" cy="34" r="9.15" fill="none" stroke="white" stroke-width=".25"/><rect x="0" y="13.84" width="16.5" height="40.32" fill="none" stroke="white" stroke-width=".25"/><rect x="88.5" y="13.84" width="16.5" height="40.32" fill="none" stroke="white" stroke-width=".25"/><g class="paths"></g><g class="moving"></g></svg><input class="scrub" type="range" min="0" max="50" value="0"><div class="legend"><span><i class="sw" style="border-color:var(--pink)"></i>runner 실제 궤적</span><span><i class="sw" style="border-color:var(--violet)"></i>teammate 실제 궤적</span><span><i class="sw" style="border-color:var(--cyan)"></i>선택 수비 궤적</span><span><i class="sw" style="border-color:white;border-top-style:dashed"></i>실제 수비 reference</span></div></section><aside class="panel"><h2 class="title"></h2><p class="subtitle"></p><div class="cards"></div><div class="box summary"></div><div class="box"><h3>이 장면에서 볼 것</h3><p>①과 ③의 청록색 endpoint와 진행 방향이 축구적으로 서로 다른 책임을 나타내는가?</p><p>②가 실제로 두 위협 사이에 위치하는가, 아니면 이상한 제3의 움직임인가?</p><p>선택 변화가 runner·teammate 공간을 막는 것으로 보이는가, 단순히 선수에게 가까워지는 것으로만 보이는가?</p></div><div class="box"><h3>검수 메모</h3><p>이 단계에서는 수치보다 궤적의 축구적 타당성을 우선한다. offside, pass window, ball transition을 아직 최종 통합하지 않았으므로 공격 위협의 최종 크기로 해석하지 않는다.</p></div></aside></div></main><script id="data" type="application/json">__DATA__</script><script>
const D=JSON.parse(document.getElementById('data').textContent),px=x=>x+52.5,py=y=>34-y;let si=0,key='compromise',idx=0,playing=false,timer;const S=()=>D.scenes[si],R=()=>S().responses[key],interp=(p,t)=>{if(t<=p[0][0])return p[0].slice(1);if(t>=p.at(-1)[0])return p.at(-1).slice(1);for(let i=1;i<p.length;i++)if(t<=p[i][0]){let u=p[i-1],v=p[i],q=(t-u[0])/(v[0]-u[0]);return[u[1]+q*(v[1]-u[1]),u[2]+q*(v[2]-u[2])]}};const timedResp=r=>r.times_s.map((t,i)=>[t,...r.path_xy[i]]),poly=(p,c,d='',w=.58)=>`<polyline points="${p.map(z=>`${px(z[1])},${py(z[2])}`).join(' ')}" fill="none" stroke="${c}" stroke-width="${w}" stroke-dasharray="${d}"/>`,near=(r,t)=>r.points.reduce((u,v)=>Math.abs(v.time_s-t)<Math.abs(u.time_s-t)?v:u);const labels={direct_cover:'① runner 방어',compromise:'② compromise',beneficiary_cover:'③ teammate 방어',observed_reference:'④ 실제 수비'};
function setupScenes(){document.querySelector('.scene').innerHTML=D.scenes.map((s,i)=>`<option value="${i}">${i+1}. frame ${s.frame_id} · ${s.runner_name}–${s.beneficiary_name}</option>`).join('')}
function cards(){let s=S();document.querySelector('.cards').innerHTML=Object.entries(labels).map(([k,l])=>{let r=s.responses[k];return `<div class="card ${k===key?'active':''}" data-k="${k}"><h3>${l}</h3><div class="metric"><span>R 잔여</span><div class="bar"><i style="width:${r.runner_horizon_mean*100}%;background:var(--orange)"></i></div><b>${(r.runner_horizon_mean*100).toFixed(1)}%</b></div><div class="metric"><span>O 잔여</span><div class="bar"><i style="width:${r.beneficiary_horizon_mean*100}%;background:var(--violet)"></i></div><b>${(r.beneficiary_horizon_mean*100).toFixed(1)}%</b></div></div>`}).join('');document.querySelectorAll('.card').forEach(c=>c.onclick=()=>setKey(c.dataset.k))}
function describe(){let s=S(),d=s.responses.direct_cover,b=s.responses.beneficiary_cover;document.querySelector('.title').textContent=`${s.runner_name} → ${s.beneficiary_name}`;document.querySelector('.subtitle').innerHTML=`수비수 <b>${s.defender_name}</b> · ${s.evaluated_action_count}개 feasible response · 극점 endpoint <b>${s.direct_terminal_separation_m.toFixed(1)}m</b> 분리`;document.querySelector('.summary').innerHTML=`<span class="big">양방향 branch gap</span><br>teammate를 막으면 runner가 <b style="color:var(--orange)">+${(s.direct_branch_gap*100).toFixed(1)}%p</b>, runner를 막으면 teammate가 <b style="color:var(--violet)">+${(s.beneficiary_branch_gap*100).toFixed(1)}%p</b> 남는다.<br>가장 작은 branch gap <b>${(s.minimum_branch_gap*100).toFixed(1)}%p</b> · compromise regret <b>${(s.normalized_compromise_regret*100).toFixed(1)}%</b>.<br><span class="warn">이 값들은 residual influence 비율의 차이지 득점확률 차이가 아니다.</span>`}
function setKey(k){key=k;document.querySelector('.response').value=k;cards();draw();render()}
function draw(){let s=S();document.querySelector('.paths').innerHTML=poly(s.runner_actual_timed,'#ff55c8')+poly(s.beneficiary_actual_timed,'#c98cff','1 .5',.42)+poly(timedResp(R()),'#4ae3d0')+(key==='observed_reference'?'':poly(s.defender_actual_timed,'white','1 .65',.38))}
function render(){let s=S(),t=idx/25,f=s.background_frames[idx],a=interp(s.runner_actual_timed,t),b=interp(s.beneficiary_actual_timed,t),d=interp(timedResp(R()),t);let g=f.players.filter(p=>![s.runner_id,s.beneficiary_id,s.defender_id].includes(p[0])).map(p=>`<circle cx="${px(p[2])}" cy="${py(p[3])}" r=".7" fill="${p[1]===s.attacking_team_id?'#ff705b':'#438cff'}" stroke="white" stroke-width=".14"><title>${D.names[p[0]]||p[0]}</title></circle>`).join('');g+=`<circle cx="${px(f.ball[0])}" cy="${py(f.ball[1])}" r=".42" fill="#111" stroke="white" stroke-width=".15"/><circle cx="${px(a[0])}" cy="${py(a[1])}" r="1.05" fill="#ff55c8" stroke="white" stroke-width=".22"><title>${s.runner_name}</title></circle><circle cx="${px(b[0])}" cy="${py(b[1])}" r=".98" fill="#ff705b" stroke="#c98cff" stroke-width=".5"><title>${s.beneficiary_name}</title></circle><circle cx="${px(d[0])}" cy="${py(d[1])}" r="1" fill="#438cff" stroke="#4ae3d0" stroke-width=".52"><title>${s.defender_name}</title></circle>`;document.querySelector('.moving').innerHTML=g;document.querySelector('.scrub').value=idx;document.querySelector('.time').textContent=`t=+${t.toFixed(2)}s`}
function setScene(i){si=Number(i);idx=0;key='compromise';document.querySelector('.response').value=key;cards();describe();draw();render()}document.querySelector('.scene').onchange=e=>setScene(e.target.value);document.querySelector('.response').onchange=e=>setKey(e.target.value);document.querySelector('.scrub').oninput=e=>{idx=Number(e.target.value);render()};document.querySelector('.play').onclick=e=>{playing=!playing;e.target.textContent=playing?'❚❚ 일시정지':'▶ 재생';if(playing)timer=setInterval(()=>{idx=idx>=50?0:idx+1;render()},80);else clearInterval(timer)};setupScenes();setScene(0);
</script></body></html>"""


def main() -> None:
    args = parse_args()
    scenes = json.loads(args.input_json.read_text(encoding="utf-8"))
    names = {}
    for match_id in sorted({scene["match_id"] for scene in scenes}):
        metadata = load_bundesliga_match_metadata(
            find_bundesliga_files(args.data_dir, match_id)["matchinfo"]
        )
        names.update(
            {player_id: player.short_name for player_id, player in metadata.players.items()}
        )
    data = json.dumps(
        {"scenes": scenes, "names": names},
        ensure_ascii=False,
        separators=(",", ":"),
    ).replace("</", "<\\/")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(TEMPLATE.replace("__DATA__", data), encoding="utf-8")
    print(f"audit: {args.output.resolve()}")


if __name__ == "__main__":
    main()
