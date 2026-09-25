#!/usr/bin/env python3
"""Render focal-runner versus named-beneficiary response choices."""

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
    parser.add_argument("--attack-rank", type=int, default=1)
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data/raw/bundesliga-integrated")
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/processed/attacker_maximin_v0_1"),
    )
    return parser.parse_args()


TEMPLATE = r"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Ginczek–Kownacki targeted dilemma</title>
<style>
:root{--bg:#081321;--panel:#111e30;--line:#30445f;--text:#eef4ff;--muted:#adbbd0;--pink:#ff55c8;--cyan:#4ae3d0;--orange:#ff8a4c;--violet:#c98cff;--gold:#ffd14f;--blue:#438cff;--red:#ff6b70}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font-family:Inter,system-ui,sans-serif}main{max-width:1580px;margin:auto;padding:24px}h1{margin:0 0 6px;font-size:2rem}h2,h3{margin:0 0 10px}p{color:var(--muted);line-height:1.5}.tag{display:inline-block;padding:5px 10px;border-radius:99px;background:#263b5a;color:#8ee5ff;font-weight:750}.note{padding:12px 15px;background:#192840;border-left:4px solid var(--violet);border-radius:6px}.layout{display:grid;grid-template-columns:minmax(760px,1fr) 530px;gap:18px}.panel{background:var(--panel);border:1px solid var(--line);border-radius:13px;padding:15px}.controls{display:flex;gap:9px;align-items:center;flex-wrap:wrap;margin:10px 0}select,button{font:inherit;color:var(--text);background:#1b2b43;border:1px solid #49617f;border-radius:8px;padding:8px 11px}button.active{color:var(--gold);border-color:var(--gold)}svg{display:block;width:100%;background:#17733a;border-radius:8px}input[type=range]{width:100%}.time{margin-left:auto}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}.card{background:#17263b;border:1px solid #344a66;border-radius:9px;padding:10px;cursor:pointer}.card.active{border:2px solid var(--cyan);padding:9px}.card h3{font-size:.93rem}.objective{min-height:2.8em;color:var(--muted);font-size:.78rem}.metric{margin-top:6px;font-size:.85rem}.legend{display:grid;grid-template-columns:1fr 1fr;gap:7px;margin-top:10px;font-size:.88rem;color:#c9d4e4}.swatch{display:inline-block;width:20px;border-top:4px solid;margin-right:7px}.box{background:#0e1928;border:1px solid #30445e;border-radius:9px;padding:11px;margin-top:10px}.trade{display:grid;grid-template-columns:1fr 1fr;gap:8px}.value{font-size:1.15rem}.good{color:#8ff0ca}.warn{color:#ffd27a}.bad{color:#ff9d9d}.small{font-size:.83rem}.timeline{display:grid;grid-template-columns:repeat(4,1fr);gap:6px;margin-top:8px}.tick{padding:7px;background:#17263b;border:1px solid #344a66;border-radius:7px;text-align:center;font-size:.8rem}.tick.on{border-color:#55d2a0}.tick.off{border-color:#ff7474}.explain{padding:11px;border-radius:8px;background:#192840;line-height:1.5;margin-top:10px}.bars{display:grid;grid-template-columns:75px 1fr auto;gap:7px;align-items:center;margin:6px 0;font-size:.84rem}.bar{height:7px;background:#273950;border-radius:5px;overflow:hidden}.bar i{display:block;height:100%}table{width:100%;border-collapse:collapse;font-size:.82rem;margin-top:7px}th,td{padding:6px;border-bottom:1px solid #2e415a;text-align:left}th{color:var(--muted)}@media(max-width:1150px){.layout{grid-template-columns:1fr}}@media(max-width:720px){.cards,.trade{grid-template-columns:1fr}.timeline{grid-template-columns:1fr 1fr}}
</style></head><body><main><h1>누굴 막을 것인가? · frame __FRAME__</h1>
<p><span class="tag">targeted dilemma v0.4</span> __RUNNER__의 직접 수신 위협 (R_G)과 __BENEFICIARY__의 수신 위협 (O_K)만 떼어, 같은 실행 가능한 수비 행동들이 두 옵션을 어떻게 바꾸는지 본다. <strong>애니메이션은 전체 2초</strong>이며, 패스 평가는 실제 킥 전 0–0.52초에만 표시한다.</p>
<p class="note"><strong>이 페이지의 질문:</strong> 수비 한 명이 Ginczek을 따라가 직접 옵션을 줄일 것인가, 아니면 Kownacki 쪽 기존 수비를 도와 2대1 커버를 만들 것인가? “다른 옵션”은 이제 모든 팀원의 최댓값이 아니라 <strong>Kownacki로 명시</strong>했다. 팀 전체 OBSO 최대값은 별도 진단값으로만 남긴다.<br><span class="warn">현재 Ginczek 궤적은 기존 t=2 탐색의 공격 후보 1이다. 여기서는 그 궤적의 0–0.52초 구간으로 수비 선택 정의를 검증하며, targeted 기준의 공격 outer maximization은 아직 수행하지 않았다.</span></p>
<div class="layout"><section class="panel"><div class="controls"><label>수비 행동 <select class="response"><option value="direct_cover">① Ginczek 직접 방어</option><option value="beneficiary_cover">② Kownacki 추가 커버</option><option value="dilemma_best" selected>③ 둘 중 큰 위협 최소화</option></select></label><button class="play">▶ 재생</button><span class="time">t=+0.52 s</span></div><div class="controls releases"><b>패스 가능 시점</b></div>
<svg viewBox="0 0 105 68"><defs><marker id="arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="4" markerHeight="4" orient="auto"><path d="M0 0L10 5L0 10z" fill="currentColor"/></marker></defs><rect width="105" height="68" fill="#17733a" stroke="white" stroke-width=".3"/><line x1="52.5" x2="52.5" y1="0" y2="68" stroke="white" stroke-width=".25"/><circle cx="52.5" cy="34" r="9.15" fill="none" stroke="white" stroke-width=".25"/><rect x="0" y="13.84" width="16.5" height="40.32" fill="none" stroke="white" stroke-width=".25"/><rect x="88.5" y="13.84" width="16.5" height="40.32" fill="none" stroke="white" stroke-width=".25"/><g class="paths"></g><g class="moving"></g></svg><input class="scrub" type="range" min="0" max="50" value="13">
<div class="legend"><span><i class="swatch" style="border-color:var(--pink)"></i>가상 Ginczek 움직임</span><span><i class="swatch" style="border-color:var(--cyan)"></i>선택 수비 행동</span><span><i class="swatch" style="border-color:white;border-top-style:dashed"></i>실제 Ginczek 움직임</span><span><i class="swatch" style="border-color:var(--orange)"></i>Ginczek event (R_G)</span><span><i class="swatch" style="border-color:var(--violet)"></i>Kownacki event (O_K)</span><span><i class="swatch" style="border-color:var(--gold)"></i>generic team max 진단</span></div></section>
<aside class="panel"><h2>같은 문제, 세 수비 목적</h2><div class="cards"><div class="card direct_cover" data-key="direct_cover"><h3>① Ginczek 방어</h3><div class="objective">패스 구간 (R_G) 최소화</div><b class="who"></b><div class="metric">R <b class="r"></b></div><div class="metric">K <b class="k"></b></div><div class="metric">max <b class="d"></b></div></div><div class="card beneficiary_cover" data-key="beneficiary_cover"><h3>② Kownacki 커버</h3><div class="objective">패스 구간 (O_K) 최소화</div><b class="who"></b><div class="metric">R <b class="r"></b></div><div class="metric">K <b class="k"></b></div><div class="metric">max <b class="d"></b></div></div><div class="card dilemma_best active" data-key="dilemma_best"><h3>③ pair best</h3><div class="objective">max((R_G,O_K)) 최소화</div><b class="who"></b><div class="metric">R <b class="r"></b></div><div class="metric">K <b class="k"></b></div><div class="metric">max <b class="d"></b></div></div></div>
<div class="explain conclusion"></div><div class="trade"><div class="box"><b style="color:var(--orange)">현재 Ginczek 옵션</b><div class="rnow value"></div></div><div class="box"><b style="color:var(--violet)">현재 Kownacki 옵션</b><div class="know value"></div></div></div><div class="timeline"></div>
<div class="box"><h3>왜 이것이 아직 강한 dilemma는 아닌가?</h3><div class="diagnosis"></div><div class="bars"><span>pair max</span><div class="bar"><i class="pairbar" style="background:var(--violet)"></i></div><b class="pair"></b><span>team max</span><div class="bar"><i class="teambar" style="background:var(--gold)"></i></div><b class="team"></b></div><p class="small">team max는 다른 모든 공격수까지 포함한 원래 팀 위협 진단값이다. targeted pair 최적화와 혼동하지 않는다.</p></div>
<h3 style="margin-top:12px">수비수별 가능한 최적 선택</h3><table><thead><tr><th>수비수</th><th>G 거리</th><th>K 거리</th><th>min R</th><th>min K</th></tr></thead><tbody class="defenders"></tbody></table><p class="small warn">현재 Kownacki는 장면을 이해하기 위한 명시적 표적이다. 전체 데이터에서는 no-run 대비 가치 증가 등 사전 규칙으로 beneficiary를 선택해야 cherry-picking을 피할 수 있다.</p></aside></div></main>
<script id="data" type="application/json">__DATA__</script><script>
const s=JSON.parse(document.getElementById('data').textContent),a=s.attack,td=a.targeted_pass_window_dilemma,names=s.player_names,px=x=>x+52.5,py=y=>34-y;let key='dilemma_best',idx=13,playing=false,timer;const rsel=document.querySelector('.response'),paths=document.querySelector('.paths'),moving=document.querySelector('.moving'),response=()=>td.responses[key],trace=()=>response().trace;
const interp=(path,t)=>{if(t<=path[0][0])return path[0].slice(1);if(t>=path.at(-1)[0])return path.at(-1).slice(1);for(let i=1;i<path.length;i++)if(t<=path[i][0]){let u=path[i-1],v=path[i],q=(t-u[0])/(v[0]-u[0]);return[u[1]+q*(v[1]-u[1]),u[2]+q*(v[2]-u[2])]}},poly=(p,c,d='',w=.5)=>`<polyline points="${p.map(z=>`${px(z[1])},${py(z[2])}`).join(' ')}" fill="none" stroke="${c}" stroke-width="${w}" stroke-dasharray="${d}"/>`,at=()=>s.attack_path_times_s.map((t,i)=>[t,...a.attack_path_xy[i]]),dt=()=>response().defender_path_times_s.map((t,i)=>[t,...response().defender_path_xy[i]]),point=()=>trace().points.find(p=>Math.round(p.release_time_s*25)===idx);
function mark(o,c,label,ball){if(!o||o.event_x==null)return'';let x=px(o.event_x),y=py(o.event_y);return `<line x1="${px(ball[0])}" y1="${py(ball[1])}" x2="${x}" y2="${y}" stroke="${c}" color="${c}" stroke-width=".3" stroke-dasharray=".8 .6" marker-end="url(#arr)"/><path d="M${x-.7},${y-.7}l1.4,1.4m0,-1.4l-1.4,1.4" stroke="${c}" stroke-width=".5"/><text x="${x+.8}" y="${y-.5}" fill="white" font-size="1.05">${label}</text>`}
function setResponse(k){key=k;rsel.value=k;document.querySelectorAll('.card').forEach(c=>c.classList.toggle('active',c.dataset.key===k));paths.innerHTML=poly(at(),'#ff55c8','',.62)+poly(dt(),'#4ae3d0','',.65)+poly(s.observed_attacker_timed,'white','1 .65',.38);update();render()}
function update(){for(let [k,r] of Object.entries(td.responses)){let c=document.querySelector('.card.'+k);c.querySelector('.who').textContent=r.defender_name;c.querySelector('.r').textContent=r.focal_peak.toFixed(6);c.querySelector('.k').textContent=r.beneficiary_peak.toFixed(6);c.querySelector('.d').textContent=r.dilemma_value.toFixed(6)}let dr=td.responses.direct_cover,bk=td.responses.beneficiary_cover,db=td.responses.dilemma_best;document.querySelector('.conclusion').innerHTML=`<b>${dr.defender_name}</b>가 Ginczek을 최대로 억제하면 Kownacki는 ${dr.beneficiary_peak.toFixed(6)}이 남는다. Kownacki 커버 행동은 그 값을 <b>${bk.beneficiary_peak.toFixed(6)}</b>으로 낮추지만, Ginczek은 ${dr.focal_peak.toFixed(6)} → <b>${bk.focal_peak.toFixed(6)}</b>으로 커진다.`;let ratio=db.dilemma_value/db.trace.team_peak;document.querySelector('.pair').textContent=db.dilemma_value.toFixed(6);document.querySelector('.team').textContent=db.trace.team_peak.toFixed(6);document.querySelector('.pairbar').style.width=(ratio*100)+'%';document.querySelector('.teambar').style.width='100%';document.querySelector('.diagnosis').innerHTML=db.focal_peak>db.beneficiary_peak?`현재 OBSO에서는 Ginczek R이 Kownacki O보다 계속 크다. 따라서 pair best는 직접 방어와 같고, <b>trade-off는 있지만 두 선택이 팽팽한 dilemma는 아니다.</b>`:`Kownacki 옵션이 pair의 병목이어서 추가 커버가 pair best를 결정한다.`;document.querySelector('.defenders').innerHTML=Object.values(td.per_defender).map(z=>`<tr><td>${z.defender_name}</td><td>${z.current_distance_to_runner_m.toFixed(1)}m</td><td>${z.current_distance_to_beneficiary_m.toFixed(1)}m</td><td>${z.direct_cover.value.toFixed(5)}</td><td>${z.beneficiary_cover.value.toFixed(5)}</td></tr>`).join('');let kw=trace().points.map(p=>{let o=p.player_options.find(o=>o.player_id===td.beneficiary_id);return{t:p.release_time_s,...o}});document.querySelector('.timeline').innerHTML=kw.map(o=>`<div class="tick ${o.onside_at_release?'on':'off'}"><b>${o.t.toFixed(2)}s</b><br>${o.onside_at_release?'K 온사이드':'K 오프사이드'}<br>${o.value.toFixed(5)}</div>`).join('')}
function render(){let t=idx/25,f=s.background_frames[idx],ap=interp(at(),t),dp=interp(dt(),t),did=response().selection.defender_id,p=point();let g=f.players.filter(x=>x[0]!==s.attacker_id&&x[0]!==did).map(x=>`<circle cx="${px(x[2])}" cy="${py(x[3])}" r=".72" fill="${x[1]===s.attacking_team_id?'#ff7057':'#438cff'}" stroke="${x[0]===td.beneficiary_id?'#c98cff':'white'}" stroke-width="${x[0]===td.beneficiary_id?'.5':'.15'}"><title>${names[x[0]]||x[0]}</title></circle>`).join('')+`<circle cx="${px(f.ball[0])}" cy="${py(f.ball[1])}" r=".43" fill="#111" stroke="white" stroke-width=".15"/>`;if(p){let k=p.player_options.find(o=>o.player_id===td.beneficiary_id);g+=mark(p.focal,'#ff8a4c','G',f.ball)+mark(k,'#c98cff','K',f.ball);document.querySelector('.rnow').textContent=p.focal.value.toFixed(6);document.querySelector('.know').textContent=k.value.toFixed(6)+` · ${k.onside_at_release?'온사이드':'오프사이드'}`}else{document.querySelector('.rnow').textContent=t>td.release_times_s.at(-1)?'평가 안 함':'–';document.querySelector('.know').textContent=t>td.release_times_s.at(-1)?'평가 안 함':'–'}moving.innerHTML=g+`<circle cx="${px(ap[0])}" cy="${py(ap[1])}" r="1.05" fill="#ff55c8" stroke="white" stroke-width=".2"/><circle cx="${px(dp[0])}" cy="${py(dp[1])}" r="1" fill="#438cff" stroke="#4ae3d0" stroke-width=".5"/>`;document.querySelector('.scrub').value=idx;let state=p?' · 패스 평가 시점':(t>td.release_times_s.at(-1)?' · 실제 킥 이후, 궤적만 표시':' · 릴리스 후보 사이');document.querySelector('.time').textContent=`t=+${t.toFixed(2)} s${state}`;document.querySelectorAll('.releases button').forEach(b=>b.classList.toggle('active',Number(b.dataset.idx)===idx))}
td.release_times_s.forEach(t=>{let b=document.createElement('button');b.dataset.idx=Math.round(t*25);b.textContent=t===0?'t=0 · V0':`t=${t.toFixed(2)}s`;b.onclick=()=>{idx=Number(b.dataset.idx);render()};document.querySelector('.releases').appendChild(b)});rsel.onchange=()=>setResponse(rsel.value);document.querySelectorAll('.card').forEach(c=>c.onclick=()=>setResponse(c.dataset.key));document.querySelector('.scrub').oninput=e=>{idx=Number(e.target.value);render()};document.querySelector('.play').onclick=e=>{playing=!playing;e.target.textContent=playing?'❚❚ 일시정지':'▶ 재생';if(playing)timer=setInterval(()=>{idx=idx>=50?0:idx+1;render()},80);else clearInterval(timer)};setResponse('dilemma_best');
</script></body></html>"""


def main() -> None:
    args = parse_args()
    match_id = normalize_bundesliga_match_id(args.match_id)
    directory = args.input_dir / match_id
    payload = json.loads(
        (directory / f"frame_{args.frame_id}_attacker_maximin.json").read_text(
            encoding="utf-8"
        )
    )
    attacks = sorted(
        (
            item
            for item in payload["searched_attacks"]
            if "targeted_pass_window_dilemma" in item
        ),
        key=lambda item: item["best_response"]["value"],
        reverse=True,
    )
    if len(attacks) < args.attack_rank:
        raise ValueError("requested attack has no targeted pass-window result")
    attack = attacks[args.attack_rank - 1]
    targeted = attack["targeted_pass_window_dilemma"]
    metadata = load_bundesliga_match_metadata(
        find_bundesliga_files(args.data_dir, match_id)["matchinfo"]
    )
    data = {
        "attacker_id": payload["attacker_id"],
        "attacking_team_id": payload["attacking_team_id"],
        "attack_path_times_s": payload["attack_path_times_s"],
        "observed_attacker_timed": payload["observed_attacker_timed"],
        "background_frames": payload["background_frames"],
        "attack": attack,
        "player_names": {
            player_id: player.short_name for player_id, player in metadata.players.items()
        },
    }
    serialized = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace(
        "</", "<\\/"
    )
    html = (
        TEMPLATE.replace("__DATA__", serialized)
        .replace("__FRAME__", str(payload["frame_id"]))
        .replace("__RUNNER__", str(payload["attacker_name"]))
        .replace("__BENEFICIARY__", str(targeted["beneficiary_name"]))
    )
    output = directory / f"frame_{args.frame_id}_targeted_dilemma_audit.html"
    output.write_text(html, encoding="utf-8")
    print(f"audit: {output.resolve()}")


if __name__ == "__main__":
    main()
