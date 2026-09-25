#!/usr/bin/env python3
"""Render the pass-release-window defensive-dilemma audit."""

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


TEMPLATE = r"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Pass-window dilemma audit</title>
<style>
:root{--bg:#081321;--panel:#111e30;--line:#30445f;--text:#edf4ff;--muted:#adbbd0;--pink:#ff55c8;--cyan:#4ae3d0;--orange:#ff8a4c;--gold:#ffd04f;--violet:#c68cff;--red:#ff6b6b;--blue:#438cff}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font-family:Inter,system-ui,sans-serif}main{max-width:1580px;margin:auto;padding:24px}h1{margin:0 0 6px;font-size:2rem}h2,h3{margin:0 0 10px}p{color:var(--muted);line-height:1.5}.status{display:inline-block;padding:5px 10px;border-radius:99px;background:#243a58;color:#8de3ff;font-weight:750}.note{padding:12px 15px;background:#192840;border-left:4px solid var(--gold);border-radius:6px}.layout{display:grid;grid-template-columns:minmax(760px,1fr) 520px;gap:18px}.panel{background:var(--panel);border:1px solid var(--line);border-radius:13px;padding:15px}svg{display:block;width:100%;background:#17733a;border-radius:8px}.controls{display:flex;gap:9px;align-items:center;flex-wrap:wrap;margin:10px 0}select,button{font:inherit;color:var(--text);background:#1b2b43;border:1px solid #49617f;border-radius:8px;padding:8px 11px}button.active{border-color:var(--gold);color:var(--gold)}input[type=range]{width:100%}.time{margin-left:auto}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}.card{background:#17263b;border:1px solid #344a66;border-radius:9px;padding:10px;cursor:pointer}.card.active{border:2px solid var(--cyan);padding:9px}.card h3{font-size:.93rem}.objective{font-size:.78rem;color:var(--muted);min-height:2.5em}.metric{margin-top:5px;font-size:.86rem}.release-buttons button{font-size:.86rem}.legend{display:grid;grid-template-columns:1fr 1fr;gap:7px;margin-top:11px;font-size:.88rem;color:#c9d4e4}.swatch{display:inline-block;width:19px;border-top:4px solid;margin-right:7px}.optionbox{background:#0e1928;border:1px solid #30445e;border-radius:9px;padding:11px;margin-top:10px}.optiongrid{display:grid;grid-template-columns:1fr 1fr;gap:9px}.headline{font-size:1.03rem}.small{font-size:.83rem}.warn{color:#ffd47c}.good{color:#8cf0c7}.bad{color:#ff9c9c}.timeline{display:grid;grid-template-columns:repeat(4,1fr);gap:6px;margin-top:8px}.tick{background:#17263b;border:1px solid #344a66;border-radius:7px;padding:7px;text-align:center;font-size:.8rem}.tick.onside{border-color:#50cb9b}.tick.offside{border-color:#ff7171}.bars{height:6px;background:#273950;border-radius:5px;overflow:hidden}.bars i{display:block;height:100%}table{width:100%;border-collapse:collapse;font-size:.84rem;margin-top:7px}th,td{padding:6px;border-bottom:1px solid #2e415a;text-align:left}th{color:var(--muted)}tr.highlight{background:#2a2446}.explain{padding:10px;border-radius:8px;background:#17263b;margin:9px 0;color:#dce7f7;line-height:1.45}@media(max-width:1150px){.layout{grid-template-columns:1fr}}@media(max-width:720px){.cards,.optiongrid{grid-template-columns:1fr}.timeline{grid-template-columns:1fr 1fr}}
</style></head><body><main>
<h1>패스를 낼 수 있는 순간에 어떤 선택이 남는가? · frame __FRAME__</h1>
<p><span class="status">pass-window audit v0.3</span> __ATTACKER__의 공격 후보 1을 <strong>t=2 종점</strong>이 아니라, 볼 소유자가 실제로 패스할 수 있던 <strong>t=0.20–0.52초</strong>에서 다시 평가한다.</p>
<p class="note"><strong>핵심:</strong> t=0은 움직임 전 공통 기준 V0일 뿐 수비 최적화에는 넣지 않는다. 각 릴리스 순간에 오프사이드를 판정하고, 그 순간 공을 받는다면 생기는 Ginczek 옵션 R과 다른 팀원 옵션 O를 계산한다. X는 선수의 이동 목적지가 아니라 <em>해당 선수가 받을 수 있는 잠재적 다음 on-ball event</em>다.</p>
<div class="layout"><section class="panel">
<div class="controls"><label>수비 목적 <select class="response"><option value="direct_cover">① R · Ginczek 직접 옵션 최소화</option><option value="other_cover">② O · 다른 팀원 옵션 최소화</option><option value="global_best" selected>③ max(R,O) · 전체 best response</option></select></label><button class="play">▶ 재생</button><span class="time">t=+0.52 s</span></div>
<div class="controls release-buttons"><b>패스 릴리스 후보</b></div>
<svg viewBox="0 0 105 68"><defs><marker id="arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="4" markerHeight="4" orient="auto"><path d="M0 0 L10 5 L0 10z" fill="currentColor"/></marker></defs><rect width="105" height="68" fill="#17733a" stroke="white" stroke-width=".3"/><line x1="52.5" x2="52.5" y1="0" y2="68" stroke="white" stroke-width=".25"/><circle cx="52.5" cy="34" r="9.15" fill="none" stroke="white" stroke-width=".25"/><rect x="0" y="13.84" width="16.5" height="40.32" fill="none" stroke="white" stroke-width=".25"/><rect x="88.5" y="13.84" width="16.5" height="40.32" fill="none" stroke="white" stroke-width=".25"/><g class="paths"></g><g class="moving"></g></svg>
<input class="scrub" type="range" min="0" max="50" value="13">
<div class="legend"><span><i class="swatch" style="border-color:var(--pink)"></i>가상 Ginczek 움직임</span><span><i class="swatch" style="border-color:var(--cyan)"></i>선택한 수비 대응</span><span><i class="swatch" style="border-color:white;border-top-style:dashed"></i>실제 Ginczek 움직임</span><span><i class="swatch" style="border-color:var(--orange)"></i>R · Ginczek 수신 옵션</span><span><i class="swatch" style="border-color:var(--gold)"></i>O · 가장 큰 다른 옵션</span><span><i class="swatch" style="border-color:var(--violet)"></i>Kownacki 옵션</span></div>
</section><aside class="panel"><h2>세 수비 목적의 결과</h2><div class="cards">
<div class="card direct_cover" data-key="direct_cover"><h3>① 직접 옵션 방어</h3><div class="objective">패스 구간의 Ginczek 최대 R을 최소화</div><b class="who"></b><div class="metric">R<sub>H</sub> <b class="r"></b></div><div class="metric">O<sub>H</sub> <b class="o"></b></div><div class="metric">team <b class="team"></b></div></div>
<div class="card other_cover" data-key="other_cover"><h3>② 다른 옵션 방어</h3><div class="objective">패스 구간의 다른 팀원 최대 O를 최소화</div><b class="who"></b><div class="metric">R<sub>H</sub> <b class="r"></b></div><div class="metric">O<sub>H</sub> <b class="o"></b></div><div class="metric">team <b class="team"></b></div></div>
<div class="card global_best active" data-key="global_best"><h3>③ 전체 best</h3><div class="objective">패스 구간의 max(R,O)를 최소화</div><b class="who"></b><div class="metric">R<sub>H</sub> <b class="r"></b></div><div class="metric">O<sub>H</sub> <b class="o"></b></div><div class="metric">team <b class="team"></b></div></div></div>
<div class="explain conclusion"></div>
<div class="optiongrid"><div class="optionbox"><b style="color:var(--orange)">R · 현재 Ginczek 옵션</b><div class="rnow"></div></div><div class="optionbox"><b style="color:var(--gold)">O · 현재 가장 큰 다른 옵션</b><div class="onow"></div></div></div>
<div class="optionbox"><b style="color:var(--violet)">Kownacki를 빠뜨리지 않았는가?</b><div class="know"></div><div class="timeline"></div><p class="small">Kownacki는 각 패스 릴리스 순간의 오프사이드 판정 후 후보에 포함된다. 단, 포함된다는 것과 그가 O의 최댓값이라는 것은 다르다.</p></div>
<h3 style="margin-top:12px">현재 릴리스 순간의 전체 수신 옵션</h3><table><thead><tr><th>선수</th><th>OBSO</th><th>온사이드</th><th>event 거리</th></tr></thead><tbody class="options"></tbody></table>
<p class="small warn">선수별 t=0 대비 증가는 시간상 변화이며 아직 Ginczek 움직임만의 인과효과는 아니다. 그 인과효과는 같은 장면의 no-run 비교 상태를 추가한 뒤 분리한다.</p>
</aside></div></main>
<script id="data" type="application/json">__DATA__</script><script>
const s=JSON.parse(document.getElementById('data').textContent),a=s.attack,pw=a.pass_window_dilemma,names=s.player_names,px=x=>x+52.5,py=y=>34-y;
const paths=document.querySelector('.paths'),moving=document.querySelector('.moving'),rsel=document.querySelector('.response');let key='global_best',idx=13,playing=false,timer;
const response=()=>pw.responses[key],trace=()=>response().trace,interp=(path,t)=>{if(t<=path[0][0])return path[0].slice(1);if(t>=path.at(-1)[0])return path.at(-1).slice(1);for(let i=1;i<path.length;i++)if(t<=path[i][0]){let u=path[i-1],v=path[i],q=(t-u[0])/(v[0]-u[0]);return[u[1]+q*(v[1]-u[1]),u[2]+q*(v[2]-u[2])];}},poly=(p,c,d='',w=.5)=>`<polyline points="${p.map(z=>`${px(z[1])},${py(z[2])}`).join(' ')}" fill="none" stroke="${c}" stroke-width="${w}" stroke-dasharray="${d}"/>`;
const at=()=>s.attack_path_times_s.map((t,i)=>[t,...a.attack_path_xy[i]]),dt=()=>response().defender_path_times_s.map((t,i)=>[t,...response().defender_path_xy[i]]),releasePoint=()=>trace().points.find(p=>Math.round(p.release_time_s*25)===idx);
function eventMark(o,color,label,ball){if(!o||o.event_x==null)return'';let x=px(o.event_x),y=py(o.event_y);return `<line x1="${px(ball[0])}" y1="${py(ball[1])}" x2="${x}" y2="${y}" stroke="${color}" color="${color}" stroke-width=".28" stroke-dasharray=".8 .6" marker-end="url(#arr)"/><path d="M${x-.7},${y-.7}l1.4,1.4m0,-1.4l-1.4,1.4" stroke="${color}" stroke-width=".5"/><text x="${x+.8}" y="${y-.5}" fill="white" font-size="1.05">${label}</text>`}
function setResponse(k){key=k;rsel.value=k;document.querySelectorAll('.card').forEach(x=>x.classList.toggle('active',x.dataset.key===k));drawPaths();updateCards();render()}
function drawPaths(){paths.innerHTML=poly(at(),'#ff55c8','',.62)+poly(dt(),'#4ae3d0','',.65)+poly(s.observed_attacker_timed,'white','1 .65',.38)}
function updateCards(){let all=pw.responses,max=Math.max(...Object.values(all).flatMap(r=>[r.trace.focal_peak,r.trace.other_peak]));for(let [k,r] of Object.entries(all)){let c=document.querySelector('.card.'+k),t=r.trace;c.querySelector('.who').textContent=r.defender_name;c.querySelector('.r').textContent=t.focal_peak.toFixed(6)+` @${t.focal_peak_time_s.toFixed(2)}s`;c.querySelector('.o').textContent=t.other_peak.toFixed(6)+` @${t.other_peak_time_s.toFixed(2)}s`;c.querySelector('.team').textContent=t.team_peak.toFixed(6)}let dr=all.direct_cover.trace,oo=all.other_cover.trace,gb=all.global_best.trace;document.querySelector('.conclusion').innerHTML=`직접 R을 가장 작게 만든 대응은 <b>${all.direct_cover.defender_name}</b>, 다른 옵션 O를 가장 작게 만든 대응은 <b>${all.other_cover.defender_name}</b>다. 전체 best response는 <b>${all.global_best.defender_name}</b>이며, 그래도 남는 위협은 <b>${gb.team_peak.toFixed(6)}</b>이다.`;updateKownacki()}
function updateKownacki(){let pts=trace().points.map(p=>{let k=p.player_options.find(o=>o.player_id===s.kownacki_id);return{t:p.release_time_s,...k}}),w=trace().player_windows.find(x=>x.player_id===s.kownacki_id);document.querySelector('.timeline').innerHTML=pts.map(p=>`<div class="tick ${p.onside_at_release?'onside':'offside'}"><b>${p.t.toFixed(2)}s</b><br>${p.onside_at_release?'온사이드':'오프사이드'}<br>${p.value.toFixed(5)}</div>`).join('');document.querySelector('.know').innerHTML=`post-onset peak <b>${w.peak_value.toFixed(6)}</b> @ ${w.peak_time_s.toFixed(2)}s · V0 대비 <b class="${w.gain_from_onset>=0?'good':'bad'}">${w.gain_from_onset>=0?'+':''}${w.gain_from_onset.toFixed(6)}</b>`}
function render(){let t=idx/25,f=s.background_frames[idx],ap=interp(at(),t),dp=interp(dt(),t),did=response().selection.defender_id,p=releasePoint();let g=f.players.filter(x=>x[0]!==s.attacker_id&&x[0]!==did).map(x=>{let ring=x[0]===s.kownacki_id?' stroke="#c68cff" stroke-width=".48"':' stroke="white" stroke-width=".15"';return `<circle cx="${px(x[2])}" cy="${py(x[3])}" r=".72" fill="${x[1]===s.attacking_team_id?'#ff7057':'#438cff'}"${ring}><title>${names[x[0]]||x[0]}</title></circle>`}).join('')+`<circle cx="${px(f.ball[0])}" cy="${py(f.ball[1])}" r=".43" fill="#111" stroke="white" stroke-width=".15"/>`;if(p){let k=p.player_options.find(o=>o.player_id===s.kownacki_id);g+=eventMark(p.focal,'#ff8a4c','R',f.ball)+eventMark(p.best_other,'#ffd04f','O · '+p.best_other.player_name,f.ball);if(k.player_id!==p.best_other.player_id)g+=eventMark(k,'#c68cff','Kownacki',f.ball);document.querySelector('.rnow').innerHTML=`${p.focal.value.toFixed(6)} · ${p.focal.onside_at_release?'온사이드':'오프사이드'} · event (${p.focal.event_x?.toFixed(1)??'–'}, ${p.focal.event_y?.toFixed(1)??'–'})`;document.querySelector('.onow').innerHTML=`<b>${p.best_other.player_name}</b> · ${p.best_other.value.toFixed(6)} · event (${p.best_other.event_x?.toFixed(1)??'–'}, ${p.best_other.event_y?.toFixed(1)??'–'})`;let sorted=[...p.player_options].sort((u,v)=>v.value-u.value);document.querySelector('.options').innerHTML=sorted.map(o=>`<tr class="${o.player_id===s.kownacki_id?'highlight':''}"><td>${o.player_name}${o.player_id===s.attacker_id?' (R)':''}</td><td>${o.value.toFixed(6)}</td><td>${o.onside_at_release?'O':'X'}</td><td>${o.player_event_distance_m?.toFixed(1)??'–'}m</td></tr>`).join('')}else{document.querySelector('.rnow').textContent='릴리스 후보 버튼을 누르면 해당 순간의 옵션이 표시된다.';document.querySelector('.onow').textContent='현재 시점은 평가한 패스 릴리스 후보가 아니다.';document.querySelector('.options').innerHTML=''}moving.innerHTML=g+`<circle cx="${px(ap[0])}" cy="${py(ap[1])}" r="1.04" fill="#ff55c8" stroke="white" stroke-width=".2"/><circle cx="${px(dp[0])}" cy="${py(dp[1])}" r="1" fill="#438cff" stroke="#4ae3d0" stroke-width=".5"/>`;document.querySelector('.scrub').value=idx;document.querySelector('.time').textContent=`t=+${t.toFixed(2)} s${p?' · 패스 평가 시점':''}`;document.querySelectorAll('.release-buttons button').forEach(b=>b.classList.toggle('active',Number(b.dataset.idx)===idx))}
pw.release_times_s.forEach(t=>{let b=document.createElement('button');b.dataset.idx=Math.round(t*25);b.textContent=t===0?`t=0 · V0`:`t=${t.toFixed(2)}s`;b.onclick=()=>{idx=Number(b.dataset.idx);render()};document.querySelector('.release-buttons').appendChild(b)});rsel.onchange=()=>setResponse(rsel.value);document.querySelectorAll('.card').forEach(c=>c.onclick=()=>setResponse(c.dataset.key));document.querySelector('.scrub').oninput=e=>{idx=Number(e.target.value);render()};document.querySelector('.play').onclick=e=>{playing=!playing;e.target.textContent=playing?'❚❚ 일시정지':'▶ 재생';if(playing)timer=setInterval(()=>{idx=idx>=50?0:idx+1;render()},80);else clearInterval(timer)};setResponse('global_best');
</script></body></html>"""


def main() -> None:
    args = parse_args()
    match_id = normalize_bundesliga_match_id(args.match_id)
    directory = args.input_dir / match_id
    input_path = directory / f"frame_{args.frame_id}_attacker_maximin.json"
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    attacks = sorted(
        (item for item in payload["searched_attacks"] if "pass_window_dilemma" in item),
        key=lambda item: item["best_response"]["value"],
        reverse=True,
    )
    if len(attacks) < args.attack_rank:
        raise ValueError("requested attack has no pass-window evaluation")
    metadata = load_bundesliga_match_metadata(
        find_bundesliga_files(args.data_dir, match_id)["matchinfo"]
    )
    data = {
        "attacker_id": payload["attacker_id"],
        "attacking_team_id": payload["attacking_team_id"],
        "attack_path_times_s": payload["attack_path_times_s"],
        "observed_attacker_timed": payload["observed_attacker_timed"],
        "background_frames": payload["background_frames"],
        "attack": attacks[args.attack_rank - 1],
        "player_names": {
            player_id: player.short_name for player_id, player in metadata.players.items()
        },
        "kownacki_id": "DFL-OBJ-002FXT",
    }
    serialized = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace(
        "</", "<\\/"
    )
    html = (
        TEMPLATE.replace("__DATA__", serialized)
        .replace("__FRAME__", str(payload["frame_id"]))
        .replace("__ATTACKER__", str(payload["attacker_name"]))
    )
    output = directory / f"frame_{args.frame_id}_pass_window_dilemma_audit.html"
    output.write_text(html, encoding="utf-8")
    print(f"audit: {output.resolve()}")


if __name__ == "__main__":
    main()
