#!/usr/bin/env python3
"""Render an interactive audit for the screened attacker maximin result."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path

from offball_value.action_value import decompose_threat_by_nearest_attacker
from offball_value.bundesliga import (
    find_bundesliga_files,
    load_bundesliga_frames,
    load_bundesliga_match_metadata,
    normalize_bundesliga_match_id,
)
from offball_value.pass_dynamics import ArrivalModelConfig, estimate_frame_velocities
from offball_value.defender_best_response import (
    DefenderBestResponseConfig,
    build_local_counterfactual_state,
    prepare_observed_background_sequence,
)
from offball_value.reference_obso import evaluate_reference_obso


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


def _decompose_selected_attacks(
    payload: dict[str, object], data_dir: Path
) -> None:
    """Materialize the explanation wrapper for every searched candidate."""

    if all(
        "fixed_decomposition" in item
        and "best_response_decomposition" in item
        for item in payload["searched_attacks"]
    ):
        return
    files = find_bundesliga_files(data_dir, payload["match_id"])
    metadata = load_bundesliga_match_metadata(files["matchinfo"])
    frame_id = int(payload["frame_id"])
    frames = load_bundesliga_frames(
        files["positions"], range(frame_id - 10, frame_id + 51)
    )
    background = prepare_observed_background_sequence(
        frames, frame_id, DefenderBestResponseConfig()
    ).states[-1]
    team_id = str(payload["attacking_team_id"])
    direction = int(payload["attacking_direction"])
    attacker_id = str(payload["attacker_id"])
    goalkeeper_ids = tuple(
        goalkeeper_id
        for team_id_key in metadata.teams
        if (goalkeeper_id := metadata.goalkeeper_id(team_id_key)) is not None
    )
    times = tuple(float(value) for value in payload["attack_path_times_s"])

    def one(item, use_response: bool):
        defender_id = (
            str(item["best_response"]["defender_id"]) if use_response else None
        )
        frame, velocities = build_local_counterfactual_state(
            background,
            attacker_id,
            item["attack_path_xy"],
            times,
            defender_id=defender_id,
            defender_path_xy=(
                item["best_defender_path_xy"] if use_response else None
            ),
            defender_path_times_s=(
                item["best_defender_path_times_s"] if use_response else None
            ),
        )
        surface = evaluate_reference_obso(
            frame,
            team_id,
            direction,
            velocities=velocities,
            goalkeeper_ids=goalkeeper_ids,
            apply_offside=True,
        )
        result = decompose_threat_by_nearest_attacker(
            surface, frame, team_id, attacker_id, direction
        )
        response = {
            "team_maximum": result.team_maximum,
            "focal_maximum": result.focal_maximum,
            "focal_maximum_x": result.focal_maximum_x,
            "focal_maximum_y": result.focal_maximum_y,
            "other_maximum": result.other_maximum,
            "other_maximum_x": result.other_maximum_x,
            "other_maximum_y": result.other_maximum_y,
            "other_player_id": result.other_player_id,
            "focal_offside": result.focal_offside,
        }
        if result.other_player_id in metadata.players:
            response["other_player_name"] = metadata.players[
                result.other_player_id
            ].short_name
        return response

    for index, item in enumerate(payload["searched_attacks"], start=1):
        item["fixed_decomposition"] = one(item, False)
        item["best_response_decomposition"] = one(item, True)
        if "dilemma_responses" in item:
            for response in item["dilemma_responses"].values():
                defender_id = str(response["selection"]["defender_id"])
                frame, velocities = build_local_counterfactual_state(
                    background,
                    attacker_id,
                    item["attack_path_xy"],
                    times,
                    defender_id=defender_id,
                    defender_path_xy=response["defender_path_xy"],
                    defender_path_times_s=response["defender_path_times_s"],
                )
                surface = evaluate_reference_obso(
                    frame,
                    team_id,
                    direction,
                    velocities=velocities,
                    goalkeeper_ids=goalkeeper_ids,
                    apply_offside=True,
                )
                split = decompose_threat_by_nearest_attacker(
                    surface, frame, team_id, attacker_id, direction
                )
                response["decomposition"] = asdict(split)
                if split.other_player_id in metadata.players:
                    response["decomposition"]["other_player_name"] = (
                        metadata.players[split.other_player_id].short_name
                    )
        print(
            f"decomposed attack {index}/{len(payload['searched_attacks'])}", flush=True
        )


def _add_decision_baseline(payload: dict[str, object], data_dir: Path) -> None:
    """Add the agreed t=0 run-onset baseline to older computed payloads."""

    if "decision_baseline" in payload:
        return
    files = find_bundesliga_files(data_dir, payload["match_id"])
    metadata = load_bundesliga_match_metadata(files["matchinfo"])
    frame_id = int(payload["frame_id"])
    frames = load_bundesliga_frames(
        files["positions"], range(frame_id - 10, frame_id + 1)
    )
    frame = frames[frame_id]
    history = tuple(frames[target] for target in range(frame_id - 10, frame_id + 1))
    velocities = estimate_frame_velocities(
        history, frame_id, ArrivalModelConfig(history_seconds=0.4)
    )
    goalkeeper_ids = tuple(
        goalkeeper_id
        for team_id in metadata.teams
        if (goalkeeper_id := metadata.goalkeeper_id(team_id)) is not None
    )
    team_id = str(payload["attacking_team_id"])
    direction = int(payload["attacking_direction"])
    attacker_id = str(payload["attacker_id"])
    surface = evaluate_reference_obso(
        frame,
        team_id,
        direction,
        velocities=velocities,
        goalkeeper_ids=goalkeeper_ids,
        apply_offside=True,
    )
    split = decompose_threat_by_nearest_attacker(
        surface, frame, team_id, attacker_id, direction
    )
    payload["decision_baseline"] = {
        "time_s": 0.0,
        "team_maximum": surface.maximum,
        "maximum_x": surface.maximum_position[0],
        "maximum_y": surface.maximum_position[1],
        "decomposition": {
            "team_maximum": split.team_maximum,
            "focal_maximum": split.focal_maximum,
            "focal_maximum_x": split.focal_maximum_x,
            "focal_maximum_y": split.focal_maximum_y,
            "other_maximum": split.other_maximum,
            "other_maximum_x": split.other_maximum_x,
            "other_maximum_y": split.other_maximum_y,
            "other_player_id": split.other_player_id,
            "focal_offside": split.focal_offside,
        },
        "velocity_model": "causal 0.4 s history at run onset",
    }
    for item in payload["searched_attacks"]:
        item["fixed_gain_from_onset"] = item["fixed"]["value"] - surface.maximum
        item["robust_gain_from_onset"] = (
            item["best_response"]["value"] - surface.maximum
        )
    winning_id = payload["winner"]["attack"]["action_id"]
    payload["winner"] = next(
        item
        for item in payload["searched_attacks"]
        if item["attack"]["action_id"] == winning_id
    )
    payload["winner"]["robust_rank"] = 1


def _render(payload: dict[str, object]) -> str:
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace(
        "</", "<\\/"
    )
    return f"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>수비 선택 강요 audit</title>
<style>
:root{{--bg:#091321;--panel:#111d2e;--line:#30415c;--text:#eef4ff;--muted:#aebbd0;--pink:#ff56c8;--cyan:#49e1cc;--gold:#ffc547;--orange:#ff884d;--red:#ff5a63;--blue:#428cff}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font-family:Inter,system-ui,sans-serif}}main{{max-width:1580px;margin:auto;padding:24px}}h1{{margin:0 0 6px;font-size:2rem}}h2{{margin:0 0 10px}}p{{color:var(--muted);line-height:1.5}}.status{{display:inline-block;padding:5px 10px;border-radius:999px;background:#263957;color:#8fdcff;font-weight:700}}.note{{padding:12px 15px;background:#18253a;border-left:4px solid var(--gold);border-radius:5px}}.layout{{display:grid;grid-template-columns:minmax(760px,1fr) 500px;gap:18px}}.panel{{background:var(--panel);border:1px solid var(--line);border-radius:13px;padding:15px}}svg{{display:block;width:100%;background:#17733a;border-radius:8px}}select,button{{font:inherit;color:var(--text);background:#1b2a41;border:1px solid #49607f;border-radius:8px;padding:8px 11px}}select{{max-width:100%}}.controls{{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:11px 0}}input[type=range]{{width:100%}}.time{{margin-left:auto}}.metrics{{display:grid;grid-template-columns:1fr auto;gap:6px 12px;margin:9px 0}}.metrics span:nth-child(odd){{color:var(--muted)}}.cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}}.card{{background:#17253a;border:1px solid #324762;border-radius:9px;padding:10px;cursor:pointer}}.card.active{{border:2px solid var(--cyan);padding:9px}}.card h3{{margin:0 0 6px;font-size:.96rem}}.card .objective{{font-size:.78rem;color:var(--muted);min-height:2.2em}}.bar{{height:7px;background:#27374e;border-radius:5px;overflow:hidden;margin:3px 0 7px}}.bar i{{display:block;height:100%}}.legend{{display:grid;grid-template-columns:1fr 1fr;gap:7px;font-size:.9rem;color:#c8d3e3}}.swatch{{display:inline-block;width:20px;border-top:4px solid;margin-right:7px}}.warn{{color:#ffd17a}}.small{{font-size:.84rem}}.optionbox{{background:#101a29;border:1px solid #31435c;border-radius:9px;padding:10px;margin-top:9px}}.optiongrid{{display:grid;grid-template-columns:1fr 1fr;gap:10px}}.formula{{font-size:.82rem;color:#bfccdc}}table{{border-collapse:collapse;width:100%;font-size:.86rem}}td,th{{padding:6px;border-bottom:1px solid #2e4058;text-align:left}}th{{color:#aebbd0}}@media(max-width:1120px){{.layout{{grid-template-columns:1fr}}}}@media(max-width:700px){{.cards{{grid-template-columns:1fr}}}}
</style></head><body><main><h1>어떤 수비 선택을 강요했는가? · frame {payload['frame_id']}</h1><p><span class="status">dilemma audit v0.2</span> {payload['attacker_name']}의 가상 오프더볼 움직임에 대해 세 가지 수비 목적을 직접 비교한다.</p>
<p class="note"><strong>읽는 순서:</strong> 분홍색 공격 움직임을 고정하고 → <strong>직접 옵션 R 방어</strong>, <strong>다른 팀 옵션 O 방어</strong>, <strong>전체 best response</strong>를 바꿔 재생한다. X는 선수의 목적지가 아니라 공이 전달되고 공격팀이 통제할 수 있는 <em>잠재적 다음 on-ball event 위치</em>다. 점선 화살표는 실제 패스가 아니라 OBSO transition 해석이다.</p>
<div class="layout"><section class="panel"><div class="controls"><label>공격 후보 <select class="candidate"></select></label><label>수비 목적 <select class="response"><option value="direct_cover">R 직접 옵션 최소화</option><option value="other_cover">O 다른 옵션 최소화</option><option value="global_best" selected>max(R,O) 최소화 · 전체 best</option></select></label><button class="play">▶ 재생</button><span class="time">t=0.00 s</span></div><svg viewBox="0 0 105 68"><defs><marker id="arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="4" markerHeight="4" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10z" fill="#fff"/></marker></defs><rect width="105" height="68" fill="#17733a" stroke="white" stroke-width=".3"/><line x1="52.5" y1="0" x2="52.5" y2="68" stroke="white" stroke-width=".25"/><circle cx="52.5" cy="34" r="9.15" fill="none" stroke="white" stroke-width=".25"/><rect x="0" y="13.84" width="16.5" height="40.32" fill="none" stroke="white" stroke-width=".25"/><rect x="88.5" y="13.84" width="16.5" height="40.32" fill="none" stroke="white" stroke-width=".25"/><g class="landscape"></g><g class="paths"></g><g class="moving"></g></svg><input class="scrub" type="range" min="0" max="50" value="0"><div class="controls"><label><input class="show-landscape" type="checkbox"> 302개 endpoint</label><label><input class="show-observed" type="checkbox" checked> 실제 공격 궤적</label><label><input class="show-options" type="checkbox" checked> R/O 잠재 이벤트</label></div><div class="legend"><span><i class="swatch" style="border-color:var(--pink)"></i>가상 공격 움직임</span><span><i class="swatch" style="border-color:var(--cyan)"></i>선택한 수비 대응</span><span><i class="swatch" style="border-color:white;border-top-style:dashed"></i>실제 공격 움직임</span><span><i class="swatch" style="border-color:var(--orange)"></i>직접 옵션 R</span><span><i class="swatch" style="border-color:var(--gold)"></i>다른 팀 옵션 O</span><span><i class="swatch" style="border-color:white;border-top-style:dotted"></i>공→잠재 이벤트 연결</span></div></section>
<aside class="panel"><h2>세 수비 선택의 대가</h2><div class="cards"><div class="card direct_cover" data-key="direct_cover"><h3>① Ginczek 직접 방어</h3><div class="objective">R을 가장 작게 만드는 실행 가능한 대응</div><b class="who"></b><div>R <b class="r"></b></div><div class="bar"><i class="rbar" style="background:var(--orange)"></i></div><div>O <b class="o"></b></div><div class="bar"><i class="obar" style="background:var(--gold)"></i></div><div>team max <b class="team"></b></div></div><div class="card other_cover" data-key="other_cover"><h3>② 다른 옵션 방어</h3><div class="objective">O를 가장 작게 만드는 실행 가능한 대응</div><b class="who"></b><div>R <b class="r"></b></div><div class="bar"><i class="rbar" style="background:var(--orange)"></i></div><div>O <b class="o"></b></div><div class="bar"><i class="obar" style="background:var(--gold)"></i></div><div>team max <b class="team"></b></div></div><div class="card global_best active" data-key="global_best"><h3>③ 전체 best response</h3><div class="objective">max(R,O)를 가장 작게 만드는 대응</div><b class="who"></b><div>R <b class="r"></b></div><div class="bar"><i class="rbar" style="background:var(--orange)"></i></div><div>O <b class="o"></b></div><div class="bar"><i class="obar" style="background:var(--gold)"></i></div><div>team max <b class="team"></b></div></div></div><p class="warn conclusion"></p><div class="optiongrid"><div class="optionbox"><b style="color:var(--orange)">R · 직접 옵션</b><div class="rname"></div><div class="formula">OBSO = transition × control × score</div><div class="rcomponents"></div></div><div class="optionbox"><b style="color:var(--gold)">O · 다른 팀 옵션</b><div class="oname"></div><div class="formula">OBSO = transition × control × score</div><div class="ocomponents"></div></div></div><p class="small">X는 추천 도착지가 아니다. 선택한 수비 대응 이후 OBSO가 가장 큰 두 잠재 이벤트 지점이다. 각 셀은 가장 가까운 공격수에게 설명용으로 귀속하며, 최적화 목적은 그대로 team max = max(R,O)이다.</p><div class="metrics"><span>run onset V0</span><b class="onset"></b><span>수비 전 t=2</span><b class="fixed"></b><span>선택한 대응 후</span><b class="selectedvalue"></b><span>선택한 수비수</span><b class="defender"></b></div><h3>관련 수비수 후보</h3><table><thead><tr><th>선수</th><th>선정 이유</th><th>현재 거리</th></tr></thead><tbody class="defenders"></tbody></table><p class="small warn">현재 dilemma 비교가 계산된 공격 후보만 드롭다운에 표시한다. 공과 비개입 선수는 실제 미래를 사용하는 retrospective audit이다.</p></aside></div></main>
<script id="data" type="application/json">{data}</script><script>
const s=JSON.parse(document.getElementById('data').textContent),items=s.searched_attacks.filter(a=>a.dilemma_responses).sort((a,b)=>b.best_response.value-a.best_response.value),px=x=>x+52.5,py=y=>34-y,svg=document.querySelector('svg'),paths=svg.querySelector('.paths'),moving=svg.querySelector('.moving'),land=svg.querySelector('.landscape'),sel=document.querySelector('.candidate'),rsel=document.querySelector('.response');
items.forEach((a,i)=>{{let o=document.createElement('option');o.value=i;o.textContent=`${{i+1}}. robust ${{a.best_response.value.toFixed(5)}} · (${{a.attack.endpoint_x.toFixed(1)}}, ${{a.attack.endpoint_y.toFixed(1)}}) · ${{a.best_defender_name}}`;sel.appendChild(o)}});
const poly=(pts,c,d='',w=.5)=>`<polyline points="${{pts.map(p=>`${{px(p[p.length-2])}},${{py(p[p.length-1])}}`).join(' ')}}" fill="none" stroke="${{c}}" stroke-width="${{w}}" stroke-dasharray="${{d}}"/>`,interp=(path,t)=>{{if(t<=path[0][0])return path[0].slice(1);if(t>=path.at(-1)[0])return path.at(-1).slice(1);for(let i=1;i<path.length;i++)if(t<=path[i][0]){{let a=path[i-1],b=path[i],q=(t-a[0])/(b[0]-a[0]);return[a[1]+q*(b[1]-a[1]),a[2]+q*(b[2]-a[2])];}}}};
let idx=0,playing=false,timer,current=()=>items[Number(sel.value)],currentResponse=()=>current().dilemma_responses[rsel.value];function timedAttack(a){{return s.attack_path_times_s.map((t,i)=>[t,...a.attack_path_xy[i]])}}function timedDef(r){{return r.defender_path_times_s.map((t,i)=>[t,...r.defender_path_xy[i]])}}
function landscape(){{if(!document.querySelector('.show-landscape').checked){{land.innerHTML='';return}}let max=Math.max(...s.fixed_attack_landscape.map(x=>x.fixed_value));land.innerHTML=s.fixed_attack_landscape.map(x=>`<circle cx="${{px(x.endpoint_x)}}" cy="${{py(x.endpoint_y)}}" r=".34" fill="#78b9ff" opacity="${{.12+.62*x.fixed_value/max}}"/>`).join('')}}
function xmark(x,y,c,label){{return `<path d="M${{px(x)-.7}},${{py(y)-.7}} l1.4,1.4 m0,-1.4 l-1.4,1.4" stroke="${{c}}" stroke-width=".52"/><text x="${{px(x)+.9}}" y="${{py(y)-.5}}" fill="white" font-size="1.05">${{label}}</text>`}}
function setResponse(key){{rsel.value=key;document.querySelectorAll('.card').forEach(x=>x.classList.toggle('active',x.dataset.key===key));idx=0;redrawPaths()}}
function comp(z,p){{let v=z[p+'_maximum'],tr=z[p+'_transition'],pc=z[p+'_pitch_control'],sc=z[p+'_score'],dist=z[p+'_owner_distance_m'];return `OBSO ${{v.toFixed(6)}}<br>transition ${{tr?.toFixed(3)??'–'}} · control ${{pc?.toFixed(3)??'–'}} · score ${{sc?.toFixed(3)??'–'}}<br>선수–이벤트 거리 ${{dist?.toFixed(1)??'–'}}m`}}
function updateInfo(){{let a=current(),selected=currentResponse(),d=selected.decomposition,all=Object.entries(a.dilemma_responses),max=Math.max(...all.flatMap(x=>[x[1].decomposition.focal_maximum,x[1].decomposition.other_maximum]));for(let [key,r] of all){{let card=document.querySelector('.card.'+key),z=r.decomposition;card.querySelector('.who').textContent=r.defender_name;card.querySelector('.r').textContent=z.focal_maximum.toFixed(6);card.querySelector('.o').textContent=z.other_maximum.toFixed(6);card.querySelector('.team').textContent=z.team_maximum.toFixed(6);card.querySelector('.rbar').style.width=(100*z.focal_maximum/max)+'%';card.querySelector('.obar').style.width=(100*z.other_maximum/max)+'%'}}document.querySelector('.onset').textContent=s.decision_baseline.team_maximum.toFixed(6);document.querySelector('.fixed').textContent=a.fixed.value.toFixed(6);document.querySelector('.selectedvalue').textContent=d.team_maximum.toFixed(6);document.querySelector('.defender').textContent=selected.defender_name;document.querySelector('.rname').textContent=s.attacker_name+' 관련 잠재 이벤트';document.querySelector('.oname').textContent=(d.other_player_name||d.other_player_id||'다른 공격수')+' 관련 잠재 이벤트';document.querySelector('.rcomponents').innerHTML=comp(d,'focal');document.querySelector('.ocomponents').innerHTML=comp(d,'other');let dr=a.dilemma_responses.direct_cover.decomposition,doo=a.dilemma_responses.other_cover.decomposition,gb=a.dilemma_responses.global_best.decomposition;document.querySelector('.conclusion').textContent=`O를 우선 막으면 R=${{doo.focal_maximum.toFixed(5)}}가 남고, R을 우선 막으면 O=${{dr.other_maximum.toFixed(5)}}가 남는다. 전체 최선은 더 큰 쪽을 ${{gb.team_maximum.toFixed(5)}}까지 낮춘 대응이다.`;document.querySelector('.defenders').innerHTML=a.relevant_defenders.map(x=>`<tr><td>${{x.defender_name}}</td><td>${{x.selection_reasons.join(', ')}}</td><td>${{x.current_distance_m.toFixed(1)}}m</td></tr>`).join('')}}
function redrawPaths(){{let a=current(),r=currentResponse(),at=timedAttack(a),dt=timedDef(r);paths.innerHTML=poly(at,'#ff56c8','',.62)+poly(dt,'#49e1cc','',.65)+(document.querySelector('.show-observed').checked?poly(s.observed_attacker_timed,'white','1 .65',.42):'');updateInfo();render()}}
function optionEvent(x,y,c,label,ball){{if(x==null)return'';return `<line x1="${{px(ball[0])}}" y1="${{py(ball[1])}}" x2="${{px(x)}}" y2="${{py(y)}}" stroke="${{c}}" stroke-width=".28" stroke-dasharray=".8 .6" opacity=".9" marker-end="url(#arr)"/>`+xmark(x,y,c,label)}}
function render(){{let a=current(),r=currentResponse(),t=idx/25,ap=interp(timedAttack(a),t),dp=interp(timedDef(r),t),f=s.background_frames[idx],did=r.selection.defender_id,g=f.players.filter(p=>p[0]!==s.attacker_id&&p[0]!==did).map(p=>`<circle cx="${{px(p[2])}}" cy="${{py(p[3])}}" r=".72" fill="${{p[1]===s.attacking_team_id?'#ff7057':'#428cff'}}" stroke="white" stroke-width=".16"/>`).join('')+`<circle cx="${{px(f.ball[0])}}" cy="${{py(f.ball[1])}}" r=".43" fill="#111" stroke="white" stroke-width=".15"/>`;if(document.querySelector('.show-options').checked&&t>1.8){{let d=r.decomposition;g+=optionEvent(d.focal_maximum_x,d.focal_maximum_y,'#ff884d','R',f.ball)+optionEvent(d.other_maximum_x,d.other_maximum_y,'#ffc547','O',f.ball)}}moving.innerHTML=g+`<circle cx="${{px(ap[0])}}" cy="${{py(ap[1])}}" r="1.05" fill="#ff56c8" stroke="white" stroke-width=".22"/><circle cx="${{px(dp[0])}}" cy="${{py(dp[1])}}" r="1.0" fill="#428cff" stroke="#49e1cc" stroke-width=".5"/>`;document.querySelector('.scrub').value=idx;document.querySelector('.time').textContent=`t=+${{t.toFixed(2)}} s`}}
sel.onchange=()=>{{idx=0;setResponse('global_best');landscape()}};rsel.onchange=()=>setResponse(rsel.value);document.querySelectorAll('.card').forEach(x=>x.onclick=()=>setResponse(x.dataset.key));document.querySelector('.scrub').oninput=e=>{{idx=Number(e.target.value);render()}};document.querySelector('.play').onclick=e=>{{playing=!playing;e.target.textContent=playing?'❚❚ 일시정지':'▶ 재생';if(playing)timer=setInterval(()=>{{idx=idx>=50?0:idx+1;render()}},40);else clearInterval(timer)}};document.querySelector('.show-landscape').onchange=landscape;document.querySelector('.show-observed').onchange=redrawPaths;document.querySelector('.show-options').onchange=render;landscape();setResponse('global_best');
</script></body></html>"""


def main() -> None:
    args = parse_args()
    match_id = normalize_bundesliga_match_id(args.match_id)
    directory = args.input_dir / match_id
    input_path = directory / f"frame_{args.frame_id}_attacker_maximin.json"
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    _decompose_selected_attacks(payload, args.data_dir)
    _add_decision_baseline(payload, args.data_dir)
    input_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    csv_path = directory / f"frame_{args.frame_id}_attacker_maximin.csv"
    rows = [
        {
            "match_id": payload["match_id"],
            "frame_id": payload["frame_id"],
            "attacker_id": payload["attacker_id"],
            "fixed_rank": item["fixed_rank"],
            "endpoint_x": item["attack"]["endpoint_x"],
            "endpoint_y": item["attack"]["endpoint_y"],
            "onset_value": payload["decision_baseline"]["team_maximum"],
            "fixed_value": item["fixed"]["value"],
            "robust_value": item["best_response"]["value"],
            "fixed_gain_from_onset": item["fixed_gain_from_onset"],
            "robust_gain_from_onset": item["robust_gain_from_onset"],
            "best_defender_id": item["best_response"]["defender_id"],
            "best_defender_name": item["best_defender_name"],
            "evaluated_response_count": item["evaluated_response_count"],
            "evaluation_seconds": item["evaluation_seconds"],
        }
        for item in payload["searched_attacks"]
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    html_path = directory / f"frame_{args.frame_id}_attacker_maximin_audit.html"
    html_path.write_text(_render(payload), encoding="utf-8")
    print(f"audit: {html_path.resolve()}")


if __name__ == "__main__":
    main()
