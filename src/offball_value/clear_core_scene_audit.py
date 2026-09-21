"""Actual-motion audit for clear local off-ball game candidates."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence


_CORE_DECISIONS = {"include", "hold", "reject"}
_VISIBILITY_LABELS = {"yes", "unclear", "no"}


def _scene_key(record: Mapping[str, object]) -> tuple[str, int, str]:
    return (
        str(record["match_id"]),
        int(record["onset_frame_id"]),
        str(record["runner_id"]),
    )


def select_clear_core_scenes(
    scenes: Sequence[Mapping[str, object]],
    reviews: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Return prior human-reviewed clear scenes in review-file order.

    This is deliberately a high-precision development set, not a model-derived
    dilemma label. A scene enters only when onset, possession, and interaction
    were all explicitly accepted by the human reviewer.
    """

    by_key = {_scene_key(scene): dict(scene) for scene in scenes}
    selected: list[dict[str, object]] = []
    for review in reviews:
        if not (
            str(review.get("onset_review", "")) == "correct"
            and str(review.get("possession_review", "")) == "settled"
            and str(review.get("interaction_review", "")) == "clear"
        ):
            continue
        key = _scene_key(review)
        if key not in by_key:
            raise ValueError(f"Clear review has no audit scene: {key}")
        scene = dict(by_key[key])
        scene["prior_review"] = {
            "onset": str(review.get("onset_review", "")),
            "possession": str(review.get("possession_review", "")),
            "interaction": str(review.get("interaction_review", "")),
            "note": str(review.get("note", "")),
        }
        selected.append(scene)
    return selected


def attach_confirmed_core_reviews(
    scenes: Sequence[Mapping[str, object]],
    reviews: Sequence[Mapping[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Validate the second-pass reviews and attach them to every scene.

    Returns ``(all_reviewed_scenes, confirmed_core_scenes)``. A confirmed core
    scene requires an explicit include decision and three unambiguous yes
    judgments. This keeps the development set machine-readable and prevents a
    held scene from entering later game evaluation by accident.
    """

    scene_by_key = {_scene_key(scene): dict(scene) for scene in scenes}
    review_by_key: dict[tuple[str, int, str], dict[str, object]] = {}
    for raw_review in reviews:
        key = _scene_key(raw_review)
        if key in review_by_key:
            raise ValueError(f"Duplicate core review: {key}")
        if key not in scene_by_key:
            raise ValueError(f"Core review has no candidate scene: {key}")
        decision = str(raw_review.get("core_decision", ""))
        if decision not in _CORE_DECISIONS:
            raise ValueError(f"Invalid core decision for {key}: {decision!r}")
        for field in ("defender_visible", "derived_visible", "tradeoff_visible"):
            label = str(raw_review.get(field, ""))
            if label not in _VISIBILITY_LABELS:
                raise ValueError(f"Invalid {field} for {key}: {label!r}")
        review_by_key[key] = {
            "core_decision": decision,
            "defender_visible": str(raw_review["defender_visible"]),
            "derived_visible": str(raw_review["derived_visible"]),
            "tradeoff_visible": str(raw_review["tradeoff_visible"]),
            "primary_defender": str(raw_review.get("primary_defender", "")),
            "derived_option": str(raw_review.get("derived_option", "")),
            "note": str(raw_review.get("note", "")),
        }

    missing = sorted(set(scene_by_key) - set(review_by_key))
    if missing:
        raise ValueError(f"Candidate scenes without a core review: {missing}")

    reviewed: list[dict[str, object]] = []
    confirmed: list[dict[str, object]] = []
    for scene in scenes:
        enriched = dict(scene)
        review = review_by_key[_scene_key(scene)]
        enriched["core_review"] = review
        reviewed.append(enriched)
        if review["core_decision"] == "include" and all(
            review[field] == "yes"
            for field in ("defender_visible", "derived_visible", "tradeoff_visible")
        ):
            confirmed.append(enriched)
    return reviewed, confirmed


def render_clear_core_scene_audit(scenes: Sequence[Mapping[str, object]]) -> str:
    """Render a self-contained review page containing observed motion only."""

    data = json.dumps(list(scenes), ensure_ascii=False, separators=(",", ":")).replace(
        "</", "<\\/"
    )
    return f"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Clear core off-ball scenes</title>
<style>:root{{--bg:#091322;--panel:#111f32;--panel2:#17283d;--line:#38506d;--text:#f3f7fc;--muted:#adbbce;--attack:#ff6d59;--defend:#428dff;--gold:#ffd04f;--cyan:#50e0d2}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font-family:Inter,Pretendard,system-ui,sans-serif}}main{{max-width:1580px;margin:auto;padding:22px}}h1,h2,h3{{margin:0 0 8px}}p{{color:var(--muted);line-height:1.48}}.notice{{background:#1a2c45;border-left:4px solid var(--gold);padding:11px 14px;border-radius:7px}}.toolbar,.controls{{display:flex;gap:9px;align-items:center;flex-wrap:wrap;margin:12px 0}}button,select,input,textarea{{font:inherit;color:var(--text);background:#182a42;border:1px solid #4a6484;border-radius:8px;padding:9px 11px}}button{{cursor:pointer}}.scene{{min-width:520px}}.status{{color:var(--cyan);font-weight:700}}.grow{{flex:1}}.layout{{display:grid;grid-template-columns:minmax(760px,1fr) 470px;gap:16px}}.panel{{background:var(--panel);border:1px solid var(--line);border-radius:13px;padding:14px}}.pitch{{width:100%;display:block;background:#17743a;border-radius:9px}}.scrub{{width:100%;padding:0}}.time{{font-variant-numeric:tabular-nums}}.legend{{display:flex;gap:15px;flex-wrap:wrap;color:#c8d4e4;font-size:.86rem;margin-top:9px}}.dot{{display:inline-block;width:12px;height:12px;border-radius:50%;margin-right:5px}}.summary{{background:var(--panel2);border:1px solid #2d4561;border-radius:9px;padding:10px;margin:9px 0}}.review{{display:grid;gap:9px}}.review label{{display:grid;gap:5px;color:#d5deeb;font-weight:650}}.review small{{color:var(--muted);font-weight:400}}textarea{{min-height:76px;resize:vertical}}.prior{{color:#dce6f3}}.warn{{color:#ffd37a}}@media(max-width:1100px){{.layout{{grid-template-columns:1fr}}.scene{{min-width:280px;max-width:100%}}}}</style></head><body><main>
<h1>Clear core local-game development scenes · v0.1</h1><p class="notice"><b>관측된 실제 움직임만 검수한다.</b> 사람의 second-pass 검수에서 포함으로 확정된 {len(scenes)}개 장면이다. 보류 장면은 이 공유용 데모에서 제외했다. 아직 beneficiary나 수비 dilemma를 모델이 정한 결과가 아니며, 실제로 runner–defender–derived option 구조가 보이는지 다시 확인한다.</p>
<div class="toolbar"><button class="previous">← 이전</button><button class="next">다음 →</button><select class="scene"></select><button class="export">리뷰 CSV 저장</button><span class="status"></span></div>
<div class="layout"><section class="panel"><h2 class="title"></h2><p class="details"></p><svg class="pitch" viewBox="0 0 105 68"><rect width="105" height="68" fill="#17743a" stroke="white" stroke-width=".35"/><line x1="52.5" y1="0" x2="52.5" y2="68" stroke="white" stroke-width=".25"/><circle cx="52.5" cy="34" r="9.15" fill="none" stroke="white" stroke-width=".25"/><rect x="0" y="13.84" width="16.5" height="40.32" fill="none" stroke="white" stroke-width=".25"/><rect x="88.5" y="13.84" width="16.5" height="40.32" fill="none" stroke="white" stroke-width=".25"/><path d="M43 3 L61 3 M58 1 L61 3 L58 5" stroke="#ffd04f" stroke-width=".65" fill="none"/><text x="43" y="7" fill="#ffe49b" font-size="1.5">공격 방향</text><g class="runner-trail"></g><g class="objects"></g></svg><div class="controls"><button class="play">▶ 재생</button><span class="time"></span><span class="grow"></span><span>t=0부터가 검출된 움직임</span></div><input class="scrub" type="range" min="0" value="0" step="1"><div class="legend"><span><i class="dot" style="background:var(--gold)"></i>runner</span><span><i class="dot" style="background:white"></i>t=0 ball carrier</span><span><i class="dot" style="background:var(--attack)"></i>공격팀</span><span><i class="dot" style="background:var(--defend)"></i>수비팀</span></div><p>선수 위에 마우스를 두면 이름이 보인다. 이 단계에서는 가상 수비 궤적이나 위협 점수를 표시하지 않는다.</p></section>
<aside class="panel"><h3>Core-set 재검수</h3><div class="summary metrics"></div><div class="summary prior"></div><div class="review"><label>① Core clear 장면에 포함할까?<select class="decision"><option value="unreviewed">미검수</option><option value="include">포함</option><option value="hold">보류</option><option value="reject">제외</option></select></label><label>② runner에게 반응할 주 수비수가 보이는가?<select class="defender-visible"><option value="unreviewed">미검수</option><option value="yes">분명함</option><option value="unclear">애매함</option><option value="no">없음</option></select></label><label>③ 그 수비수가 runner를 막을 때 열릴 다른 공격 옵션이 보이는가?<select class="derived-visible"><option value="unreviewed">미검수</option><option value="yes">분명함</option><option value="unclear">애매함</option><option value="no">없음</option></select></label><label>④ runner와 파생 옵션 사이 수비 trade-off가 보이는가?<select class="tradeoff-visible"><option value="unreviewed">미검수</option><option value="yes">분명함</option><option value="unclear">애매함</option><option value="no">없음</option></select></label><label>주 반응 수비수 <small>이름을 모르겠으면 위치나 등번호를 적어도 됨</small><input class="primary-defender" placeholder="예: Elvedi 또는 runner 안쪽 중앙수비"></label><label>파생 공격 옵션 <small>선수뿐 아니라 볼 소유자 운반·다른 공간도 가능</small><input class="derived-option" placeholder="예: Kownacki 전진 운반 / Ginczek 뒷공간"></label><label>메모<textarea class="review-note" placeholder="왜 clear인지 또는 왜 제외해야 하는지"></textarea></label></div><p class="warn">포함의 기준은 ‘이미 득점 기회가 됐는가’가 아니라, 우리가 모델링하려는 국소적 선택 구조가 눈으로 분명한가이다.</p></aside></div></main>
<script id="payload" type="application/json">{data}</script><script>
const scenes=JSON.parse(document.getElementById('payload').textContent),q=s=>document.querySelector(s),storage='clear-core-local-game-v0.1-reviews';let si=0,fi=0,playing=false,timer=null,reviews={{}};try{{reviews=JSON.parse(localStorage.getItem(storage)||'{{}}')}}catch(e){{}}
const key=s=>`${{s.match_id}}:${{s.onset_frame_id}}:${{s.runner_id}}`,px=x=>x+52.5,py=y=>34-y,point=(s,x,y)=>[s.attacking_direction*x,s.attacking_direction*y],select=q('.scene');scenes.forEach((s,i)=>{{let o=document.createElement('option');o.value=i;o.textContent=`${{i+1}}/${{scenes.length}} · ${{s.runner_name}} · frame ${{s.onset_frame_id}} · ${{s.match_label}}`;select.appendChild(o)}});
function currentReview(){{let c=scenes[si].core_review||{{}};return reviews[key(scenes[si])]||{{decision:c.core_decision||'unreviewed',defender_visible:c.defender_visible||'unreviewed',derived_visible:c.derived_visible||'unreviewed',tradeoff_visible:c.tradeoff_visible||'unreviewed',primary_defender:c.primary_defender||'',derived_option:c.derived_option||'',note:c.note||''}}}}
function save(){{let s=scenes[si];reviews[key(s)]={{decision:q('.decision').value,defender_visible:q('.defender-visible').value,derived_visible:q('.derived-visible').value,tradeoff_visible:q('.tradeoff-visible').value,primary_defender:q('.primary-defender').value,derived_option:q('.derived-option').value,note:q('.review-note').value}};localStorage.setItem(storage,JSON.stringify(reviews));let done=scenes.filter(x=>(reviews[key(x)]||{{}}).decision&&reviews[key(x)].decision!=='unreviewed').length;q('.status').textContent=`자동 저장 · ${{done}}/${{scenes.length}} 완료`}}
function renderScene(){{let s=scenes[si],r=currentReview();select.value=si;q('.title').textContent=`${{s.match_label}} · runner ${{s.runner_name}}`;q('.details').textContent=`onset frame ${{s.onset_frame_id}} · 실제 슈팅까지 ${{s.seconds_before_shot.toFixed(1)}}초 · ${{s.onset_labels.join(' + ')}}`;q('.metrics').innerHTML=`onset speed <b>${{s.onset_speed_mps.toFixed(2)}}m/s</b><br>pre → post <b>${{s.pre_speed_mps.toFixed(2)}} → ${{s.post_speed_mps.toFixed(2)}}m/s</b><br>최근 1초 team control <b>${{(100*s.settled_team_control_fraction).toFixed(0)}}%</b><br>공의 상대 진영 깊이 <b>${{s.ball_progress_m.toFixed(1)}}m</b>`;let p=s.prior_review||{{}};q('.prior').innerHTML=`이전 판정: <b>${{p.onset}} · ${{p.possession}} · ${{p.interaction}}</b>${{p.note?`<br>메모: ${{p.note}}`:''}}`;q('.decision').value=r.decision;q('.defender-visible').value=r.defender_visible;q('.derived-visible').value=r.derived_visible;q('.tradeoff-visible').value=r.tradeoff_visible;q('.primary-defender').value=r.primary_defender;q('.derived-option').value=r.derived_option;q('.review-note').value=r.note;fi=0;q('.scrub').max=s.frames.length-1;renderFrame();save()}}
function renderFrame(){{let s=scenes[si],f=s.frames[fi],trail=s.frames.slice(0,fi+1).map(x=>x.players.find(p=>p[0]===s.runner_id)).filter(Boolean).map(p=>point(s,p[2],p[3]));q('.runner-trail').innerHTML=trail.length>1?`<polyline points="${{trail.map(p=>`${{px(p[0])}},${{py(p[1])}}`).join(' ')}}" fill="none" stroke="#ffd04f" stroke-width=".45"/>`:'';q('.objects').innerHTML=f.players.map(p=>{{let n=point(s,p[2],p[3]),attack=p[1]===s.team_id,runner=p[0]===s.runner_id,carrier=p[0]===s.carrier_id;return `<circle cx="${{px(n[0])}}" cy="${{py(n[1])}}" r="${{runner ? 1.12 : 0.72}}" fill="${{attack?'#ff6d59':'#428dff'}}" stroke="${{runner?'#ffd04f':carrier?'white':'white'}}" stroke-width="${{runner ? 0.52 : carrier ? 0.48 : 0.16}}"><title>${{p[4]}}</title></circle>`}}).join('')+(f.ball?(()=>{{let b=point(s,f.ball[0],f.ball[1]);return `<circle cx="${{px(b[0])}}" cy="${{py(b[1])}}" r=".43" fill="#111" stroke="white" stroke-width=".18"/>`}})():'');q('.scrub').value=fi;q('.time').textContent=`t=${{f.relative_time_s>=0?'+':''}}${{f.relative_time_s.toFixed(2)}}초`}}
q('.play').onclick=()=>{{playing=!playing;q('.play').textContent=playing?'❚❚ 일시정지':'▶ 재생';if(playing)timer=setInterval(()=>{{fi=fi>=scenes[si].frames.length-1?0:fi+1;renderFrame()}},80);else clearInterval(timer)}};q('.scrub').oninput=()=>{{fi=Number(q('.scrub').value);renderFrame()}};q('.previous').onclick=()=>{{si=(si-1+scenes.length)%scenes.length;renderScene()}};q('.next').onclick=()=>{{si=(si+1)%scenes.length;renderScene()}};select.onchange=()=>{{si=Number(select.value);renderScene()}};['.decision','.defender-visible','.derived-visible','.tradeoff-visible'].forEach(x=>q(x).onchange=save);['.primary-defender','.derived-option','.review-note'].forEach(x=>q(x).oninput=save);
q('.export').onclick=()=>{{save();let rows=[['match_id','onset_frame_id','runner_id','runner_name','core_decision','defender_visible','derived_visible','tradeoff_visible','primary_defender','derived_option','note']];scenes.forEach(s=>{{let r=reviews[key(s)]||{{}};rows.push([s.match_id,s.onset_frame_id,s.runner_id,s.runner_name,r.decision||'unreviewed',r.defender_visible||'unreviewed',r.derived_visible||'unreviewed',r.tradeoff_visible||'unreviewed',r.primary_defender||'',r.derived_option||'',r.note||''])}});let csv=rows.map(row=>row.map(v=>'"'+String(v??'').replaceAll('"','""')+'"').join(',')).join('\\n'),url=URL.createObjectURL(new Blob([csv],{{type:'text/csv'}})),a=document.createElement('a');a.href=url;a.download='clear_core_local_game_v0_1_reviews.csv';a.click();URL.revokeObjectURL(url)}};renderScene();
</script></body></html>"""
