#!/usr/bin/env python3
"""Extract retrospective off-ball run onsets and render a 30-clip QC page."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from offball_value.animated_audit import animation_frame_payload
from offball_value.bundesliga import (
    find_bundesliga_files,
    load_bundesliga_frames,
    load_bundesliga_match_metadata,
    normalize_bundesliga_match_id,
)
from offball_value.run_onset import (
    RunOnsetCandidate,
    RunOnsetConfig,
    detect_run_onsets,
    motion_signal_records,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw/bundesliga-integrated"),
    )
    parser.add_argument("--match-id", default="J03WMX")
    parser.add_argument(
        "--scene-dir",
        type=Path,
        default=Path("data/processed/scene_extractor_v0_1"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/run_onset_v0_3"),
    )
    parser.add_argument("--audit-clips", type=int, default=30)
    parser.add_argument(
        "--exclude-reviewed-csv",
        type=Path,
        help="Optional prior QC CSV; reviewed match/frame/player keys are excluded.",
    )
    parser.add_argument("--search-radius-seconds", type=float, default=2.0)
    parser.add_argument("--animation-pre-seconds", type=float, default=1.0)
    parser.add_argument("--animation-post-seconds", type=float, default=2.0)
    return parser.parse_args()


def _match_ids(args: argparse.Namespace) -> list[str]:
    if str(args.match_id).lower() != "all":
        return [normalize_bundesliga_match_id(args.match_id)]
    return sorted(
        path.parent.name
        for path in args.scene_dir.glob("DFL-MAT-*/scene_candidates.csv")
    )


def _frame_windows(centres, radius: int) -> set[int]:
    result: set[int] = set()
    for centre in centres:
        result.update(range(int(centre) - radius, int(centre) + radius + 1))
    return result


def _even_rows(frame: pd.DataFrame, count: int) -> pd.DataFrame:
    if count <= 0 or frame.empty:
        return frame.head(0)
    if len(frame) <= count:
        return frame
    indices = np.linspace(0, len(frame) - 1, count).round().astype(int)
    return frame.iloc[np.unique(indices)]


def _match_balanced_rows(frame: pd.DataFrame, count: int) -> pd.DataFrame:
    """Preserve temporal coverage while representing every available match."""

    if count <= 0 or frame.empty:
        return frame.head(0)
    matches = list(frame.groupby("match_id", sort=True))
    base, remainder = divmod(count, len(matches))
    selected = []
    for index, (_, group) in enumerate(matches):
        quota = min(len(group), base + int(index < remainder))
        selected.append(_even_rows(group.sort_values("frame_id"), quota))
    result = pd.concat(selected, ignore_index=True)
    if len(result) < count:
        keys = set(
            zip(result["match_id"], result["frame_id"], result["player_id"])
        )
        remainder_frame = frame[
            ~frame.apply(
                lambda row: (row["match_id"], row["frame_id"], row["player_id"])
                in keys,
                axis=1,
            )
        ].sort_values(["confidence_score", "frame_id"], ascending=[False, True])
        result = pd.concat([result, remainder_frame.head(count - len(result))])
    return result.head(count)


def select_audit_candidates(candidates: pd.DataFrame, count: int) -> pd.DataFrame:
    """Balance primary onset types, time, players, and confidence."""

    if count <= 0 or candidates.empty:
        return candidates.head(0)
    selected: list[pd.DataFrame] = []
    quota = max(1, count // 3)
    for onset_type in ("acceleration", "direction_change", "deceleration_turn"):
        group = candidates[candidates["primary_type"] == onset_type].copy()
        if group.empty:
            continue
        group = group.sort_values(["frame_id", "player_id"])
        # First preserve match-time coverage, then add a high-confidence example.
        sample = _match_balanced_rows(group, quota)
        selected.append(sample)
    result = (
        pd.concat(selected, ignore_index=True)
        if selected
        else candidates.head(0)
    ).drop_duplicates(["match_id", "frame_id", "player_id"])
    if len(result) < count:
        keys = set(
            zip(result["match_id"], result["frame_id"], result["player_id"])
        )
        remaining = candidates[
            ~candidates.apply(
                lambda row: (row["match_id"], row["frame_id"], row["player_id"])
                in keys,
                axis=1,
            )
        ].sort_values(
            ["confidence_score", "frame_id", "player_id"],
            ascending=[False, True, True],
        )
        result = pd.concat([result, remaining.head(count - len(result))])
    return result.sort_values(["match_id", "frame_id", "player_id"]).head(count)


def _candidate_payload(
    row: pd.Series,
    frames,
    metadata,
    pre_frames: int,
    post_frames: int,
    config: RunOnsetConfig,
) -> dict[str, object]:
    frame_id = int(row["frame_id"])
    player_id = str(row["player_id"])
    ids = list(range(frame_id - pre_frames, frame_id + post_frames + 1))
    focal_samples = [
        (target, frames[target].players[player_id])
        for target in ids
        if target in frames and player_id in frames[target].players
    ]
    signals = motion_signal_records(
        [target for target, _ in focal_samples],
        [player.x for _, player in focal_samples],
        [player.y for _, player in focal_samples],
        config,
    )
    runner_name = metadata.players[player_id].short_name
    carrier_id = str(row["ball_carrier_id"])
    carrier_name = metadata.players[carrier_id].short_name
    return {
        "match_id": str(row["match_id"]),
        "frame_id": frame_id,
        "player_id": player_id,
        "player_name": runner_name,
        "carrier_id": carrier_id,
        "carrier_name": carrier_name,
        "team_id": str(row["team_id"]),
        "labels": str(row["onset_labels"]).split("|"),
        "primary_type": str(row["primary_type"]),
        "metrics": {
            key: float(row[key])
            for key in (
                "onset_speed_mps",
                "pre_mean_speed_mps",
                "post_mean_speed_mps",
                "speed_gain_mps",
                "direction_change_degrees",
                "post_displacement_m",
                "peak_acceleration_mps2",
                "peak_deceleration_mps2",
                "confidence_score",
                "stable_control_fraction",
                "future_known_fraction",
                "future_same_team_fraction",
            )
        },
        "frames": animation_frame_payload(frames, ids, frame_id, metadata),
        "signals": [
            {
                **signal,
                "time_s": (signal["frame_id"] - frame_id) / 25.0,
            }
            for signal in signals
        ],
    }


def render_run_onset_audit(payloads: list[dict[str, object]]) -> str:
    data = json.dumps(payloads, ensure_ascii=False, separators=(",", ":")).replace(
        "</", "<\\/"
    )
    return f"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Run onset v0.3 검수</title>
<style>body{{margin:0;background:#0d1522;color:#eef4fc;font-family:system-ui,sans-serif}}main{{max-width:1480px;margin:auto;padding:22px}}h1{{margin:0 0 7px}}p{{color:#b3bece;line-height:1.5}}.note{{background:#172338;border-left:4px solid #ffcc4d;padding:11px 14px}}.toolbar,.controls,.review{{display:flex;gap:10px;align-items:center;flex-wrap:wrap}}button,select,input{{font:inherit;background:#152239;color:#eef4fc;border:1px solid #415575;border-radius:7px;padding:8px 10px}}button{{cursor:pointer}}.layout{{display:grid;grid-template-columns:minmax(650px,1fr) 430px;gap:16px;margin-top:14px}}.panel{{background:#121d2e;border:1px solid #30405a;border-radius:12px;padding:14px}}svg.pitch{{width:100%;display:block;background:#18743a;border-radius:7px}}input[type=range]{{width:100%;padding:0}}.time{{margin-left:auto}}.metrics{{line-height:1.7;color:#c6d0de}}.metrics strong{{color:white}}.chart{{width:100%;height:105px;background:#0e1726;border-radius:7px;margin:8px 0}}.review{{margin-top:14px}}.review input{{flex:1;min-width:250px}}.status{{color:#70d9ff}}@media(max-width:1000px){{.layout{{grid-template-columns:1fr}}}}</style></head><body><main>
<h1>Run onset detector · v0.3 human QC</h1><p class="note"><strong>두 가지를 따로 볼 것:</strong> (1) 노란 세로선의 t=0이 새로운 움직임의 시작 시점으로 맞는가? (2) 그 움직임의 맥락은 안정 소유 공격, 공격 전환 지원, 수비 복귀, 소유 경합 중 무엇인가? v0.3은 공이 경기장 안에 있고, 직전 0.5초 동안 공에 가장 가까운 선수가 공격팀인지를 확인합니다.</p>
<div class="toolbar"><button class="previous">← 이전</button><button class="next">다음 →</button><select class="scene"></select><button class="export">리뷰 CSV 저장</button><span class="status"></span></div><div class="layout"><section class="panel"><h2 class="title"></h2><p class="details"></p><svg class="pitch" viewBox="0 0 105 68"><rect width="105" height="68" fill="#18743a" stroke="white" stroke-width=".35"/><line x1="52.5" y1="0" x2="52.5" y2="68" stroke="white" stroke-width=".25"/><circle cx="52.5" cy="34" r="9.15" fill="none" stroke="white" stroke-width=".25"/><rect x="0" y="13.84" width="16.5" height="40.32" fill="none" stroke="white" stroke-width=".25"/><rect x="88.5" y="13.84" width="16.5" height="40.32" fill="none" stroke="white" stroke-width=".25"/><g class="trail"></g><g class="objects"></g><text x="2" y="3.2" fill="#ffdc64" font-size="1.35">노란 링 = 검출된 러너 · 흰 링 = 볼 소유자</text></svg><div class="controls"><button class="play">▶ 재생</button><span class="time">t=-1.00 s</span></div><input class="slider" type="range" min="0" max="75" value="0" step="1"></section><aside class="panel"><h3>운동 신호</h3><svg class="chart speed" viewBox="0 0 320 80"></svg><svg class="chart acceleration" viewBox="0 0 320 80"></svg><svg class="chart turn" viewBox="0 0 320 80"></svg><p class="metrics"></p><div class="review"><label>① onset 시점 <select class="onset-review"><option value="unreviewed">미검수</option><option value="correct">정확</option><option value="too_early">너무 빠름</option><option value="too_late">너무 늦음</option><option value="not_an_onset">새 움직임이 아님</option><option value="unclear">판단 어려움</option></select></label><label>② 움직임 맥락 <select class="context-review"><option value="unreviewed">미검수</option><option value="settled_attack">안정 소유 공격</option><option value="transition_support">공격 전환 지원</option><option value="defensive_recovery">수비 복귀/커버</option><option value="contested_possession">소유 경합/불확실</option><option value="other">기타</option><option value="unclear">판단 어려움</option></select></label><input class="review-note" placeholder="근거나 수정할 시점을 적어주세요"></div></aside></div></main><script id="payload" type="application/json">{data}</script><script>
const scenes=JSON.parse(document.getElementById('payload').textContent),px=x=>x+52.5,py=y=>34-y,key=s=>`${{s.match_id}}:${{s.frame_id}}:${{s.player_id}}`,storage='run-onset-v0.3-reviews';let sceneIndex=0,frameIndex=0,playing=false,timer=null,reviews={{}};try{{reviews=JSON.parse(localStorage.getItem(storage)||'{{}}')}}catch(e){{}}
const q=s=>document.querySelector(s),select=q('.scene');scenes.forEach((s,i)=>{{let o=document.createElement('option');o.value=i;o.textContent=`${{i+1}}. ${{s.match_id.replace('DFL-MAT-','')}} · frame ${{s.frame_id}} · ${{s.player_name}} · ${{s.labels.join('+')}}`;select.appendChild(o)}});
function poly(points,color,w=.5,dash=''){{return `<polyline points="${{points.map(p=>`${{px(p[0])}},${{py(p[1])}}`).join(' ')}}" fill="none" stroke="${{color}}" stroke-width="${{w}}" stroke-dasharray="${{dash}}"/>`}}function chart(scene,field,label,color,zero=false){{let vals=scene.signals.map(s=>s[field]),lo=Math.min(...vals),hi=Math.max(...vals);if(zero){{lo=Math.min(lo,0);hi=Math.max(hi,0)}}if(Math.abs(hi-lo)<1e-6){{lo-=1;hi+=1}}let X=t=>10+(t+1)/3*300,Y=v=>66-(v-lo)/(hi-lo)*48,path=scene.signals.map(s=>`${{X(s.time_s).toFixed(1)}},${{Y(s[field]).toFixed(1)}}`).join(' '),z=zero?`<line x1="10" y1="${{Y(0)}}" x2="310" y2="${{Y(0)}}" stroke="#627089" stroke-width=".6"/>`:'';return `<rect width="320" height="80" fill="#0e1726"/><line x1="110" y1="8" x2="110" y2="72" stroke="#ffdc64" stroke-width="1.2"/>${{z}}<polyline points="${{path}}" fill="none" stroke="${{color}}" stroke-width="1.5"/><text x="8" y="12" fill="#d8e1ed" font-size="8">${{label}}</text><text x="8" y="75" fill="#8290a5" font-size="6.5">-1s</text><text x="106" y="75" fill="#ffdc64" font-size="6.5">t=0</text><text x="294" y="75" fill="#8290a5" font-size="6.5">+2s</text>`}}
function saveReview(){{let s=scenes[sceneIndex];reviews[key(s)]={{onset_review:q('.onset-review').value,context_review:q('.context-review').value,note:q('.review-note').value}};localStorage.setItem(storage,JSON.stringify(reviews));q('.status').textContent=`둘 다 검수됨 · ${{Object.values(reviews).filter(r=>r.onset_review!=='unreviewed'&&r.context_review!=='unreviewed').length}}/${{scenes.length}}`}}
function renderScene(){{let s=scenes[sceneIndex],m=s.metrics;select.value=sceneIndex;q('.title').textContent=`${{s.match_id.replace('DFL-MAT-','')}} · frame ${{s.frame_id}} · runner ${{s.player_name}}`;q('.details').textContent=`carrier ${{s.carrier_name}} · ${{s.labels.join(' + ')}}`;q('.speed').innerHTML=chart(s,'speed_mps','속도 (m/s)','#62d8ff');q('.acceleration').innerHTML=chart(s,'tangential_acceleration_mps2','접선 가속도 (m/s²)','#ff8e72',true);q('.turn').innerHTML=chart(s,'turn_rate_degrees_s','방향 변화율 (deg/s)','#c69cff',true);q('.metrics').innerHTML=`onset speed <strong>${{m.onset_speed_mps.toFixed(2)}}m/s</strong><br>pre → post <strong>${{m.pre_mean_speed_mps.toFixed(2)}} → ${{m.post_mean_speed_mps.toFixed(2)}}m/s</strong><br>speed gain <strong>${{m.speed_gain_mps.toFixed(2)}}m/s</strong><br>direction change <strong>${{m.direction_change_degrees.toFixed(1)}}°</strong><br>1초 이동 <strong>${{m.post_displacement_m.toFixed(2)}}m</strong><br>possession history <strong>${{(100*m.stable_control_fraction).toFixed(0)}}%</strong> · future same-team <strong>${{(100*m.future_same_team_fraction).toFixed(0)}}%</strong>`;let r=reviews[key(s)]||{{onset_review:'unreviewed',context_review:'unreviewed',note:''}};q('.onset-review').value=r.onset_review;q('.context-review').value=r.context_review;q('.review-note').value=r.note;frameIndex=0;q('.slider').max=s.frames.length-1;renderFrame()}}
function renderFrame(){{let s=scenes[sceneIndex],f=s.frames[frameIndex],trail=s.frames.filter(x=>x.time<=f.time).map(x=>x.players.find(p=>p[0]===s.player_id)).filter(Boolean).map(p=>[p[2],p[3]]);q('.trail').innerHTML=poly(trail,'#ffdc64',.45);q('.objects').innerHTML=f.players.map(p=>{{let focal=p[0]===s.player_id,carrier=p[0]===s.carrier_id,fill=p[1]===s.team_id?'#ff704d':'#418dff';return `<circle cx="${{px(p[2])}}" cy="${{py(p[3])}}" r="${{focal?1.2:.75}}" fill="${{fill}}" stroke="${{focal?'#ffdc64':carrier?'#fff':'#fff'}}" stroke-width="${{focal?.5:carrier?.45:.16}}"/>${{focal?`<text x="${{px(p[2])+1.4}}" y="${{py(p[3])-.8}}" fill="white" font-size="1.2">runner</text>`:''}}`;}}).join('')+(f.ball?`<circle cx="${{px(f.ball[0])}}" cy="${{py(f.ball[1])}}" r=".45" fill="#111" stroke="white" stroke-width=".15"/>`:'');q('.slider').value=frameIndex;q('.time').textContent=`t=${{f.time>=0?'+':''}}${{f.time.toFixed(2)}} s`;}}
q('.play').onclick=()=>{{playing=!playing;q('.play').textContent=playing?'❚❚ 일시정지':'▶ 재생';if(playing)timer=setInterval(()=>{{frameIndex=frameIndex>=scenes[sceneIndex].frames.length-1?0:frameIndex+1;renderFrame()}},40);else clearInterval(timer)}};q('.slider').oninput=()=>{{frameIndex=Number(q('.slider').value);renderFrame()}};q('.previous').onclick=()=>{{sceneIndex=(sceneIndex-1+scenes.length)%scenes.length;renderScene()}};q('.next').onclick=()=>{{sceneIndex=(sceneIndex+1)%scenes.length;renderScene()}};select.onchange=()=>{{sceneIndex=Number(select.value);renderScene()}};q('.onset-review').onchange=saveReview;q('.context-review').onchange=saveReview;q('.review-note').oninput=saveReview;q('.export').onclick=()=>{{saveReview();let rows=[['match_id','frame_id','player_id','onset_review','context_review','note']];scenes.forEach(s=>{{let r=reviews[key(s)]||{{onset_review:'unreviewed',context_review:'unreviewed',note:''}};rows.push([s.match_id,s.frame_id,s.player_id,r.onset_review,r.context_review,r.note])}});let csv=rows.map(row=>row.map(v=>'"'+String(v??'').replaceAll('"','""')+'"').join(',')).join('\\n'),url=URL.createObjectURL(new Blob([csv],{{type:'text/csv'}})),a=document.createElement('a');a.href=url;a.download='run_onset_v0_3_animation_reviews.csv';a.click();URL.revokeObjectURL(url)}};renderScene();
</script></body></html>"""


def main() -> None:
    args = parse_args()
    search_radius = int(round(args.search_radius_seconds * 25.0))
    pre_frames = int(round(args.animation_pre_seconds * 25.0))
    post_frames = int(round(args.animation_post_seconds * 25.0))
    config = RunOnsetConfig()
    candidates_by_match: dict[str, pd.DataFrame] = {}
    context_by_match: dict[str, tuple[object, object]] = {}
    for match_id in _match_ids(args):
        scene_path = args.scene_dir / match_id / "scene_candidates.csv"
        scenes = pd.read_csv(scene_path)
        accepted = scenes[scenes["accepted"].astype(str).str.lower() == "true"]
        if accepted.empty:
            continue
        search_ids = _frame_windows(accepted["frame_id"].astype(int), search_radius)
        margin = max(pre_frames, post_frames, 38)
        target_ids = _frame_windows(search_ids, margin)
        files = find_bundesliga_files(args.data_dir, match_id)
        metadata = load_bundesliga_match_metadata(files["matchinfo"])
        frames = load_bundesliga_frames(files["positions"], target_ids)
        candidates: tuple[RunOnsetCandidate, ...] = detect_run_onsets(
            frames,
            metadata,
            search_frame_ids=search_ids,
            config=config,
        )
        candidates_by_match[match_id] = pd.DataFrame(
            candidate.as_record() for candidate in candidates
        )
        context_by_match[match_id] = (frames, metadata)
        print(f"{match_id}: {len(candidates)} contextual onsets", flush=True)
    nonempty = [frame for frame in candidates_by_match.values() if not frame.empty]
    candidate_frame = (
        pd.concat(nonempty, ignore_index=True) if nonempty else pd.DataFrame()
    )
    if candidate_frame.empty:
        raise ValueError("The detector produced no contextual run onsets")
    audit_pool = candidate_frame
    if args.exclude_reviewed_csv is not None:
        reviewed = pd.read_csv(
            args.exclude_reviewed_csv,
            dtype={"match_id": str, "player_id": str},
        )
        review_keys = reviewed[["match_id", "frame_id", "player_id"]].drop_duplicates()
        audit_pool = candidate_frame.merge(
            review_keys.assign(previously_reviewed=True),
            on=["match_id", "frame_id", "player_id"],
            how="left",
        )
        audit_pool = audit_pool[
            audit_pool["previously_reviewed"].isna()
        ].drop(columns="previously_reviewed")
        print(
            f"new audit pool after prior-QC exclusion: {len(audit_pool)}",
            flush=True,
        )
    audit_rows = select_audit_candidates(audit_pool, args.audit_clips)
    payloads = []
    for _, row in audit_rows.iterrows():
        frames, metadata = context_by_match[str(row["match_id"])]
        payloads.append(
            _candidate_payload(
                row, frames, metadata, pre_frames, post_frames, config
            )
        )

    output_name = (
        "combined" if str(args.match_id).lower() == "all" else _match_ids(args)[0]
    )
    output_dir = args.output_dir / output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = output_dir / "run_onset_candidates.csv"
    audit_selection_path = output_dir / "run_onset_audit_selection.csv"
    audit_path = output_dir / "run_onset_audit.html"
    candidate_frame.to_csv(candidate_path, index=False)
    audit_rows.to_csv(audit_selection_path, index=False)
    audit_path.write_text(render_run_onset_audit(payloads), encoding="utf-8")
    print(candidate_frame["primary_type"].value_counts().to_string())
    print(f"candidates: {candidate_path}")
    print(f"selection:  {audit_selection_path}")
    print(f"audit:      {audit_path}")


if __name__ == "__main__":
    main()
