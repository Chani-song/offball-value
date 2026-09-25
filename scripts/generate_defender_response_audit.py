#!/usr/bin/env python3
"""Render a human-QC audit for delayed feasible defender responses."""

from __future__ import annotations

import argparse
from html import escape
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from offball_value.animated_audit import animation_frame_payload
from offball_value.bundesliga import (
    FPS,
    find_bundesliga_files,
    load_bundesliga_frames,
    load_bundesliga_match_metadata,
    normalize_bundesliga_match_id,
)
from offball_value.defender_response import (
    DefenderResponseAction,
    DefenderResponseConfig,
    generate_defender_response_actions,
    shortlist_relevant_defenders,
)
from offball_value.steering_reachable import SteeringReachabilityConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw/bundesliga-integrated"),
    )
    parser.add_argument("--match-id", default="J03WMX")
    parser.add_argument("--frame-id", type=int, default=14913)
    parser.add_argument("--player-id", default="DFL-OBJ-002G4A")
    parser.add_argument(
        "--scene-dir",
        type=Path,
        default=Path("data/processed/scene_extractor_v0_1"),
    )
    parser.add_argument(
        "--endpoint-dir",
        type=Path,
        default=Path("data/processed/attacker_endpoints_v0_6"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/defender_response_audit_v0_1"),
    )
    parser.add_argument("--response-delay", type=float, default=0.2)
    return parser.parse_args()


def _path_from_row(row: pd.Series) -> tuple[tuple[float, float], ...]:
    points = json.loads(str(row["path_xy"]))
    return (
        (float(row["start_x"]), float(row["start_y"])),
        *tuple((float(point[0]), float(point[1])) for point in points),
    )


def _choose_attack_scenarios(actions: pd.DataFrame) -> list[tuple[str, pd.Series]]:
    eligible = actions[actions["optimization_eligible"].fillna(False)].copy()
    continuous = eligible[eligible["maneuver_type"] == "continuous_steering"]
    cuts = eligible[eligible["maneuver_type"] == "plant_and_cut"]
    if continuous.empty:
        raise ValueError("No continuous steering action is available")
    straight = continuous.sort_values(
        ["control_effort_m2ps3", "grid_snap_distance_m", "action_id"]
    ).iloc[0]

    ninety = cuts[np.isclose(cuts["cut_angle_degrees"].abs(), 90.0)].copy()
    reverse = cuts[np.isclose(cuts["cut_angle_degrees"].abs(), 180.0)].copy()
    scenarios: list[tuple[str, pd.Series]] = [("straight/coast", straight)]
    if not ninety.empty:
        ninety["travel_m"] = np.hypot(
            ninety["endpoint_x"] - ninety["start_x"],
            ninety["endpoint_y"] - ninety["start_y"],
        )
        scenarios.append(
            (
                "90° plant-and-cut",
                ninety.sort_values(
                    ["travel_m", "control_effort_m2ps3", "action_id"],
                    ascending=[False, True, True],
                ).iloc[0],
            )
        )
    if not reverse.empty:
        heading = math.atan2(
            float(reverse.iloc[0]["initial_vy_mps"]),
            float(reverse.iloc[0]["initial_vx_mps"]),
        )
        reverse_heading = heading + math.pi
        reverse["reverse_projection_m"] = (
            (reverse["endpoint_x"] - reverse["start_x"]) * math.cos(reverse_heading)
            + (reverse["endpoint_y"] - reverse["start_y"]) * math.sin(reverse_heading)
        )
        scenarios.append(
            (
                "180° plant-and-return",
                reverse.sort_values(
                    ["reverse_projection_m", "control_effort_m2ps3", "action_id"],
                    ascending=[False, True, True],
                ).iloc[0],
            )
        )
    return scenarios


def _interpolate_path(
    path: tuple[tuple[float, float], ...],
    source_times: tuple[float, ...],
    target_times: tuple[float, ...],
) -> tuple[tuple[float, float], ...]:
    xs = np.asarray([point[0] for point in path], dtype=float)
    ys = np.asarray([point[1] for point in path], dtype=float)
    return tuple(
        (float(np.interp(time_s, source_times, xs)), float(np.interp(time_s, source_times, ys)))
        for time_s in target_times
    )


def _representatives(
    actions: tuple[DefenderResponseAction, ...],
    attacker_path: tuple[tuple[float, float], ...],
) -> dict[str, DefenderResponseAction]:
    if not actions:
        raise ValueError("Defender response set is empty")
    start = actions[0].full_path_xy[0]
    attacker_endpoint = attacker_path[-1]
    hold = min(
        actions,
        key=lambda action: (
            math.hypot(action.endpoint_x - start[0], action.endpoint_y - start[1]),
            action.base_action.motion.effort_m2ps3,
        ),
    )
    follow = min(
        actions,
        key=lambda action: (
            math.hypot(
                action.endpoint_x - attacker_endpoint[0],
                action.endpoint_y - attacker_endpoint[1],
            ),
            action.base_action.motion.effort_m2ps3,
        ),
    )
    attack_times = (0.0, 0.4, 0.8, 1.2, 1.6, 2.0)

    def encounter_rank(action: DefenderResponseAction) -> tuple[float, float]:
        attack = _interpolate_path(
            attacker_path,
            attack_times,
            action.response_path_times_s,
        )
        separation = min(
            math.hypot(defender[0] - runner[0], defender[1] - runner[1])
            for defender, runner in zip(action.full_path_xy, attack)
        )
        return separation, action.base_action.motion.effort_m2ps3

    encounter = min(actions, key=encounter_rank)
    return {"hold-like": hold, "follow-endpoint": follow, "closest-encounter": encounter}


def _actual_path(frames, player_id: str, frame_id: int) -> list[list[float]]:
    result = []
    for target in range(frame_id, frame_id + 51):
        frame = frames.get(target)
        if frame is None or player_id not in frame.players:
            continue
        player = frame.players[player_id]
        result.append([round((target - frame_id) / FPS, 3), player.x, player.y])
    return result


def _minimum_connector(
    defender_action: DefenderResponseAction,
    attacker_path: tuple[tuple[float, float], ...],
) -> dict[str, object]:
    attacker_times = (0.0, 0.4, 0.8, 1.2, 1.6, 2.0)
    aligned_attacker = _interpolate_path(
        attacker_path,
        attacker_times,
        defender_action.response_path_times_s,
    )
    distances = [
        math.hypot(defender[0] - attacker[0], defender[1] - attacker[1])
        for defender, attacker in zip(
            defender_action.full_path_xy, aligned_attacker
        )
    ]
    index = int(np.argmin(distances))
    return {
        "time_s": float(defender_action.response_path_times_s[index]),
        "distance_m": float(distances[index]),
        "defender": list(defender_action.full_path_xy[index]),
        "attacker": list(aligned_attacker[index]),
    }


def _render(payloads: list[dict[str, object]]) -> str:
    data = json.dumps(payloads, ensure_ascii=False, separators=(",", ":")).replace(
        "</", "<\\/"
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>수비 반응 가능 영역 검수</title>
<style>
body{{margin:0;background:#0d1522;color:#e8eef7;font-family:system-ui,sans-serif}}header,main{{max-width:1500px;margin:auto;padding:22px}}h1{{margin-bottom:6px}}p{{color:#aeb9ca}}.note{{border-left:4px solid #ffcc4d;background:#182235;padding:11px 14px;color:#dbe4f0;line-height:1.55}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(560px,1fr));gap:16px}}.card{{background:#121d2e;border:1px solid #30405a;border-radius:12px;padding:14px}}.head{{display:flex;gap:10px;align-items:center}}.badge{{background:#203b59;color:#70d9ff;border-radius:99px;padding:4px 8px;font-size:.75rem}}.details{{font-size:.86rem;min-height:3.8em;line-height:1.55}}.metric{{color:#fff;font-weight:700}}svg{{width:100%;display:block;background:#18743a;border-radius:6px}}button{{font:inherit;background:#1c2a40;color:#edf3fb;border:1px solid #455978;border-radius:7px;padding:7px 11px;cursor:pointer}}.controls{{display:flex;gap:12px;align-items:center;margin-top:10px;flex-wrap:wrap}}.controls label{{font-size:.82rem;color:#bdc9da}}input[type=range]{{width:100%}}.time{{margin-left:auto;font-variant-numeric:tabular-nums}}.legend{{display:flex;flex-wrap:wrap;gap:13px;font-size:.82rem;color:#bdc9da;margin:10px 0}}.legend span{{display:inline-flex;align-items:center;gap:5px}}i{{display:inline-block;width:18px;border-top:3px solid}}.attack{{border-color:#ff5fce}}.intercept{{border-color:#ffd166}}.actual{{border-color:#fff;border-top-style:dashed}}.delay{{border-color:#ff9f43}}.dot{{width:8px;height:8px;border:0;border-radius:50%;background:#63e6ff}}@media(max-width:650px){{.grid{{grid-template-columns:1fr}}}}
</style></head><body><header><h1>수비 반응 가능 영역 검수 · v0.2</h1>
<p>공격수의 가상 움직임마다 관련 수비수 3명을 비교합니다. 모든 배경 선수는 t=0 위치에 고정되어 있고, 분홍 공격수와 노란 수비수만 가상으로 움직입니다.</p>
<div class="legend"><span><i class="attack"></i>가상 공격 궤적</span><span><i class="delay"></i>수비 반응 전 0.2초</span><span><i class="intercept"></i>공격 경로에 가장 가까이 접근 가능한 수비 궤적</span><span><i class="actual"></i>실제 수비 궤적(선택 표시)</span><span><i class="dot"></i>수비수의 가능한 t+2 도착점</span></div>
<p class="note"><strong>볼 것:</strong> ① 이 선수가 실제로 반응할 관련 수비수인가? ② 청록색 도달 영역이 현실적인가? ③ 노란 경로처럼 공격 경로를 차단하는 움직임이 가능한가?<br><strong>아직 볼 필요 없는 것:</strong> 노란 경로가 위협을 가장 잘 줄이는 최적 수비인지는 아직 계산하지 않았습니다. 현재는 두 선수 사이의 거리만 가장 가깝게 만드는 기하학적 차단 예시입니다.</p></header><main><div class="grid" id="grid"></div></main>
<script id="payload" type="application/json">{data}</script><script>
const scenes=JSON.parse(document.getElementById('payload').textContent),NS='http://www.w3.org/2000/svg',px=x=>x+52.5,py=y=>34-y;
function poly(path,color,dash='',width=.32,opacity=.9){{return `<polyline points="${{path.map(p=>`${{px(p[0])}},${{py(p[1])}}`).join(' ')}}" fill="none" stroke="${{color}}" stroke-width="${{width}}" stroke-dasharray="${{dash}}" opacity="${{opacity}}"/>`}}
function interp(path,t){{if(!path.length)return null;if(t<=path[0][0])return path[0].slice(1);if(t>=path[path.length-1][0])return path[path.length-1].slice(1);for(let i=1;i<path.length;i++)if(t<=path[i][0]){{const a=path[i-1],b=path[i],w=(t-a[0])/(b[0]-a[0]);return [a[1]+w*(b[1]-a[1]),a[2]+w*(b[2]-a[2])];}}}}
function setup(scene,index){{const card=document.createElement('article');card.className='card';card.innerHTML=`<div class="head"><span class="badge">수비 후보 ${{scene.candidate_rank}}/3</span><strong>${{scene.attack_label_ko}} · ${{scene.defender_name}}</strong></div><p class="details">선정 이유: <span class="metric">${{scene.selection_reason_ko}}</span><br>현재 공격수와 <span class="metric">${{scene.current_distance_m.toFixed(1)}}m</span> · 실행 가능한 움직임으로 공격 경로에 최대 <span class="metric">${{scene.path_distance_m.toFixed(2)}}m</span>까지 접근 (t=+${{scene.path_time_s.toFixed(1)}}s) · t+2 공격 도착점에 최대 <span class="metric">${{scene.endpoint_distance_m.toFixed(2)}}m</span>까지 접근</p><svg viewBox="0 0 105 68"><rect width="105" height="68" fill="#18743a" stroke="white" stroke-width=".35"/><line x1="52.5" y1="0" x2="52.5" y2="68" stroke="white" stroke-width=".25"/><circle cx="52.5" cy="34" r="9.15" fill="none" stroke="white" stroke-width=".25"/><rect x="0" y="13.84" width="16.5" height="40.32" fill="none" stroke="white" stroke-width=".25"/><rect x="88.5" y="13.84" width="16.5" height="40.32" fill="none" stroke="white" stroke-width=".25"/><g class="static"></g><g class="actual"></g><g class="moving"></g></svg><div class="controls"><button>▶ 재생</button><label><input class="actual-toggle" type="checkbox"> 실제 수비 궤적 표시</label><span class="time">t=0.00 s</span></div><input class="scrubber" type="range" min="0" max="50" value="0" step="1">`;
const stat=card.querySelector('.static'),actual=card.querySelector('.actual'),moving=card.querySelector('.moving'),frame=scene.decision_frame;
let html=frame.players.filter(p=>p[0]!==scene.attacker_id&&p[0]!==scene.defender_id).map(p=>`<circle cx="${{px(p[2])}}" cy="${{py(p[3])}}" r=".8" fill="${{p[1]===scene.attacking_team_id?'#ff704d':'#418dff'}}" stroke="white" stroke-width=".18"/>`).join('');if(frame.ball)html+=`<circle cx="${{px(frame.ball[0])}}" cy="${{py(frame.ball[1])}}" r=".45" fill="#111" stroke="white" stroke-width=".16"/>`;html+=scene.endpoints.map(p=>`<circle cx="${{px(p[0])}}" cy="${{py(p[1])}}" r=".19" fill="#63e6ff" opacity=".30"/>`).join('');html+=poly(scene.attack_path,'#ff5fce','',.55,1);html+=poly(scene.delay_path,'#ff9f43','',.62,1);html+=poly(scene.closest_path.map(p=>p.slice(1)),'#ffd166','',.42,.92);const c=scene.minimum_connector;html+=`<line x1="${{px(c.defender[0])}}" y1="${{py(c.defender[1])}}" x2="${{px(c.attacker[0])}}" y2="${{py(c.attacker[1])}}" stroke="#fff" stroke-width=".22" stroke-dasharray=".6 .45"/><text x="${{px((c.defender[0]+c.attacker[0])/2)+.6}}" y="${{py((c.defender[1]+c.attacker[1])/2)-.6}}" fill="#fff" font-size="1.2">${{scene.path_distance_m.toFixed(2)}}m</text>`;stat.innerHTML=html;
function showActual(){{actual.innerHTML=card.querySelector('.actual-toggle').checked?poly(scene.actual_defender.map(p=>p.slice(1)),'#fff','1 .65',.38,.9):''}}card.querySelector('.actual-toggle').onchange=showActual;
let idx=0,playing=false,timer=null;const slider=card.querySelector('.scrubber'),button=card.querySelector('button'),time=card.querySelector('.time');function render(){{const t=idx/25,ap=interp(scene.attack_timed,t),dp=interp(scene.closest_path,t);moving.innerHTML=(ap?`<circle cx="${{px(ap[0])}}" cy="${{py(ap[1])}}" r="1.1" fill="#ff5fce" stroke="white" stroke-width=".24"/><text x="${{px(ap[0])+1.3}}" y="${{py(ap[1])-.7}}" fill="#fff" font-size="1.15">A</text>`:'')+(dp?`<circle cx="${{px(dp[0])}}" cy="${{py(dp[1])}}" r="1.0" fill="#418dff" stroke="#ffd166" stroke-width=".5"/><text x="${{px(dp[0])+1.3}}" y="${{py(dp[1])-.7}}" fill="#fff" font-size="1.15">D</text>`:'');slider.value=idx;time.textContent=`t=+${{t.toFixed(2)}} s`;}}button.onclick=()=>{{playing=!playing;button.textContent=playing?'❚❚ 일시정지':'▶ 재생';if(playing)timer=setInterval(()=>{{idx=idx>=50?0:idx+1;render()}},40);else clearInterval(timer)}};slider.oninput=()=>{{idx=Number(slider.value);render()}};showActual();render();document.getElementById('grid').appendChild(card);}}
scenes.forEach(setup);
</script></body></html>"""


def main() -> None:
    args = parse_args()
    match_id = normalize_bundesliga_match_id(args.match_id)
    files = find_bundesliga_files(args.data_dir, match_id)
    metadata = load_bundesliga_match_metadata(files["matchinfo"])
    scene_path = args.scene_dir / match_id / "scene_candidates.csv"
    scene_rows = pd.read_csv(scene_path)
    scene = scene_rows[scene_rows["frame_id"] == args.frame_id]
    if scene.empty:
        raise ValueError(f"Frame {args.frame_id} is not a scene candidate")
    scene = scene.iloc[0]
    action_path = args.endpoint_dir / match_id / "attacker_endpoint_actions.csv"
    all_actions = pd.read_csv(action_path)
    actions = all_actions[
        (all_actions["frame_id"] == args.frame_id)
        & (all_actions["player_id"] == args.player_id)
    ]
    scenarios = _choose_attack_scenarios(actions)

    frame_ids = range(args.frame_id - 25, args.frame_id + 51)
    frames = load_bundesliga_frames(files["positions"], frame_ids)
    frame = frames[args.frame_id]
    history = tuple(
        frames[target]
        for target in range(args.frame_id - 10, args.frame_id + 1)
        if target in frames
    )
    defending_team_ids = {
        player.team_id
        for player in frame.players.values()
        if player.team_id != scene["possession_team_id"]
    }
    goalkeeper_ids = tuple(
        goalkeeper_id
        for team_id in defending_team_ids
        if (goalkeeper_id := metadata.goalkeeper_id(team_id)) is not None
    )
    response_config = DefenderResponseConfig(
        response_delay_seconds=args.response_delay,
        relevant_defender_count=3,
    )
    reachability_config = SteeringReachabilityConfig()
    response_cache = {}
    defending_player_ids = tuple(
        player_id
        for player_id, player in frame.players.items()
        if player.team_id != str(scene["possession_team_id"])
        and player_id not in set(goalkeeper_ids)
    )
    for defender_id in defending_player_ids:
        response_cache[defender_id] = generate_defender_response_actions(
            frame,
            history,
            defender_id,
            response_config,
            reachability_config,
        )
    payloads = []
    summary_rows = []
    animation_frames = animation_frame_payload(
        frames,
        list(range(args.frame_id, args.frame_id + 51)),
        args.frame_id,
        metadata,
    )
    for attack_label, attack_row in scenarios:
        attack_path = _path_from_row(attack_row)
        shortlisted = shortlist_relevant_defenders(
            frame,
            history,
            str(scene["possession_team_id"]),
            args.player_id,
            attack_path,
            attacker_path_times_s=(0.0, 0.4, 0.8, 1.2, 1.6, 2.0),
            goalkeeper_ids=goalkeeper_ids,
            config=response_config,
            reachability_config=reachability_config,
            response_action_sets=response_cache,
        )
        for candidate_rank, relevant in enumerate(shortlisted, start=1):
            defender_id = relevant.defender_id
            response_set = response_cache[defender_id]
            closest_action = next(
                action
                for action in response_set.actions
                if action.action_id == relevant.minimum_feasible_path_action_id
            )
            minimum_connector = _minimum_connector(closest_action, attack_path)
            reason_translations = {
                "nearest_now": "현재 공격수와 가장 가까운 직접 마커",
                "best_endpoint_cover": "공격수의 t+2 도착점을 가장 잘 커버",
                "best_path_interceptor": "공격 경로를 가장 잘 차단",
                "next_feasible_interceptor": "그다음으로 경로 차단 가능성이 높음",
            }
            attack_label_translations = {
                "straight/coast": "직진/속도 유지",
                "90° plant-and-cut": "90° 방향 전환",
                "180° plant-and-return": "180° 반전",
            }
            payloads.append(
                {
                    "attack_label": attack_label,
                    "attack_label_ko": attack_label_translations[attack_label],
                    "candidate_rank": candidate_rank,
                    "attacker_id": args.player_id,
                    "defender_id": defender_id,
                    "defender_name": metadata.players[defender_id].short_name,
                    "attacking_team_id": str(scene["possession_team_id"]),
                    "selection_reasons": list(relevant.selection_reasons),
                    "selection_reason_ko": ", ".join(
                        reason_translations[reason]
                        for reason in relevant.selection_reasons
                    ),
                    "current_distance_m": relevant.current_distance_m,
                    "endpoint_distance_m": relevant.feasible_endpoint_distance_m,
                    "path_distance_m": relevant.minimum_feasible_path_distance_m,
                    "path_time_s": relevant.minimum_feasible_path_time_s,
                    "endpoint_count": response_set.unique_endpoint_count,
                    "endpoints": [
                        [action.endpoint_x, action.endpoint_y]
                        for action in response_set.actions
                    ],
                    "attack_path": attack_path,
                    "attack_timed": [
                        [time_s, point[0], point[1]]
                        for time_s, point in zip(
                            (0.0, 0.4, 0.8, 1.2, 1.6, 2.0), attack_path
                        )
                    ],
                    "delay_path": [
                        [response_set.decision_x, response_set.decision_y],
                        [response_set.response_start_x, response_set.response_start_y],
                    ],
                    "closest_path": [
                        [time_s, point[0], point[1]]
                        for time_s, point in zip(
                            closest_action.response_path_times_s,
                            closest_action.full_path_xy,
                        )
                    ],
                    "minimum_connector": minimum_connector,
                    "actual_defender": _actual_path(
                        frames, defender_id, args.frame_id
                    ),
                    "frames": animation_frames,
                    "decision_frame": animation_frames[0],
                }
            )
            summary_rows.append(
                {
                    "match_id": match_id,
                    "frame_id": args.frame_id,
                    "attacker_id": args.player_id,
                    "attack_label": attack_label,
                    "attacker_action_id": attack_row["action_id"],
                    "candidate_rank": candidate_rank,
                    "defender_id": defender_id,
                    "defender_name": metadata.players[defender_id].short_name,
                    "selection_reasons": "|".join(relevant.selection_reasons),
                    "current_distance_m": relevant.current_distance_m,
                    "minimum_feasible_endpoint_distance_m": (
                        relevant.feasible_endpoint_distance_m
                    ),
                    "minimum_feasible_path_distance_m": (
                        relevant.minimum_feasible_path_distance_m
                    ),
                    "minimum_feasible_path_time_s": (
                        relevant.minimum_feasible_path_time_s
                    ),
                    "response_action_count": len(response_set.actions),
                    "response_unique_endpoint_count": response_set.unique_endpoint_count,
                }
            )

    output_dir = args.output_dir / match_id
    output_dir.mkdir(parents=True, exist_ok=True)
    html_path = output_dir / "defender_response_audit.html"
    summary_path = output_dir / "defender_response_summary.csv"
    html_path.write_text(_render(payloads), encoding="utf-8")
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"cards:   {len(payloads)}")
    print(f"audit:   {html_path}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
