#!/usr/bin/env python3
"""Render the threat-aware best response beside geometry and observed paths."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from offball_value.bundesliga import (
    FPS,
    find_bundesliga_files,
    load_bundesliga_frames,
    load_bundesliga_match_metadata,
    normalize_bundesliga_match_id,
)
from offball_value.defender_response import (
    DefenderResponseConfig,
    generate_defender_response_actions,
)
from offball_value.steering_reachable import SteeringReachabilityConfig


ATTACK_TIMES_S = (0.0, 0.4, 0.8, 1.2, 1.6, 2.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw/bundesliga-integrated"),
    )
    parser.add_argument("--match-id", default="J03WMX")
    parser.add_argument("--frame-id", type=int, default=14913)
    parser.add_argument(
        "--defender-id",
        default="DFL-OBJ-J01B8N",
        help="Defender to render; defaults to Ryan Gravenberch for frame 14913.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/processed/defender_best_response_v0_1"),
    )
    parser.add_argument("--response-delay", type=float, default=0.2)
    return parser.parse_args()


def _interpolate_path(path, source_times, target_times):
    xs = np.asarray([point[0] for point in path], dtype=float)
    ys = np.asarray([point[1] for point in path], dtype=float)
    return tuple(
        (float(np.interp(time_s, source_times, xs)), float(np.interp(time_s, source_times, ys)))
        for time_s in target_times
    )


def _closest_action(actions, attack_path):
    def key(action):
        attack = _interpolate_path(
            attack_path, ATTACK_TIMES_S, action.response_path_times_s
        )
        distance = min(
            math.hypot(defender[0] - runner[0], defender[1] - runner[1])
            for defender, runner in zip(action.full_path_xy, attack)
        )
        return distance, action.base_action.motion.effort_m2ps3, action.action_id

    return min(actions, key=key)


def _actual_path(frames, player_id, frame_id):
    return [
        [round((target - frame_id) / FPS, 3), frames[target].players[player_id].x, frames[target].players[player_id].y]
        for target in range(frame_id, frame_id + 51)
        if target in frames and player_id in frames[target].players
    ]


def _render(scene: dict[str, object]) -> str:
    data = json.dumps(scene, ensure_ascii=False, separators=(",", ":")).replace(
        "</", "<\\/"
    )
    return f"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>OBSO 수비 best response 검수</title>
<style>body{{margin:0;background:#0d1522;color:#edf3fb;font-family:system-ui,sans-serif}}main{{max-width:1400px;margin:auto;padding:24px}}h1{{margin:0 0 8px}}p{{color:#b5c0d1;line-height:1.55}}.note{{background:#182235;border-left:4px solid #ffcc4d;padding:12px 15px}}.layout{{display:grid;grid-template-columns:minmax(650px,1fr) 360px;gap:18px}}.panel{{background:#121d2e;border:1px solid #30405a;border-radius:12px;padding:14px}}svg{{width:100%;display:block;background:#18743a;border-radius:7px}}button{{font:inherit;background:#1c2a40;color:#edf3fb;border:1px solid #455978;border-radius:7px;padding:8px 12px}}input[type=range]{{width:100%}}.controls,.checks{{display:flex;gap:14px;align-items:center;flex-wrap:wrap;margin-top:10px}}.time{{margin-left:auto}}.legend{{display:grid;gap:9px;color:#c5cfdd}}.swatch{{display:inline-block;width:23px;border-top:4px solid;margin-right:8px}}.metric{{font-size:1.05rem;line-height:1.7}}.warn{{color:#ffd166}}@media(max-width:950px){{.layout{{grid-template-columns:1fr}}}}</style></head>
<body><main><h1>공간을 먼저 점유하는 수비인가?</h1><p>frame {scene['frame_id']} · Skhiri 직진/속도 유지 · {scene['defender_name']}</p><p class="note"><strong>검수 질문:</strong> 청록색 OBSO best response가 실제 흰색 궤적처럼 러너에게 붙기보다 앞으로 생길 위험 공간을 선점하는가? 주황색 X는 각 시점의 <em>team-wide maximum OBSO</em> 위치다. X가 Skhiri와 무관한 곳에 있으면 global maximum 목적함수의 한계도 함께 보이는 장면이다. 배경 20명과 공은 실제 관측 미래를 재생하며, 분홍 공격수와 청록 수비수만 가상이다.</p><div class="layout"><section class="panel"><svg viewBox="0 0 105 68"><rect width="105" height="68" fill="#18743a" stroke="white" stroke-width=".35"/><line x1="52.5" y1="0" x2="52.5" y2="68" stroke="white" stroke-width=".25"/><circle cx="52.5" cy="34" r="9.15" fill="none" stroke="white" stroke-width=".25"/><rect x="0" y="13.84" width="16.5" height="40.32" fill="none" stroke="white" stroke-width=".25"/><rect x="88.5" y="13.84" width="16.5" height="40.32" fill="none" stroke="white" stroke-width=".25"/><g class="paths"></g><g class="moving"></g></svg><div class="controls"><button>▶ 재생</button><span class="time">t=0.00 s</span></div><input class="scrubber" type="range" min="0" max="50" step="1" value="0"><div class="checks"><label><input class="geometry" type="checkbox" checked> 거리 최소 경로</label><label><input class="actual" type="checkbox" checked> 실제 경로</label><label><input class="threat" type="checkbox" checked> 최대 위협점</label></div></section><aside class="panel"><div class="legend"><span><i class="swatch" style="border-color:#ff5fce"></i>가상 공격수</span><span><i class="swatch" style="border-color:#53e4d0"></i>OBSO best response</span><span><i class="swatch" style="border-color:#ffd166"></i>거리 최소 경로</span><span><i class="swatch" style="border-color:white;border-top-style:dashed"></i>실제 수비 경로</span><span><i class="swatch" style="border-color:#ff8c42"></i>team-wide 최대 위협점</span></div><hr><p class="metric">best response peak: <strong>{scene['best_peak']:.6f}</strong><br>실제 수비 상태 peak: <strong>{scene['observed_peak']:.6f}</strong><br>감소율: <strong>{scene['reduction_percent']:.3f}%</strong><br>t+2 후보 간 범위: <strong>{scene['terminal_spread']:.6f}</strong></p><p class="warn">감소량이 매우 작다면, “수비수가 필요 없다”가 아니라 현재의 전역 max가 이 국소 상호작용에 둔감하다는 뜻이다.</p></aside></div></main><script id="data" type="application/json">{data}</script><script>
const s=JSON.parse(document.getElementById('data').textContent),px=x=>x+52.5,py=y=>34-y,svg=document.querySelector('svg'),paths=svg.querySelector('.paths'),moving=svg.querySelector('.moving');
function poly(path,color,dash='',w=.45,o=.95){{return `<polyline points="${{path.map(p=>`${{px(p[p.length-2])}},${{py(p[p.length-1])}}`).join(' ')}}" fill="none" stroke="${{color}}" stroke-width="${{w}}" stroke-dasharray="${{dash}}" opacity="${{o}}"/>`}}function interp(path,t){{if(t<=path[0][0])return path[0].slice(1);if(t>=path.at(-1)[0])return path.at(-1).slice(1);for(let i=1;i<path.length;i++)if(t<=path[i][0]){{let a=path[i-1],b=path[i],q=(t-a[0])/(b[0]-a[0]);return[a[1]+q*(b[1]-a[1]),a[2]+q*(b[2]-a[2])];}}}};
const check=n=>document.querySelector('.'+n);function drawPaths(){{paths.innerHTML=poly(s.attack_timed,'#ff5fce','',.56)+poly(s.best_timed,'#53e4d0','',.55)+(check('geometry').checked?poly(s.closest_timed,'#ffd166','.8 .5',.4,.9):'')+(check('actual').checked?poly(s.actual_timed,'white','1 .7',.4,.9):'');}}['geometry','actual','threat'].forEach(n=>check(n).onchange=()=>{{drawPaths();render()}});let idx=0,playing=false,timer;const slider=check('scrubber'),button=document.querySelector('button'),time=document.querySelector('.time');function render(){{let t=idx/25,a=interp(s.attack_timed,t),d=interp(s.best_timed,t),f=s.background_frames[idx],x=f.players.filter(p=>p[0]!==s.attacker_id&&p[0]!==s.defender_id).map(p=>`<circle cx="${{px(p[2])}}" cy="${{py(p[3])}}" r=".75" fill="${{p[1]===s.attacking_team_id?'#ff704d':'#418dff'}}" stroke="white" stroke-width=".17"/>`).join('')+`<circle cx="${{px(f.ball[0])}}" cy="${{py(f.ball[1])}}" r=".45" fill="#111" stroke="white" stroke-width=".15"/>`;if(check('threat').checked){{let near=s.threat_points.reduce((u,v)=>Math.abs(v[0]-t)<Math.abs(u[0]-t)?v:u);x+=`<path d="M${{px(near[1])-.8}},${{py(near[2])-.8}} l1.6,1.6 m0,-1.6 l-1.6,1.6" stroke="#ff8c42" stroke-width=".55"/><text x="${{px(near[1])+1}}" y="${{py(near[2])-.6}}" fill="#fff" font-size="1.15">max ${{near[3].toFixed(4)}}</text>`;}}moving.innerHTML=x+`<circle cx="${{px(a[0])}}" cy="${{py(a[1])}}" r="1.05" fill="#ff5fce" stroke="white" stroke-width=".22"/><text x="${{px(a[0])+1.2}}" y="${{py(a[1])-.7}}" fill="white" font-size="1.1">A</text><circle cx="${{px(d[0])}}" cy="${{py(d[1])}}" r="1" fill="#418dff" stroke="#53e4d0" stroke-width=".5"/><text x="${{px(d[0])+1.2}}" y="${{py(d[1])-.7}}" fill="white" font-size="1.1">D</text>`;slider.value=idx;time.textContent=`t=+${{t.toFixed(2)}} s`;}}button.onclick=()=>{{playing=!playing;button.textContent=playing?'❚❚ 일시정지':'▶ 재생';if(playing)timer=setInterval(()=>{{idx=idx>=50?0:idx+1;render()}},40);else clearInterval(timer)}};slider.oninput=()=>{{idx=Number(slider.value);render()}};drawPaths();render();
</script></body></html>"""


def main() -> None:
    args = parse_args()
    match_id = normalize_bundesliga_match_id(args.match_id)
    input_path = args.input_dir / match_id / f"frame_{args.frame_id}_best_responses.json"
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    if not payload["defenders"]:
        raise ValueError("Best-response result contains no defenders")
    defender = next(
        (
            item
            for item in payload["defenders"]
            if item["defender_id"] == args.defender_id
        ),
        None,
    )
    if defender is None:
        available = ", ".join(item["defender_id"] for item in payload["defenders"])
        raise ValueError(
            f"Defender {args.defender_id} has no best-response result; available: {available}"
        )
    files = find_bundesliga_files(args.data_dir, match_id)
    metadata = load_bundesliga_match_metadata(files["matchinfo"])
    frames = load_bundesliga_frames(
        files["positions"], range(args.frame_id - 10, args.frame_id + 51)
    )
    frame = frames[args.frame_id]
    history = tuple(frames[i] for i in range(args.frame_id - 10, args.frame_id + 1))
    action_set = generate_defender_response_actions(
        frame,
        history,
        defender["defender_id"],
        DefenderResponseConfig(response_delay_seconds=args.response_delay),
        SteeringReachabilityConfig(),
    )
    closest = _closest_action(action_set.actions, payload["attacker_path_xy"])
    best = defender["best"]
    observed = defender["observed_defender_background"]
    reduction = 100.0 * (observed["horizon_peak"] - best["horizon_peak"]) / observed["horizon_peak"]
    scene = {
        "frame_id": args.frame_id,
        "attacker_id": payload["attacker_id"],
        "defender_id": defender["defender_id"],
        "defender_name": defender["defender_name"],
        "attacking_team_id": frame.players[payload["attacker_id"]].team_id,
        "background_frames": [
            {
                "time_s": (target - args.frame_id) / FPS,
                "players": [
                    [pid, state.team_id, state.x, state.y]
                    for pid, state in frames[target].players.items()
                ],
                "ball": [frames[target].ball.x, frames[target].ball.y],
            }
            for target in range(args.frame_id, args.frame_id + 51)
        ],
        "attack_timed": [[t, *xy] for t, xy in zip(payload["attacker_path_times_s"], payload["attacker_path_xy"])],
        "best_timed": [[t, *xy] for t, xy in zip(defender["best_path_times_s"], defender["best_path_xy"])],
        "closest_timed": [[t, *xy] for t, xy in zip(closest.response_path_times_s, closest.full_path_xy)],
        "actual_timed": _actual_path(frames, defender["defender_id"], args.frame_id),
        "threat_points": [[p["time_s"], p["maximum_x"], p["maximum_y"], p["maximum_obso"]] for p in best["points"]],
        "best_peak": best["horizon_peak"],
        "observed_peak": observed["horizon_peak"],
        "reduction_percent": reduction,
        "terminal_spread": defender["terminal_spread"],
    }
    html_path = (
        args.input_dir
        / match_id
        / f"frame_{args.frame_id}_{defender['defender_id']}_best_response_audit.html"
    )
    html_path.write_text(_render(scene), encoding="utf-8")
    print(f"audit: {html_path}")


if __name__ == "__main__":
    main()
