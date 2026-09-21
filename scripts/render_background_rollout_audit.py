#!/usr/bin/env python3
"""Render a balanced visual audit of held-out background rollouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


MODEL_ORDER = (
    "hold",
    "constant_velocity",
    "damped_constant_velocity",
    "empirical_reference",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        default=Path(
            "data/processed/background_rollout_benchmark_v0_1/DFL-MAT-J03WMX"
        ),
    )
    parser.add_argument("--per-stratum", type=int, default=4)
    return parser.parse_args()


def _balanced_scene_selection(
    predictions: pd.DataFrame,
    per_stratum: int,
) -> pd.DataFrame:
    at_two = predictions[
        np.isclose(predictions["horizon_seconds"], 2.0)
        & ~predictions["is_goalkeeper"].astype(bool)
    ]
    pivot = at_two.pivot_table(
        index="frame_id",
        columns="model",
        values="position_error_m",
        aggfunc="mean",
    )
    pivot["cv_minus_empirical_m"] = (
        pivot["constant_velocity"] - pivot["empirical_reference"]
    )
    ordered = pivot.sort_values("cv_minus_empirical_m")
    worst = ordered.head(per_stratum).copy()
    worst["audit_stratum"] = "empirical_worse"
    best = ordered.tail(per_stratum).copy()
    best["audit_stratum"] = "empirical_better"
    remaining = ordered.drop(index=worst.index.union(best.index))
    middle_center = len(remaining) // 2
    middle_start = max(0, middle_center - per_stratum // 2)
    similar = remaining.iloc[middle_start : middle_start + per_stratum].copy()
    similar["audit_stratum"] = "similar"
    result = pd.concat([worst, similar, best]).reset_index()
    result["audit_order"] = range(1, len(result) + 1)
    return result.sort_values("audit_order")


def _scene_payload(
    predictions: pd.DataFrame,
    selection: pd.DataFrame,
) -> list[dict[str, object]]:
    scenes: list[dict[str, object]] = []
    horizons = sorted(predictions["horizon_seconds"].unique())
    for selected in selection.itertuples(index=False):
        frame_id = int(selected.frame_id)
        scene = predictions[predictions["frame_id"] == frame_id]
        player_rows = scene.sort_values(
            ["player_id", "model", "horizon_seconds"]
        )
        players = []
        for player_id, player_frame in player_rows.groupby("player_id", sort=True):
            first = player_frame.iloc[0]
            actual = {"0": [float(first.start_x_m), float(first.start_y_m)]}
            models = {
                model: {"0": [float(first.start_x_m), float(first.start_y_m)]}
                for model in MODEL_ORDER
            }
            for horizon in horizons:
                horizon_frame = player_frame[
                    np.isclose(player_frame["horizon_seconds"], horizon)
                ]
                actual_row = horizon_frame.iloc[0]
                key = f"{horizon:g}"
                actual[key] = [
                    float(actual_row.actual_x_m),
                    float(actual_row.actual_y_m),
                ]
                for model in MODEL_ORDER:
                    row = horizon_frame[horizon_frame["model"] == model].iloc[0]
                    models[model][key] = [
                        float(row.predicted_x_m),
                        float(row.predicted_y_m),
                    ]
            players.append(
                {
                    "id": player_id,
                    "name": str(first.player_name),
                    "team": str(first.team_id),
                    "phase": str(first.team_phase),
                    "position": str(first.playing_position),
                    "goalkeeper": bool(first.is_goalkeeper),
                    "ballCarrier": bool(first.is_ball_carrier),
                    "actual": actual,
                    "models": models,
                }
            )
        metrics = {
            model: float(
                scene[
                    np.isclose(scene["horizon_seconds"], 2.0)
                    & (scene["model"] == model)
                    & ~scene["is_goalkeeper"].astype(bool)
                ]["position_error_m"].mean()
            )
            for model in MODEL_ORDER
        }
        scenes.append(
            {
                "frameId": frame_id,
                "stratum": selected.audit_stratum,
                "cvMinusEmpirical": float(selected.cv_minus_empirical_m),
                "players": players,
                "metrics": metrics,
            }
        )
    return scenes


def _render_html(scenes: list[dict[str, object]]) -> str:
    payload = json.dumps(scenes, ensure_ascii=False, separators=(",", ":"))
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Background rollout visual audit</title>
<style>
  :root {{ color-scheme: dark; font-family: Inter, ui-sans-serif, system-ui, sans-serif; }}
  body {{ margin: 0; background: #0b1220; color: #e5e7eb; }}
  main {{ max-width: 1220px; margin: 0 auto; padding: 22px; }}
  h1 {{ margin: 0 0 5px; font-size: 24px; }}
  .sub {{ color: #9ca3af; margin-bottom: 16px; line-height: 1.45; }}
  .panel {{ background: #111827; border: 1px solid #263244; border-radius: 14px; padding: 14px; }}
  .controls {{ display: flex; gap: 10px; align-items: center; flex-wrap: wrap; margin-bottom: 12px; }}
  button, select, input {{ accent-color: #f59e0b; background: #172033; color: #f3f4f6; border: 1px solid #374151; border-radius: 8px; padding: 7px 10px; }}
  button {{ cursor: pointer; }}
  #time {{ width: 260px; }}
  .grow {{ flex: 1; }}
  .badge {{ border: 1px solid #4b5563; border-radius: 999px; padding: 5px 10px; color: #d1d5db; }}
  .layout {{ display: grid; grid-template-columns: minmax(0, 1fr) 270px; gap: 14px; }}
  svg {{ width: 100%; background: #176b45; border-radius: 10px; display: block; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th, td {{ text-align: left; padding: 8px; border-bottom: 1px solid #273244; }}
  th {{ color: #9ca3af; font-weight: 600; }}
  tr.active {{ background: #3a2a0b; color: #fbbf24; }}
  .legend {{ margin-top: 12px; display: grid; gap: 7px; color: #cbd5e1; font-size: 13px; }}
  .swatch {{ display:inline-block; width:11px; height:11px; border-radius:50%; margin-right:7px; vertical-align:-1px; }}
  .actual {{ background:#f9fafb; }} .predicted {{ border:2px solid #f59e0b; box-sizing:border-box; }}
  .attack {{ background:#ef4444; }} .defend {{ background:#3b82f6; }}
  .note {{ margin-top: 12px; color: #9ca3af; font-size: 12px; line-height: 1.5; }}
  @media (max-width: 900px) {{ .layout {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body><main>
  <h1>Background rollout benchmark · visual audit</h1>
  <div class="sub">Solid circles follow the observed future. Gold rings show the selected causal background model. The 12 scenes deliberately include four where empirical reference loses to constant velocity, four near the median, and four where it wins.</div>
  <div class="panel">
    <div class="controls">
      <button id="prev">← Previous</button><button id="next">Next →</button>
      <select id="scene"></select>
      <select id="model"></select>
      <button id="play">Play</button>
      <input id="time" type="range" min="0" max="2" step="0.02" value="0">
      <span id="timeLabel" class="badge">t = 0.00 s</span><span class="grow"></span>
      <span id="sceneLabel" class="badge"></span>
    </div>
    <div class="layout">
      <svg id="pitch" viewBox="0 0 1050 680" role="img" aria-label="Football pitch rollout comparison"></svg>
      <aside>
        <table><thead><tr><th>Model</th><th>2 s FDE</th></tr></thead><tbody id="metrics"></tbody></table>
        <div class="legend">
          <div><span class="swatch attack"></span>In-possession team</div>
          <div><span class="swatch defend"></span>Out-of-possession team</div>
          <div><span class="swatch actual"></span>Observed position</div>
          <div><span class="swatch predicted"></span>Model prediction</div>
        </div>
        <div class="note">Goalkeepers use hold as the empirical-reference fallback and are excluded from the displayed FDE. Hover a player for identity and error. This page is a forecasting audit only; observed futures never enter candidate generation.</div>
      </aside>
    </div>
  </div>
<script>
const scenes={payload};
const models={json.dumps(list(MODEL_ORDER))};
const labels={{hold:'Hold',constant_velocity:'Constant velocity',damped_constant_velocity:'Damped CV',empirical_reference:'Empirical reference'}};
let sceneIndex=0, playing=false, timer=null;
const svg=document.getElementById('pitch'), sceneSelect=document.getElementById('scene'), modelSelect=document.getElementById('model'), timeInput=document.getElementById('time');
const NS='http://www.w3.org/2000/svg';
function node(tag,attrs={{}}){{const el=document.createElementNS(NS,tag);Object.entries(attrs).forEach(([k,v])=>el.setAttribute(k,v));return el;}}
function sx(x){{return (x+52.5)*10}} function sy(y){{return (34-y)*10}}
function pitch(){{svg.replaceChildren(); const g=node('g',{{stroke:'#d5eadf','stroke-width':'2','fill':'none','opacity':'0.85'}}); g.append(node('rect',{{x:10,y:10,width:1030,height:660}}));g.append(node('line',{{x1:525,y1:10,x2:525,y2:670}}));g.append(node('circle',{{cx:525,cy:340,r:91.5}}));g.append(node('circle',{{cx:525,cy:340,r:3,fill:'#d5eadf'}}));g.append(node('rect',{{x:10,y:138,width:165,height:403}}));g.append(node('rect',{{x:875,y:138,width:165,height:403}}));g.append(node('rect',{{x:10,y:248,width:55,height:183}}));g.append(node('rect',{{x:985,y:248,width:55,height:183}}));svg.append(g);svg.append(node('g',{{id:'paths'}}));svg.append(node('g',{{id:'players'}}));}}
function at(points,t){{const times=[0,.5,1,1.5,2];if(t<=0)return points['0'];for(let i=1;i<times.length;i++){{if(t<=times[i]){{const lo=times[i-1],hi=times[i],f=(t-lo)/(hi-lo),a=points[String(lo)],b=points[String(hi)];return[a[0]+f*(b[0]-a[0]),a[1]+f*(b[1]-a[1])];}}}}return points['2'];}}
function pathData(points){{return [0,.5,1,1.5,2].map((t,i)=>{{const p=points[String(t)];return `${{i?'L':'M'}}${{sx(p[0])}},${{sy(p[1])}}`;}}).join(' ');}}
function update(){{const s=scenes[sceneIndex],model=modelSelect.value,t=Number(timeInput.value);document.getElementById('timeLabel').textContent=`t = ${{t.toFixed(2)}} s`;document.getElementById('sceneLabel').textContent=`frame ${{s.frameId}} · ${{s.stratum.replaceAll('_',' ')}} · CV−emp ${{s.cvMinusEmpirical.toFixed(2)}} m`;const paths=document.getElementById('paths'),players=document.getElementById('players');paths.replaceChildren();players.replaceChildren();for(const p of s.players){{const color=p.phase==='in_possession'?'#ef4444':'#3b82f6',actual=at(p.actual,t),pred=at(p.models[model],t);paths.append(node('path',{{d:pathData(p.actual),stroke:color,'stroke-width':'2',fill:'none',opacity:'0.28'}}));paths.append(node('path',{{d:pathData(p.models[model]),stroke:'#f59e0b','stroke-width':'2',fill:'none','stroke-dasharray':'7 5',opacity:'0.65'}}));paths.append(node('line',{{x1:sx(actual[0]),y1:sy(actual[1]),x2:sx(pred[0]),y2:sy(pred[1]),stroke:'#fde68a','stroke-width':'1.5',opacity:'0.8'}}));const a=node('circle',{{cx:sx(actual[0]),cy:sy(actual[1]),r:p.goalkeeper?8:7,fill:color,stroke:p.ballCarrier?'#fff':'#111827','stroke-width':p.ballCarrier?'4':'1.5'}});const pr=node('circle',{{cx:sx(pred[0]),cy:sy(pred[1]),r:p.goalkeeper?11:10,fill:'none',stroke:'#fbbf24','stroke-width':'3'}});const err=Math.hypot(actual[0]-pred[0],actual[1]-pred[1]);const title=node('title');title.textContent=`${{p.name}} · ${{p.position}} · error ${{err.toFixed(2)}} m`;pr.append(title);players.append(a,pr);}}
 const body=document.getElementById('metrics');body.replaceChildren();for(const m of models){{const tr=document.createElement('tr');if(m===model)tr.className='active';tr.innerHTML=`<td>${{labels[m]}}</td><td>${{s.metrics[m].toFixed(2)}} m</td>`;body.append(tr);}}}}
function chooseScene(i){{sceneIndex=(i+scenes.length)%scenes.length;sceneSelect.value=String(sceneIndex);timeInput.value='0';update();}}
scenes.forEach((s,i)=>{{const o=document.createElement('option');o.value=String(i);o.textContent=`${{i+1}}. frame ${{s.frameId}} · ${{s.stratum.replaceAll('_',' ')}}`;sceneSelect.append(o);}});models.forEach(m=>{{const o=document.createElement('option');o.value=m;o.textContent=labels[m];modelSelect.append(o);}});modelSelect.value='empirical_reference';
document.getElementById('prev').onclick=()=>chooseScene(sceneIndex-1);document.getElementById('next').onclick=()=>chooseScene(sceneIndex+1);sceneSelect.onchange=()=>chooseScene(Number(sceneSelect.value));modelSelect.onchange=update;timeInput.oninput=update;
document.getElementById('play').onclick=()=>{{playing=!playing;document.getElementById('play').textContent=playing?'Pause':'Play';if(timer)clearInterval(timer);if(playing)timer=setInterval(()=>{{let t=Number(timeInput.value)+.04;if(t>2)t=0;timeInput.value=String(t);update();}},80);}};
pitch();chooseScene(0);
</script></main></body></html>"""


def main() -> None:
    args = parse_args()
    predictions = pd.read_csv(args.benchmark_dir / "player_predictions.csv")
    selection = _balanced_scene_selection(predictions, args.per_stratum)
    scenes = _scene_payload(predictions, selection)
    selection.to_csv(args.benchmark_dir / "audit_scene_selection.csv", index=False)
    (args.benchmark_dir / "background_rollout_audit.html").write_text(
        _render_html(scenes),
        encoding="utf-8",
    )
    print(selection[["audit_order", "frame_id", "audit_stratum", "cv_minus_empirical_m"]].to_string(index=False))
    print(f"audit: {args.benchmark_dir / 'background_rollout_audit.html'}")


if __name__ == "__main__":
    main()
