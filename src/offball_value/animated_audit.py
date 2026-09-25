"""Self-contained interactive tracking animations for human quality control."""

from __future__ import annotations

from html import escape
import json
from typing import Mapping, Sequence

from .bundesliga import BundesligaFrame, BundesligaMatchMeta


def animation_frame_payload(
    frames: Mapping[int, BundesligaFrame],
    frame_ids: Sequence[int],
    decision_frame_id: int,
    metadata: BundesligaMatchMeta,
) -> list[dict[str, object]]:
    payload = []
    for frame_id in frame_ids:
        frame = frames.get(frame_id)
        if frame is None:
            continue
        players = []
        for player_id, player in frame.players.items():
            player_meta = metadata.players.get(player_id)
            players.append(
                [
                    player_id,
                    player.team_id,
                    round(float(player.x), 3),
                    round(float(player.y), 3),
                    player_meta.shirt_number if player_meta else "",
                ]
            )
        payload.append(
            {
                "frame_id": frame_id,
                "time": round((frame_id - decision_frame_id) / 25.0, 3),
                "players": players,
                "ball": (
                    [round(float(frame.ball.x), 3), round(float(frame.ball.y), 3)]
                    if frame.ball is not None
                    else None
                ),
            }
        )
    return payload


def _pitch_markup(
    scene_index: int,
    endpoints: Sequence[Sequence[float]],
    baseline_endpoints: Sequence[Sequence[float]] = (),
    influence_ellipse: Mapping[str, float] | None = None,
    endpoint_vectors: Sequence[Sequence[float]] = (),
    proposal_endpoints: Sequence[Sequence[float]] = (),
    empirical_endpoints: Sequence[Sequence[float]] = (),
    representative_paths: Sequence[Sequence[Sequence[float]]] = (),
    plant_cut_endpoints: Sequence[Sequence[float]] = (),
    plant_cut_vectors: Sequence[Sequence[float]] = (),
    plant_cut_paths: Sequence[Sequence[Sequence[float]]] = (),
) -> str:
    proposal_points = "".join(
        f'<circle cx="{float(x)+52.5:.2f}" cy="{34.0-float(y):.2f}" '
        'r="0.17" fill="#d7a7ff" opacity="0.28"/>'
        for x, y in proposal_endpoints
    )
    baseline_points = "".join(
        f'<circle cx="{float(x)+52.5:.2f}" cy="{34.0-float(y):.2f}" '
        'r="0.28" fill="none" stroke="#ffd166" stroke-width="0.16" opacity="0.72"/>'
        for x, y in baseline_endpoints
    )
    endpoint_points = "".join(
        f'<circle cx="{float(x)+52.5:.2f}" cy="{34.0-float(y):.2f}" '
        'r="0.22" fill="#65ff95" opacity="0.65"/>'
        for x, y in endpoints
    )
    plant_cut_points = "".join(
        f'<circle cx="{float(x)+52.5:.2f}" cy="{34.0-float(y):.2f}" '
        'r="0.29" fill="#ff9f43" stroke="#fff0d8" stroke-width="0.08" opacity="0.88"/>'
        for x, y in plant_cut_endpoints
    )
    empirical_points = "".join(
        f'<circle cx="{float(x)+52.5:.2f}" cy="{34.0-float(y):.2f}" '
        'r="0.34" fill="none" stroke="#5de8ff" stroke-width="0.14" opacity="0.88"/>'
        for x, y in empirical_endpoints
    )
    path_markup = "".join(
        '<polyline points="'
        + " ".join(
            f"{float(point[0])+52.5:.2f},{34.0-float(point[1]):.2f}"
            for point in path
            if len(point) >= 2
        )
        + '" fill="none" stroke="#b8ffca" stroke-width="0.20" opacity="0.58"/>'
        for path in representative_paths
        if path
    )
    plant_cut_path_markup = "".join(
        '<polyline points="'
        + " ".join(
            f"{float(point[0])+52.5:.2f},{34.0-float(point[1]):.2f}"
            for point in path
            if len(point) >= 2
        )
        + '" fill="none" stroke="#ffb76b" stroke-width="0.27" opacity="0.82"/>'
        for path in plant_cut_paths
        if path
    )
    vector_lines = []
    for vector in endpoint_vectors:
        if len(vector) < 4:
            continue
        x, y, vx, vy = (float(value) for value in vector[:4])
        speed = (vx * vx + vy * vy) ** 0.5
        if speed < 0.5:
            continue
        length = min(1.8, 0.35 + 0.18 * speed)
        start_x = x - vx / speed * length
        start_y = y - vy / speed * length
        vector_lines.append(
            f'<line x1="{start_x+52.5:.2f}" y1="{34.0-start_y:.2f}" '
            f'x2="{x+52.5:.2f}" y2="{34.0-y:.2f}" '
            'stroke="#b8ffca" stroke-width="0.13" opacity="0.38" '
            f'marker-end="url(#terminal-arrow-{scene_index})"/>'
        )
    vector_markup = "".join(vector_lines)
    plant_cut_vector_lines = []
    for vector in plant_cut_vectors:
        if len(vector) < 4:
            continue
        x, y, vx, vy = (float(value) for value in vector[:4])
        speed = (vx * vx + vy * vy) ** 0.5
        if speed < 0.5:
            continue
        length = min(1.8, 0.35 + 0.18 * speed)
        start_x = x - vx / speed * length
        start_y = y - vy / speed * length
        plant_cut_vector_lines.append(
            f'<line x1="{start_x+52.5:.2f}" y1="{34.0-start_y:.2f}" '
            f'x2="{x+52.5:.2f}" y2="{34.0-y:.2f}" '
            'stroke="#ffd19a" stroke-width="0.18" opacity="0.72" '
            f'marker-end="url(#cut-arrow-{scene_index})"/>'
        )
    plant_cut_vector_markup = "".join(plant_cut_vector_lines)
    influence_markup = ""
    if influence_ellipse:
        center_x = float(influence_ellipse["center_x"]) + 52.5
        center_y = 34.0 - float(influence_ellipse["center_y"])
        major = float(influence_ellipse["major_radius_m"])
        minor = float(influence_ellipse["minor_radius_m"])
        angle = -float(influence_ellipse["angle_degrees"])
        influence_markup = (
            f'<ellipse cx="{center_x:.2f}" cy="{center_y:.2f}" '
            f'rx="{major:.2f}" ry="{minor:.2f}" '
            f'transform="rotate({angle:.2f} {center_x:.2f} {center_y:.2f})" '
            'fill="none" stroke="#d7a7ff" stroke-width="0.42" '
            'stroke-dasharray="1.25 0.75" opacity="0.95"/>'
        )
    return f"""
    <svg class="pitch" viewBox="0 0 105 68" role="img" aria-label="Tracking animation">
      <defs><marker id="terminal-arrow-{scene_index}" viewBox="0 0 5 5" refX="4" refY="2.5" markerWidth="2.5" markerHeight="2.5" orient="auto"><path d="M 0 0 L 5 2.5 L 0 5 Z" fill="#b8ffca"/></marker><marker id="cut-arrow-{scene_index}" viewBox="0 0 5 5" refX="4" refY="2.5" markerWidth="2.5" markerHeight="2.5" orient="auto"><path d="M 0 0 L 5 2.5 L 0 5 Z" fill="#ffd19a"/></marker></defs>
      <rect x="0" y="0" width="105" height="68" fill="#18743a" stroke="white" stroke-width=".35"/>
      <line x1="52.5" y1="0" x2="52.5" y2="68" stroke="white" stroke-width=".25"/>
      <circle cx="52.5" cy="34" r="9.15" fill="none" stroke="white" stroke-width=".25"/>
      <rect x="0" y="13.84" width="16.5" height="40.32" fill="none" stroke="white" stroke-width=".25"/>
      <rect x="88.5" y="13.84" width="16.5" height="40.32" fill="none" stroke="white" stroke-width=".25"/>
      <g class="influence-layer">{influence_markup}</g>
      <g class="proposal-endpoint-layer">{proposal_points}</g>
      <g class="baseline-endpoint-layer">{baseline_points}</g>
      <g class="endpoint-layer">{endpoint_points}</g>
      <g class="plant-cut-endpoint-layer">{plant_cut_points}</g>
      <g class="empirical-endpoint-layer">{empirical_points}</g>
      <g class="representative-path-layer">{path_markup}</g>
      <g class="plant-cut-path-layer">{plant_cut_path_markup}</g>
      <g class="endpoint-vector-layer">{vector_markup}</g>
      <g class="plant-cut-vector-layer">{plant_cut_vector_markup}</g>
      <g class="reference-layer"></g>
      <g class="trail-layer"></g>
      <g class="player-layer"></g>
      <g class="ball-layer"></g>
      <text class="decision-label" x="2" y="3.3">t = 0 decision</text>
    </svg>
    """


def render_tracking_animation_html(
    scenes: Sequence[dict[str, object]],
    report_kind: str,
    title: str,
    subtitle: str,
    primary_endpoint_label: str = "v0.2 feasible endpoints",
    baseline_endpoint_label: str = "v0.1 baseline endpoints",
) -> str:
    if report_kind not in {"scene", "endpoint"}:
        raise ValueError("report_kind must be scene or endpoint")
    cards = []
    for index, scene in enumerate(scenes):
        status = str(scene.get("status", ""))
        status_class = "accepted" if status == "accepted" else "rejected"
        endpoints = scene.get("endpoints", [])
        if not isinstance(endpoints, Sequence):
            endpoints = []
        baseline_endpoints = scene.get("baseline_endpoints", [])
        if not isinstance(baseline_endpoints, Sequence):
            baseline_endpoints = []
        influence_ellipse = scene.get("influence_ellipse")
        if not isinstance(influence_ellipse, Mapping):
            influence_ellipse = None
        endpoint_vectors = scene.get("endpoint_vectors", [])
        if not isinstance(endpoint_vectors, Sequence):
            endpoint_vectors = []
        proposal_endpoints = scene.get("proposal_endpoints", [])
        if not isinstance(proposal_endpoints, Sequence):
            proposal_endpoints = []
        empirical_endpoints = scene.get("empirical_endpoints", [])
        if not isinstance(empirical_endpoints, Sequence):
            empirical_endpoints = []
        representative_paths = scene.get("representative_paths", [])
        if not isinstance(representative_paths, Sequence):
            representative_paths = []
        plant_cut_endpoints = scene.get("plant_cut_endpoints", [])
        if not isinstance(plant_cut_endpoints, Sequence):
            plant_cut_endpoints = []
        plant_cut_vectors = scene.get("plant_cut_vectors", [])
        if not isinstance(plant_cut_vectors, Sequence):
            plant_cut_vectors = []
        plant_cut_paths = scene.get("plant_cut_paths", [])
        if not isinstance(plant_cut_paths, Sequence):
            plant_cut_paths = []
        if report_kind == "scene":
            review_options = (
                '<option value="unreviewed">unreviewed</option>'
                '<option value="valid">valid scene</option>'
                '<option value="invalid">invalid scene</option>'
                '<option value="ambiguous">ambiguous</option>'
            )
        else:
            review_options = (
                '<option value="unreviewed">unreviewed</option>'
                '<option value="valid">valid action space</option>'
                '<option value="too_narrow">too narrow</option>'
                '<option value="too_wide">too wide</option>'
                '<option value="bad_direction">bad direction</option>'
                '<option value="ambiguous">ambiguous</option>'
            )
        if report_kind == "endpoint":
            default_status_label = (
                "OBSERVED ENDPOINT FEASIBLE"
                if status == "accepted"
                else "OBSERVED ENDPOINT INFEASIBLE"
            )
        else:
            default_status_label = status.upper() or report_kind.upper()
        status_label = str(scene.get("status_label", default_status_label))
        cards.append(
            f"""
            <article class="scene-card" data-scene-index="{index}">
              <div class="card-head">
                <span class="badge {status_class}">{escape(status_label)}</span>
                <strong>{escape(str(scene.get('title', 'scene')))}</strong>
              </div>
              <p class="details">{escape(str(scene.get('details', '')))}</p>
              {_pitch_markup(index, endpoints, baseline_endpoints, influence_ellipse, endpoint_vectors, proposal_endpoints, empirical_endpoints, representative_paths, plant_cut_endpoints, plant_cut_vectors, plant_cut_paths)}
              <div class="controls">
                <button type="button" class="play">▶ Play</button>
                <button type="button" class="decision">Jump to t=0</button>
                <label>Speed <select class="speed"><option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option></select></label>
                <span class="time">t=0.00 s</span>
              </div>
              <input class="scrubber" type="range" min="0" max="1" value="0" step="1" aria-label="Animation frame">
              <div class="review">
                <label>Review <select class="review-value">{review_options}</select></label>
                <label>Note <input class="review-note" type="text" placeholder="What looks wrong or right?"></label>
              </div>
            </article>
            """
        )
    serialized = json.dumps(
        {"kind": report_kind, "scenes": list(scenes)},
        ensure_ascii=False,
        separators=(",", ":"),
    ).replace("</", "<\\/")
    legend = ""
    if report_kind == "endpoint":
        has_proposals = any(scene.get("proposal_endpoints") for scene in scenes)
        has_empirical = any(scene.get("empirical_endpoints") for scene in scenes)
        has_paths = any(scene.get("representative_paths") for scene in scenes)
        has_plant_cuts = any(scene.get("plant_cut_endpoints") for scene in scenes)
        legend = (
            '<div class="legend">'
            f'<span><i class="dot v2"></i>{escape(primary_endpoint_label)}</span>'
            f'<span><i class="arrow">→</i>{escape(primary_endpoint_label)} arrival direction</span>'
            f'<span><i class="dot v1"></i>{escape(baseline_endpoint_label)}</span>'
            + (
                '<span><i class="dot cut"></i>plant-and-cut endpoints</span>'
                '<span><i class="line cut"></i>plant-and-cut paths</span>'
                if has_plant_cuts
                else ""
            )
            + (
                '<span><i class="dot proposal"></i>Fernández 1 m proposal cells</span>'
                if has_proposals
                else ""
            )
            + (
                '<span><i class="dot empirical"></i>empirically supported dynamic endpoints</span>'
                if has_empirical
                else ""
            )
            + (
                '<span><i class="line path"></i>representative feasible paths</span>'
                if has_paths
                else ""
            )
            +
            '<span><i class="line influence"></i>Fernández influence contour</span>'
            '<span><i class="diamond"></i>constant-velocity endpoint</span>'
            '<span><i class="cross">×</i>observed t+2 endpoint</span>'
            '</div>'
        )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{escape(title)}</title>
<style>
:root {{ color-scheme:light; }}
body {{ margin:0; font-family:system-ui,sans-serif; background:#eef2f5; color:#17212b; }}
header,main {{ max-width:1540px; margin:auto; padding:22px; }}
.note {{ background:#fff4c9; border-left:4px solid #e2ad00; padding:11px 13px; }}
.toolbar {{ display:flex; gap:10px; flex-wrap:wrap; align-items:center; margin-top:12px; }}
.save-status {{ font-size:.82rem; color:#52616e; }}
.legend {{ display:flex; gap:14px; flex-wrap:wrap; align-items:center; margin:10px 0; font-size:.8rem; color:#43515e; }}
.legend span {{ display:inline-flex; align-items:center; gap:5px; }}
.dot {{ width:8px; height:8px; border-radius:50%; display:inline-block; box-sizing:border-box; }}
.dot.v2 {{ background:#65ff95; }}
.dot.v1 {{ border:2px solid #ffd166; background:transparent; }}
.dot.proposal {{ background:#d7a7ff; opacity:.65; }}
.dot.empirical {{ border:2px solid #25bfdc; background:transparent; }}
.dot.cut {{ background:#ff9f43; border:1px solid #fff0d8; }}
.line.influence {{ width:18px; border-top:2px dashed #b56fe8; display:inline-block; }}
.line.path {{ width:18px; border-top:2px solid #78d991; display:inline-block; }}
.line.cut {{ width:18px; border-top:2px solid #ffb76b; display:inline-block; }}
.arrow {{ color:#3caf62; font-size:1.1rem; font-weight:800; line-height:.8; }}
.diamond {{ width:8px; height:8px; background:#55d8ff; transform:rotate(45deg); display:inline-block; }}
.cross {{ color:#ff3157; font-size:1.1rem; font-weight:800; line-height:.8; }}
button,select,input {{ font:inherit; }}
button {{ border:1px solid #bac4ce; border-radius:7px; background:white; padding:7px 10px; cursor:pointer; }}
.download {{ background:#173f71; color:white; border-color:#173f71; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(440px,1fr)); gap:16px; }}
.scene-card {{ background:white; border:1px solid #d9e0e6; border-radius:11px; padding:14px; box-shadow:0 2px 8px #1d2a3510; }}
.card-head {{ display:flex; align-items:center; gap:9px; }}
.badge {{ padding:3px 7px; border-radius:999px; font-size:.72rem; font-weight:750; }}
.badge.accepted {{ background:#d9f7e1; color:#176529; }}
.badge.rejected {{ background:#ffe0e4; color:#9b1c31; }}
.details {{ color:#56636f; font-size:.82rem; min-height:2.2em; }}
.pitch {{ width:100%; background:#18743a; border-radius:5px; display:block; }}
.decision-label {{ fill:#ffe15d; font-size:1.55px; font-weight:700; opacity:0; }}
.at-decision .decision-label {{ opacity:1; }}
.controls {{ display:flex; gap:8px; align-items:center; flex-wrap:wrap; margin-top:10px; }}
.controls label,.time {{ font-size:.82rem; color:#43515e; }}
.time {{ margin-left:auto; font-variant-numeric:tabular-nums; font-weight:700; }}
.scrubber {{ width:100%; margin:9px 0; }}
.review {{ display:grid; grid-template-columns:180px 1fr; gap:10px; }}
.review label {{ font-size:.8rem; color:#4c5965; }}
.review select,.review input {{ width:100%; box-sizing:border-box; padding:6px; margin-top:3px; }}
@media(max-width:560px) {{ .grid {{ grid-template-columns:1fr; }} .review {{ grid-template-columns:1fr; }} }}
</style></head><body>
<header><h1>{escape(title)}</h1><p>{escape(subtitle)}</p>{legend}
<p class="note">Press Play or drag the timeline. Reviews are automatically saved in this browser. Use Export review CSV when you finish so the research pipeline can read them.</p>
<div class="toolbar"><button type="button" id="save-browser">Save in browser</button><button type="button" class="download" id="download">Export review CSV</button><span class="save-status" id="save-status">Not saved yet</span><span>{len(scenes)} clips</span></div>
</header><main><div class="grid">{''.join(cards)}</div></main>
<script id="report-data" type="application/json">{serialized}</script>
<script>
const REPORT=JSON.parse(document.getElementById('report-data').textContent);
const NS='http://www.w3.org/2000/svg';
const px=x=>x+52.5, py=y=>34-y;
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));
const STORAGE_KEY=`offball-audit:v1:${{location.pathname}}:${{REPORT.kind}}`;
function referenceMarkup(scene){{
  let out='';
  if(scene.cv_endpoint){{const [x,y]=scene.cv_endpoint;out+=`<path d="M ${{px(x)}},${{py(y)-.9}} L ${{px(x)+.9}},${{py(y)}} L ${{px(x)}},${{py(y)+.9}} L ${{px(x)-.9}},${{py(y)}} Z" fill="#55d8ff" stroke="white" stroke-width=".2"/>`;}}
  if(scene.observed_endpoint){{const [x,y]=scene.observed_endpoint;const color=scene.observed_feasible===false?'#ff3157':'#ff4d67';out+=`<path d="M ${{px(x)-1}},${{py(y)-1}} L ${{px(x)+1}},${{py(y)+1}} M ${{px(x)+1}},${{py(y)-1}} L ${{px(x)-1}},${{py(y)+1}}" stroke="${{color}}" stroke-width=".6"/>`;}}
  if(scene.observed_endpoint_vector){{const [x,y,vx,vy]=scene.observed_endpoint_vector,s=Math.hypot(vx,vy);if(s>=.5){{const len=Math.min(2.4,.5+.2*s),sx=x-vx/s*len,sy=y-vy/s*len;out+=`<line x1="${{px(sx)}}" y1="${{py(sy)}}" x2="${{px(x)}}" y2="${{py(y)}}" stroke="#ff4dce" stroke-width=".42"/><circle cx="${{px(sx)}}" cy="${{py(sy)}}" r=".25" fill="#ff4dce"/>`;}}}}
  return out;
}}
function setupCard(card,scene){{
  const playerLayer=card.querySelector('.player-layer'),ballLayer=card.querySelector('.ball-layer'),trailLayer=card.querySelector('.trail-layer');
  card.querySelector('.reference-layer').innerHTML=referenceMarkup(scene);
  const scrub=card.querySelector('.scrubber'),time=card.querySelector('.time'),play=card.querySelector('.play');
  scrub.max=Math.max(0,scene.frames.length-1);
  let index=0,playing=false,lastTick=0,raf=0;
  const decisionIndex=Math.max(0,scene.frames.findIndex(f=>f.time>=0));
  function render(){{
    const frame=scene.frames[index]; if(!frame)return;
    card.querySelector('.pitch').classList.toggle('at-decision',Math.abs(frame.time)<.021);
    const prior=scene.frames.slice(Math.max(0,index-12),index+1);
    const tracked=[scene.carrier_id,scene.focal_id].filter(Boolean);
    trailLayer.innerHTML=tracked.map((id,k)=>{{const pts=prior.map(f=>f.players.find(p=>p[0]===id)).filter(Boolean).map(p=>`${{px(p[2]).toFixed(2)}},${{py(p[3]).toFixed(2)}}`).join(' ');return pts?`<polyline points="${{pts}}" fill="none" stroke="${{k?'#ff69d4':'#ffe15d'}}" stroke-width=".45" opacity=".8"/>`:'';}}).join('');
    playerLayer.innerHTML=frame.players.map(p=>{{const [id,team,x,y,shirt]=p;const attack=team===scene.possession_team_id;const focal=id===scene.focal_id,carrier=id===scene.carrier_id;const fill=attack?'#ff8c42':'#3f8efc';const stroke=focal?'#ff69d4':carrier?'#ffe15d':'white';const sw=(focal||carrier) ? 0.7 : 0.22;const r=focal?1.45:1.05;return `<g><circle cx="${{px(x)}}" cy="${{py(y)}}" r="${{r}}" fill="${{fill}}" stroke="${{stroke}}" stroke-width="${{sw}}"/><text x="${{px(x)}}" y="${{py(y)+.4}}" text-anchor="middle" font-size="1" fill="#111">${{esc(shirt)}}</text></g>`;}}).join('');
    ballLayer.innerHTML=frame.ball?`<circle cx="${{px(frame.ball[0])}}" cy="${{py(frame.ball[1])}}" r=".58" fill="#111" stroke="white" stroke-width=".2"/>`:'';
    scrub.value=index;time.textContent=`t=${{frame.time>=0?'+':''}}${{frame.time.toFixed(2)}} s · frame ${{frame.frame_id}}`;
  }}
  function loop(ts){{if(!playing)return;const speed=Number(card.querySelector('.speed').value);if(ts-lastTick>=40/speed){{index+=1;lastTick=ts;if(index>=scene.frames.length)index=0;render();}}raf=requestAnimationFrame(loop);}}
  play.addEventListener('click',()=>{{playing=!playing;play.textContent=playing?'❚❚ Pause':'▶ Play';if(playing)raf=requestAnimationFrame(loop);else cancelAnimationFrame(raf);}});
  scrub.addEventListener('input',()=>{{index=Number(scrub.value);render();}});
  card.querySelector('.decision').addEventListener('click',()=>{{index=decisionIndex;render();}});
  render();
}}
document.querySelectorAll('.scene-card').forEach(card=>setupCard(card,REPORT.scenes[Number(card.dataset.sceneIndex)]));
function collectReviews(){{
  return [...document.querySelectorAll('.scene-card')].map(card=>{{const scene=REPORT.scenes[Number(card.dataset.sceneIndex)];return {{match_id:scene.match_id,frame_id:scene.decision_frame_id,player_id:scene.focal_id||'',review:card.querySelector('.review-value').value,note:card.querySelector('.review-note').value}};}});
}}
function updateSaveStatus(prefix='Saved'){{
  const reviewed=collectReviews().filter(row=>row.review!=='unreviewed').length;
  document.getElementById('save-status').textContent=`${{prefix}} · ${{reviewed}}/${{REPORT.scenes.length}} reviewed`;
}}
function saveInBrowser(){{
  try{{localStorage.setItem(STORAGE_KEY,JSON.stringify({{saved_at:new Date().toISOString(),reviews:collectReviews()}}));updateSaveStatus('Saved in browser');}}
  catch(error){{document.getElementById('save-status').textContent='Browser save failed — export CSV now';}}
}}
function restoreBrowserSave(){{
  try{{const saved=JSON.parse(localStorage.getItem(STORAGE_KEY)||'null');if(!saved?.reviews)return;const byKey=new Map(saved.reviews.map(row=>[`${{row.frame_id}}:${{row.player_id||''}}`,row]));document.querySelectorAll('.scene-card').forEach(card=>{{const scene=REPORT.scenes[Number(card.dataset.sceneIndex)],row=byKey.get(`${{scene.decision_frame_id}}:${{scene.focal_id||''}}`);if(!row)return;card.querySelector('.review-value').value=row.review||'unreviewed';card.querySelector('.review-note').value=row.note||'';}});updateSaveStatus('Restored browser save');}}
  catch(error){{document.getElementById('save-status').textContent='Could not restore browser save';}}
}}
let saveTimer=0;
document.querySelectorAll('.review-value,.review-note').forEach(input=>input.addEventListener('input',()=>{{clearTimeout(saveTimer);saveTimer=setTimeout(saveInBrowser,250);}}));
document.getElementById('save-browser').addEventListener('click',saveInBrowser);
window.addEventListener('beforeunload',saveInBrowser);
restoreBrowserSave();
document.getElementById('download').addEventListener('click',()=>{{
  saveInBrowser();
  const rows=[['report_kind','match_id','frame_id','player_id','review','note']];
  document.querySelectorAll('.scene-card').forEach(card=>{{const s=REPORT.scenes[Number(card.dataset.sceneIndex)];rows.push([REPORT.kind,s.match_id,s.decision_frame_id,s.focal_id||'',card.querySelector('.review-value').value,card.querySelector('.review-note').value]);}});
  const csv=rows.map(row=>row.map(v=>'"'+String(v??'').replaceAll('"','""')+'"').join(',')).join('\\n');
  const url=URL.createObjectURL(new Blob([csv],{{type:'text/csv'}}));const a=document.createElement('a');a.href=url;a.download=`${{REPORT.kind}}_animation_reviews.csv`;a.click();URL.revokeObjectURL(url);
}});
</script></body></html>"""
