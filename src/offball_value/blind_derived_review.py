"""A labelling screen that cannot leak the model's answer.

The rich payoff audit is a debugging tool: it prints the model's derived
pick in the story question, tags responses ``DERIVED MIN``, and embeds
``derived_option_id`` and every Q value in its payload.  A label collected
on that screen is anchored to the model and is not blind, which would burn
the new labels exactly as the first seven were burnt.

Rather than stripping a rich screen (easy to miss a field), this builds a
purpose-made minimal one from the same audit payload:

- the tracking animation, with the runner, carrier and candidate defender
  marked, and nothing else drawn;
- the candidate defender's name;
- the attacking options **in alphabetical order**, never the model's
  ranking, with no values attached;
- a recording form and CSV export.

Nothing derived from the payoff computation reaches the page.  The module
asserts this itself before returning: see ``_assert_no_leak``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence

# Any of these appearing in the emitted page means the screen leaks model
# output and the labels taken on it are not blind.
FORBIDDEN_TOKENS = (
    "derived_option_id",
    "derived_best_response_id",
    "is_derived_best",
    "is_minimax",
    "minimax_response_id",
    "direct_option_id",
    "cross_cost",
    "worst_local_q",
    "exact_minimax_worst_q",
    "peak_q",
    "relative_effect",
    "candidate_grid",
    "continuation_type",
)

_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Derived-beneficiary blind labelling</title>
<style>
:root{color-scheme:dark}
body{margin:0;background:#0d1117;color:#e6edf3;font:15px/1.55 -apple-system,BlinkMacSystemFont,"Apple SD Gothic Neo",sans-serif}
header{padding:18px 22px;border-bottom:1px solid #30363d}
h1{margin:0 0 4px;font-size:19px}
.sub{color:#8b949e;font-size:13px}
main{display:grid;grid-template-columns:minmax(0,1.35fr) minmax(320px,1fr);gap:22px;padding:22px;align-items:start}
@media(max-width:980px){main{grid-template-columns:1fr}}
select,input,textarea,button{font:inherit;background:#161b22;color:#e6edf3;border:1px solid #30363d;border-radius:7px;padding:7px 9px}
button{cursor:pointer}
button.primary{background:#1f6feb;border-color:#1f6feb;font-weight:700}
.pitch{width:100%;background:#123d1e;border-radius:10px;border:1px solid #30363d}
.row{display:flex;gap:9px;align-items:center;flex-wrap:wrap;margin:9px 0}
.card{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:15px;margin-bottom:14px}
.card h2{margin:0 0 9px;font-size:15px}
label{display:block;margin:9px 0 3px;color:#8b949e;font-size:13px}
.opts{display:flex;flex-direction:column;gap:5px;margin-top:5px}
.opts label{display:flex;gap:8px;align-items:center;color:#e6edf3;margin:0;font-size:14px}
.warn{background:#2d1b00;border:1px solid #7a5c00;color:#f0d58c;border-radius:8px;padding:11px;font-size:13px;margin-bottom:14px}
.progress{color:#8b949e;font-size:13px}
textarea{width:100%;min-height:56px;resize:vertical}
.q{font-weight:700;margin:2px 0 6px}
</style></head><body>
<header>
  <h1>Derived-beneficiary blind labelling</h1>
  <div class="sub">This screen carries no model predictions. Labels are compared with the preregistered table afterwards.</div>
</header>
<main>
  <section>
    <div class="row">
      <select data-testid="scene-select" class="scene"></select>
      <select data-testid="defender-select" class="defender"></select>
    </div>
    <svg class="pitch" data-testid="pitch-animation" viewBox="0 0 105 68" role="img" aria-label="Scene animation">
      <rect x="0" y="0" width="105" height="68" fill="#123d1e"/>
      <rect x="0" y="13.85" width="16.5" height="40.3" fill="none" stroke="white" stroke-width=".25"/>
      <rect x="88.5" y="13.85" width="16.5" height="40.3" fill="none" stroke="white" stroke-width=".25"/>
      <line x1="52.5" y1="0" x2="52.5" y2="68" stroke="white" stroke-width=".25"/>
      <circle cx="52.5" cy="34" r="9.15" fill="none" stroke="white" stroke-width=".25"/>
      <g class="moving"></g>
    </svg>
    <div class="row">
      <button class="play">Play</button>
      <input class="scrub" type="range" min="0" value="0" style="flex:1">
      <span class="time"></span>
    </div>
  </section>
  <section>
    <div class="warn">Answer <b>only with a player name from the list below</b>. A description of space cannot be a label
    (the moment someone turns it into a player, the blinding is broken). <b>The ball carrier can also be the answer.</b>
    If unsure, choose <b>None</b> — better than a forced guess.</div>
    <div class="card">
      <h2 class="ctx"></h2>
      <div class="q">Does this defender actually face the choice of following the runner?</div>
      <div class="row">
        <label style="margin:0"><input type="radio" name="reacts" value="yes"> Yes</label>
        <label style="margin:0"><input type="radio" name="reacts" value="no"> No (no dilemma)</label>
        <label style="margin:0"><input type="radio" name="reacts" value="unsure"> Not sure</label>
      </div>
      <div class="q" style="margin-top:14px">Which player's gain differs most between following and not following?</div>
      <div class="opts" data-testid="beneficiary-options"></div>
      <label>Second choice (only if tied)</label>
      <select class="alt"></select>
      <label>Confidence</label>
      <select class="confidence">
        <option value="high">High</option><option value="medium" selected>Medium</option><option value="low">Low</option>
      </select>
      <label>To name a defender who is not on the list (name + beneficiary)</label>
      <input class="outside" style="width:100%" placeholder="May be left blank">
      <label>Note</label>
      <textarea class="note"></textarea>
      <div class="row">
        <button class="primary save">Save and next</button>
        <span class="progress"></span>
      </div>
    </div>
    <div class="card">
      <h2>Export</h2>
      <div class="sub">Fill in every row, then save as CSV.</div>
      <div class="row"><button data-testid="review-export" class="export">Download CSV</button></div>
    </div>
  </section>
</main>
<script id="scene-data" type="application/json">__DATA__</script>
<script>
const DATA=JSON.parse(document.getElementById('scene-data').textContent);
const q=s=>document.querySelector(s);
const store={};let si=0,di=0,fi=0,timer=null;
const scene=()=>DATA[si], defender=()=>scene().defenders[di];
const key=()=>`${scene().match_id}:${scene().onset_frame_id}:${defender().defender_id}:${scene().repeat_tag||''}`;
const esc=t=>String(t??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
function fillSelectors(){
  q('.scene').innerHTML=DATA.map((s,i)=>`<option value="${i}">${esc(s.match_label||s.match_id)} · ${esc(s.onset_frame_id)} · runner ${esc(s.runner_name)}</option>`).join('');
  q('.scene').value=si;
  q('.defender').innerHTML=scene().defenders.map((d,i)=>`<option value="${i}">Defender ${esc(d.defender_name)}</option>`).join('');
  q('.defender').value=di;
}
function fillForm(){
  const d=defender(),saved=store[key()]||{};
  q('.ctx').textContent=`Off-ball run by ${scene().runner_name} · ball carrier ${scene().carrier_name} · defender ${d.defender_name}`;
  q('[data-testid="beneficiary-options"]').innerHTML=d.options.map(o=>
    `<label><input type="radio" name="ben" value="${esc(o.option_id)}"${saved.beneficiary===o.option_id?' checked':''}> ${esc(o.option_name)}${o.is_carrier?' (ball carrier)':''}</label>`
  ).join('')+`<label><input type="radio" name="ben" value="none"${saved.beneficiary==='none'?' checked':''}> None — no clear beneficiary</label>`;
  q('.alt').innerHTML='<option value="">(none)</option>'+d.options.map(o=>`<option value="${esc(o.option_id)}"${saved.beneficiary_alt===o.option_id?' selected':''}>${esc(o.option_name)}</option>`).join('');
  document.getElementsByName('reacts').forEach(r=>{r.checked=saved.defender_reacts===r.value});
  q('.confidence').value=saved.confidence||'medium';
  q('.outside').value=saved.outside_list_who||'';
  q('.note').value=saved.note||'';
  const total=DATA.reduce((n,s)=>n+s.defenders.length,0);
  q('.progress').textContent=`${Object.keys(store).length} / ${total} rows done`;
}
function frames(){return scene().frames}
function drawFrame(){
  const s=scene(),d=defender(),f=frames()[Math.min(fi,frames().length-1)],dir=s.attacking_direction<0?-1:1;
  const px=x=>52.5+dir*Number(x),py=y=>34+Number(y);
  let html=(f.players||[]).map(p=>{
    const attack=String(p[1])===String(s.attacking_team_id),
      isRunner=String(p[0])===String(s.runner_id),isCarrier=String(p[0])===String(s.carrier_id),
      isDef=String(p[0])===String(d.defender_id),
      r=(isRunner||isCarrier||isDef)?1.0:0.66,
      stroke=isRunner?'#ff51c7':isCarrier?'#ffd052':isDef?'#45e0d0':'white';
    return `<circle cx="${px(p[2])}" cy="${py(p[3])}" r="${r}" fill="${attack?'#ff6d59':'#438dff'}" stroke="${stroke}" stroke-width="${(isRunner||isCarrier||isDef)?'.42':'.13'}"><title>${esc(p[4]||p[0])}</title></circle>`;
  }).join('');
  if(Array.isArray(f.ball)&&f.ball.length>=2)html+=`<circle cx="${px(f.ball[0])}" cy="${py(f.ball[1])}" r=".4" fill="#111" stroke="white" stroke-width=".16"/>`;
  q('.moving').innerHTML=html;
  q('.scrub').max=frames().length-1;q('.scrub').value=fi;
  const t=Number(f.relative_time_s)||0;
  q('.time').textContent=`t=${t>=0?'+':''}${t.toFixed(2)} s · ${fi+1}/${frames().length}`;
}
function save(){
  const ben=document.querySelector('input[name="ben"]:checked'),
    reacts=document.querySelector('input[name="reacts"]:checked');
  if(!ben||!reacts){alert('Choose both the reaction and the beneficiary (if unsure, "Not sure"/"None").');return}
  store[key()]={match_id:scene().match_id,onset_frame_id:scene().onset_frame_id,
    repeat_tag:scene().repeat_tag||'',
    runner_name:scene().runner_name,carrier_name:scene().carrier_name,
    defender_id:defender().defender_id,defender_name:defender().defender_name,
    defender_reacts:reacts.value,beneficiary:ben.value,beneficiary_alt:q('.alt').value,
    confidence:q('.confidence').value,outside_list_who:q('.outside').value,note:q('.note').value};
  if(di+1<scene().defenders.length){di++}else if(si+1<DATA.length){si++;di=0;fi=0}
  fillSelectors();fillForm();drawFrame();
}
function exportCsv(){
  const cols=['match_id','onset_frame_id','repeat_tag','runner_name','carrier_name','defender_id','defender_name',
    'defender_reacts','beneficiary','beneficiary_alt','confidence','outside_list_who','note'];
  const rows=[cols].concat(Object.values(store).map(r=>cols.map(c=>r[c]??'')));
  const csv='\\ufeff'+rows.map(r=>r.map(v=>`"${String(v).replace(/"/g,'""')}"`).join(',')).join('\\n');
  const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([csv],{type:'text/csv;charset=utf-8'}));
  a.download='derived_beneficiary_blind_labels.csv';a.click();
}
q('.scene').addEventListener('change',e=>{si=Number(e.target.value);di=0;fi=0;fillSelectors();fillForm();drawFrame()});
q('.defender').addEventListener('change',e=>{di=Number(e.target.value);fillForm();drawFrame()});
q('.scrub').addEventListener('input',e=>{fi=Number(e.target.value);drawFrame()});
q('.play').addEventListener('click',()=>{if(timer){clearInterval(timer);timer=null;q('.play').textContent='Play';return}
  q('.play').textContent='Stop';timer=setInterval(()=>{fi=(fi+1)%frames().length;drawFrame()},90)});
q('.save').addEventListener('click',save);
q('.export').addEventListener('click',exportCsv);
fillSelectors();fillForm();drawFrame();
</script></body></html>
"""


def _assert_no_leak(html: str) -> None:
    leaked = [token for token in FORBIDDEN_TOKENS if token in html]
    if leaked:
        raise AssertionError(f"blind screen leaks model output: {leaked}")


def build_blind_review_payload(
    games: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Strip an audit payload down to what a blind reviewer may see."""

    scenes: list[dict[str, object]] = []
    for game in games:
        # The answer choices come from the ONSET FRAME, not from the model's
        # option catalogue. The first blind round proved why: in three games
        # the reviewer's beneficiary (a nearby, unthreatening player whose
        # space opens) had been cut by the threat-ranked top-5 catalogue, so
        # a correct label was impossible to record. Candidate generation must
        # never pre-empt the human: every attacking outfielder is listed.
        onset = min(
            game["background_frames"],  # type: ignore[index]
            key=lambda frame: abs(float(frame["relative_time_s"])),
        )
        attackers = [
            player
            for player in onset["players"]
            if str(player[1]) == str(game["attacking_team_id"])
            and str(player[0]) != str(game["runner_id"])
        ]
        if len(attackers) >= 8:
            direction = int(game["attacking_direction"])
            goalkeeper = min(
                attackers, key=lambda player: direction * float(player[2])
            )
            if str(goalkeeper[0]) != str(game["carrier_id"]):
                attackers = [
                    player for player in attackers if player is not goalkeeper
                ]
        options = sorted(
            (
                {
                    "option_id": str(player[0]),
                    "option_name": str(player[4] or player[0]),
                    "is_carrier": str(player[0]) == str(game["carrier_id"]),
                }
                for player in attackers
            ),
            # Alphabetical, never any ranking.
            key=lambda entry: entry["option_name"],
        )
        defenders = []
        for defender in game["candidate_defenders"]:  # type: ignore[index]
            defenders.append(
                {
                    "defender_id": str(defender["defender_id"]),
                    "defender_name": str(
                        defender.get("defender_name") or defender["defender_id"]
                    ),
                    "options": options,
                }
            )
        scenes.append(
            {
                "match_id": str(game["match_id"]),
                "match_label": str(game.get("match_label") or game["match_id"]),
                # Reliability repeats: the same scene shown again later in the
                # session, keyed separately so both answers are recorded. The
                # agreement rate between first and repeated answers is the
                # project's measured noise floor.
                "repeat_tag": str(game.get("repeat_tag") or ""),
                "onset_frame_id": game["onset_frame_id"],
                "runner_id": str(game["runner_id"]),
                "runner_name": str(game.get("runner_name") or game["runner_id"]),
                "carrier_id": str(game["carrier_id"]),
                "carrier_name": str(game.get("carrier_name") or game["carrier_id"]),
                "attacking_team_id": str(game["attacking_team_id"]),
                "attacking_direction": int(game["attacking_direction"]),
                "frames": [
                    {
                        "relative_time_s": frame["relative_time_s"],
                        "players": frame["players"],
                        "ball": frame.get("ball"),
                    }
                    for frame in game["background_frames"]  # type: ignore[index]
                ],
                "defenders": defenders,
            }
        )
    return scenes


def render_blind_derived_review(games: Sequence[Mapping[str, object]]) -> str:
    """Self-contained labelling page carrying no model output."""

    payload = json.dumps(
        build_blind_review_payload(games), ensure_ascii=False
    ).replace("</", "<\\/")
    html = _TEMPLATE.replace("__DATA__", payload)
    _assert_no_leak(html)
    return html
