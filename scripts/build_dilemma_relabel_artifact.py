"""Blind re-judging of the (runner, defender, beneficiary) triangle.

The judgment unit is the triangle, not the scene, because that is what the
model actually scores: for THIS defender, chased by THIS runner, with THIS
team-mate standing to gain, is he genuinely torn? A scene carries up to
three candidate defenders and each gets its own beneficiary, so the reviewer
steps through them.

WHAT IS AND IS NOT BLIND, deliberately. The beneficiary shown is R9's pick —
stage 2, already validated out of sample at 24/24 and 20/23 — so revealing it
costs nothing we are still testing. The dilemma score is stage 3 and is what
this screen exists to collect, so nothing about it reaches the page: no knee,
no commitment costs, and not the previous verdict, which would anchor the
re-judgment. A token scan refuses to emit a page that leaks any of it.

Also built for a long sitting: keyboard operation, progress that costs no
layout, and answers kept in the browser so closing the tab does not lose an
hour of work.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from offball_value.assignment_rule import attacking_team_id, onset_state  # noqa: E402
from offball_value.coupled_beneficiary import rule_r9, value_at_times  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "score_coupled_beneficiary", ROOT / "scripts" / "score_coupled_beneficiary.py"
)
_coupled = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_coupled)


def beneficiary_of(scene, defender):
    """R9's pick for this defender — stage-2 output, shown on purpose."""
    state = onset_state(scene)
    defender_id = str(defender["defender_id"])
    if defender_id not in state:
        return None
    direct = next(
        (
            row
            for row in defender["responses"]
            if str(row["response_id"]) == str(defender.get("direct_best_response_id"))
        ),
        None,
    )
    if direct is None:
        return None
    defender_at, attacker_at = _coupled.absolute_lookups(scene, defender)
    if defender_at is None:
        return None
    runner = str(scene["runner_id"])
    curves = {
        option_id: value_at_times(cell.get("candidate_grid") or [])
        for option_id, cell in direct["cells"].items()
        if cell.get("legal") is not False
        and cell.get("q") is not None
        and option_id != runner
    }
    if not curves:
        return None
    pick, _ = rule_r9(
        state,
        attacking_team_id(scene),
        runner,
        str(scene["carrier_id"]),
        defender_id,
        defender_at,
        attacker_at,
        curves,
        float(scene["horizon_seconds"]),
    )
    return pick or None

# The beneficiary is shown on purpose; its identity is stage-2 output that is
# already validated. Everything here is stage-3 output or a previous verdict.
FORBIDDEN_TOKENS = (
    "knee",
    "무릎값",
    "cross_cost",
    "exact_minimax_worst_q",
    "candidate_grid",
    "interaction_review",
    "포기 비용",
    "trade_off",
    "delivery",
    "accessibility",
)

PAGE = """<title>딜레마 재판정</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600;700&family=IBM+Plex+Sans+KR:wght@400;500;600;700&display=swap">
<style>
:root{
  --ground:#F4F7F4; --surface:#FFFFFF; --raised:#EDF2EE;
  --ink:#141B16; --muted:#5F7166; --line:#D3DED6;
  --accent:#B07A12; --accent-soft:#F6E9C9;
  --attack:#D6482F; --defend:#2C6FC4; --runner:#B8248C; --carrier:#9A7400; --gain:#0E7C6B;
  --turf:#1D4A2B; --turf-line:rgba(255,255,255,.34);
  --shadow:0 1px 2px rgba(20,27,22,.08),0 8px 24px rgba(20,27,22,.06);
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --ground:#0F1512; --surface:#18201C; --raised:#202A24;
    --ink:#E4EBE5; --muted:#8FA396; --line:#2C3A31;
    --accent:#F2C14E; --accent-soft:#3A3118;
    --attack:#FF6D59; --defend:#4A9EFF; --runner:#FF51C7; --carrier:#FFD052; --gain:#2DD4BF;
    --turf:#123D1E; --turf-line:rgba(255,255,255,.28);
    --shadow:0 1px 2px rgba(0,0,0,.4),0 10px 30px rgba(0,0,0,.35);
  }
}
:root[data-theme="dark"]{
  --ground:#0F1512; --surface:#18201C; --raised:#202A24;
  --ink:#E4EBE5; --muted:#8FA396; --line:#2C3A31;
  --accent:#F2C14E; --accent-soft:#3A3118;
  --attack:#FF6D59; --defend:#4A9EFF; --runner:#FF51C7; --carrier:#FFD052; --gain:#2DD4BF;
  --turf:#123D1E; --turf-line:rgba(255,255,255,.28);
  --shadow:0 1px 2px rgba(0,0,0,.4),0 10px 30px rgba(0,0,0,.35);
}
*{box-sizing:border-box}
body{margin:0;background:var(--ground);color:var(--ink);
  font-family:"IBM Plex Sans KR",-apple-system,BlinkMacSystemFont,"Apple SD Gothic Neo",sans-serif;
  font-size:15px;line-height:1.6;-webkit-font-smoothing:antialiased}
.bar{position:sticky;top:0;z-index:9;height:3px;background:var(--line)}
.bar>i{display:block;height:100%;background:var(--accent);width:0;transition:width .25s ease}
header{padding:20px 24px 16px;display:flex;gap:20px;align-items:baseline;flex-wrap:wrap}
h1{margin:0;font-family:"Barlow Condensed",sans-serif;font-weight:700;
  font-size:30px;letter-spacing:.02em;text-wrap:balance}
.count{font-family:"Barlow Condensed",sans-serif;font-variant-numeric:tabular-nums;
  font-size:20px;font-weight:600;color:var(--muted)}
.lede{margin:0;color:var(--muted);font-size:14px;max-width:62ch;flex-basis:100%}
main{display:grid;grid-template-columns:minmax(0,1.45fr) minmax(330px,1fr);
  gap:22px;padding:8px 24px 40px;align-items:start;max-width:1500px}
@media(max-width:1000px){main{grid-template-columns:1fr;padding:8px 16px 40px}}
.panel{background:var(--surface);border:1px solid var(--line);border-radius:12px;
  padding:16px;box-shadow:var(--shadow)}
.panel+.panel{margin-top:14px}
h2{margin:0 0 12px;font-family:"Barlow Condensed",sans-serif;font-weight:600;
  font-size:15px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}
svg.pitch{width:100%;display:block;border-radius:10px;background:var(--turf)}
.transport{display:flex;gap:10px;align-items:center;margin-top:12px}
button,select{font:inherit;font-family:inherit;color:var(--ink);background:var(--raised);
  border:1px solid var(--line);border-radius:8px;padding:8px 12px;cursor:pointer}
button:focus-visible,select:focus-visible,input:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
input[type=range]{flex:1;accent-color:var(--accent)}
.clock{font-family:"Barlow Condensed",sans-serif;font-variant-numeric:tabular-nums;
  font-size:17px;font-weight:600;min-width:76px;text-align:right;color:var(--muted)}
.who{display:grid;grid-template-columns:auto 1fr;gap:6px 14px;font-size:14px}
.who dt{font-weight:600;display:flex;align-items:center;gap:7px;white-space:nowrap}
.who dd{margin:0;color:var(--muted)}
.dot{width:10px;height:10px;border-radius:50%;flex:none}
.verdict{display:grid;grid-template-columns:1fr 1fr;gap:9px}
.verdict button{padding:14px 10px;text-align:left;line-height:1.3;border-radius:10px}
.verdict button b{display:block;font-size:15px;font-weight:600}
.verdict button span{color:var(--muted);font-size:12px}
.verdict button[aria-pressed="true"]{background:var(--accent-soft);border-color:var(--accent)}
.verdict button[aria-pressed="true"] span{color:var(--ink);opacity:.75}
.tabs{display:flex;gap:7px;margin-bottom:12px;flex-wrap:wrap}
.tabs button{padding:7px 11px;font-size:13.5px;border-radius:999px}
.tabs button[aria-selected="true"]{background:var(--accent);border-color:var(--accent);color:#1A1406;font-weight:600}
.tabs button .mk{margin-left:5px;opacity:.7;font-size:11px}
.kbd{font-family:"Barlow Condensed",sans-serif;font-weight:600;border:1px solid var(--line);
  border-radius:4px;padding:0 5px;font-size:12px;color:var(--muted);margin-left:6px}
.hint{color:var(--muted);font-size:12.5px;margin-top:10px}
.nav{display:flex;gap:9px;margin-top:14px}
.nav button{flex:1}
.go{background:var(--accent);border-color:var(--accent);color:#1A1406;font-weight:600}
:root:not([data-theme="light"]) .go{color:#1A1406}
select.full,textarea{width:100%}
textarea{font:inherit;background:var(--raised);color:var(--ink);border:1px solid var(--line);
  border-radius:8px;padding:9px;resize:vertical}
.saved{color:var(--accent);font-size:12.5px;min-height:1.2em}
@media (prefers-reduced-motion: reduce){*{transition:none!important}}
</style>

<div class="bar"><i id="bar"></i></div>
<header>
  <h1>딜레마 재판정</h1>
  <div class="count" id="count"></div>
  <p class="lede">표시된 <b>수비수</b>가 <b>러너</b>와 <b>이득 보는 선수</b> 사이에서 실제로 찢어지는지를 봐 주세요.
  한 장면에 후보 수비수가 여럿이면 차례로 판정하시면 됩니다. 확신이 서는 것만 <b>확실</b>로요.</p>
</header>
<main>
  <section>
    <div class="panel">
      <select id="pick" class="full" aria-label="장면 선택" style="margin-bottom:12px"></select>
      <svg class="pitch" viewBox="0 0 105 68" role="img" aria-label="트래킹 애니메이션">
        <rect x="0" y="0" width="105" height="68" fill="var(--turf)"/>
        <g fill="none" stroke="var(--turf-line)" stroke-width=".25">
          <rect x=".4" y=".4" width="104.2" height="67.2"/>
          <rect x=".4" y="13.85" width="16.5" height="40.3"/>
          <rect x="88.1" y="13.85" width="16.5" height="40.3"/>
          <rect x=".4" y="24.85" width="5.5" height="18.3"/>
          <rect x="99.1" y="24.85" width="5.5" height="18.3"/>
          <line x1="52.5" y1=".4" x2="52.5" y2="67.6"/>
          <circle cx="52.5" cy="34" r="9.15"/>
        </g>
        <g id="moving"></g>
      </svg>
      <div class="transport">
        <button id="play" aria-label="재생 또는 정지">▶ 재생<span class="kbd">Space</span></button>
        <input id="scrub" type="range" min="0" value="0" aria-label="시간 이동">
        <span class="clock" id="clock"></span>
      </div>
    </div>
  </section>
  <section>
    <div class="panel">
      <h2>어느 수비수를 볼까요</h2>
      <div class="tabs" id="tabs"></div>
      <dl class="who" id="who"></dl>
    </div>
    <div class="panel">
      <h2>딜레마가 있나요</h2>
      <div class="verdict" id="verdict">
        <button data-v="clear" aria-pressed="false"><b>확실<span class="kbd">1</span></b><span>찢어지는 게 분명히 보임</span></button>
        <button data-v="possible" aria-pressed="false"><b>있을 수도<span class="kbd">2</span></b><span>위험하지만 뚜렷하진 않음</span></button>
        <button data-v="unclear" aria-pressed="false"><b>모르겠음<span class="kbd">3</span></b><span>판단 유보</span></button>
        <button data-v="none" aria-pressed="false"><b>없음<span class="kbd">4</span></b><span>그런 구조가 아님</span></button>
      </div>
      <p class="hint">보수적으로 봐 주세요. 확실이 적어도 괜찮습니다. 이 판정은 <b>지금 선택된 수비수</b>에 대한 것입니다.</p>
    </div>
    <div class="panel">
      <h2>메모 <span style="text-transform:none;letter-spacing:0">(선택)</span></h2>
      <textarea id="note" rows="3" aria-label="메모"></textarea>
    </div>
    <div class="nav">
      <button id="prev">◀ 이전<span class="kbd">←</span></button>
      <button id="next">다음 ▶<span class="kbd">→</span></button>
      <button id="save" class="go">CSV 내보내기</button>
    </div>
    <p class="saved" id="saved"></p>
  </section>
</main>

<script id="scenes" type="application/json">__DATA__</script>
<script>
const D=JSON.parse(document.getElementById('scenes').textContent);
const $=id=>document.getElementById(id);
const KEY='dilemma-relabel-v1';
let si=0,di=0,fi=0,timer=null,ans={};
try{ans=JSON.parse(localStorage.getItem(KEY)||'{}')||{}}catch(e){ans={}}
const store=()=>{try{localStorage.setItem(KEY,JSON.stringify(ans))}catch(e){}};
const S=()=>D[si];
const P=()=>S().dfs[Math.min(di,S().dfs.length-1)];
const pid=()=>S().id+'|'+P().i;
const total=D.reduce((n,s)=>n+s.dfs.length,0);
const esc=t=>String(t??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const px=(s,x)=>52.5+(s.d<0?-1:1)*Number(x), py=y=>34+Number(y);
const done=()=>Object.values(ans).filter(a=>a&&a.verdict).length;
const sceneDone=s=>s.dfs.filter(d=>(ans[s.id+'|'+d.i]||{}).verdict).length;

function fill(){
  const s=S(),p=P(),a=ans[pid()]||{};
  $('pick').innerHTML=D.map((x,i)=>
    `<option value="${i}">${i+1}. ${esc(x.label)} — ${sceneDone(x)}/${x.dfs.length}</option>`).join('');
  $('pick').value=si;
  $('count').textContent=`${done()} / ${total}`;
  $('bar').style.width=(100*done()/total)+'%';
  $('tabs').innerHTML=s.dfs.map((d,i)=>{
    const v=(ans[s.id+'|'+d.i]||{}).verdict;
    return `<button data-i="${i}" aria-selected="${i===di}">${esc(d.n)}${v?'<span class="mk">✓</span>':''}</button>`}).join('');
  $('who').innerHTML=
    `<dt><i class="dot" style="background:var(--runner)"></i>오프더볼 러너</dt><dd>${esc(s.r)}</dd>`+
    `<dt><i class="dot" style="background:var(--accent)"></i>반응 수비수</dt><dd>${esc(p.n)}</dd>`+
    `<dt><i class="dot" style="background:var(--gain)"></i>이득 보는 선수</dt><dd>${esc(p.bn)}</dd>`+
    `<dt><i class="dot" style="background:var(--carrier)"></i>볼 소유자</dt><dd>${esc(s.c)}</dd>`;
  $('note').value=a.note||'';
  document.querySelectorAll('#verdict button').forEach(b=>
    b.setAttribute('aria-pressed',String(a.verdict===b.dataset.v)));
  $('scrub').max=s.f.length-1;
}
function draw(){
  const s=S(),cur=P(),fr=s.f[Math.min(fi,s.f.length-1)];
  $('moving').innerHTML=fr.p.map(p=>{
    const atk=p[1]===1,isR=p[0]===s.ri,isC=p[0]===s.ci,
      isD=p[0]===cur.i,isB=p[0]===cur.bi;
    const ring=isR?'var(--runner)':isD?'var(--accent)':isB?'var(--gain)':isC?'var(--carrier)':'none';
    const rr=(isR||isC||isD||isB)?1.15:.72;
    return `<circle cx="${px(s,p[2])}" cy="${py(p[3])}" r="${rr}" fill="${atk?'var(--attack)':'var(--defend)'}"`+
      (ring==='none'?'':` stroke="${ring}" stroke-width=".5"`)+`><title>${esc(p[4])}</title></circle>`}).join('')
    +(fr.b?`<circle cx="${px(s,fr.b[0])}" cy="${py(fr.b[1])}" r=".45" fill="#0B0B0B" stroke="#fff" stroke-width=".18"/>`:'');
  $('scrub').value=fi;
  $('clock').textContent=`t ${fr.t>=0?'+':'−'}${Math.abs(fr.t).toFixed(2)}s`;
}
function go(i,j){si=(i%D.length+D.length)%D.length;
  di=Math.min(j===undefined?0:j,S().dfs.length-1);fi=0;fill();draw();}
function step(n){ // 수비수를 먼저 훑고, 끝나면 다음 장면으로
  let d=di+n;
  if(d<0)return go(si-1,99);
  if(d>=S().dfs.length)return go(si+1,0);
  di=d;fill();draw();}
function setVerdict(v){ans[pid()]=Object.assign({},ans[pid()],{verdict:v});store();fill();
  $('saved').textContent='저장됨 — 이 브라우저에 보관됩니다.';}
function toggle(){if(timer){clearInterval(timer);timer=null;$('play').innerHTML='▶ 재생<span class="kbd">Space</span>';return}
  $('play').innerHTML='❚❚ 정지<span class="kbd">Space</span>';
  timer=setInterval(()=>{fi=(fi+1)%S().f.length;draw()},85)}

$('pick').addEventListener('change',e=>go(+e.target.value,0));
$('tabs').addEventListener('click',e=>{const b=e.target.closest('button');
  if(b){di=+b.dataset.i;fi=0;fill();draw()}});
$('scrub').addEventListener('input',e=>{fi=+e.target.value;draw()});
$('play').addEventListener('click',toggle);
document.querySelectorAll('#verdict button').forEach(b=>b.addEventListener('click',()=>setVerdict(b.dataset.v)));
$('note').addEventListener('input',e=>{ans[pid()]=Object.assign({},ans[pid()],{note:e.target.value});store()});
$('prev').addEventListener('click',()=>step(-1));
$('next').addEventListener('click',()=>step(1));
function csv(){
  const head='match_id,onset_frame_id,defender_id,defender_name,runner_name,beneficiary_id,beneficiary_name,dilemma_verdict,note';
  const rows=[];
  D.forEach(s=>s.dfs.forEach(d=>{const a=ans[s.id+'|'+d.i]||{};
    rows.push([s.m,s.o,d.i,d.n,s.r,d.bi,d.bn,a.verdict||'',
      '"'+String(a.note||'').replace(/"/g,'""')+'"'].join(','))}));
  return head+'\\n'+rows.join('\\n');}
// The viewer sandbox blocks a page-initiated download, so the save goes
// through the host capability; when it is unavailable the CSV is shown for
// copying instead, which always works.
let saver=null;
(async()=>{try{saver=await window.claude.use('downloads')}catch(e){saver=null}})();
$('save').addEventListener('click',async()=>{
  const text=csv();
  if(saver){
    try{
      await saver.save({filename:'dilemma_relabel.csv',data:text});
      $('saved').textContent='CSV를 저장했습니다.';return;
    }catch(err){
      if(err&&err.code==='declined'){$('saved').textContent='저장을 취소하셨습니다.';return}
    }
  }
  const box=$('note');
  const dump=document.createElement('textarea');
  dump.value=text;dump.rows=10;dump.style.width='100%';dump.style.marginTop='10px';
  dump.setAttribute('aria-label','CSV 내용');
  box.parentElement.appendChild(dump);dump.select();
  $('saved').textContent='저장이 막혀 있어 CSV를 아래에 펼쳤습니다 — 전체 선택해서 복사해 주세요.';});
addEventListener('keydown',e=>{
  if(/^(INPUT|TEXTAREA|SELECT)$/.test(e.target.tagName))return;
  if(e.code==='Space'){e.preventDefault();toggle()}
  else if(e.key==='ArrowRight')step(1);
  else if(e.key==='ArrowLeft')step(-1);
  else if('1234'.includes(e.key))setVerdict(['clear','possible','unclear','none'][+e.key-1]);});
go(0,0);
if(done())$('saved').textContent=`이전 작업 ${done()}건을 불러왔습니다.`;
</script>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audit_dirs", nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument(
        "--skip-verdict",
        action="append",
        default=[],
        help=(
            "Drop scenes whose previous QC verdict was this. Using the old "
            "verdict to CHOOSE what to re-judge is not a leak - the reviewer "
            "never sees it, and he asked to skip what he was already sure "
            "about. It does mean those scenes keep their old label as "
            "negatives."
        ),
    )
    parser.add_argument(
        "--max-defenders",
        type=int,
        default=0,
        help=(
            "Keep only the first N candidate defenders. They arrive ordered "
            "by marking cost, a stage-1 quantity, so trimming by it carries "
            "no stage-3 information."
        ),
    )
    options = parser.parse_args()

    scenes = {}
    for directory in options.audit_dirs:
        for scene in json.loads(
            (Path(directory) / "local_game_payoff_audits.json").read_text()
        ):
            scenes[(str(scene["match_id"]), str(scene["onset_frame_id"]))] = scene

    skip = set(options.skip_verdict)
    previous = {}
    if skip:
        import csv as _csv

        reviews = ROOT / "examples/research_audit/human_reviews/shot_context"
        for name in (
            "settled_possession_onset_v0_1_reviews.csv",
            "shot_context_onset_v0_2_reviews.csv",
        ):
            path = reviews / name
            if not path.exists():
                continue
            for row in _csv.DictReader(path.open(encoding="utf-8-sig")):
                previous[(row["match_id"], row["onset_frame_id"])] = (
                    row.get("interaction_review") or ""
                ).strip()

    payload = []
    dropped = 0
    for key, scene in sorted(scenes.items()):
        if previous.get(key) in skip:
            dropped += 1
            continue
        frames = sorted(
            scene["background_frames"], key=lambda f: float(f["relative_time_s"])
        )
        if not frames:
            continue
        attacking = str(scene["attacking_team_id"])
        names = {str(p[0]): str(p[4]) for p in frames[0]["players"]}
        pairs = []
        for defender in scene["candidate_defenders"]:
            pick = beneficiary_of(scene, defender)
            if not pick:
                continue
            pairs.append(
                {
                    "i": str(defender["defender_id"]),
                    "n": str(defender["defender_name"]),
                    "bi": pick,
                    "bn": names.get(pick, pick),
                }
            )
        if options.max_defenders > 0:
            pairs = pairs[: options.max_defenders]
        if not pairs:
            continue
        payload.append(
            {
                "id": f"{key[0]}:{key[1]}",
                "m": key[0],
                "o": key[1],
                "label": f"{str(scene['match_label']).split('·')[0].strip()} · {key[1]}",
                "d": int(scene.get("attacking_direction") or 1),
                "r": str(scene["runner_name"]),
                "ri": str(scene["runner_id"]),
                "c": str(scene["carrier_name"]),
                "ci": str(scene["carrier_id"]),
                "dfs": pairs,
                "f": [
                    {
                        "t": round(float(f["relative_time_s"]), 2),
                        "p": [
                            [
                                str(p[0]),
                                1 if str(p[1]) == attacking else 0,
                                round(float(p[2]), 1),
                                round(float(p[3]), 1),
                                str(p[4]),
                            ]
                            for p in f["players"]
                        ],
                        "b": (
                            [round(float(f["ball"][0]), 1), round(float(f["ball"][1]), 1)]
                            if f.get("ball")
                            else None
                        ),
                    }
                    for f in frames
                ],
            }
        )

    random.Random(options.seed).shuffle(payload)
    html = PAGE.replace(
        "__DATA__", json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    )
    leaked = [token for token in FORBIDDEN_TOKENS if token in html]
    if leaked:
        raise SystemExit(f"blind screen leaks model output: {leaked}")

    options.out.parent.mkdir(parents=True, exist_ok=True)
    options.out.write_text(html, encoding="utf-8")
    size = options.out.stat().st_size / 1_048_576
    triangles = sum(len(s["dfs"]) for s in payload)
    if dropped:
        print(f"이전 판정이 {'/'.join(sorted(skip))}인 장면 {dropped}개 제외")
    print(f"장면 {len(payload)}개 · 판정 {triangles}개 -> {options.out}  ({size:.1f} MB)")
    print(f"누출 검사 통과 (금지 토큰 {len(FORBIDDEN_TOKENS)}개)")
    if size > 15:
        print("경고: 16 MB 한도에 근접")


if __name__ == "__main__":
    main()
