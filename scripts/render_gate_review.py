#!/usr/bin/env python3
"""One card per scene showing every candidate defender and the gate's call.

The triple overview draws one defender per card, which is right for judging
a triple and useless for judging the rule that decides which defenders get
a card at all. Here the card is the scene: the runner, every candidate the
pipeline considered, the ones the gate kept (glowing) and the ones it
dropped (grey, dashed). The reviewer answers, per candidate, "should this
man be in the game?", and writes in the name of anyone the pipeline never
considered.

Two arms. The labelled scenes come first, with a star on each defender an
expert already marked, so the reviewer can see where the rule and the expert
agree. Then a fixed-seed sample stratified by how many defenders the gate
kept, so scenes with one, two and three survivors are all represented in
the proportion they occur.

The beneficiary drawn is the one from the rank-1 defender's game; each
candidate's own beneficiary is in the table. Trails are drawn for the runner
and kept defenders only.
"""

from __future__ import annotations

import argparse
import json
import random
from html import escape
from pathlib import Path

import pandas as pd

from render_triple_overview import (
    CSS, NAV_BAR, PLAY_JS, R_BALL, R_PLAYER, ROLE_RING, W, H,
    pitch_markings, player_bar, trail, xy,
)

REASON_KR = {
    "rank1": "1순위",
    "neighbour": "1위 옆 (Δ≤3m)",
    "run_arrives": "런이 옴 (catch≥3·종점≤11)",
    "": "탈락",
}
DROP_RING = "#d7dbe0"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--gate", type=Path,
                   default=Path("data/processed/pair_gate_v7.csv"))
    p.add_argument("--build", type=Path,
                   default=Path("/work/hdd/bbmr/kseo1/offball-out/v7_r9_ssac"))
    p.add_argument("--output", type=Path,
                   default=Path("out/gate_review/index.html"))
    p.add_argument("--sample-n", type=int, default=40)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--trail-stride", type=int, default=4)
    return p.parse_args()


def scene_svg(game: dict, cands: pd.DataFrame, stride: int) -> tuple[str, dict]:
    frames = game["background_frames"]
    onset = game["onset_frame_id"]
    flip = int(game.get("attacking_direction", 1)) < 0
    at_onset = min(frames, key=lambda f: abs(f["frame_id"] - onset))
    carrier = str(game["carrier_id"])
    attacking_team = str(game["attacking_team_id"])

    runner = str(game["runner_id"])
    kept = {str(r.defender_id) for r in cands.itertuples() if r.kept}
    dropped = {str(r.defender_id) for r in cands.itertuples() if not r.kept}
    first = cands[cands["rank"] == 1]
    ben = str(first["beneficiary_id"].iloc[0]) if len(first) and \
        isinstance(first["beneficiary_id"].iloc[0], str) else None

    def role_of(pid: str) -> str | None:
        if pid == runner:
            return "runner"
        if pid in kept:
            return "defender"
        if pid in dropped:
            return "dropped"
        if ben and pid == ben:
            return "beneficiary"
        return None

    out = [f"<svg viewBox='0 0 {W:.0f} {H:.0f}' class=pitch role=img "
           f"aria-label='피치 다이어그램'>",
           f"<rect width='{W:.0f}' height='{H:.0f}' rx='8' fill='var(--pitch-fill)'/>",
           pitch_markings(flip), "<g class=trails>"]
    for pid in [runner, *sorted(kept)]:
        line = trail(frames, pid, flip, stride)
        if line:
            out.append(f"<g style='color:{ROLE_RING[role_of(pid)]}'>{line}</g>")
    out.append("</g>")

    order = [str(p[0]) for p in at_onset["players"]]
    for slot, p in enumerate(at_onset["players"]):
        pid, team, x, y, name = str(p[0]), str(p[1]), p[2], p[3], p[4]
        px, py = xy(x, y, flip)
        attacking = team == attacking_team
        fill = "var(--team-att)" if attacking else "var(--team-def)"
        role = role_of(pid)
        side = "공격" if attacking else "수비"
        tag = {"runner": "러너", "defender": "유지 수비수", "dropped": "탈락 후보",
               "beneficiary": "수혜자"}.get(role, "")
        title = f"{name} · {side}" + (f" · {tag}" if tag else "")
        glow = ""
        if role == "dropped":
            glow = (f"<circle class='gl' data-i='{slot}' cx='{px:.1f}' cy='{py:.1f}' "
                    f"r='{R_PLAYER + 2.5:.1f}' fill='none' stroke='{DROP_RING}' "
                    f"stroke-width='2.2' stroke-dasharray='4 3' opacity='.95'/>")
        elif role:
            glow = (f"<circle class='gl' data-i='{slot}' cx='{px:.1f}' cy='{py:.1f}' "
                    f"r='{R_PLAYER + 2.5:.1f}' fill='none' stroke='{ROLE_RING[role]}' "
                    f"stroke-width='3' style='filter:drop-shadow(0 0 5px {ROLE_RING[role]})'/>")
        dot = (f"<circle class='pl' data-i='{slot}' data-n=\'{escape(title)}\' "
               f"cx='{px:.1f}' cy='{py:.1f}' r='{R_PLAYER}' fill='{fill}' "
               f"stroke='var(--dot-line)' stroke-width='1'/>")
        pip = (f"<circle class='pip' data-i='{slot}' cx='{px:.1f}' cy='{py:.1f}' "
               f"r='2' fill='#fff' opacity='.9' pointer-events='none'/>") \
            if pid == carrier else ""
        out.append(glow + dot + pip)

    bx, by = xy(at_onset["ball"][0], at_onset["ball"][1], flip)
    out.append(f"<circle class=ball cx='{bx:.1f}' cy='{by:.1f}' r='{R_BALL}' "
               f"fill='#fff' stroke='#111' stroke-width='1.4' pointer-events='none'/>")
    out.append("</svg>")

    positions, ball, clock = [], [], []
    for f in frames:
        by_id = {str(p[0]): (p[2], p[3]) for p in f["players"]}
        row = []
        for pid in order:
            if pid in by_id:
                fx, fy = xy(by_id[pid][0], by_id[pid][1], flip)
                row.append([round(fx, 1), round(fy, 1)])
            else:
                row.append(None)
        positions.append(row)
        fbx, fby = xy(f["ball"][0], f["ball"][1], flip)
        ball.append([round(fbx, 1), round(fby, 1)])
        clock.append(round(float(f["relative_time_s"]), 2))
    onset_index = min(range(len(frames)), key=lambda i: abs(frames[i]["frame_id"] - onset))
    return "".join(out), {"p": positions, "b": ball, "t": clock, "onset": onset_index}


def candidate_table(card_id: str, cands: pd.DataFrame) -> str:
    rows = []
    for r in cands.sort_values("rank").itertuples():
        k = f"{card_id}:{r.defender_id}"
        buttons = "".join(
            f"<button type=button class=v data-k=\'{escape(k)}\' data-v=\'{v}\'>{t}</button>"
            for v, t in (("ok", "담당 맞음"), ("bad", "아님"), ("meh", "애매")))
        state = ("<span class=kept>유지</span>" if r.kept else "<span class=drop>탈락</span>")
        star = " ★" if r.labelled else ""
        ben = r.beneficiary_name if isinstance(r.beneficiary_name, str) else "—"
        rows.append(
            f"<tr class='{'ck' if r.kept else 'cd'}'>"
            f"<td class=num>{r.rank}</td>"
            f"<td><b>{escape(str(r.defender_name))}</b>{star}</td>"
            f"<td>{state} <span class=why>{escape(REASON_KR.get(r.reason if isinstance(r.reason, str) else '', '탈락'))}</span></td>"
            f"<td class=num>{r.delta_m:.1f}</td><td class=num>{r.catch_m:+.1f}</td>"
            f"<td class=num>{r.path_end_m:.1f}</td>"
            f"<td>{escape(str(ben))}</td>"
            f"<td class=vb>{buttons}</td></tr>")
    return (
        "<table class=cands><tr><th>#</th><th>수비수</th><th>게이트</th>"
        "<th class=num>Δ m</th><th class=num>catch</th><th class=num>종점 m</th>"
        "<th>수혜자</th><th>이 수비수가 담당인가?</th></tr>"
        + "".join(rows) + "</table>"
        + f"<input class=note data-k=\'{escape(card_id)}\' "
          f"placeholder=\'메모 — 파이프라인이 아예 안 본 수비수가 담당이면 이름을 적어 주세요\'>"
    )


EXTRA_CSS = """
.cands{margin-top:8px}
.cands td,.cands th{padding:5px 7px;font-size:12.5px}
.cands tr.cd td{color:var(--muted)}
.cands tr.cd td b{font-weight:500}
.kept{color:#0a8a5a;font-weight:600}.drop{color:var(--muted);font-weight:600}
.why{font-size:11.5px;color:var(--muted);margin-left:3px}
.vb{white-space:nowrap}.vb .v{margin-right:3px}
.glow-drop{width:13px;height:13px;border-radius:50%;flex:none;
 border:2px dashed #b9bec5;background:var(--team-def)}
.runner-name{font-size:13px;color:var(--text-secondary)}
"""

REVIEW_JS = """
<script>
(function () {
  var KEY = 'offball-gate-review-v1';
  var data = {};
  function load() { try { data = JSON.parse(localStorage.getItem(KEY) || '{}') || {}; } catch (e) { data = {}; } }
  function save() { try { localStorage.setItem(KEY, JSON.stringify(data)); } catch (e) {} }
  function cards() { return document.querySelectorAll('.stage .card'); }
  function progress() {
    var done = 0, all = cards();
    all.forEach(function (c) {
      var btns = c.querySelectorAll('button.v'), keys = {};
      btns.forEach(function (b) { keys[b.dataset.k] = 1; });
      var ok = Object.keys(keys).every(function (k) { return data[k] && data[k].v; });
      if (ok && Object.keys(keys).length) done++;
    });
    var el = document.querySelector('.prog');
    if (el) el.textContent = done + ' / ' + all.length + ' 장면 판정함';
  }
  function paint() {
    document.querySelectorAll('button.v').forEach(function (b) {
      var e = data[b.dataset.k] || {};
      b.setAttribute('aria-pressed', e.v === b.dataset.v ? 'true' : 'false');
    });
    document.querySelectorAll('input.note').forEach(function (i) {
      var e = data[i.dataset.k] || {};
      if (e.note !== undefined) i.value = e.note;
    });
    progress();
  }
  document.addEventListener('click', function (ev) {
    var b = ev.target.closest && ev.target.closest('button.v');
    if (!b) return;
    var e = data[b.dataset.k] = data[b.dataset.k] || {};
    e.v = (e.v === b.dataset.v) ? '' : b.dataset.v;
    save(); paint();
  });
  document.addEventListener('input', function (ev) {
    if (!ev.target.matches || !ev.target.matches('input.note')) return;
    var e = data[ev.target.dataset.k] = data[ev.target.dataset.k] || {};
    e.note = ev.target.value; save(); progress();
  });
  function cell(v) { v = (v === undefined || v === null) ? '' : String(v);
    return /[",\\n]/.test(v) ? '"' + v.replace(/"/g, '""') + '"' : v; }
  function csv() {
    var lines = ['match_id,onset_frame_id,runner_id,defender_id,rank,kept,verdict,note'];
    cards().forEach(function (c) {
      var cid = c.dataset.card, note = (data[cid] || {}).note || '';
      c.querySelectorAll('tr.ck, tr.cd').forEach(function (tr) {
        var b = tr.querySelector('button.v'); if (!b) return;
        var k = b.dataset.k, v = (data[k] || {}).v || '';
        if (!v && !note) return;
        var p = k.split(':');
        lines.push([p[0], p[1], p[2], p[3], tr.dataset.rank, tr.classList.contains('ck') ? 1 : 0, v, note].map(cell).join(','));
      });
    });
    return lines.join('\\n') + '\\n';
  }
  window.addEventListener('DOMContentLoaded', function () {
    load(); paint();
    var ex = document.querySelector('.export');
    if (ex) ex.addEventListener('click', function () {
      var blob = new Blob([csv()], {type: 'text/csv;charset=utf-8'});
      var a = document.createElement('a'); a.href = URL.createObjectURL(blob);
      a.download = 'gate_review.csv'; a.click(); URL.revokeObjectURL(a.href);
    });
    var cl = document.querySelector('.clear');
    if (cl) cl.addEventListener('click', function () {
      if (!confirm('이 브라우저에 저장된 판정을 모두 지웁니다. 계속할까요?')) return;
      data = {}; save(); paint();
      document.querySelectorAll('input.note').forEach(function (i) { i.value = ''; });
    });
  });
})();
</script>
"""


def main() -> None:
    args = parse_args()
    gate = pd.read_csv(args.gate)
    key = ["match_id", "onset_frame_id", "runner_id"]
    scenes = gate.groupby(key)
    summary = scenes.agg(kept_n=("kept", "sum"), labelled=("labelled", "any"),
                         scene_dir=("scene_dir", "first")).reset_index()

    labelled = summary[summary["labelled"]].sort_values(key)
    pool = summary[~summary["labelled"]]
    rng = random.Random(args.seed)
    buckets = {1: pool[pool["kept_n"] == 1], 2: pool[pool["kept_n"] == 2],
               3: pool[pool["kept_n"] >= 3]}
    total = sum(len(b) for b in buckets.values())
    picks = []
    for k, b in buckets.items():
        n = round(args.sample_n * len(b) / total)
        picks.extend(rng.sample(list(b.index), min(n, len(b))))
    sample = pool.loc[sorted(picks)]

    arms = [("라벨 장면", labelled), ("무작위 표본", sample)]
    cards, clips, listing = [], {}, []
    for arm, subset in arms:
        for n, s in enumerate(subset.itertuples(), 1):
            path = args.build / s.scene_dir / "local_game_payoff_audits.json"
            payload = json.loads(path.read_text())
            if not payload:
                continue
            game = payload[0]
            cands = gate[(gate["match_id"] == s.match_id)
                         & (gate["onset_frame_id"] == s.onset_frame_id)
                         & (gate["runner_id"] == s.runner_id)]
            card_id = f"{s.match_id}:{int(s.onset_frame_id)}:{s.runner_id}"
            svg, clip = scene_svg(game, cands, args.trail_stride)
            clips[card_id] = clip
            match = str(game.get("match_label", s.match_id)).split(" · ")[0]
            table = candidate_table(card_id, cands)
            # data-rank on rows for the CSV export
            for r in cands.itertuples():
                table = table.replace(
                    f"<tr class='{'ck' if r.kept else 'cd'}'><td class=num>{r.rank}</td>",
                    f"<tr class='{'ck' if r.kept else 'cd'}' data-rank='{r.rank}'>"
                    f"<td class=num>{r.rank}</td>", 1)
            cards.append(
                f"<div class=card data-clip='{escape(card_id)}' data-card='{escape(card_id)}'>"
                f"<div class=hd><span class=rank>{arm} #{n}</span>"
                f"<span class=match title='{escape(match)}'>{escape(match)} · f{int(s.onset_frame_id)}</span></div>"
                f"<div class=runner-name>러너 <b>{escape(str(game.get('runner_name')))}</b>"
                f" · 볼 소유자 {escape(str(game.get('carrier_name')))}"
                f" · 후보 {len(cands)}명 중 {int(s.kept_n)}명 유지</div>"
                f"{svg}{player_bar()}{table}</div>")
            listing.append({"arm": arm, "n": n, "match_id": s.match_id,
                            "onset_frame_id": int(s.onset_frame_id),
                            "runner_id": s.runner_id, "scene_dir": s.scene_dir,
                            "kept_n": int(s.kept_n)})

    legend = (
        "<div class=legend>"
        "<span class=key><span class=tdot style='background:var(--team-att)'></span>공격팀</span>"
        "<span class=key><span class=tdot style='background:var(--team-def)'></span>수비팀</span>"
        "<span class=key><span class=gdot style='color:var(--glow-runner)'></span>러너</span>"
        "<span class=key><span class=gdot style='color:var(--glow-defender)'></span>유지된 수비수</span>"
        "<span class=key><span class=glow-drop></span>탈락한 후보</span>"
        "<span class=key><span class=gdot style='color:var(--glow-benef)'></span>수혜자 (1순위 수비수 게임)</span>"
        "<span class=key>흰 점 = 볼 소유자 · ★ = 전문가 라벨</span>"
        "<label class=key><input type=checkbox class=trailtog checked> 궤적 보기</label></div>"
    )
    bar = (
        "<div class=bar><span class=prog>0 / 0 장면 판정함</span>"
        "<button type=button class=export>CSV 내려받기</button>"
        "<button type=button class=clear>판정 지우기</button>"
        "<span style='font-size:12.5px;color:var(--muted)'>판정은 이 브라우저에만 저장됩니다.</span></div>"
    )
    html = (
        "<!doctype html><html lang=ko><meta charset=utf-8>"
        "<meta name=viewport content='width=device-width,initial-scale=1'>"
        "<title>수비수 게이트 검토</title>"
        f"<style>{CSS}{EXTRA_CSS}</style><body><div class=wrap>"
        "<h1>러너–수비수 게이트 검토</h1>"
        "<p class=sub>규칙: 1순위 유지 · 런 경로에서 1위보다 3m 이내면 유지 · "
        "런이 자기 쪽으로 와서(6m/s로 3m 여유 있게 닿고 종점 11m 이내) 유지 · 나머지 탈락. "
        "각 후보에 대해 \"이 수비수가 이 런에 반응해야 할 사람인가\"만 판정해 주세요. "
        "탈락 후보가 실은 담당이면 '담당 맞음', 유지 후보가 엉뚱하면 '아님'.</p>"
        + bar + legend + NAV_BAR
        + f"<div class=stage>{''.join(cards)}</div></div>"
        + PLAY_JS.replace("__CLIPS__", json.dumps(clips, separators=(",", ":")))
        + REVIEW_JS + "</body></html>"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html, encoding="utf-8")
    pd.DataFrame(listing).to_csv(args.output.with_name("cards.csv"), index=False)
    print(f"{args.output}  ({args.output.stat().st_size / 1024:.0f} KB) · 카드 {len(cards)}")


if __name__ == "__main__":
    main()
