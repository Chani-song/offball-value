#!/usr/bin/env python3
"""Cut the ~1,900 triples down to a shortlist two people can actually watch.

The pipeline emits one triple every 20 seconds of football. Nobody eyeballs
that. This picks a conservative few and copies their audit pages into one
folder, so the review is "open index.html and scroll" rather than "find scene
0435 under /work/hdd".

Two arms, because rank_triples.py showed the criteria disagree -- the top 5%
by second_best_q and by vacated_xt overlap on 15% of their picks. Picking one
criterion here would quietly decide a question the review is supposed to
answer, so both go in, labelled:

  consensus   mean percentile of minimax_worst_q and second_best_q. High on
              both means the moment is dangerous whatever the defender does
              AND the attack's second option also hurts. Requiring both is
              what makes it conservative: either alone admits moments where
              the defender simply covers the only real option.
  vacated_xt  PAUSA mass of the space the defender leaves behind. A different
              axis entirely -- it scores what the defence gives up rather than
              what the attack is worth.

Scene-level, not triple-level: several triples share one onset and one audit
page, so each scene appears once, represented by its best triple in that arm.

This does NOT rank by anything fitted to a reviewer's labels. Which arm picks
better scenes is the point of looking.
"""

from __future__ import annotations

import argparse
import json
import shutil
from html import escape
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ranking", type=Path,
                   default=Path("data/processed/triple_ranking_xt.csv"))
    p.add_argument("--build", type=Path,
                   default=Path("/work/hdd/bbmr/kseo1/offball-out/v5_hybrid"))
    p.add_argument("--output", type=Path, default=Path("out/review_shortlist"))
    p.add_argument("--top-n", type=int, default=20,
                   help="팔당 장면 수 (기본 20 = 경기당 약 3개)")
    p.add_argument("--dedupe-seconds", type=float, default=20.0,
                   help="같은 경기에서 이 간격 안의 온셋은 최고점 하나만 남긴다")
    p.add_argument("--fps", type=float, default=25.0,
                   help="트래킹 프레임레이트 (IDSSE = 25)")
    return p.parse_args()


def build_scene_index(build: Path) -> dict[tuple[str, int], Path]:
    """Map (match_id, onset_frame_id) -> scene directory.

    Reads the summary CSVs (2 MB over all scenes) rather than the audit JSONs
    (5 MB each); the JSONs are opened later for the shortlist only.
    """
    index: dict[tuple[str, int], Path] = {}
    paths = sorted(build.glob("scene_*/local_game_payoff_summary.csv"))
    print(f"장면 인덱스 작성 중... ({len(paths)}개)", flush=True)
    for path in paths:
        try:
            frame = pd.read_csv(path, usecols=["match_id", "onset_frame_id"])
        except (ValueError, pd.errors.EmptyDataError):
            continue
        for match_id, onset in frame.drop_duplicates().itertuples(index=False):
            index[(str(match_id), int(onset))] = path.parent
    print(f"  {len(index)}개 장면", flush=True)
    return index


def scene_detail(scene_dir: Path) -> dict:
    """Names for one scene, from its audit JSON."""
    path = scene_dir / "local_game_payoff_audits.json"
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    if not payload:
        return {}
    game = payload[0]
    names = {p[0]: p[4] for p in game["background_frames"][0]["players"]}
    defenders = {
        str(d["defender_id"]): d.get("defender_name") or str(d["defender_id"])
        for d in (game.get("candidate_defenders") or [])
    }
    # The third member of the triple is the defender's derived option, not the
    # ball carrier: carrier is whoever holds the ball, which is context.
    beneficiaries = {
        str(d["defender_id"]): names.get(d.get("derived_option_id"), "—")
        for d in (game.get("candidate_defenders") or [])
    }
    return {
        "match_label": game.get("match_label") or game.get("match_id"),
        "runner_name": game.get("runner_name"),
        "carrier_name": game.get("carrier_name"),
        "defenders": defenders,
        "beneficiaries": beneficiaries,
    }


def pick(frame: pd.DataFrame, column: str, top_n: int,
         dedupe_seconds: float, fps: float) -> pd.DataFrame:
    """Top N scenes by `column`, no two within `dedupe_seconds` of each other.

    Deduping on onset_frame_id alone is not enough. Consecutive onsets in one
    possession are separate rows but the same passage of play: at 1.8 s apart
    the pair is the same moment scored against a second defender, and at 3.1 s
    the runner and the beneficiary have simply swapped roles. Six of the first
    twenty consensus picks were such pairs -- a third of a review budget spent
    watching the same football twice.

    Greedy suppression: walk the ranking downwards and keep a scene only if it
    is far enough from every scene already kept in that match. The best of each
    cluster survives, which is the one worth watching.
    """
    ordered = frame.sort_values(column, ascending=False)
    per_scene = ordered.drop_duplicates(subset=["match_id", "onset_frame_id"])
    window = dedupe_seconds * fps
    kept: list[int] = []
    seen: dict[str, list[int]] = {}
    for row in per_scene.itertuples():
        onset = int(row.onset_frame_id)
        near = seen.setdefault(row.match_id, [])
        if any(abs(onset - other) < window for other in near):
            continue
        near.append(onset)
        kept.append(row.Index)
        if len(kept) >= top_n:
            break
    return per_scene.loc[kept].copy()


ARM_NOTE = {
    "consensus": "minimax_worst_q · second_best_q 백분위 평균 — 수비수가 최선으로 "
                 "대응해도 위험하고, 공격의 2순위 옵션도 아픈 순간",
    "vacated_xt": "수비수가 비워둔 공간의 PAUSA 합 — 수비가 무엇을 내주는가",
}


def write_index(rows: list[dict], output: Path, top_n: int,
                dedupe_seconds: float) -> None:
    parts = [
        "<!doctype html><meta charset='utf-8'>",
        "<title>Review shortlist</title>",
        "<style>",
        "body{font:15px/1.6 system-ui,-apple-system,sans-serif;margin:0;",
        "background:#fbfbfa;color:#1c1c1a}",
        ".wrap{max-width:1040px;margin:0 auto;padding:32px 16px 64px}",
        "h1{font-size:24px;margin:0 0 4px}",
        ".sub{color:#6b6b66;margin:0 0 28px}",
        "h2{font-size:17px;margin:34px 0 2px}",
        ".note{color:#6b6b66;font-size:13px;margin:0 0 14px}",
        "table{border-collapse:collapse;width:100%;background:#fff;",
        "border:1px solid #e6e5e1;border-radius:8px;overflow:hidden}",
        "th,td{padding:8px 10px;text-align:left;border-bottom:1px solid #f0efec;",
        "font-size:13.5px;vertical-align:top}",
        "th{background:#f6f5f2;font-weight:600;color:#55544f;",
        "font-size:12px;text-transform:uppercase;letter-spacing:.04em}",
        "tr:last-child td{border-bottom:0}",
        "td.num{text-align:right;font-variant-numeric:tabular-nums;color:#55544f}",
        "a{color:#1a5fb4;text-decoration:none}a:hover{text-decoration:underline}",
        ".both{display:inline-block;margin-left:6px;padding:1px 6px;border-radius:9px;",
        "background:#e8f0e4;color:#3d6b2c;font-size:11px}",
        "</style>",
        "<div class=wrap>",
        "<h1>검토 대상 장면</h1>",
        f"<p class=sub>기준별 상위 {top_n}개 장면. 같은 경기에서 "
        f"{dedupe_seconds:.0f}초 안에 붙은 온셋은 최고점 하나만 남겼습니다. "
        "각 행의 링크가 그 장면의 payoff 감사 페이지입니다. "
        "삼중항은 러너 · 수비수 · 수혜자이고, 볼 소유자는 맥락입니다.</p>",
    ]
    for arm in ("consensus", "vacated_xt"):
        arm_rows = [r for r in rows if r["arm"] == arm]
        if not arm_rows:
            continue
        parts.append(f"<h2>{escape(arm)}</h2>")
        parts.append(f"<p class=note>{escape(ARM_NOTE[arm])}</p>")
        parts.append("<table><tr><th>#</th><th>경기</th><th>프레임</th>"
                     "<th>러너</th><th>수비수</th><th>수혜자</th>"
                     "<th>볼 소유자</th><th class=num>minimax</th>"
                     "<th class=num>2nd</th><th class=num>vacated_xt</th>"
                     "<th>감사</th></tr>")
        for i, r in enumerate(arm_rows, 1):
            both = "<span class=both>양쪽</span>" if r["in_both"] else ""
            parts.append(
                f"<tr><td class=num>{i}</td>"
                f"<td>{escape(str(r['match_label']))}</td>"
                f"<td class=num>{r['onset_frame_id']}</td>"
                f"<td>{escape(str(r['runner_name']))}</td>"
                f"<td>{escape(str(r['defender_name']))}</td>"
                f"<td>{escape(str(r['beneficiary_name']))}{both}</td>"
                f"<td>{escape(str(r['carrier_name']))}</td>"
                f"<td class=num>{r['minimax_worst_q']:.3f}</td>"
                f"<td class=num>{r['second_best_q']:.3f}</td>"
                f"<td class=num>{r['vacated_xt']:.2f}</td>"
                f"<td><a href='{escape(r['href'])}'>열기</a></td></tr>"
            )
        parts.append("</table>")
    parts.append("</div>")
    (output / "index.html").write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.ranking)
    print(f"삼중항 {len(frame)}개 · 장면 "
          f"{frame.groupby(['match_id', 'onset_frame_id']).ngroups}개\n")

    frame["consensus"] = (
        frame["minimax_worst_q"].rank(pct=True)
        + frame["second_best_q"].rank(pct=True)
    ) / 2

    index = build_scene_index(args.build)

    picks = {
        arm: pick(frame, arm, args.top_n, args.dedupe_seconds, args.fps)
        for arm in ("consensus", "vacated_xt")
    }
    keys = {
        arm: {(r.match_id, int(r.onset_frame_id)) for r in sel.itertuples()}
        for arm, sel in picks.items()
    }
    overlap = keys["consensus"] & keys["vacated_xt"]
    union = keys["consensus"] | keys["vacated_xt"]
    print(f"\nconsensus {len(keys['consensus'])} · vacated_xt {len(keys['vacated_xt'])}"
          f" · 겹침 {len(overlap)} · 합집합 {len(union)}개 장면")

    args.output.mkdir(parents=True, exist_ok=True)
    details: dict[tuple[str, int], dict] = {}
    rows: list[dict] = []
    missing = 0
    for arm, sel in picks.items():
        for r in sel.itertuples():
            key = (r.match_id, int(r.onset_frame_id))
            scene_dir = index.get(key)
            if scene_dir is None:
                missing += 1
                continue
            if key not in details:
                details[key] = scene_detail(scene_dir)
            detail = details[key]
            name = f"{scene_dir.name}_{r.match_id}_{int(r.onset_frame_id)}.html"
            target = args.output / name
            if not target.exists():
                shutil.copy2(scene_dir / "local_game_payoff_audit.html", target)
            rows.append({
                "arm": arm,
                "match_label": detail.get("match_label", r.match_id),
                "onset_frame_id": int(r.onset_frame_id),
                "runner_name": detail.get("runner_name", r.runner_id),
                "carrier_name": detail.get("carrier_name", "?"),
                "defender_name": (detail.get("defenders") or {}).get(
                    r.defender_id, r.defender_id),
                "beneficiary_name": (detail.get("beneficiaries") or {}).get(
                    r.defender_id, "—"),
                "minimax_worst_q": float(r.minimax_worst_q),
                "second_best_q": float(r.second_best_q),
                "vacated_xt": float(r.vacated_xt),
                "in_both": key in overlap,
                "href": name,
            })
    if missing:
        print(f"경고: 장면 디렉토리를 못 찾은 삼중항 {missing}개")

    write_index(rows, args.output, args.top_n, args.dedupe_seconds)
    size = sum(f.stat().st_size for f in args.output.glob("*.html")) / 1e6
    print(f"\n{args.output}/index.html")
    print(f"  감사 페이지 {len(list(args.output.glob('scene_*.html')))}개 · {size:.0f} MB")


if __name__ == "__main__":
    main()
