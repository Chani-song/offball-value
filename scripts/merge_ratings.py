#!/usr/bin/env python3
"""Merge the raters' CSVs from the rating page and reveal where each scene came from.

Reads every exported CSV (one per rater, from out/rating_v1/offball_dilemma_rating_v4.html)
and the private scene key (data/processed/rating_v1/scene_key.csv), and writes one
row per scene: each rater's off-ball answer (sure / unclear / no, from v4 on) and
score, the mean, median, spread and the notes, next to the source (Chani
strong/medium or solver 2v1/3v1) and the players involved.

Printed:
  completion   how many scenes each rater scored or marked "can't judge", and
               answered the off-ball question
  agreement    pairwise Spearman correlation on the scenes both scored
  by source    mean score per source -- does the solver's pick look like a
               dilemma to people as often as Chani's strong/medium scenes? --
               and how the off-ball answers split
  candidates   top scenes by mean (at least two scores), for the top-3/top-5 talk
  disagree     scenes whose scores span 2 or more points, to discuss first

A scene marked "can't judge" counts in completion, not in the mean.

Usage:
    python scripts/merge_ratings.py ratings/*.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OFFBALL = {"sure": "확신", "unclear": "애매", "no": "아님"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("csvs", nargs="+", type=Path)
    p.add_argument("--key", type=Path, default=ROOT / "data/processed/rating_v1/scene_key.csv")
    p.add_argument("--output", type=Path, default=ROOT / "data/processed/rating_v1/merged.csv")
    p.add_argument("--top", type=int, default=10)
    return p.parse_args()


def read(path: Path) -> pd.DataFrame:
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    d = pd.DataFrame(rows)
    need = {"rater", "scene", "rating", "unsure", "note"}
    if not need <= set(d.columns):
        raise SystemExit(f"{path}: not a rating-page CSV (columns {list(d.columns)})")
    d["rating"] = pd.to_numeric(d["rating"], errors="coerce")
    d["unsure"] = d["unsure"].astype(str).str.strip().isin(["1", "true", "True"])
    # v4 on; older CSVs have no off-ball column
    d["offball"] = d["offball"].astype(str).str.strip() if "offball" in d else ""
    d["offball"] = d["offball"].where(d["offball"].isin(OFFBALL))
    return d


def main() -> None:
    args = parse_args()
    key = pd.read_csv(args.key).set_index("code")
    frames = [read(p) for p in args.csvs]
    ratings = pd.concat(frames, ignore_index=True)
    if "version" in ratings:
        versions = sorted(set(ratings["version"].astype(str)))
        if len(versions) > 1:
            raise SystemExit(f"CSVs from different page versions {versions}: scene codes differ between them")
        print(f"페이지 버전 {versions[0]}")
    raters = list(dict.fromkeys(ratings["rater"]))
    wide = ratings.pivot_table(index="scene", columns="rater", values="rating", aggfunc="last")
    unsure = ratings[ratings["unsure"]].groupby("scene")["rater"].apply(lambda s: ", ".join(s))
    notes = (ratings[ratings["note"].astype(str).str.strip() != ""]
             .assign(n=lambda d: d["rater"] + ": " + d["note"].astype(str))
             .groupby("scene")["n"].apply(" | ".join))
    off = ratings.dropna(subset=["offball"]).pivot_table(index="scene", columns="rater", values="offball",
                                                         aggfunc="last")
    out = key.copy()
    for r in raters:
        out[f"offball_{r}"] = off.get(r)
        out[f"rating_{r}"] = wide.get(r)
    answers = out[[f"offball_{r}" for r in raters]]
    for v in OFFBALL:
        out[f"offball_{v}"] = (answers == v).sum(axis=1)
    scores = out[[f"rating_{r}" for r in raters]]
    out["n_rated"] = scores.notna().sum(axis=1)
    out["mean"] = scores.mean(axis=1)
    out["median"] = scores.median(axis=1)
    out["sd"] = scores.std(axis=1, ddof=0)
    out["range"] = scores.max(axis=1) - scores.min(axis=1)
    out["unsure_by"] = unsure.reindex(out.index).fillna("")
    out["notes"] = notes.reindex(out.index).fillna("")
    out["group"] = out["source"].map({"chani": "찬의"}).fillna("솔버") + " " + out["detail"].astype(str)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.reset_index().to_csv(args.output, index=False)

    print(f"평가자 {len(raters)}명: {', '.join(raters)} · 장면 {len(out)}")
    for r in raters:
        d = ratings[ratings["rater"] == r]
        done = int((d["rating"].notna() | d["unsure"]).sum())
        print(f"  {r}: {done}/{len(out)} (판단 불가 {int(d['unsure'].sum())}) · 오프더볼 답 {int(d['offball'].notna().sum())}")
    if len(raters) > 1:
        print("\n평가자 간 일치 (함께 점수 매긴 장면의 순위 상관)")
        for i, a in enumerate(raters):
            for b in raters[i + 1:]:
                both = scores[[f"rating_{a}", f"rating_{b}"]].dropna()
                rho = both.rank().corr().iloc[0, 1] if len(both) > 2 else float("nan")
                print(f"  {a} – {b}: {rho:+.2f} ({len(both)}장면)")
    print("\n출처별 평균 점수")
    for g, d in out.groupby("group"):
        split = " ".join(f"{lab} {int(d[f'offball_{v}'].sum())}" for v, lab in OFFBALL.items())
        print(f"  {g:<14} {d['mean'].mean():.2f}  ({len(d)}장면, 평균 점수가 4 이상 {int((d['mean'] >= 4).sum())})"
              f" · 오프더볼 답 {split}")
    cand = out[out["n_rated"] >= 2].sort_values(["mean", "sd"], ascending=[False, True]).head(args.top)
    print(f"\ntop {args.top} 후보 (점수 2개 이상, 평균 높은 순 · 같으면 의견이 모인 순)")
    for code, r in cand.iterrows():
        sc = " ".join(f"{x:.0f}" if pd.notna(x) else "-" for x in r[[f'rating_{q}' for q in raters]])
        print(f"  {code} 평균 {r['mean']:.2f} [{sc}] · {r['group']} · 러너 {r['runners']} / 수비 {r['defenders']}")
    dis = out[out["range"] >= 2].sort_values("range", ascending=False)
    print(f"\n의견이 2점 이상 갈린 장면 {len(dis)}개 (먼저 토의): {', '.join(dis.index[:15])}")
    print(f"\n→ {args.output}")


if __name__ == "__main__":
    main()
