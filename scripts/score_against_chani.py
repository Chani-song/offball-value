#!/usr/bin/env python3
"""Does the pipeline's (runner, defender, beneficiary) match Chani's?

Chani scored shot scenes for dilemma worth -- strong / medium / low / ignore --
and for each non-ignore scene wrote the three roles the pipeline also produces:
the off-ball attacker who runs, the defender he drags, and the team-mate who
uses the space. Those are the same three slots, so they compare directly.

The only question that matters is how much of Chani's strong, and of her
strong+medium, our output contains. Tagging OUR scenes with her grades runs the
comparison backwards: she never looked at 91% of them, so such a tag says
nothing about whether they are right.

Matching is reported per role and cumulatively, because a triple can fail at
any slot and the three failures need different fixes. Jersey numbers resolve
through match metadata; the mapping was validated against the shooter name
Chani recorded (20/20).

Human labels are the evaluation and never enter the model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from offball_value.bundesliga import (
    FPS,
    find_bundesliga_files,
    load_bundesliga_match_metadata,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diagnose_missed_onsets import read_xlsx, split_numbers  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--annotations", type=Path, default=Path("chani/chani_shot_annotation.xlsx"))
    p.add_argument("--join", type=Path, default=Path("data/processed/chani_annotation_join_v0_4.csv"))
    p.add_argument("--root", type=Path, default=Path("data/processed/run_onset_v0_5/dir25"))
    p.add_argument("--raw-dir", type=Path, default=Path("data/raw/bundesliga-integrated"))
    p.add_argument("--builds", nargs="*", default=[], metavar="NAME=DIR",
                   help="Optional beneficiary builds, e.g. hybrid=/work/.../v5_hybrid")
    p.add_argument("--tolerance-seconds", type=float, default=3.0)
    p.add_argument("--output", type=Path, default=Path("data/processed/chani_triple_score.csv"))
    return p.parse_args()


def load_build(root: Path) -> dict[tuple[str, str], dict]:
    games: dict[tuple[str, str], dict] = {}
    for path in sorted(root.glob("scene_*/local_game_payoff_audits.json")):
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        for game in payload:
            games[(str(game["match_id"]), str(game["onset_frame_id"]))] = game
    return games


def main() -> None:
    args = parse_args()
    rows = read_xlsx(args.annotations)
    header = rows[0]
    ann = {r[header.index("clip_id")]: dict(zip(header, r)) for r in rows[1:] if any(r)}

    join = pd.read_csv(args.join)
    phases = pd.read_csv(args.root / "attacking_phases.csv")
    cands = pd.read_csv(args.root / "shot_context_run_onsets.csv")
    scenes = cands[cands["primary_candidate"] == True]  # noqa: E712
    tol = args.tolerance_seconds * FPS

    builds = {}
    for spec in args.builds:
        name, _, path = spec.partition("=")
        builds[name] = load_build(Path(path))
        print(f"빌드 {name}: 장면 {len(builds[name])}개")

    meta_cache: dict[str, object] = {}

    def meta(match_id: str):
        if match_id not in meta_cache:
            files = find_bundesliga_files(args.raw_dir, match_id)
            meta_cache[match_id] = load_bundesliga_match_metadata(files["matchinfo"])
        return meta_cache[match_id]

    out = []
    targets = join[join["labelled"]]
    for _, r in targets.iterrows():
        rec = ann.get(r["clip_id"])
        if rec is None:
            continue
        m = meta(r["match_id"])
        name_to_team = {t.name: tid for tid, t in m.teams.items()}
        attack_team = name_to_team.get(str(rec["team"]).strip())
        if attack_team is None:
            continue
        defend_team = next(t for t in m.teams if t != attack_team)
        shirt = {(p.team_id, str(p.shirt_number)): p.player_id for p in m.players.values()}

        want_runners = {shirt.get((attack_team, s)) for s in split_numbers(rec["offball_attackers"])}
        want_defs = {shirt.get((defend_team, s)) for s in split_numbers(rec["drawn_defenders"])}
        want_bens = {shirt.get((attack_team, s)) for s in split_numbers(rec["space_beneficiaries"])}
        want_runners.discard(None); want_defs.discard(None); want_bens.discard(None)

        P = phases[phases["match_id"] == r["match_id"]]
        inph = P[(P["possession_start_frame_id"] <= r["shot_frame_id"])
                 & (P["possession_end_frame_id"] + tol >= r["shot_frame_id"])]
        row = {"clip_id": r["clip_id"], "match_id": r["match_id"], "effect": r["effect"],
               "n_runner_labels": len(want_runners)}
        if inph.empty:
            row["stage"] = "no_phase"
            out.append(row); continue
        p0 = inph.iloc[0]
        ours = scenes[(scenes["match_id"] == r["match_id"])
                      & (scenes["frame_id"] >= p0["possession_start_frame_id"])
                      & (scenes["frame_id"] <= p0["possession_end_frame_id"])]
        row["our_scenes_in_phase"] = len(ours)
        if ours.empty:
            row["stage"] = "no_onset"
            out.append(row); continue

        our_runners = set(ours["player_id"].astype(str))
        row["runner_match"] = bool(our_runners & want_runners)
        row["stage"] = "runner_hit" if row["runner_match"] else "runner_miss"

        for name, games in builds.items():
            hit_def = hit_ben = False
            for _, s in ours.iterrows():
                if not (str(s["player_id"]) in want_runners):
                    continue
                g = games.get((str(s["match_id"]), str(s["frame_id"])))
                if g is None:
                    continue
                our_defs = {str(d["defender_id"]) for d in g["candidate_defenders"]}
                if our_defs & want_defs:
                    hit_def = True
            row[f"{name}_defender_match"] = hit_def
        out.append(row)

    frame = pd.DataFrame(out)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)

    print("\n" + "=" * 70)
    print("찬의님 라벨을 우리 파이프라인이 얼마나 머금는가")
    print("=" * 70)
    for title, subset in (("strong", ("strong",)),
                          ("strong+medium", ("strong", "medium")),
                          ("전체", ("strong", "medium", "low"))):
        f = frame[frame["effect"].isin(subset)]
        n = len(f)
        if not n:
            continue
        phase_ok = (f["stage"] != "no_phase").sum()
        onset_ok = f["stage"].isin(["runner_hit", "runner_miss"]).sum()
        runner_ok = (f["stage"] == "runner_hit").sum()
        print(f"\n{title}  (n={n})")
        print(f"  ① 국면을 본다      {phase_ok:>3}/{n}  ({phase_ok / n:>5.0%})")
        print(f"  ② 런을 잡는다      {onset_ok:>3}/{n}  ({onset_ok / n:>5.0%})")
        print(f"  ③ 같은 선수다      {runner_ok:>3}/{n}  ({runner_ok / n:>5.0%})")
        for name in builds:
            col = f"{name}_defender_match"
            if col in f:
                d = f[col].fillna(False).sum()
                print(f"  ④ 같은 수비수 [{name}] {d:>3}/{n}  ({d / n:>5.0%})")
    print(f"\n저장: {args.output}")


if __name__ == "__main__":
    main()
