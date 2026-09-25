#!/usr/bin/env python3
"""StatsBomb 360 open-play passes with velocities and leak-free targets (B1-SB).

B1-SB fits 준현's velocity logistic on StatsBomb instead of on our own six
matches. It has far more passes, and the data are independent of every scene
the solver analyses, so no scene is priced by a model that saw its own match.
Rows come out in the same format as scripts/build_pass_dataset.py, so the
same feature code scores both.

  competitions  men's club football only (1. Bundesliga, La Liga, Ligue 1,
                MLS: 133 matches with 360 frames). Our scenes are men's club
                football; international tournaments and women's competitions
                are left out.
  label         1 = completed (no outcome), 0 = "Incomplete". "Out", "Pass
                Offside", "Unknown" and "Injury Clearance" are dropped, so the
                label means what ours means -- who controls the ball first --
                and offside stays the game's own legality rule.
  velocities    from the previous freeze frame within 2 s, players re-matched
                greedily by nearest position within side (the only identity a
                freeze frame gives), as scripts/train_xpass_360_kinematic.py.
                ONE CORRECTION to that script. Each freeze frame is written
                from the perspective of the team performing its event, so when
                the previous event belongs to the other team its coordinates
                are mirrored (120 - x, 80 - y) and its teammate flags inverted.
                Measured 2026-09-25 on 4 matches: across different-team pairs
                the matched-player distance is a median 55.0 units unflipped
                and 3.15 flipped, against 3.76 for same-team pairs. The old
                script did not flip, so about 30 % of its pairs carried
                nonsense velocities.
                A match implying more than 11 m/s (the project's
                observed_speed_cap_mps) is treated as no match. A row is kept
                only when the passer, the receiver and the defenders nearest
                each of them all have a real velocity -- those are the only
                velocities B1's features read.
  target        leak-free, as observed_passes.infer_intended_target. Only the
                DIRECTION start -> end is used, never the end point's
                distance. The receiver is the team-mate closest to that ray
                (at least 3 m along it, at most 8 m off it), and the target is
                his projection onto it. A completed pass's end_location is the
                receiver's feet by definition; using it as the target is the
                leak that sank the 2026-09-18 model.
  coordinates   StatsBomb units -> metres (105/120, 68/80). The attack always
                runs toward +x, so attack_direction is +1.

Usage:
    python scripts/build_statsbomb_pass_dataset.py \
        --output data/processed/pass_models/statsbomb_passes.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLUB_MEN = ("1. Bundesliga", "La Liga", "Ligue 1", "Major League Soccer")
FAILED = "Incomplete"
SET_PIECES = {"Corner", "Free Kick", "Goal Kick", "Kick Off", "Throw-in"}
MAX_GAP_S = 2.0
SPEED_CAP_MPS = 11.0
MIN_ALONG_M, MAX_PERP_M = 3.0, 8.0
PX, PY = 105.0 / 120.0, 68.0 / 80.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, default=ROOT / "data/raw/statsbomb-open-data/data")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--limit", type=int, default=None, help="first N matches only (testing)")
    return p.parse_args()


def club_matches(root: Path) -> list[tuple[str, str]]:
    out = []
    for f in sorted((root / "matches").rglob("*.json")):
        for m in json.loads(f.read_text(encoding="utf-8")):
            name = m["competition"]["competition_name"]
            mid = str(m["match_id"])
            if name in CLUB_MEN and (root / "three-sixty" / f"{mid}.json").exists():
                out.append((mid, name))
    return sorted(set(out))


def seconds(e) -> float:
    h, m, s = (e.get("timestamp") or "0:0:0").split(":")
    return int(e.get("period", 1)) * 10000 + int(h) * 3600 + int(m) * 60 + float(s)


def metres(xy):
    return (float(xy[0]) * PX, float(xy[1]) * PY)


def with_velocities(now, past, gap_s, flip):
    """[(x, y, vx, vy or None, teammate, actor, keeper)] for the current frame."""
    pools = {True: [], False: []}
    for p in past:
        x, y = float(p["location"][0]), float(p["location"][1])
        side = bool(p.get("teammate"))
        if flip:
            x, y, side = 120.0 - x, 80.0 - y, not side
        pools[side].append(metres((x, y)))
    out = []
    for p in now:
        side = bool(p.get("teammate"))
        here = metres(p["location"])
        pool, vel = pools[side], None
        if pool:
            j = min(range(len(pool)), key=lambda k: math.hypot(pool[k][0] - here[0], pool[k][1] - here[1]))
            was = pool[j]
            dist = math.hypot(was[0] - here[0], was[1] - here[1])
            if dist <= SPEED_CAP_MPS * gap_s:
                pool.pop(j)
                vel = ((here[0] - was[0]) / gap_s, (here[1] - was[1]) / gap_s)
        out.append((here[0], here[1], vel, side, bool(p.get("actor")), bool(p.get("keeper"))))
    return out


def nearest(points, anchor):
    return min(range(len(points)), key=lambda k: math.hypot(points[k][0] - anchor[0], points[k][1] - anchor[1]))


def collect(args):
    mid, competition, root = args
    try:
        ff = {str(r["event_uuid"]): r["freeze_frame"]
              for r in json.loads((root / "three-sixty" / f"{mid}.json").read_text(encoding="utf-8"))
              if r.get("freeze_frame")}
        events = json.loads((root / "events" / f"{mid}.json").read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError, FileNotFoundError):
        return mid, [], {"unreadable": 1}
    seq = sorted((e for e in events if str(e.get("id")) in ff), key=seconds)
    rows, why = [], {}

    def skip(reason):
        why[reason] = why.get(reason, 0) + 1

    for i, e in enumerate(seq):
        if e.get("type", {}).get("name") != "Pass":
            continue
        pd_ = e.get("pass", {})
        if (pd_.get("type") or {}).get("name") in SET_PIECES:
            continue
        if not e.get("location") or not pd_.get("end_location"):
            skip("no coordinates")
            continue
        outcome = (pd_.get("outcome") or {}).get("name")
        if outcome is not None and outcome != FAILED:
            skip(f"outcome {outcome}")
            continue
        if i == 0:
            skip("no previous frame")
            continue
        prev = seq[i - 1]
        gap = seconds(e) - seconds(prev)
        if not (0.05 < gap <= MAX_GAP_S):
            skip("previous frame too far")
            continue
        flip = str(prev["team"]["id"]) != str(e["team"]["id"])
        bodies = with_velocities(ff[str(e["id"])], ff[str(prev["id"])], gap, flip)
        actor = [b for b in bodies if b[4]]
        if len(actor) != 1 or actor[0][2] is None:
            skip("passer missing or unmatched")
            continue
        passer = actor[0]
        start, end = metres(e["location"]), metres(pd_["end_location"])
        dx, dy = end[0] - start[0], end[1] - start[1]
        length = math.hypot(dx, dy)
        if length < 0.5:
            skip("no heading")
            continue
        ux, uy = dx / length, dy / length
        best = None
        for b in bodies:
            if not b[3] or b[4]:
                continue
            rx, ry = b[0] - start[0], b[1] - start[1]
            along = rx * ux + ry * uy
            perp = abs(-uy * rx + ux * ry)
            if along >= MIN_ALONG_M and perp <= MAX_PERP_M and (best is None or perp < best[1]):
                best = (b, perp, along)
        if best is None:
            skip("no intended receiver")
            continue
        receiver, _, along = best
        if receiver[2] is None:
            skip("receiver unmatched")
            continue
        opps = [b for b in bodies if not b[3]]
        if not opps:
            skip("no opponent in frame")
            continue
        pts = [(o[0], o[1]) for o in opps]
        if (opps[nearest(pts, (passer[0], passer[1]))][2] is None
                or opps[nearest(pts, (receiver[0], receiver[1]))][2] is None):
            skip("nearest defender unmatched")
            continue
        target = (start[0] + ux * along, start[1] + uy * along)
        rows.append({
            "match_id": mid, "competition": competition, "label": 0 if outcome else 1,
            "attack_direction": 1, "flipped_pair": flip, "gap_s": round(gap, 3),
            "passer": [round(passer[0], 3), round(passer[1], 3), round(passer[2][0], 3), round(passer[2][1], 3)],
            "receiver": [round(receiver[0], 3), round(receiver[1], 3),
                         round(receiver[2][0], 3), round(receiver[2][1], 3)],
            "target": [round(target[0], 3), round(target[1], 3)],
            "opponents": [[round(o[0], 3), round(o[1], 3),
                           round(o[2][0], 3) if o[2] else 0.0, round(o[2][1], 3) if o[2] else 0.0,
                           int(o[5])] for o in opps],
            "pass_distance": round(math.hypot(target[0] - passer[0], target[1] - passer[1]), 3),
        })
    return mid, rows, why


def main() -> None:
    args = parse_args()
    matches = club_matches(args.data_root)
    if args.limit:
        matches = matches[:args.limit]
    print(f"클럽 남자 경기 {len(matches)}개", flush=True)
    rows, why = [], {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for mid, got, w in pool.map(collect, [(m, c, args.data_root) for m, c in matches]):
            rows.extend(got)
            for k, v in w.items():
                why[k] = why.get(k, 0) + v
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(r) + "\n" for r in rows))
    lab = sum(r["label"] for r in rows)
    flipped = sum(r["flipped_pair"] for r in rows)
    print(f"패스 {len(rows):,} · 성공률 {lab / max(len(rows), 1):.4f} · 뒤집은 쌍 {flipped:,}", flush=True)
    print("제외:", dict(sorted(why.items(), key=lambda kv: -kv[1])), flush=True)
    print(f"→ {args.output}", flush=True)


if __name__ == "__main__":
    main()
