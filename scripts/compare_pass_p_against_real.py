#!/usr/bin/env python3
"""Which P is plausible for our counterfactual passes -- 0.625 or 0.971?

Swapping the delivery model moves the median P on genuine pass cells from
0.625 to 0.971. There is no ground truth for those cells: they are passes the
option catalogue invented and nobody played. But there is a bound.

For each pass-length band, StatsBomb says what fraction of REAL passes of that
length completed. Real passes are SELECTED -- a player judged them worth
attempting -- while our catalogue enumerates every candidate including the ones
nobody would try. So the real rate is an UPPER bound on what our cells should
score, band by band. A model that sits above it is claiming our invented passes
beat the ones professionals chose to play.

Usage:
    python scripts/compare_pass_p_against_real.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from offball_value.xpass import (
    event_features_and_label,
    is_open_play_pass,
)

PASS_TYPES = {"through_ball_to_space", "receive_to_feet", "cutback_to_space"}
BANDS = [(0, 10), (10, 15), (15, 20), (20, 25), (25, 30), (30, 40), (40, 200)]
# StatsBomb pitch units are 120 x 80 over a 105 m x 68 m pitch, so the two axes
# scale differently. An earlier version applied the x scale to the euclidean
# distance, overstating lateral passes by up to 2.9 %.
X_SCALE_M = 105.0 / 120.0
Y_SCALE_M = 68.0 / 80.0


def collect_ours(directory: Path):
    out = []
    for path in sorted(directory.glob("scene_*/local_game_payoff_audits.json")):
        for scene in json.loads(path.read_text(encoding="utf-8")):
            for defender in scene["candidate_defenders"]:
                for response in defender["responses"]:
                    for cell in (response.get("cells") or {}).values():
                        if cell.get("legal") is False or cell.get("q") is None:
                            continue
                        if str(cell.get("continuation_type")) not in PASS_TYPES:
                            continue
                        components = cell.get("delivery_components") or {}
                        distance = components.get("pass_distance_m")
                        if distance is None:
                            continue
                        out.append((float(distance), float(cell["delivery"])))
    return np.array(out) if out else np.zeros((0, 2))


def main() -> None:
    root = Path("/work/hdd/bbmr/kseo1/offball-out")
    hybrid = collect_ours(root / "control")
    x360 = collect_ours(Path("out"))
    kin = collect_ours(root / "x360kin_main")

    events_dir = Path("data/raw/statsbomb-open-data/data/events")
    frames_dir = Path("data/raw/statsbomb-open-data/data/three-sixty")
    lengths, labels = [], []
    for frames_path in sorted(frames_dir.glob("*.json")):
        events_path = events_dir / frames_path.name
        if not events_path.exists():
            continue
        try:
            events = json.loads(events_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        for event in events:
            if not is_open_play_pass(event):
                continue
            base, label = event_features_and_label(event)
            start = event["location"][:2]
            end = event["pass"]["end_location"][:2]
            lengths.append(float(np.hypot(
                (end[0] - start[0]) * X_SCALE_M,
                (end[1] - start[1]) * Y_SCALE_M,
            )))
            labels.append(label)
    lengths = np.asarray(lengths)
    labels = np.asarray(labels, dtype=float)

    print("=" * 96)
    print("패스 길이별: 실제 성공률(상한) vs 우리 셀에 매겨진 P")
    print("=" * 96)
    print(f"  실제 패스 {len(lengths):,}개 · 우리 패스 셀 하이브리드 {len(hybrid):,} "
          f"· xpass360 {len(x360):,} · x360kin {len(kin):,}\n")
    print(f"  {'길이(m)':<12}{'실제 n':>9}{'실제 성공률':>12}"
          f"{'하이브리드':>11}{'xpass360':>11}{'x360kin':>11}")
    for lo, hi in BANDS:
        m = (lengths >= lo) & (lengths < hi)
        if m.sum() < 200:
            continue
        real = float(labels[m].mean())
        row = f"  [{lo}, {hi})".ljust(12) + f"{m.sum():>9,}{real:>12.3f}"
        for arr in (hybrid, x360, kin):
            if arr.size:
                k = (arr[:, 0] >= lo) & (arr[:, 0] < hi)
                row += f"{np.median(arr[k, 1]):>11.3f}" if k.sum() >= 30 else f"{'-':>11}"
            else:
                row += f"{'-':>11}"
        print(row)

    print("\n  '실제 성공률' 은 상한이다: 실제 패스는 선수가 시도할 만하다고 판단한 것들이고,")
    print("  우리 셀은 아무도 시도하지 않을 후보까지 전부 세어놓은 것이므로.")
    print("  이 선을 넘는 모델은 '우리가 지어낸 패스가 프로가 고른 패스보다 낫다'고 말하는 셈.")


if __name__ == "__main__":
    main()
