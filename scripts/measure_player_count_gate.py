#!/usr/bin/env python3
"""How much of each match does RunOnsetConfig.required_player_count=22 discard?

detect_run_onsets rejects any frame whose player count is not exactly 22
(run_onset.py:566). DFL-MAT-J03WN1 has a ~7-minute red card and yields 2 run
onsets where peer matches yield 52-85, which points straight at this gate. But
a red card is not the only way to drop below 22: substitutions, a player off
the pitch for treatment, and momentary tracking dropouts do it too, and every
such frame is silently invisible to onset detection in EVERY match.

This measures the real distribution instead of assuming. Read-only.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from offball_value.bundesliga import (
    FPS,
    find_bundesliga_files,
    list_bundesliga_match_ids,
    load_bundesliga_frame_clock,
    load_bundesliga_frames,
)

DATA = "data/raw/bundesliga-integrated"
STRIDE = 25          # 1 Hz sample: enough to characterise minutes-long stretches
MAX_FRAME = 400_000


def main() -> None:
    print(f"{'match':16} {'표본':>7} {'22명':>8} {'21명':>8} {'기타':>8}   22명 비율")
    rows = []
    for match_id in list_bundesliga_match_ids(DATA):
        files = find_bundesliga_files(DATA, match_id)
        clock = load_bundesliga_frame_clock(files["positions"])
        start = min(v[0] for v in clock.section_start.values())
        ids = range(start, MAX_FRAME, STRIDE)
        frames = load_bundesliga_frames(files["positions"], ids)
        counts = Counter(len(f.players) for f in frames.values() if f is not None)
        total = sum(counts.values())
        if not total:
            print(f"{match_id:16} 프레임 없음")
            continue
        n22 = counts.get(22, 0)
        n21 = counts.get(21, 0)
        other = total - n22 - n21
        print(f"{match_id:16} {total:>7} {n22:>8} {n21:>8} {other:>8}   {n22 / total:>8.1%}")
        rows.append((match_id, total, n22, n21, other, counts))
        del frames

    print("\n경기별 선수 수 분포 (상위 6개):")
    for match_id, total, *_rest, counts in rows:
        top = ", ".join(f"{k}명:{v}" for k, v in counts.most_common(6))
        print(f"  {match_id:16} {top}")

    print("\n요약")
    good = [n22 / total for _m, total, n22, *_ in rows]
    print(f"  22명 비율  중앙값 {np.median(good):.1%}  최소 {min(good):.1%}  최대 {max(good):.1%}")
    lost = [(1 - g) * 100 for g in good]
    print(f"  게이트로 버려지는 비율  중앙값 {np.median(lost):.1f}%  최대 {max(lost):.1f}%")
    print(f"\n  (표본은 {STRIDE / FPS:.1f}초 간격, 킥오프 프레임부터)")


if __name__ == "__main__":
    main()
