#!/usr/bin/env python3
"""Which competitions does the 360 training set actually come from?

Every comparison between our pass cells and StatsBomb's completion rates
assumes the two populations are comparable. Our scenes are 2. Bundesliga; the
360 matches have never been checked, and a rate estimated on a different level
of football is a different yardstick.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path("data/raw/statsbomb-open-data/data")


def main() -> None:
    matches = {}
    for path in sorted((ROOT / "matches").rglob("*.json")):
        for match in json.loads(path.read_text(encoding="utf-8")):
            matches[str(match["match_id"])] = match
    have360 = {p.stem for p in (ROOT / "three-sixty").glob("*.json")}
    tally = Counter()
    seasons = Counter()
    for match_id in have360:
        match = matches.get(match_id)
        if match is None:
            tally["(매치 메타 없음)"] += 1
            continue
        competition = match.get("competition", {}).get("competition_name", "?")
        country = match.get("competition", {}).get("country_name", "?")
        season = match.get("season", {}).get("season_name", "?")
        tally[f"{country} · {competition}"] += 1
        seasons[f"{country} · {competition} · {season}"] += 1

    total = sum(tally.values())
    print(f"360 프레임이 있는 경기 {len(have360):,}개\n")
    print(f"  {'대회':<46}{'경기':>7}{'비중':>9}")
    for name, count in tally.most_common():
        print(f"  {name:<44}{count:>7}{count/total:>9.1%}")
    print(f"\n  시즌까지 (상위 12개)")
    for name, count in seasons.most_common(12):
        print(f"    {name:<58}{count:>5}")
    print("\n  우리 장면은 2. Bundesliga 다. 위 구성과 다르면 성공률 기준선도 다르다.")


if __name__ == "__main__":
    main()
