# Data Inventory

Last local inspection: 2026-05-17

This directory is the local data workspace for the off-ball value project. Raw
datasets are intentionally not tracked by Git because the local raw data is
large and mostly reproducible from public sources.

Current local size:

| Path | Size |
|---|---:|
| `data/raw/metrica-sample-data` | 233 MB |
| `data/raw/skillcorner-opendata` | 1.8 GB |
| `data/raw/statsbomb-open-data` | 13 GB |
| `data/raw/bundesliga-integrated` | 5.0 GB |
| `data/processed` | 3.1 MB |
| `data/raw` total | 20 GB |

## Summary

| Dataset | Local matches | Main data type | Current role in this project |
|---|---:|---|---|
| Metrica sample data | 3 | Tracking + events | Fast prototype and visual sanity checks |
| SkillCorner open data | 10 | Broadcast tracking + dynamic events + phases | Main open tracking candidate for off-ball movement experiments |
| StatsBomb open data + 360 | 3,464 event files, 326 360 files | Events + freeze frames | Event-context and freeze-frame validation |
| Bundesliga integrated / IDSSE | 7 | Tracking + events + metadata XML | Later benchmark-style integrated dataset |

## Local Layout

```text
data/
  README.md
  raw/
    metrica-sample-data/
    skillcorner-opendata/
    statsbomb-open-data/
    bundesliga-integrated/
  processed/
    demo_offball_counterfactual_game1.csv
    metrica_draft_offball_value.csv
    visualizations/
```

`data/raw/` and `data/processed/` are ignored by Git. Share large raw data via a
separate archive or cloud storage if needed.

## Metrica Sample Data

Source: `data/raw/metrica-sample-data`

Metrica is the smallest and easiest tracking dataset here. It is best for
loader development, first-pass movement extraction, and visual debugging. Game
1 and Game 2 use CSV tracking/event files. Game 3 uses a newer EPTS-style
tracking text file, metadata XML, and events JSON.

Local files:

| Game | Tracking rows | Event rows | Notes |
|---|---:|---:|---|
| Sample Game 1 | 145,006 per team file | 1,745 | Used for current toy baseline |
| Sample Game 2 | 141,156 per team file | 1,935 | Same CSV structure as Game 1 |
| Sample Game 3 | 143,761 tracking text lines | JSON events | Different format, not yet wired into current loaders |

Current implemented use:

- `scripts/run_metrica_baseline.py`
- `scripts/demo_offball_counterfactual.py`
- `scripts/visualize_metrica_top_runs.py`
- `src/offball_value/loaders.py`
- `src/offball_value/adapters.py`

Current processed outputs from Metrica Game 1:

| File | Rows / files | Meaning |
|---|---:|---|
| `data/processed/metrica_draft_offball_value.csv` | 7,228 rows | Toy candidate movement scores |
| `data/processed/demo_offball_counterfactual_game1.csv` | 7,228 rows | Adapter-based toy counterfactual output |
| `data/processed/visualizations/metrica_top_runs/summary.csv` | 10 rows | Top-scene visualization summary |
| `data/processed/visualizations/metrica_top_runs/*.png` | 10 images | Top-ranked scene diagnostics |

Important limitation:

The current checked-in Metrica CSVs were produced by an earlier toy baseline.
They should still be treated cautiously because some top-ranked scenes may
involve loose-ball or ball-near-runner situations. The current code now includes
foot-control possession helpers and the demo script filters to controlled
attacking possessions by default.

## SkillCorner Open Data

Source: `data/raw/skillcorner-opendata`

SkillCorner is currently the strongest open tracking candidate for this project.
It contains broadcast-derived tracking, dynamic events, and phases of play for
10 matches.

Local structure:

```text
data/raw/skillcorner-opendata/data/
  matches.json
  matches/
    <match_id>/
      <match_id>_match.json
      <match_id>_tracking_extrapolated.jsonl
      <match_id>_dynamic_events.csv
      <match_id>_phases_of_play.csv
```

Local match-level counts:

| Match ID | Tracking frames | Dynamic events | Phases |
|---:|---:|---:|---:|
| 1886347 | 59,061 | 5,079 | 454 |
| 1899585 | 60,530 | 4,713 | 460 |
| 1925299 | 61,301 | 5,220 | 492 |
| 1953632 | 59,250 | 4,823 | 431 |
| 1996435 | 57,621 | 5,292 | 448 |
| 2006229 | 59,270 | 4,991 | 438 |
| 2011166 | 71,851 | 3,966 | 429 |
| 2013725 | 70,251 | 4,999 | 486 |
| 2015213 | 72,101 | 4,582 | 506 |
| 2017461 | 71,451 | 4,188 | 437 |
| **Total** | **642,687** | **47,853** | **4,581** |

Important fields observed:

- tracking JSONL: `frame`, `timestamp`, `period`, `ball_data`, `possession`,
  `player_data`
- dynamic events CSV: `event_type`, `player_id`, `team_id`,
  `player_in_possession_id`, `frame_start`, `frame_end`, `x_start`, `y_start`,
  `x_end`, `y_end`
- engineered context fields include passing options, off-ball run associations,
  line breaks, xThreat-like values, pass quality, defensive structure, and
  possession danger fields

Project interpretation:

SkillCorner should be used carefully. It is very useful because it already has
possession and off-ball-run-related fields, but many fields are engineered
labels or downstream annotations. For model inputs, avoid leakage from fields
that directly encode the target outcome. Use those richer fields first for
validation and qualitative interpretation.

For the project state definition, SkillCorner possession should be converted
into foot-control states with `possession.player_id` and tracking distance. Pass
success can be labeled after the receiver's next controlled touch, while
pass-flight frames remain outside the state set.

## StatsBomb Open Data + 360

Source: `data/raw/statsbomb-open-data`

StatsBomb is event data, not continuous tracking. It is useful for validating
event-level context and checking freeze-frame option structure around selected
events. It is not the primary source for continuous off-ball movement modeling.

Local structure:

```text
data/raw/statsbomb-open-data/data/
  competitions.json
  matches/
  events/
  lineups/
  three-sixty/
```

Local counts:

| Item | Count |
|---|---:|
| competition-season rows in `competitions.json` | 75 |
| match JSON files under `matches/` | 75 |
| total matches / event files | 3,464 |
| lineup files | 3,464 |
| 360 files | 326 |

Largest local competitions by match count:

| Competition | Match count |
|---|---:|
| La Liga | 868 |
| Ligue 1 | 435 |
| Premier League | 418 |
| Serie A | 381 |
| 1. Bundesliga | 340 |
| FA Women's Super League | 326 |
| FIFA World Cup | 147 |
| Women's World Cup | 116 |
| Indian Super League | 115 |
| UEFA Euro | 102 |

Typical event fields:

- `id`, `index`, `period`, `timestamp`, `minute`, `second`
- `type`, `team`, `player`, `possession`, `possession_team`
- event-specific payloads such as `pass`, `carry`, `shot`
- `location` and event-specific end locations

Typical 360 fields:

- `event_uuid`
- `visible_area`
- `freeze_frame`

Project interpretation:

StatsBomb can support event-conditioned checks such as:

- whether a high-value off-ball context coincides with a pass, shot, or carry
- whether freeze-frame geometry agrees with our option/state-value logic
- whether later state-value definitions align with known event outcomes

## Bundesliga Integrated Dataset / IDSSE

Source: `data/raw/bundesliga-integrated`

This is the IDSSE / Sportec open tracking and event dataset, mirrored from
Figshare via Hugging Face. The local README reports a CC-BY-4.0 license and
describes 7 complete Bundesliga 2022/23 matches with TRACAB optical tracking.

Local match IDs:

```text
J03WMX
J03WN1
J03WOH
J03WOY
J03WPY
J03WQQ
J03WR9
```

Each local match has three XML files:

| Prefix | Meaning | Approx local size |
|---|---|---:|
| `DFL_02_01_matchinformation_...xml` | Match metadata, teams, players, pitch, venue | 12 KB |
| `DFL_03_02_events_raw_...xml` | Match events | 600-760 KB |
| `DFL_04_03_positions_raw_observed_...xml` | Player/ball/referee positions | 333-399 MB |

Local event counts by match, counted as top-level `<Event ...>` elements:

| Match ID | Event count |
|---|---:|
| J03WMX | 1,715 |
| J03WN1 | 1,363 |
| J03WOH | 1,389 |
| J03WOY | 1,489 |
| J03WPY | 1,504 |
| J03WQQ | 1,586 |
| J03WR9 | 1,452 |
| **Total** | **10,498** |

Observed XML structure:

- match information XML includes competition, teams, players, lineups, venue,
  pitch dimensions, kickoff time, and match result
- event XML contains top-level `Event` elements with location and timestamp,
  and nested action types such as `KickOff`, `Play`, `Pass`, `ThrowIn`,
  `TacklingGame`, and other ball actions
- position XML contains `Positions`, `MetaData`, and per-entity `FrameSet`
  blocks; each `Frame` includes fields such as `N`, `T`, `X`, `Y`, `D`, `S`,
  `A`, and `M`
- local `FrameSet` counts range from 53 to 62 per match, representing tracked
  entities such as players, ball, and officials

Project interpretation:

Bundesliga is the best candidate for a later benchmark-style experiment because
it combines event and position data in a synchronized official format. It is not
yet wired into the project loaders. The likely next step is to add a Sportec /
IDSSE XML parser or use `kloppy` as an intermediate loader.

## Current Research Usefulness

For the off-ball value project, the datasets fit different stages:

1. **Metrica**: smallest loop for debugging movement detection, actual-vs-
   counterfactual visualizations, and baseline sanity checks.
2. **SkillCorner**: most useful open source for realistic off-ball tracking
   experiments because it has possession, phases, dynamic events, and engineered
   off-ball-run metadata.
3. **StatsBomb + 360**: useful for event-aligned validation and freeze-frame
   option structure, but not for continuous movement trajectories.
4. **Bundesliga / IDSSE**: strongest benchmark candidate after loaders mature,
   because it has synchronized events and 25 fps position data across 7 full
   matches.

## Data Handling Notes

- Do not commit raw datasets to Git.
- `data/raw/` and `data/processed/` are ignored in `.gitignore`.
- Use download scripts or external archives to share raw data.
- If sharing with collaborators, a practical compromise is:
  - share `data/processed/` for quick result review;
  - share `data/raw/metrica-sample-data/` for exact Metrica reproducibility;
  - ask collaborators to run the download scripts for SkillCorner and StatsBomb;
  - share Bundesliga only through the original Figshare/Hugging Face source or a
    private cloud archive if necessary.

## Recreate Local Downloads

```bash
.venv/bin/python scripts/download_metrica.py
.venv/bin/python scripts/download_skillcorner.py
.venv/bin/python scripts/download_statsbomb.py
git lfs install
git clone https://huggingface.co/datasets/pysport/idsse-data data/raw/bundesliga-integrated
```

Bundesliga can also be downloaded from the original Figshare DOI listed in the
dataset README.
