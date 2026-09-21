# Datasets

## Primary dataset: IDSSE Bundesliga

The current research pipeline uses IDSSE, an integrated dataset of seven full
matches from the 2022/23 Bundesliga and 2. Bundesliga.

- Tracking: every player and the ball at 25 Hz
- Events: synchronized match events
- Metadata: teams, players, roles, and pitch information
- Dataset DOI: <https://doi.org/10.6084/m9.figshare.28196177>
- Paper DOI: <https://doi.org/10.1038/s41597-025-04505-y>
- License: CC BY 4.0

IDSSE is used because the project needs continuous 22-player motion to detect
off-ball onset, preserve the 11-v-11 background, and generate counterfactual
defender responses.

## Secondary datasets retained locally

The repository also contains loader support or local copies for other open
sources. They are not used in the current v0.1 local-game audit.

### Metrica Sports sample data

- Type: synchronized events and tracking
- Use: parser and visualization prototyping
- Limitation: very small sample

### SkillCorner open data

- Type: broadcast tracking, dynamic events, and phases of play
- Use: possible external method checks
- Limitation: provider format and visibility differ from IDSSE

### StatsBomb open data and 360

- Type: events and selected freeze frames
- Use: event-level or static option-map validation
- Limitation: not continuous full tracking for all match states

## Repository policy

Raw datasets are stored below `data/raw/` and ignored by Git. Small derived
human-review payloads may be committed only when they are needed to audit the
method, remain within the source license, and include attribution.
