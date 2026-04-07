# Open datasets for this project

## 1. Metrica Sports sample data
- URL: https://github.com/metrica-sports/sample-data
- Type: synchronized event + tracking
- Best for: quick prototyping, parser development, visualization
- Caveat: tiny sample

## 2. SkillCorner open data
- URL: https://github.com/SkillCorner/opendata
- Type: broadcast tracking + dynamic events + phases of play
- Best for: actual off-ball movement prototypes on open tracking data
- Caveat: only 10 matches, repository structure may evolve

## 3. StatsBomb open data
- URL: https://github.com/statsbomb/open-data
- Type: event data, lineups, selected 360 freeze frames
- Best for: event-level contextual checks, option maps at decision moments
- Caveat: not continuous full tracking

## 4. Bundesliga integrated event + position dataset
- Paper: https://www.nature.com/articles/s41597-025-04505-y
- Dataset landing page: https://springernature.figshare.com/articles/dataset/An_integrated_dataset_of_spatiotemporal_and_event_data_in_elite_soccer/28196177
- Type: official integrated event + position data
- Best for: benchmark-style experiments and reproducibility
- Caveat: download flow is less convenient than the GitHub datasets above

## Suggested practical order

1. Metrica
2. SkillCorner
3. StatsBomb 360
4. Bundesliga integrated dataset
