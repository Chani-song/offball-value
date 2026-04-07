# Bundesliga integrated dataset setup

This dataset is public, but its download flow is less convenient than the GitHub datasets.

Use the official paper and figshare landing page:

- https://www.nature.com/articles/s41597-025-04505-y
- https://springernature.figshare.com/articles/dataset/An_integrated_dataset_of_spatiotemporal_and_event_data_in_elite_soccer/28196177

Suggested local placement after manual download/unzip:

```text
data/raw/bundesliga-integrated/
```

Then add a parser later in `src/offball_value/loaders.py`.
