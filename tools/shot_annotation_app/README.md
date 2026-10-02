# Manual shot annotation

Browse shot clips, select one or more runners, defenders, and beneficiaries,
choose strong / medium / low / ignore, add comments, and use Save & Next.
Saved annotations are restored from XLSX after restart; saving the same `clip_id`
updates its existing row.

```sh
cd <repo>
export OFFBALL_ANNOTATION_MEDIA_DIR=~/idsse_shots/output
./tools/shot_annotation_app/start.sh
```

Browser: http://127.0.0.1:8787

Requires Python 3.11 or newer. The launcher creates `<repo>/.venv` if needed and
installs the app's requirements there.

- `OFFBALL_ANNOTATION_MEDIA_DIR`: local shot clips directory. Supports the original
  `output/ALL_SHOT_CLIPS/*.mp4` or `output/<match_id>/clips/*.mp4` layout, or a
  directory directly containing `J03…_shot_….mp4` files. Optional metadata comes
  from `shot_metadata.csv` beside collected clips or `ALL_7_MATCHES_SHOTS.csv` in
  the output directory. Defaults to `~/idsse_shots/output` when available; otherwise
  startup prints instructions for setting it.
- `OFFBALL_ANNOTATION_XLSX`: defaults to
  `<repo>/annotations/shot_annotations.xlsx`, the canonical tracked annotation file.
  Set this to another path to work on a separate copy. CSV is a generated export
  beside the XLSX and is refreshed on save or CSV download.
- Optional tracking rings use cached data in `<media_dir>/tracking_overlay_cache`
  or Sportec XML in `idsse-data` beside the original output directory. Set
  `IDSSE_ROOT` to the directory containing `idsse-data` if it is elsewhere.

Paths accept `~`; relative environment paths are resolved from the app directory
when using `start.sh`. Raw video clips, tracking caches, and generated CSV exports
are local files and should not be committed.
