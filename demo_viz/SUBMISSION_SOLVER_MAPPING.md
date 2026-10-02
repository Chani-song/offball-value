# Solver mapping for the submission showcase

Development notes, not a public page. For every solver-derived showcase entry:
what it is, what it would need, and whether the public demo may switch
**Solver solution** on. Regenerate the underlying metadata with
`python -m demo_viz.ingest_showcase`.

## The rule

A scene shows solver output only when a real, compatible `mit_ssac2027`
artifact exists for it. There is no fallback. The three states are kept
distinct in the interface:

| state | when | what the panel says |
| --- | --- | --- |
| available | a real artifact is loaded | the solved quantities |
| unavailable | no artifact for a scene we can otherwise show | *Not computed for this scene* |
| unresolved | the scene itself is not identified in published data | *Scene data not published* |

## The five solver-derived entries

| Showcase | Rating | Scenario | Match | Runner | Defender | Beneficiary | Underlying scene | Artifact | Public solver layer |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S46 | 5/5 | 2v1 | Fortuna Düsseldorf vs 1. FC Nürnberg | #8 M. Karbownik | #17 J. Castrop | #23 Shinta Appelkamp | unresolved | none | **off** |
| S48 | 5/4 | 2v1 | Fortuna Düsseldorf vs F.C. Hansa Rostock | #25 Matthias Zimmermann | #21 Anderson Lucoqui | #11 F. Klaus | unresolved | none | **off** |
| S53 | 4/5 | 2v1 | Fortuna Düsseldorf vs SSV Jahn Regensburg | #11 F. Klaus | #20 L. Guwara | #25 Matthias Zimmermann | unresolved | none | **off** |
| S58 | 5/4 | 3v1 | Fortuna Düsseldorf vs 1. FC Kaiserslautern | #9 Dawid Kownacki | #32 R. Bormuth | #28 R. Hennings | unresolved | none | **off** |
| S66 | 5/4 | 3v1 | Fortuna Düsseldorf vs 1. FC Kaiserslautern | #28 R. Hennings | #32 R. Bormuth | #31 M. Sobottka | unresolved | none | **off** |

## Why none can be enabled

**No underlying scene.** All five are run-onset-centred states from the
stage-3 pipeline, not shot annotations, so none of them is among the 45 scenes
the demo publishes. Mapping evidence (team pair, half, hand-labelled roles)
finds no repository scene for any of them, and copying the tracking embedded in
the curated source would republish licensed data. They are therefore listed in
the showcase and disabled, with the reason on the entry.

**No artifact.** The solved policies for these states live on Delta
(`/work/hdd/bbmr/kseo1/offball-out`), which is not reachable from this machine.
A bounded search of the locations the docs and configs name found no
stage-3, `passer2on1` or `fixedpasser` output locally. The only real solver
artifacts present are `mit_ssac2027_ref/results/exact_100`, which are the
solver's own declared 2v1 study states — **not** these Bundesliga scenes, and
never attached to one.

The split metrics the curated source carries for each entry are recorded in
`submission_showcase.json` under `solver.split_metrics`; they describe the
pipeline run that selected the scene and are not a solver result the demo can
draw.

## To enable one

Three things have to line up, in this order:

1. the underlying stage-3 state is published as demo scene data, or the scene
   is otherwise identified in `web_data/`, giving `mapping.status = verified`;
2. the corresponding solved artifact is available locally as a run directory
   (`manifest.json`, `rows.jsonl`, `policies/`);
3. `solver.artifact` is set to `"<run directory>#<state index>"` and
   `solver.status` to `available`.

Then `python -m demo_viz.web.export_solver --from-manifest` writes the overlay
and the layer turns on for that scene. The adapter, the renderer and the tests
are already in place and are exercised against real artifacts today; only the
data is missing.
