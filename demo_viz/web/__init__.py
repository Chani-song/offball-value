"""Static browser build of the explorer.

``demo_viz/app`` is the local Dash explorer. This package produces the
dependency-free version that GitHub Pages serves: a data exporter
(:mod:`demo_viz.web.export_data`) and the static site under ``web/site``.

The browser re-implements the residual-space arithmetic rather than being fed
pre-rendered fields, so a visitor can pick any runner, defender and beneficiary
and get the same numbers Python would produce.
``demo_viz/web/validate.py`` checks that claim against the Python implementation.
"""
