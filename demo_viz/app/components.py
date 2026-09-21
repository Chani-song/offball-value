"""Layout for the interactive explorer. Short labels, dark surfaces."""

from __future__ import annotations

from dash import dcc, html

from .. import palette
from ..core.figures import DEFAULT_LAYERS
from .state import scene_options

LAYER_OPTIONS = [
    {"label": "Trail", "value": "trail"},
    {"label": "Tether", "value": "tether"},
    {"label": "Wake", "value": "wake"},
    {"label": "Lane", "value": "lane"},
    {"label": "Ghost", "value": "ghost"},
    {"label": "Labels", "value": "labels"},
    {"label": "Paths", "value": "paths"},
    {"label": "Hints", "value": "candidates"},
]

MODE_OPTIONS = [
    {"label": "Manual", "value": "manual"},
    {"label": "Annotation", "value": "annotation"},
    {"label": "Pipeline", "value": "pipeline"},
]

EFFECT_OPTIONS = [
    {"label": "Strong", "value": "strong"},
    {"label": "Medium", "value": "medium"},
]

ROLE_SLOTS = (
    ("runner", "Runner", palette.RUNNER),
    ("beneficiary", "Beneficiary", palette.BENEFICIARY),
    ("defender", "Defender", palette.DEFENDER),
)

CARD = {
    "background": palette.PANEL,
    "border": f"1px solid {palette.GRID}",
    "borderRadius": "10px",
    "padding": "12px 14px",
}

LABEL = {
    "color": palette.TEXT_MUTED,
    "fontSize": "10px",
    "letterSpacing": "0.09em",
    "textTransform": "uppercase",
    "fontWeight": 700,
    "marginBottom": "6px",
}

BUTTON = {
    "background": palette.PANEL,
    "color": palette.TEXT_PRIMARY,
    "border": f"1px solid {palette.GRID}",
    "borderRadius": "7px",
    "padding": "7px 12px",
    "fontSize": "12px",
    "cursor": "pointer",
    "fontWeight": 600,
}


def _button(label: str, identifier: str, accent: str | None = None):
    style = dict(BUTTON)
    if accent:
        style["borderColor"] = accent
        style["color"] = accent
    return html.Button(label, id=identifier, n_clicks=0, style=style)


#: Initial dropdown/mode values; the CLI overrides these before the app starts.
DEFAULTS = {"scene": None, "mode": "annotation"}


def layout():
    options = scene_options()
    first = DEFAULTS.get("scene") or (options[0]["value"] if options else "synthetic")
    if first not in {o["value"] for o in options}:
        first = options[0]["value"] if options else "synthetic"
    return html.Div(
        style={
            "background": palette.INK,
            "height": "100vh",
            "overflow": "hidden",
            "color": palette.TEXT_PRIMARY,
            "fontFamily": "Inter, Helvetica, Arial, sans-serif",
            "padding": "12px 16px 14px 16px",
            "boxSizing": "border-box",
        },
        children=[
            dcc.Store(id="store-selection", data={}),
            dcc.Store(id="store-scene", data={"scene_id": first}),
            dcc.Interval(id="tick", interval=80, disabled=True),
            dcc.Download(id="download"),
            _top_bar(options, first),
            source_sheet(),
            html.Div(
                style={"display": "flex", "gap": "12px", "marginTop": "10px",
                       "alignItems": "stretch", "height": "calc(100vh - 106px)"},
                children=[
                    html.Div(style={"flex": "1 1 auto", "minWidth": 0},
                             children=[_pitch_card()]),
                    html.Div(style={"flex": "0 0 336px", "minWidth": 0},
                             children=[_side_panel()]),
                ],
            ),
        ],
    )


def _top_bar(options, first):
    return html.Div(
        style={**CARD, "display": "flex", "gap": "18px", "alignItems": "flex-end",
               "flexWrap": "wrap"},
        children=[
            html.Div(style={"flex": "1 1 320px", "minWidth": "240px"}, children=[
                html.Div("Scene", style=LABEL),
                dcc.Dropdown(
                    id="scene", options=options, value=first, clearable=False,
                    searchable=True,
                ),
            ]),
            html.Div(children=[
                html.Div("Mode", style=LABEL),
                dcc.RadioItems(
                    id="mode", options=MODE_OPTIONS, value=DEFAULTS.get("mode", "annotation"),
                    inline=True, className="chips",
                    inputStyle={"marginRight": "5px"},
                    labelStyle={"marginRight": "12px", "fontSize": "12px"},
                ),
            ]),
            html.Div(children=[
                html.Div("Effect", style=LABEL),
                dcc.Checklist(
                    id="effects", options=EFFECT_OPTIONS, value=["strong", "medium"],
                    inline=True, className="chips",
                    inputStyle={"marginRight": "5px"},
                    labelStyle={"marginRight": "12px", "fontSize": "12px"},
                ),
            ]),
            dcc.Store(id="pick", data="runner"),
            dcc.Store(id="hints-open", data=None),
            html.Div(style={"display": "flex", "gap": "8px"}, children=[
                _button("Auto triplet", "btn-auto", palette.BENEFICIARY),
                _button("Reset", "btn-reset"),
                _button("Source", "btn-source"),
            ]),
        ],
    )


def _pitch_card():
    return html.Div(style={**CARD, "padding": "10px 12px 12px 12px", "height": "100%",
                           "boxSizing": "border-box", "display": "flex",
                           "flexDirection": "column"}, children=[
        html.Div(
            style={"display": "flex", "alignItems": "center", "gap": "10px",
                   "marginBottom": "6px"},
            children=[
                html.Div(id="pitch-caption", style={"color": palette.TEXT_SECONDARY,
                                                    "fontSize": "12px",
                                                    "overflow": "hidden",
                                                    "textOverflow": "ellipsis",
                                                    "whiteSpace": "nowrap"}),
                html.Div(style={"flex": "1 1 auto"}),
                dcc.RadioItems(
                    id="view-mode",
                    options=[{"label": "Full pitch", "value": "full"},
                             {"label": "Focus", "value": "focus"}],
                    value="full", inline=True, className="chips segmented",
                    inputStyle={"marginRight": "5px"},
                    labelStyle={"marginRight": "10px", "fontSize": "11.5px"},
                ),
            ],
        ),
        dcc.Graph(
            id="pitch",
            config={"displayModeBar": False, "scrollZoom": False,
                    "doubleClick": False, "responsive": True},
            style={"height": "calc(100% - 118px)", "minHeight": "290px"},
        ),
        html.Div(
            style={"display": "flex", "alignItems": "center", "gap": "12px",
                   "marginTop": "6px"},
            children=[
                html.Button("▶", id="btn-play", n_clicks=0,
                            style={**BUTTON, "width": "44px", "fontSize": "14px"}),
                html.Button("Run", id="btn-jump-run", n_clicks=0,
                            style={**BUTTON, "padding": "6px 9px", "fontSize": "11px"}),
                html.Button("Reaction", id="btn-jump-reaction", n_clicks=0,
                            style={**BUTTON, "padding": "6px 9px", "fontSize": "11px"}),
                html.Button("Peak", id="btn-jump-peak", n_clicks=0,
                            style={**BUTTON, "padding": "6px 9px", "fontSize": "11px"}),
                html.Div(style={"flex": "1 1 auto"}, children=[
                    dcc.Slider(id="time", min=0, max=1, step=1, value=0,
                               marks={}, updatemode="drag", tooltip=None),
                ]),
                html.Div(id="time-readout", style={
                    "fontFamily": "ui-monospace, Menlo, monospace", "fontSize": "12px",
                    "color": palette.TEXT_SECONDARY, "minWidth": "74px",
                    "textAlign": "right"}),
            ],
        ),
        html.Div(
            style={"display": "flex", "gap": "16px", "alignItems": "center",
                   "marginTop": "10px", "flexWrap": "wrap"},
            children=[
                dcc.Checklist(
                    id="layers", options=LAYER_OPTIONS, value=list(DEFAULT_LAYERS),
                    inline=True, className="chips",
                    inputStyle={"marginRight": "5px"},
                    labelStyle={"marginRight": "14px", "fontSize": "12px"},
                ),
                html.Div(style={"flex": "1 1 auto"}),
                dcc.RadioItems(
                    id="wake-mode",
                    options=[{"label": "Opened space", "value": "gain"},
                             {"label": "Total space", "value": "space"}],
                    value="gain", inline=True, className="chips",
                    inputStyle={"marginRight": "5px"},
                    labelStyle={"marginRight": "12px", "fontSize": "12px"},
                ),
            ],
        ),
    ])


def _side_panel():
    return html.Div(
        style={"display": "flex", "flexDirection": "column", "gap": "10px",
               "height": "100%", "overflow": "hidden"},
        children=[
            _role_dock(),
            html.Div(style={**CARD, "padding": "10px 12px", "flex": "0 0 auto"},
                     children=[
                html.Div(id="stat-label", style=LABEL),
                html.Div(id="stat-value", style={"fontSize": "30px", "fontWeight": 700,
                                                 "lineHeight": "1.05"}),
                dcc.Graph(id="chart", config={"displayModeBar": False},
                          style={"height": "138px", "marginTop": "2px"}),
            ]),
            html.Div(id="hints-card",
                     style={**CARD, "flex": "1 1 auto", "overflowY": "auto",
                            "minHeight": "58px", "padding": "10px 12px"}, children=[
                html.Button(
                    id="hints-toggle", n_clicks=0,
                    style={"display": "flex", "alignItems": "center", "gap": "7px",
                           "width": "100%", "background": "none", "border": "0",
                           "padding": "0 0 2px 0", "cursor": "pointer",
                           "color": "inherit", "font": "inherit", "textAlign": "left"},
                    children=[
                        html.Span(id="hints-chev", style={"color": palette.TEXT_MUTED,
                                                          "fontSize": "10px",
                                                          "width": "9px"}),
                        html.Span(id="candidates-label", style=LABEL),
                        html.Span(id="hints-count",
                                  style={"color": palette.TEXT_MUTED, "fontSize": "11px",
                                         "marginLeft": "auto"}),
                    ],
                ),
                html.Div(id="candidates"),
            ]),
            html.Div(style={**CARD, "padding": "10px 12px", "flex": "0 0 auto",
                            "maxHeight": "104px", "overflowY": "auto"}, children=[
                html.Div("Notes", style=LABEL),
                html.Div(id="notes", style={"fontSize": "12px",
                                            "color": palette.TEXT_SECONDARY,
                                            "lineHeight": "1.5"}),
            ]),
            html.Div(style={"display": "flex", "gap": "8px", "alignItems": "center",
                            "flex": "0 0 auto"},
                     children=[
                         _button("PNG", "btn-png"),
                         _button("JSON", "btn-json"),
                         html.Div(id="export-status",
                                  style={"color": palette.TEXT_MUTED,
                                         "fontSize": "11px"}),
                     ]),
        ],
    )


INDEX_CSS = f"""
<style>
  /* Dash 4 themes its components through CSS variables, so one override here
     darkens the dropdown, radios, checklists and slider consistently instead
     of chasing individual class names. */
  :root {{
      --Dash-Text-Primary: {palette.TEXT_SECONDARY};
      --Dash-Text-Strong: {palette.TEXT_PRIMARY};
      --Dash-Text-Weak: {palette.TEXT_MUTED};
      --Dash-Text-Disabled: {palette.TEXT_MUTED};
      --Dash-Fill-Primary-Hover: {palette.PANEL};
      --Dash-Fill-Primary-Active: {palette.PANEL};
      --Dash-Fill-Interactive-Strong: {palette.BENEFICIARY};
      --Dash-Fill-Interactive-Weak: {palette.GRID};
      --Dash-Fill-Inverse-Strong: {palette.INK};
      --Dash-Fill-Inverse-strong: {palette.INK};
      --Dash-Fill-Disabled: {palette.GRID};
      --Dash-Stroke-Strong: {palette.TEXT_MUTED};
      --Dash-Stroke-Weak: {palette.GRID};
      --Dash-Shading-Strong: rgba(0,0,0,0.55);
      --Dash-Shading-Weak: rgba(0,0,0,0.35);
      --Dash-Tooltip-Background-Color: {palette.PANEL};
      --Dash-Tooltip-Border-Color: {palette.GRID};
  }}
  body {{ margin: 0; background: {palette.INK}; }}
  * {{ scrollbar-color: {palette.GRID} {palette.PANEL}; }}
  button:hover {{ border-color: {palette.TEXT_SECONDARY} !important; }}

  .dash-dropdown, [class*="dash-dropdown-menu"], [class*="dash-dropdown-listbox"] {{
      background: {palette.INK} !important;
      border-color: {palette.GRID} !important;
  }}
  .dash-dropdown-value-item span {{ color: {palette.TEXT_PRIMARY} !important; }}
  .dash-dropdown-trigger-icon path {{ fill: {palette.TEXT_MUTED} !important; }}

  .dash-options-list {{ background: transparent !important; }}
  .dash-options-list-option {{ background: transparent !important; cursor: pointer; }}
  .dash-options-list-option-text {{ color: {palette.TEXT_SECONDARY} !important; }}
  .dash-options-list-option[aria-selected="true"] .dash-options-list-option-text {{
      color: {palette.TEXT_PRIMARY} !important;
  }}
  .dash-options-list-option-checkbox {{ accent-color: {palette.BENEFICIARY}; }}

  /* the view toggle reads as a segmented control, not a pair of radios */
  .segmented .dash-options-list-option {{
      border: 1px solid {palette.GRID}; border-radius: 7px; padding: 3px 10px;
      margin-right: 6px !important;
  }}
  .segmented .dash-options-list-option[aria-selected="true"] {{
      background: {palette.PANEL}; border-color: {palette.TEXT_MUTED};
  }}
  .segmented .dash-options-list-option[aria-selected="true"]
  .dash-options-list-option-text {{ color: {palette.TEXT_PRIMARY} !important; }}
  .segmented .dash-options-list-option-checkbox {{ display: none !important; }}
  .segmented .dash-options-list-option-wrapper {{ display: none !important; }}

  /* the readout next to the slider already shows the time in seconds, and a
     raw frame index is not a useful thing to type into */
  .dash-slider-tooltip, .dash-range-slider-input {{ display: none !important; }}
  .dash-slider-mark {{ color: {palette.TEXT_MUTED} !important; font-size: 10px !important; }}
</style>
"""


# ---------------------------------------------------------------------------
# role dock
# ---------------------------------------------------------------------------
def _slot(role: str, title: str, colour: str):
    """One role slot: a drop target that arms on click and holds a player chip."""

    return html.Div(
        id=f"slot-{role}",
        n_clicks=0,
        className="slot",
        style={
            "border": f"1px dashed {palette.GRID}",
            "borderRadius": "9px",
            "padding": "7px 9px",
            "cursor": "pointer",
            "background": palette.INK,
            "transition": "border-color .12s, background .12s",
        },
        children=[
            html.Div(title, style={"color": colour, "fontSize": "10px",
                                   "fontWeight": 700, "letterSpacing": "0.08em",
                                   "textTransform": "uppercase",
                                   "marginBottom": "5px"}),
            html.Div(id=f"body-{role}",
                     style={"display": "flex", "flexWrap": "wrap", "gap": "5px",
                            "minHeight": "24px", "alignItems": "center"}),
        ],
    )


def _role_dock():
    return html.Div(style={**CARD, "padding": "11px 12px"}, children=[
        html.Div(
            style={"border": f"1px solid {palette.GRID}", "borderRadius": "11px",
                   "padding": "9px", "background": "rgba(217,207,184,0.04)"},
            children=[
                html.Div("Attack", style={**LABEL, "marginBottom": "7px",
                                          "color": palette.ATTACK_NEUTRAL}),
                _slot("runner", "Runner", palette.RUNNER),
                html.Div(
                    style={"display": "flex", "justifyContent": "center",
                           "margin": "5px 0"},
                    children=[html.Button(
                        "⇅ Swap", id="btn-swap", n_clicks=0,
                        style={**BUTTON, "padding": "3px 12px", "fontSize": "11px",
                               "borderRadius": "999px"},
                    )],
                ),
                _slot("beneficiary", "Beneficiary", palette.BENEFICIARY),
            ],
        ),
        html.Div(
            style={"border": f"1px solid {palette.GRID}", "borderRadius": "11px",
                   "padding": "9px", "background": "rgba(95,115,146,0.07)",
                   "marginTop": "9px"},
            children=[
                html.Div("Defence", style={**LABEL, "marginBottom": "7px",
                                           "color": palette.DEFEND_NEUTRAL}),
                _slot("defender", "Defender", palette.DEFENDER),
            ],
        ),
        html.Div(id="dock-meta", style={"color": palette.TEXT_MUTED,
                                        "fontSize": "11px", "marginTop": "8px"}),
        html.Div(id="hint", style={"color": palette.TEXT_SECONDARY,
                                   "fontSize": "11px", "marginTop": "4px"}),
    ])


# ---------------------------------------------------------------------------
# source panel
# ---------------------------------------------------------------------------
FACTS = (
    ("Measured", palette.BENEFICIARY,
     "Player and ball positions from IDSSE Bundesliga tracking at 25 Hz. "
     "Opened space is the beneficiary's goal-weighted residual space minus the same "
     "quantity with the selected defenders replaced by a no-reaction baseline, computed "
     "with offball_value.goal_weighted_influence on the Fernández & Bornn (2018) "
     "influence surface."),
    ("Human", palette.TEXT_SECONDARY,
     "Which players are runner, defender and beneficiary comes from manual annotation of "
     "45 scenes (shot_annotations.xlsx, effect strong and medium). The notes are a "
     "reviewer's qualitative comment, not a metric."),
    ("Explanatory", palette.DEFENDER,
     "The held defender is a what-if device, not a learned or optimised defensive "
     "response — this repository has none. Reacts most is an exploratory geometric "
     "heuristic on the goal-side marking target, not a research-pipeline output. Run "
     "starts come from the repository's kinematic onset detector where it fires, "
     "otherwise from a labelled acceleration cue."),
    ("Not shown", palette.TEXT_MUTED,
     "No calibrated xT, pass probability, dribble probability or learned threat surface "
     "is drawn, because this repository does not produce one."),
)


def source_sheet():
    rows = []
    for title, colour, body in FACTS:
        rows.append(html.Div(
            style={"display": "grid", "gridTemplateColumns": "104px 1fr",
                   "gap": "16px", "marginBottom": "10px"},
            children=[
                html.Div(title.upper(), style={**LABEL, "color": colour,
                                               "marginBottom": 0, "paddingTop": "3px"}),
                html.Div(body, style={"color": palette.TEXT_SECONDARY,
                                      "fontSize": "12.5px", "lineHeight": "1.6"}),
            ],
        ))
    return html.Div(
        id="source-sheet",
        style={"display": "none"},
        children=html.Div(
            style={"position": "fixed", "inset": 0, "zIndex": 120,
                   "background": "rgba(2,5,6,0.72)", "display": "flex",
                   "alignItems": "center", "justifyContent": "center",
                   "padding": "24px"},
            children=html.Div(
                style={**CARD, "maxWidth": "720px", "width": "100%",
                       "maxHeight": "84vh", "overflowY": "auto", "padding": "20px 22px",
                       "boxShadow": "0 24px 70px rgba(0,0,0,0.6)"},
                children=[
                    html.Div(
                        style={"display": "flex", "alignItems": "center",
                               "gap": "14px", "marginBottom": "14px"},
                        children=[
                            html.H2("Source & method",
                                    style={"margin": 0, "fontSize": "15px",
                                           "fontWeight": 700, "flex": "1 1 auto"}),
                            _button("Close", "btn-source-close"),
                        ],
                    ),
                    *rows,
                    html.Div(
                        style={"color": palette.TEXT_MUTED, "fontSize": "11.5px",
                               "marginTop": "14px"},
                        children="Research prototype. Not a validated off-ball value "
                                 "metric, player ranking or coaching recommendation.",
                    ),
                ],
            ),
        ),
    )
