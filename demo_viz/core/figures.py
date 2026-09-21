"""Plotly figures for the interactive app.

Same visual language as the rendered videos: the role colours and layer
semantics come from :mod:`demo_viz.palette` and mirror
:mod:`demo_viz.render.layers`, expressed as Plotly traces so the pitch is
clickable.

Every player marker carries its player id in ``customdata``, so a pitch click
resolves to a player exactly rather than by nearest-neighbour search.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go

from .. import palette
from ..scene import Scene
from .influence import InfluenceCache
from .selection import Selection

LAYERS = ("paths", "trail", "tether", "wake", "lane", "ghost", "labels",
          "candidates", "fit")
DEFAULT_LAYERS = ("trail", "tether", "wake", "lane", "ghost", "labels", "candidates")

_ROLE_COLOUR = {
    "runner": palette.RUNNER,
    "defender": palette.DEFENDER,
    "beneficiary": palette.BENEFICIARY,
}


@dataclass
class ViewOptions:
    layers: tuple[str, ...] = DEFAULT_LAYERS
    trail_seconds: float = 1.8
    wake_mode: str = "gain"          # "gain" (factual - baseline) | "space" (factual)

    def on(self, name: str) -> bool:
        return name in self.layers


def _viewport(scene: Scene, index: int, selection: Selection, margin_m: float = 13.0,
              minimum_m: float = 46.0) -> tuple[float, float, float, float]:
    """Crop to the picked players and the ball, so a close look stays clickable."""

    points = []
    for player_id in selection.runners + selection.defenders + selection.beneficiaries:
        player = scene.players.get(player_id)
        if player is not None:
            points.append(scene.view_xy(player.xy[index]))
    ball = scene.view_xy(scene.ball_xy[index])
    if np.all(np.isfinite(ball)):
        points.append(ball)
    points = [p for p in points if np.all(np.isfinite(p))]
    half_l, half_w = scene.pitch_length / 2 + 2, scene.pitch_width / 2 + 2
    if not points:
        return -half_l, half_l, -half_w, half_w
    arr = np.array(points)
    lo, hi = arr.min(axis=0) - margin_m, arr.max(axis=0) + margin_m
    width = max(float(hi[0] - lo[0]), minimum_m)
    height = max(float(hi[1] - lo[1]), minimum_m / 1.6)
    centre = (lo + hi) / 2
    x0 = float(np.clip(centre[0] - width / 2, -half_l, half_l - width))
    y0 = float(np.clip(centre[1] - height / 2, -half_w, half_w - height))
    return x0, x0 + width, y0, y0 + height


# ---------------------------------------------------------------------------
# pitch
# ---------------------------------------------------------------------------
def pitch_shapes(scene: Scene) -> list[dict]:
    """Pitch markings as Plotly shapes, drawn under every trace."""

    half_l, half_w = scene.pitch_length / 2, scene.pitch_width / 2
    line = dict(color=palette.PITCH_LINE, width=1.1)
    faint = dict(color=palette.PITCH_LINE, width=0.9)
    shapes = [
        dict(type="rect", x0=-half_l, y0=-half_w, x1=half_l, y1=half_w,
             line=line, fillcolor=palette.PITCH_DARK, layer="below"),
        dict(type="line", x0=0, y0=-half_w, x1=0, y1=half_w, line=faint, layer="below"),
        dict(type="circle", x0=-9.15, y0=-9.15, x1=9.15, y1=9.15, line=faint, layer="below"),
    ]
    for sign in (-1, 1):
        base = sign * half_l
        for length, width in ((16.5, 40.32), (5.5, 18.32)):
            shapes.append(dict(
                type="rect",
                x0=min(base, base - sign * length), x1=max(base, base - sign * length),
                y0=-width / 2, y1=width / 2, line=faint, layer="below",
            ))
        shapes.append(dict(
            type="rect",
            x0=min(base, base + sign * 2.0), x1=max(base, base + sign * 2.0),
            y0=-3.66, y1=3.66, line=faint, fillcolor=palette.INK, layer="below",
        ))
    return shapes


# ---------------------------------------------------------------------------
# main figure
# ---------------------------------------------------------------------------
def scene_figure(
    scene: Scene,
    index: int,
    selection: Selection,
    cache: InfluenceCache | None = None,
    options: ViewOptions | None = None,
    freeze_index: int | None = None,
    baseline: str = "hold",
    candidates: tuple[str, ...] = (),
) -> go.Figure:
    """The clickable pitch for one frame and one role selection."""

    options = options or ViewOptions()
    index = int(np.clip(index, 0, scene.n_frames - 1))
    traces: list[go.BaseTraceType] = []

    traces.append(_space_trace(scene, index, selection, cache, options, freeze_index, baseline))
    traces += _path_traces(scene, index, selection, options)
    traces += _lane_traces(scene, index, selection, options)
    traces += _tether_traces(scene, index, selection, options)
    traces += _ghost_traces(scene, index, selection, options, freeze_index, baseline, cache)
    traces += _trail_traces(scene, index, selection, options)
    traces += _player_traces(scene, index, selection, options, candidates)
    traces.append(_ball_trace(scene, index))

    if options.on("fit") and not selection.is_empty:
        x0, x1, y0, y1 = _viewport(scene, index, selection)
    else:
        half_l, half_w = scene.pitch_length / 2 + 2, scene.pitch_width / 2 + 2
        x0, x1, y0, y1 = -half_l, half_l, -half_w, half_w
    figure = go.Figure(data=traces)
    figure.update_layout(
        paper_bgcolor=palette.INK,
        plot_bgcolor=palette.INK,
        font=dict(family="Inter, Helvetica, Arial, sans-serif",
                  color=palette.TEXT_PRIMARY, size=12),
        shapes=pitch_shapes(scene),
        xaxis=dict(range=[x0, x1], visible=False, constrain="domain", fixedrange=True),
        yaxis=dict(range=[y0, y1], visible=False,
                   scaleanchor="x", scaleratio=1, fixedrange=True),
        margin=dict(l=6, r=6, t=6, b=6),
        showlegend=False,
        autosize=True,
        hovermode="closest",
        clickmode="event",
        dragmode=False,
        transition=dict(duration=0),
        uirevision=scene.scene_id,
    )
    return figure


# ---------------------------------------------------------------------------
# layers
# ---------------------------------------------------------------------------
def _space_trace(scene, index, selection, cache, options, freeze_index, baseline):
    blank = go.Heatmap(z=[[0.0]], x=[0.0], y=[0.0], showscale=False, opacity=0.0,
                       hoverinfo="skip")
    if cache is None or not options.on("wake") or not selection.beneficiaries:
        return blank
    slot = cache.slot_for(index)
    swap = ()
    if selection.defenders and freeze_index is not None:
        swap = tuple((d, int(freeze_index), baseline) for d in selection.defenders)
    factual = cache.combined(selection.beneficiaries, slot)
    if options.wake_mode == "gain" and swap:
        counter = cache.combined(selection.beneficiaries, slot, swap)
        field = factual.field - counter.field
        scale = _CYAN_SCALE
        peak = float(np.nanmax(np.abs(field))) or 1.0
        z = np.clip(field / peak, 0.0, 1.0)
    else:
        field = factual.field
        scale = _CYAN_SCALE
        peak = float(np.nanmax(field)) or 1.0
        z = np.clip(field / peak, 0.0, 1.0)
    if scene.flip:
        z = z[::-1, ::-1]
    return go.Heatmap(
        z=np.round(z, 3), x=cache.xgrid, y=cache.ygrid, zmin=0.0, zmax=1.0,
        colorscale=scale, showscale=False, hoverinfo="skip", zsmooth="best",
    )


_CYAN_SCALE = [
    [0.00, "rgba(92,232,245,0)"],
    [0.18, "rgba(92,232,245,0.10)"],
    [0.45, "rgba(92,232,245,0.32)"],
    [0.75, "rgba(92,232,245,0.55)"],
    [1.00, "rgba(92,232,245,0.78)"],
]


def _path_traces(scene, index, selection, options):
    if not options.on("paths"):
        return []
    xs: list[float | None] = []
    ys: list[float | None] = []
    for player in scene.players.values():
        if player.is_goalkeeper:
            continue
        track = scene.view_xy(player.xy)
        track = track[np.all(np.isfinite(track), axis=1)]
        if len(track) < 2:
            continue
        xs.extend(track[:, 0].tolist() + [None])
        ys.extend(track[:, 1].tolist() + [None])
    return [go.Scatter(x=xs, y=ys, mode="lines", hoverinfo="skip",
                       line=dict(color=palette.TEXT_MUTED, width=1), opacity=0.28)]


def _trail_traces(scene, index, selection, options):
    if not options.on("trail"):
        return []
    span = max(2, int(round(options.trail_seconds * scene.fps)))
    out = []
    for runner_id in selection.runners:
        player = scene.players.get(runner_id)
        if player is None:
            continue
        track = scene.view_xy(player.xy[max(0, index - span): index + 1])
        track = track[np.all(np.isfinite(track), axis=1)]
        if len(track) < 2:
            continue
        out.append(go.Scatter(x=track[:, 0], y=track[:, 1], mode="lines", hoverinfo="skip",
                              line=dict(color=palette.RUNNER, width=9, shape="spline"),
                              opacity=0.16))
        out.append(go.Scatter(x=track[:, 0], y=track[:, 1], mode="lines", hoverinfo="skip",
                              line=dict(color=palette.RUNNER, width=3, shape="spline"),
                              opacity=0.9))
    return out


def _tether_traces(scene, index, selection, options):
    if not options.on("tether") or not selection.runners or not selection.defenders:
        return []
    runner = scene.players.get(selection.runners[0])
    if runner is None:
        return []
    start = scene.view_xy(runner.xy[index])
    if not np.all(np.isfinite(start)):
        return []
    xs: list[float | None] = []
    ys: list[float | None] = []
    labels_x, labels_y, labels = [], [], []
    for defender_id in selection.defenders:
        defender = scene.players.get(defender_id)
        if defender is None:
            continue
        end = scene.view_xy(defender.xy[index])
        if not np.all(np.isfinite(end)):
            continue
        xs.extend([float(start[0]), float(end[0]), None])
        ys.extend([float(start[1]), float(end[1]), None])
        middle = (start + end) / 2
        labels_x.append(float(middle[0]))
        labels_y.append(float(middle[1]))
        labels.append(f"{float(np.linalg.norm(end - start)):.0f} m")
    out = [go.Scatter(x=xs, y=ys, mode="lines", hoverinfo="skip",
                      line=dict(color=palette.DEFENDER, width=2, dash="dash"), opacity=0.85)]
    if options.on("labels") and labels:
        out.append(go.Scatter(
            x=labels_x, y=labels_y, mode="text", text=labels, hoverinfo="skip",
            textfont=dict(color=palette.DEFENDER, size=11),
        ))
    return out


def _ghost_traces(scene, index, selection, options, freeze_index, baseline, cache):
    if not options.on("ghost") or not selection.defenders or freeze_index is None:
        return []
    from ..quantities import baseline_tracks

    tracks, _ = baseline_tracks(scene, int(freeze_index), baseline,
                                defender_ids=tuple(selection.defenders))
    xs, ys, text = [], [], []
    link_x: list[float | None] = []
    link_y: list[float | None] = []
    for defender_id in selection.defenders:
        track = tracks.get(defender_id)
        player = scene.players.get(defender_id)
        if track is None or player is None:
            continue
        held = scene.view_xy(track[index])
        now = scene.view_xy(player.xy[index])
        if not (np.all(np.isfinite(held)) and np.all(np.isfinite(now))):
            continue
        xs.append(float(held[0]))
        ys.append(float(held[1]))
        text.append(f"#{player.label} if no reaction · pulled "
                    f"{float(np.linalg.norm(now - held)):.0f} m")
        link_x.extend([float(held[0]), float(now[0]), None])
        link_y.extend([float(held[1]), float(now[1]), None])
    if not xs:
        return []
    return [
        go.Scatter(x=link_x, y=link_y, mode="lines", hoverinfo="skip",
                   line=dict(color=palette.GHOST, width=1.2, dash="dot"), opacity=0.55),
        go.Scatter(x=xs, y=ys, mode="markers", text=text, hoverinfo="text",
                   marker=dict(color=palette.GHOST, size=20, opacity=0.18,
                               line=dict(color=palette.GHOST, width=2))),
    ]


def _lane_traces(scene, index, selection, options, maximum_m: float = 34.0):
    if not options.on("lane") or not selection.beneficiaries:
        return []
    ball = scene.view_xy(scene.ball_xy[index])
    if not np.all(np.isfinite(ball)):
        return []
    xs: list[float | None] = []
    ys: list[float | None] = []
    for beneficiary_id in selection.beneficiaries:
        player = scene.players.get(beneficiary_id)
        if player is None:
            continue
        target = scene.view_xy(player.xy[index])
        if not np.all(np.isfinite(target)):
            continue
        if float(np.linalg.norm(target - ball)) > maximum_m:
            continue
        xs.extend([float(ball[0]), float(target[0]), None])
        ys.extend([float(ball[1]), float(target[1]), None])
    if not xs:
        return []
    return [go.Scatter(x=xs, y=ys, mode="lines", hoverinfo="skip",
                       line=dict(color=palette.BENEFICIARY, width=6), opacity=0.22),
            go.Scatter(x=xs, y=ys, mode="lines", hoverinfo="skip",
                       line=dict(color=palette.BENEFICIARY, width=2), opacity=0.85)]


def _player_traces(scene, index, selection, options, candidates):
    groups: dict[str, dict[str, list]] = {}
    for key in ("attack", "defend", "runner", "defender", "beneficiary"):
        groups[key] = {"x": [], "y": [], "text": [], "ids": [], "label": []}

    for player in scene.players.values():
        xy = player.xy[index]
        if not np.all(np.isfinite(xy)):
            continue
        view = scene.view_xy(xy)
        role = selection.role_of(player.player_id)
        key = role or player.side
        bucket = groups[key]
        bucket["x"].append(float(view[0]))
        bucket["y"].append(float(view[1]))
        bucket["ids"].append(player.player_id)
        bucket["label"].append(str(player.label))
        team = scene.attacking_team_name if player.side == "attack" else scene.defending_team_name
        suffix = f" · {role}" if role else ""
        bucket["text"].append(f"#{player.label} {player.name}<br>{team}{suffix}")

    out: list[go.BaseTraceType] = []

    # candidate rings sit under the markers
    if options.on("candidates") and candidates:
        ring_x, ring_y = [], []
        for player_id in candidates:
            player = scene.players.get(player_id)
            if player is None:
                continue
            xy = player.xy[index]
            if not np.all(np.isfinite(xy)):
                continue
            view = scene.view_xy(xy)
            ring_x.append(float(view[0]))
            ring_y.append(float(view[1]))
        if ring_x:
            out.append(go.Scatter(
                x=ring_x, y=ring_y, mode="markers", hoverinfo="skip",
                marker=dict(size=36, color="rgba(0,0,0,0)",
                            line=dict(color=palette.DEFENDER, width=2)), opacity=0.55,
            ))

    for key, colour, size in (
        ("attack", palette.ATTACK_NEUTRAL, 20),
        ("defend", palette.DEFEND_NEUTRAL, 20),
        ("runner", palette.RUNNER, 28),
        ("defender", palette.DEFENDER, 28),
        ("beneficiary", palette.BENEFICIARY, 28),
    ):
        bucket = groups[key]
        if not bucket["x"]:
            continue
        is_role = key in _ROLE_COLOUR
        out.append(go.Scatter(
            x=bucket["x"], y=bucket["y"],
            customdata=bucket["ids"],
            text=bucket["label"] if options.on("labels") else None,
            mode="markers+text" if options.on("labels") else "markers",
            textposition="middle center",
            textfont=dict(color=palette.INK, size=11 if is_role else 10,
                          family="Inter, Helvetica, Arial"),
            hovertext=bucket["text"], hoverinfo="text",
            marker=dict(
                color=colour, size=size,
                line=dict(color=palette.INK if not is_role else colour,
                          width=1 if not is_role else 3),
                opacity=1.0 if is_role else 0.55,
            ),
        ))
    return out


def _ball_trace(scene, index):
    ball = scene.view_xy(scene.ball_xy[index])
    if not np.all(np.isfinite(ball)):
        return go.Scatter(x=[], y=[], mode="markers", hoverinfo="skip")
    return go.Scatter(
        x=[float(ball[0])], y=[float(ball[1])], mode="markers",
        hovertext=["ball"], hoverinfo="text",
        marker=dict(color=palette.BALL, size=11,
                    line=dict(color=palette.INK, width=2)),
    )


# ---------------------------------------------------------------------------
# side chart
# ---------------------------------------------------------------------------
def space_chart(
    scene: Scene,
    cache: InfluenceCache | None,
    selection: Selection,
    index: int,
    freeze_index: int | None,
    baseline: str = "hold",
    height: int = 190,
) -> go.Figure:
    """Beneficiary space over time, observed against the no-reaction baseline."""

    figure = go.Figure()
    figure.update_layout(
        paper_bgcolor=palette.INK, plot_bgcolor=palette.INK,
        font=dict(family="Inter, Helvetica, Arial", color=palette.TEXT_SECONDARY, size=10),
        margin=dict(l=36, r=10, t=8, b=26), height=height, showlegend=False,
        xaxis=dict(gridcolor=palette.GRID, zeroline=False, fixedrange=True,
                   title=dict(text="s", standoff=4)),
        yaxis=dict(gridcolor=palette.GRID, zeroline=False, fixedrange=True),
        hovermode="x unified",
    )
    if cache is None or not selection.beneficiaries:
        figure.add_annotation(text="pick a beneficiary", showarrow=False,
                              font=dict(color=palette.TEXT_MUTED, size=11))
        figure.update_xaxes(visible=False)
        figure.update_yaxes(visible=False)
        return figure

    times = cache.times
    factual = np.sum([cache.residual_series(pid) for pid in selection.beneficiaries], axis=0)
    counter = None
    if selection.defenders and freeze_index is not None:
        swap = tuple((d, int(freeze_index), baseline) for d in selection.defenders)
        counter = np.sum(
            [cache.residual_series(pid, swap) for pid in selection.beneficiaries], axis=0
        )
        figure.add_trace(go.Scatter(
            x=times, y=counter, mode="lines", name="no reaction",
            line=dict(color=palette.DEFENDER, width=1.6, dash="dash"),
            fill=None, hovertemplate="no reaction %{y:.1f}<extra></extra>",
        ))
        figure.add_trace(go.Scatter(
            x=times, y=factual, mode="lines", name="observed",
            line=dict(color=palette.BENEFICIARY, width=2.2),
            fill="tonexty", fillcolor="rgba(92,232,245,0.16)",
            hovertemplate="observed %{y:.1f}<extra></extra>",
        ))
    else:
        figure.add_trace(go.Scatter(
            x=times, y=factual, mode="lines", name="observed",
            line=dict(color=palette.BENEFICIARY, width=2.2),
            hovertemplate="observed %{y:.1f}<extra></extra>",
        ))

    now = float(scene.times[int(np.clip(index, 0, scene.n_frames - 1))])
    figure.add_vline(x=now, line=dict(color=palette.TEXT_PRIMARY, width=1))
    if freeze_index is not None:
        figure.add_vline(x=float(scene.times[int(freeze_index)]),
                         line=dict(color=palette.RUNNER, width=1, dash="dot"))
    return figure
