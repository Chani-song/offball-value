"""Browser-friendly interactive viewer, exported as a self-contained HTML file.

Plotly is already an available dependency, so the viewer needs no build step:
one HTML file with a time scrubber, play/pause, layer toggles and the story
beats marked on the timeline.  It shows the same quantities as the video and
carries the same provenance footer, so a reviewer cannot mistake an
explanatory layer for a measured one.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import palette
from .scene import Scene
from .story import CHAIN, Storyboard


def _pitch_shapes(scene: Scene) -> list[dict]:
    half_l, half_w = scene.pitch_length / 2, scene.pitch_width / 2
    line = dict(color=palette.PITCH_LINE, width=1.1)
    shapes = [
        dict(type="rect", x0=-half_l, y0=-half_w, x1=half_l, y1=half_w,
             line=line, fillcolor=palette.PITCH_DARK, layer="below"),
        dict(type="line", x0=0, y0=-half_w, x1=0, y1=half_w, line=line, layer="below"),
        dict(type="circle", x0=-9.15, y0=-9.15, x1=9.15, y1=9.15, line=line, layer="below"),
    ]
    for sign in (-1, 1):
        base = sign * half_l
        shapes.append(dict(type="rect", x0=min(base, base - sign * 16.5),
                           x1=max(base, base - sign * 16.5), y0=-20.16, y1=20.16,
                           line=line, layer="below"))
        shapes.append(dict(type="rect", x0=min(base, base - sign * 5.5),
                           x1=max(base, base - sign * 5.5), y0=-9.16, y1=9.16,
                           line=line, layer="below"))
    return shapes


def _frame_data(scene: Scene, index: int, trail_frames: int) -> list[dict]:
    """Traces for one frame, in a fixed order the figure's traces mirror."""

    import plotly.graph_objects as go

    groups = {"attack": ([], [], []), "defend": ([], [], [])}
    roles = {"runner": ([], [], []), "defender": ([], [], []), "beneficiary": ([], [], [])}
    for player in scene.players.values():
        xy = player.xy[index]
        if not np.all(np.isfinite(xy)):
            continue
        view = scene.view_xy(xy)
        role = scene.role_of(player.player_id)
        bucket = roles[role] if role else groups[player.side]
        bucket[0].append(float(view[0]))
        bucket[1].append(float(view[1]))
        bucket[2].append(f"#{player.label} {player.name}")

    data = []
    for side, colour, size in (("attack", palette.ATTACK_NEUTRAL, 13),
                               ("defend", palette.DEFEND_NEUTRAL, 13)):
        xs, ys, text = groups[side]
        data.append(go.Scatter(x=xs, y=ys, text=text, mode="markers", hoverinfo="text",
                               marker=dict(color=colour, size=size,
                                           line=dict(color=palette.INK, width=1)),
                               name=scene.attacking_team_name if side == "attack"
                               else scene.defending_team_name))
    for role in ("runner", "defender", "beneficiary"):
        xs, ys, text = roles[role]
        data.append(go.Scatter(
            x=xs, y=ys, text=text, mode="markers+text", hoverinfo="text",
            textposition="top center",
            textfont=dict(color=palette.ROLE_COLOURS[role], size=11),
            marker=dict(color=palette.ROLE_COLOURS[role], size=20,
                        line=dict(color=palette.INK, width=2)),
            name=palette.ROLE_LABELS[role],
        ))

    ball = scene.view_xy(scene.ball_xy[index])
    data.append(go.Scatter(
        x=[float(ball[0])] if np.all(np.isfinite(ball)) else [],
        y=[float(ball[1])] if np.all(np.isfinite(ball)) else [],
        mode="markers", hoverinfo="skip", name="ball",
        marker=dict(color=palette.BALL, size=9, line=dict(color=palette.INK, width=1)),
    ))

    # trail
    start = max(0, index - trail_frames)
    trail_x: list[float] = []
    trail_y: list[float] = []
    for runner_id in scene.runner_ids:
        track = scene.view_xy(scene.players[runner_id].xy[start : index + 1])
        track = track[np.all(np.isfinite(track), axis=1)]
        trail_x.extend(track[:, 0].tolist() + [None])      # type: ignore[arg-type]
        trail_y.extend(track[:, 1].tolist() + [None])      # type: ignore[arg-type]
    data.append(go.Scatter(x=trail_x, y=trail_y, mode="lines", hoverinfo="skip",
                           line=dict(color=palette.RUNNER, width=4), opacity=0.75,
                           name="runner trail"))

    # tether
    tether_x: list[float] = []
    tether_y: list[float] = []
    if scene.runner_ids:
        runner_xy = scene.view_xy(scene.players[scene.runner_ids[0]].xy[index])
        for defender_id in scene.defender_ids:
            defender_xy = scene.view_xy(scene.players[defender_id].xy[index])
            if np.all(np.isfinite(runner_xy)) and np.all(np.isfinite(defender_xy)):
                tether_x.extend([float(runner_xy[0]), float(defender_xy[0]), None])
                tether_y.extend([float(runner_xy[1]), float(defender_xy[1]), None])
    data.append(go.Scatter(x=tether_x, y=tether_y, mode="lines", hoverinfo="skip",
                           line=dict(color=palette.DEFENDER, width=2, dash="dash"),
                           name="runner-defender link"))

    # ghost defenders
    stack = scene.surfaces
    freeze = getattr(stack, "freeze_index", None) if stack is not None else None
    ghost_x: list[float] = []
    ghost_y: list[float] = []
    if freeze is not None:
        for defender_id in scene.defender_ids:
            held = scene.view_xy(scene.players[defender_id].xy[freeze])
            if np.all(np.isfinite(held)):
                ghost_x.append(float(held[0]))
                ghost_y.append(float(held[1]))
    data.append(go.Scatter(x=ghost_x, y=ghost_y, mode="markers", hoverinfo="skip",
                           marker=dict(color=palette.GHOST, size=20, opacity=0.35,
                                       line=dict(color=palette.GHOST, width=2)),
                           name="defender held position"))
    return data


def _space_trace(scene: Scene, index: int, downsample: int = 2):
    import plotly.graph_objects as go

    stack = scene.surfaces
    if stack is None:
        return go.Heatmap(z=[[0]], x=[0], y=[0], showscale=False, opacity=0.0,
                          hoverinfo="skip", name="space")
    field = stack.at(index)
    if scene.flip:
        field = field[::-1, ::-1]
    peak = float(np.nanmax(stack.factual)) or 1.0
    step = max(1, int(downsample))
    # the HTML carries one grid per frame, so keep the payload light
    field = np.round(np.clip(field[::step, ::step] / peak, 0, 1), 3)
    return go.Heatmap(
        z=field, x=stack.xgrid[::step], y=stack.ygrid[::step],
        colorscale=[[0.0, "rgba(92,232,245,0)"], [0.25, "rgba(92,232,245,0.18)"],
                    [0.65, "rgba(92,232,245,0.45)"], [1.0, "rgba(92,232,245,0.72)"]],
        zmin=0, zmax=1, showscale=False, hoverinfo="skip", name="open space",
        zsmooth="best",
    )


def export_html(
    scene: Scene,
    storyboard: Storyboard,
    path: Path,
    stride: int = 3,
    include_plotlyjs: str = "cdn",
    field_downsample: int = 3,
) -> Path:
    """Write a self-contained interactive viewer for one scene."""

    import plotly.graph_objects as go

    indices = list(range(0, scene.n_frames, max(1, int(stride))))
    if indices[-1] != scene.n_frames - 1:
        indices.append(scene.n_frames - 1)
    trail_frames = int(round(1.8 * scene.fps))

    base = [_space_trace(scene, indices[0], field_downsample)] + _frame_data(
        scene, indices[0], trail_frames)
    frames = [
        go.Frame(
            data=[_space_trace(scene, i, field_downsample)]
            + _frame_data(scene, i, trail_frames),
            name=f"{scene.times[i]:.2f}",
            layout=dict(annotations=_annotations(scene, storyboard, i)),
        )
        for i in indices
    ]

    half_l, half_w = scene.pitch_length / 2, scene.pitch_width / 2
    figure = go.Figure(data=base, frames=frames)
    figure.update_layout(
        template="plotly_dark",
        paper_bgcolor=palette.INK,
        plot_bgcolor=palette.INK,
        font=dict(family="DejaVu Sans, Helvetica, Arial", color=palette.TEXT_PRIMARY, size=12),
        title=dict(
            text=f"<b>{scene.title}</b><br>"
                 f"<span style='font-size:12px;color:{palette.TEXT_SECONDARY}'>"
                 f"{scene.subtitle}</span>",
            x=0.02, xanchor="left", y=0.965,
        ),
        shapes=_pitch_shapes(scene),
        xaxis=dict(range=[-half_l - 3, half_l + 3], visible=False, constrain="domain"),
        yaxis=dict(range=[-half_w - 3, half_w + 3], visible=False,
                   scaleanchor="x", scaleratio=1),
        margin=dict(l=24, r=24, t=96, b=112),
        legend=dict(orientation="h", y=-0.10, x=0, bgcolor="rgba(0,0,0,0)"),
        annotations=_annotations(scene, storyboard, indices[0]),
        updatemenus=[
            dict(
                type="buttons", direction="left", x=0.02, y=-0.185, xanchor="left",
                bgcolor=palette.PANEL, bordercolor=palette.GRID,
                buttons=[
                    dict(label="▶ Play", method="animate",
                         args=[None, dict(frame=dict(duration=1000 * stride / scene.fps,
                                                     redraw=True),
                                          fromcurrent=True, mode="immediate")]),
                    dict(label="⏸ Pause", method="animate",
                         args=[[None], dict(frame=dict(duration=0, redraw=False),
                                            mode="immediate")]),
                ],
            ),
        ],
        sliders=[dict(
            active=0, x=0.02, len=0.96, y=-0.055, xanchor="left",
            bgcolor=palette.PANEL, activebgcolor=palette.TEXT_SECONDARY,
            bordercolor=palette.GRID, tickcolor=palette.TEXT_MUTED,
            currentvalue=dict(prefix="t = ", suffix=" s",
                              font=dict(size=13, color=palette.TEXT_SECONDARY)),
            steps=[dict(method="animate", label=f"{scene.times[i]:.1f}",
                        args=[[f"{scene.times[i]:.2f}"],
                              dict(mode="immediate",
                                   frame=dict(duration=0, redraw=True))])
                   for i in indices],
        )],
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(
        str(path), include_plotlyjs=include_plotlyjs, auto_play=False,
        full_html=True, div_id="offball-demo",
    )
    _inject_chrome(path, scene, storyboard)
    return path


def _annotations(scene: Scene, storyboard: Storyboard, index: int) -> list[dict]:
    t = float(scene.times[index])
    beat = storyboard.beat_at(t)
    progress = storyboard.chain_progress(t)
    annotations = [dict(
        x=0.02, y=1.02, xref="paper", yref="paper", xanchor="left", yanchor="bottom",
        showarrow=False, align="left",
        text="   ".join(
            f"<span style='color:{palette.ROLE_COLOURS.get(role, palette.SPACE_WAKE) if lit > 0.5 else palette.TEXT_MUTED}'>"
            f"<b>{label}</b></span>"
            for (label, role), lit in zip(CHAIN, progress)
        ),
        font=dict(size=12),
    ), dict(
        x=0.02, y=-0.135, xref="paper", yref="paper", xanchor="left", yanchor="top",
        showarrow=False, align="left",
        text=f"<b>{beat.title}</b>  ·  <span style='color:{palette.TEXT_SECONDARY}'>"
             f"{beat.caption}</span>",
        font=dict(size=12),
    )]
    gain = scene.series.get("_value_gain")
    times = scene.series.get("_surface_t")
    if gain is not None and times is not None:
        value = float(gain[int(np.argmin(np.abs(times - t)))])
        colour = palette.DELTA_POS if value >= 0 else palette.DELTA_NEG
        annotations.append(dict(
            x=0.98, y=1.02, xref="paper", yref="paper", xanchor="right", yanchor="bottom",
            showarrow=False,
            text=f"<span style='color:{palette.TEXT_MUTED};font-size:11px'>"
                 f"SPACE GAINED VS HELD DEFENDER</span>  "
                 f"<span style='color:{colour};font-size:20px'><b>{value:+.1f}</b></span>",
        ))
    return annotations


_FOOTER_CSS = """
<style>
  body {{ background: {ink}; color: {text}; font-family: 'DejaVu Sans', Helvetica, Arial;
          margin: 0; }}
  .offball-footer {{ max-width: 1180px; margin: 4px auto 28px auto; padding: 14px 22px;
      border-top: 1px solid {grid}; font-size: 12px; line-height: 1.7; color: {muted}; }}
  .offball-footer b {{ letter-spacing: .08em; font-size: 11px; }}
  .tag-measured {{ color: {cyan}; }}
  .tag-human {{ color: {text2}; }}
  .tag-explanatory {{ color: {amber}; }}
</style>
"""


def _inject_chrome(path: Path, scene: Scene, storyboard: Storyboard) -> None:
    """Add the provenance footer that the video carries, to the HTML export."""

    html = path.read_text()
    css = _FOOTER_CSS.format(
        ink=palette.INK, text=palette.TEXT_PRIMARY, text2=palette.TEXT_SECONDARY,
        muted=palette.TEXT_MUTED, grid=palette.GRID, cyan=palette.SPACE_WAKE,
        amber=palette.DEFENDER,
    )
    rows = [
        ("measured", "MEASURED", "player and ball positions from IDSSE tracking at 25 Hz"),
    ]
    if scene.surfaces is not None:
        rows.append(("measured", "MEASURED",
                     "shaded space: goal-weighted residual influence "
                     "(offball_value.goal_weighted_influence, repo v0.1)"))
        rows.append(("explanatory", "EXPLANATORY",
                     "the held-position defender is a what-if device, not a learned "
                     "defensive best response"))
    rows.append(("human", "HUMAN",
                 "runner / reacting defender / beneficiary come from manual annotation"
                 if scene.source == "annotation"
                 else f"roles supplied by the {scene.source} adapter"))
    rows.append(("human", "NOT SHOWN",
                 "no calibrated xT, pass probability or dribble probability is rendered; "
                 "this repository does not currently produce them"))
    footer = "<div class='offball-footer'>" + "".join(
        f"<div><b class='tag-{kind}'>{tag}</b> &nbsp; {text}</div>" for kind, tag, text in rows
    ) + f"<div style='margin-top:8px'>scene <code>{scene.scene_id}</code></div></div>"
    html = html.replace("</head>", css + "</head>", 1)
    html = html.replace("</body>", footer + "</body>", 1)
    path.write_text(html)
