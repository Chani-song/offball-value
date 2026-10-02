"""A clean, broadcast-leaning football pitch drawn in metres.

The pitch is rendered in *view* coordinates (see :meth:`demo_viz.scene.Scene.view_xy`):
origin at the centre spot, attacking team always moving toward +x.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import Arc, Circle, Rectangle

from .. import palette

# Standard markings, in metres, for a 105 x 68 pitch.
PENALTY_AREA_LENGTH = 16.5
PENALTY_AREA_WIDTH = 40.32
GOAL_AREA_LENGTH = 5.5
GOAL_AREA_WIDTH = 18.32
PENALTY_SPOT = 11.0
CENTRE_CIRCLE_R = 9.15
GOAL_WIDTH = 7.32
GOAL_DEPTH = 2.0
CORNER_R = 1.0


def tracked(text: str, space: str = "\u2009") -> str:
    """Poor-man's letter-spacing for small all-caps labels."""

    return space.join(text)


@dataclass(frozen=True)
class PitchStyle:
    """Colours and weights for the pitch surface and its markings."""

    grass: str = palette.PITCH_DARK
    band: str = palette.PITCH_LIGHT
    line: str = palette.PITCH_LINE
    line_alpha: float = 0.48
    line_width: float = 1.15
    band_count: int = 6
    band_alpha: float = 0.75
    vignette: float = 0.34
    margin_m: float = 3.2


def draw_pitch(
    ax: Axes,
    length: float = 105.0,
    width: float = 68.0,
    style: PitchStyle = PitchStyle(),
) -> None:
    """Draw grass, mow bands, a soft vignette and all standard markings."""

    half_l, half_w = length / 2.0, width / 2.0
    ax.set_facecolor(palette.INK)
    ax.set_xlim(-half_l - style.margin_m, half_l + style.margin_m)
    ax.set_ylim(-half_w - style.margin_m, half_w + style.margin_m)
    ax.set_aspect("equal")
    ax.axis("off")

    # grass
    ax.add_patch(
        Rectangle(
            (-half_l, -half_w), length, width,
            facecolor=style.grass, edgecolor="none", zorder=0,
        )
    )
    # mow bands
    band_w = length / style.band_count
    for index in range(style.band_count):
        if index % 2:
            continue
        ax.add_patch(
            Rectangle(
                (-half_l + index * band_w, -half_w), band_w, width,
                facecolor=style.band, edgecolor="none",
                alpha=style.band_alpha, zorder=0.1,
            )
        )
    # vignette: a soft radial darkening that keeps the eye on the middle
    if style.vignette > 0:
        res = 96
        ys, xs = np.mgrid[0:res, 0:res]
        radial = np.hypot((xs / (res - 1) - 0.5) * 2.0, (ys / (res - 1) - 0.5) * 2.0)
        alpha = np.clip((radial - 0.55) / 0.85, 0.0, 1.0) ** 1.6 * style.vignette
        rgba = np.zeros((res, res, 4))
        rgba[..., 3] = alpha
        ax.imshow(
            rgba,
            extent=(-half_l, half_l, -half_w, half_w),
            origin="lower", zorder=0.2, interpolation="bilinear", aspect="auto",
        )

    kwargs = dict(
        color=style.line,
        lw=style.line_width,
        alpha=style.line_alpha,
        zorder=1.0,
        fill=False,
    )
    line_kwargs = dict(color=style.line, lw=style.line_width, alpha=style.line_alpha, zorder=1.0)

    ax.add_patch(Rectangle((-half_l, -half_w), length, width, **kwargs))
    ax.plot([0, 0], [-half_w, half_w], **line_kwargs)
    ax.add_patch(Circle((0, 0), CENTRE_CIRCLE_R, **kwargs))
    ax.plot([0], [0], marker="o", ms=2.4, color=style.line, alpha=style.line_alpha, zorder=1.0)

    for sign in (-1, 1):
        base = sign * half_l
        # penalty area
        ax.add_patch(
            Rectangle(
                (base - (PENALTY_AREA_LENGTH if sign > 0 else 0), -PENALTY_AREA_WIDTH / 2),
                PENALTY_AREA_LENGTH, PENALTY_AREA_WIDTH, **kwargs,
            )
        )
        # goal area
        ax.add_patch(
            Rectangle(
                (base - (GOAL_AREA_LENGTH if sign > 0 else 0), -GOAL_AREA_WIDTH / 2),
                GOAL_AREA_LENGTH, GOAL_AREA_WIDTH, **kwargs,
            )
        )
        # penalty spot and arc
        spot_x = base - sign * PENALTY_SPOT
        ax.plot([spot_x], [0], marker="o", ms=2.2, color=style.line, alpha=style.line_alpha, zorder=1.0)
        # only the portion of the penalty arc outside the box is drawn
        span = np.degrees(np.arccos(
            (PENALTY_AREA_LENGTH - PENALTY_SPOT) / CENTRE_CIRCLE_R
        ))
        theta1, theta2 = (180 - span, 180 + span) if sign > 0 else (-span, span)
        ax.add_patch(
            Arc(
                (spot_x, 0), 2 * CENTRE_CIRCLE_R, 2 * CENTRE_CIRCLE_R,
                angle=0, theta1=theta1, theta2=theta2,
                color=style.line, lw=style.line_width, alpha=style.line_alpha, zorder=1.0,
            )
        )
        # goal
        ax.add_patch(
            Rectangle(
                (base, -GOAL_WIDTH / 2) if sign > 0 else (base - GOAL_DEPTH, -GOAL_WIDTH / 2),
                GOAL_DEPTH, GOAL_WIDTH,
                facecolor=palette.INK, edgecolor=style.line,
                lw=style.line_width, alpha=0.75, zorder=0.9,
            )
        )
        # corner arcs: the quarter that opens toward the middle of the pitch
        for y_sign in (-1, 1):
            theta1 = 0.0 if sign < 0 else 90.0
            if y_sign > 0:
                theta1 = 270.0 if sign < 0 else 180.0
            ax.add_patch(
                Arc(
                    (base, y_sign * half_w), 2 * CORNER_R, 2 * CORNER_R,
                    angle=0, theta1=theta1, theta2=theta1 + 90,
                    color=style.line, lw=style.line_width, alpha=style.line_alpha, zorder=1.0,
                )
            )


def attack_arrow(ax: Axes, length: float = 105.0, width: float = 68.0, alpha: float = 1.0) -> None:
    """A small 'attacking this way' cue in the bottom-left of the pitch."""

    if alpha <= 0.01:
        return
    y = -width / 2 + 3.0
    x0 = -length / 2 + 4.0
    ax.annotate(
        "",
        xy=(x0 + 11.0, y), xytext=(x0, y),
        arrowprops=dict(
            arrowstyle="-|>,head_width=0.22,head_length=0.5",
            color=palette.TEXT_SECONDARY, lw=1.3, alpha=0.7 * alpha,
        ),
        zorder=3.0,
    )
    ax.text(
        x0, y + 1.9, tracked("ATTACKING DIRECTION"),
        color=palette.TEXT_MUTED, fontsize=6.2, alpha=0.85 * alpha,
        weight="bold", zorder=3.0,
    )
