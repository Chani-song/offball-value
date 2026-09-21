"""Collision-aware label placement for the pitch layer.

Annotations on a tracking plot collide constantly: role chips, ghost captions,
field captions and distance readouts all want the same few pixels.  This module
keeps a list of occupied rectangles in *display* space and nudges each new label
to the first candidate offset that is free and still inside the axes, then draws
it with a short leader line back to its anchor.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from matplotlib.axes import Axes

from .. import palette

# candidate offsets, in points, tried in order
_DIRECTIONS = {
    "up": (0.0, 1.0),
    "right": (1.0, 0.25),
    "left": (-1.0, 0.25),
    "down": (0.0, -1.0),
    "up-right": (0.78, 0.78),
    "up-left": (-0.78, 0.78),
    "down-right": (0.78, -0.78),
    "down-left": (-0.78, -0.78),
}
_DISTANCES = (18.0, 30.0, 44.0, 60.0, 78.0)


@dataclass
class _Box:
    x0: float
    y0: float
    x1: float
    y1: float

    def overlaps(self, other: "_Box", pad: float = 4.0) -> bool:
        return not (
            self.x1 + pad < other.x0
            or other.x1 + pad < self.x0
            or self.y1 + pad < other.y0
            or other.y1 + pad < self.y0
        )


class LabelPlacer:
    """Greedy, deterministic label placement in display coordinates."""

    def __init__(self, ax: Axes):
        self.ax = ax
        self.boxes: list[_Box] = []
        self._dpi = ax.figure.dpi

    # -- occupancy ------------------------------------------------------
    def reserve_point(self, xy_data, radius_px: float) -> None:
        x, y = self.ax.transData.transform(xy_data)
        self.boxes.append(_Box(x - radius_px, y - radius_px, x + radius_px, y + radius_px))

    def reserve_points(self, points, radius_px: float) -> None:
        for point in points:
            if np.all(np.isfinite(point)):
                self.reserve_point(point, radius_px)

    # -- placement ------------------------------------------------------
    def place(
        self,
        xy_data,
        text: str,
        color: str,
        fontsize: float = 6.8,
        weight: str = "bold",
        alpha: float = 1.0,
        prefer: tuple[str, ...] = ("up", "up-right", "right", "up-left", "left", "down"),
        chip: bool = True,
        chip_face: str | None = None,
        text_color: str | None = None,
        leader: bool = True,
        clearance_px: float = 16.0,
        zorder: float = 6.0,
    ) -> bool:
        """Draw ``text`` near ``xy_data`` without overlapping anything placed before."""

        if not np.all(np.isfinite(np.asarray(xy_data, dtype=float))):
            return False
        anchor = np.asarray(self.ax.transData.transform(xy_data), dtype=float)
        scale = self._dpi / 72.0
        width = (len(text) * fontsize * 0.635 + 12.0) * scale
        height = (fontsize * 1.62 + 5.0) * scale

        axes_box = self.ax.get_window_extent()
        best = None
        for distance in _DISTANCES:
            for name in prefer:
                dx, dy = _DIRECTIONS[name]
                centre = anchor + np.array([dx, dy]) * (distance + clearance_px) * scale
                box = _Box(
                    centre[0] - width / 2, centre[1] - height / 2,
                    centre[0] + width / 2, centre[1] + height / 2,
                )
                if (
                    box.x0 < axes_box.x0 + 3 or box.x1 > axes_box.x1 - 3
                    or box.y0 < axes_box.y0 + 3 or box.y1 > axes_box.y1 - 3
                ):
                    continue
                if any(box.overlaps(other) for other in self.boxes):
                    continue
                best = (centre, box)
                break
            if best is not None:
                break
        if best is None:
            return False

        centre, box = best
        self.boxes.append(box)
        target = self.ax.transData.inverted().transform(centre)
        kwargs = dict(
            ha="center", va="center", fontsize=fontsize, weight=weight,
            alpha=alpha, zorder=zorder,
        )
        if chip:
            kwargs["color"] = text_color or palette.INK
            kwargs["bbox"] = dict(
                boxstyle="round,pad=0.30",
                facecolor=chip_face or color,
                edgecolor="none",
                alpha=alpha * 0.94,
            )
        else:
            kwargs["color"] = text_color or color
        if leader:
            self.ax.annotate(
                text, xy=tuple(xy_data), xytext=tuple(target),
                arrowprops=dict(arrowstyle="-", color=color, lw=0.9, alpha=alpha * 0.6,
                                shrinkA=0, shrinkB=2),
                **kwargs,
            )
        else:
            self.ax.text(target[0], target[1], text, **kwargs)
        return True
