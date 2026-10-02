"""A smooth 'broadcast camera' over the pitch.

A full-pitch view wastes most of the frame: an off-ball sequence typically
lives inside a 40 x 30 m box.  This module precomputes a per-frame viewport
that follows the cast and the ball, smooths it so the camera glides instead of
jittering, and blends between the establishing wide shot and the close shot as
the story starts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..scene import Scene


@dataclass(frozen=True)
class CameraConfig:
    """How tightly and how smoothly the viewport follows the cast."""

    margin_m: float = 8.0
    minimum_width_m: float = 44.0
    maximum_width_m: float = 80.0
    smooth_seconds: float = 1.6
    include_ball: bool = True
    ball_attach_m: float = 16.0     # ignore the ball once it is this far from the cast
    include_all_players: bool = False
    lead_seconds: float = 0.6       # look slightly ahead so motion is not cut off


class Camera:
    """Per-frame ``(x0, x1, y0, y1)`` viewports in *view* coordinates."""

    def __init__(self, scene: Scene, aspect: float, config: CameraConfig = CameraConfig()):
        self.scene = scene
        self.aspect = float(aspect)          # width / height of the pitch axes
        self.config = config
        self._close = self._build_close_track()
        self._wide = self._wide_view()

    # ------------------------------------------------------------------
    def _subject_points(self) -> np.ndarray:
        """Cast tracks, plus the ball only while it is near the cast.

        A ball that has been cleared to the far corner would otherwise drag the
        viewport back out to a full-pitch shot for no narrative gain.
        """

        scene, config = self.scene, self.config
        tracks: list[np.ndarray] = []
        ids = list(scene.players) if config.include_all_players else list(scene.role_ids)
        for player_id in ids:
            player = scene.players.get(player_id)
            if player is not None:
                tracks.append(scene.view_xy(player.xy))
        if not tracks:
            tracks.append(np.zeros((scene.n_frames, 2)))
        stack = np.stack(tracks, axis=0)
        if config.include_ball:
            ball = scene.view_xy(scene.ball_xy)
            centre = np.nanmean(stack, axis=0)
            far = np.linalg.norm(ball - centre, axis=1) > config.ball_attach_m
            ball = ball.copy()
            ball[far] = np.nan
            stack = np.concatenate([stack, ball[None, :, :]], axis=0)
        return stack

    def _build_close_track(self) -> np.ndarray:
        scene, config = self.scene, self.config
        points = self._subject_points()
        lead = max(0, int(round(config.lead_seconds * scene.fps)))

        with np.errstate(invalid="ignore"):
            lo = np.nanmin(points, axis=0)   # (T, 2)
            hi = np.nanmax(points, axis=0)
        lo = np.where(np.isfinite(lo), lo, 0.0)
        hi = np.where(np.isfinite(hi), hi, 0.0)
        if lead:
            lo = np.minimum(lo, np.roll(lo, -lead, axis=0))
            hi = np.maximum(hi, np.roll(hi, -lead, axis=0))
            lo[-lead:] = lo[-lead - 1]
            hi[-lead:] = hi[-lead - 1]

        centre = (lo + hi) / 2.0
        extent = (hi - lo) + 2.0 * config.margin_m
        width = np.maximum(extent[:, 0], extent[:, 1] * self.aspect)
        width = np.clip(width, config.minimum_width_m, config.maximum_width_m)

        window = max(1, int(round(config.smooth_seconds * scene.fps)))
        centre = np.stack([_smooth(centre[:, 0], window), _smooth(centre[:, 1], window)], axis=1)
        width = _smooth(width, window)
        # keep the viewport inside the pitch (plus a little bleed)
        bleed = 4.0
        half_l = scene.pitch_length / 2.0 + bleed
        half_w = scene.pitch_width / 2.0 + bleed
        width = np.minimum(width, 2 * half_l)
        height = width / self.aspect
        height = np.minimum(height, 2 * half_w)
        width = height * self.aspect
        centre[:, 0] = np.clip(centre[:, 0], -half_l + width / 2, half_l - width / 2)
        centre[:, 1] = np.clip(centre[:, 1], -half_w + height / 2, half_w - height / 2)
        return np.stack([centre[:, 0], centre[:, 1], width], axis=1)

    def _wide_view(self) -> np.ndarray:
        scene = self.scene
        width = scene.pitch_length + 2 * 3.2
        height = width / self.aspect
        if height < scene.pitch_width + 2 * 3.2:
            height = scene.pitch_width + 2 * 3.2
            width = height * self.aspect
        return np.array([0.0, 0.0, width])

    # ------------------------------------------------------------------
    def viewport(self, index: int, zoom: float = 1.0) -> tuple[float, float, float, float]:
        """Blend wide (``zoom=0``) to close (``zoom=1``) and return axis limits."""

        zoom = float(min(1.0, max(0.0, zoom)))
        close = self._close[int(np.clip(index, 0, len(self._close) - 1))]
        wide = self._wide
        # blend in log-width so the push-in feels linear on screen
        width = float(np.exp(np.log(wide[2]) + (np.log(close[2]) - np.log(wide[2])) * zoom))
        cx = float(wide[0] + (close[0] - wide[0]) * zoom)
        cy = float(wide[1] + (close[1] - wide[1]) * zoom)
        height = width / self.aspect
        return cx - width / 2, cx + width / 2, cy - height / 2, cy + height / 2

    def scale(self, index: int, zoom: float = 1.0) -> float:
        """How much bigger things look than in the wide shot (>= 1)."""

        x0, x1, _, _ = self.viewport(index, zoom)
        return float(self._wide[2] / max(x1 - x0, 1e-6))


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(values) < 3:
        return np.asarray(values, dtype=float)
    kernel = np.ones(window) / window
    padded = np.pad(np.asarray(values, dtype=float), (window, window), mode="edge")
    smooth = np.convolve(padded, kernel, mode="same")[window:-window]
    # a second pass makes the motion feel like a camera operator, not a filter
    padded = np.pad(smooth, (window, window), mode="edge")
    return np.convolve(padded, kernel, mode="same")[window:-window]


def focus_box(
    scene: Scene,
    index: int,
    player_ids,
    aspect: float,
    margin_m: float = 9.0,
    minimum_width_m: float = 27.0,
) -> tuple[float, float, float, float]:
    """A close viewport around ``player_ids`` at one frame, in view coordinates.

    Used by the role introduction at the head of a demo video: the point is to
    say *which players these are*, so the box is tight and the ball is ignored.
    Several players spread across the pitch widen the box rather than splitting
    it, which keeps the move a simple push-in instead of a cut.
    """

    points = [
        scene.view_xy(scene.players[pid].xy[index])
        for pid in player_ids
        if pid in scene.players
    ]
    if not points:
        return Camera(scene, aspect).viewport(index, 1.0)

    array = np.asarray(points, dtype=float)
    lo, hi = array.min(axis=0), array.max(axis=0)
    centre = (lo + hi) / 2.0
    extent = (hi - lo) + 2.0 * margin_m
    width = max(float(extent[0]), float(extent[1]) * aspect, float(minimum_width_m))

    bleed = 4.0
    half_l = scene.pitch_length / 2.0 + bleed
    half_w = scene.pitch_width / 2.0 + bleed
    width = min(width, 2 * half_l)
    height = min(width / aspect, 2 * half_w)
    width = height * aspect
    cx = float(np.clip(centre[0], -half_l + width / 2, half_l - width / 2))
    cy = float(np.clip(centre[1], -half_w + height / 2, half_w - height / 2))
    return cx - width / 2, cx + width / 2, cy - height / 2, cy + height / 2


def blend_boxes(first, second, weight: float) -> tuple[float, float, float, float]:
    """Ease between two viewports; widths blend in log space, as in ``Camera``."""

    weight = float(min(1.0, max(0.0, weight)))
    weight = weight * weight * (3.0 - 2.0 * weight)
    ax0, ax1, ay0, ay1 = first
    bx0, bx1, by0, by1 = second
    a_cx, a_cy, a_w = (ax0 + ax1) / 2, (ay0 + ay1) / 2, ax1 - ax0
    b_cx, b_cy, b_w = (bx0 + bx1) / 2, (by0 + by1) / 2, bx1 - bx0
    width = float(np.exp(np.log(a_w) + (np.log(b_w) - np.log(a_w)) * weight))
    aspect = a_w / max(ay1 - ay0, 1e-6)
    height = width / aspect
    cx = a_cx + (b_cx - a_cx) * weight
    cy = a_cy + (b_cy - a_cy) * weight
    return cx - width / 2, cx + width / 2, cy - height / 2, cy + height / 2
