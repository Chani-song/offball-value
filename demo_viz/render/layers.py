"""Story layers drawn on top of the pitch.

Each function takes a ``strength`` in ``[0, 1]`` so the storyboard can fade a
layer in and out instead of snapping it on.  Every layer is independent: a
scene that lacks the quantity a layer needs simply skips it.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.patches import Circle, FancyArrowPatch, PathPatch, Polygon
from matplotlib.path import Path as MplPath

from .. import palette
from ..scene import Scene
from .labels import LabelPlacer

PLAYER_R = 1.05
ROLE_R = 1.30
BALL_R = 0.46


def mark_scale(unit: float) -> float:
    """Marker radius factor for a viewport ``unit`` (view width / pitch length).

    Fully zoomed out the markers keep their true size; zoomed in they shrink
    a little so a close shot does not turn into a wall of discs.
    """

    return 0.52 + 0.48 * float(min(1.0, max(0.12, unit)))


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def fade_cmap(color: str, peak_alpha: float = 0.62, gamma: float = 1.35):
    """Transparent -> ``color`` colormap, for soft field overlays."""

    rgb = to_rgb(color)
    stops = np.linspace(0.0, 1.0, 64)
    colours = [(*rgb, peak_alpha * (stop ** gamma)) for stop in stops]
    return LinearSegmentedColormap.from_list("fade", colours)


def _glow(ax: Axes, xy, radius: float, color: str, alpha: float, rings: int = 4, zorder: float = 2.0):
    for index in range(rings, 0, -1):
        scale = 1.0 + 1.35 * index / rings
        ax.add_patch(
            Circle(
                xy, radius * scale,
                facecolor=color, edgecolor="none",
                alpha=alpha * (0.16 * (1.0 - (index - 1) / rings)),
                zorder=zorder,
            )
        )


def _view(scene: Scene, xy) -> np.ndarray:
    return scene.view_xy(np.asarray(xy, dtype=float))


def _finite(xy) -> bool:
    return bool(np.all(np.isfinite(np.asarray(xy, dtype=float))))


# ---------------------------------------------------------------------------
# A. base tracking
# ---------------------------------------------------------------------------
def draw_players(
    ax: Axes,
    scene: Scene,
    index: int,
    neutral_alpha: float = palette.NEUTRAL_ALPHA_FULL,
    role_strength: dict[str, float] | None = None,
    show_numbers: bool = True,
    label_roles: bool = True,
    unit: float = 1.0,
    placer: LabelPlacer | None = None,
) -> None:
    """All 22 players; the story triplet is emphasised by ``role_strength``."""

    role_strength = role_strength or {}
    for player in scene.players.values():
        xy = player.xy[index]
        if not _finite(xy):
            continue
        view = _view(scene, xy)
        role = scene.role_of(player.player_id)
        strength = role_strength.get(role, 0.0) if role else 0.0
        base = palette.ATTACK_NEUTRAL if player.side == "attack" else palette.DEFEND_NEUTRAL
        colour = palette.ROLE_COLOURS.get(role or "", base)
        # role players keep the team fill until the reveal, then take their hue
        fill = base if strength < 0.02 else _blend(base, colour, strength)
        alpha = neutral_alpha if not role else max(neutral_alpha, 0.35 + 0.65 * strength)
        radius = (PLAYER_R + (ROLE_R - PLAYER_R) * strength) * mark_scale(unit)

        if strength > 0.02:
            _glow(ax, view, radius, colour, strength, zorder=2.0)
        ax.add_patch(
            Circle(
                view, radius,
                facecolor=fill,
                edgecolor=colour if strength > 0.02 else palette.INK,
                lw=0.8 + 1.7 * strength,
                alpha=alpha,
                zorder=3.0 + strength,
            )
        )
        if player.is_goalkeeper:
            ax.add_patch(
                Circle(view, radius * 0.45, facecolor=palette.INK, edgecolor="none",
                       alpha=alpha * 0.55, zorder=3.05 + strength)
            )
        if show_numbers and player.shirt:
            ax.text(
                view[0], view[1], str(player.shirt),
                ha="center", va="center",
                fontsize=5.4 + 1.6 * strength,
                color=palette.INK if strength < 0.5 else palette.INK,
                weight="bold", alpha=alpha, zorder=3.4 + strength,
            )
        if label_roles and role and strength > 0.25 and placer is not None:
            placer.place(
                view,
                f"{palette.ROLE_LABELS[role]}  #{player.shirt}",
                palette.ROLE_COLOURS[role],
                fontsize=7.0, alpha=strength,
                prefer=_ROLE_PREFERENCE[role],
                clearance_px=10.0, zorder=6.5,
            )


_ROLE_PREFERENCE = {
    "runner": ("up", "up-right", "right", "up-left", "left", "down-right"),
    "defender": ("right", "up-right", "down-right", "up", "left", "down"),
    "beneficiary": ("left", "up-left", "down-left", "up", "right", "down"),
}


def _blend(first: str, second: str, weight: float) -> tuple[float, float, float]:
    a, b = np.asarray(to_rgb(first)), np.asarray(to_rgb(second))
    return tuple(a + (b - a) * min(1.0, max(0.0, weight)))  # type: ignore[return-value]


def draw_ball(ax: Axes, scene: Scene, index: int, alpha: float = 1.0, unit: float = 1.0) -> None:
    xy = scene.ball_xy[index]
    if not _finite(xy):
        return
    view = _view(scene, xy)
    radius = BALL_R * mark_scale(unit)
    _glow(ax, view, radius, palette.BALL, 0.55 * alpha, rings=3, zorder=4.4)
    ax.add_patch(
        Circle(view, radius, facecolor=palette.BALL, edgecolor=palette.BALL_EDGE,
               lw=0.9, alpha=alpha, zorder=4.6)
    )


def draw_ball_trail(ax: Axes, scene: Scene, index: int, seconds: float = 1.2, alpha: float = 0.7) -> None:
    span = max(2, int(round(seconds * scene.fps)))
    start = max(0, index - span)
    track = scene.view_xy(scene.ball_xy[start : index + 1])
    mask = np.all(np.isfinite(track), axis=1)
    track = track[mask]
    if len(track) < 2:
        return
    segments = np.stack([track[:-1], track[1:]], axis=1)
    weights = np.linspace(0.0, 1.0, len(segments)) ** 1.6
    ax.add_collection(
        LineCollection(
            segments, colors=[(*to_rgb(palette.BALL), alpha * w) for w in weights],
            linewidths=0.6 + 1.1 * weights, capstyle="round", zorder=4.2,
        )
    )


# ---------------------------------------------------------------------------
# C. runner trail
# ---------------------------------------------------------------------------
def draw_trail(
    ax: Axes,
    scene: Scene,
    player_id: str,
    index: int,
    seconds: float = 1.8,
    color: str = palette.RUNNER,
    strength: float = 1.0,
    width: float = 4.2,
) -> None:
    """A glowing, smoothly fading trail behind one player."""

    if strength <= 0.01 or player_id not in scene.players:
        return
    span = max(2, int(round(seconds * scene.fps)))
    start = max(0, index - span)
    track = scene.view_xy(scene.players[player_id].xy[start : index + 1])
    mask = np.all(np.isfinite(track), axis=1)
    track = track[mask]
    if len(track) < 2:
        return
    segments = np.stack([track[:-1], track[1:]], axis=1)
    weights = np.linspace(0.0, 1.0, len(segments)) ** 1.7
    rgb = to_rgb(color)
    # wide soft glow underneath
    ax.add_collection(
        LineCollection(
            segments, colors=[(*rgb, 0.16 * strength * w) for w in weights],
            linewidths=width * 3.1 * (0.4 + 0.6 * weights), capstyle="round", zorder=1.6,
        )
    )
    ax.add_collection(
        LineCollection(
            segments, colors=[(*rgb, 0.92 * strength * w) for w in weights],
            linewidths=width * (0.28 + 0.72 * weights), capstyle="round", zorder=1.8,
        )
    )


# ---------------------------------------------------------------------------
# D. defender tether
# ---------------------------------------------------------------------------
def draw_tether(
    ax: Axes,
    scene: Scene,
    runner_id: str,
    defender_id: str,
    index: int,
    strength: float = 1.0,
    phase: float = 0.0,
    show_distance: bool = True,
    unit: float = 1.0,
    placer: LabelPlacer | None = None,
) -> None:
    """A subtle bowed link implying the defender is being carried by the runner."""

    if strength <= 0.01:
        return
    if runner_id not in scene.players or defender_id not in scene.players:
        return
    a = scene.players[runner_id].xy[index]
    b = scene.players[defender_id].xy[index]
    if not (_finite(a) and _finite(b)):
        return
    pa, pb = _view(scene, a), _view(scene, b)
    delta = pb - pa
    distance = float(np.linalg.norm(delta))
    if distance < 1e-6:
        return
    normal = np.array([-delta[1], delta[0]]) / distance
    bow = normal * min(2.6 * unit + 0.6, 0.13 * distance)
    control = (pa + pb) / 2.0 + bow
    path = MplPath([tuple(pa), tuple(control), tuple(pb)],
                   [MplPath.MOVETO, MplPath.CURVE3, MplPath.CURVE3])
    ax.add_patch(
        PathPatch(path, facecolor="none", edgecolor=palette.DEFENDER,
                  lw=5.0, alpha=0.10 * strength, capstyle="round", zorder=1.9)
    )
    ax.add_patch(
        PathPatch(path, facecolor="none", edgecolor=palette.DEFENDER,
                  lw=1.5, alpha=0.78 * strength, linestyle=(phase, (4.5, 3.0)),
                  capstyle="round", zorder=2.0)
    )
    if show_distance and strength > 0.4 and placer is not None:
        mid = (pa + pb) / 2.0 + bow * 1.45
        placer.place(
            mid, f"{distance:.1f} m", palette.DEFENDER,
            fontsize=6.4, alpha=strength, chip=True,
            chip_face=palette.INK, text_color=palette.DEFENDER,
            leader=False, clearance_px=0.0, zorder=5.0,
            prefer=("up", "down", "right", "left"),
        )


# ---------------------------------------------------------------------------
# E. space wake
# ---------------------------------------------------------------------------
def draw_space_field(
    ax: Axes,
    scene: Scene,
    index: int,
    strength: float = 1.0,
    mode: str = "delta",
    label: bool = False,
    placer: LabelPlacer | None = None,
) -> str | None:
    """Filled contours of the opened space.

    ``mode='delta'``     factual minus held-defender counterfactual (both from
                         ``goal_weighted_influence.target_residual_influence``).
    ``mode='residual'``  the factual residual space itself.
    Returns the caption describing what was drawn, or ``None`` if nothing was.
    """

    stack = scene.surfaces
    if stack is None or strength <= 0.01:
        return None
    field = stack.delta_at(index) if mode == "delta" else stack.at(index)
    if field is None:
        field = stack.at(index)
        mode = "residual"
    if scene.flip:
        field = field[::-1, ::-1]
    peak = float(np.nanmax(field))
    if not np.isfinite(peak) or peak <= 1e-9:
        return None

    xgrid, ygrid = stack.xgrid, stack.ygrid
    colour = palette.SPACE_WAKE
    cut = peak * 0.10
    # an interpolated image reads as a soft field; stacked contour bands read as
    # onion rings, which is the wrong visual metaphor for "open space"
    ax.imshow(
        np.clip((field - cut) / max(peak - cut, 1e-9), 0.0, 1.0),
        extent=(float(xgrid[0]), float(xgrid[-1]), float(ygrid[0]), float(ygrid[-1])),
        origin="lower", cmap=fade_cmap(colour, peak_alpha=0.66 * strength, gamma=1.15),
        vmin=0.0, vmax=1.0, interpolation="bicubic", aspect="auto", zorder=1.2,
    )
    if label and placer is not None and strength > 0.35:
        centroid = _field_centroid(xgrid, ygrid, field, peak * 0.55)
        if centroid is not None:
            placer.place(
                centroid, "SPACE OPENED" if mode == "delta" else "OPEN SPACE",
                colour, fontsize=6.6, alpha=strength, chip=True,
                chip_face=palette.INK, text_color=colour, leader=False,
                clearance_px=0.0, zorder=5.2,
            )
    return mode


def draw_delta_outline(
    ax: Axes,
    scene: Scene,
    index: int,
    strength: float = 1.0,
    label: bool = True,
    placer: LabelPlacer | None = None,
    outline: bool = True,
) -> bool:
    """Outline the region that exists *only* because the defender moved.

    This is ``factual - counterfactual`` of the repository's residual space
    formula, drawn as a boundary rather than a second heatmap so the frame does
    not carry two competing fields.
    """

    stack = scene.surfaces
    if stack is None or strength <= 0.01:
        return False
    delta = stack.delta_at(index)
    if delta is None:
        return False
    if scene.flip:
        delta = delta[::-1, ::-1]
    peak = float(np.nanmax(delta))
    if not np.isfinite(peak) or peak <= 1e-6:
        return False
    level = peak * 0.42
    if not outline:
        if label and strength > 0.45 and placer is not None:
            centroid = _field_centroid(stack.xgrid, stack.ygrid, delta, level)
            if centroid is not None:
                placer.place(
                    centroid, "SPACE THE DEFENDER LEFT BEHIND", palette.DELTA_POS,
                    fontsize=6.4, alpha=strength, chip=True,
                    chip_face=palette.INK, text_color=palette.DELTA_POS, leader=True,
                    prefer=("up", "up-left", "left", "up-right", "down"),
                    clearance_px=4.0, zorder=5.3,
                )
        return True
    # upsample before contouring: a 1.5 m grid contours into visible polygons
    xs, ys, smooth = _upsample(stack.xgrid, stack.ygrid, delta)
    ax.contourf(
        xs, ys, smooth, levels=[level, peak * 10],
        colors=[palette.DELTA_POS], alpha=0.18 * strength, zorder=1.34,
    )
    ax.contour(
        xs, ys, smooth, levels=[level],
        colors=[palette.DELTA_POS], linewidths=1.5, alpha=0.9 * strength, zorder=1.36,
    )
    if label and strength > 0.45 and placer is not None:
        centroid = _field_centroid(stack.xgrid, stack.ygrid, delta, level)
        if centroid is not None:
            placer.place(
                centroid, "SPACE THE DEFENDER LEFT BEHIND", palette.DELTA_POS,
                fontsize=6.4, alpha=strength, chip=True,
                chip_face=palette.INK, text_color=palette.DELTA_POS, leader=True,
                prefer=("up", "up-left", "left", "up-right", "down"),
                clearance_px=4.0, zorder=5.3,
            )
    return True


def _upsample(xgrid, ygrid, field, factor: int = 4):
    """Smoothly resample a coarse pitch grid so contours read as curves."""

    try:
        from scipy.ndimage import zoom
    except Exception:                                    # pragma: no cover
        return xgrid, ygrid, field
    smooth = zoom(np.asarray(field, dtype=float), factor, order=3, mode="nearest")
    xs = np.linspace(float(xgrid[0]), float(xgrid[-1]), smooth.shape[1])
    ys = np.linspace(float(ygrid[0]), float(ygrid[-1]), smooth.shape[0])
    return xs, ys, smooth


def _field_centroid(xgrid, ygrid, field, threshold: float):
    mask = field >= threshold
    if not mask.any():
        return None
    xx, yy = np.meshgrid(xgrid, ygrid)
    weights = np.where(mask, field, 0.0)
    total = weights.sum()
    if total <= 0:
        return None
    return float((xx * weights).sum() / total), float((yy * weights).sum() / total)


def draw_geometric_wake(
    ax: Axes,
    scene: Scene,
    index: int,
    strength: float = 1.0,
    freeze_index: int | None = None,
) -> None:
    """Purely geometric 'vacated corridor' aid, used when no surface exists.

    This is an explanatory shape, not a measured quantity; the caller is
    responsible for labelling it as illustrative.
    """

    if strength <= 0.01 or not scene.defender_ids:
        return
    freeze_index = 0 if freeze_index is None else freeze_index
    for defender_id in scene.defender_ids:
        track = scene.players[defender_id].xy
        start, now = track[freeze_index], track[index]
        if not (_finite(start) and _finite(now)):
            continue
        p0, p1 = _view(scene, start), _view(scene, now)
        delta = p1 - p0
        distance = float(np.linalg.norm(delta))
        if distance < 1.5:
            continue
        unit = delta / distance
        normal = np.array([-unit[1], unit[0]])
        half_start, half_end = 2.2, 2.2 + 0.42 * distance
        polygon = np.array([
            p0 + normal * half_start, p1 + normal * half_end,
            p1 - normal * half_end, p0 - normal * half_start,
        ])
        for scale, alpha in ((1.35, 0.07), (1.0, 0.16)):
            ax.add_patch(
                Polygon(
                    (polygon - polygon.mean(axis=0)) * scale + polygon.mean(axis=0),
                    closed=True, facecolor=palette.SPACE_WAKE, edgecolor="none",
                    alpha=alpha * strength, zorder=1.15,
                )
            )


# ---------------------------------------------------------------------------
# G. counterfactual ghost
# ---------------------------------------------------------------------------
def _baseline_label(scene: Scene) -> str:
    stack = scene.surfaces
    mode = getattr(stack, "baseline_mode", "hold") if stack is not None else "hold"
    return "IF HELD" if mode == "hold" else "IF NO REACTION"


def draw_ghost(
    ax: Axes,
    scene: Scene,
    index: int,
    strength: float = 1.0,
    freeze_index: int | None = None,
    label: bool = True,
    unit: float = 1.0,
    placer: LabelPlacer | None = None,
    marker: bool = True,
) -> None:
    """Translucent 'if the defender had held position' marker plus displacement."""

    if strength <= 0.01 or not scene.defender_ids:
        return
    if freeze_index is None:
        stack = scene.surfaces
        freeze_index = getattr(stack, "freeze_index", None) if stack is not None else None
    if freeze_index is None:
        return
    stack = scene.surfaces
    baselines = getattr(stack, "baseline_tracks", {}) if stack is not None else {}
    for defender_id in scene.defender_ids:
        track = scene.players[defender_id].xy
        if freeze_index >= len(track):
            continue
        baseline = baselines.get(defender_id)
        held = baseline[index] if baseline is not None else track[freeze_index]
        now = track[index]
        if not (_finite(held) and _finite(now)):
            continue
        p_held, p_now = _view(scene, held), _view(scene, now)
        radius = ROLE_R * mark_scale(unit)
        if not marker:
            shift = float(np.linalg.norm(p_now - p_held))
            if label and strength > 0.4 and placer is not None:
                placer.place(
                    p_held, f"{_baseline_label(scene)} · pulled {shift:.1f} m", palette.GHOST,
                    fontsize=6.4, alpha=0.92 * strength, chip=True,
                    chip_face=palette.INK, text_color=palette.GHOST,
                    prefer=("down-left", "left", "down", "up-left", "down-right"),
                    clearance_px=8.0, zorder=5.1,
                )
            continue
        ax.add_patch(
            Circle(p_held, radius, facecolor="none", edgecolor=palette.GHOST,
                   lw=1.5, linestyle=(0, (2.2, 2.2)), alpha=0.72 * strength, zorder=2.6)
        )
        ax.add_patch(
            Circle(p_held, radius, facecolor=palette.GHOST, edgecolor="none",
                   alpha=0.12 * strength, zorder=2.5)
        )
        shift = float(np.linalg.norm(p_now - p_held))
        if shift > 1.2:
            ax.add_patch(
                FancyArrowPatch(
                    tuple(p_held), tuple(p_now),
                    arrowstyle="-|>,head_width=0.18,head_length=0.42",
                    color=palette.GHOST, lw=1.2, alpha=0.6 * strength,
                    linestyle=(0, (1.4, 2.0)), shrinkA=8, shrinkB=10, zorder=2.55,
                )
            )
        if label and strength > 0.4 and placer is not None:
            placer.place(
                p_held, f"{_baseline_label(scene)} · pulled {shift:.1f} m", palette.GHOST,
                fontsize=6.4, alpha=0.92 * strength, chip=True,
                chip_face=palette.INK, text_color=palette.GHOST,
                prefer=("down-left", "left", "down", "up-left", "down-right"),
                clearance_px=8.0, zorder=5.1,
            )


# ---------------------------------------------------------------------------
# F. beneficiary reveal
# ---------------------------------------------------------------------------
def draw_beneficiary_lane(
    ax: Axes,
    scene: Scene,
    index: int,
    strength: float = 1.0,
    label: bool = True,
    unit: float = 1.0,
    maximum_distance_m: float = 34.0,
    placer: LabelPlacer | None = None,
    shapes: bool = True,
) -> None:
    """The geometric lane from the ball to the beneficiary.

    Drawn only when the beneficiary is actually within passing range; a 50 m
    'lane' would be a graphic pretending to be an option.
    """

    if strength <= 0.01 or not scene.beneficiary_ids:
        return
    ball = scene.ball_xy[index]
    if not _finite(ball):
        return
    origin = _view(scene, ball)
    for beneficiary_id in scene.beneficiary_ids:
        target_xy = scene.players[beneficiary_id].xy[index]
        if not _finite(target_xy):
            continue
        target = _view(scene, target_xy)
        delta = target - origin
        distance = float(np.linalg.norm(delta))
        if distance < 2.0 or distance > maximum_distance_m:
            continue
        direction = delta / distance
        normal = np.array([-direction[1], direction[0]])
        half = 1.15 * mark_scale(unit)
        if not shapes:
            if label and strength > 0.4 and placer is not None:
                mid = origin + delta * 0.52 + normal * (half + 1.6 * unit)
                placer.place(
                    mid, f"open lane  {distance:.0f} m", palette.BENEFICIARY,
                    fontsize=6.2, alpha=strength, chip=True,
                    chip_face=palette.INK, text_color=palette.BENEFICIARY, leader=False,
                    clearance_px=0.0, zorder=5.0,
                )
            continue
        polygon = np.array([
            origin + normal * half * 0.45, target + normal * half,
            target - normal * half, origin - normal * half * 0.45,
        ])
        ax.add_patch(
            Polygon(polygon, closed=True, facecolor=palette.BENEFICIARY,
                    edgecolor="none", alpha=0.20 * strength, zorder=1.45)
        )
        ax.add_patch(
            FancyArrowPatch(
                tuple(origin), tuple(target),
                arrowstyle="-|>,head_width=0.22,head_length=0.5",
                color=palette.BENEFICIARY, lw=1.5, alpha=0.85 * strength,
                shrinkA=4, shrinkB=12, zorder=2.2,
            )
        )
        # pulsing reveal ring
        pulse = 1.0 + 0.45 * math.sin(index * 0.24)
        ax.add_patch(
            Circle(target, ROLE_R * mark_scale(unit) * (1.55 + 0.32 * pulse), facecolor="none",
                   edgecolor=palette.BENEFICIARY, lw=1.3,
                   alpha=0.45 * strength * (1.15 - 0.3 * pulse), zorder=2.3)
        )
        if label and strength > 0.4 and placer is not None:
            mid = origin + delta * 0.52 + normal * (half + 1.6 * unit)
            placer.place(
                mid, f"open lane  {distance:.0f} m", palette.BENEFICIARY,
                fontsize=6.2, alpha=strength, chip=True,
                chip_face=palette.INK, text_color=palette.BENEFICIARY, leader=False,
                clearance_px=0.0, zorder=5.0,
            )


def draw_future_path(
    ax: Axes,
    scene: Scene,
    player_id: str,
    index: int,
    seconds: float = 2.0,
    color: str = palette.BENEFICIARY,
    strength: float = 1.0,
) -> None:
    """Faint forward look at where a highlighted player actually goes next."""

    if strength <= 0.01 or player_id not in scene.players:
        return
    span = max(2, int(round(seconds * scene.fps)))
    end = min(scene.n_frames, index + span)
    track = scene.view_xy(scene.players[player_id].xy[index:end])
    mask = np.all(np.isfinite(track), axis=1)
    track = track[mask]
    if len(track) < 2:
        return
    segments = np.stack([track[:-1], track[1:]], axis=1)
    weights = np.linspace(1.0, 0.0, len(segments)) ** 1.4
    ax.add_collection(
        LineCollection(
            segments, colors=[(*to_rgb(color), 0.40 * strength * w) for w in weights],
            linewidths=1.5, linestyle="dotted", capstyle="round", zorder=1.7,
        )
    )


# ---------------------------------------------------------------------------
# run vector: make "the runner moved" impossible to miss
# ---------------------------------------------------------------------------
def draw_run_vector(
    ax: Axes,
    scene: Scene,
    player_id: str,
    index: int,
    onset_index: int,
    strength: float = 1.0,
    color: str = palette.RUNNER,
    unit: float = 1.0,
    placer: LabelPlacer | None = None,
    label: str = "ran",
    minimum_m: float = 3.0,
) -> None:
    """Origin marker, travelled path and displacement for one player.

    The 1-2 s glow trail shows *speed*; this shows the whole run since the beat
    started, which is what the causal claim is actually about.
    """

    if strength <= 0.01 or player_id not in scene.players:
        return
    onset_index = int(max(0, min(onset_index, scene.n_frames - 1)))
    if index <= onset_index:
        return
    track = scene.view_xy(scene.players[player_id].xy[onset_index : index + 1])
    mask = np.all(np.isfinite(track), axis=1)
    track = track[mask]
    if len(track) < 2:
        return
    origin, now = track[0], track[-1]
    travelled = float(np.sum(np.linalg.norm(np.diff(track, axis=0), axis=1)))
    if travelled < minimum_m:
        return

    if placer is not None:
        if strength > 0.4:
            placer.place(
                origin, f"{label} {travelled:.0f} m from here", color,
                fontsize=6.3, alpha=0.95 * strength, chip=True,
                chip_face=palette.INK, text_color=color,
                prefer=("down", "down-left", "left", "down-right", "up-left"),
                clearance_px=6.0, zorder=5.4,
            )
        return

    rgb = to_rgb(color)
    ax.plot(track[:, 0], track[:, 1], color=rgb, lw=1.4, alpha=0.40 * strength,
            linestyle=(0, (1.2, 2.4)), zorder=1.55, solid_capstyle="round")
    radius = PLAYER_R * mark_scale(unit)
    ax.add_patch(
        Circle(origin, radius * 0.92, facecolor="none", edgecolor=rgb,
               lw=1.3, alpha=0.62 * strength, linestyle=(0, (2.0, 2.0)), zorder=1.6)
    )
    ax.add_patch(
        Circle(origin, radius * 0.92, facecolor=rgb, edgecolor="none",
               alpha=0.10 * strength, zorder=1.58)
    )
    if float(np.linalg.norm(now - origin)) > radius * 2.6:
        ax.add_patch(
            FancyArrowPatch(
                tuple(origin), tuple(now),
                arrowstyle="-|>,head_width=0.20,head_length=0.46",
                color=rgb, lw=1.3, alpha=0.48 * strength,
                shrinkA=radius * 4, shrinkB=radius * 9, zorder=1.62,
            )
        )


def draw_gain_badge(
    ax: Axes,
    scene: Scene,
    index: int,
    value: float,
    strength: float = 1.0,
    unit: float = 1.0,
    placer: LabelPlacer | None = None,
) -> None:
    """A ring plus a signed badge on the beneficiary at the payoff beat."""

    if strength <= 0.01 or not scene.beneficiary_ids:
        return
    for beneficiary_id in scene.beneficiary_ids:
        xy = scene.players[beneficiary_id].xy[index]
        if not _finite(xy):
            continue
        view = _view(scene, xy)
        radius = ROLE_R * mark_scale(unit)
        for scale, alpha in ((2.5, 0.10), (1.95, 0.16), (1.5, 0.22)):
            ax.add_patch(
                Circle(view, radius * scale, facecolor="none",
                       edgecolor=palette.BENEFICIARY, lw=1.1,
                       alpha=alpha * strength, zorder=2.35)
            )
        if placer is not None and strength > 0.45:
            placer.place(
                view, f"{value:+.1f} space vs held defender", palette.BENEFICIARY,
                fontsize=6.6, alpha=strength, chip=True,
                chip_face=palette.BENEFICIARY, text_color=palette.INK,
                prefer=("down-left", "down", "left", "down-right"),
                clearance_px=radius * 2.4, zorder=6.6,
            )


# ---------------------------------------------------------------------------
# the exploit: where the annotated shot was taken
# ---------------------------------------------------------------------------
def draw_shot_marker(
    ax: Axes,
    scene: Scene,
    index: int,
    strength: float = 1.0,
    unit: float = 1.0,
    placer: LabelPlacer | None = None,
    window_s: float = 1.2,
) -> None:
    """Mark the annotated shot while the clock is near it.

    The window's zero is the shot the annotator reviewed, so this is the moment
    the sequence was selected for. It is drawn as a transient cue rather than a
    permanent one, because for several of these scenes the measured payoff for
    the beneficiary arrives *after* the shot rather than at it.
    """

    if strength <= 0.01 or "shot" not in scene.moments:
        return
    t_shot = float(scene.moments["shot"])
    delta = abs(float(scene.times[index]) - t_shot)
    if delta > window_s:
        return
    fade = strength * (1.0 - (delta / window_s) ** 2)
    shot_index = scene.index_at(t_shot)
    xy = scene.ball_xy[shot_index]
    if not _finite(xy):
        return
    view = _view(scene, xy)
    radius = ROLE_R * mark_scale(unit)
    grow = 1.0 + 2.6 * (delta / window_s)
    ax.add_patch(
        Circle(view, radius * 1.7 * grow, facecolor="none", edgecolor=palette.BALL,
               lw=1.4, alpha=0.55 * fade, zorder=4.7)
    )
    ax.add_patch(
        Circle(view, radius * 0.8, facecolor=palette.BALL, edgecolor="none",
               alpha=0.20 * fade, zorder=4.65)
    )
    if placer is not None and fade > 0.35:
        placer.place(
            view, "SHOT", palette.BALL,
            fontsize=6.6, alpha=fade, chip=True,
            chip_face=palette.BALL, text_color=palette.INK, leader=False,
            prefer=("up-right", "right", "up", "down-right", "up-left"),
            clearance_px=radius * 2.0, zorder=6.8,
        )
