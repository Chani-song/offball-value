"""Composition of the full demo frame: pitch, chain rail, side panel, timeline.

One :class:`SceneFigure` owns the matplotlib figure for a whole render.  The
pitch layer is redrawn per frame (cheap, and far easier to reason about than
artist bookkeeping); the header is drawn once.
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass, field

import matplotlib
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch, Rectangle

from .. import palette
from ..scene import Scene
from ..story import CHAIN, Storyboard
from . import layers as L
from .camera import Camera, CameraConfig
from .labels import LabelPlacer
from .pitch import PitchStyle, attack_arrow, draw_pitch, tracked

def _unicode_font() -> str | None:
    """A locally installed font that can render the annotators' notes.

    The notes in ``shot_annotations.xlsx`` are written in Korean; DejaVu Sans
    has no Hangul coverage, so the note block is simply skipped when no
    suitable font is present rather than rendering a row of tofu boxes.
    """

    from matplotlib import font_manager

    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Apple SD Gothic Neo", "Nanum Gothic", "NanumGothic", "Noto Sans CJK KR",
                 "Noto Sans KR", "Arial Unicode MS", "PingFang SC", "Hiragino Sans"):
        if name in available:
            return name
    return None


UNICODE_FONT = _unicode_font()


matplotlib.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "figure.facecolor": palette.INK,
        "savefig.facecolor": palette.INK,
        "axes.facecolor": palette.INK,
        "text.color": palette.TEXT_PRIMARY,
        "axes.edgecolor": palette.GRID,
        "axes.labelcolor": palette.TEXT_SECONDARY,
        "xtick.color": palette.TEXT_MUTED,
        "ytick.color": palette.TEXT_MUTED,
        "path.simplify": True,
    }
)


@dataclass
class FigureConfig:
    """Frame size and which optional panels and layers are drawn."""

    width_px: int = 1920
    height_px: int = 1080
    dpi: int = 120
    pitch_style: PitchStyle = field(default_factory=PitchStyle)
    camera: CameraConfig = field(default_factory=CameraConfig)
    wake_mode: str = "residual"       # "residual" | "delta" | "geometric" | "off"
    delta_outline: bool = True        # outline the factual-minus-held difference
    show_chart: bool = True
    show_numbers: bool = True
    use_camera: bool = True
    show_minimap: bool = True


class SceneFigure:
    """The whole demo frame. Call :meth:`draw` once per rendered frame."""

    def __init__(self, scene: Scene, storyboard: Storyboard, config: FigureConfig | None = None):
        self.scene = scene
        self.storyboard = storyboard
        self.config = config or FigureConfig()
        config = self.config

        width_in = config.width_px / config.dpi
        height_in = config.height_px / config.dpi
        self.fig = Figure(figsize=(width_in, height_in), dpi=config.dpi, facecolor=palette.INK)

        # --- geometry -----------------------------------------------------
        top, bottom, left = 0.836, 0.112, 0.028
        height = top - bottom
        pitch_ratio = 16.0 / 10.0 if config.use_camera else (
            (scene.pitch_length + 2 * config.pitch_style.margin_m)
            / (scene.pitch_width + 2 * config.pitch_style.margin_m)
        )
        pitch_width_frac = height * (config.height_px / config.width_px) * pitch_ratio
        self.ax_pitch = self.fig.add_axes((left, bottom, pitch_width_frac, height))
        self.aspect = pitch_ratio

        rail_left = left + pitch_width_frac + 0.030
        rail_width = max(0.15, 0.974 - rail_left)
        chart_height = 0.172 if config.show_chart else 0.0
        chart_bottom = bottom + 0.055
        panel_bottom = chart_bottom + chart_height + 0.076
        self.ax_panel = self.fig.add_axes(
            (rail_left, panel_bottom, rail_width, top - panel_bottom)
        )
        _blank(self.ax_panel)
        self.ax_chart = None
        if config.show_chart:
            self.ax_chart = self.fig.add_axes((rail_left, chart_bottom, rail_width, chart_height))

        self.ax_header = self.fig.add_axes((left, 0.888, 0.974 - left, 0.094))
        _blank(self.ax_header)
        self.ax_chain = self.fig.add_axes((left, 0.850, pitch_width_frac, 0.032))
        _blank(self.ax_chain)
        self.ax_footer = self.fig.add_axes((left, 0.020, 0.974 - left, 0.070))
        _blank(self.ax_footer)

        self.camera = Camera(scene, self.aspect, config.camera) if config.use_camera else None
        self._wrap_cache: dict[tuple[str, float], str] = {}
        self._minimap_axes = None
        self._draw_header()

    # -- text metrics ---------------------------------------------------
    def _renderer(self):
        try:
            return self.fig.canvas.get_renderer()
        except Exception:                                   # pragma: no cover
            from matplotlib.backends.backend_agg import FigureCanvasAgg

            FigureCanvasAgg(self.fig)
            return self.fig.canvas.get_renderer()

    def _text_width_px(self, text: str, fontsize: float, weight: str = "normal") -> float:
        artist = self.fig.text(0, 0, text, fontsize=fontsize, weight=weight)
        try:
            width = artist.get_window_extent(self._renderer()).width
        finally:
            artist.remove()
        return float(width)

    def _wrap(self, text: str, fontsize: float, width_px: float) -> str:
        key = (text, fontsize)
        cached = self._wrap_cache.get(key)
        if cached is not None:
            return cached
        words = text.split()
        lines, current = [], ""
        for word in words:
            candidate = f"{current} {word}".strip()
            if current and self._text_width_px(candidate, fontsize) > width_px:
                lines.append(current)
                current = word
            else:
                current = candidate
        if current:
            lines.append(current)
        wrapped = "\n".join(lines)
        self._wrap_cache[key] = wrapped
        return wrapped

    def _panel_px(self) -> float:
        return float(self.ax_panel.get_window_extent().width)

    def _line_step(self, fontsize: float, factor: float = 1.62) -> float:
        height_px = float(self.ax_panel.get_window_extent().height)
        return fontsize * factor * self.config.dpi / 72.0 / max(height_px, 1e-6)

    # ------------------------------------------------------------------
    def draw(self, index: int) -> None:
        """Redraw every panel for one scene frame."""

        t = float(self.scene.times[index])
        state = self.storyboard.state(t)
        beat = self.storyboard.beat_at(t)

        unit, viewport = self._draw_pitch_layer(index, state)
        self._draw_chain(t)
        self._draw_panel(beat, state, index, unit)
        self._draw_footer(index, t)
        if self.ax_chart is not None:
            self._draw_chart(t)
        self._draw_minimap(index, state, viewport)

    def _draw_minimap(self, index: int, state: dict[str, float], viewport) -> None:
        if getattr(self, "_minimap_axes", None) is not None:
            self.fig.delaxes(self._minimap_axes)
            self._minimap_axes = None
        if not self.config.show_minimap or viewport is None:
            return
        pitch = self.ax_pitch.get_position()
        width = 0.112
        height = width * (self.config.width_px / self.config.height_px) * (68.0 / 105.0)
        pad_x, pad_y = 0.014, 0.020
        rect = (pitch.x0 + pad_x, pitch.y0 + pad_y, width, height)
        draw_minimap(self.fig, self.scene, index, rect, state, viewport)
        self._minimap_axes = self.fig.axes[-1]

    # ------------------------------------------------------------------
    def _draw_pitch_layer(self, index: int, state: dict[str, float]):
        scene, ax, config = self.scene, self.ax_pitch, self.config
        ax.clear()
        draw_pitch(ax, scene.pitch_length, scene.pitch_width, config.pitch_style)

        unit = 1.0
        viewport = None
        if self.camera is not None:
            x0, x1, y0, y1 = self.camera.viewport(index, state.get("zoom", 1.0))
            viewport = (x0, x1, y0, y1)
            ax.set_xlim(x0, x1)
            ax.set_ylim(y0, y1)
            unit = (x1 - x0) / scene.pitch_length
        attack_arrow(ax, scene.pitch_length, scene.pitch_width,
                     alpha=(1.0 - state["mute"]) if unit > 0.8 else 0.0)

        # ---- fields (bottom of the stack) ------------------------------
        mode = config.wake_mode
        delta_drawn = False
        if state["wake"] > 0.01 and mode != "off":
            if mode == "geometric":
                L.draw_geometric_wake(
                    ax, scene, index, state["wake"],
                    freeze_index=getattr(scene.surfaces, "freeze_index", None),
                )
            else:
                drawn = L.draw_space_field(ax, scene, index, state["wake"], mode=mode)
                if drawn is None:
                    L.draw_geometric_wake(ax, scene, index, state["wake"])
                elif config.delta_outline and mode != "delta":
                    delta_drawn = L.draw_delta_outline(ax, scene, index, state["wake"], label=False)

        if state["lane"] > 0.01:
            L.draw_beneficiary_lane(ax, scene, index, state["lane"], unit=unit, label=False)
        if state["future"] > 0.01:
            for beneficiary_id in scene.beneficiary_ids:
                L.draw_future_path(ax, scene, beneficiary_id, index, strength=state["future"])

        onset_index = scene.moment_index("onset", 0) or 0
        for runner_id in scene.runner_ids:
            L.draw_run_vector(ax, scene, runner_id, index, onset_index,
                              strength=state["runner"] * 0.9, unit=unit)
            L.draw_trail(ax, scene, runner_id, index, strength=state["runner"])
        if state["defender"] > 0.01 and scene.runner_ids:
            phase = (index * 0.9) % 7.5
            for defender_id in scene.defender_ids:
                L.draw_tether(
                    ax, scene, scene.runner_ids[0], defender_id, index,
                    strength=state["defender"], phase=phase, unit=unit, show_distance=False,
                )
        if state["ghost"] > 0.01:
            L.draw_ghost(ax, scene, index, state["ghost"], unit=unit, label=False)

        neutral = palette.NEUTRAL_ALPHA_FULL + (
            palette.NEUTRAL_ALPHA_MUTED - palette.NEUTRAL_ALPHA_FULL
        ) * state["mute"]
        L.draw_players(
            ax, scene, index,
            neutral_alpha=neutral,
            role_strength={
                "runner": state["runner"],
                "defender": state["defender"],
                "beneficiary": state["beneficiary"],
            },
            show_numbers=config.show_numbers,
            unit=unit,
            label_roles=False,
        )
        L.draw_ball_trail(ax, scene, index, alpha=0.55)
        L.draw_ball(ax, scene, index, unit=unit)

        # ---- labels last, so they can avoid everything already placed --
        placer = LabelPlacer(ax)
        marker_px = self._marker_radius_px(unit)
        placer.reserve_points(
            [scene.view_xy(p.xy[index]) for p in scene.players.values()], marker_px
        )
        placer.reserve_points([scene.view_xy(scene.ball_xy[index])], marker_px * 0.7)
        for role, ids in (("runner", scene.runner_ids), ("defender", scene.defender_ids),
                          ("beneficiary", scene.beneficiary_ids)):
            strength = state.get(role, 0.0)
            if strength <= 0.25:
                continue
            for player_id in ids:
                player = scene.players.get(player_id)
                if player is None:
                    continue
                placer.place(
                    scene.view_xy(player.xy[index]),
                    f"{palette.ROLE_LABELS[role]}  #{player.label}",
                    palette.ROLE_COLOURS[role],
                    fontsize=7.0, alpha=strength,
                    prefer=L._ROLE_PREFERENCE[role],
                    clearance_px=marker_px * 0.5, zorder=6.5,
                )
        for runner_id in scene.runner_ids:
            L.draw_run_vector(ax, scene, runner_id, index, onset_index,
                              strength=state["runner"] * 0.9, unit=unit,
                              placer=placer, label="ran")
        if state["ghost"] > 0.4:
            L.draw_ghost(ax, scene, index, state["ghost"], unit=unit,
                         label=True, placer=placer, marker=False)
        if state["beneficiary"] > 0.45:
            L.draw_gain_badge(ax, scene, index, self._gain_at(index),
                              strength=state["beneficiary"], unit=unit, placer=placer)
        L.draw_shot_marker(ax, scene, index, strength=1.0, unit=unit, placer=placer)
        # The gain badge already names the difference; do not say it twice.
        if delta_drawn and state["wake"] > 0.45 and state["beneficiary"] <= 0.45:
            L.draw_delta_outline(ax, scene, index, state["wake"], label=True,
                                 placer=placer, outline=False)
        if state["lane"] > 0.4:
            L.draw_beneficiary_lane(ax, scene, index, state["lane"], unit=unit,
                                    label=True, placer=placer, shapes=False)
        return unit, viewport

    def _gain_at(self, index: int) -> float:
        times = self.scene.series.get("_surface_t")
        gain = self.scene.series.get("_value_gain")
        if times is None or gain is None:
            return 0.0
        t = float(self.scene.times[index])
        return float(gain[int(np.argmin(np.abs(times - t)))])

    def _marker_radius_px(self, unit: float) -> float:
        width_px = self.ax_pitch.get_window_extent().width
        metres = unit * self.scene.pitch_length
        return float(L.ROLE_R * L.mark_scale(unit) * width_px / max(metres, 1e-6))

    # ------------------------------------------------------------------
    def _draw_header(self) -> None:
        scene, ax = self.scene, self.ax_header
        ax.clear(); _blank(ax)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.text(0.0, 0.66, scene.title, fontsize=20.5, weight="bold",
                color=palette.TEXT_PRIMARY, va="center", ha="left")
        ax.text(0.0, 0.16, scene.subtitle, fontsize=10.2,
                color=palette.TEXT_SECONDARY, va="center", ha="left")
        ax.text(1.0, 0.68, tracked("OFF-BALL VALUE"), fontsize=9.2, weight="bold",
                color=palette.TEXT_SECONDARY, va="center", ha="right")
        ax.text(1.0, 0.18, scene.scene_id, fontsize=8.4,
                color=palette.TEXT_MUTED, va="center", ha="right", family="monospace")

    # ------------------------------------------------------------------
    def _draw_chain(self, t: float) -> None:
        ax = self.ax_chain
        ax.clear(); _blank(ax)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        progress = self.storyboard.chain_progress(t)
        n = len(CHAIN)
        gap = 0.017
        width = (1.0 - gap * (n - 1)) / n
        for index, ((label, role), lit) in enumerate(zip(CHAIN, progress)):
            x0 = index * (width + gap)
            colour = palette.ROLE_COLOURS.get(role, palette.SPACE_WAKE)
            ax.add_patch(
                FancyBboxPatch(
                    (x0, 0.10), width, 0.80,
                    boxstyle="round,pad=0,rounding_size=0.12",
                    facecolor=_mix(palette.PANEL, colour, 0.26 * lit),
                    edgecolor=_mix(palette.GRID, colour, lit),
                    lw=1.0 + 1.0 * lit,
                )
            )
            ax.text(
                x0 + width / 2, 0.50, tracked(label),
                ha="center", va="center", fontsize=8.0, weight="bold",
                color=_mix(palette.TEXT_MUTED, palette.TEXT_PRIMARY, lit),
            )
            if index < n - 1:
                ax.text(x0 + width + gap / 2, 0.50, "❯", ha="center", va="center",
                        fontsize=7.5, color=palette.TEXT_MUTED, alpha=0.55)

    # ------------------------------------------------------------------
    def _draw_panel(self, beat, state: dict[str, float], index: int, unit: float) -> None:
        scene, ax = self.scene, self.ax_panel
        ax.clear(); _blank(ax)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        width_px = self._panel_px()
        accent = _beat_colour(beat.key)
        y = 0.995

        chip = (
            "MANUAL ANNOTATION · EFFECT "
            + str(scene.provenance.get("annotation_effect", "")).upper()
            if scene.source == "annotation" else scene.source.upper()
        )
        ax.text(0.0, y, chip, fontsize=6.6, weight="bold", color=palette.TEXT_MUTED,
                va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.34", facecolor=palette.PANEL, edgecolor="none"))
        y -= self._line_step(6.6) + 0.030

        step = self.storyboard.index_of(beat)
        ax.add_patch(Rectangle((0.0, y - 0.046), 0.010, 0.046, facecolor=accent, edgecolor="none"))
        ax.text(0.028, y - 0.004, f"{step}", fontsize=8.4, color=palette.TEXT_MUTED,
                va="top", ha="left", family="monospace")
        ax.text(0.072, y - 0.002, beat.title, fontsize=12.6, weight="bold",
                color=palette.TEXT_PRIMARY, va="top", ha="left")
        y -= self._line_step(12.6) + 0.018

        wrapped = self._wrap(beat.caption, 9.0, width_px)
        lines = wrapped.split("\n")
        if len(lines) > 3:                      # keep the caption to three lines
            lines = lines[:3]
            lines[-1] = lines[-1].rstrip() + "\u2026"
            wrapped = "\n".join(lines)
        ax.text(0.0, y, wrapped, fontsize=9.0, color=palette.TEXT_SECONDARY,
                va="top", ha="left", linespacing=1.62)
        y -= self._line_step(9.0) * len(lines) + 0.020

        y = self._divider(ax, y)
        y = self._draw_stat(ax, y, index)
        y = self._divider(ax, y - 0.006)
        y = self._draw_cast(ax, y, state)
        if scene.notes and y > 0.16:
            y = self._divider(ax, y - 0.004)
            self._draw_note(ax, y, width_px)

    def _draw_stat(self, ax, y: float, index: int) -> float:
        """Hero number: space gained against the held-defender counterfactual."""

        scene = self.scene
        times = scene.series.get("_surface_t")
        gain = scene.series.get("_value_gain")
        if times is None or gain is None:
            return y
        t = float(scene.times[index])
        slot = int(np.argmin(np.abs(times - t)))
        value = float(gain[slot])
        peak = float(np.nanmax(np.abs(gain))) or 1.0
        colour = palette.DELTA_POS if value >= 0 else palette.DELTA_NEG

        ax.text(0.0, y, tracked("SPACE GAINED VS HELD DEFENDER"), fontsize=7.2,
                weight="bold", color=palette.TEXT_MUTED, va="top", ha="left")
        y -= self._line_step(7.2) + 0.014
        ax.text(0.0, y, f"{value:+.1f}", fontsize=26, weight="bold", color=colour,
                va="top", ha="left")
        ax.text(0.31, y - 0.008, "goal-weighted\nspace units", fontsize=7.0,
                color=palette.TEXT_MUTED, va="top", ha="left", linespacing=1.45)
        y -= self._line_step(26, factor=1.10)
        # a signed bar makes the sign and magnitude readable at a glance
        ax.add_patch(Rectangle((0.0, y), 1.0, 0.010, facecolor=palette.PANEL, edgecolor="none"))
        ax.add_patch(Rectangle((0.5, y), 0.5 * value / peak, 0.010,
                               facecolor=colour, edgecolor="none"))
        ax.add_patch(Rectangle((0.4985, y - 0.005), 0.003, 0.020,
                               facecolor=palette.TEXT_MUTED, edgecolor="none"))
        return y - 0.028

    def _draw_cast(self, ax, y: float, state: dict[str, float]) -> float:
        scene = self.scene
        ax.text(0.0, y, tracked("CAST"), fontsize=7.2, weight="bold",
                color=palette.TEXT_MUTED, va="top", ha="left")
        y -= self._line_step(7.2) + 0.016
        for role, ids in (
            ("runner", scene.runner_ids),
            ("defender", scene.defender_ids),
            ("beneficiary", scene.beneficiary_ids),
        ):
            colour = palette.ROLE_COLOURS[role]
            alpha = 0.30 + 0.70 * state.get(role, 0.0)
            names = [scene.players[i] for i in ids if i in scene.players]
            if not names:
                continue
            ax.add_patch(Rectangle((0.002, y - 0.030), 0.009, 0.030,
                                   facecolor=colour, edgecolor="none", alpha=alpha))
            ax.text(0.032, y - 0.001, palette.ROLE_LABELS[role], fontsize=7.0, weight="bold",
                    color=colour, va="top", ha="left", alpha=alpha)
            y -= self._line_step(7.0) + 0.006
            line = "   ".join(f"#{p.label} {p.name}" for p in names)
            wrapped = self._wrap(line, 9.0, self._panel_px() * 0.92)
            ax.text(0.032, y, wrapped, fontsize=9.0, color=palette.TEXT_PRIMARY,
                    va="top", ha="left", alpha=alpha, linespacing=1.5)
            y -= self._line_step(9.0) * (wrapped.count("\n") + 1) + 0.020

        # The neutral team fills need a key too, not only the three roles.
        # Every optional block is guarded: the panel must never spill into the
        # chart below it, whatever a scene's cast list happens to contain.
        if y < 0.12:
            return y
        for colour, label, name in (
            (palette.ATTACK_NEUTRAL, "ATTACKING", scene.attacking_team_name),
            (palette.DEFEND_NEUTRAL, "DEFENDING", scene.defending_team_name),
        ):
            ax.add_patch(Rectangle((0.002, y - 0.022), 0.009, 0.019,
                                   facecolor=colour, edgecolor="none", alpha=0.85))
            ax.text(0.032, y - 0.012, label, fontsize=6.4, weight="bold",
                    color=palette.TEXT_MUTED, va="center", ha="left")
            ax.text(0.205, y - 0.012, _elide(name, 26), fontsize=7.8,
                    color=palette.TEXT_SECONDARY, va="center", ha="left")
            y -= self._line_step(7.8) + 0.008
        return y - 0.006

    def _draw_note(self, ax, y: float, width_px: float) -> float:
        """The human annotator's own comment on this scene, when renderable."""

        text = self.scene.notes.replace("\n", " ").strip()
        if not text:
            return y
        ascii_only = all(ord(ch) < 0x2E80 for ch in text)
        font = None if ascii_only else UNICODE_FONT
        if font is None and not ascii_only:
            return y
        ax.text(0.0, y, tracked("ANNOTATOR'S NOTE"), fontsize=6.8, weight="bold",
                color=palette.TEXT_MUTED, va="top", ha="left")
        y -= self._line_step(6.8) + 0.012
        kwargs = {"family": font} if font else {}
        line = self._line_step(7.8)
        budget = max(1, int((y - 0.012) / max(line, 1e-6)))
        wrapped = self._wrap_font(text, 7.8, width_px, font)
        lines = wrapped.split("\n")
        if len(lines) > budget:
            lines = lines[:budget]
            lines[-1] = lines[-1][: max(1, len(lines[-1]) - 1)] + "\u2026"
            wrapped = "\n".join(lines)
        ax.text(0.0, y, wrapped, fontsize=7.8, color=palette.TEXT_SECONDARY,
                va="top", ha="left", linespacing=1.55, style="italic" if ascii_only else "normal",
                **kwargs)
        return y - self._line_step(7.8) * (wrapped.count("\n") + 1)

    def _wrap_font(self, text: str, fontsize: float, width_px: float, font: str | None) -> str:
        if font is None:
            return self._wrap(text, fontsize, width_px)
        # CJK text has no spaces to break on; wrap by measured character runs
        lines, current = [], ""
        for ch in text:
            candidate = current + ch
            artist = self.fig.text(0, 0, candidate, fontsize=fontsize, family=font)
            try:
                too_wide = artist.get_window_extent(self._renderer()).width > width_px
            finally:
                artist.remove()
            if too_wide and current:
                lines.append(current)
                current = ch
            else:
                current = candidate
        if current:
            lines.append(current)
        return "\n".join(lines)

    @staticmethod
    def _divider(ax, y: float) -> float:
        ax.plot([0.0, 1.0], [y, y], color=palette.GRID, lw=1.0, solid_capstyle="butt")
        return y - 0.034

    # ------------------------------------------------------------------
    def _draw_chart(self, t: float) -> None:
        scene, ax = self.scene, self.ax_chart
        ax.clear()
        ax.set_facecolor(palette.INK)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(palette.GRID)
            ax.spines[spine].set_linewidth(0.9)

        times = scene.series.get("_surface_t")
        factual = scene.series.get("_factual_value")
        counter = scene.series.get("_counterfactual_value")

        ax.text(0.0, 1.235, tracked("SPACE HELD BY THE BENEFICIARY"),
                transform=ax.transAxes, fontsize=7.8, weight="bold",
                color=palette.TEXT_SECONDARY, va="bottom", ha="left")
        ax.text(0.0, 1.06, "goal-weighted residual space (repo v0.1) · higher = more usable space",
                transform=ax.transAxes, fontsize=6.8, color=palette.TEXT_MUTED,
                va="bottom", ha="left")

        if times is None or factual is None:
            ax.text(0.5, 0.5, "no space quantity available for this scene",
                    ha="center", va="center", fontsize=8.5, color=palette.TEXT_MUTED,
                    transform=ax.transAxes)
            ax.set_xticks([]); ax.set_yticks([])
            return

        ax.grid(axis="y", color=palette.GRID, lw=0.7, alpha=0.75)
        ax.set_axisbelow(True)
        if counter is not None:
            ax.fill_between(times, counter, factual, where=factual >= counter,
                            color=palette.DELTA_POS, alpha=0.22, linewidth=0, interpolate=True)
            ax.fill_between(times, counter, factual, where=factual < counter,
                            color=palette.DELTA_NEG, alpha=0.15, linewidth=0, interpolate=True)
            ax.plot(times, counter, color=palette.DEFENDER, lw=1.6,
                    linestyle=(0, (3.5, 2.5)), solid_capstyle="round")
        ax.plot(times, factual, color=palette.BENEFICIARY, lw=2.0, solid_capstyle="round")

        for key, colour in (("onset", palette.RUNNER), ("reaction", palette.DEFENDER)):
            value = scene.moments.get(key)
            if value is not None:
                ax.axvline(value, color=colour, lw=0.9, alpha=0.45, linestyle=(0, (2.0, 2.5)))
        ax.axvline(t, color=palette.TEXT_PRIMARY, lw=1.2, alpha=0.8)

        index = int(np.argmin(np.abs(times - t)))
        ax.plot([t], [factual[index]], marker="o", ms=6.0, color=palette.BENEFICIARY,
                markeredgecolor=palette.INK, markeredgewidth=1.5, zorder=5, clip_on=False)
        if counter is not None:
            ax.plot([t], [counter[index]], marker="o", ms=4.6, color=palette.DEFENDER,
                    markeredgecolor=palette.INK, markeredgewidth=1.3, zorder=5, clip_on=False)

        span = float(times[-1] - times[0])
        ax.set_xlim(float(times[0]), float(times[-1]) + 0.30 * span)
        lo, hi = float(np.nanmin(factual)), float(np.nanmax(factual))
        if counter is not None:
            lo = min(lo, float(np.nanmin(counter))); hi = max(hi, float(np.nanmax(counter)))
        pad = 0.14 * max(hi - lo, 1e-6)
        ax.set_ylim(lo - pad, hi + pad)
        # Direct labels instead of a legend box. When the two curves end close
        # together they would print on top of each other, so nudge them apart.
        label_y = float(factual[-1])
        other_y = float(counter[-1]) if counter is not None else None
        if other_y is not None:
            gap = 0.11 * (hi - lo + 2 * pad)
            if abs(label_y - other_y) < gap:
                midpoint = (label_y + other_y) / 2.0
                high = label_y >= other_y
                label_y = midpoint + (gap / 2 if high else -gap / 2)
                other_y = midpoint + (-gap / 2 if high else gap / 2)
        ax.text(times[-1] + 0.03 * span, label_y, "observed", fontsize=7.2, weight="bold",
                color=palette.BENEFICIARY, va="center", ha="left")
        if other_y is not None:
            ax.text(times[-1] + 0.03 * span, other_y, "defender held", fontsize=7.2,
                    weight="bold", color=palette.DEFENDER, va="center", ha="left")
        ax.tick_params(labelsize=7.0, length=2.5, pad=2)
        ax.set_xlabel("seconds relative to the annotated shot", fontsize=7.0, labelpad=1)

    # ------------------------------------------------------------------
    def _draw_footer(self, index: int, t: float) -> None:
        scene, ax = self.scene, self.ax_footer
        ax.clear(); _blank(ax)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)

        t0, t1 = float(scene.times[0]), float(scene.times[-1])
        span = max(t1 - t0, 1e-6)
        bar_y, bar_h = 0.62, 0.16
        ax.add_patch(Rectangle((0, bar_y), 1.0, bar_h, facecolor=palette.PANEL, edgecolor="none"))
        progress = (t - t0) / span
        ax.add_patch(Rectangle((0, bar_y), progress, bar_h,
                               facecolor=palette.TEXT_SECONDARY, edgecolor="none", alpha=0.55))
        for beat in self.storyboard.beats:
            if beat.chain_step < 0:
                continue
            x = (beat.start - t0) / span
            colour = _beat_colour(beat.key)
            ax.add_patch(Rectangle((x - 0.0016, bar_y - 0.10), 0.0032, bar_h + 0.20,
                                   facecolor=colour, edgecolor="none"))
        ax.text(0.0, bar_y - 0.30, f"t = {t:+.2f} s", fontsize=8.0, family="monospace",
                color=palette.TEXT_SECONDARY, ha="left", va="top")
        self._draw_grounding_strip(ax, bar_y - 0.30)

    def _draw_grounding_strip(self, ax, y: float) -> None:
        """One honest line: which layers are measured, human, or explanatory."""

        scene = self.scene
        blocks: list[tuple[str, str, str]] = [
            ("MEASURED",
             "IDSSE tracking 25 Hz"
             + ("  ·  goal-weighted residual space (repo v0.1)"
                if scene.surfaces is not None else ""),
             palette.SPACE_WAKE),
            ("HUMAN",
             "runner / defender / beneficiary from manual annotation"
             if scene.source == "annotation" else f"roles from {scene.source}",
             palette.TEXT_SECONDARY),
        ]
        if scene.surfaces is not None and getattr(scene.surfaces, "freeze_positions", None):
            blocks.append(("EXPLANATORY",
                           "held-position defender is a what-if device, "
                           "not a learned defensive response", palette.DEFENDER))
        if self.config.wake_mode == "geometric":
            blocks.append(("ILLUSTRATIVE", "space wake is a drawing aid, not a metric",
                           palette.TEXT_MUTED))

        width = float(ax.get_window_extent().width)
        cursor = 1.0
        for tag, text, colour in reversed(blocks):
            body_px = self._text_width_px(text, 6.8)
            tag_px = self._text_width_px(tag, 6.2, "bold")
            ax.text(cursor, y, text, fontsize=6.8, color=palette.TEXT_MUTED,
                    ha="right", va="top")
            cursor -= body_px / width + 0.006
            ax.text(cursor, y, tag, fontsize=6.2, weight="bold", color=colour,
                    ha="right", va="top")
            cursor -= tag_px / width + 0.016

    def _provenance_line(self) -> str:
        scene = self.scene
        parts = [f"tracking IDSSE {scene.provenance.get('source_fps', 25):g} Hz"]
        parts.append("roles: manual annotation" if scene.source == "annotation"
                     else f"roles: {scene.source}")
        if scene.surfaces is not None and self.config.wake_mode in ("delta", "residual"):
            parts.append("space: goal-weighted residual influence (repo v0.1)")
            parts.append("counterfactual: defender held at pre-run position (explanatory device)")
        elif self.config.wake_mode == "geometric":
            parts.append("space wake: illustrative geometry, not a measured quantity")
        return "   ·   ".join(parts)


def _beat_colour(key: str) -> str:
    return {
        "runner": palette.RUNNER,
        "defender": palette.DEFENDER,
        "space": palette.SPACE_WAKE,
        "beneficiary": palette.BENEFICIARY,
    }.get(key, palette.TEXT_SECONDARY)


def _elide(text: str, width: int) -> str:
    return text if len(text) <= width else text[: width - 1] + "…"


def _blank(ax) -> None:
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_facecolor("none")
    ax.patch.set_alpha(0.0)


def _mix(first: str, second: str, weight: float) -> tuple[float, float, float]:
    a, b = np.asarray(to_rgb(first)), np.asarray(to_rgb(second))
    return tuple(a + (b - a) * float(min(1.0, max(0.0, weight))))  # type: ignore[return-value]


def draw_minimap(fig, scene, index, rect, state, viewport=None) -> None:
    """Full-pitch inset showing every player plus the current camera viewport.

    The close shot is what makes the story legible; this keeps the 11-v-11
    context that the close shot necessarily crops away.
    """

    ax = fig.add_axes(rect, zorder=8)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.patch.set_visible(False)
    half_l, half_w = scene.pitch_length / 2, scene.pitch_width / 2
    ax.set_xlim(-half_l - 1, half_l + 1)
    ax.set_ylim(-half_w - 1, half_w + 1)
    ax.set_aspect("equal")
    ax.add_patch(Rectangle((-half_l - 3.0, -half_w - 3.0), scene.pitch_length + 6.0,
                           scene.pitch_width + 6.0, facecolor=palette.INK,
                           edgecolor=palette.GRID, lw=0.9, alpha=0.88, zorder=0))
    ax.add_patch(Rectangle((-half_l, -half_w), scene.pitch_length, scene.pitch_width,
                           facecolor=palette.PITCH_DARK, edgecolor=palette.PITCH_LINE,
                           lw=0.7, alpha=0.95, zorder=1))
    ax.plot([0, 0], [-half_w, half_w], color=palette.PITCH_LINE, lw=0.5, alpha=0.45, zorder=2)
    for sign in (-1, 1):
        ax.add_patch(Rectangle(
            (sign * half_l - (16.5 if sign > 0 else 0), -20.16), 16.5, 40.32,
            facecolor="none", edgecolor=palette.PITCH_LINE, lw=0.5, alpha=0.45, zorder=2))
    if viewport is not None:
        x0, x1, y0, y1 = viewport
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor=palette.TEXT_PRIMARY,
                               edgecolor=palette.TEXT_PRIMARY, lw=0.8, alpha=0.10))
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor="none",
                               edgecolor=palette.TEXT_PRIMARY, lw=0.9, alpha=0.55))
    for player in scene.players.values():
        xy = player.xy[index]
        if not np.all(np.isfinite(xy)):
            continue
        view = scene.view_xy(xy)
        role = scene.role_of(player.player_id)
        colour = palette.ROLE_COLOURS.get(role or "", None)
        if colour is None:
            colour = palette.ATTACK_NEUTRAL if player.side == "attack" else palette.DEFEND_NEUTRAL
            size, alpha = 5.0, 0.55
        else:
            size, alpha = 13.0, 0.35 + 0.65 * state.get(role, 0.0)
        ax.scatter([view[0]], [view[1]], s=size, color=colour, alpha=alpha,
                   edgecolors="none", zorder=3)
    ball = scene.view_xy(scene.ball_xy[index])
    if np.all(np.isfinite(ball)):
        ax.scatter([ball[0]], [ball[1]], s=6.0, color=palette.BALL, alpha=0.95,
                   edgecolors="none", zorder=4)
    ax.set_xlim(-half_l - 3.5, half_l + 3.5)
    ax.set_ylim(-half_w - 3.5, half_w + 3.5)
    ax.text(-half_l - 3.0, half_w + 5.0, tracked("FULL PITCH"), fontsize=5.6, weight="bold",
            color=palette.TEXT_MUTED, ha="left", va="bottom")
