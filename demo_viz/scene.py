"""Canonical scene abstraction.

Everything downstream of this module — renderers, story beats, exporters —
talks only to :class:`Scene`.  Adapters (Excel annotations, IDSSE tracking,
pipeline output, synthetic smoke tests) are responsible for producing one.
That keeps the demo independent of whichever intermediate file format the
research pipeline happens to emit this week.

Coordinate convention follows ``offball_value.bundesliga``: metres, origin at
the centre spot, ``x`` in ``[-52.5, 52.5]`` and ``y`` in ``[-34, 34]``.  The
renderer additionally applies :meth:`Scene.view_xy`, which mirrors the pitch so
that the attacking team always plays left-to-right on screen.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

FIELD_LENGTH = 105.0
FIELD_WIDTH = 68.0

@dataclass
class ScenePlayer:
    """One tracked player over the whole scene window."""

    player_id: str
    team_id: str
    side: str                      # "attack" | "defend"
    shirt: str | None = None
    name: str = ""
    xy: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    speed: np.ndarray | None = None       # m/s, from the source feed when present
    is_goalkeeper: bool = False

    @property
    def label(self) -> str:
        return self.shirt or self.name or self.player_id[-4:]

    def velocity(self, index: int, fps: float, window: int = 5) -> tuple[float, float]:
        """Central-difference velocity in m/s over ``window`` samples."""

        n = len(self.xy)
        if n < 2:
            return 0.0, 0.0
        half = max(1, window // 2)
        lo = max(0, index - half)
        hi = min(n - 1, index + half)
        if hi == lo:
            return 0.0, 0.0
        dt = (hi - lo) / fps
        delta = self.xy[hi] - self.xy[lo]
        return float(delta[0] / dt), float(delta[1] / dt)


@dataclass
class Scene:
    """Everything the renderer may draw, with optional quantities left as ``None``."""

    scene_id: str
    source: str                            # "annotation" | "pipeline" | "synthetic"
    fps: float
    times: np.ndarray                      # (T,) seconds, 0 at scene start
    players: dict[str, ScenePlayer]
    ball_xy: np.ndarray                    # (T, 2); NaN where the ball is unknown
    attacking_team_id: str
    defending_team_id: str
    attacking_direction: int = 1           # +1 attacks the +x goal in raw coords

    # --- roles -----------------------------------------------------------
    runner_ids: tuple[str, ...] = ()
    defender_ids: tuple[str, ...] = ()
    beneficiary_ids: tuple[str, ...] = ()
    carrier_ids: tuple[str, ...] = ()

    # --- labelling -------------------------------------------------------
    title: str = ""
    subtitle: str = ""
    attacking_team_name: str = ""
    defending_team_name: str = ""
    notes: str = ""
    frame_ids: np.ndarray | None = None

    # --- optional quantities (``None`` means "not available", never faked") --
    moments: dict[str, float] = field(default_factory=dict)      # seconds
    scores: dict[str, float] = field(default_factory=dict)
    series: dict[str, np.ndarray] = field(default_factory=dict)  # (T,) time series
    surfaces: object | None = None                               # SurfaceStack
    pass_probability: dict[str, np.ndarray] | None = None        # hook, unused today
    dribble_probability: dict[str, np.ndarray] | None = None     # hook, unused today
    threat_surface: np.ndarray | None = None                     # hook, unused today

    video_path: Path | None = None
    provenance: dict[str, object] = field(default_factory=dict)

    pitch_length: float = FIELD_LENGTH
    pitch_width: float = FIELD_WIDTH

    # ------------------------------------------------------------------
    @property
    def n_frames(self) -> int:
        return int(len(self.times))

    @property
    def duration(self) -> float:
        return float(self.times[-1] - self.times[0]) if self.n_frames else 0.0

    def role_of(self, player_id: str) -> str | None:
        if player_id in self.runner_ids:
            return "runner"
        if player_id in self.defender_ids:
            return "defender"
        if player_id in self.beneficiary_ids:
            return "beneficiary"
        return None

    @property
    def role_ids(self) -> tuple[str, ...]:
        return tuple(self.runner_ids) + tuple(self.defender_ids) + tuple(self.beneficiary_ids)

    def player(self, player_id: str) -> ScenePlayer:
        return self.players[player_id]

    def players_on(self, side: str) -> list[ScenePlayer]:
        return [p for p in self.players.values() if p.side == side]

    def by_shirt(self, shirt: str, side: str) -> ScenePlayer | None:
        for player in self.players.values():
            if player.side == side and str(player.shirt) == str(shirt):
                return player
        return None

    # --- viewing transform ---------------------------------------------
    @property
    def flip(self) -> bool:
        """True when raw coordinates must be mirrored to attack left-to-right."""

        return self.attacking_direction < 0

    def view_xy(self, xy: np.ndarray) -> np.ndarray:
        """Map raw pitch coordinates into on-screen coordinates."""

        arr = np.asarray(xy, dtype=float)
        return -arr if self.flip else arr

    @property
    def attacking_goal_xy(self) -> tuple[float, float]:
        """On-screen goal the attacking team is shooting at (always +x)."""

        return (self.pitch_length / 2.0, 0.0)

    # --- time helpers ---------------------------------------------------
    def index_at(self, t: float) -> int:
        if self.n_frames == 0:
            return 0
        return int(np.clip(np.searchsorted(self.times, t), 0, self.n_frames - 1))

    def moment_index(self, key: str, default: int | None = None) -> int | None:
        value = self.moments.get(key)
        if value is None:
            return default
        return self.index_at(float(value))

    # --- summary --------------------------------------------------------
    def describe(self) -> str:
        def names(ids: Sequence[str]) -> str:
            return ", ".join(
                f"#{self.players[i].label} {self.players[i].name}".strip()
                for i in ids
                if i in self.players
            ) or "--"

        lines = [
            f"scene      {self.scene_id}  ({self.source})",
            f"title      {self.title}",
            f"window     {self.n_frames} frames @ {self.fps:g} Hz = {self.duration:.2f} s",
            f"attack     {self.attacking_team_name} ({len(self.players_on('attack'))} tracked)",
            f"defend     {self.defending_team_name} ({len(self.players_on('defend'))} tracked)",
            f"runner     {names(self.runner_ids)}",
            f"defender   {names(self.defender_ids)}",
            f"beneficiary{names(self.beneficiary_ids)}",
            "moments    " + (
                ", ".join(f"{k}={v:+.2f}s" for k, v in self.moments.items()) or "--"
            ),
            "series     " + (", ".join(sorted(self.series)) or "--"),
            f"surfaces   {'yes' if self.surfaces is not None else 'no'}",
            f"video      {self.video_path or '--'}",
        ]
        return "\n".join(lines)


def build_players(
    records: Iterable[Mapping[str, object]],
    attacking_team_id: str,
) -> dict[str, ScenePlayer]:
    """Convenience constructor used by the adapters."""

    players: dict[str, ScenePlayer] = {}
    for record in records:
        player_id = str(record["player_id"])
        team_id = str(record["team_id"])
        players[player_id] = ScenePlayer(
            player_id=player_id,
            team_id=team_id,
            side="attack" if team_id == attacking_team_id else "defend",
            shirt=(str(record["shirt"]) if record.get("shirt") is not None else None),
            name=str(record.get("name", "")),
            xy=np.asarray(record["xy"], dtype=float),
            speed=(
                np.asarray(record["speed"], dtype=float)
                if record.get("speed") is not None
                else None
            ),
            is_goalkeeper=bool(record.get("is_goalkeeper", False)),
        )
    return players
