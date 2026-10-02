"""Build a canonical :class:`~demo_viz.scene.Scene` from one annotated clip.

Join key: the spreadsheet stores ``period`` + ``period_seconds`` of the shot,
which maps onto an IDSSE frame number through the first frame of that half.
Shirt numbers from the spreadsheet resolve to DFL person ids through the match
information file.  The same window convention as the annotation app is used by
default (8 s before the shot, 3 s after), so on-screen time lines up with the
reviewed video clip.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..annotations import AnnotatedClip, find_clip
from ..config import demo_paths, ensure_repo_on_path
from ..scene import Scene, ScenePlayer
from .idsse import (
    SECTION_BY_PERIOD,
    find_match_files,
    load_window_cached,
    match_metadata,
    section_start_frames,
)

ensure_repo_on_path()

from offball_value.bundesliga import FPS  # noqa: E402

CLIP_BEFORE_S = 8.0
CLIP_AFTER_S = 3.0


def frame_for(match_id: str, period: int, period_seconds: float, idsse_dir: Path | None = None) -> int:
    """IDSSE frame number for a time measured from the start of a half."""

    files = find_match_files(match_id, idsse_dir)
    starts = section_start_frames(files.positions)
    section = SECTION_BY_PERIOD[int(period)]
    if section not in starts:
        raise ValueError(f"{files.positions.name} has no {section} frames")
    return int(round(starts[section] + period_seconds * FPS))


def _resolve_shirts(metadata, team_id: str, shirts) -> tuple[list[str], list[str]]:
    """Map shirt numbers of one team onto DFL person ids."""

    by_shirt = {
        str(player.shirt_number): player.player_id
        for player in metadata.players.values()
        if player.team_id == team_id and player.shirt_number
    }
    found, missing = [], []
    for shirt in shirts:
        player_id = by_shirt.get(str(shirt))
        (found if player_id else missing).append(player_id or str(shirt))
    return found, missing


def build_scene_from_clip(
    clip: AnnotatedClip,
    before_s: float = CLIP_BEFORE_S,
    after_s: float = CLIP_AFTER_S,
    idsse_dir: Path | None = None,
    stride: int = 1,
) -> Scene:
    """Join one annotated clip to its tracking window and roles."""

    paths = demo_paths()
    metadata = match_metadata(clip.match_id, idsse_dir)

    # --- which team is attacking? the spreadsheet stores the shooting team ---
    def normalise(name: str) -> str:
        return "".join(ch for ch in name.casefold() if ch.isalnum())

    wanted = normalise(clip.team)
    attacking_team_id = None
    for team_id, team in metadata.teams.items():
        key = normalise(team.name)
        if key == wanted or (wanted and (wanted in key or key in wanted)):
            attacking_team_id = team_id
            break
    if attacking_team_id is None:
        attacking_team_id = metadata.home_team_id
    defending_team_id = next(t for t in metadata.teams if t != attacking_team_id)

    # --- frame window -------------------------------------------------------
    shot_frame = frame_for(clip.match_id, clip.period, clip.period_seconds, idsse_dir)
    first_frame = shot_frame - int(round(before_s * FPS))
    last_frame = shot_frame + int(round(after_s * FPS))
    window = load_window_cached(clip.match_id, clip.period, first_frame, last_frame, idsse_dir)

    keep = slice(None, None, max(1, int(stride)))
    frame_ids = window.frame_ids[keep]
    fps = FPS / max(1, int(stride))
    times = (frame_ids - shot_frame) / FPS  # 0 at the annotated shot

    # --- players ------------------------------------------------------------
    players: dict[str, ScenePlayer] = {}
    for object_id, track in window.tracks.items():
        team_id = window.team_of[object_id]
        if team_id == "BALL":
            continue
        meta = metadata.players.get(object_id)
        players[object_id] = ScenePlayer(
            player_id=object_id,
            team_id=team_id,
            side="attack" if team_id == attacking_team_id else "defend",
            shirt=(meta.shirt_number if meta else None),
            name=(meta.short_name if meta else object_id),
            xy=track[keep],
            speed=window.speeds[object_id][keep],
            is_goalkeeper=bool(meta and meta.position == "TW"),
        )

    ball_xy = window.ball_xy[keep]

    # --- roles --------------------------------------------------------------
    runner_ids, missing_runners = _resolve_shirts(metadata, attacking_team_id, clip.runner_shirts)
    defender_ids, missing_defenders = _resolve_shirts(
        metadata, defending_team_id, clip.defender_shirts
    )
    beneficiary_ids, missing_beneficiaries = _resolve_shirts(
        metadata, attacking_team_id, clip.beneficiary_shirts
    )
    # A role player may have been substituted out of this window; drop silently
    # tracked ones but record the fact in provenance.
    untracked = [
        player_id
        for player_id in runner_ids + defender_ids + beneficiary_ids
        if player_id not in players
    ]
    runner_ids = [p for p in runner_ids if p in players]
    defender_ids = [p for p in defender_ids if p in players]
    beneficiary_ids = [p for p in beneficiary_ids if p in players]

    # --- attacking direction ------------------------------------------------
    direction = _attacking_direction(players, attacking_team_id)

    # --- ball carrier at the start of the window ----------------------------
    carrier_ids = _carrier_ids(players, ball_xy, attacking_team_id)

    video_path = None
    if paths.clip_dir and clip.video_filename:
        candidate = paths.clip_dir / clip.video_filename
        video_path = candidate if candidate.exists() else None

    attack_name = metadata.teams[attacking_team_id].name
    defend_name = metadata.teams[defending_team_id].name

    scene = Scene(
        scene_id=clip.clip_id,
        source="annotation",
        fps=fps,
        times=times,
        players=players,
        ball_xy=ball_xy,
        attacking_team_id=attacking_team_id,
        defending_team_id=defending_team_id,
        attacking_direction=direction,
        runner_ids=tuple(runner_ids),
        defender_ids=tuple(defender_ids),
        beneficiary_ids=tuple(beneficiary_ids),
        carrier_ids=tuple(carrier_ids),
        title=f"{attack_name} vs {defend_name}",
        subtitle=(
            f"{clip.match_clock}  ·  shot {clip.shot_number or '?'}  ·  "
            f"{clip.shooter or 'unknown shooter'}"
            + (f"  ·  {clip.shot_result}" if clip.shot_result else "")
        ),
        attacking_team_name=attack_name,
        defending_team_name=defend_name,
        notes=clip.notes,
        frame_ids=frame_ids,
        moments={"shot": 0.0},
        video_path=video_path,
        provenance={
            "adapter": "annotation",
            "annotation_effect": clip.effect,
            "match_id": clip.match_id,
            "period": clip.period,
            "period_seconds": clip.period_seconds,
            "shot_frame_id": shot_frame,
            "shot_result": clip.shot_result,
            "window_frames": [int(first_frame), int(last_frame)],
            "source_fps": FPS,
            "stride": int(stride),
            "unresolved_shirts": {
                "runner": missing_runners,
                "defender": missing_defenders,
                "beneficiary": missing_beneficiaries,
            },
            "untracked_role_players": untracked,
        },
    )
    return scene


def _attacking_direction(players: dict[str, ScenePlayer], attacking_team_id: str) -> int:
    """+1 when the attacking team shoots at the +x goal in raw coordinates."""

    keeper = next(
        (
            player
            for player in players.values()
            if player.team_id == attacking_team_id and player.is_goalkeeper
        ),
        None,
    )
    if keeper is not None and len(keeper.xy):
        mean_x = float(np.nanmean(keeper.xy[:, 0]))
        if np.isfinite(mean_x):
            return 1 if mean_x < 0 else -1
    team_x = [
        float(np.nanmean(player.xy[:, 0]))
        for player in players.values()
        if player.team_id == attacking_team_id and len(player.xy)
    ]
    team_x = [value for value in team_x if np.isfinite(value)]
    if not team_x:
        return 1
    return -1 if sum(team_x) / len(team_x) > 0 else 1


def _carrier_ids(
    players: dict[str, ScenePlayer],
    ball_xy: np.ndarray,
    attacking_team_id: str,
    max_distance_m: float = 2.0,
) -> list[str]:
    """Attacking players who are within ``max_distance_m`` of the ball at some point."""

    carriers: list[str] = []
    for player in players.values():
        if player.team_id != attacking_team_id or not len(player.xy):
            continue
        distance = np.linalg.norm(player.xy - ball_xy, axis=1)
        if np.nanmin(distance) <= max_distance_m:
            carriers.append(player.player_id)
    return carriers


def scene_from_clip_id(clip_id: str, **kwargs) -> Scene:
    return build_scene_from_clip(find_clip(clip_id), **kwargs)
