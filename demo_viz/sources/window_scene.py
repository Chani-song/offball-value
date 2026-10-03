"""Build a :class:`~demo_viz.scene.Scene` from a tracking window instead of an annotated shot.

Five curated showcase scenes (S46 S48 S53 S58 S66) were found by the stage-3 pipeline, not annotated: each is a
run onset with one runner, defender and beneficiary, and no shot row in the spreadsheet. ``data/pipeline_scenes.json``
lists them -- match, half, the onset frame (time 0), the frame window the solver bundle's panels are indexed against
(``clip_first_frame`` .. ``clip_last_frame`` of the bundle's ``scenes.csv``) and the three role ids -- and this module
joins that to the IDSSE tracking the way ``annotation_scene.py`` joins a shot: the same window loader, player records,
attacking-direction and carrier rules. The scene's first frame is the window's first frame, so a bundle panel's
``start_frame - clip_first_frame`` is its frame index here too.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from ..config import ensure_repo_on_path
from ..scene import Scene, ScenePlayer
from .annotation_scene import _attacking_direction, _carrier_ids
from .idsse import SECTION_BY_PERIOD, find_match_files, load_window_cached, match_metadata, section_start_frames

ensure_repo_on_path()
from offball_value.bundesliga import FPS, short_bundesliga_match_id  # noqa: E402

SPEC_PATH = Path(__file__).resolve().parent.parent / "data" / "pipeline_scenes.json"
#: The index label for these scenes: not an annotation strength, the source that found them.
EFFECT = "solver"


@lru_cache(maxsize=1)
def pipeline_specs() -> tuple[dict, ...]:
    """Every listed pipeline scene, in file order (none when the file is absent)."""

    if not SPEC_PATH.exists():
        return ()
    return tuple(json.loads(SPEC_PATH.read_text())["scenes"])


def pipeline_scene_ids() -> list[str]:
    return [spec["scene_id"] for spec in pipeline_specs()]


def find_spec(scene_id: str) -> dict | None:
    return next((spec for spec in pipeline_specs() if spec["scene_id"] == scene_id), None)


def onset_clock(match_id: str, period: int, frame: int, idsse_dir: Path | None = None) -> tuple[int, str]:
    """(seconds into the half, match clock "mm:ss") of a frame."""

    starts = section_start_frames(find_match_files(match_id, idsse_dir).positions)
    seconds = (int(frame) - starts[SECTION_BY_PERIOD[int(period)]]) / FPS
    total = int(seconds) + (45 * 60 if int(period) == 2 else 0)
    return int(seconds), f"{total // 60:02d}:{total % 60:02d}"


def scene_id_for(match_id: str, period: int, onset_frame: int, idsse_dir: Path | None = None) -> str:
    """``J03WPY:run_P1_0733`` -- the half and its seconds at the onset, as the annotated ids carry the shot's."""

    seconds, _ = onset_clock(match_id, period, onset_frame, idsse_dir)
    return f"{short_bundesliga_match_id(match_id)}:run_P{int(period)}_{seconds:04d}"


def build_scene_from_spec(spec: dict, idsse_dir: Path | None = None, stride: int = 1) -> Scene:
    """Join one listed pipeline scene to its tracking window and roles."""

    metadata = match_metadata(spec["match_id"], idsse_dir)
    attacking_team_id = next((t for t, team in metadata.teams.items() if team.name == spec["attack_team"]), None)
    if attacking_team_id is None:
        raise ValueError(f"{spec['code']}: no team named {spec['attack_team']!r} in {spec['match_id']}")
    defending_team_id = next(t for t in metadata.teams if t != attacking_team_id)

    first_frame, last_frame, zero = int(spec["first_frame"]), int(spec["last_frame"]), int(spec["onset_frame"])
    window = load_window_cached(spec["match_id"], int(spec["period"]), first_frame, last_frame, idsse_dir)
    keep = slice(None, None, max(1, int(stride)))
    frame_ids = window.frame_ids[keep]
    times = (frame_ids - zero) / FPS  # 0 at the run onset

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
    for role in ("runner", "defender", "beneficiary"):
        if spec[f"{role}_id"] not in players:
            raise ValueError(f"{spec['code']}: the {role} {spec[f'{role}_id']} is not tracked in the window")

    ball_xy = window.ball_xy[keep]
    attack_name = metadata.teams[attacking_team_id].name
    defend_name = metadata.teams[defending_team_id].name
    _, clock = onset_clock(spec["match_id"], spec["period"], zero, idsse_dir)
    runner = players[spec["runner_id"]]
    return Scene(
        scene_id=spec["scene_id"],
        source="pipeline",
        fps=FPS / max(1, int(stride)),
        times=times,
        players=players,
        ball_xy=ball_xy,
        attacking_team_id=attacking_team_id,
        defending_team_id=defending_team_id,
        attacking_direction=_attacking_direction(players, attacking_team_id),
        runner_ids=(spec["runner_id"],),
        defender_ids=(spec["defender_id"],),
        beneficiary_ids=(spec["beneficiary_id"],),
        carrier_ids=tuple(_carrier_ids(players, ball_xy, attacking_team_id)),
        title=f"{attack_name} vs {defend_name}",
        subtitle=f"{clock}  ·  run onset {spec['code']}  ·  {runner.name}",
        attacking_team_name=attack_name,
        defending_team_name=defend_name,
        notes=spec.get("note", ""),
        frame_ids=frame_ids,
        moments={"run onset": 0.0},
        provenance={
            "adapter": "pipeline_window",
            "showcase_code": spec["code"],
            "match_id": spec["match_id"],
            "period": int(spec["period"]),
            "onset_frame_id": zero,
            "window_frames": [first_frame, last_frame],
            "source_fps": FPS,
            "stride": int(stride),
        },
    )
