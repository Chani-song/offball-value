"""Where every non-strategic player actually was at each decision instant.

The solved game re-decides at t = 0, step, 2*step, ... after the run's onset.
Background players make no decision, so all the game needs from them is a
position at each of those instants. It comes from the tracking.

Frames in our scene payloads are 0.08 s apart and centred on the onset, so an
instant such as 1.0 s usually falls between two frames; positions are linearly
interpolated between the frames either side. The clip ends at +2.96 s, so the
3.0 s instant is taken from the last frame -- 0.04 s early, under 0.3 m at a
sprint -- and the time actually used is recorded next to every position.

Coordinates are converted to the solver's corner origin here, once.

`background_tracks` is a copy of andrew-passer2on1's; `player_track` is added
here for the passer, who in the 3v1 game is scripted the same way.
"""

from __future__ import annotations

import numpy as np

FIELD_LENGTH, FIELD_WIDTH = 105.0, 68.0


def _series(frames, t0):
    """(relative times, {player_id: array[frames, 2] or None}) in corner origin."""
    times = np.array([float(f["relative_time_s"]) - t0 for f in frames])
    ids = {str(p[0]) for f in frames for p in f["players"]}
    tracks = {pid: np.full((len(frames), 2), np.nan) for pid in ids}
    for i, f in enumerate(frames):
        for p in f["players"]:
            tracks[str(p[0])][i] = (float(p[2]) + FIELD_LENGTH / 2.0,
                                    float(p[3]) + FIELD_WIDTH / 2.0)
    return times, tracks


def _at(times, track, t):
    """Linear interpolation over the frames where this player is present."""
    ok = np.isfinite(track[:, 0])
    if ok.sum() < 2:
        return None
    tt, xy = times[ok], track[ok]
    t = float(np.clip(t, tt[0], tt[-1]))
    return np.array([np.interp(t, tt, xy[:, 0]), np.interp(t, tt, xy[:, 1])])


def background_tracks(payload: dict, strategic_ids, steps: int, step_seconds: float,
                      *, velocity_half_window: float = 0.16) -> dict:
    """Every defending player not in `strategic_ids`, at each decision instant.

    Returns positions and velocities [steps + 1, m, 2] (corner origin), the ids
    and names in that order, the instants requested and the instants actually
    covered by the clip, and any defender left out for lack of tracking.
    """
    frames = payload["background_frames"]
    onset = payload["onset_frame_id"]
    at = min(frames, key=lambda f: abs(f["frame_id"] - onset))
    t0 = float(at["relative_time_s"])
    attacking = str(payload["attacking_team_id"])
    strategic = {str(s) for s in strategic_ids}
    names = {str(p[0]): p[4] for p in at["players"]}
    candidates = [str(p[0]) for p in at["players"]
                  if str(p[1]) != attacking and str(p[0]) not in strategic]
    times, tracks = _series(frames, t0)
    wanted = [k * step_seconds for k in range(steps + 1)]
    used = [float(np.clip(t, times.min(), times.max())) for t in wanted]

    ids, positions, velocities, dropped = [], [], [], []
    for pid in candidates:
        rows, vels = [], []
        for t in used:
            p = _at(times, tracks[pid], t)
            a = _at(times, tracks[pid], t - velocity_half_window)
            b = _at(times, tracks[pid], t + velocity_half_window)
            if p is None or a is None or b is None:
                rows = None
                break
            rows.append(p)
            vels.append((b - a) / (2 * velocity_half_window))
        if rows is None:
            dropped.append(pid)
            continue
        ids.append(pid)
        positions.append(rows)
        velocities.append(vels)
    shape = (steps + 1, len(ids), 2)
    pos = np.transpose(np.array(positions), (1, 0, 2)) if ids else np.zeros(shape)
    vel = np.transpose(np.array(velocities), (1, 0, 2)) if ids else np.zeros(shape)
    return {
        "ids": ids, "names": [names.get(i, "") for i in ids],
        "positions": pos, "velocities": vel,
        "times_requested": wanted, "times_used": used, "dropped": dropped,
    }


def player_track(payload: dict, player_id: str, steps: int, step_seconds: float,
                 *, velocity_half_window: float = 0.16) -> dict:
    """One player's position and velocity at each decision instant.

    Same interpolation and clamping as `background_tracks`, so the scripted
    passer and the scripted defenders are read off the tracking identically.
    """
    frames = payload["background_frames"]
    onset = payload["onset_frame_id"]
    at = min(frames, key=lambda f: abs(f["frame_id"] - onset))
    t0 = float(at["relative_time_s"])
    times, tracks = _series(frames, t0)
    pid = str(player_id)
    if pid not in tracks:
        raise ValueError(f"player {pid} not in the scene's tracking")
    wanted = [k * step_seconds for k in range(steps + 1)]
    used = [float(np.clip(t, times.min(), times.max())) for t in wanted]
    positions, velocities = [], []
    for t in used:
        p = _at(times, tracks[pid], t)
        a = _at(times, tracks[pid], t - velocity_half_window)
        b = _at(times, tracks[pid], t + velocity_half_window)
        if p is None or a is None or b is None:
            raise ValueError(f"player {pid} lacks tracking around t = {t:.2f} s")
        positions.append(p)
        velocities.append((b - a) / (2 * velocity_half_window))
    return {"positions": np.array(positions), "velocities": np.array(velocities),
            "times_requested": wanted, "times_used": used}
