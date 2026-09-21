"""IDSSE (DFL) raw-XML window loader.

``offball_value.bundesliga.load_bundesliga_frames`` already reads these files,
but it builds one ``BundesligaFrame`` per target frame via ElementTree, which
is more object churn than a demo needs when the positions file is ~400 MB.
This module keeps the *same coordinate and metadata conventions* (it reuses
``load_bundesliga_match_metadata`` verbatim) and only swaps in a line-streaming
scan for the position window, which reads a full match file in about a second.

The result is cached on disk as a compressed ``.npz`` so later renders never
touch the XML again.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import demo_paths, ensure_repo_on_path

ensure_repo_on_path()

from offball_value.bundesliga import (  # noqa: E402
    FPS,
    load_bundesliga_match_metadata,
    normalize_bundesliga_match_id,
    short_bundesliga_match_id,
)

#: First frame number of each game section in the IDSSE feed.  The files are
#: written with a fixed offset per half, which we verify on load rather than
#: assume.
_FRAMESET_RE = re.compile(
    r'<FrameSet\s+GameSection="([^"]+)"\s+MatchId="([^"]+)"\s+TeamId="([^"]+)"\s+PersonId="([^"]+)"'
)
_FRAME_RE = re.compile(
    r'<Frame N="(\d+)" T="([^"]+)" X="(-?[\d.]+)" Y="(-?[\d.]+)"'
    r'(?: D="(-?[\d.]+)")?(?: S="(-?[\d.]+)")?'
)

SECTION_BY_PERIOD = {1: "firstHalf", 2: "secondHalf"}


@dataclass(frozen=True)
class MatchFiles:
    """The three IDSSE XML files that belong to one match."""

    match_id: str
    matchinfo: Path
    positions: Path
    events: Path | None


def find_match_files(match_id: str, idsse_dir: Path | None = None) -> MatchFiles:
    """Locate a match's XML files, accepting either id form."""

    base = idsse_dir or demo_paths().idsse_dir
    if base is None:
        raise FileNotFoundError(
            "No IDSSE directory found. Set OFFBALL_IDSSE_DIR to the folder holding "
            "DFL_02_01_matchinformation_*.xml and DFL_04_03_positions_raw_observed_*.xml"
        )
    normalized = normalize_bundesliga_match_id(match_id)
    def one(pattern: str, required: bool = True) -> Path | None:
        hits = sorted(base.glob(pattern))
        if not hits:
            if required:
                raise FileNotFoundError(f"No file matching {pattern} under {base}")
            return None
        return hits[0]

    return MatchFiles(
        match_id=normalized,
        matchinfo=one(f"DFL_02_01_matchinformation_*_{normalized}.xml"),  # type: ignore[arg-type]
        positions=one(f"DFL_04_03_positions_raw_observed_*_{normalized}.xml"),  # type: ignore[arg-type]
        events=one(f"DFL_03_02_events_raw_*_{normalized}.xml", required=False),
    )


def section_start_frames(positions_xml: Path, cache_dir: Path | None = None) -> dict[str, int]:
    """First frame number of each half.

    The first ``FrameSet`` in the file is *not* a reliable half marker: a player
    who is substituted on at minute 60 has a second-half FrameSet that starts
    tens of thousands of frames late.  ``offball_value.bundesliga`` reads the
    clock from the BALL track for this reason, and so does this function - it
    takes the smallest frame number seen in each section, which is the ball's.
    Results are cached because it costs one full pass over a ~400 MB file.
    """

    cache_dir = cache_dir or (demo_paths().cache_dir / "clocks")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{positions_xml.stem}.json"
    if cache_path.exists():
        try:
            return {key: int(value) for key, value in json.loads(cache_path.read_text()).items()}
        except Exception:
            pass

    starts: dict[str, int] = {}
    ball_starts: dict[str, int] = {}
    with positions_xml.open("r", encoding="utf-8") as handle:
        current: tuple[str, str] | None = None
        pending = False
        for line in handle:
            if line.startswith("<FrameSet"):
                match = _FRAMESET_RE.match(line)
                current = (match.group(1), match.group(3)) if match else None
                pending = current is not None
            elif pending and line.startswith("<Frame "):
                section, team = current                       # type: ignore[misc]
                frame_id = int(line[10 : line.index('"', 10)])
                if section not in starts or frame_id < starts[section]:
                    starts[section] = frame_id
                if team == "BALL" and section not in ball_starts:
                    ball_starts[section] = frame_id
                pending = False
            elif line.startswith("</FrameSet"):
                current, pending = None, False
    starts.update(ball_starts)          # the ball is authoritative where present
    if "firstHalf" not in starts:
        raise ValueError(f"Could not read a firstHalf frame clock from {positions_xml}")
    cache_path.write_text(json.dumps(starts, indent=1))
    return starts


@dataclass
class TrackingWindow:
    """Raw per-object tracks over a frame window, in repo coordinates."""

    match_id: str
    period: int
    frame_ids: np.ndarray                 # (T,)
    fps: float
    tracks: dict[str, np.ndarray]         # object_id -> (T, 2) metres, NaN when absent
    speeds: dict[str, np.ndarray]         # object_id -> (T,) m/s
    team_of: dict[str, str]               # object_id -> team id, or "BALL"

    @property
    def ball_xy(self) -> np.ndarray:
        for object_id, team in self.team_of.items():
            if team == "BALL":
                return self.tracks[object_id]
        return np.full((len(self.frame_ids), 2), np.nan)


def read_window(
    positions_xml: Path,
    period: int,
    first_frame: int,
    last_frame: int,
    include_referees: bool = False,
) -> TrackingWindow:
    """Stream one positions file and keep only ``[first_frame, last_frame]``."""

    section = SECTION_BY_PERIOD[int(period)]
    n_frames = last_frame - first_frame + 1
    index_of = {frame: i for i, frame in enumerate(range(first_frame, last_frame + 1))}

    tracks: dict[str, np.ndarray] = {}
    speeds: dict[str, np.ndarray] = {}
    team_of: dict[str, str] = {}
    current: tuple[str, str, str] | None = None

    with positions_xml.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("<Frame "):
                if current is None:
                    continue
                frame_id = int(line[10 : line.index('"', 10)])
                slot = index_of.get(frame_id)
                if slot is None:
                    continue
                match = _FRAME_RE.match(line)
                if match is None:
                    continue
                object_id = current[2]
                array = tracks.get(object_id)
                if array is None:
                    array = np.full((n_frames, 2), np.nan)
                    tracks[object_id] = array
                    speeds[object_id] = np.full(n_frames, np.nan)
                    team_of[object_id] = current[1]
                array[slot, 0] = float(match.group(3))
                array[slot, 1] = float(match.group(4))
                if match.group(6) is not None:
                    # IDSSE reports S in km/h; the repo works in m/s.
                    speeds[object_id][slot] = float(match.group(6)) / 3.6
            elif line.startswith("<FrameSet"):
                match = _FRAMESET_RE.match(line)
                if match is None or match.group(1) != section:
                    current = None
                    continue
                team = match.group(3)
                if team != "BALL" and not team.startswith("DFL-CLU-") and not include_referees:
                    current = None
                    continue
                current = (match.group(1), team, match.group(4))
            elif line.startswith("</FrameSet"):
                current = None

    return TrackingWindow(
        match_id=short_bundesliga_match_id(positions_xml.stem.split("_")[-1]),
        period=int(period),
        frame_ids=np.arange(first_frame, last_frame + 1),
        fps=FPS,
        tracks=tracks,
        speeds=speeds,
        team_of=team_of,
    )


# ---------------------------------------------------------------------------
# disk cache
# ---------------------------------------------------------------------------
#: Bumped when the cache payload changes meaning. v2 stores float64: the feed
#: gives two decimal places, and float32 was quantising them by ~1e-6 m, which
#: is physically irrelevant but breaks exact agreement with the browser port.
_CACHE_VERSION = "v2"


def _cache_key(match_id: str, period: int, first: int, last: int) -> str:
    raw = f"{_CACHE_VERSION}|{match_id}|{period}|{first}|{last}"
    digest = hashlib.sha1(raw.encode()).hexdigest()[:10]
    return f"{short_bundesliga_match_id(match_id)}_P{period}_{first}_{last}_{digest}"


def load_window_cached(
    match_id: str,
    period: int,
    first_frame: int,
    last_frame: int,
    idsse_dir: Path | None = None,
    cache_dir: Path | None = None,
    refresh: bool = False,
) -> TrackingWindow:
    """Read a frame window, reusing the ``.npz`` cache when one exists."""

    paths = demo_paths()
    cache_dir = cache_dir or (paths.cache_dir / "windows")
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _cache_key(match_id, period, first_frame, last_frame)
    npz_path = cache_dir / f"{key}.npz"
    meta_path = cache_dir / f"{key}.json"

    if npz_path.exists() and meta_path.exists() and not refresh:
        meta = json.loads(meta_path.read_text())
        with np.load(npz_path) as blob:
            tracks = {
                name[len("xy_") :]: blob[name] for name in blob.files if name.startswith("xy_")
            }
            speeds = {
                name[len("sp_") :]: blob[name] for name in blob.files if name.startswith("sp_")
            }
            frame_ids = blob["frame_ids"]
        return TrackingWindow(
            match_id=meta["match_id"],
            period=meta["period"],
            frame_ids=frame_ids,
            fps=meta["fps"],
            tracks=tracks,
            speeds=speeds,
            team_of=meta["team_of"],
        )

    files = find_match_files(match_id, idsse_dir)
    window = read_window(files.positions, period, first_frame, last_frame)
    payload: dict[str, np.ndarray] = {"frame_ids": window.frame_ids}
    for object_id, array in window.tracks.items():
        payload[f"xy_{object_id}"] = array
        payload[f"sp_{object_id}"] = window.speeds[object_id]
    np.savez_compressed(npz_path, **payload)
    meta_path.write_text(
        json.dumps(
            {
                "match_id": short_bundesliga_match_id(files.match_id),
                "period": int(period),
                "fps": float(window.fps),
                "team_of": window.team_of,
                "first_frame": int(first_frame),
                "last_frame": int(last_frame),
                "source": str(files.positions),
            },
            indent=1,
        )
    )
    window.match_id = short_bundesliga_match_id(files.match_id)
    return window


def match_metadata(match_id: str, idsse_dir: Path | None = None):
    """Team and player metadata, straight from the repository loader."""

    return load_bundesliga_match_metadata(find_match_files(match_id, idsse_dir).matchinfo)
