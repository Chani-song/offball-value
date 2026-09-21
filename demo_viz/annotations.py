"""Adapter for the manual annotation spreadsheet ``shot_annotations.xlsx``.

The spreadsheet is the human ground truth for which scenes are worth showing
and which shirt numbers play which role in the causal story.  Roles are stored
as free-text shirt-number lists (``"23"``, ``"6, 34"``, ``"9,25"``), so parsing
is deliberately forgiving.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import pandas as pd

from .config import demo_paths

EFFECT_ORDER = {"strong": 0, "medium": 1, "low": 2, "ignore": 3}

REQUIRED_COLUMNS = (
    "clip_id",
    "match_id",
    "period",
    "period_seconds",
    "offball_attackers",
    "drawn_defenders",
    "space_beneficiaries",
    "effect",
)


def parse_shirt_list(value: object) -> tuple[str, ...]:
    """``"6, 34"`` -> ``("6", "34")``; blanks and NaN -> ``()``."""

    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ()
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "-"}:
        return ()
    return tuple(token for token in re.split(r"[^0-9]+", text) if token)


@dataclass(frozen=True)
class AnnotatedClip:
    """One reviewed row of ``shot_annotations.xlsx``."""

    clip_id: str
    match_id: str
    shot_number: int | None
    period: int
    period_seconds: float
    match_clock: str
    team: str
    shooter: str
    shot_result: str
    video_filename: str
    runner_shirts: tuple[str, ...]
    defender_shirts: tuple[str, ...]
    beneficiary_shirts: tuple[str, ...]
    effect: str
    notes: str

    @property
    def is_renderable(self) -> bool:
        return bool(self.runner_shirts and self.defender_shirts and self.beneficiary_shirts)

    @property
    def shape(self) -> str:
        return (
            f"{len(self.runner_shirts)}R-"
            f"{len(self.defender_shirts)}D-"
            f"{len(self.beneficiary_shirts)}B"
        )

    def summary(self) -> str:
        return (
            f"{self.clip_id:<28} {self.effect:<6} {self.shape:<8} "
            f"runner={','.join(self.runner_shirts) or '-':<7} "
            f"defenders={','.join(self.defender_shirts) or '-':<7} "
            f"beneficiaries={','.join(self.beneficiary_shirts) or '-'}"
        )


def _clean(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def load_annotations(path: Path | None = None) -> list[AnnotatedClip]:
    """Every row of the spreadsheet, as records."""

    path = path or demo_paths().annotations_xlsx
    if path is None:
        raise FileNotFoundError(
            "shot_annotations.xlsx not found. Set OFFBALL_ANNOTATIONS to its location."
        )
    frame = pd.read_excel(path)
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing expected columns: {missing}")

    clips: list[AnnotatedClip] = []
    for _, row in frame.iterrows():
        try:
            period = int(float(row["period"]))
            period_seconds = float(row["period_seconds"])
        except (TypeError, ValueError):
            continue
        shot_number = None
        try:
            shot_number = int(float(row.get("shot_number")))
        except (TypeError, ValueError):
            pass
        clips.append(
            AnnotatedClip(
                clip_id=_clean(row["clip_id"]),
                match_id=_clean(row["match_id"]),
                shot_number=shot_number,
                period=period,
                period_seconds=period_seconds,
                match_clock=_clean(row.get("match_clock")),
                team=_clean(row.get("team")),
                shooter=_clean(row.get("shooter")),
                shot_result=_clean(row.get("shot_result")),
                video_filename=_clean(row.get("video_filename")),
                runner_shirts=parse_shirt_list(row.get("offball_attackers")),
                defender_shirts=parse_shirt_list(row.get("drawn_defenders")),
                beneficiary_shirts=parse_shirt_list(row.get("space_beneficiaries")),
                effect=_clean(row.get("effect")).lower(),
                notes=_clean(row.get("notes")),
            )
        )
    return clips


def select_clips(
    effect: str | tuple[str, ...] = "strong",
    path: Path | None = None,
    renderable_only: bool = True,
) -> list[AnnotatedClip]:
    """Rows with the given effect label, optionally only complete triplets."""

    wanted = (effect,) if isinstance(effect, str) else tuple(effect)
    wanted_set = {value.lower() for value in wanted}
    clips = [clip for clip in load_annotations(path) if clip.effect in wanted_set]
    if renderable_only:
        clips = [clip for clip in clips if clip.is_renderable]
    clips.sort(key=lambda c: (EFFECT_ORDER.get(c.effect, 9), c.clip_id))
    return clips


def find_clip(clip_id: str, path: Path | None = None) -> AnnotatedClip:
    """One row by clip id, accepting a partial match on the suffix."""

    wanted = clip_id.strip()
    for clip in load_annotations(path):
        if clip.clip_id == wanted or clip.clip_id.endswith(wanted) or clip.video_filename.startswith(wanted):
            return clip
    raise KeyError(f"clip_id {clip_id!r} is not in the annotation spreadsheet")
