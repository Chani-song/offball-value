"""Read the solver/evaluation bundle, and hand the adapter real numbers.

    local_inputs/offball_demo_data_20260930/

26 scenes, 69 real moments (each scene's 0 / 0.6 / 1.2 s), solved by the
upstream run at ``kyuhyeok-dev@8a5c69d``. Produced by
``scripts/analyze_eval.py`` and ``scripts/extract_panel_policy.py``; this
module only reads it.

LICENCE
-------
**The bundle carries licensed tracking.** ``tracking/*.csv`` and the raw
coordinates in it never leave this machine: they are used here for mapping and
validation only. What crosses into the public build is the derived material --
equilibrium probabilities, the solver's own 0.6 s paths, values, ranks,
regrets, the dilemma flags and the static comparison. ``DATA_BUNDLE_PROVENANCE.md``
says which is which, and ``tests/test_bundle_integration.py`` fails the build
if a tracking file or a raw-coordinate series reaches the site.

COORDINATES
-----------
Panels are corner-origin metres on the solver's pitch (0..105 x 0..68); this
repository's scenes are centre-origin and *unnormalised* (the browser flips
them at draw time). The conversion is therefore just a translation:

    x_scene = x_panel - 52.5      y_scene = y_panel - 34.0

Checked field for field against the published scene payloads: 0.0000 m on
every body of every moment (``test_bundle_integration``).

FRAMES
------
``scenes.csv`` gives each scene's ``clip_first_frame``; a panel gives the
moment's DFL ``start_frame``. So

    scene frame index = start_frame - clip_first_frame

which puts S05's three moments at 55, 70 and 85 -- 15 frames, 0.6 s apart.
"""

from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

#: The bundle this module reads, and the upstream it was produced by.
BUNDLE = "offball_demo_data_20260930"
UPSTREAM_SHA = "8a5c69d0bb1c42948c3f9d50becd2ffc41e72344"

#: Later bundles that add scenes, read after the first from the same folder.
#: Same solver, flags and files as the first, at the starts picked for them:
#: eval_batch2 (2026-10-03) S02 S03 S04 S08 S10 S35; eval_batch3 (2026-10-04)
#: S06 S27. A code an earlier bundle already covers is never replaced.
EXTRA_BUNDLES = ("offball_demo_data_batch2_20261003", "offball_demo_data_batch3_20261004")

DEFAULT_ROOT = (Path(__file__).resolve().parent.parent.parent
                / "local_inputs" / BUNDLE)

#: Corner-origin panel metres -> this repository's centre-origin scene metres.
HALF_LENGTH, HALF_WIDTH = 52.5, 34.0

#: The panel files predate upstream's English cleanup, so a defender's command
#: still carries its Korean name. This is the rename as `e89e78f` performed it
#: on `src/offball_value/stage3_read.py` -- recovered from that diff line by
#: line, not translated here. `test_bundle_integration` re-derives it from the
#: commit and fails if this table drifts.
COMMAND_EN = {
    "멈추기(감속)": "slow down",
    "옆으로": "sideways",
    "볼 쪽": "toward ball",
    "러너 쪽": "toward runner",
    "수혜자 쪽": "toward beneficiary",
    "골문 쪽": "toward goal",
}

#: The solver's role names, as this demo's story roles. The ball carrier is the
#: passer. A 3v1 game's "beneficiary" is the scripted-passer variant's second
#: attacker and has no story role here, so it is left out rather than forced
#: into one.
STORY_ROLE_OF = {
    "ball carrier": "passer",
    "runner": "runner",
    "defender": "defender",
}


def english(name: str) -> str:
    """A command's current upstream English name."""

    return COMMAND_EN.get(name, name)


def to_scene_xy(point: Sequence[float]) -> list[float]:
    return [float(point[0]) - HALF_LENGTH, float(point[1]) - HALF_WIDTH]


@dataclass(frozen=True)
class Moment:
    """One solved moment: its panel, its analysis row and where it sits."""

    code: str            # "S05"
    key: str             # "S05", "S05@0.6", "S05@1.2"
    dt: float
    frame: int           # scene frame index
    panel: dict
    analysis: dict
    bundle: str = BUNDLE  # which bundle the moment was read from


def available(root: Path | None = None) -> bool:
    root = Path(root or os.environ.get("OFFBALL_BUNDLE", DEFAULT_ROOT))
    return (root / "panels").is_dir() and (root / "analysis" / "moments.json").is_file()


def roots(root: Path | None = None) -> list[tuple[str, Path]]:
    """The installed bundles as (name, folder), the first bundle first.

    The later ones are looked for beside the first, so moving the first with
    ``OFFBALL_BUNDLE`` moves them too. One that is not installed is skipped.
    """

    first = Path(root or os.environ.get("OFFBALL_BUNDLE", DEFAULT_ROOT))
    found = [(BUNDLE, first)]
    for name in EXTRA_BUNDLES:
        if available(first.parent / name):
            found.append((name, first.parent / name))
    return found


class BundleEvaluationSource:
    """The bundle as an :class:`~.adapter.EvaluationSource`.

    Returns ``None`` for anything it does not cover, which the adapter turns
    into an explicit unavailable state -- a scene with no solved moment, a
    moment the run skipped, a role the game does not contain.
    """

    definition_version = f"kyuhyeok-dev@{UPSTREAM_SHA[:7]}"

    def __init__(self, root: Path | None = None, showcase=None):
        self.root = Path(root or os.environ.get("OFFBALL_BUNDLE", DEFAULT_ROOT))
        self.name = BUNDLE
        self._roots = roots(self.root)
        self._bundle_of: dict[str, tuple[str, Path]] = {}
        self._scenes = self._read_scenes()
        self._analysis = {}
        for _, folder in self._roots:
            for row in json.loads((folder / "analysis" / "moments.json").read_text()):
                self._analysis.setdefault(row["code"], row)
        self._by_scene = self._index(showcase)

    # -- loading ----------------------------------------------------------
    def _read_scenes(self) -> dict[str, dict]:
        """Every bundle's scenes; a code keeps the first bundle that has it."""

        scenes: dict[str, dict] = {}
        for name, folder in self._roots:
            with (folder / "scenes.csv").open(encoding="utf-8-sig") as handle:
                for row in csv.DictReader(handle):
                    if row["code"] not in scenes:
                        scenes[row["code"]] = row
                        self._bundle_of[row["code"]] = (name, folder)
        return scenes

    def _index(self, showcase) -> dict[str, list[Moment]]:
        """scene_id -> its solved moments, in time order.

        The join is the curated showcase: its ``showcase_id`` *is* the
        bundle's code, and only a verified mapping names a published scene. A
        code with no published scene (S53) is not indexed, so nothing can show
        a policy on a scene the demo does not publish.
        """

        if showcase is None:
            from ..core.showcase import load as load_showcase

            showcase = load_showcase() or ()
        out: dict[str, list[Moment]] = {}
        for entry in showcase:
            code = entry.showcase_id
            if not entry.scene_id or code not in self._scenes:
                continue
            first = int(self._scenes[code]["clip_first_frame"])
            name, folder = self._bundle_of[code]
            moments = []
            for key, dt in ((code, 0.0), (f"{code}@0.6", 0.6), (f"{code}@1.2", 1.2)):
                path = folder / "panels" / f"eval-{key}.json"
                if not path.exists():
                    continue            # a moment the run skipped: left out
                panel = json.loads(path.read_text())
                moments.append(Moment(
                    code=code, key=key, dt=dt,
                    frame=int(panel["start_frame"]) - first,
                    panel=panel, analysis=self._analysis.get(key, {}), bundle=name))
            if moments:
                out[entry.scene_id] = moments
        return out

    # -- what the adapter asks for ---------------------------------------
    def scenes(self) -> list[str]:
        return sorted(self._by_scene)

    def moments(self, scene_id: str) -> list[Moment]:
        return self._by_scene.get(scene_id, [])

    def name_for(self, scene_id: str) -> str:
        """The bundle a scene's solved moments come from; the first bundle's
        name for a scene with none."""

        moments = self._by_scene.get(scene_id)
        return moments[0].bundle if moments else self.name

    def _at(self, scene_id: str, frame: int | None) -> Moment | None:
        """The solved moment at `frame`, or the first when none is asked for.

        Only an *exact* frame matches. Between solved moments there is no
        solved game, and returning a neighbour's numbers would present one
        moment's evaluation as another's.
        """

        moments = self._by_scene.get(scene_id)
        if not moments:
            return None
        if frame is None:
            return moments[0]
        for moment in moments:
            if moment.frame == frame:
                return moment
        return None

    def _player(self, moment: Moment, role: str) -> dict | None:
        """The analysis row for a story role at this moment."""

        for entry in moment.analysis.get("players", ()):
            if STORY_ROLE_OF.get(entry.get("role")) == role:
                return entry
        return None

    def _options(self, moment: Moment, role: str) -> list[dict] | None:
        for body in moment.panel.get("bodies", {}).values():
            if STORY_ROLE_OF.get(body.get("role")) == role:
                return body.get("options")
        return None

    def evaluation(self, scene_id: str, role: str,
                   frame: int | None) -> Mapping[str, Any] | None:
        moment = self._at(scene_id, frame)
        if moment is None:
            return None
        row = self._player(moment, role)
        if row is None:
            return None
        options = self._options(moment, role) or []
        observed = row.get("observed")
        out: dict[str, Any] = {}
        if row.get("rank") is not None:
            out["observed_action_rank"] = {"value": int(row["rank"])}
        if row.get("rank_frac") is not None:
            out["relative_rank"] = {"value": float(row["rank_frac"])}
        if row.get("similarity") is not None:
            out["similarity_to_optimal"] = {"value": float(row["similarity"])}
        if row.get("regret") is not None:
            out["regret"] = {"value": float(row["regret"])}
        if isinstance(observed, int) and 0 <= observed < len(options):
            option = options[observed]
            fit = row.get("fit_m") or [None, None]
            out["observed_action"] = {
                "action_id": f"command_{observed}",
                "kind": "move",
                "description": english(option.get("name", "")),
                "target": to_scene_xy(option["end"]),
                "projection_distance_m": fit[0],
                "confidence": None,
                "provenance": "evaluation_pipeline",
            }
        return out or None

    def counterfactual(self, scene_id: str, role: str,
                       frame: int | None) -> Mapping[str, Any] | None:
        """The static comparison upstream actually computes at this moment.

        `analyze_eval`: the defender is held to his observed command, the
        attack's best column there *appears* worth `M[d_obs, a*]`; once the
        defender may answer it is worth `min_d M[d, a*]`. The difference is the
        loss. These are **not** `static_counterfactual.py`'s whole-window
        V / S / R / A, which this bundle does not contain -- those slots stay
        unavailable.
        """

        moment = self._at(scene_id, frame)
        if moment is None or not moment.analysis:
            return None
        a = moment.analysis
        if a.get("static_attack_appeared") is None:
            return None
        out = {
            "static": {"semantics": "the defender held to his observed command; "
                                    "the attack's best answer to it",
                       "value": {"value": float(a["static_attack_appeared"])}},
            "responsive": {"semantics": "the same attacking choice once the "
                                        "defender may answer it",
                           "value": {"value": float(a["static_attack_responsive"])}},
        }
        if a.get("static_attack_loss") is not None:
            out["value_change"] = {"value": -float(a["static_attack_loss"])}
        observed = self.evaluation(scene_id, role, frame) or {}
        if "observed_action" in observed:
            out["observed_action"] = observed["observed_action"]
        return out

    #: What a frame series carries, and its natural range.
    #:
    #: Every metric the panel shows needs one: the interface switches a row to
    #: the solved moment nearest the playhead *through its series*, so a metric
    #: without one would sit at the first moment's value beside rows that had
    #: moved -- rank 1 next to a relative rank of 0.25.
    SERIES = (("observed_action_rank", "rank", None),
              ("relative_rank", "rank_frac", (0.0, 1.0)),
              ("similarity_to_optimal", "similarity", (0.0, 1.0)),
              ("regret", "regret", None))

    def frame_series(self, scene_id: str, role: str) -> Mapping[str, Any] | None:
        moments = self._by_scene.get(scene_id)
        if not moments:
            return None
        out: dict[str, Any] = {}
        for name, key, domain in self.SERIES:
            frames, values = [], []
            for moment in moments:
                row = self._player(moment, role)
                if row is None or row.get(key) is None:
                    continue
                frames.append(moment.frame)
                values.append(float(row[key]))
            if frames:
                entry = {"frames": frames, "values": values}
                if domain:
                    entry["domain"] = list(domain)
                out[name] = entry        # interpolate stays false: three solves
        return out or None

    def clip_metrics(self, scene_id: str, role: str):
        """Nothing. `analysis/summary.json` aggregates over *moments* for a
        corpus summary -- mean / median / max across 69 moments and 26 scenes,
        grouped by game kind. That is not a per-clip player score, and this
        module will not invent one."""

        return None


# ---------------------------------------------------------------------------
# the Game solution payload
# ---------------------------------------------------------------------------
def panel_payload(source: BundleEvaluationSource, scene_id: str) -> dict | None:
    """One scene's solved moments, in the shape the browser's solver layer reads.

    Carries the equilibrium policy and **the solver's own 0.6 s path** per
    command, converted to scene coordinates. No tracking: every coordinate
    here is solver output or a body's start, which the published scene already
    contains.
    """

    moments = source.moments(scene_id)
    if not moments:
        return None
    panels = []
    for moment in moments:
        panel, analysis = moment.panel, moment.analysis
        bodies = {}
        for body in panel.get("bodies", {}).values():
            # The panel payload draws every body the game contains, keyed by
            # its story role where it has one and by its solver role where it
            # does not. A 3v1 game's beneficiary is the second attacker --
            # Figure 2's "Teammate" -- and has a real equilibrium policy; the
            # story schema has no role for him, but the pitch still shows him
            # rather than drawing two thirds of the equilibrium.
            role = STORY_ROLE_OF.get(body.get("role"), body.get("role"))
            bodies[role] = {
                "solver_role": body["role"],
                "pos": to_scene_xy(body["pos"]),
                "options": [{
                    "command": i,
                    "label": english(o.get("name", "")),
                    "legacy_label": o.get("name", ""),
                    "prob": float(o.get("prob", 0.0)),
                    "path": [to_scene_xy(p) for p in (o.get("path") or ())],
                    "end": to_scene_xy(o["end"]),
                    "aim": [float(v) for v in (o.get("aim") or (0, 0))],
                    # render_figure2_abstract: a stop that reached speed 0
                    # inside the 0.6 s is "Stop", one still braking is "Slow
                    # down". The test is exact equality with zero, as upstream.
                    "rests": (math.hypot(*(o.get("end_velocity") or (1.0, 0.0)))
                              == 0.0),
                } for i, o in enumerate(body.get("options", ()))],
            }
        release = panel.get("release") or {}
        panels.append({
            "dt": moment.dt,
            "frame": moment.frame,
            "value": float(panel["value"]),
            "kind": panel.get("kind"),
            "bodies": bodies,
            "release": {"prob": float(release.get("prob", 0.0)),
                        "to": release.get("to"),
                        "target": to_scene_xy(release["target"]) if release.get("target") else None},
            "passes": [{"prob": float(q.get("prob", 0.0)), "to": q.get("to"),
                        "where": q.get("where"),
                        "target": to_scene_xy(q["target"]) if q.get("target") else None}
                       for q in panel.get("passes", ())],
            "dilemma": {
                "defender_mixed": bool(analysis.get("defender_mixed")),
                "attack_mixed": bool(analysis.get("attack_mixed")),
                "no_saddle": bool(analysis.get("no_saddle")),
                "saddle_gap": analysis.get("saddle_gap"),
                # upstream's own criterion: the defender mixes AND there is no
                # saddle point. Not recomputed anywhere else.
                "is_dilemma": bool(analysis.get("defender_mixed")
                                   and analysis.get("no_saddle")),
            },
            "static": {k: analysis.get(k) for k in
                       ("static_attack_appeared", "static_attack_responsive",
                        "static_attack_loss", "static_attack_loss_rel")},
        })
    first = moments[0].panel.get("provenance", {})
    return {
        "available": True,
        "kind": "bundle_panels",
        "scene_id": scene_id,
        "code": moments[0].code,
        "label": f"{moments[0].code} · solved at {', '.join(f'{m.dt:.1f} s' for m in moments)}",
        "caveat": "",
        "panels": panels,
        "provenance": {
            "bundle": moments[0].bundle,
            "upstream": UPSTREAM_SHA,
            "repository": "Chani-song/offball-value@kyuhyeok-dev",
            "script": "analyze_eval.py + extract_panel_policy.py",
            "match": first.get("match_label"),
            "pass_model": moments[0].panel.get("pass_model"),
        },
    }


@lru_cache(maxsize=1)
def source() -> BundleEvaluationSource | None:
    """The bundle if it is installed here, else None."""

    return BundleEvaluationSource() if available() else None
