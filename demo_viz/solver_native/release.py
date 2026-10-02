"""The solver's own release quantities, computed by the research code itself.

This is the chain the current solver actually uses, traced in
``demo_viz/PASS_MODEL_TRACE.md`` and reproduced here without redefining any of
it:

    legal  =  inside_pitch(target)  and not offside(receiver, ...)
    completion_proxy  =  ExpectedPass.predict(...)      33-feature logistic
    positional_threat =  positional_threat_all(target, carrier, defenders)
    release_payoff    =  legal ? completion_proxy * positional_threat : 0

Every term comes from the imported implementation. The feature builder is
Andrew's ``pass_features``; the threat and the offside rule are Kyuhyeok's
``positional_threat_all`` and ``offside_flags``, vendored byte-identical beside
this file. Nothing is ported to JavaScript: the browser is handed numbers this
code produced.

OPTIONAL BY DESIGN
------------------
Two things live outside this repository and are not committed: Andrew's
``defensive_positioning`` package, and the fitted ``experimental_pass.json``.
:func:`availability` reports what is missing and every entry point degrades to
``None`` rather than raising, so a fresh public checkout still builds a site --
it simply has no solver-pass payload.

COORDINATES
-----------
This repository puts the origin on the centre spot. The imported code puts it
on a corner. Conversion happens once, here, exactly as
``ssac_stack._to_corner_origin`` does it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import numpy as np

FIELD_LENGTH, FIELD_WIDTH = 105.0, 68.0
GROUND, DRIVEN, LOFTED = 0, 1, 2

#: Where the research stack and the fitted model are looked for. Both may be
#: overridden, and both may simply be absent.
DEFAULT_RESEARCH_SRC = Path.home() / "Research" / "offball_demo" / "mit_ssac2027_ref" / "src"
DEFAULT_MODEL = (Path(__file__).resolve().parent.parent.parent
                 / "local_inputs" / "models" / "experimental_pass.json")


def research_src() -> Path:
    return Path(os.environ.get("OFFBALL_RESEARCH_SRC", DEFAULT_RESEARCH_SRC)).expanduser()


def model_path() -> Path:
    return Path(os.environ.get("OFFBALL_PASS_MODEL", DEFAULT_MODEL)).expanduser()


@dataclass(frozen=True)
class Availability:
    ok: bool
    reason: str
    research: bool
    model: bool


def availability() -> Availability:
    """Whether the solver-native chain can run here, and what is missing."""

    research = (research_src() / "defensive_positioning").is_dir()
    model = model_path().is_file()
    if research and model:
        return Availability(True, "", True, True)
    missing = []
    if not research:
        missing.append(f"defensive_positioning package (looked in {research_src()})")
    if not model:
        missing.append(f"experimental_pass.json (looked at {model_path()})")
    return Availability(False, "; ".join(missing), research, model)


@lru_cache(maxsize=1)
def _stack():
    """Import the research stack once, or return ``None`` when it is absent."""

    import sys

    state = availability()
    if not state.ok:
        return None
    path = str(research_src())
    if path not in sys.path:
        sys.path.insert(0, path)
    try:
        from defensive_positioning.expected_pass import (
            ExpectedPass,
            POSITION_FEATURES,
            position_features,
        )
        from defensive_positioning.models import DEFAULT_PASSES, PlayerState, Scenario
        from defensive_positioning.rules import inside_pitch

        from ._passer2on1_payoff import offside_flags, positional_threat_all
    except Exception:                                      # pragma: no cover
        return None
    model = ExpectedPass.load(model_path(), allow_proxy=True)
    return {
        "model": model, "features": POSITION_FEATURES,
        "position_features": position_features, "inside_pitch": inside_pitch,
        "offside_flags": offside_flags, "threat_all": positional_threat_all,
        "passes": DEFAULT_PASSES, "PlayerState": PlayerState, "Scenario": Scenario,
    }


def _corner(points):
    """Centre-spot origin -> corner origin, the imported code's convention."""

    return np.asarray(points, dtype=float) + np.array([FIELD_LENGTH / 2.0, FIELD_WIDTH / 2.0])


@lru_cache(maxsize=4)
def _scenario(attacking_direction: int):
    """A Scenario carries only the pitch and direction for these calls.

    Its three players are required by the constructor and unused by
    ``positional_threat``; they sit on the centre spot so the constructor's
    inside-the-pitch check passes. This mirrors ``ssac_stack._scenario``.
    """

    stack = _stack()
    centre = (FIELD_LENGTH / 2.0, FIELD_WIDTH / 2.0)
    player = stack["PlayerState"](position=centre, velocity=(0.0, 0.0))
    return stack["Scenario"](carrier=player, receiver=player, defender=player,
                             pitch_length=FIELD_LENGTH, pitch_width=FIELD_WIDTH,
                             attack_direction=int(attacking_direction))


#: The subset of the 33 features the panel shows, with the public wording from
#: PASS_MODEL_TRACE.md section 10. Definitions are the traced ones.
PANEL_FEATURES = (
    ("pass_length", "Pass length", "m"),
    ("forward_distance", "Forward distance", "m"),
    ("receiver_target_gap", "Receiver-target gap", "m"),
    ("defender_near_passer_passer_pressure", "Passer pressure", ""),
    ("defender_near_receiver_receiver_pressure", "Receiver pressure", ""),
    ("defender_near_passer_lane_pressure", "Lane pressure", ""),
    ("same_defender", "Same defender", ""),
)


def release_quantities(
    carrier_xy: Sequence[float],
    receiver_xy: Sequence[float],
    target_xy: Sequence[float],
    defender_xys: Sequence[Sequence[float]],
    attacking_direction: int,
    family: int = GROUND,
) -> dict | None:
    """The four solver quantities for one release target, in this repo's frame.

    ``defender_xys`` must be the **whole defending side**: the model selects its
    own two defender roles from whatever it is handed, and Kyuhyeok's wrapper
    hands it the controlled defender plus the background. Passing a single
    defender reproduces Andrew's reduced 2v1 convention instead, which is not
    what the meeting pipeline ran.

    Returns ``None`` when the research stack or the fitted model is absent.
    """

    stack = _stack()
    if stack is None:
        return None
    defenders = np.asarray(list(defender_xys), dtype=float)
    if defenders.ndim != 2 or defenders.shape[0] == 0:
        raise ValueError("supply at least one defender position")

    carrier = _corner(carrier_xy)
    receiver = _corner(receiver_xy)
    target = _corner(target_xy)
    defence = _corner(defenders)
    scenario = _scenario(int(attacking_direction))

    inside = bool(stack["inside_pitch"](target, FIELD_LENGTH, FIELD_WIDTH))
    offside = bool(np.asarray(
        stack["offside_flags"](scenario, receiver, carrier, defence[None, ...])).reshape(-1)[0])
    legal = inside and not offside

    features = np.asarray(stack["position_features"](
        carrier, receiver, defence, target, int(family),
        int(attacking_direction), FIELD_LENGTH, FIELD_WIDTH), dtype=float)
    completion = float(stack["model"].probability(features))
    threat = float(np.asarray(
        stack["threat_all"](target, carrier, defence, scenario)).reshape(-1)[0])
    payoff = completion * threat if legal else 0.0

    names = list(stack["features"])
    lookup = dict(zip(names, features.tolist()))
    return {
        "legal": legal,
        "inside_pitch": inside,
        "offside": offside,
        "completion_proxy": completion,
        "positional_threat": threat,
        "release_payoff": payoff,
        "features": {name: lookup[name] for name, _, _ in PANEL_FEATURES},
        "all_features": lookup,
    }


def solver_release_targets(receiver_xy: Sequence[float], attacking_direction: int):
    """The solver's own 18 release choices: ``receiver + direction * offset``.

    These are *not* the demo's exploratory rays. ``DEFAULT_PASSES`` is read
    from the imported module so the set cannot drift from the solver's.
    """

    stack = _stack()
    if stack is None:
        return None
    receiver = np.asarray(receiver_xy, dtype=float)
    out = []
    for choice in stack["passes"]:
        target = receiver + int(attacking_direction) * np.asarray(choice.offset, dtype=float)
        out.append({"offset": tuple(float(v) for v in choice.offset),
                    "family": choice.family,
                    "target": (float(target[0]), float(target[1]))})
    return out
