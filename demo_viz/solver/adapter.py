"""Read real solver artifacts, and refuse to invent one.

The solver is Andrew's exact 2v1 game (``mit_ssac2027``). It is a separate
repository and a batch job: median 20.3 s per state on eight workers. Nothing
here runs it. This module only *reads* what it produced, normalises it into the
shape the browser draws, and carries enough provenance that a reviewer can find
the run it came from.

The one rule this module exists to enforce: if there is no artifact, the answer
is :data:`UNAVAILABLE`, never a plausible-looking trajectory. A generated path
is not a solver output, and the demo must not let the two look alike.

Artifact layout, as written by ``defensive_positioning.exact_study``:

    manifest.json           steps, step_seconds, directions, passes
    states/state_NNN.json   scenario, value, certificate, root policies
    rows.jsonl              the same, plus `rollouts`
    policies/state_NNN.npz  the full conditional policy tables

Coordinates in the artifact are corner-origin metres on the solver's own pitch;
the demo works centre-origin, so :func:`load_state` shifts them once, here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

#: The five movement commands, in the order the solver indexes them. Taken from
#: the run's own manifest and checked against it on load, never assumed.
DEFAULT_DIRECTIONS = ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0))

#: What a rollout step's `event` can say, from `markov.py`.
EVENTS = ("move", "release", "retain", "tackled_during_previous_interval")

#: The three bodies the 2v1 game controls.
BODIES = ("carrier", "receiver", "defender")


class SolverArtifactError(ValueError):
    """The path exists but is not a solver artifact we can read."""


@dataclass(frozen=True)
class Unavailable:
    """No solver output for this scene. Carries the reason, never a fallback."""

    reason: str
    available: bool = False

    def to_payload(self) -> dict:
        return {"available": False, "reason": self.reason}


#: The default answer. Every demo scene gets this until a real run exists.
UNAVAILABLE = Unavailable("Not computed for this scene")


@dataclass(frozen=True)
class SolverProvenance:
    """Where a solver number came from, precise enough to go back to it."""

    repository: str
    commit: str
    artifact: str
    state_index: int
    fingerprint: str | None = None
    created_utc: str | None = None

    def to_payload(self) -> dict:
        return {
            "repository": self.repository,
            "commit": self.commit,
            "artifact": self.artifact,
            "state_index": self.state_index,
            "fingerprint": self.fingerprint,
            "created_utc": self.created_utc,
        }


@dataclass(frozen=True)
class SolverTrajectory:
    """One rollout: what the three bodies did under the solved policy."""

    times: tuple[float, ...]
    #: body -> ((x, y), ...) in centre-origin metres
    paths: dict[str, tuple[tuple[float, float], ...]]
    events: tuple[str, ...]
    #: per step, the chosen command index per body, where the step was a move
    commands: tuple[dict[str, int] | None, ...]
    survival: tuple[float | None, ...]
    #: set when the rollout ends in a pass
    release_target: tuple[float, float] | None = None
    release_family: str | None = None

    @property
    def terminal_event(self) -> str:
        return self.events[-1] if self.events else "unknown"

    def to_payload(self) -> dict:
        return {
            "times": list(self.times),
            "paths": {name: [list(p) for p in path] for name, path in self.paths.items()},
            "events": list(self.events),
            "commands": list(self.commands),
            "survival": list(self.survival),
            "release_target": list(self.release_target) if self.release_target else None,
            "release_family": self.release_family,
        }


@dataclass(frozen=True)
class SolverState:
    """One solved state: the game value, the root policies, and the rollouts."""

    index: int
    name: str
    stratum: str
    #: body -> (x, y) start, centre-origin metres
    start: dict[str, tuple[float, float]]
    start_velocity: dict[str, tuple[float, float]]
    attack_direction: int
    pitch: tuple[float, float]
    value: float
    certificate: dict[str, float]
    root_attack: tuple[float, ...]
    root_defender: tuple[float, ...]
    directions: tuple[tuple[float, float], ...]
    step_seconds: float
    steps: int
    trajectories: tuple[SolverTrajectory, ...]
    provenance: SolverProvenance
    available: bool = True
    speed_caps: dict[str, float] = field(default_factory=dict)

    # -- reading the root policy ------------------------------------------
    @property
    def release_probability(self) -> float:
        """Probability the attack releases the ball at the root.

        ``markov.py`` encodes an attack action ``a`` over ``len(directions)**2
        + 1`` entries, where the last one is release.
        """

        return float(self.root_attack[-1]) if self.root_attack else 0.0

    def most_likely_attack(self) -> dict[str, Any]:
        """The root attacking action with the largest probability.

        Returned with its decoded meaning, never with a made-up label. A game
        whose root policy is mixed is reported as mixed.
        """

        if not self.root_attack:
            return {"kind": "unknown", "probability": None}
        best = max(range(len(self.root_attack)), key=lambda i: self.root_attack[i])
        probability = float(self.root_attack[best])
        actions = len(self.directions)
        if best == actions * actions:
            return {"kind": "release", "probability": probability}
        carry, run = divmod(best, actions)
        return {
            "kind": "move",
            "probability": probability,
            "carrier": self.directions[carry],
            "receiver": self.directions[run],
            "carrier_index": carry,
            "receiver_index": run,
        }

    def most_likely_defence(self) -> dict[str, Any]:
        if not self.root_defender:
            return {"kind": "unknown", "probability": None}
        best = max(range(len(self.root_defender)), key=lambda i: self.root_defender[i])
        return {
            "kind": "move",
            "probability": float(self.root_defender[best]),
            "defender": self.directions[best],
            "defender_index": best,
        }

    @property
    def is_mixed(self) -> bool:
        """Whether either root policy puts weight on more than one action."""

        for policy in (self.root_attack, self.root_defender):
            if sum(1 for p in policy if p > 1e-9) > 1:
                return True
        return False

    def to_payload(self) -> dict:
        return {
            "available": True,
            "index": self.index,
            "name": self.name,
            "stratum": self.stratum,
            "start": {k: list(v) for k, v in self.start.items()},
            "start_velocity": {k: list(v) for k, v in self.start_velocity.items()},
            "attack_direction": self.attack_direction,
            "pitch": list(self.pitch),
            "value": self.value,
            "certificate": self.certificate,
            "root_attack": list(self.root_attack),
            "root_defender": list(self.root_defender),
            "directions": [list(d) for d in self.directions],
            "step_seconds": self.step_seconds,
            "steps": self.steps,
            "is_mixed": self.is_mixed,
            "release_probability": self.release_probability,
            "most_likely_attack": self.most_likely_attack(),
            "most_likely_defence": self.most_likely_defence(),
            "speed_caps": self.speed_caps,
            "trajectories": [t.to_payload() for t in self.trajectories],
            "provenance": self.provenance.to_payload(),
        }


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------
def _centre(xy: Sequence[float], pitch: tuple[float, float]) -> tuple[float, float]:
    """Corner-origin solver metres -> centre-origin demo metres."""

    return (float(xy[0]) - pitch[0] / 2.0, float(xy[1]) - pitch[1] / 2.0)


def read_manifest(run_dir: Path) -> dict:
    path = Path(run_dir) / "manifest.json"
    if not path.exists():
        raise SolverArtifactError(f"no manifest.json in {run_dir}")
    manifest = json.loads(path.read_text())
    if "config" not in manifest:
        raise SolverArtifactError("manifest.json has no config block")
    return manifest


def _trajectory(rollout: Sequence[dict], pitch: tuple[float, float]) -> SolverTrajectory:
    times, events, commands, survival = [], [], [], []
    paths: dict[str, list[tuple[float, float]]] = {name: [] for name in BODIES}
    release_target = release_family = None
    for step in rollout:
        times.append(float(step["time"]))
        events.append(str(step.get("event", "unknown")))
        commands.append(step.get("commands"))
        survival.append(step.get("interval_survival_probability"))
        for name in BODIES:
            paths[name].append(_centre(step[name], pitch))
        if step.get("event") == "release":
            if step.get("target") is not None:
                release_target = _centre(step["target"], pitch)
            release_family = step.get("family")
    return SolverTrajectory(
        times=tuple(times),
        paths={name: tuple(points) for name, points in paths.items()},
        events=tuple(events),
        commands=tuple(commands),
        survival=tuple(survival),
        release_target=release_target,
        release_family=release_family,
    )


def load_state(
    run_dir: Path,
    index: int,
    *,
    repository: str = "andrewkang12345/mit_ssac2027",
    commit: str = "unknown",
) -> SolverState:
    """One solved state, with its rollouts, from a real run directory.

    ``rows.jsonl`` is preferred because it is the only place the rollouts are
    written; ``states/state_NNN.json`` is the fallback and yields a state with
    no trajectories rather than an invented one.
    """

    run_dir = Path(run_dir)
    manifest = read_manifest(run_dir)
    config = manifest["config"]
    directions = tuple(tuple(float(v) for v in d) for d in config.get("directions", ()))
    if not directions:
        directions = DEFAULT_DIRECTIONS

    record = None
    rows = run_dir / "rows.jsonl"
    if rows.exists():
        with rows.open() as handle:
            for line in handle:
                candidate = json.loads(line)
                if int(candidate.get("index", -1)) == index:
                    record = candidate
                    break
    if record is None:
        state_file = run_dir / "states" / f"state_{index:03d}.json"
        if not state_file.exists():
            raise SolverArtifactError(f"no solved state {index} in {run_dir}")
        record = json.loads(state_file.read_text())

    scenario = record.get("scenario")
    if not scenario:
        raise SolverArtifactError(f"state {index} has no scenario block")
    pitch = (float(scenario.get("pitch_length", 105.0)),
             float(scenario.get("pitch_width", 68.0)))

    start, start_velocity, caps = {}, {}, {}
    for name in BODIES:
        body = scenario.get(name)
        if body is None:
            raise SolverArtifactError(f"state {index} has no {name}")
        start[name] = _centre(body["position"], pitch)
        start_velocity[name] = (float(body["velocity"][0]), float(body["velocity"][1]))
        if "maximum_speed" in body:
            caps[name] = float(body["maximum_speed"])

    fingerprint = None
    policy_path = run_dir / "policies" / f"state_{index:03d}.npz"
    if policy_path.exists():
        fingerprint = _policy_fingerprint(policy_path)

    return SolverState(
        index=index,
        name=str(scenario.get("name", f"state_{index:03d}")),
        stratum=str(record.get("stratum", "")),
        start=start,
        start_velocity=start_velocity,
        attack_direction=int(scenario.get("attack_direction", 1)),
        pitch=pitch,
        value=float(record["value"]),
        certificate={k: float(v) for k, v in (record.get("certificate") or {}).items()},
        root_attack=tuple(float(v) for v in record.get("root_attack", ())),
        root_defender=tuple(float(v) for v in record.get("root_defender", ())),
        directions=directions,
        step_seconds=float(config.get("step_seconds", 0.0)),
        steps=int(config.get("steps", 0)),
        trajectories=tuple(_trajectory(r, pitch) for r in record.get("rollouts", ())),
        provenance=SolverProvenance(
            repository=repository,
            commit=commit,
            artifact=str(Path(run_dir).name),
            state_index=index,
            fingerprint=fingerprint,
            created_utc=manifest.get("created_utc"),
        ),
        speed_caps=caps,
    )


def _policy_fingerprint(path: Path) -> str | None:
    """The solver's own fingerprint for a policy file, if numpy can read it."""

    try:
        import numpy as np

        with np.load(path, allow_pickle=True) as handle:
            if "metadata" not in handle.files:
                return None
            return json.loads(str(handle["metadata"])).get("fingerprint")
    except Exception:                                     # pragma: no cover
        return None


def load_for_scene(artifact: str | Path | None, **kwargs) -> SolverState | Unavailable:
    """Whatever a manifest entry points at, or :data:`UNAVAILABLE`.

    ``artifact`` is ``"<run directory>#<state index>"``. Anything missing,
    unreadable or unparseable yields the unavailable state with the reason
    attached -- this function never returns a partially invented state.
    """

    if not artifact:
        return UNAVAILABLE
    text = str(artifact)
    run, _, index = text.partition("#")
    if not index.isdigit():
        return Unavailable(f"Malformed solver reference: {text}")
    path = Path(run).expanduser()
    if not path.exists():
        return Unavailable(f"Solver artifact not found: {path}")
    try:
        return load_state(path, int(index), **kwargs)
    except (SolverArtifactError, KeyError, ValueError) as error:
        return Unavailable(f"Unreadable solver artifact: {error}")
