"""Role selection state, kept JSON-serialisable so Dash can store it."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

ROLES = ("runner", "defender", "beneficiary")

#: role name -> the attribute holding its ids (beneficiary does not pluralise)
_FIELD = {"runner": "runners", "defender": "defenders", "beneficiary": "beneficiaries"}


@dataclass
class Selection:
    runners: list[str] = field(default_factory=list)
    defenders: list[str] = field(default_factory=list)
    beneficiaries: list[str] = field(default_factory=list)
    pick: str = "runner"                 # which role the next pitch click fills
    source: str = "manual"               # manual | annotation | pipeline | suggested

    # -- access ---------------------------------------------------------
    def ids(self, role: str) -> list[str]:
        return getattr(self, _FIELD[role])

    def role_of(self, player_id: str) -> str | None:
        for role in ROLES:
            if player_id in self.ids(role):
                return role
        return None

    @property
    def runner(self) -> str | None:
        return self.runners[0] if self.runners else None

    @property
    def is_empty(self) -> bool:
        return not (self.runners or self.defenders or self.beneficiaries)

    # -- mutation -------------------------------------------------------
    def toggle(self, role: str, player_id: str, single: bool = False) -> None:
        bucket = self.ids(role)
        if player_id in bucket:
            bucket.remove(player_id)
            return
        for other in ROLES:                       # a player holds one role at a time
            if other != role and player_id in self.ids(other):
                self.ids(other).remove(player_id)
        if single:
            bucket.clear()
        bucket.append(player_id)

    def clear(self, role: str | None = None) -> None:
        for name in (ROLES if role is None else (role,)):
            self.ids(name).clear()

    def advance(self) -> None:
        """Move the pick target on, so runner -> defender -> beneficiary flows."""

        if self.pick == "runner" and self.runners:
            self.pick = "defender"
        elif self.pick == "defender" and self.defenders:
            self.pick = "beneficiary"

    # -- serialisation --------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "Selection":
        if not data:
            return cls()
        return cls(
            runners=list(data.get("runners", [])),
            defenders=list(data.get("defenders", [])),
            beneficiaries=list(data.get("beneficiaries", [])),
            pick=data.get("pick", "runner"),
            source=data.get("source", "manual"),
        )

    # -- clicking --------------------------------------------------------
    def apply_click(self, scene, player_id: str) -> "Selection":
        """Assign a clicked player to a role and move the pick target on.

        Three rules, in order:

        1. a player who already holds a role loses it, so a second click is
           always "undo" no matter which pick target is armed;
        2. the clicked side wins over the pick target, so clicking a defender
           while 'Runner' is armed sets the defender;
        3. an attacker fills the runner slot if it is empty, otherwise the
           beneficiary slot.
        """

        player = scene.players.get(player_id)
        if player is None:
            return self
        existing = self.role_of(player_id)
        if existing is not None:
            role = existing                       # a second click drops the role
        elif player.side == "defend":
            role = "defender"                     # the clicked side always wins
        elif self.pick == "runner" or not self.runners:
            role = "runner"
        else:
            role = "beneficiary"                  # runner is taken, so this is a gainer
        self.toggle(role, player_id, single=(role == "runner"))
        self.source = "manual"
        self.pick = role
        self.advance()
        return self

    @classmethod
    def from_scene(cls, scene, source: str = "annotation") -> "Selection":
        return cls(
            runners=list(scene.runner_ids),
            defenders=list(scene.defender_ids),
            beneficiaries=list(scene.beneficiary_ids),
            pick="runner",
            source=source,
        )
