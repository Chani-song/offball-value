"""R2: counterfactual assignment by ARRIVAL-TIME RESPONSIBILITY.

The reviewer's criterion is "had the runner not existed, whom would this
defender have engaged?". R1 approximates that with proximity plus two
hand-tuned exceptions (a 15 m carrier-override radius, a 3 m covered-
exclusion radius, a one-skip cap). His own explanations say the exceptions
are not really about distance:

- G1/G2: the carrier is picked when he is carrying INTO the defender's
  zone — a trajectory condition, not a 15 m cap.
- G2: the nearest man is skipped when his defensive RESPONSIBILITY lies
  with other defenders, not merely when someone stands within 3 m.
- 37471: that skip must repeat until a genuinely free attacker is found.

R2 replaces the whole ladder with one quantity: **who can this defender
reach before anybody else, and soonest**. Reachability uses the package's
existing arrival model (acceleration, top speed, reaction time, turn
penalty), so an attacker's velocity and the defender's own momentum are
already in it.

    t_ours(a)   = this defender's time to a's onset position
    t_others(a) = the best other defender's time to the same point
    responsible(a) iff t_ours(a) <= t_others(a)
    charge = argmin over responsible attackers of t_ours(a)
             (fallback: argmin of t_ours - t_others when nobody qualifies)

Why the R1 exceptions fall out with NO fitted constants:
- a carrier driving at the defender is closing the gap, so the arrival
  model gives him a short time — no 15 m radius;
- an attacker another defender reaches sooner fails `responsible`, so he
  is skipped — no 3 m radius, and the skip repeats automatically because
  every attacker is scored rather than walked in order;
- turn penalty and momentum make "the space the defender is leaving"
  expensive to re-cover, which is the vacated-zone intuition.

A first attempt used the pointwise Fernández influence difference instead
and scored 3/24 on the round-1 training picks: influence at a teammate's
own position is ~0 whenever the defender is more than a few metres away,
so the margin carried no signal and unreachable isolated attackers won.
That failure is recorded here so it is not retried.

STATUS: developed AFTER the round-2 blind test was scored. Round 1 is its
development set; round 2 has already been spent on the pre-registered R1
race, so any round-2 number for R2 is a single declared post-hoc look and
never an adoption argument. R2 is NOT wired into the pipeline.
"""

from __future__ import annotations

import math
from typing import Mapping

from .bundesliga import BundesligaObjectState
from .pass_dynamics import ArrivalModelConfig, VelocityEstimate, player_time_to_point

ARRIVAL_CONFIG = ArrivalModelConfig()


def _as_player(
    state_row: Mapping[str, object], player_id: str
) -> tuple[BundesligaObjectState, VelocityEstimate]:
    """Adapt an assignment-rule state row to the arrival model's types."""
    player = BundesligaObjectState(
        object_id=player_id,
        team_id=str(state_row["team"]),
        x=float(state_row["x"]),
        y=float(state_row["y"]),
    )
    vx = float(state_row["vx"])
    vy = float(state_row["vy"])
    velocity = VelocityEstimate(
        vx=vx,
        vy=vy,
        speed=math.hypot(vx, vy),
        sample_count=2,
        window_seconds=2 * 0.2,
    )
    return player, velocity


def arrival_table(
    state: Mapping[str, Mapping[str, object]],
    attacking_team: str,
    runner_id: str,
    defender_id: str,
    config: ArrivalModelConfig = ARRIVAL_CONFIG,
) -> list[dict[str, object]]:
    """Per non-runner attacker: our arrival time, the best rival's, the gap."""
    our_player, our_velocity = _as_player(state[defender_id], defender_id)
    rivals = [
        pid
        for pid, row in state.items()
        if str(row["team"]) != attacking_team and pid != defender_id
    ]
    rows: list[dict[str, object]] = []
    for pid, row in state.items():
        if str(row["team"]) != attacking_team or pid == runner_id:
            continue
        point = (float(row["x"]), float(row["y"]))
        ours = float(player_time_to_point(our_player, our_velocity, point, config))
        best_other = float("inf")
        for rival_id in rivals:
            rival, rival_velocity = _as_player(state[rival_id], rival_id)
            best_other = min(
                best_other,
                float(player_time_to_point(rival, rival_velocity, point, config)),
            )
        rows.append(
            {
                "option_id": pid,
                "ours": ours,
                "best_other": best_other,
                "gap": ours - best_other,
                "responsible": ours <= best_other,
            }
        )
    # Deterministic: soonest first, ties by option id.
    rows.sort(key=lambda row: (round(float(row["ours"]), 9), str(row["option_id"])))
    return rows


def rule_r2(
    state: Mapping[str, Mapping[str, object]],
    attacking_team: str,
    runner_id: str,
    carrier_id: str,
    defender_id: str,
    config: ArrivalModelConfig = ARRIVAL_CONFIG,
) -> tuple[str, str]:
    """The defender's counterfactual charge, plus the branch that chose it.

    ``carrier_id`` is accepted for signature parity with R0/R1 and is
    deliberately unused: the carrier competes on the same footing as every
    other attacker, which is the point of the rule.
    """
    del carrier_id
    rows = arrival_table(state, attacking_team, runner_id, defender_id, config)
    if not rows:
        raise ValueError("no non-runner attackers to assign")
    responsible = [row for row in rows if row["responsible"]]
    if responsible:
        return str(responsible[0]["option_id"]), "responsible"
    # Nobody is ours outright — take the attacker we are least far behind on.
    fallback = min(
        rows, key=lambda row: (round(float(row["gap"]), 9), str(row["option_id"]))
    )
    return str(fallback["option_id"]), "least_behind"
