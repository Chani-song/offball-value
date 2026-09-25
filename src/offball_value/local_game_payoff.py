"""Provisional payoff audit for the confirmed local off-ball scenes.

This module is intentionally a *development-set diagnostic*, not the final
counterfactual estimator.  The focal runner and all non-focal players follow
their observed futures.  One candidate defender is replaced by either the
legacy goal-side target baseline or a target-agnostic physically reachable
response, and every displayed attacking continuation is re-evaluated under
the same response.

The provisional option payoff is

``delivery-or-retention x geometric goal danger x goal-side accessibility``.

All three components are exposed so that a football review can identify
whether an implausible result comes from local-game construction, response
generation, delivery, terminal danger, or influence coverage.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from functools import lru_cache
import json
import math
from pathlib import Path
from collections.abc import Mapping, Sequence

import numpy as np

from .bundesliga import (
    FPS,
    BundesligaFrame,
    BundesligaObjectState,
)
from .dynamic_marking import interpolate_timed_point
from .causal_defender_policy import (
    CausalPolicyConfig,
    rollout_causal_defender_path,
)
from .defender_trajectory_search import (
    DefenderTrajectorySearchConfig,
    generate_feasible_defender_trajectories,
)
from .goal_weighted_influence import (
    GoalWeightedInfluenceConfig,
    influence_pitch_grid,
)
from .local_game_structure import (
    LocalGameStructureConfig,
    build_structural_local_game,
    estimate_onset_velocity,
    moving_goal_side_target,
    simulate_goal_side_response_trace,
)
from .carry_dynamics import carry_point_estimate
from .r9_selection import select_r9_beneficiary

# Adopted for structure, not for a claimed accuracy win: round 1 R6 23 /
# R9 24, round 2 R6 18 / R9 19 with McNemar p = 1.0. The docstring of
# coupled_beneficiary.py states the claim is explicitly not made.
R9_LABEL = "coupled_r9 (round1 24/24, round2 19/21; adopted for structure)"
from .assignment_rule import (
    PROVISIONAL_LABEL as ASSIGNMENT_RULE_LABEL,
    attacking_team_id as _assignment_attacking_team,
    onset_state as _assignment_onset_state,
    rule_r1 as _assignment_rule_r1,
)
from .obso import is_offside_position, score_at_points
from .pass_dynamics import (
    ArrivalModelConfig,
    VelocityEstimate,
    player_time_to_point,
    point_reception_estimate,
)
from .post_reception_value import goal_side_accessibility
from .ssac_stack import (
    carry_survival as ssac_carry_survival,
    completion_probability as ssac_completion_probability,
    defending_positions as ssac_defending_positions,
    threat_at_point as ssac_threat_at_point,
)
from .delivery_calibration import apply_calibration
from .kinematic_xpass import TrackedPlayer, load_kinematic_model
from .xpass import (
    load_xpass_model,
    predict_pass_success,
    predict_pass_success_360,
    predict_pass_success_360_kinematic,
)


TimedPoint = tuple[float, float, float]


@dataclass(frozen=True)
class _SceneRuntime:
    onset_players: Mapping[str, Sequence[object]]
    player_paths: Mapping[str, tuple[TimedPoint, ...]]
    ball_path: tuple[TimedPoint, ...]


@dataclass(frozen=True)
class LocalGamePayoffConfig:
    """Pre-registered settings for the first eight-scene payoff audit."""

    release_step_seconds: float = 0.4
    minimum_event_duration_seconds: float = 0.2
    influence_grid_resolution_m: float = 3.0
    compromise_runner_weights: tuple[float, ...] = (0.25, 0.50, 0.75)
    local_non_runner_limit: int = 3
    # Raised from 0.002 after the 36179 gate-boundary case: a derived branch
    # worth ~0.002 passed on relative effect alone and produced a spurious
    # dilemma label.  Confirmed dilemma branches in the development set all
    # have peak Q >= 0.06, so 0.01 separates the regimes with margin.
    minimum_peak_q: float = 0.01
    minimum_relative_response_effect: float = 0.05
    carry_pressure_action_time_seconds: float = 0.35
    carry_pressure_sigma_seconds: float = 0.25
    carry_path_step_seconds: float = 0.2
    # Adopted as the default after an eight-scene comparison showed identical
    # local-game labels with a more principled multi-defender pressure price.
    carry_pressure_model: str = "independent_race"
    # Continuation-design C (v0.1): through-ball templates into the space
    # ahead of a moving receiver.  Without them the only receiver action is
    # "meet the ball on the observed path", so optimal defense degenerates
    # into intercepting observed coordinates instead of denying space.
    # Empty tuple disables the templates.
    through_ball_lead_distances_m: tuple[float, ...] = (2.0, 4.0, 6.0)
    through_ball_goal_side_offset_m: float = 1.0
    through_ball_minimum_run_speed_mps: float = 0.5
    # Design-C deferred template, now implemented behind a flag (default off
    # so the v0.3 canonical catalogue is unchanged): receive coming toward
    # the ball, lead metres short of the receiver's release position.
    cutback_lead_distances_m: tuple[float, ...] = ()
    # Carrier-leverage channel (v0.3.2): the carry option also prices
    # carrying INTO the space ahead of the observed carry direction, so a
    # candidate defender standing in (or leaving) that space moves the
    # carry value — the mechanism behind the two remaining human-label
    # mismatches (53833-B, 131237).  Empty tuple disables.
    carry_lead_distances_m: tuple[float, ...] = (2.0, 4.0, 6.0)
    # Adaptive confrontation leads (carry-suppressibility diagnosis): the
    # fixed grid stops short of the defender actually blocking the carry ray
    # (12.5 m away in 53833-B), so the vacate-versus-stay value never enters
    # the candidate set.  Per anchor, the distance along the observed carry
    # ray to the first blocking defender (observed frame, so the columns
    # stay response-independent) is added as leads d*-1 and d*+1, within
    # this maximum and the corridor half-width below.
    carry_confrontation_max_lead_m: float = 15.0
    carry_confrontation_corridor_m: float = 2.5
    # Carry-lead pricing (v0.3.5-v0.3.7 experiment chain):
    # - "contest_only" (v0.3.5): retention x G x endpoint tackle race.  The
    #   race saturates for standing defenders a few metres away, so a
    #   defender parked in the carry lane never suppressed the row.
    # - "goal_side_accessibility" (v0.3.6): additionally multiplies the
    #   global goal-side accessibility at the lead point.  Response
    #   sensitive, but the whole back line drags A down with depth and
    #   cancels the G gradient — advancing the ball into vacated space then
    #   never beats standing still, which is football-illogical.
    # - "extension_race" (v0.3.7, NEGATIVE RESULT): the forward extension
    #   priced exactly like the observed carry path — every defender races
    #   the carrier to sampled points along the extension, weakest sample
    #   wins, endpoint included.  Semantically right for a defender standing
    #   mid-corridor, but empirically covering responses hover goal-side of
    #   the lane rather than in the narrow corridor, so replacing the global
    #   accessibility term with the race LOST the 131237 auto-pick
    #   (follow-gain pins-off 4/7 -> 3/7).  Kept as an optional mode;
    #   revisit as a multiplicative complement after the G-gradient work.
    carry_lead_pricing: str = "goal_side_accessibility"
    carry_extension_sample_step_m: float = 2.0
    # Goal-danger surface G (v0.3.8): path to an obso EPV grid CSV.  None
    # keeps the geometric proxy.  Two defects motivate the learned surface
    # (scripts/explore_progression_goal_danger.py, 9.75M StatsBomb events):
    # - LATERAL (decisive, label-independent): the proxy prices the corner
    #   flag 0.579 ABOVE the top of the D 0.529, and the touchline 8 m from
    #   goal 0.548 above it too.  The learned surface orders them correctly
    #   (0.128 < 0.165), centre-vs-wing contrast x2.07 at 25 m vs x1.16.
    # - DEPTH: the 0.45x(linear progress) term flattens the approach zone.
    #   25->17 m central is x1.165 in the proxy against x1.713 in the grid
    #   AS CONSUMED.  (An earlier x1.97 figure was a coarse-bin artefact;
    #   after removing the goal-shot label tautology and stratifying on
    #   progression the defensible causal band is x1.3-1.7.)
    # The surface is an OUTCOME frequency, not a defender-occupancy model
    # like Fernandez & Bornn's pitch value, so it does not carry an
    # explicit defensive-shape term into G -- but it is still measured
    # under the average real defence at z, so it is not defence-free.
    goal_danger_grid_path: str | None = None
    # Derived-beneficiary source (2026-09-06 decision). The round-2 blind
    # test (docs/blind_session_round2_final_results.md) put the R1
    # counterfactual-assignment rule at the top of the race (14.5/22,
    # chance E=2.5) with the shipped pairwise_cross_cost selection LAST
    # (5.5/22, significantly below the top three) — but R1's 0.5 lead over
    # the best Q-variant failed the pre-registered adoption bar, so this
    # default is PROVISIONAL: chosen on parsimony, cross-population
    # stability, interpretability and cost, not on a claimed statistical
    # win. "payoff" restores the previous cross-cost-selected derived
    # branch. Human pins in the manifest always take precedence.
    # "coupled_r9" is the rule the labels actually favour: 24/24 on round 1
    # and 19/21 on round 2, against R1's 20 and 14.5/22. It was never the
    # default because R1 needs only positions while R9 needs Q, which does
    # not exist until the responses are priced -- an ordering problem, not a
    # disagreement, and the build prices everything before it forms the pair.
    derived_option_source: str = "assignment_rule_v1"
    # De-confounding diagnostic (v0.3.9): the response search is
    # value-guided, so changing any pricing term also changes WHICH
    # defender trajectories get proposed and screened (53833-B: 33 vs 44
    # responses, only 24 shared).  Every cross-version comparison then
    # confounds "the value model changed" with "the search went
    # somewhere else".  Point this at an existing audit JSON to replay its
    # exact search-candidate trajectories instead of generating and
    # screening new ones, so two pricing models can be compared on
    # identical inputs.  Reference responses (actual / target baseline)
    # are structural and are still rebuilt locally.
    frozen_response_catalogue_path: str | None = None
    # Terminal-structure channel (v0.3.2): every option also carries a
    # discounted "virtual post-horizon window" priced from the terminal
    # states — deliverability proxy × geometric danger × the product of
    # per-defender contest terms.  This is what prices a box two-versus-one
    # that only materializes after the horizon, gives path tails and arrival
    # control a value, and is zeroed by terminal offside.  0 disables.
    # 0.4 from a coarse sweep against the seven human derived labels: the
    # selection rule is stable for 0.1-0.4 and collapses at 0.5 (terminal
    # floors drowning the event windows); treat as pre-registered pending
    # validation on newly labelled scenes.
    terminal_structure_discount: float = 0.4
    # Record every (release × continuation) candidate per cell so the payoff
    # matrix can be exported with unfolded attack columns (joint analysis
    # with the game-theoretic solver).
    record_candidate_grid: bool = False
    # v0.3 main line (2026-08-28 lab decision): also price the causal
    # observation-limited policy as a non-optimizer reference, so every audit
    # reports the pair (causal value, oracle bound) whose gap is the value of
    # anticipation.
    include_causal_policy_reference: bool = False
    # 2026-08-28 lab decision: the delivery term can be a learned xPass model
    # (StatsBomb open data) instead of the mechanistic flight/interception
    # machinery.  Timing/event selection stays mechanistic in both modes; the
    # switch only changes the priced P.  Selection-bias caveat documented in
    # docs/causal_defender_policy_v0_3_design.md.
    delivery_model: str = "mechanistic"
    # Stage 3 prices possession with its own threat and completion model.
    # "ssac" swaps BOTH plus the carry hazard, so stage 2 ranks on the same
    # scale stage 3 solves on. Swapping one piece is what broke commensurability
    # before: pass options moved 1.9x and carry options did not move at all.
    # Ablations, because "adopt the whole stack" bundles three substitutions
    # and one judgement call that is NOT in the adopted code: dropping the
    # accessibility factor. If that one line is what reorders the shortlist,
    # that is this pipeline's problem, not the adopted stack's.
    #   native            nothing changed
    #   ssac              all three, accessibility folded away
    #   ssac_threat       adopted threat only
    #   ssac_pass         adopted completion model only
    #   native_no_access  accessibility dropped, nothing else
    #   ssac_ourcarry     adopted threat and completion, OUR carry model
    #   ssac_keepaccess   all three adopted, accessibility KEPT
    #
    # The last one separates two changes that "ssac" makes at once. Dropping
    # accessibility was justified as redundant with the adopted threat's own
    # defender term, but that term is one defender's distance
    # (1 - exp(-d/9)), while goal_side_accessibility reads the whole
    # defensive shape off an influence grid including the keeper. One player
    # against eleven is not redundancy, so the drop needs its own arm.
    #
    # The last one exists because the adopted tackle hazard is three hand-set
    # constants, while carry_dynamics.py prices a carry the way the pass side
    # is priced -- same logistic, same race sigma -- on a carry speed measured
    # from tracking (0.944 over 5,059 paired top speeds). It has never been in
    # the build path; it has only ever run in a prototype script.
    value_stack: str = "native"
    ssac_pass_model_path: str = "andrew/models/experimental_pass.json"
    # The adopted tackle hazard's three constants are hand-set defaults in the
    # other package, not fitted. Exposed here so a sweep can tell whether
    # adopting the stack also means adopting somebody's tuning.
    ssac_tackle_rate: float = 2.2
    ssac_tackle_radius_m: float = 1.0
    ssac_tackle_softness_m: float = 0.45
    xpass_model_path: str = "data/processed/xpass_v0/xpass_hist_gbdt.joblib"
    # Defender-aware xPass trained on StatsBomb 360 freeze frames: the same
    # geometry features plus lane pressure, blockers in the corridor and
    # cover at the reception point (test AUC 0.935 against the geometry-only
    # model's 0.863, and calibrated across lane occupancy - 51% predicted and
    # observed with an opponent inside 1 m of the lane, 96% beyond 8 m).
    # Because defenders are IN the model, this mode uses it alone rather than
    # multiplying by the mechanistic interaction terms, which removes the
    # double-count the hybrid mode accepted as a v0 bias and which measurement
    # showed leaves P moving only 3% across the defender's own responses.
    xpass_360_model_path: str = (
        "data/processed/xpass_360/xpass_360_hist_gbdt.joblib"
    )
    # Optional monotone map from the raw pass delivery onto an observed
    # completion rate, fitted on real Bundesliga passes by
    # scripts/calibrate_hybrid_delivery.py. None disables it, which is the
    # default: the fit corrects the LEVEL (median 0.32 -> 0.77 against a true
    # 0.81) but measurably flattens the response to a defender placed on the
    # lane (-0.208 raw -> -0.077 calibrated), and an argmax over P x G x A
    # consumes that spacing. Applies to pass cells only -- carry retention and
    # the terminal floor are different quantities and are left alone.
    delivery_calibration_path: str | None = None
    # xpass360 plus ten arrival-race features differenced from a 0.97 s frame
    # pair (scripts/train_xpass_360_kinematic.py). Worth +0.0039 AUC over the
    # static set on 204,857 StatsBomb passes, fold sd 0.0015 -- small there
    # because observed passes rarely contain a lost race, which is exactly the
    # region the option catalogue lives in. At inference the pipeline supplies
    # 25 fps velocities, cleaner than the frame pairs the fit saw.
    # Ablations for the Q = P x G x A shape itself. With 21 labels the
    # approximate version (dividing the recorded q by the cell's own factor)
    # left the answers almost untouched -- G removable with no change at all --
    # so these rebuild it exactly instead of dividing after the fact. Both off
    # by default; they exist to test the product, not to ship without a factor.
    disable_goal_danger: bool = False
    disable_accessibility: bool = False
    xpass_360_kinematic_model_path: str = (
        "data/processed/xpass_360_kinematic/xpass_360_kinematic_gbdt.joblib"
    )
    # Among responses whose anchor score ties within this relative tolerance,
    # the displayed representative is chosen by terminal goal-side positioning
    # instead of raw index/effort.  Human review found that value-indifferent
    # path tails otherwise produce football-implausible display anchors.
    # Widened 0.5% -> 2% after the 61841 review: run-leading, goal-side
    # near-ties sat at +0.6-0.9% and could not be displayed.
    anchor_tie_tolerance: float = 0.02
    response_search_mode: str = "target_conditioned"
    target_agnostic_raw_response_limit: int = 160
    response_screen_keep_per_option: int = 4
    response_screen_keep_minimax: int = 8
    response_screen_keep_low_effort: int = 4
    # Extra responses drawn uniformly at random from the ones the screen
    # rejected. The screen selects on a cheap approximation of Q whose
    # WITHIN-option rank correlation with the real thing is only 0.153, so
    # everything it keeps is correlated in the same wrong direction and the
    # stored minimax can be overstated by up to 40%. A random draw is not
    # smarter, only uncorrelated, and that is exactly what the tail needs:
    # 40 extra responses cut the worst case from 40.3% to 11.6% and the
    # mean from 5.0% to 1.8%. Zero reproduces every existing artifact.
    # See docs/screening_bias_v0_1.md.
    # Union the top-K attackers by vacated-space overlap into each defender's
    # option catalogue.  0 keeps the historical marking-release-only filter.
    vacated_union_count: int = 0
    response_screen_keep_random: int = 0
    response_screen_random_seed: int = 0
    local_game_selection_mode: str = "response_effect"

    def validate(self) -> None:
        positive = {
            "release_step_seconds": self.release_step_seconds,
            "minimum_event_duration_seconds": self.minimum_event_duration_seconds,
            "influence_grid_resolution_m": self.influence_grid_resolution_m,
            "carry_pressure_action_time_seconds": (
                self.carry_pressure_action_time_seconds
            ),
            "carry_pressure_sigma_seconds": self.carry_pressure_sigma_seconds,
            "carry_path_step_seconds": self.carry_path_step_seconds,
        }
        if any(value <= 0.0 for value in positive.values()):
            raise ValueError("payoff-audit time and grid settings must be positive")
        if self.local_non_runner_limit < 1:
            raise ValueError("local_non_runner_limit must be positive")
        if self.minimum_peak_q < 0.0:
            raise ValueError("minimum_peak_q cannot be negative")
        if not 0.0 <= self.minimum_relative_response_effect <= 1.0:
            raise ValueError("minimum relative effect must lie in [0, 1]")
        if any(not 0.0 < weight < 1.0 for weight in self.compromise_runner_weights):
            raise ValueError("compromise weights must lie in (0, 1)")
        if self.response_search_mode not in {
            "target_conditioned",
            "target_agnostic",
        }:
            raise ValueError("unknown response_search_mode")
        if self.carry_pressure_model not in {
            "nearest_defender",
            "independent_race",
        }:
            raise ValueError("unknown carry_pressure_model")
        if not 0.0 <= self.anchor_tie_tolerance < 0.5:
            raise ValueError("anchor_tie_tolerance must lie in [0, 0.5)")
        if any(value <= 0.0 for value in self.through_ball_lead_distances_m):
            raise ValueError("through-ball lead distances must be positive")
        if self.through_ball_goal_side_offset_m < 0.0:
            raise ValueError("through-ball goal-side offset cannot be negative")
        if self.through_ball_minimum_run_speed_mps <= 0.0:
            raise ValueError("through-ball minimum run speed must be positive")
        if any(value <= 0.0 for value in self.cutback_lead_distances_m):
            raise ValueError("cutback lead distances must be positive")
        if any(value <= 0.0 for value in self.carry_lead_distances_m):
            raise ValueError("carry lead distances must be positive")
        if self.carry_confrontation_max_lead_m < 0.0:
            raise ValueError("confrontation max lead cannot be negative")
        if self.carry_confrontation_corridor_m <= 0.0:
            raise ValueError("confrontation corridor must be positive")
        if self.derived_option_source not in {
            "assignment_rule_v1", "coupled_r9", "payoff"
        }:
            raise ValueError("unknown derived_option_source")
        if self.carry_lead_pricing not in {
            "contest_only",
            "goal_side_accessibility",
            "extension_race",
        }:
            raise ValueError("unknown carry_lead_pricing")
        if self.carry_extension_sample_step_m <= 0.0:
            raise ValueError("carry extension sample step must be positive")
        if self.delivery_model not in {
            "mechanistic", "xpass", "hybrid", "xpass360", "xpass360_kinematic"
        }:
            raise ValueError("unknown delivery_model")
        if min(self.ssac_tackle_rate, self.ssac_tackle_radius_m) < 0 or (
            self.ssac_tackle_softness_m <= 0
        ):
            raise ValueError("invalid ssac tackle parameters")
        if self.value_stack not in {
            "native", "ssac", "ssac_threat", "ssac_pass",
            "native_no_access", "ssac_ourcarry", "ssac_keepaccess"
        }:
            raise ValueError("unknown value_stack")
        if self.value_stack in {
            "ssac", "ssac_pass", "ssac_ourcarry", "ssac_keepaccess"
        } and not Path(
            self.ssac_pass_model_path
        ).exists():
            raise ValueError(
                f"ssac pass model not found: {self.ssac_pass_model_path}"
            )
        if self.delivery_calibration_path and not Path(
            self.delivery_calibration_path
        ).exists():
            raise ValueError(
                f"delivery calibration not found: {self.delivery_calibration_path}"
            )
        if not 0.0 <= self.terminal_structure_discount <= 1.0:
            raise ValueError("terminal_structure_discount must lie in [0, 1]")
        counts = (
            self.target_agnostic_raw_response_limit,
            self.response_screen_keep_per_option,
            self.response_screen_keep_minimax,
            self.response_screen_keep_low_effort,
        )
        if any(value < 1 for value in counts):
            raise ValueError("response-search counts must be positive")
        if self.local_game_selection_mode not in {
            "response_effect",
            "pairwise_cross_cost",
        }:
            raise ValueError("unknown local_game_selection_mode")


def _time_of(frame: Mapping[str, object]) -> float:
    if "relative_time_s" in frame:
        return float(frame["relative_time_s"])
    return float(frame["time_s"])


def _frame_lookup(frame: Mapping[str, object]) -> dict[str, Sequence[object]]:
    return {str(player[0]): player for player in frame["players"]}  # type: ignore[index]


def _onset_frame(frames: Sequence[Mapping[str, object]]) -> Mapping[str, object]:
    return min(frames, key=lambda frame: abs(_time_of(frame)))


def _observed_player_paths(
    frames: Sequence[Mapping[str, object]],
) -> dict[str, tuple[TimedPoint, ...]]:
    rows: dict[str, list[TimedPoint]] = {}
    for frame in frames:
        time_s = _time_of(frame)
        for player in frame["players"]:  # type: ignore[index]
            rows.setdefault(str(player[0]), []).append(
                (time_s, float(player[2]), float(player[3]))
            )
    return {
        player_id: tuple(sorted(samples))
        for player_id, samples in rows.items()
        if len(samples) >= 2
    }


def _observed_ball_path(
    frames: Sequence[Mapping[str, object]],
) -> tuple[TimedPoint, ...]:
    samples = [
        (_time_of(frame), float(frame["ball"][0]), float(frame["ball"][1]))  # type: ignore[index]
        for frame in frames
        if frame.get("ball") is not None
    ]
    if not samples:
        raise ValueError("payoff audit requires an observed ball path")
    return tuple(sorted(samples))


def _scene_runtime(game: Mapping[str, object]) -> _SceneRuntime:
    frames: Sequence[Mapping[str, object]] = game["background_frames"]  # type: ignore[assignment]
    return _SceneRuntime(
        onset_players=_frame_lookup(_onset_frame(frames)),
        player_paths=_observed_player_paths(frames),
        ball_path=_observed_ball_path(frames),
    )


def _path_velocity(
    path_txy: Sequence[TimedPoint],
    time_s: float,
    window_seconds: float = 0.2,
) -> tuple[float, float]:
    lower = max(float(path_txy[0][0]), float(time_s) - window_seconds / 2.0)
    upper = min(float(path_txy[-1][0]), float(time_s) + window_seconds / 2.0)
    if upper - lower <= 1e-9:
        lower = max(float(path_txy[0][0]), float(time_s) - window_seconds)
        upper = min(float(path_txy[-1][0]), float(time_s) + window_seconds)
    if upper - lower <= 1e-9:
        return 0.0, 0.0
    before = interpolate_timed_point(path_txy, lower)
    after = interpolate_timed_point(path_txy, upper)
    return (
        float((after[0] - before[0]) / (upper - lower)),
        float((after[1] - before[1]) / (upper - lower)),
    )


def _frame_state(
    game: Mapping[str, object],
    time_s: float,
    defender_id: str | None = None,
    defender_path_txy: Sequence[TimedPoint] | None = None,
    attach_ball_to_carrier: bool = False,
    runtime: _SceneRuntime | None = None,
) -> tuple[BundesligaFrame, dict[str, VelocityEstimate]]:
    """Interpolate the retrospective 22-player background at one time."""

    state = runtime or _scene_runtime(game)
    onset_players = state.onset_players
    paths = state.player_paths
    ball_path = state.ball_path
    players: dict[str, BundesligaObjectState] = {}
    velocities: dict[str, VelocityEstimate] = {}
    for player_id, source in onset_players.items():
        path = paths[player_id]
        x, y = interpolate_timed_point(path, time_s)
        vx, vy = _path_velocity(path, time_s)
        if player_id == defender_id and defender_path_txy is not None:
            x, y = interpolate_timed_point(defender_path_txy, time_s)
            vx, vy = _path_velocity(defender_path_txy, time_s)
        speed = math.hypot(vx, vy)
        players[player_id] = BundesligaObjectState(
            object_id=player_id,
            team_id=str(source[1]),
            x=float(x),
            y=float(y),
            speed=float(speed * 3.6),
        )
        velocities[player_id] = VelocityEstimate(
            vx=float(vx),
            vy=float(vy),
            speed=float(speed),
            sample_count=2,
            window_seconds=0.2,
        )

    if attach_ball_to_carrier:
        carrier = players[str(game["carrier_id"])]
        ball_xy = (float(carrier.x), float(carrier.y))
    else:
        ball_xy = interpolate_timed_point(ball_path, time_s)
    frame = BundesligaFrame(
        match_id=str(game["match_id"]),
        frame_id=int(round(int(game["onset_frame_id"]) + float(time_s) * FPS)),
        period=1,
        game_section="retrospective_payoff_audit",
        timestamp=None,
        players=players,
        ball=BundesligaObjectState(
            object_id="BALL",
            team_id="BALL",
            x=float(ball_xy[0]),
            y=float(ball_xy[1]),
        ),
    )
    return frame, velocities


def _sample_times(
    start: float,
    stop: float,
    step: float,
    *,
    include_stop: bool = True,
) -> tuple[float, ...]:
    if stop < start - 1e-9:
        return ()
    values: list[float] = []
    value = float(start)
    while value <= stop + 1e-9:
        values.append(round(value, 10))
        value += step
    if include_stop and (not values or stop - values[-1] > 1e-8):
        values.append(float(stop))
    return tuple(dict.fromkeys(values))


def _path_effort(path_txy: Sequence[TimedPoint]) -> float:
    if len(path_txy) < 3:
        return 0.0
    velocities = []
    for before, after in zip(path_txy, path_txy[1:]):
        dt = max(float(after[0] - before[0]), 1e-9)
        velocities.append(
            (
                0.5 * (before[0] + after[0]),
                (after[1] - before[1]) / dt,
                (after[2] - before[2]) / dt,
            )
        )
    effort = 0.0
    for before, after in zip(velocities, velocities[1:]):
        dt = max(float(after[0] - before[0]), 1e-9)
        ax = (after[1] - before[1]) / dt
        ay = (after[2] - before[2]) / dt
        effort += (ax * ax + ay * ay) * dt
    return float(effort)


def _deduplicate_responses(
    responses: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    selected: dict[tuple[float, float], dict[str, object]] = {}
    for response in responses:
        path: Sequence[Sequence[float]] = response["path_txy"]  # type: ignore[assignment]
        key = (round(float(path[-1][1]), 2), round(float(path[-1][2]), 2))
        incumbent = selected.get(key)
        if incumbent is None or float(response["effort_m2ps3"]) < float(
            incumbent["effort_m2ps3"]
        ):
            selected[key] = response
    return list(selected.values())


@lru_cache(maxsize=4)
def _frozen_response_catalogue(
    path: str,
) -> Mapping[tuple[str, str], tuple[dict[str, object], ...]]:
    """Search-candidate trajectories keyed by (scene id, defender id)."""

    scenes = json.loads(Path(path).read_text(encoding="utf-8"))
    catalogue: dict[tuple[str, str], tuple[dict[str, object], ...]] = {}
    for scene in scenes:
        scene_id = str(scene["onset_frame_id"])
        for defender in scene["candidate_defenders"]:
            frozen: list[dict[str, object]] = []
            for response in defender["responses"]:
                if not response.get("is_search_candidate", False):
                    continue
                replay = {
                    "response_id": str(response["response_id"]),
                    "label": str(response["label"]),
                    "kind": str(response.get("kind", "feasible_response")),
                    "is_search_candidate": True,
                    "path_txy": [
                        list(map(float, row)) for row in response["path_txy"]
                    ],
                    "effort_m2ps3": float(response["effort_m2ps3"]),
                }
                # Physical diagnostics describe the trajectory, not the
                # pricing, so they replay unchanged — and the continuation
                # validator requires them on every search candidate.
                if "physical_diagnostics" in response:
                    replay["physical_diagnostics"] = dict(
                        response["physical_diagnostics"]  # type: ignore[arg-type]
                    )
                for extra in ("target_policy", "target_path_txy"):
                    if extra in response:
                        replay[extra] = response[extra]
                frozen.append(replay)
            catalogue[(scene_id, str(defender["defender_id"]))] = tuple(frozen)
    return catalogue


def _response_candidates(
    game: Mapping[str, object],
    defender: Mapping[str, object],
    payoff_config: LocalGamePayoffConfig,
    ensure_ids: Sequence[str] = (),
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Return non-optimizer references and feasible response search paths."""

    displayed_ids = _defender_displayed_option_ids(
        game, defender, ensure_ids, payoff_config.vacated_union_count
    )
    displayed = set(displayed_ids)
    cells = {
        str(cell["option_id"]): cell
        for cell in defender["cells"]  # type: ignore[index]
        if str(cell["option_id"]) in displayed
    }
    actual_path = [list(map(float, row)) for row in defender["actual_path_txy"]]  # type: ignore[index]
    actual = {
        "response_id": "actual",
        "label": "Actual defense",
        "kind": "actual_reference",
        "is_search_candidate": False,
        "path_txy": actual_path,
        "effort_m2ps3": _path_effort(tuple(tuple(row) for row in actual_path)),
    }
    responses: list[dict[str, object]] = []
    runner_path = tuple(tuple(map(float, row)) for row in game["runner_path_txy"])  # type: ignore[index]
    runner_response = [
        list(map(float, row)) for row in defender["runner_response_path_txy"]  # type: ignore[index]
    ]
    runner_target = [
        list(map(float, row))
        for row in defender.get("runner_response_target_path_txy", [])  # type: ignore[union-attr]
    ]
    runner_target_response = {
        "response_id": "focus_runner",
        "label": f"Target baseline · {game['runner_name']} goal-side",
        "kind": "target_conditioned_baseline",
        "is_search_candidate": payoff_config.response_search_mode
        == "target_conditioned",
        "path_txy": runner_response,
        "target_policy": "anticipated_runner_goal_side",
        "target_path_txy": runner_target,
        "effort_m2ps3": _path_effort(
            tuple(tuple(row) for row in runner_response)
        ),
    }
    if payoff_config.response_search_mode == "target_agnostic":
        structure_config = LocalGameStructureConfig(**game["config"])  # type: ignore[arg-type]
        frames: Sequence[Mapping[str, object]] = game["background_frames"]  # type: ignore[assignment]
        initial_velocity = estimate_onset_velocity(
            frames,
            str(defender["defender_id"]),
            structure_config.velocity_history_seconds,
        )
        start_xy = (
            float(defender["start_xy"][0]),  # type: ignore[index]
            float(defender["start_xy"][1]),  # type: ignore[index]
        )
        default_search = DefenderTrajectorySearchConfig()
        # Plan bounded steering over the full scene horizon.  A shorter
        # planning window freezes the terminal segment into a straight
        # coast, which structurally excludes responses that keep curving
        # with an option late in the scene (found in human review of the
        # Klaus audit: the best Ginczek-covering paths need control after
        # 1.8 s).  The generator still floors to the integration step and
        # neutrally coasts any sub-step remainder.
        planning_horizon = (
            math.floor(
                float(game["horizon_seconds"])
                / default_search.integration_step_seconds
                + 1e-9
            )
            * default_search.integration_step_seconds
        )
        search_config = DefenderTrajectorySearchConfig(
            planning_horizon_seconds=planning_horizon,
            response_delay_seconds=structure_config.response_delay_seconds,
            maximum_speed_mps=structure_config.maximum_speed_mps,
            maximum_acceleration_mps2=(
                structure_config.maximum_acceleration_mps2
            ),
            maximum_deceleration_mps2=(
                structure_config.maximum_deceleration_mps2
            ),
            maximum_normal_acceleration_mps2=(
                structure_config.maximum_normal_acceleration_mps2
            ),
            maximum_representatives=(
                payoff_config.target_agnostic_raw_response_limit
            ),
        )
        trajectories = generate_feasible_defender_trajectories(
            start_xy,
            initial_velocity,
            float(game["horizon_seconds"]),
            search_config,
        )
        search = [
            {
                "response_id": trajectory.trajectory_id,
                "label": (
                    f"Feasible response {index:03d} · "
                    f"end ({trajectory.endpoint_x:.1f}, {trajectory.endpoint_y:.1f})"
                ),
                "kind": "target_agnostic_feasible",
                "is_search_candidate": True,
                "path_txy": [list(row) for row in trajectory.path_txy],
                "target_policy": None,
                "target_path_txy": [],
                "effort_m2ps3": trajectory.effort_m2ps3,
                "physical_diagnostics": {
                    "terminal_speed_mps": trajectory.terminal_speed_mps,
                    "terminal_heading_radians": (
                        trajectory.terminal_heading_radians
                    ),
                    "maximum_path_speed_mps": (
                        trajectory.maximum_path_speed_mps
                    ),
                    "maximum_tangential_acceleration_mps2": (
                        trajectory.maximum_tangential_acceleration_mps2
                    ),
                    "maximum_tangential_deceleration_mps2": (
                        trajectory.maximum_tangential_deceleration_mps2
                    ),
                    "maximum_normal_acceleration_mps2": (
                        trajectory.maximum_normal_acceleration_mps2
                    ),
                    "source": trajectory.source,
                },
            }
            for index, trajectory in enumerate(trajectories, 1)
        ]
        references = [actual, runner_target_response]
        if payoff_config.include_causal_policy_reference:
            causal_path = rollout_causal_defender_path(
                game,
                str(defender["defender_id"]),
                CausalPolicyConfig(),
            )
            references.append(
                {
                    "response_id": "causal_policy",
                    "label": "인과 정책 · t≤T 관측만 사용",
                    "kind": "causal_policy_reference",
                    "is_search_candidate": False,
                    "path_txy": [list(row) for row in causal_path],
                    "target_policy": "causal_worst_case_replanning",
                    "target_path_txy": [],
                    "effort_m2ps3": _path_effort(causal_path),
                }
            )
        return references, search

    responses.append(runner_target_response)
    for option_id in displayed_ids:
        cell = cells[str(option_id)]
        path = [list(map(float, row)) for row in cell["option_response_path_txy"]]
        target_path = [
            list(map(float, row))
            for row in cell.get("option_response_target_path_txy", [])
        ]
        responses.append(
            {
                "response_id": f"focus:{option_id}",
                "label": f"옵션 억제 · {cell['option_name']}",
                "kind": "option_focus",
                "focus_option_id": str(option_id),
                "is_search_candidate": True,
                "path_txy": path,
                "target_policy": "anticipated_option_goal_side",
                "target_path_txy": target_path,
                "effort_m2ps3": _path_effort(tuple(tuple(row) for row in path)),
            }
        )

    compromise_options = sorted(
        cells.values(),
        key=lambda cell: (
            -float(cell["structural_tradeoff_score"]),
            -float(cell["option_allocation_effect_fraction"]),
            str(cell["option_id"]),
        ),
    )[:2]
    structure_config = LocalGameStructureConfig(**game["config"])  # type: ignore[arg-type]
    frames: Sequence[Mapping[str, object]] = game["background_frames"]  # type: ignore[assignment]
    initial_velocity = estimate_onset_velocity(
        frames,
        str(defender["defender_id"]),
        structure_config.velocity_history_seconds,
    )
    start_xy = (float(defender["start_xy"][0]), float(defender["start_xy"][1]))  # type: ignore[index]
    goal_xy = (float(game["goal_xy"][0]), float(game["goal_xy"][1]))  # type: ignore[index]
    horizon = float(game["horizon_seconds"])
    for primary in compromise_options:
        option_path = tuple(
            tuple(map(float, row)) for row in primary["option_path_txy"]
        )
        for runner_weight in payoff_config.compromise_runner_weights:
            blended = tuple(
                (
                    float(time_s),
                    float(
                        runner_weight * runner_x
                        + (1.0 - runner_weight)
                        * interpolate_timed_point(option_path, time_s)[0]
                    ),
                    float(
                        runner_weight * runner_y
                        + (1.0 - runner_weight)
                        * interpolate_timed_point(option_path, time_s)[1]
                    ),
                )
                for time_s, runner_x, runner_y in runner_path
            )
            path, target_path = simulate_goal_side_response_trace(
                start_xy,
                initial_velocity,
                blended,
                goal_xy,
                horizon,
                structure_config,
            )
            payload_path = [list(row) for row in path]
            responses.append(
                {
                    "response_id": (
                        f"compromise:{primary['option_id']}:"
                        f"r{int(round(100 * runner_weight))}"
                    ),
                    "label": (
                        f"중간 대응 · runner {int(round(100 * runner_weight))}% / "
                        f"{primary['option_name']} "
                        f"{int(round(100 * (1-runner_weight)))}%"
                    ),
                    "kind": "compromise",
                    "focus_option_id": str(primary["option_id"]),
                    "runner_weight": float(runner_weight),
                    "is_search_candidate": True,
                    "path_txy": payload_path,
                    "target_policy": "anticipated_blended_goal_side",
                    "target_path_txy": [list(row) for row in target_path],
                    "effort_m2ps3": _path_effort(path),
                }
            )
    return [actual], _deduplicate_responses(responses)


def _vacated_overlap_ranking(
    game: Mapping[str, object],
    defender: Mapping[str, object],
) -> list[tuple[str, float]]:
    """Rank options by how much of the space this defender gives up they cover.

    The catalogue's own criterion is marking release: "if he chases the
    runner, whose marking does he have to abandon".  That only sees one of
    the two ways an attacker benefits.  The other is spatial — he was never
    marked by this defender, he simply moves into the ground the defender
    left.  A player who benefits that way scores LAST on marking release
    (there was no marking to release), so the top-N deletes him before the
    beneficiary rule can ever see him.

    Both 40843/Irvine and 13749/Tanaka failed exactly this way: each was
    ranked 9 of 9 on marking release and 1st or 2nd on vacated overlap.  The
    earlier nearest-attacker union was aimed at the same gap but approximated
    it by distance, and missed both (Irvine sits 12.8 m out while a different
    attacker is nearest at 3.3 m).

    This scores the same quantity the beneficiary rule maximises, so the
    filter and the selector finally speak about the same thing.
    """
    from .assignment_rule import onset_state
    from .vacated_space import _region_grid, _state_v, _state_xy, coverage_field

    state = onset_state(game)
    defender_id = str(defender["defender_id"])
    if defender_id not in state:
        return []
    attacking_team_id = str(game["attacking_team_id"])
    chase = defender.get("runner_response_path_txy") or ()
    cells = defender.get("cells") or ()
    if not chase or not cells:
        return []

    xgrid, ygrid = _region_grid(state[defender_id])
    others = np.zeros((len(ygrid), len(xgrid)))
    for player_id, row in state.items():
        if str(row["team"]) == attacking_team_id or player_id == defender_id:
            continue
        np.maximum(
            others,
            coverage_field(_state_xy(row), _state_v(row), xgrid, ygrid),
            out=others,
        )
    before = np.maximum(
        others,
        coverage_field(
            _state_xy(state[defender_id]),
            _state_v(state[defender_id]),
            xgrid,
            ygrid,
        ),
    )

    paths = {
        str(cell["option_id"]): tuple(
            (float(t), float(x), float(y)) for t, x, y in cell["option_path_txy"]
        )
        for cell in cells
        if cell.get("option_path_txy")
    }
    totals = {option_id: 0.0 for option_id in paths}
    horizon = float(game["horizon_seconds"])
    step = 0.25
    time_s = step
    while time_s <= horizon + 1e-9:
        defender_xy = interpolate_timed_point(chase, time_s)
        defender_v = _path_velocity(chase, time_s)
        lost = np.clip(
            before
            - np.maximum(
                others, coverage_field(defender_xy, defender_v, xgrid, ygrid)
            ),
            0.0,
            None,
        )
        lost_total = float(lost.sum())
        if lost_total > 1e-9:
            for option_id, path in paths.items():
                option_xy = interpolate_timed_point(path, time_s)
                option_v = _path_velocity(path, time_s)
                totals[option_id] += float(
                    (lost * coverage_field(option_xy, option_v, xgrid, ygrid)).sum()
                )
        time_s = round(time_s + step, 6)
    return sorted(totals.items(), key=lambda item: (-item[1], item[0]))


def _defender_displayed_option_ids(
    game: Mapping[str, object],
    defender: Mapping[str, object],
    ensure_ids: Sequence[str] = (),
    vacated_union_count: int = 0,
) -> list[str]:
    """Rank the displayed option catalogue for this defender specifically.

    The affected local option set is defender-dependent by design, but the
    structural payload exposes one scene-level top list; with several
    candidate defenders that pooled ranking can crowd a defender's own
    plausible derived options out of the catalogue (found in the 131237
    three-defender audit, where the strongest derived option for one
    defender was displaced by options another defender controls).  The ball
    carrier is always retained, matching the structural convention.
    """

    carrier_id = str(game["carrier_id"])
    count = int(game["config"]["displayed_option_count"])  # type: ignore[index]
    ranked = sorted(
        defender["cells"],  # type: ignore[arg-type]
        key=lambda cell: (
            -float(cell["structural_tradeoff_score"]),
            -float(cell["option_allocation_effect_fraction"]),
            str(cell["option_id"]),
        ),
    )
    option_ids = [str(cell["option_id"]) for cell in ranked]
    if carrier_id in option_ids:
        option_ids = [carrier_id] + [
            option_id for option_id in option_ids if option_id != carrier_id
        ]
    displayed = option_ids[:count]

    # Union in the attacker this defender is currently closest to.  The
    # blind label round proved the threat-ranked top-N systematically
    # deletes the defender's OWN CURRENT MAN — dedicated cover of an
    # already-covered player shows ~0 release despite a large runner-side
    # cross-cost, so the most natural beneficiary ("the man the defender
    # must abandon") ranked 8-9 of 9 in all three off-list failures
    # (135654 Kownacki, 131237 Sobottka, 49266 Ljubicic), each of whom was
    # that defender's nearest attacker at onset.
    players = _frame_lookup(_onset_frame(game["background_frames"]))  # type: ignore[arg-type]
    defender_row = players.get(str(defender["defender_id"]))
    runner_id = str(game["runner_id"])
    if defender_row is not None:
        candidates = [
            option_id
            for option_id in option_ids
            if option_id != runner_id and option_id in players
        ]
        if candidates:
            nearest = min(
                candidates,
                key=lambda option_id: math.hypot(
                    float(players[option_id][2]) - float(defender_row[2]),
                    float(players[option_id][3]) - float(defender_row[3]),
                ),
            )
            if nearest not in displayed:
                displayed = [*displayed, nearest]
    # Union in the attackers who cover the ground this defender gives up.
    # See _vacated_overlap_ranking: marking release and spatial vacation are
    # two different ways to benefit, and ranking by the first alone deletes
    # everyone who benefits by the second.
    if vacated_union_count > 0:
        for option_id, overlap in _vacated_overlap_ranking(game, defender)[
            :vacated_union_count
        ]:
            if (
                overlap > 1e-9
                and option_id != runner_id
                and option_id in option_ids
                and option_id not in displayed
            ):
                displayed = [*displayed, option_id]
    # Callers may require specific options to survive display — e.g. the
    # provisional assignment-rule beneficiary, which must be priceable and
    # visible whenever it is the selected derived branch.
    for ensure_id in ensure_ids:
        if ensure_id and ensure_id in option_ids and ensure_id not in displayed:
            displayed = [*displayed, ensure_id]
    return displayed


def _goalkeeper_ids(game: Mapping[str, object]) -> tuple[str, ...]:
    onset = _onset_frame(game["background_frames"])  # type: ignore[arg-type]
    players = _frame_lookup(onset)
    team_id = str(game["attacking_team_id"])
    direction = int(game["attacking_direction"])
    defenders = [
        player_id for player_id, player in players.items() if str(player[1]) != team_id
    ]
    if not defenders:
        return ()
    goalkeeper = max(defenders, key=lambda player_id: direction * float(players[player_id][2]))
    return (goalkeeper,)


def _option_paths(
    game: Mapping[str, object],
    defender: Mapping[str, object],
    ensure_ids: Sequence[str] = (),
    vacated_union_count: int = 0,
) -> list[dict[str, object]]:
    cells = {str(cell["option_id"]): cell for cell in defender["cells"]}  # type: ignore[index]
    options = [
        {
            "option_id": str(game["runner_id"]),
            "option_name": str(game["runner_name"]),
            "option_type": "runner_receive",
            "path_txy": [list(map(float, row)) for row in game["runner_path_txy"]],  # type: ignore[index]
            "is_runner": True,
            "is_carrier": False,
        }
    ]
    for option_id in _defender_displayed_option_ids(
        game,
        defender,
        tuple(i for i in ensure_ids if i in cells),
        vacated_union_count,
    ):
        cell = cells[str(option_id)]
        is_carrier = str(option_id) == str(game["carrier_id"])
        options.append(
            {
                "option_id": str(option_id),
                "option_name": str(cell["option_name"]),
                "option_type": "carrier_carry" if is_carrier else "teammate_receive",
                "path_txy": [
                    list(map(float, row)) for row in cell["option_path_txy"]
                ],
                "is_runner": False,
                "is_carrier": is_carrier,
                "structural_tradeoff_score": float(
                    cell["structural_tradeoff_score"]
                ),
            }
        )
    return options


def _empty_cell(
    option_type: str,
    reason: str,
    event_xy: tuple[float, float],
) -> dict[str, object]:
    return {
        "q": 0.0,
        "delivery": 0.0,
        "goal": 0.0,
        "accessibility": 0.0,
        "release_time_s": None,
        "event_time_s": None,
        "event_xy": [float(event_xy[0]), float(event_xy[1])],
        "legal": False,
        "invalid_reason": reason,
        "delivery_components": {},
        "option_type": option_type,
        "proxy_flags": [
            "uncalibrated_delivery",
            "geometric_goal_danger",
            "retrospective_observed_background",
        ],
    }


def _pass_cell(
    game: Mapping[str, object],
    defender_id: str,
    defender_path_txy: Sequence[TimedPoint],
    option: Mapping[str, object],
    config: LocalGamePayoffConfig,
    arrival_config: ArrivalModelConfig,
    influence_config: GoalWeightedInfluenceConfig,
    influence_xgrid: np.ndarray,
    influence_ygrid: np.ndarray,
    goalkeeper_ids: Sequence[str],
    runtime: _SceneRuntime,
) -> dict[str, object]:
    receiver_id = str(option["option_id"])
    option_path = tuple(
        tuple(map(float, row)) for row in option["path_txy"]  # type: ignore[index]
    )
    horizon = float(game["horizon_seconds"])
    goal_xy = (float(game["goal_xy"][0]), float(game["goal_xy"][1]))
    latest_release = horizon - config.minimum_event_duration_seconds
    release_times = _sample_times(
        0.0,
        latest_release,
        config.release_step_seconds,
        include_stop=True,
    )
    best: dict[str, object] | None = None
    candidate_grid: list[dict[str, object]] = []
    last_xy = interpolate_timed_point(option_path, min(horizon, option_path[-1][0]))

    def _register_candidate(
        release_time: float,
        release_ball_xy: tuple[float, float],
        estimate,
        event_time: float,
        target_xy: tuple[float, float],
        event_velocity: tuple[float, float],
        continuation_type: str,
        through_ball_lead_m: float | None,
    ) -> None:
        nonlocal best
        if estimate is None or event_time > horizon + 1e-8:
            return
        event_frame, event_velocities = _frame_state(
            game,
            event_time,
            defender_id,
            defender_path_txy,
            attach_ball_to_carrier=False,
            runtime=runtime,
        )
        accessibility = goal_side_accessibility(
            event_frame,
            event_velocities,
            receiver_id,
            str(game["attacking_team_id"]),
            int(game["attacking_direction"]),
            (float(target_xy[0]), float(target_xy[1])),
            event_velocity_xy=event_velocity,
            goalkeeper_ids=goalkeeper_ids,
            config=influence_config,
            xgrid=influence_xgrid,
            ygrid=influence_ygrid,
        )
        mechanistic_delivery = float(
            np.clip(estimate.receive_probability, 0.0, 1.0)
        )
        xpass_geometry = None
        if config.delivery_model in {"xpass", "hybrid"}:
            xpass_geometry = predict_pass_success(
                load_xpass_model(config.xpass_model_path),
                release_ball_xy,
                (float(target_xy[0]), float(target_xy[1])),
                int(game["attacking_direction"]),
            )
        if config.delivery_model == "xpass360_kinematic":
            release_frame, release_velocities = _frame_state(
                game,
                release_time,
                defender_id,
                defender_path_txy,
                attach_ball_to_carrier=False,
                runtime=runtime,
            )
            attacking_team = str(game["attacking_team_id"])
            carrier = str(game["carrier_id"])
            tracked = []
            for player in release_frame.players.values():
                pid = str(player.object_id)
                velocity = release_velocities.get(pid)
                tracked.append(
                    TrackedPlayer(
                        float(player.x),
                        float(player.y),
                        float(velocity.vx) if velocity is not None else 0.0,
                        float(velocity.vy) if velocity is not None else 0.0,
                        teammate=str(player.team_id) == attacking_team,
                        actor=pid == carrier,
                        keeper=pid in goalkeeper_ids,
                    )
                )
            delivery = predict_pass_success_360_kinematic(
                load_kinematic_model(config.xpass_360_kinematic_model_path),
                release_ball_xy,
                (float(target_xy[0]), float(target_xy[1])),
                int(game["attacking_direction"]),
                tracked,
            )
        elif config.delivery_model == "xpass360":
            # The defending side's positions at the RELEASE moment, taken from
            # the COUNTERFACTUAL defender path. This is the channel through
            # which his trajectory is allowed to change the pass, and it is
            # the one the hybrid mode could not open without double-counting.
            release_frame, _ = _frame_state(
                game,
                release_time,
                defender_id,
                defender_path_txy,
                attach_ball_to_carrier=False,
                runtime=runtime,
            )
            attacking_team = str(game["attacking_team_id"])
            opponents = [
                (float(player.x), float(player.y))
                for player in release_frame.players.values()
                if str(player.team_id) != attacking_team
                and str(player.object_id) not in goalkeeper_ids
            ]
            delivery = predict_pass_success_360(
                load_xpass_model(config.xpass_360_model_path),
                release_ball_xy,
                (float(target_xy[0]), float(target_xy[1])),
                int(game["attacking_direction"]),
                opponents,
            )
        elif config.delivery_model == "xpass":
            delivery = float(xpass_geometry)
        elif config.delivery_model == "hybrid":
            # xPass supplies the learned geometric execution rate; the
            # defender-interaction terms stay mechanistic so a counterfactual
            # defender path can still move P.  The average defensive presence
            # already baked into xPass double-counts with these terms — an
            # accepted v0 bias, to be revisited with the 360-trained model.
            defender_interaction = float(
                np.clip(
                    estimate.path_survival_probability
                    * estimate.receiver_first_probability
                    * estimate.secure_possession_probability
                    * estimate.pressure_execution_probability,
                    0.0,
                    1.0,
                )
            )
            delivery = float(
                np.clip(xpass_geometry * defender_interaction, 0.0, 1.0)
            )
        else:
            delivery = mechanistic_delivery
        if config.delivery_calibration_path:
            delivery = apply_calibration(
                config.delivery_calibration_path, float(delivery)
            )
        stack = config.value_stack
        if stack in {"ssac", "ssac_threat", "ssac_pass", "ssac_ourcarry",
                     "ssac_keepaccess"}:
            # The full stack is the intended configuration; the two partial
            # ones exist only to attribute a reordering to a factor.
            carrier_state = event_frame.players.get(str(game["carrier_id"]))
            support_xy = (
                (float(carrier_state.x), float(carrier_state.y))
                if carrier_state is not None
                else (float(target_xy[0]), float(target_xy[1]))
            )
            defender_state = event_frame.players.get(str(defender_id))
            defender_xy = (
                (float(defender_state.x), float(defender_state.y))
                if defender_state is not None
                else support_xy
            )
            if stack in {"ssac", "ssac_threat", "ssac_ourcarry",
                         "ssac_keepaccess"}:
                goal = ssac_threat_at_point(
                    (float(target_xy[0]), float(target_xy[1])),
                    support_xy,
                    defender_xy,
                    int(game["attacking_direction"]),
                )
            else:
                goal = float(
                    score_at_points(
                        [(float(target_xy[0]), float(target_xy[1]))],
                        int(game["attacking_direction"]),
                        epv_grid_path=config.goal_danger_grid_path,
                    )[0]
                )
            receiver_state = event_frame.players.get(receiver_id)
            receiver_xy = (
                (float(receiver_state.x), float(receiver_state.y))
                if receiver_state is not None
                else (float(target_xy[0]), float(target_xy[1]))
            )
            if stack in {"ssac", "ssac_pass", "ssac_ourcarry",
                         "ssac_keepaccess"}:
                delivery = ssac_completion_probability(
                    config.ssac_pass_model_path,
                    support_xy,
                    receiver_xy,
                    (float(target_xy[0]), float(target_xy[1])),
                    ssac_defending_positions(
                        event_frame, str(game["attacking_team_id"])
                    ),
                    int(game["attacking_direction"]),
                )
            # Only the full stack folds accessibility away, so a partial arm
            # keeps it and differs from native in exactly one factor.
            access = (
                1.0 if stack in {"ssac", "ssac_ourcarry"}
                else float(np.clip(accessibility.accessibility_score, 0.0, 1.0))
            )
        elif stack == "native_no_access":
            goal = float(
                score_at_points(
                    [(float(target_xy[0]), float(target_xy[1]))],
                    int(game["attacking_direction"]),
                    epv_grid_path=config.goal_danger_grid_path,
                )[0]
            )
            access = 1.0
        else:
            goal = float(
                score_at_points(
                    [(float(target_xy[0]), float(target_xy[1]))],
                    int(game["attacking_direction"]),
                    epv_grid_path=config.goal_danger_grid_path,
                )[0]
            )
            access = float(np.clip(accessibility.accessibility_score, 0.0, 1.0))
        goal, access = _apply_factor_ablations(config, goal, access)
        q_value = float(delivery * goal * access)
        candidate = {
            "q": q_value,
            "delivery": delivery,
            "goal": goal,
            "accessibility": access,
            "release_time_s": float(release_time),
            "event_time_s": float(event_time),
            "event_xy": [float(target_xy[0]), float(target_xy[1])],
            "legal": True,
            "invalid_reason": None,
            "option_type": str(option["option_type"]),
            "continuation_type": continuation_type,
            "through_ball_lead_m": through_ball_lead_m,
            "delivery_components": {
                "mechanistic_delivery": mechanistic_delivery,
                "xpass_geometry": xpass_geometry,
                "pass_execution": float(estimate.pass_execution_probability),
                "passer_pressure": float(estimate.pressure_execution_probability),
                "distance_execution": float(estimate.distance_execution_probability),
                "path_survival": float(estimate.path_survival_probability),
                "receiver_first": float(estimate.receiver_first_probability),
                "secure_possession": float(
                    estimate.secure_possession_probability
                ),
                "pass_distance_m": float(estimate.pass_distance_m),
                "ball_arrival_time_s": float(estimate.ball_arrival_time_s),
                "receiver_arrival_time_s": float(estimate.receiver_arrival_time_s),
                "target_defender_margin_s": float(
                    estimate.defender_time_margin_s
                ),
                "path_margin_s": float(estimate.path_time_margin_s),
                "path_blocker_id": estimate.path_suppression_defender_id,
                "target_blocker_id": estimate.nearest_defender_id,
                "passer_pressure_defender_id": estimate.passer_pressure_defender_id,
            },
            "accessibility_components": {
                "covered_fraction": float(accessibility.covered_fraction),
                "intrinsic_value": float(accessibility.intrinsic_value),
                "residual_value": float(accessibility.residual_value),
                "residual_peak": float(accessibility.residual_peak),
            },
            "proxy_flags": [
                "uncalibrated_delivery",
                "geometric_goal_danger",
                "retrospective_observed_background",
                (
                    "observed_receiver_future"
                    if continuation_type == "receive_to_feet"
                    else "synthetic_through_ball_target"
                ),
            ],
        }
        if config.record_candidate_grid:
            candidate_grid.append(
                {
                    "release": float(release_time),
                    "type": continuation_type,
                    "lead": through_ball_lead_m,
                    "q": round(q_value, 6),
                    "event_time": round(float(event_time), 3),
                }
            )
        rank = (
            q_value,
            delivery * goal,
            -float(event_time),
            -float(release_time),
        )
        if best is None or rank > best["_rank"]:  # type: ignore[index]
            best = {**candidate, "_rank": rank}

    for release_time in release_times:
        release_frame, release_velocities = _frame_state(
            game,
            release_time,
            defender_id,
            defender_path_txy,
            attach_ball_to_carrier=True,
            runtime=runtime,
        )
        if release_frame.ball is None:
            continue
        ball_xy = (float(release_frame.ball.x), float(release_frame.ball.y))
        if is_offside_position(
            release_frame,
            receiver_id,
            str(game["attacking_team_id"]),
            ball_xy,
            int(game["attacking_direction"]),
        ):
            continue

        receiver_now = release_frame.players[receiver_id]

        # --- Continuation: through balls into the receiver's running lane ---
        release_velocity = release_velocities[receiver_id]
        if release_velocity.speed >= config.through_ball_minimum_run_speed_mps:
            unit_x = release_velocity.vx / release_velocity.speed
            unit_y = release_velocity.vy / release_velocity.speed
            for lead in config.through_ball_lead_distances_m:
                base_x = float(receiver_now.x) + float(lead) * unit_x
                base_y = float(receiver_now.y) + float(lead) * unit_y
                goal_dx = goal_xy[0] - base_x
                goal_dy = goal_xy[1] - base_y
                goal_norm = math.hypot(goal_dx, goal_dy)
                if goal_norm > 1e-9:
                    base_x += (
                        config.through_ball_goal_side_offset_m * goal_dx / goal_norm
                    )
                    base_y += (
                        config.through_ball_goal_side_offset_m * goal_dy / goal_norm
                    )
                through_target = (base_x, base_y)
                through_estimate = point_reception_estimate(
                    release_frame,
                    release_velocities,
                    receiver_id,
                    str(game["attacking_team_id"]),
                    ball_xy,
                    through_target,
                    arrival_config,
                    passer_id=str(game["carrier_id"]),
                )
                through_event = release_time + max(
                    config.minimum_event_duration_seconds,
                    float(through_estimate.ball_arrival_time_s),
                    float(through_estimate.receiver_arrival_time_s),
                )
                run_dx = through_target[0] - float(receiver_now.x)
                run_dy = through_target[1] - float(receiver_now.y)
                run_distance = math.hypot(run_dx, run_dy)
                arrival = max(
                    float(through_estimate.receiver_arrival_time_s), 1e-6
                )
                run_speed = min(
                    arrival_config.player_max_speed_mps,
                    run_distance / arrival,
                )
                event_velocity = (
                    (run_dx / run_distance * run_speed, run_dy / run_distance * run_speed)
                    if run_distance > 1e-9
                    else (0.0, 0.0)
                )
                _register_candidate(
                    release_time,
                    ball_xy,
                    through_estimate,
                    through_event,
                    through_target,
                    event_velocity,
                    "through_ball_to_space",
                    float(lead),
                )

        # --- Continuation: cutback / lay-off coming toward the ball ---
        for cut_lead in config.cutback_lead_distances_m:
            toward_x = ball_xy[0] - float(receiver_now.x)
            toward_y = ball_xy[1] - float(receiver_now.y)
            toward_norm = math.hypot(toward_x, toward_y)
            if toward_norm <= float(cut_lead) + 1e-6:
                continue
            cut_target = (
                float(receiver_now.x) + float(cut_lead) * toward_x / toward_norm,
                float(receiver_now.y) + float(cut_lead) * toward_y / toward_norm,
            )
            cut_estimate = point_reception_estimate(
                release_frame,
                release_velocities,
                receiver_id,
                str(game["attacking_team_id"]),
                ball_xy,
                cut_target,
                arrival_config,
                passer_id=str(game["carrier_id"]),
            )
            cut_event = release_time + max(
                config.minimum_event_duration_seconds,
                float(cut_estimate.ball_arrival_time_s),
                float(cut_estimate.receiver_arrival_time_s),
            )
            cut_arrival = max(float(cut_estimate.receiver_arrival_time_s), 1e-6)
            cut_speed = min(
                arrival_config.player_max_speed_mps,
                float(cut_lead) / cut_arrival,
            )
            _register_candidate(
                release_time,
                ball_xy,
                cut_estimate,
                cut_event,
                cut_target,
                (
                    toward_x / toward_norm * cut_speed,
                    toward_y / toward_norm * cut_speed,
                ),
                "cutback_layoff",
                float(cut_lead),
            )

        # --- Continuation: meet the ball on the observed path (to feet) ---
        initial_distance = math.hypot(
            receiver_now.x - ball_xy[0], receiver_now.y - ball_xy[1]
        )
        target_time = min(
            horizon,
            release_time
            + max(
                config.minimum_event_duration_seconds,
                initial_distance / max(arrival_config.pass_speed_mps, 1e-9),
            ),
        )
        estimate = None
        target_xy = interpolate_timed_point(option_path, target_time)
        for _ in range(6):
            target_xy = interpolate_timed_point(option_path, target_time)
            estimate = point_reception_estimate(
                release_frame,
                release_velocities,
                receiver_id,
                str(game["attacking_team_id"]),
                ball_xy,
                target_xy,
                arrival_config,
                passer_id=str(game["carrier_id"]),
            )
            duration = max(
                config.minimum_event_duration_seconds,
                float(estimate.ball_arrival_time_s),
                float(estimate.receiver_arrival_time_s),
            )
            proposed = release_time + duration
            if proposed > horizon + 1e-8:
                estimate = None
                break
            updated = 0.5 * target_time + 0.5 * proposed
            if abs(updated - target_time) <= 0.01:
                target_time = float(proposed)
                break
            target_time = float(updated)
        if estimate is None or target_time > horizon + 1e-8:
            continue

        # Re-evaluate at the converged moving-receiver target.
        target_xy = interpolate_timed_point(option_path, target_time)
        estimate = point_reception_estimate(
            release_frame,
            release_velocities,
            receiver_id,
            str(game["attacking_team_id"]),
            ball_xy,
            target_xy,
            arrival_config,
            passer_id=str(game["carrier_id"]),
        )
        event_time = release_time + max(
            config.minimum_event_duration_seconds,
            float(estimate.ball_arrival_time_s),
            float(estimate.receiver_arrival_time_s),
        )
        if event_time > horizon + 1e-8:
            continue
        target_xy = interpolate_timed_point(option_path, event_time)
        # A final point estimate keeps P and A aligned at the same event target.
        estimate = point_reception_estimate(
            release_frame,
            release_velocities,
            receiver_id,
            str(game["attacking_team_id"]),
            ball_xy,
            target_xy,
            arrival_config,
            passer_id=str(game["carrier_id"]),
        )
        event_time = release_time + max(
            config.minimum_event_duration_seconds,
            float(estimate.ball_arrival_time_s),
            float(estimate.receiver_arrival_time_s),
        )
        _register_candidate(
            release_time,
            ball_xy,
            estimate,
            event_time,
            (float(target_xy[0]), float(target_xy[1])),
            _path_velocity(option_path, event_time),
            "receive_to_feet",
            None,
        )
    if config.terminal_structure_discount > 0.0:
        terminal_frame, terminal_velocities = _frame_state(
            game,
            horizon,
            defender_id,
            defender_path_txy,
            attach_ball_to_carrier=True,
            runtime=runtime,
        )
        ball_h = (float(terminal_frame.ball.x), float(terminal_frame.ball.y))
        z_h = interpolate_timed_point(option_path, horizon)
        if not is_offside_position(
            terminal_frame,
            receiver_id,
            str(game["attacking_team_id"]),
            ball_h,
            int(game["attacking_direction"]),
        ):
            terminal_distance = math.hypot(
                z_h[0] - ball_h[0], z_h[1] - ball_h[1]
            )
            deliver = config.terminal_structure_discount * math.exp(
                -terminal_distance / arrival_config.pass_distance_scale_m
            )
            contest = 1.0
            for player_id, player in terminal_frame.players.items():
                if (
                    player.team_id == str(game["attacking_team_id"])
                    or player_id in goalkeeper_ids
                ):
                    continue
                arrival = float(
                    player_time_to_point(
                        player,
                        terminal_velocities[player_id],
                        z_h,
                        arrival_config,
                    )
                )
                contest *= _logistic(
                    (arrival - config.carry_pressure_action_time_seconds)
                    / config.carry_pressure_sigma_seconds
                )
            goal_h = float(
                score_at_points(
                    [(float(z_h[0]), float(z_h[1]))],
                    int(game["attacking_direction"]),
                    epv_grid_path=config.goal_danger_grid_path,
                )[0]
            )
            goal_h, contest = _apply_factor_ablations(config, goal_h, contest)
            q_terminal = float(deliver * goal_h * contest)
            if config.record_candidate_grid:
                candidate_grid.append(
                    {
                        "release": float(horizon),
                        "type": "terminal_structure",
                        "lead": None,
                        "q": round(q_terminal, 6),
                        "event_time": round(float(horizon), 3),
                    }
                )
            rank = (q_terminal, deliver * goal_h, -float(horizon), -float(horizon))
            if best is None or rank > best["_rank"]:  # type: ignore[index]
                best = {
                    "q": q_terminal,
                    "delivery": float(deliver),
                    "goal": goal_h,
                    "accessibility": float(contest),
                    "release_time_s": float(horizon),
                    "event_time_s": float(horizon),
                    "event_xy": [float(z_h[0]), float(z_h[1])],
                    "legal": True,
                    "invalid_reason": None,
                    "option_type": str(option["option_type"]),
                    "continuation_type": "terminal_structure",
                    "through_ball_lead_m": None,
                    "delivery_components": {
                        "terminal_distance_m": float(terminal_distance),
                        "terminal_discount": float(
                            config.terminal_structure_discount
                        ),
                    },
                    "accessibility_components": {
                        "contest_product": float(contest),
                    },
                    "proxy_flags": [
                        "uncalibrated_delivery",
                        "geometric_goal_danger",
                        "post_horizon_structure_proxy",
                    ],
                    "_rank": rank,
                }

    if best is None:
        empty = _empty_cell(
            str(option["option_type"]),
            "offside_or_no_delivery_within_horizon",
            last_xy,
        )
        if config.record_candidate_grid:
            empty["candidate_grid"] = candidate_grid
        return empty
    best.pop("_rank", None)
    if config.record_candidate_grid:
        best["candidate_grid"] = candidate_grid
    return best


def _apply_factor_ablations(config, goal: float, access: float):
    """Return (G, A) with either factor pinned to 1 when ablated."""
    if config.disable_goal_danger:
        goal = 1.0
    if config.disable_accessibility:
        access = 1.0
    return float(goal), float(access)


def _logistic(value: float) -> float:
    return float(1.0 / (1.0 + math.exp(-float(np.clip(value, -60.0, 60.0)))))


def _carry_retention_at_time(
    game: Mapping[str, object],
    defender_id: str,
    defender_path_txy: Sequence[TimedPoint],
    carrier_path_txy: Sequence[TimedPoint],
    time_s: float,
    goalkeeper_ids: Sequence[str],
    config: LocalGamePayoffConfig,
    arrival_config: ArrivalModelConfig,
    runtime: _SceneRuntime,
) -> tuple[float, str | None, float]:
    frame, velocities = _frame_state(
        game,
        time_s,
        defender_id,
        defender_path_txy,
        attach_ball_to_carrier=True,
        runtime=runtime,
    )
    target_xy = interpolate_timed_point(carrier_path_txy, time_s)
    excluded = set(goalkeeper_ids)
    arrivals = [
        (
            float(
                player_time_to_point(
                    defender,
                    velocities[other_id],
                    target_xy,
                    arrival_config,
                )
            ),
            other_id,
        )
        for other_id, defender in frame.players.items()
        if defender.team_id != str(game["attacking_team_id"])
        and other_id not in excluded
    ]
    if not arrivals:
        return 1.0, None, math.inf
    nearest_time, nearest_id = min(arrivals)
    if config.carry_pressure_model == "independent_race":
        # Every converging defender independently gets a chance to break the
        # carry, so retention is the product of per-defender survival terms.
        # A second defender arriving shortly after the first now lowers the
        # value instead of being priced at zero.
        retention = 1.0
        for arrival_time, _ in arrivals:
            retention *= _logistic(
                (arrival_time - config.carry_pressure_action_time_seconds)
                / config.carry_pressure_sigma_seconds
            )
    else:
        retention = _logistic(
            (nearest_time - config.carry_pressure_action_time_seconds)
            / config.carry_pressure_sigma_seconds
        )
    return float(retention), nearest_id, float(nearest_time)


def _carry_extension_race(
    game: Mapping[str, object],
    runtime: _SceneRuntime,
    defender_id: str,
    defender_path_txy: Sequence[TimedPoint],
    event_time: float,
    event_xy: tuple[float, float],
    unit_xy: tuple[float, float],
    lead: float,
    arrival_time: float,
    goalkeeper_ids: Sequence[str],
    config: LocalGamePayoffConfig,
    arrival_config: ArrivalModelConfig,
) -> float:
    """Interception race along the forward carry extension.

    Prices the hypothetical segment event_xy -> event_xy + lead*unit exactly
    like the observed carry path: at sampled pass-through instants every
    defending outfielder races the carrier to the sampled point, per-sample
    survival follows carry_pressure_model, and the weakest sample wins.  The
    endpoint is always sampled, so this subsumes the arrival contest.
    """

    step = max(config.carry_extension_sample_step_m, 1e-6)
    distances = [
        min(float(lead), step * index)
        for index in range(1, int(float(lead) / step) + 1)
    ]
    if not distances or distances[-1] < float(lead) - 1e-9:
        distances.append(float(lead))
    duration = max(float(arrival_time) - float(event_time), 1e-9)
    weakest = 1.0
    for distance in distances:
        fraction = distance / float(lead)
        sample_time = float(event_time) + fraction * duration
        point = (
            event_xy[0] + distance * unit_xy[0],
            event_xy[1] + distance * unit_xy[1],
        )
        frame, velocities = _frame_state(
            game,
            sample_time,
            defender_id,
            defender_path_txy,
            attach_ball_to_carrier=True,
            runtime=runtime,
        )
        survival = 1.0
        nearest = math.inf
        for player_id, player in frame.players.items():
            if (
                player.team_id == str(game["attacking_team_id"])
                or player_id in goalkeeper_ids
            ):
                continue
            arrival = float(
                player_time_to_point(
                    player,
                    velocities[player_id],
                    point,
                    arrival_config,
                )
            )
            if config.carry_pressure_model == "independent_race":
                survival *= _logistic(
                    (arrival - config.carry_pressure_action_time_seconds)
                    / config.carry_pressure_sigma_seconds
                )
            else:
                nearest = min(nearest, arrival)
        if config.carry_pressure_model != "independent_race":
            survival = (
                1.0
                if math.isinf(nearest)
                else _logistic(
                    (nearest - config.carry_pressure_action_time_seconds)
                    / config.carry_pressure_sigma_seconds
                )
            )
        weakest = min(weakest, survival)
    return float(weakest)


def _confrontation_leads(
    game: Mapping[str, object],
    runtime: _SceneRuntime,
    event_time: float,
    event_xy: tuple[float, float],
    unit_xy: tuple[float, float],
    config: LocalGamePayoffConfig,
) -> tuple[float, ...]:
    """Leads reaching the first defender blocking the observed carry ray.

    Computed from the OBSERVED frame (no counterfactual substitution) so the
    candidate set is identical across defender responses.
    """

    frame, _ = _frame_state(game, event_time, runtime=runtime)
    nearest_along: float | None = None
    for player in frame.players.values():
        if player.team_id == str(game["attacking_team_id"]):
            continue
        dx = float(player.x) - event_xy[0]
        dy = float(player.y) - event_xy[1]
        along = dx * unit_xy[0] + dy * unit_xy[1]
        if along <= 0.5 or along > config.carry_confrontation_max_lead_m:
            continue
        perpendicular = abs(-unit_xy[1] * dx + unit_xy[0] * dy)
        if perpendicular > config.carry_confrontation_corridor_m:
            continue
        if nearest_along is None or along < nearest_along:
            nearest_along = along
    if nearest_along is None:
        return ()
    return tuple(
        lead for lead in (nearest_along - 1.0, nearest_along + 1.0) if lead >= 0.5
    )


def _carry_cell(
    game: Mapping[str, object],
    defender_id: str,
    defender_path_txy: Sequence[TimedPoint],
    option: Mapping[str, object],
    config: LocalGamePayoffConfig,
    arrival_config: ArrivalModelConfig,
    influence_config: GoalWeightedInfluenceConfig,
    influence_xgrid: np.ndarray,
    influence_ygrid: np.ndarray,
    goalkeeper_ids: Sequence[str],
    runtime: _SceneRuntime,
) -> dict[str, object]:
    carrier_path = tuple(
        tuple(map(float, row)) for row in option["path_txy"]  # type: ignore[index]
    )
    horizon = float(game["horizon_seconds"])
    carry_grid: list[dict[str, object]] = []
    event_times = _sample_times(
        config.minimum_event_duration_seconds,
        horizon,
        config.release_step_seconds,
        include_stop=True,
    )
    best: dict[str, object] | None = None
    for event_time in event_times:
        path_times = _sample_times(
            config.minimum_event_duration_seconds,
            event_time,
            config.carry_path_step_seconds,
            include_stop=True,
        )
        samples = [
            _carry_retention_at_time(
                game,
                defender_id,
                defender_path_txy,
                carrier_path,
                time_s,
                goalkeeper_ids,
                config,
                arrival_config,
                runtime,
            )
            for time_s in path_times
        ]
        if not samples:
            continue
        weakest_index = min(range(len(samples)), key=lambda index: samples[index][0])
        path_retention, pressure_id, pressure_time = samples[weakest_index]
        terminal_retention, terminal_id, terminal_time = samples[-1]
        event_xy = interpolate_timed_point(carrier_path, event_time)
        event_velocity = _path_velocity(carrier_path, event_time)
        event_frame, event_velocities = _frame_state(
            game,
            event_time,
            defender_id,
            defender_path_txy,
            attach_ball_to_carrier=True,
            runtime=runtime,
        )
        accessibility = goal_side_accessibility(
            event_frame,
            event_velocities,
            str(game["carrier_id"]),
            str(game["attacking_team_id"]),
            int(game["attacking_direction"]),
            (float(event_xy[0]), float(event_xy[1])),
            event_velocity_xy=event_velocity,
            goalkeeper_ids=goalkeeper_ids,
            config=influence_config,
            xgrid=influence_xgrid,
            ygrid=influence_ygrid,
        )
        stack = config.value_stack
        if stack in {"ssac", "ssac_threat", "ssac_pass", "ssac_ourcarry",
                     "ssac_keepaccess"}:
            # The carrier IS the ball, so the support is the receiver, which is
            # this module's convention for retained possession.
            runner_state = event_frame.players.get(str(game["runner_id"]))
            support_xy = (
                (float(runner_state.x), float(runner_state.y))
                if runner_state is not None
                else (float(event_xy[0]), float(event_xy[1]))
            )
            defender_state = event_frame.players.get(str(defender_id))
            defender_xy = (
                (float(defender_state.x), float(defender_state.y))
                if defender_state is not None
                else support_xy
            )
            if stack in {"ssac", "ssac_threat", "ssac_ourcarry",
                         "ssac_keepaccess"}:
                goal = ssac_threat_at_point(
                    (float(event_xy[0]), float(event_xy[1])),
                    support_xy,
                    defender_xy,
                    int(game["attacking_direction"]),
                )
            else:
                goal = float(
                    score_at_points(
                        [(float(event_xy[0]), float(event_xy[1]))],
                        int(game["attacking_direction"]),
                        epv_grid_path=config.goal_danger_grid_path,
                    )[0]
                )
            if stack in {"ssac", "ssac_pass", "ssac_keepaccess"}:
                # The hazard integrates the whole approach, so a defender who
                # sits on the carrier for two seconds costs more than one who
                # arrives at the end -- the gap pass-shaped pricing cannot see.
                path_retention = ssac_carry_survival(
                    [row for row in carrier_path
                     if float(row[0]) <= float(event_time)],
                    defender_path_txy,
                    tackle_rate=config.ssac_tackle_rate,
                    tackle_radius_m=config.ssac_tackle_radius_m,
                    tackle_softness_m=config.ssac_tackle_softness_m,
                )
            elif stack == "ssac_ourcarry":
                # carrier_first x secure x path_retention, on the same race
                # form the pass side uses, so a carry and a pass finally
                # answer one question on one scale. None means the carry was
                # not priceable here; keep the existing value rather than
                # inventing a zero.
                ours = carry_point_estimate(
                    event_frame,
                    event_velocities,
                    str(game["carrier_id"]),
                    str(game["attacking_team_id"]),
                    (float(event_xy[0]), float(event_xy[1])),
                    arrival_config,
                    goalkeeper_ids,
                )
                if ours is not None:
                    path_retention = float(
                        np.clip(ours.carry_probability, 0.0, 1.0)
                    )
            access = (
                1.0 if stack in {"ssac", "ssac_ourcarry"}
                else float(np.clip(accessibility.accessibility_score, 0.0, 1.0))
            )
        elif stack == "native_no_access":
            goal = float(
                score_at_points(
                    [(float(event_xy[0]), float(event_xy[1]))],
                    int(game["attacking_direction"]),
                    epv_grid_path=config.goal_danger_grid_path,
                )[0]
            )
            access = 1.0
        else:
            goal = float(
                score_at_points(
                    [(float(event_xy[0]), float(event_xy[1]))],
                    int(game["attacking_direction"]),
                    epv_grid_path=config.goal_danger_grid_path,
                )[0]
            )
            access = float(np.clip(accessibility.accessibility_score, 0.0, 1.0))
        goal, access = _apply_factor_ablations(config, goal, access)
        q_value = float(path_retention * goal * access)
        candidate = {
            "q": q_value,
            "delivery": float(path_retention),
            "goal": goal,
            "accessibility": access,
            "release_time_s": 0.0,
            "event_time_s": float(event_time),
            "event_xy": [float(event_xy[0]), float(event_xy[1])],
            "legal": True,
            "invalid_reason": None,
            "option_type": "carrier_carry",
            "continuation_type": "carrier_carry",
            "through_ball_lead_m": None,
            "delivery_components": {
                "path_min_retention": float(path_retention),
                "terminal_retention": float(terminal_retention),
                "weakest_pressure_time_s": float(path_times[weakest_index]),
                "weakest_pressure_defender_id": pressure_id,
                "weakest_pressure_arrival_time_s": float(pressure_time),
                "terminal_pressure_defender_id": terminal_id,
                "terminal_pressure_arrival_time_s": float(terminal_time),
            },
            "accessibility_components": {
                "covered_fraction": float(accessibility.covered_fraction),
                "intrinsic_value": float(accessibility.intrinsic_value),
                "residual_value": float(accessibility.residual_value),
                "residual_peak": float(accessibility.residual_peak),
            },
            "proxy_flags": [
                "uncalibrated_carry_retention",
                "geometric_goal_danger",
                "retrospective_observed_background",
                "observed_carrier_path_with_possession_reattached",
            ],
        }
        if config.record_candidate_grid:
            carry_grid.append(
                {
                    "release": 0.0,
                    "type": "carrier_carry",
                    "lead": None,
                    "q": round(q_value, 6),
                    "event_time": round(float(event_time), 3),
                }
            )
        rank = (q_value, path_retention * goal, -float(event_time))
        if best is None or rank > best["_rank"]:  # type: ignore[index]
            best = {**candidate, "_rank": rank}

        # --- Carrier leverage: carrying into the space ahead ---
        carrier_speed = math.hypot(*event_velocity)
        if (
            config.carry_lead_distances_m
            and carrier_speed >= config.through_ball_minimum_run_speed_mps
        ):
            unit_x = event_velocity[0] / carrier_speed
            unit_y = event_velocity[1] / carrier_speed
            leads = tuple(config.carry_lead_distances_m) + _confrontation_leads(
                game,
                runtime,
                event_time,
                (float(event_xy[0]), float(event_xy[1])),
                (unit_x, unit_y),
                config,
            )
            for lead in leads:
                # The carrier only REACHES the forward point later; pricing
                # the contest at that arrival instant is what lets the
                # candidate defender's counterfactual path move this value.
                arrival_time = event_time + float(lead) / carrier_speed
                if arrival_time > horizon + 1e-9:
                    continue
                z_lead = (
                    float(event_xy[0]) + float(lead) * unit_x,
                    float(event_xy[1]) + float(lead) * unit_y,
                )
                lead_frame, lead_velocities = _frame_state(
                    game,
                    arrival_time,
                    defender_id,
                    defender_path_txy,
                    attach_ball_to_carrier=True,
                    runtime=runtime,
                )
                if config.carry_lead_pricing == "extension_race":
                    contest = _carry_extension_race(
                        game,
                        runtime,
                        defender_id,
                        defender_path_txy,
                        float(event_time),
                        (float(event_xy[0]), float(event_xy[1])),
                        (unit_x, unit_y),
                        float(lead),
                        float(arrival_time),
                        goalkeeper_ids,
                        config,
                        arrival_config,
                    )
                else:
                    contest = 1.0
                    for player_id, player in lead_frame.players.items():
                        if (
                            player.team_id == str(game["attacking_team_id"])
                            or player_id in goalkeeper_ids
                        ):
                            continue
                        arrival = float(
                            player_time_to_point(
                                player,
                                lead_velocities[player_id],
                                z_lead,
                                arrival_config,
                            )
                        )
                        contest *= _logistic(
                            (arrival - config.carry_pressure_action_time_seconds)
                            / config.carry_pressure_sigma_seconds
                        )
                goal_lead = float(
                    score_at_points(
                        [z_lead],
                        int(game["attacking_direction"]),
                        epv_grid_path=config.goal_danger_grid_path,
                    )[0]
                )
                access_lead = 1.0
                if config.carry_lead_pricing == "goal_side_accessibility":
                    lead_accessibility = goal_side_accessibility(
                        lead_frame,
                        lead_velocities,
                        str(game["carrier_id"]),
                        str(game["attacking_team_id"]),
                        int(game["attacking_direction"]),
                        z_lead,
                        event_velocity_xy=event_velocity,
                        goalkeeper_ids=goalkeeper_ids,
                        config=influence_config,
                        xgrid=influence_xgrid,
                        ygrid=influence_ygrid,
                    )
                    access_lead = float(
                        np.clip(lead_accessibility.accessibility_score, 0.0, 1.0)
                    )
                q_lead = float(path_retention * goal_lead * access_lead * contest)
                if config.record_candidate_grid:
                    carry_grid.append(
                        {
                            "release": 0.0,
                            "type": "carry_into_space",
                            "lead": float(lead),
                            "q": round(q_lead, 6),
                            "event_time": round(float(arrival_time), 3),
                        }
                    )
                lead_rank = (q_lead, path_retention * goal_lead, -float(arrival_time))
                if best is None or lead_rank > best["_rank"]:  # type: ignore[index]
                    best = {
                        "q": q_lead,
                        "delivery": float(path_retention),
                        "goal": goal_lead,
                        "accessibility": float(access_lead * contest),
                        "release_time_s": 0.0,
                        "event_time_s": float(arrival_time),
                        "event_xy": [z_lead[0], z_lead[1]],
                        "legal": True,
                        "invalid_reason": None,
                        "option_type": "carrier_carry",
                        "continuation_type": "carry_into_space",
                        "through_ball_lead_m": float(lead),
                        "delivery_components": {
                            "path_min_retention": float(path_retention),
                            "weakest_pressure_defender_id": pressure_id,
                        },
                        "accessibility_components": {
                            "contest_product": float(contest),
                            "goal_side_accessibility": float(access_lead),
                        },
                        "proxy_flags": [
                            "uncalibrated_carry_retention",
                            "geometric_goal_danger",
                            "carry_space_extension_proxy",
                        ],
                        "_rank": lead_rank,
                    }
    if config.terminal_structure_discount > 0.0:
        terminal_frame, terminal_velocities = _frame_state(
            game,
            horizon,
            defender_id,
            defender_path_txy,
            attach_ball_to_carrier=True,
            runtime=runtime,
        )
        z_h = interpolate_timed_point(carrier_path, horizon)
        deliver = float(config.terminal_structure_discount)
        contest = 1.0
        for player_id, player in terminal_frame.players.items():
            if (
                player.team_id == str(game["attacking_team_id"])
                or player_id in goalkeeper_ids
            ):
                continue
            arrival = float(
                player_time_to_point(
                    player,
                    terminal_velocities[player_id],
                    z_h,
                    arrival_config,
                )
            )
            contest *= _logistic(
                (arrival - config.carry_pressure_action_time_seconds)
                / config.carry_pressure_sigma_seconds
            )
        goal_h = float(
            score_at_points(
                [(float(z_h[0]), float(z_h[1]))],
                int(game["attacking_direction"]),
                epv_grid_path=config.goal_danger_grid_path,
            )[0]
        )
        goal_h, contest = _apply_factor_ablations(config, goal_h, contest)
        q_terminal = float(deliver * goal_h * contest)
        if config.record_candidate_grid:
            carry_grid.append(
                {
                    "release": float(horizon),
                    "type": "terminal_structure",
                    "lead": None,
                    "q": round(q_terminal, 6),
                    "event_time": round(float(horizon), 3),
                }
            )
        rank = (q_terminal, deliver * goal_h, -float(horizon))
        if best is None or rank > best["_rank"]:  # type: ignore[index]
            best = {
                "q": q_terminal,
                "delivery": deliver,
                "goal": goal_h,
                "accessibility": float(contest),
                "release_time_s": float(horizon),
                "event_time_s": float(horizon),
                "event_xy": [float(z_h[0]), float(z_h[1])],
                "legal": True,
                "invalid_reason": None,
                "option_type": "carrier_carry",
                "continuation_type": "terminal_structure",
                "through_ball_lead_m": None,
                "delivery_components": {
                    "terminal_discount": deliver,
                },
                "accessibility_components": {
                    "contest_product": float(contest),
                },
                "proxy_flags": [
                    "uncalibrated_carry_retention",
                    "geometric_goal_danger",
                    "post_horizon_structure_proxy",
                ],
                "_rank": rank,
            }

    if best is None:
        empty = _empty_cell(
            "carrier_carry",
            "no_carry_event_within_horizon",
            interpolate_timed_point(carrier_path, horizon),
        )
        if config.record_candidate_grid:
            empty["candidate_grid"] = carry_grid
        return empty
    best.pop("_rank", None)
    if config.record_candidate_grid:
        best["candidate_grid"] = carry_grid
    return best


def _evaluate_cell(
    game: Mapping[str, object],
    defender_id: str,
    defender_path_txy: Sequence[TimedPoint],
    option: Mapping[str, object],
    config: LocalGamePayoffConfig,
    arrival_config: ArrivalModelConfig,
    influence_config: GoalWeightedInfluenceConfig,
    influence_xgrid: np.ndarray,
    influence_ygrid: np.ndarray,
    goalkeeper_ids: Sequence[str],
    runtime: _SceneRuntime,
) -> dict[str, object]:
    if bool(option["is_carrier"]):
        return _carry_cell(
            game,
            defender_id,
            defender_path_txy,
            option,
            config,
            arrival_config,
            influence_config,
            influence_xgrid,
            influence_ygrid,
            goalkeeper_ids,
            runtime,
        )
    return _pass_cell(
        game,
        defender_id,
        defender_path_txy,
        option,
        config,
        arrival_config,
        influence_config,
        influence_xgrid,
        influence_ygrid,
        goalkeeper_ids,
        runtime,
    )


def _select_anchor_index(
    scores: Sequence[float],
    tail_costs: Sequence[float],
    efforts: Sequence[float],
    tolerance: float,
) -> int:
    """Pick a representative among near-tied minima by tail sensibleness.

    Value-indifferent path tails (segments after the last priced pass window)
    otherwise decide anchors arbitrarily; among scores within the relative
    tolerance of the minimum, prefer the response whose terminal position
    keeps goal-side contact with the anchored option(s).
    """

    best = min(scores)
    limit = best + max(1e-9, abs(best) * tolerance)
    eligible = [index for index, score in enumerate(scores) if score <= limit]
    return min(eligible, key=lambda index: (tail_costs[index], efforts[index], index))


def _response_effects(
    q_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    minimum = np.min(q_matrix, axis=1)
    maximum = np.max(q_matrix, axis=1)
    effect = maximum - minimum
    relative = np.divide(
        effect,
        np.maximum(maximum, 1e-9),
        out=np.zeros_like(effect),
        where=maximum > 1e-9,
    )
    return minimum, maximum, relative


def _local_option_indices(
    options: Sequence[Mapping[str, object]],
    peak_q: np.ndarray,
    relative_effect: np.ndarray,
    config: LocalGamePayoffConfig,
) -> tuple[int, ...]:
    runner_index = next(
        index for index, option in enumerate(options) if bool(option["is_runner"])
    )
    eligible = [
        index
        for index, option in enumerate(options)
        if index != runner_index
        and float(peak_q[index]) >= config.minimum_peak_q
        and float(relative_effect[index])
        >= config.minimum_relative_response_effect
    ]
    eligible.sort(
        key=lambda index: (
            -float(relative_effect[index]),
            -float(peak_q[index]),
            str(options[index]["option_id"]),
        )
    )
    return (runner_index, *eligible[: config.local_non_runner_limit])


def _eligible_cross_cost_indices(
    options: Sequence[Mapping[str, object]],
    peak_q: np.ndarray,
    relative_effect: np.ndarray,
    config: LocalGamePayoffConfig,
) -> tuple[int, ...]:
    """Keep the runner and substantively non-trivial derived candidates."""

    runner_index = next(
        index for index, option in enumerate(options) if bool(option["is_runner"])
    )
    return (
        runner_index,
        *(
            index
            for index in range(len(options))
            if index != runner_index
            and float(peak_q[index]) >= config.minimum_peak_q
            and float(relative_effect[index])
            >= config.minimum_relative_response_effect
        ),
    )


def _cross_cost(
    options: Sequence[Mapping[str, object]],
    q_matrix: np.ndarray,
    local_indices: Sequence[int],
    option_minimum_indices: Sequence[int],
) -> tuple[str | None, dict[str, object]]:
    runner_index = next(
        index for index, option in enumerate(options) if bool(option["is_runner"])
    )
    runner_response = int(option_minimum_indices[runner_index])
    candidates = []
    for option_index in local_indices:
        if option_index == runner_index:
            continue
        option_response = int(option_minimum_indices[option_index])
        runner_cost = float(
            q_matrix[runner_index, option_response]
            - q_matrix[runner_index, runner_response]
        )
        derived_cost = float(
            q_matrix[option_index, runner_response]
            - q_matrix[option_index, option_response]
        )
        runner_relative = runner_cost / max(
            float(q_matrix[runner_index, option_response]), 1e-9
        )
        derived_relative = derived_cost / max(
            float(q_matrix[option_index, runner_response]), 1e-9
        )
        strength = min(max(0.0, runner_relative), max(0.0, derived_relative))
        candidates.append(
            (
                strength,
                min(max(0.0, runner_cost), max(0.0, derived_cost)),
                option_index,
                {
                    "runner": runner_cost,
                    "derived": derived_cost,
                    "runner_relative": float(runner_relative),
                    "derived_relative": float(derived_relative),
                    "strength": float(strength),
                    "runner_best_response_index": runner_response,
                    "derived_best_response_index": option_response,
                },
            )
        )
    if not candidates:
        return None, {
            "runner": 0.0,
            "derived": 0.0,
            "runner_relative": 0.0,
            "derived_relative": 0.0,
            "strength": 0.0,
            "runner_best_response_index": runner_response,
            "derived_best_response_index": None,
        }
    selected = max(candidates, key=lambda row: (row[0], row[1], -row[2]))
    return str(options[selected[2]]["option_id"]), selected[3]


def _screen_pass_value(
    game: Mapping[str, object],
    defender_id: str,
    defender_path_txy: Sequence[TimedPoint],
    option: Mapping[str, object],
    config: LocalGamePayoffConfig,
    arrival_config: ArrivalModelConfig,
    runtime: _SceneRuntime,
) -> float:
    """Cheap delivery × goal screen; influence A is deferred to finalists."""

    receiver_id = str(option["option_id"])
    option_path = tuple(
        tuple(map(float, row)) for row in option["path_txy"]  # type: ignore[index]
    )
    horizon = float(game["horizon_seconds"])
    goal_xy = (float(game["goal_xy"][0]), float(game["goal_xy"][1]))
    releases = _sample_times(
        0.0,
        horizon - config.minimum_event_duration_seconds,
        config.release_step_seconds,
        include_stop=True,
    )
    best = 0.0
    for release_time in releases:
        release_frame, velocities = _frame_state(
            game,
            release_time,
            defender_id,
            defender_path_txy,
            attach_ball_to_carrier=True,
            runtime=runtime,
        )
        if release_frame.ball is None:
            continue
        ball_xy = (float(release_frame.ball.x), float(release_frame.ball.y))
        if is_offside_position(
            release_frame,
            receiver_id,
            str(game["attacking_team_id"]),
            ball_xy,
            int(game["attacking_direction"]),
        ):
            continue
        receiver_now = release_frame.players[receiver_id]

        # Through-ball templates share the screen so per-option minima keep
        # the paths that matter against the space continuation as well.
        release_velocity = velocities[receiver_id]
        if release_velocity.speed >= config.through_ball_minimum_run_speed_mps:
            unit_x = release_velocity.vx / release_velocity.speed
            unit_y = release_velocity.vy / release_velocity.speed
            for lead in config.through_ball_lead_distances_m:
                base_x = float(receiver_now.x) + float(lead) * unit_x
                base_y = float(receiver_now.y) + float(lead) * unit_y
                goal_dx = goal_xy[0] - base_x
                goal_dy = goal_xy[1] - base_y
                goal_norm = math.hypot(goal_dx, goal_dy)
                if goal_norm > 1e-9:
                    base_x += (
                        config.through_ball_goal_side_offset_m * goal_dx / goal_norm
                    )
                    base_y += (
                        config.through_ball_goal_side_offset_m * goal_dy / goal_norm
                    )
                through_estimate = point_reception_estimate(
                    release_frame,
                    velocities,
                    receiver_id,
                    str(game["attacking_team_id"]),
                    ball_xy,
                    (base_x, base_y),
                    arrival_config,
                    passer_id=str(game["carrier_id"]),
                )
                through_event = release_time + max(
                    config.minimum_event_duration_seconds,
                    float(through_estimate.ball_arrival_time_s),
                    float(through_estimate.receiver_arrival_time_s),
                )
                if through_event > horizon + 1e-8:
                    continue
                best = max(
                    best,
                    float(
                        np.clip(through_estimate.receive_probability, 0.0, 1.0)
                    )
                    * float(
                        score_at_points(
                            [(base_x, base_y)],
                            int(game["attacking_direction"]),
                            epv_grid_path=config.goal_danger_grid_path,
                        )[0]
                    ),
                )

        target_time = min(
            horizon,
            release_time
            + max(
                config.minimum_event_duration_seconds,
                math.hypot(receiver_now.x - ball_xy[0], receiver_now.y - ball_xy[1])
                / max(arrival_config.pass_speed_mps, 1e-9),
            ),
        )
        estimate = None
        for _ in range(6):
            target_xy = interpolate_timed_point(option_path, target_time)
            estimate = point_reception_estimate(
                release_frame,
                velocities,
                receiver_id,
                str(game["attacking_team_id"]),
                ball_xy,
                target_xy,
                arrival_config,
                passer_id=str(game["carrier_id"]),
            )
            proposed = release_time + max(
                config.minimum_event_duration_seconds,
                float(estimate.ball_arrival_time_s),
                float(estimate.receiver_arrival_time_s),
            )
            if proposed > horizon + 1e-8:
                estimate = None
                break
            updated = 0.5 * target_time + 0.5 * proposed
            if abs(updated - target_time) <= 0.01:
                target_time = float(proposed)
                break
            target_time = float(updated)
        if estimate is None or target_time > horizon + 1e-8:
            continue
        target_xy = interpolate_timed_point(option_path, target_time)
        estimate = point_reception_estimate(
            release_frame,
            velocities,
            receiver_id,
            str(game["attacking_team_id"]),
            ball_xy,
            target_xy,
            arrival_config,
            passer_id=str(game["carrier_id"]),
        )
        event_time = release_time + max(
            config.minimum_event_duration_seconds,
            float(estimate.ball_arrival_time_s),
            float(estimate.receiver_arrival_time_s),
        )
        if event_time > horizon + 1e-8:
            continue
        target_xy = interpolate_timed_point(option_path, event_time)
        delivery = float(np.clip(estimate.receive_probability, 0.0, 1.0))
        goal = float(
            score_at_points(
                [(float(target_xy[0]), float(target_xy[1]))],
                int(game["attacking_direction"]),
                epv_grid_path=config.goal_danger_grid_path,
            )[0]
        )
        best = max(best, delivery * goal)
    return float(best)


def _screen_carry_value(
    game: Mapping[str, object],
    defender_id: str,
    defender_path_txy: Sequence[TimedPoint],
    option: Mapping[str, object],
    config: LocalGamePayoffConfig,
    arrival_config: ArrivalModelConfig,
    goalkeeper_ids: Sequence[str],
    runtime: _SceneRuntime,
) -> float:
    """Cheap path-retention × goal screen for the observed carrier action."""

    carrier_path = tuple(
        tuple(map(float, row)) for row in option["path_txy"]  # type: ignore[index]
    )
    horizon = float(game["horizon_seconds"])
    best = 0.0
    for event_time in _sample_times(
        config.minimum_event_duration_seconds,
        horizon,
        config.release_step_seconds,
        include_stop=True,
    ):
        samples = [
            _carry_retention_at_time(
                game,
                defender_id,
                defender_path_txy,
                carrier_path,
                time_s,
                goalkeeper_ids,
                config,
                arrival_config,
                runtime,
            )[0]
            for time_s in _sample_times(
                config.minimum_event_duration_seconds,
                event_time,
                config.carry_path_step_seconds,
                include_stop=True,
            )
        ]
        if not samples:
            continue
        event_xy = interpolate_timed_point(carrier_path, event_time)
        goal = float(
            score_at_points(
                [(float(event_xy[0]), float(event_xy[1]))],
                int(game["attacking_direction"]),
                epv_grid_path=config.goal_danger_grid_path,
            )[0]
        )
        retention = min(samples)
        best = max(best, retention * goal)
        # Screen parity for the carry-into-space extension, so per-option
        # minima keep the paths that suppress the forward carry space.
        event_velocity = _path_velocity(carrier_path, event_time)
        carrier_speed = math.hypot(*event_velocity)
        if (
            config.carry_lead_distances_m
            and carrier_speed >= config.through_ball_minimum_run_speed_mps
        ):
            unit_x = event_velocity[0] / carrier_speed
            unit_y = event_velocity[1] / carrier_speed
            screen_leads = tuple(
                config.carry_lead_distances_m
            ) + _confrontation_leads(
                game,
                runtime,
                event_time,
                (float(event_xy[0]), float(event_xy[1])),
                (unit_x, unit_y),
                config,
            )
            for lead in screen_leads:
                arrival_time = event_time + float(lead) / carrier_speed
                if arrival_time > horizon + 1e-9:
                    continue
                z_lead = (
                    float(event_xy[0]) + float(lead) * unit_x,
                    float(event_xy[1]) + float(lead) * unit_y,
                )
                if config.carry_lead_pricing == "extension_race":
                    contest = _carry_extension_race(
                        game,
                        runtime,
                        defender_id,
                        defender_path_txy,
                        float(event_time),
                        (float(event_xy[0]), float(event_xy[1])),
                        (unit_x, unit_y),
                        float(lead),
                        float(arrival_time),
                        goalkeeper_ids,
                        config,
                        arrival_config,
                    )
                else:
                    frame, velocities = _frame_state(
                        game,
                        arrival_time,
                        defender_id,
                        defender_path_txy,
                        attach_ball_to_carrier=True,
                        runtime=runtime,
                    )
                    contest = 1.0
                    for player_id, player in frame.players.items():
                        if (
                            player.team_id == str(game["attacking_team_id"])
                            or player_id in goalkeeper_ids
                        ):
                            continue
                        arrival = float(
                            player_time_to_point(
                                player,
                                velocities[player_id],
                                z_lead,
                                arrival_config,
                            )
                        )
                        contest *= _logistic(
                            (arrival - config.carry_pressure_action_time_seconds)
                            / config.carry_pressure_sigma_seconds
                        )
                goal_lead = float(
                    score_at_points(
                        [z_lead],
                        int(game["attacking_direction"]),
                        epv_grid_path=config.goal_danger_grid_path,
                    )[0]
                )
                best = max(best, retention * goal_lead * contest)
    return float(best)


def _screen_search_responses(
    game: Mapping[str, object],
    defender: Mapping[str, object],
    options: Sequence[Mapping[str, object]],
    responses: Sequence[dict[str, object]],
    config: LocalGamePayoffConfig,
    arrival_config: ArrivalModelConfig,
    goalkeeper_ids: Sequence[str],
    runtime: _SceneRuntime,
) -> list[dict[str, object]]:
    """Keep option minima and pairwise compromises before costly influence A."""

    if config.response_search_mode != "target_agnostic":
        return list(responses)
    # Nothing to screen, and the regret normalisation below reduces over the
    # response axis, so an empty one raises rather than returning nothing.
    # Seen on run_onset_v0_5 task 435 (DFL-MAT-J03WQQ), both delivery arms.
    if not responses or not options:
        return list(responses)
    matrix = np.zeros((len(options), len(responses)), dtype=float)
    for response_index, response in enumerate(responses):
        path = tuple(
            tuple(map(float, row)) for row in response["path_txy"]  # type: ignore[index]
        )
        screen_values = {}
        for option_index, option in enumerate(options):
            if bool(option["is_carrier"]):
                value = _screen_carry_value(
                    game,
                    str(defender["defender_id"]),
                    path,
                    option,
                    config,
                    arrival_config,
                    goalkeeper_ids,
                    runtime,
                )
            else:
                value = _screen_pass_value(
                    game,
                    str(defender["defender_id"]),
                    path,
                    option,
                    config,
                    arrival_config,
                    runtime,
                )
            matrix[option_index, response_index] = value
            screen_values[str(option["option_id"])] = float(value)
        response["screen_values"] = screen_values

    chosen: set[int] = set()
    keep = min(config.response_screen_keep_per_option, len(responses))
    for option_index in range(len(options)):
        chosen.update(
            int(index)
            for index in np.argsort(matrix[option_index], kind="stable")[:keep]
        )

    minimum = np.min(matrix, axis=1, keepdims=True)
    maximum = np.max(matrix, axis=1, keepdims=True)
    regret = np.divide(
        matrix - minimum,
        np.maximum(maximum - minimum, 1e-9),
        out=np.zeros_like(matrix),
        where=(maximum - minimum) > 1e-9,
    )
    runner_index = next(
        index for index, option in enumerate(options) if bool(option["is_runner"])
    )
    pairwise_keep = min(2, len(responses))
    for option_index in range(len(options)):
        if option_index == runner_index:
            continue
        pairwise = np.maximum(regret[runner_index], regret[option_index])
        chosen.update(
            int(index)
            for index in np.argsort(pairwise, kind="stable")[:pairwise_keep]
        )
    all_option_worst = np.max(regret, axis=0)
    chosen.update(
        int(index)
        for index in np.argsort(all_option_worst, kind="stable")[
            : min(config.response_screen_keep_minimax, len(responses))
        ]
    )
    efforts = np.asarray(
        [float(response["effort_m2ps3"]) for response in responses], dtype=float
    )
    chosen.update(
        int(index)
        for index in np.argsort(efforts, kind="stable")[
            : min(config.response_screen_keep_low_effort, len(responses))
        ]
    )
    # Uncorrelated tail insurance against the screen's own proxy error. The
    # seed is fixed and the pool is sorted, so a rerun keeps the same set.
    if config.response_screen_keep_random > 0:
        rejected = sorted(set(range(len(responses))) - chosen)
        if rejected:
            generator = np.random.default_rng(config.response_screen_random_seed)
            chosen.update(
                int(rejected[index])
                for index in generator.choice(
                    len(rejected),
                    size=min(config.response_screen_keep_random, len(rejected)),
                    replace=False,
                )
            )
    selected = [response for index, response in enumerate(responses) if index in chosen]
    for rank, response in enumerate(selected, 1):
        response["screen_rank"] = rank
        response["raw_search_count"] = len(responses)
        response["screened_search_count"] = len(selected)
    return selected


def _build_defender_payoff_game(
    game: Mapping[str, object],
    defender: Mapping[str, object],
    config: LocalGamePayoffConfig,
    arrival_config: ArrivalModelConfig,
    influence_config: GoalWeightedInfluenceConfig,
    influence_xgrid: np.ndarray,
    influence_ygrid: np.ndarray,
    goalkeeper_ids: Sequence[str],
    runtime: _SceneRuntime,
) -> dict[str, object] | None:
    # Provisional assignment-rule beneficiary (2026-09-06 decision): compute
    # it up front so the pick is guaranteed to survive display and pricing.
    rule_derived_id: str | None = None
    rule_derived_branch: str | None = None
    if config.derived_option_source == "assignment_rule_v1":
        assignment_state = _assignment_onset_state(game)
        rule_derived_id, rule_derived_branch = _assignment_rule_r1(
            assignment_state,
            _assignment_attacking_team(game),
            str(game["runner_id"]),
            str(game["carrier_id"]),
            str(defender["defender_id"]),
        )
    ensure_ids = (rule_derived_id,) if rule_derived_id else ()

    options = _option_paths(game, defender, ensure_ids, config.vacated_union_count)
    reference_responses, raw_search_responses = _response_candidates(
        game, defender, config, ensure_ids
    )
    if config.frozen_response_catalogue_path:
        # Replay a previous run's trajectories so a pricing change can be
        # measured without the value-guided search moving underneath it.
        key = (str(game["onset_frame_id"]), str(defender["defender_id"]))
        catalogue = _frozen_response_catalogue(
            config.frozen_response_catalogue_path
        )
        if key not in catalogue:
            raise ValueError(
                f"frozen catalogue has no responses for scene/defender {key}"
            )
        search_responses = [dict(response) for response in catalogue[key]]
    else:
        search_responses = _screen_search_responses(
            game,
            defender,
            options,
            raw_search_responses,
            config,
            arrival_config,
            goalkeeper_ids,
            runtime,
        )
    responses = [*reference_responses, *search_responses]
    for response in responses:
        response_path = tuple(
            tuple(map(float, row)) for row in response["path_txy"]  # type: ignore[index]
        )
        response["cells"] = {
            str(option["option_id"]): _evaluate_cell(
                game,
                str(defender["defender_id"]),
                response_path,
                option,
                config,
                arrival_config,
                influence_config,
                influence_xgrid,
                influence_ygrid,
                goalkeeper_ids,
                runtime,
            )
            for option in options
        }

    # No playable response means this defender has no game: every reduction
    # below is over the response axis and an empty one raises rather than
    # returning nothing. Seen on run_onset_v0_5 task 435 (DFL-MAT-J03WQQ), in
    # both delivery arms and both value stacks. Dropping the defender is the
    # honest result -- there is nothing to solve, not a zero-valued game.
    if not search_responses or not options:
        return None
    q_matrix = np.asarray(
        [
            [
                float(response["cells"][str(option["option_id"])]["q"])  # type: ignore[index]
                for response in search_responses
            ]
            for option in options
        ],
        dtype=float,
    )
    minimum_q, peak_q, relative_effect = _response_effects(q_matrix)
    efforts = [float(response["effort_m2ps3"]) for response in search_responses]

    goal_xy = (float(game["goal_xy"][0]), float(game["goal_xy"][1]))
    horizon = float(game["horizon_seconds"])
    response_end_xy = [
        interpolate_timed_point(
            tuple(tuple(map(float, row)) for row in response["path_txy"]),
            horizon,
        )
        for response in search_responses
    ]
    option_tail_target = []
    for option in options:
        option_path = tuple(tuple(map(float, row)) for row in option["path_txy"])
        option_end = interpolate_timed_point(option_path, horizon)
        option_tail_target.append(
            moving_goal_side_target(option_end, goal_xy, 1.5)
        )

    def _tail_cost(response_index: int, option_indices: Sequence[int]) -> float:
        end_x, end_y = response_end_xy[response_index]
        return float(
            np.mean(
                [
                    math.hypot(
                        end_x - option_tail_target[option_index][0],
                        end_y - option_tail_target[option_index][1],
                    )
                    for option_index in option_indices
                ]
            )
        )

    # Value layer: exact per-option minima drive cross-cost and the minimax
    # bound.  Display layer: within the anchor tie tolerance, a representative
    # with the most sensible tail is shown instead.
    exact_minimum_indices = tuple(
        min(
            range(len(search_responses)),
            key=lambda response_index: (
                float(q_matrix[option_index, response_index]),
                efforts[response_index],
                str(search_responses[response_index]["response_id"]),
            ),
        )
        for option_index in range(len(options))
    )
    display_minimum_indices = tuple(
        _select_anchor_index(
            [float(value) for value in q_matrix[option_index]],
            [
                _tail_cost(response_index, (option_index,))
                for response_index in range(len(search_responses))
            ],
            efforts,
            config.anchor_tie_tolerance,
        )
        for option_index in range(len(options))
    )
    if config.derived_option_source == "coupled_r9":
        # Same two inputs the validated scorer used: the direct-best response
        # for the value curves, the baseline response for coverage geometry.
        runner_option_index_early = next(
            index for index, option in enumerate(options) if bool(option["is_runner"])
        )
        rule_derived_id, r9_detail = select_r9_beneficiary(
            game,
            str(defender["defender_id"]),
            responses,
            str(search_responses[
                display_minimum_indices[runner_option_index_early]
            ]["response_id"]),
            str(game["runner_id"]),
        )
        rule_derived_branch = "coupled_r9" if rule_derived_id else None

    pinned_derived_id = dict(
        game.get("confirmed_derived_ids", {})  # type: ignore[arg-type]
    ).get(str(defender["defender_id"]))
    human_label_option_id = (
        dict(game.get("human_labels", {}))  # type: ignore[arg-type]
        .get("derived_by_defender", {})
        .get(str(defender["defender_id"]))
    )
    derived_pinned = False
    derived_selection_source = None
    derived_option_id_payoff: str | None = None
    if config.local_game_selection_mode == "pairwise_cross_cost":
        eligible_indices = _eligible_cross_cost_indices(
            options,
            peak_q,
            relative_effect,
            config,
        )
        runner_index = next(
            index
            for index, option in enumerate(options)
            if bool(option["is_runner"])
        )
        pinned_index = next(
            (
                index
                for index, option in enumerate(options)
                if pinned_derived_id is not None
                and str(option["option_id"]) == str(pinned_derived_id)
            ),
            None,
        )
        rule_index = next(
            (
                index
                for index, option in enumerate(options)
                if rule_derived_id is not None
                and str(option["option_id"]) == str(rule_derived_id)
            ),
            None,
        )
        if pinned_index is None and rule_index is not None:
            # The provisional assignment rule chooses the beneficiary; the
            # payoff machinery still prices the (runner, beneficiary) pair
            # and its cross-cost, and the payoff-selected alternative is
            # kept alongside for comparison. Human pins take precedence.
            derived_selection_source = (
                R9_LABEL if config.derived_option_source == "coupled_r9"
                else ASSIGNMENT_RULE_LABEL
            )
            derived_option_id_payoff, _ = _cross_cost(
                options,
                q_matrix,
                eligible_indices,
                exact_minimum_indices,
            )
            derived_option_id, cross_cost = _cross_cost(
                options,
                q_matrix,
                (runner_index, rule_index),
                exact_minimum_indices,
            )
            derived_option_id = str(options[rule_index]["option_id"])
        elif pinned_index is not None:
            # Human review pinned the derived branch for this defender; the
            # cross-cost is still reported as a diagnostic of that pair, but
            # the gates and ranking no longer choose the pair.
            derived_pinned = True
            derived_option_id, cross_cost = _cross_cost(
                options,
                q_matrix,
                (runner_index, pinned_index),
                exact_minimum_indices,
            )
            derived_option_id = str(options[pinned_index]["option_id"])
        else:
            derived_option_id, cross_cost = _cross_cost(
                options,
                q_matrix,
                eligible_indices,
                exact_minimum_indices,
            )
        derived_index = next(
            (
                index
                for index, option in enumerate(options)
                if str(option["option_id"]) == derived_option_id
            ),
            None,
        )
        local_indices = (
            (runner_index, derived_index)
            if derived_index is not None
            else (runner_index,)
        )
    else:
        local_indices = _local_option_indices(
            options, peak_q, relative_effect, config
        )
        derived_option_id, cross_cost = _cross_cost(
            options,
            q_matrix,
            local_indices,
            exact_minimum_indices,
        )
    local_matrix = q_matrix[np.asarray(local_indices, dtype=int)]
    worst_per_response = np.max(local_matrix, axis=0)
    exact_minimax_worst_q = float(np.min(worst_per_response))
    minimax_index = _select_anchor_index(
        [float(value) for value in worst_per_response],
        [
            _tail_cost(response_index, local_indices)
            for response_index in range(len(search_responses))
        ],
        efforts,
        config.anchor_tie_tolerance,
    )
    minimax_response = search_responses[minimax_index]
    cross_cost["runner_best_response_id"] = search_responses[
        int(cross_cost["runner_best_response_index"])
    ]["response_id"]
    if cross_cost["derived_best_response_index"] is not None:
        cross_cost["derived_best_response_id"] = search_responses[
            int(cross_cost["derived_best_response_index"])
        ]["response_id"]
    else:
        cross_cost["derived_best_response_id"] = None

    local_ids = {str(options[index]["option_id"]) for index in local_indices}
    for option_index, option in enumerate(options):
        option.update(
            {
                "in_local_set": str(option["option_id"]) in local_ids,
                "minimum_q": float(minimum_q[option_index]),
                "peak_q": float(peak_q[option_index]),
                "response_effect": float(
                    peak_q[option_index] - minimum_q[option_index]
                ),
                "relative_effect": float(relative_effect[option_index]),
                "best_cover_response_id": str(
                    search_responses[display_minimum_indices[option_index]][
                        "response_id"
                    ]
                ),
            }
        )

    best_option_ids_by_response: dict[str, list[str]] = {}
    for option in options:
        best_option_ids_by_response.setdefault(
            str(option["best_cover_response_id"]), []
        ).append(str(option["option_id"]))
    # Display anchors: tail-sensible representatives of the exact minima.
    runner_option_index = next(
        index for index, option in enumerate(options) if bool(option["is_runner"])
    )
    runner_best_id = str(
        search_responses[display_minimum_indices[runner_option_index]][
            "response_id"
        ]
    )
    derived_option_index = next(
        (
            index
            for index, option in enumerate(options)
            if derived_option_id is not None
            and str(option["option_id"]) == str(derived_option_id)
        ),
        None,
    )
    derived_best_id = (
        str(
            search_responses[display_minimum_indices[derived_option_index]][
                "response_id"
            ]
        )
        if derived_option_index is not None
        else None
    )
    for response in responses:
        response_id = str(response["response_id"])
        response["best_for_option_ids"] = best_option_ids_by_response.get(
            response_id, []
        )
        response["is_direct_best"] = response_id == runner_best_id
        response["is_derived_best"] = (
            derived_best_id is not None and response_id == derived_best_id
        )

    local_option_ids = [str(options[index]["option_id"]) for index in local_indices]
    for response in responses:
        local_values = {
            option_id: float(response["cells"][option_id]["q"])  # type: ignore[index]
            for option_id in local_option_ids
        }
        worst_id = max(
            local_values,
            key=lambda option_id: (local_values[option_id], option_id),
        )
        response.update(
            {
                "worst_local_option_id": worst_id,
                "worst_local_q": float(local_values[worst_id]),
                "local_total_q": float(sum(local_values.values())),
                "is_minimax": response is minimax_response,
            }
        )

    def response_display_rank(response: Mapping[str, object]) -> tuple[object, ...]:
        response_id = str(response["response_id"])
        if response_id == "actual":
            priority = 0
        elif str(response["kind"]) == "target_conditioned_baseline":
            priority = 1
        elif bool(response.get("is_direct_best")):
            priority = 2
        elif bool(response.get("is_derived_best")):
            priority = 3
        elif bool(response.get("is_minimax")):
            priority = 4
        else:
            priority = 5
        return (priority, float(response["effort_m2ps3"]), response_id)

    responses.sort(key=response_display_rank)

    actual = reference_responses[0]
    actual_worst_q = float(actual["worst_local_q"])
    minimax_worst_q = float(minimax_response["worst_local_q"])
    minimax_attack_id = str(minimax_response["worst_local_option_id"])
    actual_attack_id = str(actual["worst_local_option_id"])
    return {
        "defender_id": str(defender["defender_id"]),
        "defender_name": str(defender["defender_name"]),
        "current_distance_to_runner_m": float(
            defender["current_distance_to_runner_m"]
        ),
        "actual_response_id": "actual",
        "minimax_response_id": str(minimax_response["response_id"]),
        "minimax_response_label": str(minimax_response["label"]),
        "direct_option_id": str(game["runner_id"]),
        "direct_best_response_id": runner_best_id,
        "derived_option_id": derived_option_id,
        "derived_pinned": derived_pinned,
        # Evaluation only: the reviewer's label for this defender and whether
        # the model's own selection reproduced it. Never read by any model
        # stage — the label is not an input.
        "human_label_option_id": human_label_option_id,
        "matches_human_label": (
            None
            if human_label_option_id is None
            else str(derived_option_id) == str(human_label_option_id)
        ),
        "derived_selection_source": derived_selection_source,
        "derived_rule_branch": rule_derived_branch
        if derived_selection_source is not None
        else None,
        "derived_option_id_payoff": derived_option_id_payoff,
        "derived_best_response_id": derived_best_id,
        "local_option_ids": local_option_ids,
        "actual_worst_q": actual_worst_q,
        "minimax_worst_q": minimax_worst_q,
        "exact_minimax_worst_q": exact_minimax_worst_q,
        "actual_minus_minimax_q": float(actual_worst_q - minimax_worst_q),
        "actual_reselected_option_id": actual_attack_id,
        "minimax_reselected_option_id": minimax_attack_id,
        "cross_cost": cross_cost,
        "options": options,
        "responses": responses,
        "raw_search_response_count": len(raw_search_responses),
        "search_response_count": len(search_responses),
        "actual_is_optimizer_candidate": False,
    }


def build_local_game_payoff_audit(
    scene: Mapping[str, object],
    config: LocalGamePayoffConfig = LocalGamePayoffConfig(),
    structural_config: LocalGameStructureConfig = LocalGameStructureConfig(),
) -> dict[str, object]:
    """Build one defender-conditioned payoff audit from a confirmed scene."""

    config.validate()
    structural_config.validate()
    structural = build_structural_local_game(scene, structural_config)
    runtime = _scene_runtime(structural)
    arrival_config = ArrivalModelConfig()
    influence_config = GoalWeightedInfluenceConfig(
        grid_resolution_m=config.influence_grid_resolution_m
    )
    influence_xgrid, influence_ygrid = influence_pitch_grid(influence_config)
    goalkeeper_ids = _goalkeeper_ids(structural)
    defender_games = [
        built
        for defender in structural["candidate_defenders"]  # type: ignore[index]
        if (built := _build_defender_payoff_game(
            structural,
            defender,
            config,
            arrival_config,
            influence_config,
            influence_xgrid,
            influence_ygrid,
            goalkeeper_ids,
            runtime,
        )) is not None
    ]
    target_agnostic = config.response_search_mode == "target_agnostic"
    return {
        "schema_version": (
            "target-agnostic-local-game-audit-v0.2"
            if target_agnostic
            else "local-game-payoff-audit-v0.1"
        ),
        "status": (
            "retrospective target-agnostic continuation audit"
            if target_agnostic
            else "retrospective provisional payoff audit"
        ),
        "match_id": str(structural["match_id"]),
        "match_label": str(structural["match_label"]),
        "onset_frame_id": int(structural["onset_frame_id"]),
        "runner_id": str(structural["runner_id"]),
        "runner_name": str(structural["runner_name"]),
        "carrier_id": str(structural["carrier_id"]),
        "carrier_name": str(structural["carrier_name"]),
        "attacking_team_id": str(structural["attacking_team_id"]),
        "attacking_direction": int(structural["attacking_direction"]),
        "horizon_seconds": float(structural["horizon_seconds"]),
        "goal_xy": list(structural["goal_xy"]),
        "background_frames": list(structural["background_frames"]),
        "candidate_defenders": defender_games,
        "human_review": dict(structural.get("human_review", {})),
        "config": {
            "payoff": asdict(config),
            "structure": asdict(structural_config),
            "arrival": asdict(arrival_config),
            "influence": asdict(influence_config),
        },
        "value_contract": (
            "ssac_completion_or_carry_survival × ssac_positional_threat"
            if config.value_stack == "ssac"
            else "delivery_or_path_min_retention × geometric_goal_danger × "
            "post_success_goal_side_accessibility"
        ),
        "notes": [
            "Observed attacker and background futures are retrospective audit inputs, not online forecasts.",
            "Only the selected defender path changes; the other 21 observed player paths remain fixed.",
            "Actual defense is a reference and never enters the minimax search set.",
            "Pass delivery and carry retention share a 0-1 interface but are not calibrated probabilities.",
            "Carrier carry reattaches possession to the observed carrier path and is explicitly provisional.",
            "G is a geometric terminal proxy; A is residual influence accessibility, not shot probability.",
            (
                "The local pair contains the runner and the strongest two-sided cross-cost continuation, not the global team maximum."
                if config.local_game_selection_mode == "pairwise_cross_cost"
                else "The local option set is defender-dependent and selected by response sensitivity, not global team maximum."
            ),
            (
                "Target-agnostic search paths are generated without attacker targets; direct, derived, and minimax labels are assigned only after value evaluation."
                if target_agnostic
                else "Target-conditioned responses are heuristic development baselines."
            ),
            (
                "In pairwise mode the derived continuation must first pass minimum peak-Q and response-effect gates, then maximize the two-sided cross-cost against the runner-direct minimum path."
                if config.local_game_selection_mode == "pairwise_cross_cost"
                else "Response-effect mode constructs a sensitivity-filtered local option set."
            ),
            "Displayed anchors are representatives of near-tied minima: within the anchor tie tolerance, the path with the best terminal goal-side positioning is shown, because tails after the last priced pass window carry no value.",
        ],
    }


def build_local_game_payoff_audits(
    scenes: Sequence[Mapping[str, object]],
    config: LocalGamePayoffConfig = LocalGamePayoffConfig(),
    structural_config: LocalGameStructureConfig = LocalGameStructureConfig(),
) -> list[dict[str, object]]:
    """Build payoff audits in the confirmed human-review order."""

    return [
        build_local_game_payoff_audit(scene, config, structural_config)
        for scene in scenes
    ]
