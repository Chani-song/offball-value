from __future__ import annotations

import unittest

import pandas as pd

from offball_value.bundesliga import (
    BundesligaFrame,
    BundesligaMatchMeta,
    BundesligaObjectState,
    BundesligaPlayerMeta,
    BundesligaTeamMeta,
)
from offball_value.run_onset import (
    KinematicRunOnset,
    RunOnsetCandidate,
    RunOnsetConfig,
    _context_candidate,
)
from offball_value.shot_context_onset import (
    SettledPossessionConfig,
    assess_settled_shot_context_onset,
)


# A ball held at (0, 0) until frame 15, then kicked along x at 1 m per frame
# to a receiver standing at (20, 0), who has it from frame 34 on.
RELEASE, ARRIVAL = 15, 34


def _ball_x(frame_id: int, flight: bool) -> float:
    if not flight or frame_id <= RELEASE:
        return 0.3
    if frame_id >= ARRIVAL:
        return 19.7
    return float(frame_id - RELEASE)


def _frames(holder: str, flight: bool = True) -> dict[int, BundesligaFrame]:
    """`holder` stands on the ball at (0, 0); the runner otherwise waits at (10, 10)."""
    spots = {"passer": (-10.0, -10.0), "runner": (10.0, 10.0), "defender": (-20.0, 20.0)}
    spots[holder] = (0.0, 0.0)
    teams = {"passer": "attack", "runner": "attack", "receiver": "attack", "defender": "defence"}
    frames = {}
    for frame_id in range(0, 61):
        players = {
            pid: BundesligaObjectState(pid, teams[pid], *spots[pid])
            for pid in ("passer", "runner", "defender")
        }
        players["receiver"] = BundesligaObjectState("receiver", "attack", 20.0, 0.0)
        frames[frame_id] = BundesligaFrame(
            match_id="match",
            frame_id=frame_id,
            period=1,
            game_section="firstHalf",
            timestamp=None,
            players=players,
            ball=BundesligaObjectState("ball", "BALL", _ball_x(frame_id, flight), 0.0, z=0.0),
        )
    return frames


def _meta() -> BundesligaMatchMeta:
    teams = {"attack": "home", "defence": "away"}
    players = {
        pid: BundesligaPlayerMeta(pid, team, pid, position="MF")
        for pid, team in (("passer", "attack"), ("runner", "attack"),
                          ("receiver", "attack"), ("defender", "defence"))
    }
    return BundesligaMatchMeta(
        match_id="match",
        home_team_id="attack",
        away_team_id="defence",
        home_team_name="Attack",
        away_team_name="Defence",
        teams={tid: BundesligaTeamMeta(tid, tid, role) for tid, role in teams.items()},
        players=players,
    )


def _onset(frame_id: int = 20) -> KinematicRunOnset:
    return KinematicRunOnset(
        frame_id=frame_id,
        labels=("acceleration",),
        x=10.0,
        y=10.0,
        onset_speed_mps=2.0,
        pre_mean_speed_mps=1.0,
        post_mean_speed_mps=4.0,
        speed_gain_mps=3.0,
        direction_change_degrees=0.0,
        post_displacement_m=4.0,
        peak_acceleration_mps2=3.0,
        peak_deceleration_mps2=0.0,
        confidence_score=1.0,
    )


def _config(in_flight: bool) -> RunOnsetConfig:
    return RunOnsetConfig(required_player_count=None, in_flight_control=in_flight)


class InFlightOnsetGateTest(unittest.TestCase):
    def test_ball_in_flight_is_rejected_without_the_option(self) -> None:
        candidate = _context_candidate(_onset(), "runner", _frames("passer"), _meta(), _config(False))

        self.assertIsNone(candidate)

    def test_ball_in_flight_from_a_team_mate_keeps_the_passer_as_carrier(self) -> None:
        candidate = _context_candidate(_onset(), "runner", _frames("passer"), _meta(), _config(True))

        self.assertIsNotNone(candidate)
        self.assertEqual(candidate.ball_carrier_id, "passer")
        self.assertTrue(candidate.ball_in_flight)
        self.assertEqual(candidate.stable_control_fraction, 1.0)
        self.assertEqual(candidate.future_known_fraction, 1.0)
        self.assertEqual(candidate.future_same_team_fraction, 1.0)

    def test_pass_and_move_takes_the_next_team_mate_as_carrier(self) -> None:
        candidate = _context_candidate(_onset(), "runner", _frames("runner"), _meta(), _config(True))

        self.assertIsNotNone(candidate)
        self.assertEqual(candidate.ball_carrier_id, "receiver")
        self.assertTrue(candidate.ball_in_flight)

    def test_ball_last_touched_by_an_opponent_is_still_rejected(self) -> None:
        candidate = _context_candidate(_onset(), "runner", _frames("defender"), _meta(), _config(True))

        self.assertIsNone(candidate)

    def test_controlled_ball_gives_the_same_candidate_either_way(self) -> None:
        frames = _frames("passer", flight=False)

        off = _context_candidate(_onset(), "runner", frames, _meta(), _config(False))
        on = _context_candidate(_onset(), "runner", frames, _meta(), _config(True))

        self.assertIsNotNone(off)
        self.assertEqual(off, on)
        self.assertFalse(on.ball_in_flight)
        self.assertEqual(on.ball_carrier_id, "passer")


def _settled_frame(frame_id: int, owner: str | None) -> BundesligaFrame:
    """Ball at x = 10 on `owner`'s foot, or 6 m from everyone when owner is None."""
    attack_x = 10.0 if owner == "attack" else 2.0
    defence_x = 10.0 if owner == "defence" else 18.0
    ball_y = 0.0 if owner else 6.0
    return BundesligaFrame(
        match_id="match",
        frame_id=frame_id,
        period=1,
        game_section="firstHalf",
        timestamp=None,
        players={
            "carrier": BundesligaObjectState("carrier", "attack", attack_x, 0.0),
            "runner": BundesligaObjectState("runner", "attack", 15.0, 5.0),
            "defender": BundesligaObjectState("defender", "defence", defence_x, 0.0),
        },
        ball=BundesligaObjectState("ball", "BALL", 10.0, ball_y, z=0.0),
    )


def _settled_candidate() -> RunOnsetCandidate:
    return RunOnsetCandidate(
        match_id="match", frame_id=100, player_id="runner", team_id="attack",
        ball_carrier_id="carrier", labels=("acceleration",), x=15.0, y=5.0,
        onset_speed_mps=2.0, pre_mean_speed_mps=1.0, post_mean_speed_mps=4.0,
        speed_gain_mps=3.0, direction_change_degrees=0.0, post_displacement_m=4.0,
        peak_acceleration_mps2=3.0, peak_deceleration_mps2=0.0, confidence_score=1.0,
        ball_distance_m=8.0, stable_control_fraction=1.0, future_known_fraction=1.0,
        future_same_team_fraction=1.0,
    )


def _assess(last_owner: str, in_flight: bool):
    # last owner on the ball up to frame 79, then nobody on it through the onset at 100
    frames = {f: _settled_frame(f, last_owner if f < 80 else None) for f in range(60, 151)}
    return assess_settled_shot_context_onset(
        _settled_candidate(),
        frames=frames,
        events=pd.DataFrame(columns=["frame_id", "event_type", "from_open_play"]),
        phase_start_frame_id=60,
        shot_frame_id=150,
        attacking_direction=1,
        config=SettledPossessionConfig(in_flight_control=in_flight),
    )


class InFlightSettledHistoryTest(unittest.TestCase):
    def test_unknown_history_is_rejected_without_the_option(self) -> None:
        assessment = _assess("attack", in_flight=False)

        self.assertFalse(assessment.accepted)
        self.assertIn("insufficient_known_control_history", assessment.rejection_reasons)

    def test_unknown_history_after_own_control_counts_as_own(self) -> None:
        assessment = _assess("attack", in_flight=True)

        self.assertTrue(assessment.accepted, assessment.rejection_reasons)
        self.assertEqual(assessment.team_control_fraction, 1.0)

    def test_unknown_history_after_opponent_control_counts_as_opponent(self) -> None:
        assessment = _assess("defence", in_flight=True)

        self.assertFalse(assessment.accepted)
        self.assertIn("recent_opponent_control", assessment.rejection_reasons)


if __name__ == "__main__":
    unittest.main()
