from __future__ import annotations

import unittest

import pandas as pd

from offball_value.bundesliga import BundesligaFrame, BundesligaObjectState
from offball_value.run_onset import RunOnsetCandidate
from offball_value.shot_context_onset import assess_settled_shot_context_onset


def _candidate(frame_id: int = 100) -> RunOnsetCandidate:
    return RunOnsetCandidate(
        match_id="match",
        frame_id=frame_id,
        player_id="runner",
        team_id="attack",
        ball_carrier_id="carrier",
        labels=("acceleration",),
        x=15.0,
        y=5.0,
        onset_speed_mps=2.0,
        pre_mean_speed_mps=1.0,
        post_mean_speed_mps=4.0,
        speed_gain_mps=3.0,
        direction_change_degrees=0.0,
        post_displacement_m=4.0,
        peak_acceleration_mps2=3.0,
        peak_deceleration_mps2=0.0,
        confidence_score=1.0,
        ball_distance_m=8.0,
        stable_control_fraction=1.0,
        future_known_fraction=1.0,
        future_same_team_fraction=1.0,
    )


def _frame(frame_id: int, owner: str = "attack", ball_x: float = 10.0) -> BundesligaFrame:
    carrier_x = ball_x if owner == "attack" else ball_x - 8.0
    defender_x = ball_x if owner == "defense" else ball_x + 8.0
    return BundesligaFrame(
        match_id="match",
        frame_id=frame_id,
        period=1,
        game_section="firstHalf",
        timestamp=None,
        players={
            "carrier": BundesligaObjectState("carrier", "attack", carrier_x, 0.0),
            "runner": BundesligaObjectState("runner", "attack", 15.0, 5.0),
            "defender": BundesligaObjectState("defender", "defense", defender_x, 0.0),
        },
        ball=BundesligaObjectState("ball", "BALL", ball_x, 0.0, z=0.0),
    )


def _empty_events() -> pd.DataFrame:
    return pd.DataFrame(columns=["frame_id", "event_type", "from_open_play"])


class ShotContextOnsetTest(unittest.TestCase):
    def test_settled_opponent_half_onset_is_accepted(self) -> None:
        frames = {frame_id: _frame(frame_id) for frame_id in range(75, 151)}

        assessment = assess_settled_shot_context_onset(
            _candidate(),
            frames=frames,
            events=_empty_events(),
            phase_start_frame_id=80,
            shot_frame_id=150,
            attacking_direction=1,
        )

        self.assertTrue(assessment.accepted)
        self.assertEqual(assessment.rejection_reasons, ())
        self.assertEqual(assessment.opponent_control_frames, 0)

    def test_recent_opponent_control_rejects_immediate_turnover(self) -> None:
        frames = {
            frame_id: _frame(frame_id, "defense" if frame_id < 88 else "attack")
            for frame_id in range(75, 151)
        }

        assessment = assess_settled_shot_context_onset(
            _candidate(),
            frames=frames,
            events=_empty_events(),
            phase_start_frame_id=80,
            shot_frame_id=150,
            attacking_direction=1,
        )

        self.assertFalse(assessment.accepted)
        self.assertIn("recent_opponent_control", assessment.rejection_reasons)

    def test_unstable_event_and_own_half_are_explicit_reasons(self) -> None:
        frames = {
            frame_id: _frame(frame_id, ball_x=-10.0)
            for frame_id in range(75, 151)
        }
        events = pd.DataFrame(
            [{"frame_id": 98, "event_type": "tacklinggame", "from_open_play": None}]
        )

        assessment = assess_settled_shot_context_onset(
            _candidate(),
            frames=frames,
            events=events,
            phase_start_frame_id=80,
            shot_frame_id=150,
            attacking_direction=1,
        )

        self.assertFalse(assessment.accepted)
        self.assertIn("unstable_event_window", assessment.rejection_reasons)
        self.assertIn("ball_not_in_opponent_half", assessment.rejection_reasons)


if __name__ == "__main__":
    unittest.main()
