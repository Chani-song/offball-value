from __future__ import annotations

import math
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
    PassAnchorConfig,
    RunOnsetConfig,
    _context_candidate,
    _nearest_player,
    detect_pass_anchored_run_onsets,
    team_controls_ball,
)
from scripts.extract_settled_possession_run_onsets import _pass_anchors, _repeated_pass_anchors


FPS = 25
TEAMS = {"passer": "attack", "runner": "attack", "defender": "defence", "far": "defence"}


def _frame(frame_id: int, positions: dict[str, tuple[float, float]], ball, z: float | None = 0.0):
    return BundesligaFrame(
        match_id="match",
        frame_id=frame_id,
        period=1,
        game_section="firstHalf",
        timestamp=None,
        players={pid: BundesligaObjectState(pid, TEAMS[pid], x, y) for pid, (x, y) in positions.items()},
        ball=BundesligaObjectState("ball", "BALL", ball[0], ball[1], z=z),
    )


def _meta(keeper: str | None = None) -> BundesligaMatchMeta:
    return BundesligaMatchMeta(
        match_id="match",
        home_team_id="attack",
        away_team_id="defence",
        home_team_name="Attack",
        away_team_name="Defence",
        teams={tid: BundesligaTeamMeta(tid, tid, role) for tid, role in (("attack", "home"), ("defence", "away"))},
        players={
            pid: BundesligaPlayerMeta(pid, team, pid, position="TW" if pid == keeper else "MF")
            for pid, team in TEAMS.items()
        },
    )


# The pass at frame 60. The passer stands on the ball at the origin all along;
# the runner crosses at `runner_speed` along y = 10 and is at (-8, 10) at the
# pass; the defender stands `gap` m from him at the pass and sets off at
# `defender_start` (None: never), reaching 6 m/s in 10 frames.
PASS = 60


def _defender_x(frame_id: int, start: int | None, at_pass: float) -> float:
    if start is None or frame_id <= start:
        return at_pass
    return at_pass + sum(6.0 * min(k, 10) / 10 for k in range(1, frame_id - start + 1)) / FPS


def _pass_frames(
    runner_speed: float = 5.0,
    gap: float = 4.0,
    defender_start: int | None = 70,
    ball_with: str = "passer",
) -> dict[int, BundesligaFrame]:
    frames = {}
    for frame_id in range(150):
        runner_x = -8.0 + runner_speed * (frame_id - PASS) / FPS
        positions = {
            "passer": (0.0, 0.0),
            "runner": (runner_x, 10.0),
            "defender": (_defender_x(frame_id, defender_start, -8.0), 10.0 + gap),
            "far": (30.0, -20.0),
        }
        ball = (0.3, 0.0) if ball_with == "passer" or frame_id < 50 else (30.3, -20.0)
        frames[frame_id] = _frame(frame_id, positions, ball)
    return frames


def _detect(frames, meta=None, anchors=((PASS, "passer", "attack"),), **anchor_settings):
    return detect_pass_anchored_run_onsets(
        frames,
        meta or _meta(),
        list(anchors),
        config=RunOnsetConfig(required_player_count=None, in_flight_control=True),
        anchor_config=PassAnchorConfig(**anchor_settings),
    )


class PassAnchoredOnsetTest(unittest.TestCase):
    def test_runner_at_speed_with_a_reacting_defender_is_a_candidate(self) -> None:
        found = _detect(_pass_frames())

        self.assertEqual(len(found), 1)
        candidate = found[0].candidate
        self.assertEqual((candidate.frame_id, candidate.player_id, candidate.ball_carrier_id),
                         (PASS, "runner", "passer"))
        self.assertEqual(candidate.labels, ("pass_anchor",))
        self.assertEqual(candidate.primary_type, "pass_anchor")
        self.assertAlmostEqual(candidate.post_mean_speed_mps, 5.0, places=3)
        self.assertEqual(found[0].passer_id, "passer")
        (defender_id, distance, onset_frame), = found[0].reactions
        self.assertEqual(defender_id, "defender")
        self.assertAlmostEqual(distance, 4.0)
        self.assertTrue(PASS <= onset_frame <= PASS + FPS, onset_frame)

    def test_defender_who_does_not_react_makes_no_candidate(self) -> None:
        self.assertEqual(_detect(_pass_frames(defender_start=None)), ())

    def test_reaction_after_the_window_is_not_one(self) -> None:
        self.assertEqual(_detect(_pass_frames(defender_start=100)), ())

    def test_defender_beyond_the_radius_does_not_count(self) -> None:
        frames = _pass_frames(gap=12.0)

        self.assertEqual(_detect(frames), ())
        self.assertEqual(len(_detect(frames, reaction_radius_m=15.0)), 1)

    def test_slow_runner_is_not_a_candidate(self) -> None:
        self.assertEqual(_detect(_pass_frames(runner_speed=3.0)), ())

    def test_keepers_are_neither_runners_nor_reacting_defenders(self) -> None:
        self.assertEqual(_detect(_pass_frames(), meta=_meta(keeper="runner")), ())
        self.assertEqual(_detect(_pass_frames(), meta=_meta(keeper="defender")), ())

    def test_passer_is_not_a_runner_of_his_own_pass(self) -> None:
        self.assertEqual(_detect(_pass_frames(), anchors=((PASS, "runner", "attack"),)), ())

    def test_anchor_without_its_passer_on_the_pitch_is_skipped(self) -> None:
        self.assertEqual(_detect(_pass_frames(), anchors=((PASS, "ghost", "attack"),)), ())

    def test_candidate_must_pass_the_context_gate(self) -> None:
        # the ball is on the far defender's foot from frame 50
        self.assertEqual(_detect(_pass_frames(ball_with="far")), ())

    def test_one_candidate_per_pass_frame_and_runner(self) -> None:
        anchors = ((PASS, "passer", "attack"), (PASS, "passer", "attack"))

        self.assertEqual(len(_detect(_pass_frames(), anchors=anchors)), 1)


# A ball on the passer's foot at the origin through frame 15, then held 3 m up
# over a defender standing 0.5 m from it (a lofted pass, frozen).
def _lofted_frames(height: float) -> dict[int, BundesligaFrame]:
    frames = {}
    for frame_id in range(61):
        positions = {"passer": (-0.3, 0.0), "runner": (10.0, 10.0), "defender": (8.0, 0.5),
                     "far": (30.0, -20.0)}
        on_foot = frame_id <= 15
        frames[frame_id] = _frame(frame_id, positions, (0.0, 0.0) if on_foot else (8.0, 0.0),
                                  z=0.0 if on_foot else height)
    return frames


def _onset(frame_id: int = 20) -> KinematicRunOnset:
    return KinematicRunOnset(
        frame_id=frame_id, labels=("acceleration",), x=10.0, y=10.0, onset_speed_mps=2.0,
        pre_mean_speed_mps=1.0, post_mean_speed_mps=4.0, speed_gain_mps=3.0,
        direction_change_degrees=0.0, post_displacement_m=4.0, peak_acceleration_mps2=3.0,
        peak_deceleration_mps2=0.0, confidence_score=1.0,
    )


def _gate(frames, cap: float | None):
    config = RunOnsetConfig(required_player_count=None, in_flight_control=True,
                            max_control_ball_height_m=cap)
    return _context_candidate(_onset(), "runner", frames, _meta(), config)


class OnsetBallHeightCapTest(unittest.TestCase):
    def test_nobody_is_on_a_ball_above_the_cap(self) -> None:
        frame = _lofted_frames(3.0)[20]

        self.assertEqual(_nearest_player(frame), ("defender", 0.5))
        self.assertEqual(_nearest_player(frame, 1.5), ("defender", math.inf))
        self.assertTrue(team_controls_ball(frame, "defence", 1.5))
        self.assertFalse(team_controls_ball(frame, "defence", 1.5, max_ball_height_m=1.5))

    def test_ball_without_a_height_is_not_capped(self) -> None:
        frame = _lofted_frames(3.0)[20]
        frame = _frame(20, {pid: (p.x, p.y) for pid, p in frame.players.items()}, (8.0, 0.0), z=None)

        self.assertEqual(_nearest_player(frame, 1.5), ("defender", 0.5))

    def test_lofted_pass_over_a_defender_stays_the_passers_team(self) -> None:
        frames = _lofted_frames(3.0)

        self.assertIsNone(_gate(frames, cap=None))
        candidate = _gate(frames, cap=1.5)
        self.assertIsNotNone(candidate)
        self.assertEqual(candidate.ball_carrier_id, "passer")
        self.assertTrue(candidate.ball_in_flight)
        self.assertEqual(candidate.stable_control_fraction, 1.0)
        self.assertEqual(candidate.future_same_team_fraction, 1.0)

    def test_ball_below_the_cap_is_still_the_defenders(self) -> None:
        self.assertIsNone(_gate(_lofted_frames(1.0), cap=1.5))

    def test_cap_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            RunOnsetConfig(max_control_ball_height_m=0.0).validate()


class PassAnchorsFromEventsTest(unittest.TestCase):
    def test_passes_and_crosses_of_the_possessing_team_inside_the_window(self) -> None:
        events = pd.DataFrame(
            [
                {"frame_id": 100, "event_type": "pass", "player_id": "a1", "team_id": "attack"},
                {"frame_id": 100, "event_type": "pass", "player_id": "a1", "team_id": "attack"},
                {"frame_id": 150, "event_type": "cross", "player_id": "a2", "team_id": "attack"},
                {"frame_id": 200, "event_type": "pass", "player_id": "a1", "team_id": "attack"},
                {"frame_id": 201, "event_type": "pass", "player_id": "a1", "team_id": "attack"},
                {"frame_id": 120, "event_type": "pass", "player_id": "d1", "team_id": "defence"},
                {"frame_id": 130, "event_type": "shotatgoal", "player_id": "a2", "team_id": "attack"},
                {"frame_id": None, "event_type": "pass", "player_id": "a1", "team_id": "attack"},
                {"frame_id": 140, "event_type": "pass", "player_id": None, "team_id": "attack"},
            ]
        )
        phases = pd.DataFrame(
            [{"team_id": "attack", "search_start_frame_id": 100, "search_end_frame_id": 200}]
        )

        self.assertEqual(
            sorted(_pass_anchors(events, phases)),
            [(100, "a1", "attack"), (150, "a2", "attack"), (200, "a1", "attack")],
        )


def _row(frame_id: int, source: str, player: str = "r1") -> dict:
    return {"frame_id": frame_id, "player_id": player, "candidate_source": source}


class RepeatedPassAnchorsTest(unittest.TestCase):
    def test_a_run_is_one_scene_at_its_first_pass_unless_the_detector_has_it(self) -> None:
        fresh = pd.DataFrame(
            [
                _row(1000, "kinematic"),
                _row(1075, "pass_anchor"),          # 3 s after the kinematic onset
                _row(2000, "pass_anchor"),          # first pass of a new run: kept
                _row(2070, "pass_anchor"),
                _row(2080, "pass_anchor"),          # 3.2 s after the first: a new scene
                _row(2010, "pass_anchor", "r2"),    # another runner
                _row(2000, "kinematic", "r3"),
            ]
        )

        self.assertEqual(
            _repeated_pass_anchors(fresh, window_frames=75),
            {1: "kinematic", 3: "pass_anchor"},
        )


if __name__ == "__main__":
    unittest.main()
