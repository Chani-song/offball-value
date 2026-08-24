from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from offball_value.bundesliga import (
    BundesligaFrame,
    BundesligaMatchMeta,
    BundesligaObjectState,
    BundesligaPlayerMeta,
    BundesligaTeamMeta,
    load_bundesliga_events,
)
from offball_value.shot_context import (
    ShotAnchor,
    ShotContextConfig,
    extract_shot_anchors,
    infer_shot_attacking_phase,
)


def _metadata() -> BundesligaMatchMeta:
    return BundesligaMatchMeta(
        match_id="match",
        home_team_id="attack",
        away_team_id="defense",
        home_team_name="Attack",
        away_team_name="Defense",
        teams={
            "attack": BundesligaTeamMeta("attack", "Attack", "home"),
            "defense": BundesligaTeamMeta("defense", "Defense", "away"),
        },
        players={
            "gk_a": BundesligaPlayerMeta("gk_a", "attack", "GK A", position="TW"),
            "a": BundesligaPlayerMeta("a", "attack", "Attacker"),
            "gk_d": BundesligaPlayerMeta("gk_d", "defense", "GK D", position="TW"),
            "d": BundesligaPlayerMeta("d", "defense", "Defender"),
        },
    )


def _frame(frame_id: int, owner: str, ball_x: float) -> BundesligaFrame:
    attacking_x = ball_x if owner == "attack" else ball_x - 8.0
    defending_x = ball_x if owner == "defense" else ball_x + 8.0
    return BundesligaFrame(
        match_id="match",
        frame_id=frame_id,
        period=1,
        game_section="firstHalf",
        timestamp=None,
        players={
            "gk_a": BundesligaObjectState("gk_a", "attack", -50.0, 0.0),
            "a": BundesligaObjectState("a", "attack", attacking_x, 0.0),
            "gk_d": BundesligaObjectState("gk_d", "defense", 50.0, 0.0),
            "d": BundesligaObjectState("d", "defense", defending_x, 0.0),
        },
        ball=BundesligaObjectState("ball", "BALL", ball_x, 0.0, z=0.0),
    )


class ShotContextTest(unittest.TestCase):
    def test_direct_shot_xml_restores_actor_and_shot_metadata(self) -> None:
        xml = """<Events><Event MatchId="match" EventId="shot-1" CalculatedFrame="100" X-Source-Position="80" Y-Source-Position="34"><ShotAtGoal Team="attack" Player="a" xG="0.21" BuildUp="passOpenPlay" TakerSetup="otherPassFromOpenPlay" /></Event></Events>"""
        with TemporaryDirectory() as directory:
            path = Path(directory) / "events.xml"
            path.write_text(xml, encoding="utf-8")
            events = load_bundesliga_events(path)

        row = events.iloc[0]
        self.assertEqual(row["event_type"], "shotatgoal")
        self.assertEqual(row["team_id"], "attack")
        self.assertEqual(row["player_id"], "a")
        self.assertAlmostEqual(float(row["xg"]), 0.21)
        self.assertTrue(bool(row["from_open_play"]))

    def test_extract_shots_excludes_set_piece_shots(self) -> None:
        events = pd.DataFrame(
            [
                {
                    "match_id": "match",
                    "event_id": "open",
                    "frame_id": 100,
                    "period": 1,
                    "event_type": "shotatgoal",
                    "team_id": "attack",
                    "player_id": "a",
                    "from_open_play": True,
                    "xg": 0.1,
                    "x_start": 20.0,
                    "y_start": 1.0,
                    "shot_build_up": "passOpenPlay",
                    "shot_setup": "otherPassFromOpenPlay",
                },
                {
                    "match_id": "match",
                    "event_id": "corner",
                    "frame_id": 200,
                    "period": 1,
                    "event_type": "shotatgoal",
                    "team_id": "attack",
                    "player_id": "a",
                    "from_open_play": False,
                    "xg": 0.2,
                    "x_start": 15.0,
                    "y_start": 0.0,
                    "shot_build_up": "cornerKick",
                    "shot_setup": "cornerKick",
                },
            ]
        )

        anchors = extract_shot_anchors(events)

        self.assertEqual([anchor.event_id for anchor in anchors], ["open"])

    def test_phase_starts_after_confirmed_opponent_control(self) -> None:
        frames = {}
        for frame_id in range(101):
            owner = "defense" if frame_id <= 20 else "attack"
            if frame_id == 50:
                owner = "defense"  # one isolated touch must not split possession
            ball_x = -10.0 if frame_id <= 60 else 10.0
            frames[frame_id] = _frame(frame_id, owner, ball_x)
        shot = ShotAnchor(
            match_id="match",
            event_id="shot",
            frame_id=100,
            period=1,
            team_id="attack",
            player_id="a",
            xg=0.1,
            x=10.0,
            y=0.0,
            from_open_play=True,
            build_up="passOpenPlay",
            setup="otherPassFromOpenPlay",
        )

        phase = infer_shot_attacking_phase(
            shot,
            frames,
            _metadata(),
            ShotContextConfig(max_lookback_seconds=4.0),
        )

        self.assertEqual(phase.possession_start_frame_id, 21)
        self.assertEqual(phase.attacking_half_entry_frame_id, 61)
        self.assertEqual(phase.boundary_reason, "confirmed_opponent_control")
        self.assertAlmostEqual(phase.duration_seconds, 79 / 25)


if __name__ == "__main__":
    unittest.main()
