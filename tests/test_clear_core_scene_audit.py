import unittest

from offball_value.clear_core_scene_audit import (
    attach_confirmed_core_reviews,
    render_clear_core_scene_audit,
    select_clear_core_scenes,
)


class ClearCoreSceneAuditTest(unittest.TestCase):
    def test_selection_requires_all_three_prior_acceptance_labels(self):
        scenes = [
            {"match_id": "m", "onset_frame_id": 1, "runner_id": "r"},
            {"match_id": "m", "onset_frame_id": 2, "runner_id": "r"},
        ]
        reviews = [
            {
                "match_id": "m",
                "onset_frame_id": "1",
                "runner_id": "r",
                "onset_review": "correct",
                "possession_review": "settled",
                "interaction_review": "clear",
                "note": "keep",
            },
            {
                "match_id": "m",
                "onset_frame_id": "2",
                "runner_id": "r",
                "onset_review": "correct",
                "possession_review": "settled",
                "interaction_review": "possible",
                "note": "hold",
            },
        ]
        selected = select_clear_core_scenes(scenes, reviews)
        self.assertEqual([scene["onset_frame_id"] for scene in selected], [1])
        self.assertEqual(selected[0]["prior_review"]["note"], "keep")

    def test_html_has_actual_only_review_contract(self):
        html = render_clear_core_scene_audit([])
        self.assertIn("관측된 실제 움직임만 검수", html)
        self.assertIn("확정된 0개 장면", html)
        self.assertIn("보류 장면은 이 공유용 데모에서 제외", html)
        self.assertIn("파생 공격 옵션", html)
        self.assertIn("clear_core_local_game_v0_1_reviews.csv", html)
        self.assertIn("scenes[si].core_review", html)
        self.assertNotIn("Team derived", html.split("<script id=", 1)[0])

    def test_second_pass_requires_include_and_three_yes_labels(self):
        scenes = [
            {"match_id": "m", "onset_frame_id": 1, "runner_id": "r"},
            {"match_id": "m", "onset_frame_id": 2, "runner_id": "r"},
        ]
        reviews = [
            {
                "match_id": "m",
                "onset_frame_id": "1",
                "runner_id": "r",
                "core_decision": "include",
                "defender_visible": "yes",
                "derived_visible": "yes",
                "tradeoff_visible": "yes",
                "primary_defender": "d",
                "derived_option": "k",
            },
            {
                "match_id": "m",
                "onset_frame_id": "2",
                "runner_id": "r",
                "core_decision": "hold",
                "defender_visible": "yes",
                "derived_visible": "unclear",
                "tradeoff_visible": "unclear",
            },
        ]
        reviewed, confirmed = attach_confirmed_core_reviews(scenes, reviews)
        self.assertEqual(len(reviewed), 2)
        self.assertEqual([scene["onset_frame_id"] for scene in confirmed], [1])
        self.assertEqual(confirmed[0]["core_review"]["derived_option"], "k")


if __name__ == "__main__":
    unittest.main()
