from __future__ import annotations

import unittest

from offball_value.meeting_scene_gallery import (
    build_meeting_gallery_payload,
    render_meeting_scene_gallery,
)


def _observed(index: int) -> dict[str, object]:
    return {
        "match_id": "match",
        "onset_frame_id": index,
        "runner_id": f"runner-{index}",
        "runner_name": f"Runner {index}",
        "carrier_name": f"Carrier {index}",
        "core_review": {"core_decision": "include"},
    }


class MeetingSceneGalleryTests(unittest.TestCase):
    def test_build_payload_keeps_eight_unique_items(self) -> None:
        payload = build_meeting_gallery_payload(
            {"runner_name": "F. Klaus"}, [_observed(i) for i in range(8)]
        )

        self.assertEqual(payload["scene_count"], 8)
        self.assertEqual(payload["prototype"]["runner_name"], "F. Klaus")
        self.assertEqual(len(payload["observed"]), 8)

    def test_build_payload_requires_eight_confirmed_scenes(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly eight"):
            build_meeting_gallery_payload({}, [_observed(0)])

        scenes = [_observed(i) for i in range(8)]
        scenes[-1]["core_review"] = {"core_decision": "hold"}
        with self.assertRaisesRegex(ValueError, "human-confirmed"):
            build_meeting_gallery_payload({}, scenes)

    def test_render_embeds_payload_and_actual_dynamic_distinction(self) -> None:
        payload = build_meeting_gallery_payload(
            {"runner_name": "F. Klaus"}, [_observed(i) for i in range(8)]
        )
        html = render_meeting_scene_gallery(payload)

        self.assertIn("8-scene meeting gallery", html)
        self.assertIn("COUNTERFACTUAL PROTOTYPE", html)
        self.assertIn("HUMAN-CONFIRMED OBSERVED", html)
        self.assertIn('"scene_count":8', html)
        self.assertIn("actual motion ↔ old dynamic prototype", html)


if __name__ == "__main__":
    unittest.main()
