from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import select_new_scenes
from scripts.select_new_scenes import select_fresh


def scene(frame: int, runner: str = "r1", match: str = "m", speed: float = 5.0, **extra) -> dict:
    return {"match_id": match, "runner_id": runner, "onset_frame_id": frame, "post_speed_mps": speed, **extra}


class SameSceneTest(unittest.TestCase):
    def test_same_runner_within_window_is_not_new(self) -> None:
        fresh, _ = select_fresh([scene(1075)], [scene(1000)])
        self.assertEqual(fresh, [])

    def test_same_runner_outside_window_is_new(self) -> None:
        fresh, _ = select_fresh([scene(1076)], [scene(1000)])
        self.assertEqual([s["onset_frame_id"] for s in fresh], [1076])

    def test_other_runner_or_match_at_same_frame_is_new(self) -> None:
        fresh, _ = select_fresh([scene(1000, runner="r2"), scene(1000, match="m2")], [scene(1000)])
        self.assertEqual(len(fresh), 2)


class RunSpeedPrefilterTest(unittest.TestCase):
    def test_drops_runs_below_the_cut_and_keeps_the_cut(self) -> None:
        new = [scene(100, speed=3.79), scene(500, speed=3.8), scene(900, speed=6.0)]
        fresh, dropped = select_fresh(new, [], min_run_speed=3.8)
        self.assertEqual([s["onset_frame_id"] for s in fresh], [500, 900])
        self.assertEqual(dropped["slow"], 1)

    def test_no_cut_keeps_every_new_scene(self) -> None:
        fresh, dropped = select_fresh([scene(100, speed=0.5)], [])
        self.assertEqual(len(fresh), 1)
        self.assertEqual(dropped["slow"], 0)

    def test_scene_without_a_run_speed_counts_as_slow(self) -> None:
        bare = {"match_id": "m", "runner_id": "r1", "onset_frame_id": 100}
        fresh, _ = select_fresh([bare], [], min_run_speed=3.8)
        self.assertEqual(fresh, [])

    def test_old_scenes_are_removed_before_the_speed_cut_is_counted(self) -> None:
        fresh, dropped = select_fresh([scene(1000, speed=1.0), scene(5000, speed=1.0)], [scene(1000)],
                                      min_run_speed=3.8)
        self.assertEqual(fresh, [])
        self.assertEqual(dropped["slow"], 1)


class NearShotTest(unittest.TestCase):
    def test_keeps_only_onsets_inside_the_window_before_a_shot(self) -> None:
        new = [scene(700), scene(900), scene(1000), scene(1001)]
        fresh, dropped = select_fresh(new, [], near_shot_frames={"m": [1000]}, shot_window_frames=300)
        self.assertEqual([s["onset_frame_id"] for s in fresh], [700, 900])
        self.assertEqual(dropped["not_near_shots"], 2)


class MainWritesChunksTest(unittest.TestCase):
    def test_writes_new_scenes_as_chunks_with_a_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, payload in (("old", [scene(1000)]),
                                  ("new", [scene(1000), scene(3000, speed=2.0), scene(5000), scene(7000)])):
                (root / name).mkdir()
                (root / name / "chunk0.json").write_text(json.dumps(payload))
                (root / name / "manifest.json").write_text(json.dumps({"chunk_count": 1}))
            argv = ["select_new_scenes.py", "--new", str(root / "new"), "--old", str(root / "old"),
                    "--output", str(root / "out"), "--chunk-size", "1", "--min-run-speed", "3.8",
                    "--mark-run-speed", "5.0"]
            with mock.patch.object(sys, "argv", argv), mock.patch("builtins.print"):
                select_new_scenes.main()
            manifest = json.loads((root / "out" / "manifest.json").read_text())
            self.assertEqual((manifest["scene_count"], manifest["chunk_count"], manifest["dropped_slow"]), (2, 2, 1))
            self.assertEqual((manifest["run_speed_mark"], manifest["scenes_meeting_mark"]), (5.0, 2))
            frames = [json.loads((root / "out" / f"chunk{k}.json").read_text())[0]["onset_frame_id"]
                      for k in range(2)]
            self.assertEqual(frames, [5000, 7000])

    def test_a_scene_in_any_old_pool_is_not_new(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, payload in (("old1", [scene(1000)]), ("old2", [scene(3000)]),
                                  ("new", [scene(1000), scene(3000), scene(5000)])):
                (root / name).mkdir()
                (root / name / "chunk0.json").write_text(json.dumps(payload))
                (root / name / "manifest.json").write_text(json.dumps({"chunk_count": 1}))
            argv = ["select_new_scenes.py", "--new", str(root / "new"),
                    "--old", str(root / "old1"), str(root / "old2"), "--output", str(root / "out")]
            with mock.patch.object(sys, "argv", argv), mock.patch("builtins.print"):
                select_new_scenes.main()
            manifest = json.loads((root / "out" / "manifest.json").read_text())
            self.assertEqual(manifest["scene_count"], 1)
            self.assertEqual(manifest["compared_to"], [str(root / "old1"), str(root / "old2")])
            chunk = json.loads((root / "out" / "chunk0.json").read_text())
            self.assertEqual([s["onset_frame_id"] for s in chunk], [5000])


if __name__ == "__main__":
    unittest.main()
