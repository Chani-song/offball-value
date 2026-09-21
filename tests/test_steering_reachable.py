from __future__ import annotations

import math
import unittest

from offball_value.bundesliga import BundesligaObjectState
from offball_value.empirical_action_space import CausalMotionState
from offball_value.steering_reachable import (
    SteeringReachabilityConfig,
    SteeringEndpointActionSet,
    enumerate_steering_motions,
    generate_plant_cut_endpoint_actions,
    generate_steering_endpoint_actions,
    steering_step,
)
from offball_value.fernandez_influence import fernandez_influence_ellipse


class SteeringReachableTest(unittest.TestCase):
    def test_plant_cut_adds_deep_reverse_and_ninety_degree_actions(self) -> None:
        speed = 7.669
        heading = math.pi
        start = (8.2, -9.91)
        velocity = (-speed, 0.0)
        ellipse = fernandez_influence_ellipse(start, velocity, (0.0, 0.0))
        base = SteeringEndpointActionSet(
            player_id="runner",
            start_x=start[0],
            start_y=start[1],
            initial_vx_mps=velocity[0],
            initial_vy_mps=velocity[1],
            initial_speed_mps=speed,
            reference_heading_radians=heading,
            influence_ellipse=ellipse,
            proposal_endpoints=(),
            actions=(),
        )
        actions = generate_plant_cut_endpoint_actions(base)

        self.assertTrue(actions)
        self.assertEqual(len(actions), len({action.action_id for action in actions}))
        self.assertIn(90.0, {action.motion.cut_angle_degrees for action in actions})
        self.assertIn(180.0, {action.motion.cut_angle_degrees for action in actions})
        forward_displacements = [
            (action.endpoint_x - start[0]) * math.cos(heading)
            + (action.endpoint_y - start[1]) * math.sin(heading)
            for action in actions
        ]
        self.assertLess(min(forward_displacements), 1.0)
        for action in actions:
            self.assertEqual(action.motion.maneuver_type, "plant_and_cut")
            self.assertLessEqual(action.motion.maximum_path_speed_mps, 9.0)
            self.assertLessEqual(
                action.motion.maximum_tangential_deceleration_mps2,
                9.0,
            )
            self.assertEqual(action.motion.path_xy[-1][0], action.endpoint_x)
            self.assertEqual(action.motion.path_xy[-1][1], action.endpoint_y)

    def test_high_speed_left_and_right_turns_survive_state_pruning(self) -> None:
        config = SteeringReachabilityConfig(
            horizon_seconds=0.1,
            path_sample_seconds=(0.1,),
        )
        states = enumerate_steering_motions(
            (0.0, 0.0),
            (9.0, 0.0),
            0.0,
            config,
        )
        headings = [state.heading_radians for state in states]

        self.assertLess(min(headings), -0.05)
        self.assertGreater(max(headings), 0.05)

    def test_pure_normal_control_preserves_speed_and_turns(self) -> None:
        config = SteeringReachabilityConfig()
        result = steering_step(
            0.0,
            0.0,
            8.0,
            0.0,
            0.0,
            config.max_normal_acceleration_mps2,
            config,
        )

        self.assertAlmostEqual(result.speed_mps, 8.0)
        self.assertAlmostEqual(
            result.heading_radians,
            config.max_normal_acceleration_mps2
            / 8.0
            * config.integration_step_seconds,
        )
        self.assertGreater(result.y, 0.0)

    def test_stationary_launch_covers_all_spatial_sectors(self) -> None:
        config = SteeringReachabilityConfig(
            horizon_seconds=0.5,
            path_sample_seconds=(0.5,),
            control_direction_count=8,
            state_heading_bins=16,
        )
        states = enumerate_steering_motions(
            (0.0, 0.0),
            (0.0, 0.0),
            0.0,
            config,
        )
        sectors = {
            int(
                math.floor(
                    ((math.atan2(state.y, state.x) + 2.0 * math.pi)
                     % (2.0 * math.pi))
                    / (2.0 * math.pi / 8)
                )
            )
            for state in states
            if math.hypot(state.x, state.y) >= 0.1
        }
        self.assertEqual(sectors, set(range(8)))

    def test_brake_then_reaccelerate_can_finish_behind_start(self) -> None:
        config = SteeringReachabilityConfig(
            state_heading_bins=16,
            control_direction_count=8,
        )
        states = enumerate_steering_motions(
            (0.0, 0.0),
            (4.0, 0.0),
            0.0,
            config,
        )
        self.assertLess(min(state.x for state in states), -0.25)

    def test_generated_actions_obey_caps_and_keep_continuous_witness(self) -> None:
        config = SteeringReachabilityConfig(
            horizon_seconds=0.5,
            path_sample_seconds=(0.5,),
            control_direction_count=8,
            state_heading_bins=16,
        )
        state = CausalMotionState(
            vx_mps=4.0,
            vy_mps=0.0,
            speed_mps=4.0,
            ax_mps2=0.0,
            ay_mps2=0.0,
            longitudinal_acceleration_mps2=0.0,
            heading_radians=0.0,
            sample_count=11,
            observed_window_seconds=0.4,
        )
        result = generate_steering_endpoint_actions(
            BundesligaObjectState("runner", "attack", 0.0, 0.0),
            BundesligaObjectState("ball", "BALL", 0.0, 0.0),
            state,
            config=config,
        )

        self.assertTrue(result.optimization_actions)
        for action in result.optimization_actions:
            self.assertAlmostEqual(action.motion.path_xy[-1][0], action.endpoint_x)
            self.assertAlmostEqual(action.motion.path_xy[-1][1], action.endpoint_y)
            self.assertLessEqual(
                action.motion.maximum_path_speed_mps,
                config.max_speed_mps + 1e-9,
            )
            self.assertLessEqual(
                action.motion.maximum_tangential_acceleration_mps2,
                config.max_tangential_acceleration_mps2 + 1e-9,
            )
            self.assertLessEqual(
                action.motion.maximum_tangential_deceleration_mps2,
                config.max_tangential_deceleration_mps2 + 1e-9,
            )
            self.assertLessEqual(
                action.motion.maximum_normal_acceleration_mps2,
                config.max_normal_acceleration_mps2 + 1e-9,
            )
            self.assertLessEqual(
                action.grid_snap_distance_m,
                math.sqrt(0.5),
            )


if __name__ == "__main__":
    unittest.main()
