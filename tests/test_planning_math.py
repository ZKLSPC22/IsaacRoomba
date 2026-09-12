"""Tests for `envs.planning_math`.

These tests pin the *existing* reward and heuristic formulas. They import `torch`
only, so they run on CPU without Isaac Gym.

The heuristic under test is a **value estimate** used as an MCTS leaf value. It is
not part of the environment reward and it is **not** potential-based reward
shaping; PBRS would require an added reward term of the form
`gamma * Phi(next_state) - Phi(state)`, which this codebase does not implement.
"""

import math
import sys
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import yaml

from envs.planning_math import (
    compute_planning_rewards,
    compute_straight_line_heuristic,
)

CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "config.yaml"

# Canonical task constants as shipped in `configs/config.yaml` under `task:`.
# `ConfigContractTests` asserts these match the real configuration, so the values
# used below cannot silently drift from the running system.
GOAL_RADIUS = 0.5
GOAL_REWARD = 10.0
COLLISION_PENALTY = -5.0
STEP_COST = -0.1

# Robot / timing defaults shipped in `configs/config.yaml`.
MAX_LINEAR_VELOCITY = 0.6
MACRO_ACTION_TICKS = 15
DT = 1.0 / 30.0
MACRO_ACTION_DURATION = MACRO_ACTION_TICKS * DT  # 0.5 s
GAMMA = 0.95


def make_states(start_x, start_z, goal_x, goal_z):
    """Build the 15D explicit state rows used by the heuristic."""
    states = torch.zeros((len(start_x), 15), dtype=torch.float32)
    states[:, 0] = torch.tensor(start_x, dtype=torch.float32)
    states[:, 2] = torch.tensor(start_z, dtype=torch.float32)
    states[:, 13] = torch.tensor(goal_x, dtype=torch.float32)
    states[:, 14] = torch.tensor(goal_z, dtype=torch.float32)
    return states


def rewards(dist_to_goal, bumped, **overrides):
    """Call `compute_planning_rewards` with the canonical task constants."""
    kwargs = {
        "goal_radius": GOAL_RADIUS,
        "goal_reward": GOAL_REWARD,
        "collision_penalty": COLLISION_PENALTY,
        "step_cost": STEP_COST,
    }
    kwargs.update(overrides)
    return compute_planning_rewards(dist_to_goal, bumped, **kwargs)


def heuristic(states, is_terminal, gamma=GAMMA, **overrides):
    """Call `compute_straight_line_heuristic` with the canonical constants."""
    kwargs = {
        "max_linear_velocity": MAX_LINEAR_VELOCITY,
        "macro_action_duration": MACRO_ACTION_DURATION,
        "goal_radius": GOAL_RADIUS,
        "goal_reward": GOAL_REWARD,
        "step_cost": STEP_COST,
    }
    kwargs.update(overrides)
    return compute_straight_line_heuristic(states, is_terminal, gamma, **kwargs)


def expected_value(H, gamma=GAMMA):
    """Closed-form heuristic value for a given ideal horizon `H`."""
    if abs(1.0 - gamma) < 1e-9:
        return GOAL_REWARD + H * STEP_COST
    return (gamma ** (H - 1)) * GOAL_REWARD + STEP_COST * ((1.0 - gamma ** H) / (1.0 - gamma))


class ConfigContractTests(unittest.TestCase):
    """The task constants are owned by `configs/config.yaml`.

    `envs.planning_math` deliberately defines no task constants and no defaults,
    so these tests tie the values used everywhere else back to the single
    authoritative configuration.
    """

    def test_task_section_exists_with_expected_values(self):
        task = yaml.safe_load(CONFIG_PATH.read_text())["task"]

        self.assertEqual(task["goal_radius"], GOAL_RADIUS)
        self.assertEqual(task["goal_reward"], GOAL_REWARD)
        self.assertEqual(task["collision_penalty"], COLLISION_PENALTY)
        self.assertEqual(task["step_cost"], STEP_COST)

    def test_macro_action_duration_matches_configured_ticks_and_dt(self):
        """The heuristic's speed estimate must match the real macro-action."""
        config = yaml.safe_load(CONFIG_PATH.read_text())

        self.assertEqual(config["planning"]["macro_action_ticks"], MACRO_ACTION_TICKS)
        # Physics runs at 30 Hz; see RoombaSimulator._setup_simulator.
        self.assertAlmostEqual(DT, 1.0 / 30.0)
        self.assertAlmostEqual(MACRO_ACTION_DURATION, MACRO_ACTION_TICKS * DT)

    def test_robot_speed_limit_matches_config(self):
        config = yaml.safe_load(CONFIG_PATH.read_text())
        self.assertEqual(config["robot"]["max_linear_velocity"], MAX_LINEAR_VELOCITY)

    def test_task_constants_are_not_defined_in_planning_math(self):
        """Guard against reintroducing a second copy of the constants."""
        from envs import planning_math

        for name in ("GOAL_RADIUS_M", "GOAL_REWARD", "COLLISION_PENALTY", "STEP_COST"):
            self.assertFalse(
                hasattr(planning_math, name),
                f"{name} should live in configs/config.yaml, not in planning_math",
            )


class RequiredSignatureTests(unittest.TestCase):
    """Every task constant is required; there are no silent fallback defaults."""

    def test_reward_requires_all_task_constants(self):
        dist = torch.tensor([1.0])
        bumped = torch.tensor([False])

        with self.assertRaises(TypeError):
            compute_planning_rewards(dist, bumped)
        with self.assertRaises(TypeError):
            compute_planning_rewards(dist, bumped, goal_radius=GOAL_RADIUS)

    def test_heuristic_requires_all_task_constants(self):
        states = make_states([0.0], [0.0], [1.0], [0.0])

        with self.assertRaises(TypeError):
            compute_straight_line_heuristic(
                states,
                torch.tensor([False]),
                GAMMA,
                max_linear_velocity=MAX_LINEAR_VELOCITY,
                macro_action_duration=MACRO_ACTION_DURATION,
            )


class RewardTests(unittest.TestCase):
    def test_constants_are_honored_when_overridden(self):
        """Values come from the caller, not from a hidden module default."""
        values = rewards(torch.tensor([0.25]), torch.tensor([False]), goal_reward=1.0, step_cost=0.0)
        self.assertAlmostEqual(values[0].item(), 1.0, places=6)

    def test_nonterminal_without_bump_returns_step_cost(self):
        values = rewards(torch.tensor([1.0]), torch.tensor([False]))
        self.assertAlmostEqual(values[0].item(), -0.1, places=6)

    def test_terminal_without_bump_returns_goal_plus_step_cost(self):
        values = rewards(torch.tensor([0.25]), torch.tensor([False]))
        self.assertAlmostEqual(values[0].item(), 9.9, places=5)

    def test_nonterminal_with_bump_returns_collision_plus_step_cost(self):
        values = rewards(torch.tensor([1.0]), torch.tensor([True]))
        self.assertAlmostEqual(values[0].item(), -5.1, places=5)

    def test_terminal_with_bump_returns_goal_plus_collision_plus_step_cost(self):
        values = rewards(torch.tensor([0.25]), torch.tensor([True]))
        self.assertAlmostEqual(values[0].item(), 4.9, places=5)

    def test_goal_radius_boundary_is_exclusive(self):
        # `dist < goal_radius` is the reached test, so exactly 0.5 is a failure.
        values = rewards(torch.tensor([0.5, 0.499999]), torch.tensor([False, False]))
        self.assertAlmostEqual(values[0].item(), -0.1, places=6)
        self.assertAlmostEqual(values[1].item(), 9.9, places=5)

    def test_batched_combinations_produce_expected_vector(self):
        dist = torch.tensor([0.1, 1.0, 0.2, 2.0])
        bumped = torch.tensor([False, False, True, True])
        expected = [9.9, -0.1, 4.9, -5.1]
        for actual, want in zip(rewards(dist, bumped).tolist(), expected):
            self.assertAlmostEqual(actual, want, places=5)

    def test_output_matches_input_shape_and_device(self):
        dist = torch.tensor([0.1, 1.0], dtype=torch.float32)
        values = rewards(dist, torch.tensor([False, True]))

        self.assertEqual(values.shape, dist.shape)
        self.assertEqual(values.dtype, torch.float32)
        self.assertEqual(values.device, dist.device)

    def test_reward_output_is_float32_even_for_float64_input(self):
        """Pins existing behavior: the masks are cast with `.float()`.

        The original inline expression did the same, so a float64 input has always
        produced a float32 reward. This is existing semantics, not a new choice.
        """
        dist = torch.tensor([0.1, 1.0], dtype=torch.float64)
        values = rewards(dist, torch.tensor([False, True], dtype=torch.bool))

        self.assertEqual(values.dtype, torch.float32)
        self.assertEqual(values.shape, dist.shape)

    def test_matches_original_literal_expression(self):
        """Bit-level agreement with the expression this function replaced."""
        torch.manual_seed(0)
        dist = torch.rand(64) * 3.0
        bumped = torch.rand(64) > 0.5

        reached = dist < 0.5
        original = (reached.float() * 10.0) - (bumped.float() * 5.0) - 0.1
        extracted = rewards(dist, bumped)

        self.assertTrue(torch.equal(original, extracted))


class HeuristicTests(unittest.TestCase):
    def test_terminal_states_return_zero(self):
        states = make_states([0.0, 1.0], [0.0, 1.0], [0.1, 2.0], [0.0, 2.0])
        is_terminal = torch.tensor([True, False])
        values = heuristic(states, is_terminal)

        self.assertEqual(values[0].item(), 0.0)
        self.assertNotEqual(values[1].item(), 0.0)

    def test_one_ideal_step_from_goal_returns_goal_plus_step_cost(self):
        max_dist_per_step = MAX_LINEAR_VELOCITY * MACRO_ACTION_DURATION
        # Exactly one ideal macro-action of remaining distance.
        goal_x = GOAL_RADIUS + max_dist_per_step
        states = make_states([0.0], [0.0], [goal_x], [0.0])
        values = heuristic(states, torch.tensor([False]))

        # H == 1 -> gamma^0 * 10 + (-0.1) * (1 - gamma^1) / (1 - gamma) == 9.9
        expected = GOAL_REWARD + STEP_COST
        self.assertAlmostEqual(values[0].item(), expected, places=5)

    def test_farther_states_have_lower_values(self):
        near = make_states([0.0], [0.0], [1.0], [0.0])
        mid = make_states([0.0], [0.0], [2.0], [0.0])
        far = make_states([0.0], [0.0], [3.0], [0.0])
        not_terminal = torch.tensor([False])

        near_value = heuristic(near, not_terminal)[0].item()
        mid_value = heuristic(mid, not_terminal)[0].item()
        far_value = heuristic(far, not_terminal)[0].item()

        self.assertGreater(near_value, mid_value)
        self.assertGreater(mid_value, far_value)

    def test_inside_goal_radius_floors_horizon_at_one_step(self):
        states = make_states([0.0], [0.0], [0.25], [0.0])
        values = heuristic(states, torch.tensor([False]))
        self.assertAlmostEqual(values[0].item(), GOAL_REWARD + STEP_COST, places=5)

    def test_gamma_one_uses_explicit_limit(self):
        max_dist_per_step = MAX_LINEAR_VELOCITY * MACRO_ACTION_DURATION
        goal_x = GOAL_RADIUS + (3.0 * max_dist_per_step)  # H == 3
        states = make_states([0.0], [0.0], [goal_x], [0.0])
        values = heuristic(states, torch.tensor([False]), gamma=1.0)

        expected = GOAL_REWARD + 3.0 * STEP_COST  # 9.7, no division by zero
        self.assertAlmostEqual(values[0].item(), expected, places=5)

    def test_gamma_near_one_also_uses_the_limit_branch(self):
        max_dist_per_step = MAX_LINEAR_VELOCITY * MACRO_ACTION_DURATION
        goal_x = GOAL_RADIUS + (3.0 * max_dist_per_step)
        states = make_states([0.0], [0.0], [goal_x], [0.0])
        values = heuristic(states, torch.tensor([False]), gamma=1.0 - 1e-12)
        expected = GOAL_REWARD + 3.0 * STEP_COST
        self.assertAlmostEqual(values[0].item(), expected, places=5)

    def test_fractional_horizon_matches_closed_form(self):
        max_dist_per_step = MAX_LINEAR_VELOCITY * MACRO_ACTION_DURATION
        goal_x = GOAL_RADIUS + (2.5 * max_dist_per_step)  # H == 2.5
        states = make_states([0.0], [0.0], [goal_x], [0.0])
        values = heuristic(states, torch.tensor([False]), gamma=GAMMA)

        H = 2.5
        expected = (GAMMA ** (H - 1)) * GOAL_REWARD + STEP_COST * (
            (1.0 - GAMMA ** H) / (1.0 - GAMMA)
        )
        self.assertAlmostEqual(values[0].item(), expected, places=5)
        self.assertNotAlmostEqual(values[0].item(), (GAMMA ** (math.floor(H) - 1)) * GOAL_REWARD, places=3)

    def test_max_distance_per_macro_action_uses_velocity_times_duration(self):
        """Halving `velocity * duration` doubles the estimated horizon."""
        max_dist_per_step = MAX_LINEAR_VELOCITY * MACRO_ACTION_DURATION
        # Exactly 3 ideal macro-actions of remaining distance at the base duration.
        goal_x = GOAL_RADIUS + (3.0 * max_dist_per_step)
        states = make_states([0.0], [0.0], [goal_x], [0.0])
        not_terminal = torch.tensor([False])

        base = heuristic(states, not_terminal)
        doubled = heuristic(states, not_terminal, macro_action_duration=2.0 * MACRO_ACTION_DURATION)

        # Same remaining distance, so doubling the per-step distance halves H.
        self.assertAlmostEqual(base[0].item(), expected_value(3.0), places=5)
        self.assertAlmostEqual(doubled[0].item(), expected_value(1.5), places=5)
        # A shorter horizon means fewer negative step costs, i.e. a higher value.
        self.assertGreater(doubled[0].item(), base[0].item())

    def test_batched_values_have_expected_shape(self):
        states = make_states([0.0, 1.0, 2.0], [0.0, 1.0, 2.0], [3.0, 3.0, 3.0], [0.0, 0.0, 0.0])
        is_terminal = torch.tensor([False, True, False])
        values = heuristic(states, is_terminal)

        self.assertEqual(values.shape, (3,))
        self.assertEqual(values.dtype, torch.float32)

    def test_output_dtype_and_device_follow_state_dtype(self):
        """Unlike the reward, the heuristic has no `.float()` cast in its path."""
        states = make_states([0.0], [0.0], [2.0], [0.0]).to(torch.float64)
        values = heuristic(states, torch.tensor([False]))

        self.assertEqual(values.dtype, torch.float64)
        self.assertEqual(values.device, states.device)

    def test_does_not_mutate_input_tensors(self):
        states = make_states([0.0, 1.0], [0.0, 1.0], [2.0, 2.0], [0.0, 0.0])
        is_terminal = torch.tensor([False, True])
        states_before = states.clone()
        terminal_before = is_terminal.clone()

        heuristic(states, is_terminal)

        self.assertTrue(torch.equal(states, states_before))
        self.assertTrue(torch.equal(is_terminal, terminal_before))

    def test_terminal_entries_do_not_affect_non_terminal_entries(self):
        states = make_states([0.0, 3.0], [0.0, 0.0], [2.0, 0.0], [0.0, 0.0])
        mixed = heuristic(states, torch.tensor([False, True]))
        only_first = heuristic(states[:1], torch.tensor([False]))

        self.assertAlmostEqual(mixed[0].item(), only_first[0].item(), places=6)

    def test_reward_and_heuristic_agree_on_goal_reward_for_a_terminal_step(self):
        """The heuristic's one-step value equals the reward shape on a successful step."""
        max_dist_per_step = MAX_LINEAR_VELOCITY * MACRO_ACTION_DURATION
        goal_x = GOAL_RADIUS + max_dist_per_step
        states = make_states([0.0], [0.0], [goal_x], [0.0])

        value = heuristic(states, torch.tensor([False]))[0].item()
        reward = rewards(torch.tensor([0.25]), torch.tensor([False]))[0].item()

        self.assertAlmostEqual(value, reward, places=5)

    def test_constants_are_honored_when_overridden(self):
        """Values come from the caller, not from a hidden module default."""
        states = make_states([0.0], [0.0], [1.0], [0.0])
        values = heuristic(states, torch.tensor([False]), goal_reward=1.0, step_cost=0.0)
        self.assertLessEqual(values[0].item(), 1.0)


if __name__ == "__main__":
    unittest.main()
