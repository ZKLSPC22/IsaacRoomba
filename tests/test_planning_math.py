"""Tests for `envs.planning_math`.

Pins the reward, bumper, and leaf-heuristic formulas, plus the geodesic sampling
contract the planner actually runs with. Imports `torch` (and `numpy` for an
independent sampling reference), so it runs on CPU without Isaac Gym.

`ConfigContractTests` ties every value repeated below back to the real
`configs/config.yaml`, so the constants here cannot silently drift from the
running system.
"""

import math
import sys
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import yaml

from envs.planning_math import (
    compute_bumped,
    compute_planning_rewards,
    compute_straight_line_heuristic,
    VALID_SAMPLING_MODES,
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

# The sampling mode shipped in `configs/config.yaml` under `task.heuristic_sampling`.
# `ConfigContractTests` asserts it matches, so the tests cannot silently exercise a
# different interpolation than the one the planner runs with.
HEURISTIC_SAMPLING = "bilinear"

# Bumper sensitivity in newtons, from `configs/config.yaml` under
# `task.bumped_threshold`. `ConfigContractTests` asserts it matches.
BUMPED_THRESHOLD = 2.0


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
        "sampling_mode": HEURISTIC_SAMPLING,
    }
    kwargs.update(overrides)
    return compute_straight_line_heuristic(states, is_terminal, gamma, **kwargs)


def expected_value(H, gamma=GAMMA):
    """Closed-form heuristic value for a given ideal horizon `H`."""
    if abs(1.0 - gamma) < 1e-9:
        return GOAL_REWARD + H * STEP_COST
    return (gamma ** (H - 1)) * GOAL_REWARD + STEP_COST * ((1.0 - gamma ** H) / (1.0 - gamma))


# --- Geodesic distance-field fixtures ---------------------------------------
# `compute_straight_line_heuristic` samples this field instead of measuring a
# straight-line distance when `distance_field`, `room_width` and `room_depth` are
# all supplied.

#: 50 x 50 cells over a 5.0 m x 5.0 m room, i.e. 0.1 m cells.
GEODESIC_ROOM_SIZE = 5.0
GEODESIC_CELLS = 50
GEODESIC_RESOLUTION = GEODESIC_ROOM_SIZE / GEODESIC_CELLS

#: Grid rows/cols overwritten with a large finite value, standing in for the
#: inflated obstacle margin: `[gz, gx]` indices covering roughly x in
#: [-1.5, -0.5] m and z in [0.5, 1.5] m. Real fields stay finite everywhere
#: (`core/room.py` inherits the nearest reachable cell), so a large finite value
#: is the realistic stand-in.
GEODESIC_OBSTACLE_ROWS = slice(30, 40)
GEODESIC_OBSTACLE_COLS = slice(10, 20)


def geodesic_field_np() -> np.ndarray:
    """Synthetic `(depth_cells, width_cells)` distance field in metres.

    The per-axis slopes differ (0.8 m per metre along X, 0.2 along Z), so a
    transposed tensor or a swapped sampling axis changes the sampled numbers.
    """
    index = np.arange(GEODESIC_CELLS)
    x = (index + 0.5) * GEODESIC_RESOLUTION - GEODESIC_ROOM_SIZE / 2.0
    z = (index + 0.5) * GEODESIC_RESOLUTION - GEODESIC_ROOM_SIZE / 2.0

    field = (0.8 * x)[np.newaxis, :] + (0.2 * z)[:, np.newaxis] + 3.0
    field[GEODESIC_OBSTACLE_ROWS, GEODESIC_OBSTACLE_COLS] = float(field.max()) + 2.0
    return field


def geodesic_field_tensor() -> torch.Tensor:
    """The same field as the `[1, 1, depth_cells, width_cells]` planner tensor."""
    field = geodesic_field_np()
    return torch.tensor(field, dtype=torch.float32).view(1, 1, *field.shape)


def sample_geodesic_field(field: np.ndarray, x: float, z: float) -> float:
    """Independent NumPy reference for the documented `grid_sample` convention.

    `align_corners=True` maps normalized `-1`/`+1` onto grid indices `0` and
    `size - 1`, and `padding_mode="border"` clamps anything outside the room, so
    this bilinear lookup is the value the planner must reproduce.
    """
    height, width = field.shape  # (depth, width)
    fx = ((x / (GEODESIC_ROOM_SIZE / 2.0)) + 1.0) * 0.5 * (width - 1)
    fz = ((z / (GEODESIC_ROOM_SIZE / 2.0)) + 1.0) * 0.5 * (height - 1)
    fx = min(max(fx, 0.0), width - 1)
    fz = min(max(fz, 0.0), height - 1)

    x0, z0 = math.floor(fx), math.floor(fz)
    x1, z1 = min(x0 + 1, width - 1), min(z0 + 1, height - 1)
    wx, wz = fx - x0, fz - z0

    return float(
        field[z0, x0] * (1.0 - wx) * (1.0 - wz)
        + field[z0, x1] * wx * (1.0 - wz)
        + field[z1, x0] * (1.0 - wx) * wz
        + field[z1, x1] * wx * wz
    )


def expected_geodesic_value(sampled_distance: float, gamma=GAMMA) -> float:
    """Heuristic value implied by one sampled geodesic distance."""
    max_dist_per_step = MAX_LINEAR_VELOCITY * MACRO_ACTION_DURATION
    d_remain = max(sampled_distance - GOAL_RADIUS, 0.0)
    H = max(d_remain / max_dist_per_step, 1.0)
    return expected_value(H, gamma)


def nearest_cell_value(field: np.ndarray, x: float, z: float) -> float:
    """Independent NumPy reference for `grid_sample(mode="nearest")`.

    Reads the field value at the containing cell centre, with no interpolation.
    """
    height, width = field.shape  # (depth, width)
    fx = ((x / (GEODESIC_ROOM_SIZE / 2.0)) + 1.0) * 0.5 * (width - 1)
    fz = ((z / (GEODESIC_ROOM_SIZE / 2.0)) + 1.0) * 0.5 * (height - 1)
    fx = min(max(fx, 0.0), width - 1)
    fz = min(max(fz, 0.0), height - 1)
    return float(field[round(fz), round(fx)])


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

    def test_shipped_heuristic_sampling_mode_is_supported(self):
        """The configured interpolation must be one the heuristic accepts."""
        task = yaml.safe_load(CONFIG_PATH.read_text())["task"]

        self.assertEqual(task["heuristic_sampling"], HEURISTIC_SAMPLING)
        self.assertIn(HEURISTIC_SAMPLING, VALID_SAMPLING_MODES)

    def test_bumped_threshold_matches_config(self):
        """Both environments read this one key, so it must be the pinned value."""
        task = yaml.safe_load(CONFIG_PATH.read_text())["task"]
        self.assertEqual(task["bumped_threshold"], BUMPED_THRESHOLD)

    def test_wheels_reset_is_configured(self):
        """`task.wheels_reset` gates the Markov wheel-DOF reset in `set_states`."""
        task = yaml.safe_load(CONFIG_PATH.read_text())["task"]
        self.assertIsInstance(task["wheels_reset"], bool)

    def test_solver_iterations_are_configured(self):
        """Required PhysX keys, direct-indexed by `RoombaSimulator`."""
        simulation = yaml.safe_load(CONFIG_PATH.read_text())["simulation"]

        self.assertGreaterEqual(simulation["num_position_iterations"], 1)
        self.assertGreaterEqual(simulation["num_velocity_iterations"], 1)

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

    def test_heuristic_rejects_an_unknown_sampling_mode(self):
        """A typo in task.heuristic_sampling must fail, not fall back silently."""
        states = make_states([0.0], [0.0], [1.0], [0.0])

        with self.assertRaises(ValueError):
            heuristic(states, torch.tensor([False]), sampling_mode="bicubic")


class BumperTests(unittest.TestCase):
    """The bumper reads the horizontal contact force only.

    Regression guard: the mask used the full force norm, so world Y -- which
    carries the ground reaction and reaches 14.8 N while the robot stands still --
    made it fire on every macro-action regardless of collisions.
    """

    def bump(self, forces, threshold=BUMPED_THRESHOLD):
        return compute_bumped(torch.tensor(forces, dtype=torch.float32), threshold=threshold)

    def test_vertical_force_alone_is_not_a_bump(self):
        """The resting ground reaction must not register as a collision."""
        self.assertFalse(self.bump([[0.0, 14.8, 0.0]])[0].item())

    def test_world_x_force_is_a_bump(self):
        self.assertTrue(self.bump([[15.0, 0.0, 0.0]])[0].item())

    def test_world_z_force_is_a_bump(self):
        self.assertTrue(self.bump([[0.0, 0.0, 15.0]])[0].item())

    def test_threshold_is_exclusive(self):
        """`> threshold`, matching the reward's exclusive `< goal_radius` test."""
        at_threshold = self.bump([[BUMPED_THRESHOLD, 0.0, 0.0]])
        just_above = self.bump([[BUMPED_THRESHOLD + 1e-3, 0.0, 0.0]])

        self.assertFalse(at_threshold[0].item())
        self.assertTrue(just_above[0].item())

    def test_both_horizontal_axes_contribute_to_one_norm(self):
        """X and Z combine as a magnitude, not a per-axis maximum."""
        # 1.6 N on each axis is 2.26 N combined: a bump, though neither axis
        # alone clears the 2.0 N threshold.
        self.assertTrue(self.bump([[1.6, 0.0, 1.6]])[0].item())

    def test_free_space_noise_floor_is_not_a_bump(self):
        """The measured horizontal peak in free space is 2e-6 N."""
        self.assertFalse(self.bump([[2e-6, 0.0, -2e-6]])[0].item())

    def test_batch_shape_and_dtype(self):
        mask = self.bump([[0.0, 20.0, 0.0], [20.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

        self.assertEqual(mask.shape, (3,))
        self.assertEqual(mask.dtype, torch.bool)

    def test_threshold_is_required(self):
        """No hidden default: the owner is `task.bumped_threshold`."""
        with self.assertRaises(TypeError):
            compute_bumped(torch.zeros((1, 3)))


class GeodesicArgumentTests(unittest.TestCase):
    """The two distance sources are mutually exclusive and behave differently."""

    def setUp(self):
        self.field = geodesic_field_tensor()
        self.not_terminal = torch.tensor([False])

    def test_partial_geodesic_arguments_raise(self):
        """A partial set would silently change which distance is optimized."""
        states = make_states([0.0], [0.0], [1.0], [0.0])
        partial_sets = {
            "field only": {"distance_field": self.field},
            "width only": {"room_width": GEODESIC_ROOM_SIZE},
            "depth only": {"room_depth": GEODESIC_ROOM_SIZE},
            "field and width": {
                "distance_field": self.field,
                "room_width": GEODESIC_ROOM_SIZE,
            },
            "field and depth": {
                "distance_field": self.field,
                "room_depth": GEODESIC_ROOM_SIZE,
            },
            "width and depth": {
                "room_width": GEODESIC_ROOM_SIZE,
                "room_depth": GEODESIC_ROOM_SIZE,
            },
        }

        for name, kwargs in partial_sets.items():
            with self.subTest(partial=name), self.assertRaises(ValueError):
                heuristic(states, self.not_terminal, **kwargs)

    def test_omitting_all_three_uses_the_straight_line_distance(self):
        """The state's own goal columns are then the only distance source."""
        max_dist_per_step = MAX_LINEAR_VELOCITY * MACRO_ACTION_DURATION
        states = make_states([0.0], [0.0], [GOAL_RADIUS + (3.0 * max_dist_per_step)], [0.0])

        values = heuristic(states, self.not_terminal)

        self.assertAlmostEqual(values[0].item(), expected_value(3.0), places=5)

    def test_straight_line_mode_ignores_the_goal_columns_only_used_by_it(self):
        """Geodesic mode takes its distance from the field, not the state."""
        # Same pose, a goal at 2 m and a goal at 4 m: straight-line mode must
        # differ, geodesic mode must not (the field was built for one goal).
        near_goal = make_states([0.0], [0.0], [GOAL_RADIUS + 2.0], [0.0])
        far_goal = make_states([0.0], [0.0], [GOAL_RADIUS + 4.0], [0.0])
        geodesic_kwargs = {
            "distance_field": self.field,
            "room_width": GEODESIC_ROOM_SIZE,
            "room_depth": GEODESIC_ROOM_SIZE,
        }

        straight_near = heuristic(near_goal, self.not_terminal)[0].item()
        straight_far = heuristic(far_goal, self.not_terminal)[0].item()
        geodesic_near = heuristic(near_goal, self.not_terminal, **geodesic_kwargs)[0].item()
        geodesic_far = heuristic(far_goal, self.not_terminal, **geodesic_kwargs)[0].item()

        self.assertNotAlmostEqual(straight_near, straight_far, places=3)
        self.assertAlmostEqual(geodesic_near, geodesic_far, places=6)


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


class GeodesicHeuristicTests(unittest.TestCase):
    """The leaf heuristic samples a precomputed 2D distance field.

    `distance_field` is shaped `[1, 1, depth_cells, width_cells]`, matching the
    room-local `[gx, gz]` occupancy grid transposed so that the height axis is Z.
    """

    def setUp(self):
        self.field = geodesic_field_np()
        self.field_tensor = geodesic_field_tensor()

    def call_heuristic(self, states, is_terminal, **overrides):
        kwargs = {
            "distance_field": self.field_tensor,
            "room_width": GEODESIC_ROOM_SIZE,
            "room_depth": GEODESIC_ROOM_SIZE,
        }
        kwargs.update(overrides)
        return heuristic(states, is_terminal, **kwargs)

    def test_geodesic_heuristic_sampling(self):
        # Room-local query poses: free space, the obstacle margin, and a
        # duplicate pose that is terminal (so its value must be exactly 0).
        # The free poses stay clear of the obstacle block's rows, otherwise the
        # bilinear lookup would blend the ramp with the margin value.
        queries = [
            (0.0, 0.0),
            (1.0, -1.0),
            (-1.5, -1.75),
            (0.0, 0.0),
            (-1.0, 1.0),
        ]
        is_terminal = torch.tensor([False, False, False, True, False])
        states = make_states(
            [x for x, _ in queries],
            [z for _, z in queries],
            [GOAL_RADIUS] * len(queries),
            [0.0] * len(queries),
        )

        values = self.call_heuristic(states, is_terminal)
        self.assertEqual(values.shape, (len(queries),))

        for i, (x, z) in enumerate(queries):
            with self.subTest(pose=(x, z)):
                # No NaN/inf anywhere, including inside the obstacle margin.
                self.assertTrue(
                    torch.isfinite(values[i]).item(),
                    msg=f"value at ({x}, {z}) was {values[i].item()}",
                )
                if is_terminal[i].item():
                    self.assertEqual(values[i].item(), 0.0)
                    continue

                sampled = sample_geodesic_field(self.field, x, z)
                self.assertAlmostEqual(
                    values[i].item(), expected_geodesic_value(sampled), places=4
                )

        # Monotone in the sampled distance: nearer in the field means a higher
        # value. The samples are 1.48 m, 3.00 m, 3.59 m and 7.45 m, so a
        # transposed or swapped sampling axis breaks this ordering.
        self.assertGreater(values[2].item(), values[0].item())
        self.assertGreater(values[0].item(), values[1].item())
        self.assertGreater(values[1].item(), values[4].item())

        # A pose outside the room is clamped to the border, not turned into NaN.
        outside = make_states([10.0], [10.0], [GOAL_RADIUS], [0.0])
        outside_values = self.call_heuristic(outside, torch.tensor([False]))
        self.assertTrue(torch.isfinite(outside_values[0]).item())

    def test_sampling_uses_the_field_not_the_state_distance(self):
        """The two distance sources disagree here, so they cannot be confused."""
        # The field reads 3.05 m at (0.0, 0.0); the straight line to the goal in
        # the state is ~1.25 m.
        states = make_states([0.0], [0.0], [GOAL_RADIUS + 1.25], [0.0])
        not_terminal = torch.tensor([False])

        geodesic = self.call_heuristic(states, not_terminal)[0].item()
        straight = heuristic(states, not_terminal)[0].item()

        sampled = sample_geodesic_field(self.field, 0.0, 0.0)
        self.assertAlmostEqual(geodesic, expected_geodesic_value(sampled), places=4)
        self.assertNotAlmostEqual(geodesic, straight, places=3)

    def test_nearest_mode_quantizes_to_the_containing_cell(self):
        """`nearest` reproduces the pre-toggle sampling instead of blending."""
        # (0.09, 0.05) sits inside the cell centred at (0.05, 0.05), well away
        # from that centre, so bilinear and nearest must disagree.
        states = make_states([0.09], [0.05], [GOAL_RADIUS], [0.0])
        not_terminal = torch.tensor([False])

        nearest = self.call_heuristic(states, not_terminal, sampling_mode="nearest")[0].item()
        bilinear = self.call_heuristic(states, not_terminal, sampling_mode="bilinear")[0].item()

        self.assertAlmostEqual(
            nearest,
            expected_geodesic_value(nearest_cell_value(self.field, 0.09, 0.05)),
            places=4,
        )
        # Bilinear blends the neighbouring cells; the fixture field rises with
        # +x, so the blend lands on a larger distance, i.e. a lower value.
        self.assertLess(bilinear, nearest)

    def test_out_of_room_poses_clamp_under_both_modes(self):
        """`padding_mode="border"` must not depend on the interpolation mode."""
        outside = make_states([10.0], [10.0], [GOAL_RADIUS], [0.0])

        for mode in VALID_SAMPLING_MODES:
            with self.subTest(mode=mode):
                values = self.call_heuristic(outside, torch.tensor([False]), sampling_mode=mode)
                self.assertTrue(torch.isfinite(values[0]).item())


if __name__ == "__main__":
    unittest.main()
