"""Tests for how `envs/planning_env.py` and `envs/rl_env.py` bind configuration.

These tests cover the environment layer only: which configured value reaches which
helper, that the geodesic field is cached per goal, and that `set_states` honors
`task.wheels_reset`. The formulas themselves are covered by
`tests/test_planning_math.py` and `tests/test_distance_field.py`.

CPU-only and Isaac-Gym-free. Instances are built with `object.__new__` so that
`RoombaSimulator` is never constructed; the physics core is replaced by fakes that
supply the tensors and config values each method reads. Minimal stubs are
registered for the `isaacgym` / `gym` names that the modules import at scope, so
this file runs without a GPU install (`AGENTS.md`).
"""

import importlib
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import yaml

# --- Import stubs -----------------------------------------------------------
# `envs/planning_env.py` and `envs/rl_env.py` import `gym.spaces` and pull in
# `core.simulator` (and through it `isaacgym.gymapi`, `isaacgym.gymtorch`). None of
# those names is ever called below, so they only have to resolve for the import to
# succeed.


def _module_available(name: str) -> bool:
    try:
        importlib.import_module(name)
    except ImportError:
        return False
    return True


def _register_import_stubs() -> None:
    """Register the minimal module namespace the environment layer imports."""
    if not _module_available("isaacgym"):
        isaacgym = types.ModuleType("isaacgym")
        isaacgym.gymapi = types.ModuleType("isaacgym.gymapi")
        isaacgym.gymtorch = types.ModuleType("isaacgym.gymtorch")
        sys.modules.setdefault("isaacgym", isaacgym)
        sys.modules.setdefault("isaacgym.gymapi", isaacgym.gymapi)
        sys.modules.setdefault("isaacgym.gymtorch", isaacgym.gymtorch)

    if not _module_available("gym"):
        gym = types.ModuleType("gym")
        gym.spaces = types.ModuleType("gym.spaces")
        sys.modules.setdefault("gym", gym)
        sys.modules.setdefault("gym.spaces", gym.spaces)


_register_import_stubs()

import envs.planning_env as planning_env_module
import envs.rl_env as rl_env_module
from core.room import OccupancyMap, Room
from envs.planning_env import RoombaPlanningEnv
from envs.rl_env import RoombaRLEnv

CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "config.yaml"

DT = 1.0 / 30.0
MACRO_ACTION_TICKS = 15
NUM_ENVS = 3

#: Room extents used for the `room_width` / `room_depth` normalization.
ROOM_WIDTH = 5.0
ROOM_DEPTH = 5.0

#: Non-square cell counts, so a missing transpose in the cached field is visible
#: as a wrong shape rather than a coincidentally correct one.
WIDTH_CELLS = 6
DEPTH_CELLS = 3


def shipped_config():
    return yaml.safe_load(CONFIG_PATH.read_text())


def make_env(cls, **attributes):
    """Build an environment without its simulator-heavy `__init__`.

    Every method under test is pure Python over config values and tensors, so the
    physics core is supplied by the caller as a fake.
    """
    env = object.__new__(cls)
    env.device = torch.device("cpu")
    env.num_envs = NUM_ENVS
    for name, value in attributes.items():
        setattr(env, name, value)
    return env


class RecordingGym:
    """No-op stand-in for the `gymapi.Gym` handle that records the calls made."""

    def __init__(self):
        self.calls = []

    def refresh_net_contact_force_tensor(self, sim):
        self.calls.append("refresh_net_contact_force_tensor")

    def refresh_actor_root_state_tensor(self, sim):
        self.calls.append("refresh_actor_root_state_tensor")

    def end_access_image_tensors(self, sim):
        self.calls.append("end_access_image_tensors")


class FakeOccupancyMap:
    """Records distance-field queries and returns an orientation-sensitive field."""

    def __init__(self, width_cells=WIDTH_CELLS, depth_cells=DEPTH_CELLS):
        self.width_cells = width_cells
        self.depth_cells = depth_cells
        self.calls = []

    def compute_distance_field(self, goal_x, goal_z):
        self.calls.append((goal_x, goal_z))
        # (width_cells, depth_cells), varying along both axes.
        return np.arange(self.width_cells * self.depth_cells, dtype=np.float32).reshape(
            self.width_cells, self.depth_cells
        )


class FakeSimulator:
    """Stand-in for `RoombaSimulator` covering only what these paths touch."""

    def __init__(
        self,
        num_envs=NUM_ENVS,
        contact_forces=None,
        chassis_body_idx=0,
        occupancy_map=None,
        room=None,
        dt=DT,
    ):
        self.gym = RecordingGym()
        self.sim = object()
        self.device = torch.device("cpu")
        self.dt = dt
        self.chassis_body_idx = chassis_body_idx
        self.occupancy_map = occupancy_map if occupancy_map is not None else FakeOccupancyMap()
        self.room = room if room is not None else Room(width=ROOM_WIDTH, depth=ROOM_DEPTH, obstacles=[])
        self.robot_actor_indices = torch.arange(num_envs, dtype=torch.int32)
        self.root_states = torch.zeros((num_envs, 13), dtype=torch.float32)
        self.contact_forces_view = (
            contact_forces
            if contact_forces is not None
            else torch.zeros((num_envs, 1, 3), dtype=torch.float32)
        )
        # Reset / teleport bookkeeping.
        self.reset_dof_calls = 0
        self.set_root_state_calls = 0
        self.sync_graphics_calls = 0

    def set_actor_root_states(self, root_states, actor_ids):
        self.set_root_state_calls += 1

    def reset_dof_states(self, env_ids=None):
        self.reset_dof_calls += 1

    def sync_graphics(self):
        self.sync_graphics_calls += 1


def forces(*rows):
    """`[num_envs, 1, 3]` chassis contact forces from per-env `(x, y, z)` rows."""
    return torch.tensor([[list(row)] for row in rows], dtype=torch.float32)


class PlanningBumperTests(unittest.TestCase):
    """`RoombaPlanningEnv._compute_bumped` is configured, not hard-coded."""

    def build(self, contact_forces, threshold):
        sim = FakeSimulator(contact_forces=contact_forces)
        env = make_env(
            RoombaPlanningEnv,
            sim=sim,
            bumped_threshold=threshold,
        )
        return env, sim

    def test_threshold_comes_from_the_attribute(self):
        """The same forces must flip when `task.bumped_threshold` changes."""
        contact_forces = forces((3.0, 0.0, 0.0), (3.0, 0.0, 0.0))

        permissive, _ = self.build(contact_forces, threshold=2.0)
        strict, _ = self.build(contact_forces, threshold=10.0)

        self.assertTrue(permissive._compute_bumped().all().item())
        self.assertFalse(strict._compute_bumped().any().item())

    def test_vertical_force_alone_is_ignored(self):
        """Regression guard for the resting ground reaction (14.8 N at standstill)."""
        contact_forces = forces((0.0, 14.8, 0.0))
        env, _ = self.build(contact_forces, threshold=2.0)

        self.assertFalse(env._compute_bumped()[0].item())

    def test_contact_forces_are_refreshed_before_reading(self):
        """Reading a stale tensor would report the previous macro-action's contact."""
        env, sim = self.build(forces((0.0, 0.0, 0.0)), threshold=2.0)

        env._compute_bumped()

        self.assertIn("refresh_net_contact_force_tensor", sim.gym.calls)

    def test_mask_has_one_flag_per_env(self):
        contact_forces = forces((0.0, 0.0, 0.0), (20.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        env, _ = self.build(contact_forces, threshold=2.0)

        mask = env._compute_bumped()

        self.assertEqual(mask.shape, (NUM_ENVS,))
        self.assertEqual(mask.dtype, torch.bool)

    def test_chassis_body_index_selects_the_chassis(self):
        """A wrong body index would test a wheel or caster instead of the chassis."""
        contact_forces = torch.zeros((NUM_ENVS, 2, 3), dtype=torch.float32)
        contact_forces[:, 1, 0] = 20.0  # only body 1 is in contact
        sim = FakeSimulator(contact_forces=contact_forces, chassis_body_idx=1)
        env = make_env(RoombaPlanningEnv, sim=sim, bumped_threshold=2.0)

        self.assertTrue(env._compute_bumped().all().item())

        sim.chassis_body_idx = 0
        self.assertFalse(env._compute_bumped().any().item())


class DistanceFieldCacheTests(unittest.TestCase):
    """The Dijkstra field is recomputed only when the goal actually changes."""

    #: The tolerance `_get_or_update_distance_field` uses, mirrored here.
    TOLERANCE = 1e-4

    def build(self):
        occupancy_map = FakeOccupancyMap()
        sim = FakeSimulator(occupancy_map=occupancy_map)
        env = make_env(
            RoombaPlanningEnv,
            sim=sim,
            _cached_goal=None,
            _cached_distance_field=None,
        )
        return env, occupancy_map

    def test_shape_is_depth_by_width(self):
        """`grid_sample` maps its height axis to Z, so the cache must be transposed."""
        env, occupancy_map = self.build()

        field = env._get_or_update_distance_field(1.0, -0.5)

        self.assertEqual(field.shape, (1, 1, DEPTH_CELLS, WIDTH_CELLS))
        expected = torch.from_numpy(occupancy_map.compute_distance_field(1.0, -0.5).T.copy())
        self.assertTrue(
            torch.equal(field[0, 0], expected),
            "cached field content does not match the transposed map output",
        )

    def test_field_is_float32_on_the_env_device(self):
        env, _ = self.build()

        field = env._get_or_update_distance_field(0.0, 0.0)

        self.assertEqual(field.dtype, torch.float32)
        self.assertEqual(field.device, env.device)

    def test_repeated_goal_queries_hit_the_cache(self):
        env, occupancy_map = self.build()

        first = env._get_or_update_distance_field(1.0, 2.0)
        second = env._get_or_update_distance_field(1.0, 2.0)

        self.assertEqual(len(occupancy_map.calls), 1)
        self.assertIs(first, second)

    def test_goal_within_tolerance_reuses_the_field(self):
        env, occupancy_map = self.build()
        env._get_or_update_distance_field(1.0, 2.0)

        env._get_or_update_distance_field(1.0 + (self.TOLERANCE / 2.0), 2.0)

        self.assertEqual(len(occupancy_map.calls), 1)

    def test_goal_beyond_tolerance_recomputes(self):
        env, occupancy_map = self.build()
        env._get_or_update_distance_field(1.0, 2.0)

        env._get_or_update_distance_field(1.0 + (self.TOLERANCE * 10.0), 2.0)

        self.assertEqual(len(occupancy_map.calls), 2)

    def test_each_axis_is_compared_independently(self):
        """A change in either coordinate alone must invalidate the cache."""
        env, occupancy_map = self.build()
        env._get_or_update_distance_field(0.0, 0.0)

        env._get_or_update_distance_field(0.0, 1.0)
        env._get_or_update_distance_field(1.0, 1.0)

        self.assertEqual(occupancy_map.calls, [(0.0, 0.0), (0.0, 1.0), (1.0, 1.0)])


class HeuristicBindingTests(unittest.TestCase):
    """`compute_heuristic_values` forwards the field and the configured constants."""

    def setUp(self):
        self.config = shipped_config()
        self.occupancy_map = FakeOccupancyMap()
        self.sim = FakeSimulator(occupancy_map=self.occupancy_map)
        self.env = make_env(
            RoombaPlanningEnv,
            sim=self.sim,
            config=self.config,
            macro_action_ticks=MACRO_ACTION_TICKS,
            goal_radius=self.config["task"]["goal_radius"],
            goal_reward=self.config["task"]["goal_reward"],
            step_cost=self.config["task"]["step_cost"],
            heuristic_sampling=self.config["task"]["heuristic_sampling"],
            _cached_goal=None,
            _cached_distance_field=None,
        )
        self.states = torch.zeros((NUM_ENVS, 15), dtype=torch.float32)
        self.states[:, 13] = 1.5
        self.states[:, 14] = -0.5
        self.is_terminal = torch.zeros(NUM_ENVS, dtype=torch.bool)

    def call_and_capture(self):
        """Run the binding with the pure function replaced by a recorder."""
        captured = {}

        def fake_heuristic(states, is_terminal, gamma, **kwargs):
            captured["states"] = states
            captured["is_terminal"] = is_terminal
            captured["gamma"] = gamma
            captured["kwargs"] = kwargs
            return torch.zeros(states.shape[0], dtype=torch.float32)

        with mock.patch.object(
            planning_env_module, "compute_straight_line_heuristic", fake_heuristic
        ):
            values = self.env.compute_heuristic_values(self.states, self.is_terminal, 0.95)

        return captured, values

    def test_field_is_built_for_the_goal_in_the_state(self):
        """Goals live in state columns 13/14, in room-local coordinates."""
        self.call_and_capture()

        self.assertEqual(self.occupancy_map.calls, [(1.5, -0.5)])

    def test_call_arguments_match_the_configuration(self):
        captured, _ = self.call_and_capture()
        kwargs = captured["kwargs"]
        robot = self.config["robot"]

        self.assertEqual(captured["gamma"], 0.95)
        self.assertEqual(kwargs["max_linear_velocity"], robot["max_linear_velocity"])
        self.assertEqual(kwargs["goal_radius"], self.config["task"]["goal_radius"])
        self.assertEqual(kwargs["goal_reward"], self.config["task"]["goal_reward"])
        self.assertEqual(kwargs["step_cost"], self.config["task"]["step_cost"])
        self.assertEqual(kwargs["sampling_mode"], self.config["task"]["heuristic_sampling"])
        self.assertAlmostEqual(kwargs["macro_action_duration"], MACRO_ACTION_TICKS * DT)

    def test_room_extent_is_taken_from_the_room(self):
        """Normalization uses the real extents, not the cell counts."""
        captured, _ = self.call_and_capture()

        self.assertEqual(captured["kwargs"]["room_width"], self.sim.room.width)
        self.assertEqual(captured["kwargs"]["room_depth"], self.sim.room.depth)

    def test_the_transposed_memoized_field_is_passed_through(self):
        captured, _ = self.call_and_capture()

        field = captured["kwargs"]["distance_field"]
        self.assertEqual(field.shape, (1, 1, DEPTH_CELLS, WIDTH_CELLS))
        self.assertIs(field, self.env._cached_distance_field)

    def test_the_batch_is_forwarded_unchanged(self):
        captured, values = self.call_and_capture()

        self.assertIs(captured["states"], self.states)
        self.assertIs(captured["is_terminal"], self.is_terminal)
        self.assertEqual(values.shape, (NUM_ENVS,))

    def test_transpose_matches_a_real_occupancy_map(self):
        """Pin the axis mapping against a real map over a non-square room."""
        room = Room(width=6.0, depth=3.0, obstacles=[])
        real_map = OccupancyMap(
            room, resolution=0.5, robot_radius=0.17, safety_margin=0.05, min_start_goal_dist=1.0
        )
        env = make_env(
            RoombaPlanningEnv,
            sim=FakeSimulator(occupancy_map=real_map, room=room),
            _cached_goal=None,
            _cached_distance_field=None,
        )

        field = env._get_or_update_distance_field(0.0, 0.0)

        self.assertEqual((real_map.width_cells, real_map.depth_cells), (12, 6))
        # The map is (width, depth); the cache must flip it to (depth, width).
        self.assertEqual(field.shape, (1, 1, real_map.depth_cells, real_map.width_cells))
        expected = torch.from_numpy(real_map.compute_distance_field(0.0, 0.0).T.copy())
        self.assertTrue(torch.equal(field[0, 0], expected))


class SetStatesTests(unittest.TestCase):
    """`set_states` is the teleport half of `G(s, a)`, so its inputs are pinned."""

    def build(self, wheels_reset):
        sim = FakeSimulator()
        env = make_env(
            RoombaPlanningEnv,
            sim=sim,
            goals=torch.zeros((NUM_ENVS, 2), dtype=torch.float32),
            wheels_reset=wheels_reset,
        )
        states = torch.zeros((NUM_ENVS, 15), dtype=torch.float32)
        states[:, 0] = torch.tensor([1.0, 2.0, 3.0])
        states[:, 13] = torch.tensor([9.0, 9.0, 9.0])
        states[:, 14] = torch.tensor([-9.0, -9.0, -9.0])
        return env, sim, states

    def test_wheels_reset_true_zeroes_the_wheel_dof_state(self):
        """Wheel DOF state is not in the 15D state, so it must be reset explicitly."""
        env, sim, states = self.build(wheels_reset=True)

        env.set_states(states)

        self.assertEqual(sim.reset_dof_calls, 1)

    def test_wheels_reset_false_skips_the_reset(self):
        env, sim, states = self.build(wheels_reset=False)

        env.set_states(states)

        self.assertEqual(sim.reset_dof_calls, 0)

    def test_root_state_columns_are_written_and_pushed_to_the_physics_engine(self):
        env, sim, states = self.build(wheels_reset=True)

        env.set_states(states)

        self.assertTrue(torch.equal(sim.root_states[:, 0], torch.tensor([1.0, 2.0, 3.0])))
        self.assertEqual(sim.set_root_state_calls, 1)

    def test_goals_are_taken_from_the_state(self):
        """The planner's goal lives in the state, not in a separate tensor."""
        env, _, states = self.build(wheels_reset=True)

        env.set_states(states)

        self.assertTrue(torch.equal(env.goals[:, 0], torch.full((NUM_ENVS,), 9.0)))
        self.assertTrue(torch.equal(env.goals[:, 1], torch.full((NUM_ENVS,), -9.0)))

    def test_graphics_are_synced_after_the_teleport(self):
        """Visual sensors would otherwise render the pre-teleport pose."""
        env, sim, states = self.build(wheels_reset=True)

        env.set_states(states)

        self.assertEqual(sim.sync_graphics_calls, 1)


class RlEnvBumperTests(unittest.TestCase):
    """`RoombaRLEnv` shares the bumper threshold and the horizontal-only test."""

    def build(self, contact_forces, threshold=None, config=None):
        config = config if config is not None else shipped_config()
        config["sensors"]["enable_bumper"] = True
        config["sensors"]["enable_lidar"] = False
        config["sensors"]["enable_camera"] = False
        sim = FakeSimulator(contact_forces=contact_forces)
        env = make_env(
            RoombaRLEnv,
            sim=sim,
            config=config,
            bumped_threshold=(
                threshold if threshold is not None else config["task"]["bumped_threshold"]
            ),
        )
        return env

    def test_bumper_observation_shape_and_dtype(self):
        env = self.build(forces((20.0, 0.0, 0.0), (0.0, 20.0, 0.0), (0.0, 0.0, 0.0)))

        obs = env._compute_observations()

        self.assertEqual(obs["bumper"].shape, (NUM_ENVS, 1))
        self.assertEqual(obs["bumper"].dtype, torch.float32)

    def test_vertical_force_alone_is_ignored(self):
        """Same regression as the planning env; both must exclude world Y."""
        env = self.build(forces((0.0, 14.8, 0.0)))

        obs = env._compute_observations()

        self.assertEqual(obs["bumper"][0].item(), 0.0)

    def test_configured_threshold_is_applied(self):
        env = self.build(forces((3.0, 0.0, 0.0)), threshold=10.0)

        obs = env._compute_observations()

        self.assertEqual(obs["bumper"][0].item(), 0.0)

    def test_threshold_comes_from_the_task_section(self):
        """The key is read at construction, so the same forces flip with it."""
        config = shipped_config()
        config["task"]["bumped_threshold"] = 25.0
        env = self.build(forces((20.0, 0.0, 0.0)), config=config)

        obs = env._compute_observations()

        self.assertEqual(obs["bumper"][0].item(), 0.0)

    def test_both_environments_agree_on_the_same_forces(self):
        """Guards against the two copies of the bumper test drifting apart."""
        contact_forces = forces((0.0, 14.8, 0.0), (5.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        config = shipped_config()
        threshold = config["task"]["bumped_threshold"]

        planning_env = make_env(
            RoombaPlanningEnv,
            sim=FakeSimulator(contact_forces=contact_forces),
            bumped_threshold=threshold,
        )
        rl_env = self.build(contact_forces, threshold=threshold, config=config)

        planning_mask = planning_env._compute_bumped()
        rl_mask = rl_env._compute_observations()["bumper"].squeeze(-1) > 0.5

        self.assertTrue(torch.equal(planning_mask, rl_mask))


if __name__ == "__main__":
    unittest.main()
