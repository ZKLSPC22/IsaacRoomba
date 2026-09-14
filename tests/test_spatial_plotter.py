"""Contract tests for `tracking.spatial_plotter.TrajectoryVisualizer`.

These tests are strictly CPU-only: the `Agg` backend is selected before
`matplotlib.pyplot` is imported, and neither Isaac Gym, CUDA, nor the simulator
is imported. Every artifact is written into a `tempfile.mkdtemp()` directory
that is removed in `tearDown`.
"""

import matplotlib

matplotlib.use("Agg")

import math
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import matplotlib.axes
import matplotlib.pyplot as plt
import yaml

sys.path.append(str(Path(__file__).resolve().parent.parent))

from tracking.spatial_plotter import (
    EMPTY_ROOM_BOUNDS,
    STANDARD_ROOM_BOUNDS,
    TrajectoryVisualizer,
)

# PNG file signature as defined by the PNG specification (RFC 2083).
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


class TestTrajectoryVisualizerContract(unittest.TestCase):
    """Public contract of the headless 2D trajectory visualizer."""

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.room_bounds = (10.0, 10.0)
        self.obstacles = [{"x": 0.0, "z": 0.0, "width": 2.0, "depth": 2.0}]
        self.goal = (3.0, 3.0)
        self.goal_radius = 0.5
        self.robot_radius = 0.17
        self.safety_margin = 0.05

    def tearDown(self):
        plt.close("all")
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _make_visualizer(self):
        return TrajectoryVisualizer(
            output_dir=self.temp_dir,
            room_bounds=self.room_bounds,
            obstacles=self.obstacles,
            goal=self.goal,
            goal_radius=self.goal_radius,
            robot_radius=self.robot_radius,
            safety_margin=self.safety_margin,
        )

    def _write_run_log(self, obstacles=None, room_type="custom", room_extent=(5.0, 4.0)):
        """Write a minimal `run.yaml` shaped like the real logger's output."""
        environment = {
            "task": {"goal_radius": 0.4},
            "robot": {"radius": 0.2},
            "room": {
                "type": room_type,
                "custom": {
                    "width": room_extent[0],
                    "depth": room_extent[1],
                    "obstacles": obstacles if obstacles is not None else [],
                },
            },
        }
        metadata = {
            "schema_version": 1,
            "run_id": "mock_run",
            "scenario": {"seed": 0, "goal": {"x": 1.25, "z": -0.75}},
            "configuration": {"environment": environment},
        }
        (self.temp_dir / "run.yaml").write_text(yaml.safe_dump(metadata, sort_keys=False))
        return metadata

    def _write_steps_csv(self, rows=None, fieldnames=None):
        rows = rows if rows is not None else [
            {
                "step": 0,
                "robot_x": -1.0,
                "robot_z": -0.5,
                "robot_yaw": 0.0,
                "distance_to_goal_m": 2.5,
                "root_value": 0.0,
                "tree_size": 1,
            },
            {
                "step": 1,
                "robot_x": -0.5,
                "robot_z": 0.0,
                "robot_yaw": 0.5,
                "distance_to_goal_m": 1.9,
                "root_value": -0.25,
                "tree_size": 31,
            },
        ]
        fieldnames = fieldnames or list(rows[0].keys())
        lines = [",".join(fieldnames)]
        for row in rows:
            lines.append(",".join(str(row[name]) for name in fieldnames))
        csv_path = self.temp_dir / "steps.csv"
        csv_path.write_text("\n".join(lines) + "\n")
        return csv_path

    def _assert_is_valid_png(self, frame_path):
        """A rendered frame must be a non-empty file with a PNG signature."""
        self.assertTrue(frame_path.exists(), f"missing frame: {frame_path}")
        self.assertGreater(frame_path.stat().st_size, 0, f"empty frame: {frame_path}")
        with open(frame_path, "rb") as handle:
            self.assertEqual(handle.read(8), PNG_MAGIC, f"not a PNG: {frame_path}")

    def test_frame_file_creation_and_format(self):
        visualizer = self._make_visualizer()

        frame_path = visualizer.render_step(step=0, robot_pose=(-1.0, -1.0, 0.0))

        self.assertTrue(frame_path.exists())
        self.assertEqual(frame_path.parent.name, "frames")
        self.assertEqual(frame_path.parent.parent, self.temp_dir)
        self.assertEqual(frame_path.name, "step_000.png")
        self.assertGreater(frame_path.stat().st_size, 0)

        with open(frame_path, "rb") as handle:
            self.assertEqual(handle.read(8), PNG_MAGIC)

    def test_kinematic_history_tracking(self):
        visualizer = self._make_visualizer()
        poses = [
            (-2.0, -2.0, 0.0),
            (-1.5, -1.8, 0.2),
            (-1.0, -1.5, 0.5),
            (-0.5, -1.0, 0.8),
        ]

        for step, pose in enumerate(poses):
            visualizer.render_step(step=step, robot_pose=pose)

        self.assertEqual(len(visualizer.history_x), 4)
        self.assertEqual(len(visualizer.history_z), 4)
        for index, (x, z, _yaw) in enumerate(poses):
            self.assertTrue(math.isclose(visualizer.history_x[index], x, abs_tol=1e-5))
            self.assertTrue(math.isclose(visualizer.history_z[index], z, abs_tol=1e-5))

        for step in range(4):
            self._assert_is_valid_png(visualizer.output_dir / f"step_{step:03d}.png")

    def test_boundary_and_out_of_bounds_resilience(self):
        visualizer = self._make_visualizer()
        # On-perimeter poses plus poses well outside the 10 m x 10 m room: an
        # out-of-bounds pose must degrade visually, never raise or clip the axes.
        poses = [
            (5.0, 5.0, 0.0),
            (-5.0, -5.0, 0.0),
            (12.0, 0.0, 0.0),
            (0.0, -15.0, 0.0),
        ]

        for step, pose in enumerate(poses):
            try:
                frame_path = visualizer.render_step(step=step, robot_pose=pose)
            except (ValueError, IndexError, ArithmeticError) as exc:  # pragma: no cover
                self.fail(f"render_step raised for pose {pose}: {exc!r}")
            self._assert_is_valid_png(frame_path)

    def test_resource_cleanup_no_leaked_figures(self):
        visualizer = self._make_visualizer()

        self.assertEqual(len(plt.get_fignums()), 0)

        visualizer.render_step(step=1, robot_pose=(0.0, 0.0, 0.0))

        self.assertEqual(len(plt.get_fignums()), 0)

        for step in range(100, 125):
            visualizer.render_step(step=step, robot_pose=(0.0, 0.0, 0.0))

        self.assertEqual(len(plt.get_fignums()), 0)

    def test_overlay_callback_executed_with_axes(self):
        visualizer = self._make_visualizer()
        spy_callback = mock.MagicMock()

        frame_path = visualizer.render_step(
            step=1, robot_pose=(0.0, 0.0, 0.0), overlay_fn=spy_callback
        )

        spy_callback.assert_called_once()
        overlay_axes = spy_callback.call_args[0][0]
        self.assertIsInstance(overlay_axes, matplotlib.axes.Axes)
        self._assert_is_valid_png(frame_path)

        # Absent delegate is a supported configuration, and must not re-invoke the spy.
        frame_path = visualizer.render_step(
            step=2, robot_pose=(0.0, 0.0, 0.0), overlay_fn=None
        )
        self.assertEqual(spy_callback.call_count, 1)
        self._assert_is_valid_png(frame_path)

    def test_hud_rendering_with_various_metric_types(self):
        visualizer = self._make_visualizer()
        # `None` and `{}` both mean "no HUD"; the rich case mixes str/int/float/
        # bool/None to prove formatting is value-agnostic and cannot raise.
        hud_cases = [
            None,
            {},
            {
                "Alg": "MCTS",
                "Step": 42,
                "Q/N": -3.1415,
                "Expanded": 16,
                "Reached": False,
                "NoneVal": None,
            },
        ]

        for step, hud_metrics in enumerate(hud_cases):
            try:
                frame_path = visualizer.render_step(
                    step=step, robot_pose=(0.0, 0.0, 0.0), hud_metrics=hud_metrics
                )
            except (TypeError, ValueError, KeyError, AttributeError) as exc:  # pragma: no cover
                self.fail(f"HUD formatting failed for {hud_metrics!r}: {exc!r}")
            self._assert_is_valid_png(frame_path)

    def test_from_run_log_and_render_from_csv(self):
        obstacles = [{"x": 0.0, "z": 0.0, "width": 1.0, "depth": 1.0}]
        self._write_run_log(obstacles=obstacles)
        csv_path = self._write_steps_csv()

        visualizer = TrajectoryVisualizer.from_run_log(self.temp_dir)

        # Geometry, radii, and goal come from the log, not the constructor defaults.
        self.assertEqual((visualizer.room_width, visualizer.room_depth), (5.0, 4.0))
        self.assertEqual(visualizer.obstacles, obstacles)
        self.assertEqual((visualizer.goal_x, visualizer.goal_z), (1.25, -0.75))
        self.assertEqual(visualizer.goal_radius, 0.4)
        self.assertEqual(visualizer.robot_radius, 0.2)
        # Output stays inside the replayed run directory, as the CLI expects.
        self.assertEqual(visualizer.output_dir, self.temp_dir / "frames")

        frames = visualizer.render_from_csv(csv_path)

        self.assertEqual([f.name for f in frames], ["step_000.png", "step_001.png"])
        for frame_path in frames:
            self.assertEqual(frame_path.parent, self.temp_dir / "frames")
            self._assert_is_valid_png(frame_path)

        # History is rebuilt from the CSV, not accumulated from elsewhere.
        self.assertEqual(visualizer.history_x, [-1.0, -0.5])
        self.assertEqual(visualizer.history_z, [-0.5, 0.0])

    def test_render_from_csv_replay_is_repeatable(self):
        self._write_run_log()
        csv_path = self._write_steps_csv()
        visualizer = TrajectoryVisualizer.from_run_log(self.temp_dir)

        first = visualizer.render_from_csv(csv_path)
        second = visualizer.render_from_csv(csv_path)

        self.assertEqual(len(first), len(second))
        self.assertEqual(visualizer.history_x, [-1.0, -0.5])
        for frame_path in second:
            self._assert_is_valid_png(frame_path)

    def test_render_from_csv_rejects_legacy_log_without_pose_columns(self):
        # Every pre-existing run in logs/mcts/ has this 13-column layout.
        legacy = [
            {
                "step": 0,
                "action_v_normalized": 1.0,
                "action_w_normalized": 0.0,
                "root_value": 0.0,
                "tree_size": 1,
                "max_depth": 0,
                "distance_to_goal_m": 2.5,
                "executed_reward": 0.0,
                "search_time_sec": 0.0,
                "execution_time_sec": 0.0,
                "expansion_calls": 0,
                "physics_ticks": 0,
                "decisions_per_sec": 0.0,
            }
        ]
        visualizer = self._make_visualizer()
        csv_path = self._write_steps_csv(rows=legacy)

        with self.assertRaises(ValueError) as ctx:
            visualizer.render_from_csv(csv_path)

        message = str(ctx.exception)
        for column in ("robot_x", "robot_z", "robot_yaw"):
            self.assertIn(column, message)
        self.assertFalse((self.temp_dir / "frames" / "step_000.png").exists())

    def test_from_run_log_resolves_room_presets(self):
        # `standard` and `empty` are presets: the log stores the name only, so the
        # loader must expand them to the same geometry as core.room.Room.
        self._write_run_log(room_type="standard")
        standard = TrajectoryVisualizer.from_run_log(self.temp_dir)
        self.assertEqual(
            (standard.room_width, standard.room_depth),
            STANDARD_ROOM_BOUNDS,
        )
        self.assertEqual(len(standard.obstacles), 2)

        self._write_run_log(room_type="empty")
        empty = TrajectoryVisualizer.from_run_log(self.temp_dir)
        self.assertEqual((empty.room_width, empty.room_depth), EMPTY_ROOM_BOUNDS)
        self.assertEqual(empty.obstacles, [])

    def test_from_run_log_prefers_recorded_scenario_geometry(self):
        metadata = self._write_run_log(room_type="custom", room_extent=(5.0, 4.0))
        metadata["scenario"]["geometry"] = {
            "room_bounds": [7.0, 6.0],
            "obstacles": [{"x": 1.0, "z": 1.0, "width": 0.5, "depth": 0.5}],
        }
        (self.temp_dir / "run.yaml").write_text(yaml.safe_dump(metadata, sort_keys=False))

        visualizer = TrajectoryVisualizer.from_run_log(self.temp_dir)

        self.assertEqual((visualizer.room_width, visualizer.room_depth), (7.0, 6.0))
        self.assertEqual(
            visualizer.obstacles,
            [{"x": 1.0, "z": 1.0, "width": 0.5, "depth": 0.5}],
        )


if __name__ == "__main__":
    unittest.main()
