"""Tests for `tracking.run_logger`.

These tests never launch Isaac Gym and never write into the repository's real
`logs/` directory: every run logger is pointed at a `tempfile.TemporaryDirectory`.
"""

import csv
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.append(str(Path(__file__).resolve().parent.parent))

import yaml

from tracking.run_logger import (
    MCTSRunLogger,
    RUN_LOG_SCHEMA_VERSION,
    STATUS_COMPLETED,
    STATUS_INTERRUPTED,
    STATUS_RUNNING,
    STEP_COLUMNS,
    TENSORBOARD_TAGS,
    TERMINATION_GOAL_REACHED,
    TERMINATION_INTERRUPTED,
    TensorBoardUnavailableError,
)


class _RecordingWriter:
    """Minimal stand-in for `torch.utils.tensorboard.SummaryWriter`."""

    def __init__(self):
        self.scalars = []
        self.flush_count = 0
        self.close_count = 0

    def add_scalar(self, tag, value, step):
        self.scalars.append((tag, value, step))

    def flush(self):
        self.flush_count += 1

    def close(self):
        self.close_count += 1


def _step_metrics(step=1):
    """A complete, valid per-step metrics dictionary."""
    return {
        "step": step,
        "action_v_normalized": 0.5,
        "action_w_normalized": -1.0,
        "root_value": -3.25,
        "tree_size": 241,
        "max_depth": 2,
        "distance_to_goal_m": 2.7518,
        "executed_reward": -0.1,
        "search_time_sec": 0.688068,
        "execution_time_sec": 0.033009,
        "expansion_calls": 16,
        "physics_ticks": 255,
        "decisions_per_sec": 1.3864,
    }


# Shared setup code for test classes
class RunLoggerTestBase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.output_dir = Path(self._tmp.name) / "mcts"

    # Default arguments for child classes
    def make_logger(self, **kwargs):
        kwargs.setdefault("environment_config", {"env": {"num_planning_envs": 16}})
        kwargs.setdefault("planner_config", {"base": {"num_iterations": 100}})
        kwargs.setdefault("experiment_config", {"mcts": {"seed": 0}})
        kwargs.setdefault("scenario", {"seed": 0, "start": {"x": 1.0, "z": 2.0}, "goal": {"x": -1.5, "z": 0.5}})
        kwargs.setdefault("initial_distance", 3.5)
        logger = MCTSRunLogger(output_dir=self.output_dir, **kwargs)
        self.addCleanup(logger.close)
        return logger


class RunDirectoryTests(RunLoggerTestBase):
    def test_creates_unique_run_directory(self):
        first = self.make_logger()
        second = self.make_logger()

        self.assertTrue(first.run_dir.is_dir())
        self.assertTrue(second.run_dir.is_dir())
        self.assertNotEqual(first.run_dir, second.run_dir)
        self.assertEqual(first.run_dir.parent, self.output_dir)
        self.assertTrue((first.run_dir / "run.yaml").is_file())
        self.assertTrue((first.run_dir / "steps.csv").is_file())


class MetadataTests(RunLoggerTestBase):
    def test_initial_metadata_is_valid_and_running(self):
        logger = self.make_logger()
        payload = yaml.safe_load(logger.run_path.read_text())

        self.assertEqual(payload["schema_version"], RUN_LOG_SCHEMA_VERSION)
        self.assertEqual(payload["run_id"], logger.run_dir.name)
        self.assertEqual(payload["result"]["status"], STATUS_RUNNING)
        self.assertIsNone(payload["result"]["termination_reason"])
        self.assertIsNone(payload["result"]["success"])
        self.assertIsNone(payload["finished_at_utc"])
        self.assertIsNotNone(payload["started_at_utc"])
        self.assertIn("commit", payload["git"])
        self.assertIn("dirty", payload["git"])

    def test_resolved_configurations_are_preserved(self):
        environment = {"env": {"num_planning_envs": 16}, "robot": {"radius": 0.17}}
        planner = {"base": {"c_param": 1.414, "num_iterations": 100, "gamma": 0.95}}
        experiment = {"mcts": {"seed": 0, "logging": {"tensorboard": False}}}

        logger = self.make_logger(
            environment_config=environment,
            planner_config=planner,
            experiment_config=experiment,
        )
        configuration = yaml.safe_load(logger.run_path.read_text())["configuration"]

        self.assertEqual(configuration["environment"], environment)
        self.assertEqual(configuration["planner"], planner)
        self.assertEqual(configuration["experiment"], experiment)

    def test_seed_and_scenario_are_plain_numeric_values(self):
        logger = self.make_logger()
        scenario = yaml.safe_load(logger.run_path.read_text())["scenario"]

        self.assertEqual(scenario["seed"], 0)
        self.assertIsInstance(scenario["seed"], int)
        for key in ("start", "goal"):
            for axis in ("x", "z"):
                value = scenario[key][axis]
                self.assertIsInstance(value, float, f"{key}.{axis} should be a plain float")
        self.assertEqual(scenario["start"]["x"], 1.0)
        self.assertEqual(scenario["goal"]["z"], 0.5)

    def test_special_objects_are_converted_to_yaml_safe_values(self):
        import numpy as np

        class FakeTensor:
            def __init__(self, value):
                self._value = value

            def tolist(self):
                return [self._value]

        logger = self.make_logger(
            environment_config={
                "numpy_scalar": np.float32(0.25),
                "numpy_array": np.array([1.0, 2.0]),
                "path": Path("configs/config.yaml"),
                "tensor_like": FakeTensor(3.5),
            }
        )
        environment = yaml.safe_load(logger.run_path.read_text())["configuration"]["environment"]

        self.assertEqual(environment["numpy_scalar"], 0.25)
        self.assertEqual(environment["numpy_array"], [1.0, 2.0])
        self.assertEqual(environment["path"], "configs/config.yaml")
        self.assertEqual(environment["tensor_like"], [3.5])

    def test_numpy_scalars_are_serializable_in_scenario_and_configs(self):
        """Regression: `OccupancyMap.sample_valid_pose` returns `np.float64`.

        NumPy scalars subclass the Python scalars, but PyYAML dispatches on the
        exact type, so a bare `isinstance(value, float)` check is not sufficient.
        """
        import numpy as np

        logger = self.make_logger(
            scenario={
                "seed": np.int64(0),
                "start": {"x": np.float64(-0.7249999999999999), "z": np.float64(1.25)},
                "goal": {"x": np.float32(-2.1), "z": np.float64(0.5)},
            },
            initial_distance=np.float64(3.5),
            environment_config={
                "numpy_bool": np.bool_(True),
                "numpy_int64": np.int64(16),
                "numpy_float64": np.float64(0.05),
                "numpy_float32": np.float32(1.5),
                "numpy_zero_d": np.array(2.5),
                "numpy_nested": {"values": np.array([[1.0, 2.0], [3.0, 4.0]])},
            },
        )
        logger.finalize(
            status=STATUS_COMPLETED,
            termination_reason=TERMINATION_GOAL_REACHED,
            success=np.bool_(True),
            steps=np.int64(7),
            final_distance=np.float64(0.42),
        )

        payload = yaml.safe_load(logger.run_path.read_text())

        self.assertEqual(payload["scenario"]["seed"], 0)
        self.assertIsInstance(payload["scenario"]["seed"], int)
        self.assertEqual(payload["scenario"]["start"]["x"], -0.7249999999999999)
        self.assertIsInstance(payload["scenario"]["start"]["x"], float)
        self.assertAlmostEqual(payload["scenario"]["goal"]["x"], -2.1, places=5)

        environment = payload["configuration"]["environment"]
        self.assertIs(environment["numpy_bool"], True)
        self.assertEqual(environment["numpy_int64"], 16)
        self.assertEqual(environment["numpy_float64"], 0.05)
        self.assertAlmostEqual(environment["numpy_float32"], 1.5, places=6)
        self.assertEqual(environment["numpy_zero_d"], 2.5)
        self.assertEqual(environment["numpy_nested"]["values"], [[1.0, 2.0], [3.0, 4.0]])

        result = payload["result"]
        self.assertIs(result["success"], True)
        self.assertEqual(result["steps"], 7)
        self.assertAlmostEqual(result["final_distance"], 0.42, places=6)
        self.assertAlmostEqual(result["initial_distance"], 3.5, places=6)


class StepsCsvTests(RunLoggerTestBase):
    def test_steps_csv_header_is_exact(self):
        logger = self.make_logger()
        header = logger.steps_path.read_text().splitlines()[0]
        self.assertEqual(header, ",".join(STEP_COLUMNS))

    def test_logging_a_step_writes_one_complete_row(self):
        logger = self.make_logger()
        logger.log_step(_step_metrics(step=1))

        rows = list(csv.DictReader(logger.steps_path.read_text().splitlines()))
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(set(row), set(STEP_COLUMNS))
        self.assertEqual(row["step"], "1")
        self.assertEqual(row["tree_size"], "241")
        self.assertEqual(row["distance_to_goal_m"], "2.7518")
        self.assertEqual(row["executed_reward"], "-0.1")
        for column in STEP_COLUMNS:
            self.assertNotEqual(row[column], "", f"column {column} should be populated")

    def test_missing_metric_fails_fast(self):
        logger = self.make_logger()
        metrics = _step_metrics()
        del metrics["tree_size"]

        with self.assertRaises(KeyError):
            logger.log_step(metrics)

    def test_multiple_steps_preserve_order(self):
        logger = self.make_logger()
        for step in (1, 2, 3):
            metrics = _step_metrics(step=step)
            metrics["distance_to_goal_m"] = 3.0 - step
            logger.log_step(metrics)

        rows = list(csv.DictReader(logger.steps_path.read_text().splitlines()))
        self.assertEqual([row["step"] for row in rows], ["1", "2", "3"])
        self.assertEqual([row["distance_to_goal_m"] for row in rows], ["2.0", "1.0", "0.0"])


class FinalizeTests(RunLoggerTestBase):
    def test_finalize_records_completion_details(self):
        logger = self.make_logger()
        logger.log_step(_step_metrics(step=1))
        logger.finalize(
            status=STATUS_COMPLETED,
            termination_reason=TERMINATION_GOAL_REACHED,
            success=True,
            steps=1,
            final_distance=0.42,
        )

        result = yaml.safe_load(logger.run_path.read_text())["result"]
        self.assertEqual(result["status"], STATUS_COMPLETED)
        self.assertEqual(result["termination_reason"], TERMINATION_GOAL_REACHED)
        self.assertIs(result["success"], True)
        self.assertEqual(result["steps"], 1)
        self.assertEqual(result["initial_distance"], 3.5)
        self.assertEqual(result["final_distance"], 0.42)
        self.assertIsNotNone(result["elapsed_seconds"])
        self.assertGreaterEqual(result["elapsed_seconds"], 0.0)

    def test_finalize_records_finished_timestamp(self):
        logger = self.make_logger()
        logger.finalize(status=STATUS_COMPLETED, termination_reason=TERMINATION_GOAL_REACHED, success=True)
        payload = yaml.safe_load(logger.run_path.read_text())
        self.assertIsNotNone(payload["finished_at_utc"])

    def test_rejects_invalid_status_and_reason(self):
        logger = self.make_logger()
        with self.assertRaises(ValueError):
            logger.finalize(status="finished")
        with self.assertRaises(ValueError):
            logger.finalize(status=STATUS_COMPLETED, termination_reason="ran_out_of_time")

    def test_interrupted_finalize_leaves_both_files_readable(self):
        logger = self.make_logger()
        logger.log_step(_step_metrics(step=1))
        logger.log_step(_step_metrics(step=2))
        logger.finalize(
            status=STATUS_INTERRUPTED,
            termination_reason=TERMINATION_INTERRUPTED,
            steps=2,
        )
        logger.close()

        result = yaml.safe_load(logger.run_path.read_text())["result"]
        self.assertEqual(result["status"], STATUS_INTERRUPTED)
        self.assertEqual(result["termination_reason"], TERMINATION_INTERRUPTED)
        self.assertIsNone(result["success"])

        rows = list(csv.DictReader(logger.steps_path.read_text().splitlines()))
        self.assertEqual(len(rows), 2)

    def test_close_is_idempotent(self):
        logger = self.make_logger()
        logger.close()
        logger.close()


class AtomicWriteTests(RunLoggerTestBase):
    def test_yaml_update_leaves_no_partial_target(self):
        logger = self.make_logger()
        logger.log_step(_step_metrics(step=1))
        logger.finalize(status=STATUS_COMPLETED, termination_reason=TERMINATION_GOAL_REACHED, success=True)

        # Target parses as a complete document and no temporary file lingers.
        payload = yaml.safe_load(logger.run_path.read_text())
        self.assertEqual(payload["result"]["status"], STATUS_COMPLETED)
        self.assertEqual(list(logger.run_dir.glob("*.tmp")), [])


class TensorBoardIntegrationTests(RunLoggerTestBase):
    def test_tensorboard_disabled_does_not_import_tensorboard(self):
        modules_before = set(sys.modules)

        logger = self.make_logger(enable_tensorboard=False)
        logger.log_step(_step_metrics(step=1))
        logger.finalize(status=STATUS_COMPLETED, termination_reason=TERMINATION_GOAL_REACHED, success=True)
        logger.close()

        newly_imported = {name for name in set(sys.modules) - modules_before if "tensorboard" in name}
        self.assertEqual(newly_imported, set())
        self.assertFalse(logger.tensorboard_enabled)
        self.assertFalse((logger.run_dir / "tensorboard").exists())

    def test_tensorboard_receives_same_values_as_csv(self):
        writer = _RecordingWriter()
        logger = self.make_logger(enable_tensorboard=False, tensorboard_writer=writer)
        metrics = _step_metrics(step=7)
        logger.log_step(metrics)

        rows = list(csv.DictReader(logger.steps_path.read_text().splitlines()))
        row = rows[0]

        self.assertTrue(logger.tensorboard_enabled)
        self.assertEqual(len(writer.scalars), len(TENSORBOARD_TAGS))

        logged = {tag: (value, step) for tag, value, step in writer.scalars}
        for column, tag in TENSORBOARD_TAGS.items():
            self.assertIn(tag, logged, f"missing TensorBoard tag {tag}")
            value, step = logged[tag]
            self.assertEqual(step, 7)
            self.assertEqual(value, float(row[column]))

    def test_writer_is_flushed_and_closed_in_every_termination_path(self):
        writer = _RecordingWriter()
        logger = self.make_logger(enable_tensorboard=False, tensorboard_writer=writer)
        logger.finalize(
            status=STATUS_INTERRUPTED,
            termination_reason=TERMINATION_INTERRUPTED,
            steps=0,
        )
        logger.close()

        self.assertEqual(writer.flush_count, 1)
        self.assertEqual(writer.close_count, 1)

    def test_enabling_tensorboard_without_package_raises_focused_error(self):
        # Setting the module entry to None makes `from ... import` raise ImportError,
        # which stands in for TensorBoard not being installed.
        # The backslash continuation keeps this a single `with` statement, which
        # Python 3.8 requires (parenthesized context managers need 3.10+).
        with mock.patch.dict(sys.modules, {"torch.utils.tensorboard": None}), \
                self.assertRaises(TensorBoardUnavailableError) as context:
            self.make_logger(enable_tensorboard=True)

        message = str(context.exception)
        self.assertIn("tensorboard", message)
        self.assertIn("mcts.logging.tensorboard", message)


if __name__ == "__main__":
    unittest.main()
