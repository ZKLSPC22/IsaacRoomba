"""Reusable per-run logging for IsaacRoomba experiments.

Each run writes a self-describing directory::

    <output_dir>/<run_id>/
    ├── run.yaml                  # authoritative, schema-versioned metadata
    ├── steps.csv                 # authoritative per-step metrics
    └── tensorboard/              # derived visualization (only when enabled)
        └── events.out.tfevents...

``run.yaml`` and ``steps.csv`` are the portable source of truth. TensorBoard is a
derived visualization layer only: no metric or configuration value is stored
exclusively in TensorBoard event files.
"""

from __future__ import annotations

import csv
import datetime
import os
import subprocess
import time
from collections.abc import Mapping
from pathlib import Path

import yaml

# Version of the `run.yaml` document layout. Bump when fields change meaning.
RUN_LOG_SCHEMA_VERSION = 1

RUN_METADATA_FILENAME = "run.yaml"
STEPS_FILENAME = "steps.csv"
TENSORBOARD_DIRNAME = "tensorboard"

# Per-step metric columns, in order. Names carry explicit units so that a CSV is
# interpretable without the code that produced it.
STEP_COLUMNS = (
    "step",
    "action_v_normalized",
    "action_w_normalized",
    "root_value",
    "tree_size",
    "max_depth",
    "distance_to_goal_m",
    "executed_reward",
    "search_time_sec",
    "execution_time_sec",
    "expansion_calls",
    "physics_ticks",
    "decisions_per_sec",
)

# TensorBoard scalar tags keyed by step-metric column. The per-step metrics
# dictionary is built once; the same values go to CSV and TensorBoard.
TENSORBOARD_TAGS = {
    "distance_to_goal_m": "navigation/distance_to_goal_m",
    "executed_reward": "navigation/executed_reward",
    "root_value": "search/root_value",
    "tree_size": "search/tree_size",
    "max_depth": "search/max_depth",
    "expansion_calls": "search/expansion_calls",
    "search_time_sec": "performance/search_time_sec",
    "execution_time_sec": "performance/execution_time_sec",
    "physics_ticks": "performance/physics_ticks",
    "decisions_per_sec": "performance/decisions_per_sec",
}

STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_INTERRUPTED = "interrupted"
STATUS_FAILED = "failed"

VALID_STATUSES = frozenset({STATUS_RUNNING, STATUS_COMPLETED, STATUS_INTERRUPTED, STATUS_FAILED})

TERMINATION_GOAL_REACHED = "goal_reached"
TERMINATION_NO_PROGRESS = "no_progress"
TERMINATION_EXECUTION_HORIZON = "execution_horizon"
TERMINATION_INTERRUPTED = "interrupted"
TERMINATION_ERROR = "error"

VALID_TERMINATION_REASONS = frozenset(
    {
        None,
        TERMINATION_GOAL_REACHED,
        TERMINATION_NO_PROGRESS,
        TERMINATION_EXECUTION_HORIZON,
        TERMINATION_INTERRUPTED,
        TERMINATION_ERROR,
    }
)


class TensorBoardUnavailableError(RuntimeError):
    """Raised when TensorBoard logging is requested but unavailable."""


# Cleans up Python object for loading into a yaml
def _to_yaml_safe(value):
    """Recursively convert `value` into plain YAML-serializable Python objects.

    Handles tensors, NumPy scalars/arrays, ``Path`` objects, and nested
    containers. NumPy and torch are not imported here; objects that expose
    ``tolist``/``item`` are converted through those methods so this module has no
    hard dependency on either library.

    Native scalars are matched by *exact* type rather than ``isinstance``. NumPy
    scalar types subclass the Python scalars (``np.float64`` is a ``float``), but
    PyYAML's safe representer dispatches on the exact type and therefore cannot
    serialize them. Routing subclasses through the ``tolist``/``item`` branch
    converts them to genuine Python scalars, which YAML can represent.
    """
    if value is None:
        return None

    if type(value) in (bool, int, float, str):
        return value

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, Mapping):
        return {str(key): _to_yaml_safe(item) for key, item in value.items()}

    if isinstance(value, (list, tuple, set, frozenset)):
        return [_to_yaml_safe(item) for item in value]

    # NumPy scalars/arrays and tensors: try a lossless element-wise conversion
    # before falling back to `item()` and finally to `str`.
    for accessor in ("tolist", "item"):
        method = getattr(value, accessor, None)
        if callable(method):
            try:
                converted = method()
            except Exception:  # pragma: no cover - defensive fallback
                continue
            if converted is value:  # pragma: no cover - defensive guard
                break
            return _to_yaml_safe(converted)

    return str(value)


def collect_git_metadata(repo_root) -> dict:
    """Return `{"commit", "dirty", "error"}` describing the working tree.

    Failures are recorded as `None` values plus an `error` string; a missing or
    broken Git installation must never abort an experiment.
    """
    metadata = {"commit": None, "dirty": None, "error": None}
    repo_root = str(repo_root)

    def run_git(*args):
        return subprocess.run(
            ["git", *args],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )

    try:
        commit = run_git("rev-parse", "HEAD")
        if commit.returncode == 0:
            metadata["commit"] = commit.stdout.strip()
        else:
            metadata["error"] = f"git rev-parse HEAD failed (exit {commit.returncode})"

        # Untracked files count as dirty: uncommitted code changes results.
        status = run_git("status", "--porcelain")
        if status.returncode == 0:
            metadata["dirty"] = bool(status.stdout.strip())
        else:
            metadata["error"] = "git status --porcelain failed (exit %d)" % status.returncode
    except (OSError, subprocess.SubprocessError) as exc:
        metadata["error"] = f"{type(exc).__name__}: {exc}"

    return metadata


class MCTSRunLogger:
    """Writes `run.yaml`, `steps.csv`, and optional TensorBoard scalars for one run."""

    def __init__(
        self,
        output_dir="logs/mcts",
        run_id=None,
        *,
        environment_config=None,
        planner_config=None,
        experiment_config=None,
        scenario=None,
        initial_distance=None,
        enable_tensorboard=False,
        tensorboard_writer=None,
        repo_root=None,
        timestamp=None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.run_id = run_id or self._generate_run_id(timestamp)
        self.run_dir = self._create_unique_run_dir(self.run_id)

        self.run_path = self.run_dir / RUN_METADATA_FILENAME
        self.steps_path = self.run_dir / STEPS_FILENAME

        self._started_perf = time.perf_counter()
        self._closed = False
        self._steps_file = None
        self._csv_writer = None

        self._metadata = {
            "schema_version": RUN_LOG_SCHEMA_VERSION,
            "run_id": self.run_dir.name,
            "started_at_utc": _utc_now_iso(),
            "finished_at_utc": None,
            "git": collect_git_metadata(
                repo_root if repo_root is not None else Path(__file__).resolve().parent.parent
            ),
            "configuration": {
                "environment": _to_yaml_safe(environment_config),
                "planner": _to_yaml_safe(planner_config),
                "experiment": _to_yaml_safe(experiment_config),
            },
            "scenario": _to_yaml_safe(scenario),
            "result": {
                "status": STATUS_RUNNING,
                "termination_reason": None,
                "success": None,
                "steps": 0,
                "initial_distance": _to_yaml_safe(initial_distance),
                "final_distance": None,
                "elapsed_seconds": None,
            },
        }

        # Initial metadata is on disk before the execution loop starts, so an
        # interrupted run is still identifiable.
        self._write_metadata()

        # Build the TensorBoard writer before opening the CSV so that a failed
        # writer setup cannot leak an open file handle.
        self._writer = tensorboard_writer
        if self._writer is None and enable_tensorboard:
            self._writer = self._create_tensorboard_writer()

        try:
            self._steps_file = open(self.steps_path, mode="w", newline="")
            self._csv_writer = csv.writer(self._steps_file)
            self._csv_writer.writerow(list(STEP_COLUMNS))
            self._steps_file.flush()
        except BaseException:
            # Never leak a CSV handle or TensorBoard writer on a failed setup.
            self.close()
            raise

    # ------------------------------------------------------------------ setup

    def _generate_run_id(self, timestamp=None):
        stamp = timestamp or datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
        return f"mcts_{stamp}"

    def _create_unique_run_dir(self, run_id):
        """Create and return a run directory, never reusing an existing one."""
        candidate = self.output_dir / run_id
        suffix = 2
        while True:
            try:
                candidate.mkdir(parents=True, exist_ok=False)
                return candidate
            except FileExistsError:
                candidate = self.output_dir / f"{run_id}_{suffix:02d}"
                suffix += 1
                if suffix > 1000:  # pragma: no cover - defensive guard
                    raise RuntimeError(f"Could not allocate a unique run directory under {self.output_dir}")

    def _create_tensorboard_writer(self):
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as exc:
            raise TensorBoardUnavailableError(
                "TensorBoard logging is enabled but the 'tensorboard' package is not "
                "importable. Install 'tensorboard' in the active environment, or set "
                "'mcts.logging.tensorboard: false' in configs/experiments.yaml to log "
                "CSV/YAML only."
            ) from exc

        return SummaryWriter(log_dir=str(self.run_dir / TENSORBOARD_DIRNAME))

    # --------------------------------------------------------------- metadata

    def _write_metadata(self):
        """Atomically rewrite `run.yaml` via a temporary file and `os.replace`.

        The payload is re-sanitized here so that values added late (e.g. by
        `finalize`) cannot introduce a YAML-unrepresentable object.
        """
        payload = yaml.safe_dump(
            _to_yaml_safe(self._metadata), sort_keys=False, default_flow_style=False
        )
        tmp_path = self.run_path.parent / (self.run_path.name + ".tmp")
        with open(tmp_path, mode="w") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, self.run_path)

    @property
    def metadata(self):
        """A deep copy of the current `run.yaml` payload, as plain Python values."""
        return _to_yaml_safe(self._metadata)

    @property
    def tensorboard_enabled(self):
        return self._writer is not None

    # ------------------------------------------------------------------ steps

    def log_step(self, metrics: Mapping):
        """Write one metrics row to CSV and (if enabled) the same values to TensorBoard.

        `metrics` must contain every column in `STEP_COLUMNS`; missing keys raise
        `KeyError` so a misconfigured runner fails fast rather than writing an
        incomplete row.
        """
        missing = [column for column in STEP_COLUMNS if column not in metrics]
        if missing:
            raise KeyError(f"Missing required step metric(s): {', '.join(missing)}")

        row = [metrics[column] for column in STEP_COLUMNS]
        self._csv_writer.writerow(row)
        self._steps_file.flush()

        if self._writer is not None:
            step = int(metrics["step"])
            for column, tag in TENSORBOARD_TAGS.items():
                self._writer.add_scalar(tag, float(metrics[column]), step)

    # ----------------------------------------------------------- finalisation

    def finalize(
        self,
        *,
        status,
        termination_reason=None,
        success=None,
        steps=None,
        initial_distance=None,
        final_distance=None,
    ):
        """Record the run outcome and persist `run.yaml`."""
        if status not in VALID_STATUSES:
            raise ValueError(f"Invalid status {status!r}; expected one of {sorted(VALID_STATUSES)}")
        if termination_reason not in VALID_TERMINATION_REASONS:
            raise ValueError(
                f"Invalid termination_reason {termination_reason!r}; "
                f"expected one of {sorted(str(r) for r in VALID_TERMINATION_REASONS)}"
            )

        result = self._metadata["result"]
        result["status"] = status
        result["termination_reason"] = termination_reason
        result["success"] = success
        if steps is not None:
            result["steps"] = int(steps)
        if initial_distance is not None:
            result["initial_distance"] = _to_yaml_safe(initial_distance)
        if final_distance is not None:
            result["final_distance"] = _to_yaml_safe(final_distance)

        self._metadata["finished_at_utc"] = _utc_now_iso()
        result["elapsed_seconds"] = round(time.perf_counter() - self._started_perf, 6)
        self._write_metadata()

    def close(self):
        """Flush and close the CSV file and the TensorBoard writer. Idempotent."""
        if self._closed:
            return

        try:
            if self._writer is not None:
                self._writer.flush()
                self._writer.close()
        finally:
            if self._steps_file is not None and not self._steps_file.closed:
                self._steps_file.flush()
                self._steps_file.close()
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, exc_tb):
        self.close()
        return False


def _utc_now_iso():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
