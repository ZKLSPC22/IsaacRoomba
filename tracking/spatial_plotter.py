"""Headless 2D trajectory visualization for IsaacRoomba runs.

`TrajectoryVisualizer` renders a top-down (X horizontal, Z vertical) view of the
room, the robot pose history, and optional diagnostics into standalone PNG
frames. It is deliberately free of Isaac Gym, CUDA, and PyTorch dependencies so
that it can run on any machine and be unit tested on CPU; the `Agg` backend is
selected at import time and no window is ever created.

Frames are composed in four layers, each drawn independently of the others:

1. Static scene — room bounds, obstacle bodies, inflated obstacle footprints, goal.
2. Kinematics — robot body, heading, and breadcrumb trail.
3. Overlay — an optional caller-supplied callback invoked with the axes.
4. HUD — an optional free-form metrics box, rendered value-agnostically.

Layers 3 and 4 know nothing about MCTS or the environment: callers pass plain
dictionaries and callables, which keeps this module reusable for replays.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib

# Headless backend must be selected before pyplot is imported.
matplotlib.use("Agg")

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import yaml

# Built-in room presets, mirroring `core.room.Room.empty()` / `Room.standard()`.
# They are duplicated rather than imported because `core.room` imports isaacgym,
# which would make this module unusable off a GPU machine.
EMPTY_ROOM_BOUNDS = (10.0, 10.0)
EMPTY_ROOM_OBSTACLES: List[Dict[str, float]] = []
STANDARD_ROOM_BOUNDS = (20.0, 20.0)
STANDARD_ROOM_OBSTACLES: List[Dict[str, float]] = [
    {"x": 0.0, "z": 0.0, "width": 2.0, "depth": 2.0},  # Central pillar
    {"x": 5.0, "z": 5.0, "width": 8.0, "depth": 1.0},  # Dividing wall
]


class TrajectoryVisualizer:
    """Renders per-step top-down PNG frames of a navigation run.

    Parameters
    ----------
    output_dir:
        Run directory (or any directory); frames are written to
        ``<output_dir>/<subfolder>/``.
    room_bounds:
        ``(width, depth)`` of the room in metres, centred on the origin.
    obstacles:
        Axis-aligned boxes as dictionaries with ``x``, ``z``, ``width`` and
        ``depth`` keys, matching `core.room.BoxObstacle`.
    goal:
        ``(x, z)`` goal centre in room-local metres.
    goal_radius:
        Radius of the goal acceptance region in metres.
    robot_radius:
        Robot body radius in metres.
    safety_margin:
        Clearance added to `robot_radius` when drawing inflated footprints.
    subfolder:
        Directory created under `output_dir` to hold the frames.
    """

    def __init__(
        self,
        output_dir: Path | str,
        room_bounds: Tuple[float, float],
        obstacles: List[Dict[str, float]],
        goal: Tuple[float, float],
        goal_radius: float = 0.5,
        robot_radius: float = 0.17,
        safety_margin: float = 0.05,
        subfolder: str = "frames",
    ):
        self.room_width = float(room_bounds[0])
        self.room_depth = float(room_bounds[1])
        self.goal_x = float(goal[0])
        self.goal_z = float(goal[1])
        self.goal_radius = float(goal_radius)
        self.robot_radius = float(robot_radius)
        self.safety_margin = float(safety_margin)
        self.obstacles = obstacles

        self.inflation = float(robot_radius + safety_margin)

        self.output_dir = Path(output_dir) / subfolder
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.history_x: List[float] = []
        self.history_z: List[float] = []

    @classmethod
    def from_run_log(cls, run_dir: Path | str) -> TrajectoryVisualizer:
        """Build a visualizer from the `run.yaml` of a previously logged run.

        The room geometry is taken from `scenario.geometry` when a run records it
        directly; otherwise it is reconstructed from the resolved environment
        config, which stores the preset name (`empty`/`standard`/`custom`) rather
        than the expanded room. Radii and the goal fall back to the same defaults
        the environment uses when an older log omits them.

        Frames are written to `<run_dir>/frames/`, keeping a replay's artifacts
        next to the run it replays.
        """
        run_dir = Path(run_dir)
        metadata = yaml.safe_load((run_dir / "run.yaml").read_text())

        goal_dict = metadata["scenario"]["goal"]
        goal = (float(goal_dict["x"]), float(goal_dict["z"]))

        env_cfg = metadata.get("configuration", {}).get("environment", {})
        goal_radius = float(env_cfg.get("task", {}).get("goal_radius", 0.5))
        robot_radius = float(env_cfg.get("robot", {}).get("radius", 0.17))

        # A run may record its geometry directly; that always wins.
        geom = metadata.get("scenario", {}).get("geometry", {}) or {}
        room_bounds = tuple(geom["room_bounds"]) if "room_bounds" in geom else None
        obstacles = geom.get("obstacles", None)

        if room_bounds is None:
            room_cfg = env_cfg.get("room", {})
            room_type = room_cfg.get("type")
            if room_type == "custom":
                custom = room_cfg["custom"]
                room_bounds = (float(custom["width"]), float(custom["depth"]))
                obstacles = custom.get("obstacles", [])
            elif room_type == "standard":
                room_bounds = STANDARD_ROOM_BOUNDS
                obstacles = STANDARD_ROOM_OBSTACLES
            else:
                # "empty" and any unrecognised preset the environment accepts.
                room_bounds = EMPTY_ROOM_BOUNDS
                obstacles = EMPTY_ROOM_OBSTACLES

        return cls(
            output_dir=run_dir,
            room_bounds=room_bounds,
            obstacles=obstacles or [],
            goal=goal,
            goal_radius=goal_radius,
            robot_radius=robot_radius,
        )

    def render_from_csv(self, csv_path: Path | str) -> List[Path]:
        """Replay a `steps.csv` into frames, returning the paths written.

        `robot_x`, `robot_z`, and `robot_yaw` are required: without a recorded
        pose there is nothing to draw. Logs written before those columns existed
        raise `ValueError` here rather than producing a misleading flipbook.
        History is cleared first so a replay is self-contained and does not
        inherit positions from earlier renders on the same instance.
        """
        csv_path = Path(csv_path)
        frames: List[Path] = []

        with open(csv_path, newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames or []
            required = {"step", "robot_x", "robot_z", "robot_yaw"}
            if not required.issubset(fieldnames):
                raise ValueError(
                    f"CSV missing required columns: {required - set(fieldnames)}"
                )

            self.history_x.clear()
            self.history_z.clear()

            for row in reader:
                step = int(row["step"])
                pose = (
                    float(row["robot_x"]),
                    float(row["robot_z"]),
                    float(row["robot_yaw"]),
                )

                hud: Dict[str, Any] = {}
                for key, label in [
                    ("distance_to_goal_m", "Dist"),
                    ("root_value", "Root Q/N"),
                    ("tree_size", "Nodes"),
                ]:
                    if key in row and row[key] != "":
                        hud[label] = row[key]

                frames.append(
                    self.render_step(step=step, robot_pose=pose, hud_metrics=hud)
                )

        return frames

    def _draw_static_scene(self, ax: plt.Axes) -> None:
        """Layer 1: room bounds, obstacle footprints and bodies, and the goal."""
        hw = self.room_width / 2.0
        hd = self.room_depth / 2.0

        ax.add_patch(
            patches.Rectangle(
                (-hw, -hd),
                self.room_width,
                self.room_depth,
                fc="none",
                ec="black",
                lw=2,
                zorder=1,
            )
        )

        for obstacle in self.obstacles:
            ox = float(obstacle["x"])
            oz = float(obstacle["z"])
            width = float(obstacle["width"])
            depth = float(obstacle["depth"])

            # Inflated footprint: the region the robot centre must stay out of.
            ax.add_patch(
                patches.Rectangle(
                    (ox - width / 2.0 - self.inflation, oz - depth / 2.0 - self.inflation),
                    width + 2.0 * self.inflation,
                    depth + 2.0 * self.inflation,
                    ls="--",
                    ec="gray",
                    fc="none",
                    lw=1,
                    zorder=2,
                )
            )
            ax.add_patch(
                patches.Rectangle(
                    (ox - width / 2.0, oz - depth / 2.0),
                    width,
                    depth,
                    fc="dimgray",
                    ec="black",
                    lw=1.5,
                    zorder=3,
                )
            )

        ax.add_patch(
            patches.Circle(
                (self.goal_x, self.goal_z),
                radius=self.goal_radius,
                color="green",
                alpha=0.2,
                zorder=2,
            )
        )
        ax.plot(
            self.goal_x,
            self.goal_z,
            marker="x",
            color="green",
            markersize=8,
            linestyle="none",
            zorder=3,
        )

    def _draw_kinematics(self, ax: plt.Axes, x: float, z: float, yaw: float) -> None:
        """Layer 2: breadcrumb trail, robot body, and heading arrow."""
        # The trail is only meaningful once at least two poses are known: a
        # single-point plot would render no line at all.
        if len(self.history_x) > 1:
            ax.plot(
                self.history_x,
                self.history_z,
                color="royalblue",
                lw=1.5,
                alpha=0.7,
                zorder=4,
            )

        ax.add_patch(
            patches.Circle(
                (x, z),
                radius=self.robot_radius,
                color="deepskyblue",
                ec="black",
                lw=1.2,
                zorder=5,
            )
        )

        # Planar heading: yaw rotates the +X forward axis toward +Z.
        arrow_len = self.robot_radius * 1.3
        ax.arrow(
            x,
            z,
            arrow_len * math.cos(yaw),
            arrow_len * math.sin(yaw),
            head_width=0.05,
            head_length=0.05,
            fc="crimson",
            ec="crimson",
            zorder=6,
        )

    def _draw_hud(self, ax: plt.Axes, hud_metrics: Dict[str, Any]) -> None:
        """Layer 4: free-form metrics card in axes-relative coordinates.

        Values are stringified generically, so callers may pass any mix of
        strings, numbers, booleans, or `None`.
        """
        text = "\n".join(f"{key}: {value}" for key, value in hud_metrics.items())
        ax.text(
            0.03,
            0.97,
            text,
            transform=ax.transAxes,
            fontsize=8,
            fontfamily="monospace",
            va="top",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.85),
            zorder=10,
        )

    def render_step(
        self,
        step: int,
        robot_pose: Tuple[float, float, float],
        hud_metrics: Optional[Dict[str, Any]] = None,
        overlay_fn: Optional[Callable[[plt.Axes], None]] = None,
    ) -> Path:
        """Render one frame to ``<output_dir>/step_<step:03d>.png``.

        Parameters
        ----------
        step:
            Control step index; zero-padded to three digits in the filename.
        robot_pose:
            ``(x, z, yaw)`` in room-local metres and radians. Out-of-bounds
            poses are drawn as-is and simply fall outside the visible axes.
        hud_metrics:
            Optional mapping rendered verbatim as a diagnostic text card.
            `None` and `{}` both mean "draw no HUD".
        overlay_fn:
            Optional callback invoked with the axes after the kinematics and
            before the HUD, letting callers add algorithm-specific overlays.

        Returns
        -------
        Path
            Location of the written PNG frame.
        """
        x = float(robot_pose[0])
        z = float(robot_pose[1])
        yaw = float(robot_pose[2])
        self.history_x.append(x)
        self.history_z.append(z)

        fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
        try:
            # Layer 1: static scene geometry
            self._draw_static_scene(ax)

            # Layer 2: kinematics (trail, body, heading)
            self._draw_kinematics(ax, x, z, yaw)

            # Layer 3: pluggable overlay callback
            if overlay_fn is not None:
                overlay_fn(ax)

            # Layer 4: HUD metrics card
            if hud_metrics:
                self._draw_hud(ax, hud_metrics)

            # Axis bounds, scaling, and decoration removal
            hw = self.room_width / 2.0
            hd = self.room_depth / 2.0
            ax.set_xlim(-hw, hw)
            ax.set_ylim(-hd, hd)
            ax.set_aspect("equal")
            ax.axis("off")

            # Persist frame
            frame_path = self.output_dir / f"step_{step:03d}.png"
            fig.savefig(frame_path, bbox_inches="tight", pad_inches=0.05)
            return frame_path
        finally:
            plt.close(fig)
