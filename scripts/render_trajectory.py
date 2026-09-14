"""Offline 2D replay of a logged MCTS run, with an optional animated GIF.

Usage::

    python scripts/render_trajectory.py logs/mcts/mcts_20260914_055025/
    python scripts/render_trajectory.py logs/mcts/<run>/ --gif --fps 10

Reads the authoritative `run.yaml` and `steps.csv` of a run directory and writes
one top-down PNG per logged step under `<run_dir>/frames/`. Nothing here touches
Isaac Gym, CUDA, or the simulator: it is pure CPU post-processing of a log, so a
run can be replayed on any machine and re-rendered at will.

The run's `steps.csv` must contain the pose columns (`robot_x`, `robot_z`,
`robot_yaw`) added with the spatial logging integration. Runs logged before that
change cannot be replayed; `render_from_csv` reports the missing columns.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.append(str(Path(__file__).resolve().parent.parent))

from tracking.spatial_plotter import TrajectoryVisualizer

RUN_METADATA_FILENAME = "run.yaml"
STEPS_FILENAME = "steps.csv"
GIF_FILENAME = "trajectory.gif"
DEFAULT_GIF_FPS = 5


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="render_trajectory.py",
        description=(
            "Replay a logged MCTS run directory into 2D top-down PNG frames "
            "(written to <run_dir>/frames/) and optionally an animated GIF."
        ),
        epilog=(
            "Example: python scripts/render_trajectory.py "
            "logs/mcts/mcts_20260914_055025/ --gif --fps 10"
        ),
    )
    parser.add_argument(
        "run_dir",
        type=str,
        help="Path to a run directory containing run.yaml and steps.csv.",
    )
    parser.add_argument(
        "--gif",
        action="store_true",
        help=f"Stitch the generated frames into {GIF_FILENAME} in the run directory.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=DEFAULT_GIF_FPS,
        help=f"Frame rate for the GIF animation (default: {DEFAULT_GIF_FPS}).",
    )
    return parser


def _load_gif_backend():
    """Return a callable that writes a GIF, or None if no backend is installed.

    Prefers Pillow (already a matplotlib dependency) and falls back to imageio.
    """
    try:
        from PIL import Image
    except ImportError:
        Image = None

    if Image is not None:
        def write_gif(frame_paths, gif_path, duration_ms):
            images = [Image.open(path) for path in frame_paths]
            try:
                images[0].save(
                    gif_path,
                    save_all=True,
                    append_images=images[1:],
                    duration=duration_ms,
                    loop=0,
                )
            finally:
                # Pillow keeps file handles open until each image is closed.
                for image in images:
                    image.close()

        return write_gif

    try:
        import imageio.v2 as imageio
    except ImportError:
        return None

    def write_gif_imageio(frame_paths, gif_path, duration_ms):
        # imageio expresses frame duration in seconds.
        frames = [imageio.imread(path) for path in frame_paths]
        imageio.mimsave(gif_path, frames, duration=duration_ms / 1000.0, loop=0)

    return write_gif_imageio


def main(argv=None):
    args = build_arg_parser().parse_args(argv)

    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        print(f"Error: run directory not found: {run_dir}")
        return 1

    run_path = run_dir / RUN_METADATA_FILENAME
    steps_path = run_dir / STEPS_FILENAME
    missing = [path.name for path in (run_path, steps_path) if not path.is_file()]
    if missing:
        print(f"Error: {run_dir} is missing {', '.join(missing)}; not a run directory.")
        return 1

    if args.gif and args.fps <= 0:
        print(f"Error: --fps must be positive, got {args.fps}.")
        return 1

    try:
        visualizer = TrajectoryVisualizer.from_run_log(run_dir)
        frame_paths = visualizer.render_from_csv(steps_path)
    except ValueError as exc:
        print(f"Error: cannot replay {steps_path}: {exc}")
        return 1
    except (OSError, yaml.YAMLError) as exc:
        print(f"Error: cannot read run log in {run_dir}: {exc}")
        return 1

    print(f"Rendered {len(frame_paths)} frame(s) to {visualizer.output_dir}")
    if not frame_paths:
        print("Warning: steps.csv contained no rows; nothing to animate.")
        return 0

    if args.gif:
        write_gif = _load_gif_backend()
        if write_gif is None:
            print(
                "Warning: neither Pillow nor imageio is installed; "
                f"skipping GIF generation (frames are still in {visualizer.output_dir})."
            )
            return 0

        gif_path = run_dir / GIF_FILENAME
        write_gif(frame_paths, gif_path, int(1000 / args.fps))
        print(f"Saved GIF animation to {gif_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
