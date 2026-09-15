"""Open-loop action-sequence replay harness for the planning pipeline.

Purpose
-------
This is `scripts/run_mcts.py` with the MCTS search removed and replaced by a
fixed list of actions, picked by toggle from `ACTION_MODES` (below). Everything
else is kept as close to the original as possible on purpose: the same
environment, the same 15D state construction, the same single `env.generate()`
call per macro-action, the same `current_state = next_states[0]` bookkeeping, and
the same logging chain (`MCTSRunLogger` -> `run.yaml` + `steps.csv` + TensorBoard
+ `frames/`).

It exists to answer physical questions about the transition model — for example
whether `[0, -1]` turns the robot faster than `[0, 1]` — where the planner is a
confounder that has to be held constant and removed.

Log-chain parity, and its limits
--------------------------------
Both scripts write exactly the 16 `STEP_COLUMNS`, in the same order, through the
same `log_step` call, so a `logs/debug_mcts` CSV and a `logs/mcts` CSV are
directly comparable for the physical columns (`step`, `robot_x`, `robot_z`,
`robot_yaw`, `action_*`, `distance_to_goal_m`, `executed_reward`,
`execution_time_sec`, `physics_ticks`, `decisions_per_sec`). The search columns
(`root_value`, `tree_size`, `max_depth`, `expansion_calls`) and `search_time_sec`
have no search behind them and are written with the same zero sentinels the
Step 0 spawn row uses, so no value can be mistaken for a measurement. `run.yaml`
records the selected mode name, the full action list and its length, so a log is
self-describing.

Deliberate differences from `run_mcts.py` (the complete list):
  1. No `MCTSSolver` is constructed. `configs/planners.yaml` is still loaded and
     recorded in `run.yaml`, so a log says which action grid and hyperparameters
     the run would have used, but nothing is searched.
  2. The executed action is `ACTION_SEQUENCE[steps]` instead of
     `mcts.actions[mcts.search(root)]`.
  3. The loop bound is `len(ACTION_SEQUENCE)` instead of `max_execution_steps`.
  4. The no-progress detector is gone, along with its `TERMINATION_NO_PROGRESS`
     import: it would truncate exactly the turn-in-place and reverse sequences
     this script exists to run, and would record `termination_reason:
     no_progress` for them.
  5. Steps, rewards and `dones` are otherwise untouched, so `termination_reason`
     is still `goal_reached` on success and `execution_horizon` when the list
     runs out.

Comparability caveat
--------------------
`RoombaPlanningEnv.set_states` currently leaves `sim.reset_dof_states()` disabled
(commented out). With it enabled, every `generate()` call zeroes the wheel DOF
state before applying targets, so no run propagates wheel state across a
teleport; with it disabled, this script replays genuine continuous physics from a
single spawn, while `run_mcts.py` teleports carry per-batch-slot wheel hysteresis
into each transition. Record which configuration a measurement was taken under:
the two are not interchangeable.

Diagnostics
-----------
`steps.csv` alone cannot show *which* stage of the command chain is asymmetric,
so each step also prints, and mirrors into the PNG HUD, the commanded wheel
speeds, the achieved wheel speeds, the per-step delta-yaw and all four
quaternion components. Two measurement notes:
  * `robot_yaw` uses the two-component `atan2(-2 qy qw, qw^2 - qy^2)` form, which
    is exact only while `qx = qz = 0`. A robot that pitches or rolls under the
    turning couple biases the *recorded* yaw asymmetrically; the printed
    quaternion columns are what distinguishes that artifact from real physics.
  * The raw yaw is not unwrapped, so per-step delta-yaw is wrap-corrected into
    [-pi, pi) before accumulating; turns of ~1 rad per step are far below pi, so
    the correction is exact rather than approximate.
The run ends with a per-phase turn table and a turn-rate symmetry check that
compares the total angle swept by the +omega and -omega phases.
"""
import sys
import time
import math
import yaml

from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from envs.planning_env import RoombaPlanningEnv
from tracking.run_logger import (
    MCTSRunLogger,
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_INTERRUPTED,
    TERMINATION_ERROR,
    TERMINATION_EXECUTION_HORIZON,
    TERMINATION_GOAL_REACHED,
    TERMINATION_INTERRUPTED,
)
import torch
import numpy as np


# =============================================================================
# ACTION MODES — the only input to this script
# =============================================================================
# Each mode is a sequence of `[linear_velocity, angular_velocity]` pairs,
# normalized to [-1, 1] exactly like `mcts.actions` in `configs/planners.yaml`,
# so a sequence replays comparable actions to a logged MCTS run. Index i of the
# selected sequence is executed at step i+1.
#
# Toggle which mode runs by editing `ACTION_MODE`; nothing else in the file needs
# to change. Every mode is validated the same way (see
# `_validate_action_sequence`), which mirrors `MCTSSolver._create_action_grid`
# with one deliberate relaxation: the stationary `[0.0, 0.0]` action is allowed
# here. The ban in `planners.yaml` is a search-design decision (it would let the
# planner park forever); it is not an environment constraint, and a settle phase
# between turn phases is what makes the turn totals below comparable.
#
# `turn_symmetry`: the turn-in-place probe. `TURN_OMEGA = 1.0` means
# `1.0 * robot.max_angular_velocity` (2.0 rad/s) held for
# `macro_action_ticks * dt` (15 / 30 = 0.5 s), i.e. about 1.0 rad per step. The
# stationary gaps separate the phases and let the wheels settle, so a phase's
# total turn is not contaminated by the previous phase's decay. The block is
# repeated so a slow asymmetry has time to show up rather than being averaged
# away by one short phase.
TURN_OMEGA = 1.0
TURN_STEPS = 4
SETTLE_STEPS = 2

ACTION_MODES = {
    "turn_symmetry": (
        [[0.0, +TURN_OMEGA]] * TURN_STEPS + [[0.0, 0.0]] * SETTLE_STEPS
        + [[0.0, -TURN_OMEGA]] * TURN_STEPS + [[0.0, 0.0]] * SETTLE_STEPS
    ) * 2,
    # Placeholder for ad-hoc sequences: replace the body with the [v, w] pairs to
    # replay. Validation applies unchanged.
    "custom": [
        [0.0, 0.0],
    ],
}

# Toggle: the key of the `ACTION_MODES` entry to replay. Direct-indexed with an
# explicit key list so a typo fails fast here instead of silently running a
# different sequence than the one written in `run.yaml`.
ACTION_MODE = "turn_symmetry"
if ACTION_MODE not in ACTION_MODES:
    raise KeyError(
        f"Unknown ACTION_MODE {ACTION_MODE!r}; available modes: {sorted(ACTION_MODES)}."
    )
ACTION_SEQUENCE = ACTION_MODES[ACTION_MODE]


def _validate_action_sequence(sequence):
    """Validate an action list against the rules the planner's action grid obeys.

    Returns the sequence as a list of `[float, float]` pairs. A malformed entry
    fails fast here rather than being clamped inside `generate()`, which would
    silently turn an out-of-range request such as `[0.0, 2.0]` into
    `[0.0, 1.0]` and produce a valid-looking run that measured a different
    action than the one written down.
    """
    if not isinstance(sequence, (list, tuple)):
        raise TypeError(
            f"ACTION_SEQUENCE must be a list of [v, w] pairs; got {type(sequence).__name__}."
        )
    if len(sequence) == 0:
        raise ValueError("ACTION_SEQUENCE is empty; add at least one [v, w] pair.")

    validated = []
    for index, entry in enumerate(sequence):
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            raise ValueError(
                f"ACTION_SEQUENCE[{index}] must be a [linear_velocity, angular_velocity] "
                f"pair; got {entry!r}."
            )
        for component in entry:
            if isinstance(component, bool) or not isinstance(component, (int, float)):
                raise TypeError(
                    f"ACTION_SEQUENCE[{index}] must contain numbers; got {component!r}."
                )
            if not -1.0 <= float(component) <= 1.0:
                raise ValueError(
                    f"ACTION_SEQUENCE[{index}] component {component!r} is outside [-1, 1]. "
                    "Actions are normalized; scaling happens in generate()."
                )
        validated.append([float(entry[0]), float(entry[1])])
    return validated


def seed_everything(seed: int):
    """Seed NumPy and Torch for reproducible start/goal sampling and expansion order."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def extract_yaw_from_quaternion(qy: float, qw: float) -> float:
    """Compute planar yaw angle (radians) in 2D top-down (X=horizontal, Z=vertical) frame.

    In Y-up frame (+X forward, +Z right), a CCW yaw rotates forward (+X) into -Z.
    """
    return math.atan2(-2.0 * qy * qw, qw * qw - qy * qy)

def _finalize_safely(run_logger, **kwargs):
    """Finalize a run without masking the exception that triggered finalization."""
    try:
        run_logger.finalize(**kwargs)
    except Exception as exc:
        print(f"Warning: failed to finalize run log: {exc}")


def _wrap_angle(angle: float) -> float:
    """Wrap an angle in radians into [-pi, pi]."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _print_phase_summary(phases):
    """Tabulate per-phase turn totals, then compare +omega and -omega phases.

    `phases` is the list built by the execution loop: one entry per contiguous run
    of an identical action, each holding its step range, its total absolute turn
    (the angle swept — the honest measure of "how fast it turns"), and its net
    signed turn (which cancels for a symmetric probe, so a non-zero net exposes a
    side that turns further in one direction).

    The symmetry check compares rotate-in-place phases only (`|v| == 0`,
    `|w| > 0`); a phase with a linear component folds translation into the turn
    and is not a clean left/right comparison.
    """
    if not phases:
        print("\nPhase summary: no phases recorded.")
        return

    print("\n=== Phase summary (contiguous runs of one action) ===")
    print(
        f"{'steps':>11}  {'v':>6}  {'w':>6}  {'n':>3}  "
        f"{'sum|dyaw|':>10}  {'net dyaw':>9}  {'rad/step':>9}"
    )
    for phase in phases:
        count = phase["steps"]
        mean_turn = phase["abs_turn"] / count if count else 0.0
        print(
            f"{phase['first_step']:>4}-{phase['last_step']:<4}  "
            f"{phase['action'][0]:>6.2f}  {phase['action'][1]:>6.2f}  "
            f"{count:>3}  {phase['abs_turn']:>10.4f}  "
            f"{phase['net_turn']:>9.4f}  {mean_turn:>9.5f}"
        )

    rotational = [
        phase for phase in phases
        if abs(phase["action"][0]) < 1e-9 and abs(phase["action"][1]) > 1e-9
    ]
    if not rotational:
        print("\nTurn symmetry: the sequence has no rotate-in-place phase.")
        return

    print("\n=== Turn symmetry (rotate-in-place phases only: v == 0, w != 0) ===")
    mean_turns = {}
    for label, group in (
        ("w > 0", [phase for phase in rotational if phase["action"][1] > 0.0]),
        ("w < 0", [phase for phase in rotational if phase["action"][1] < 0.0]),
    ):
        count = sum(phase["steps"] for phase in group)
        swept = sum(phase["abs_turn"] for phase in group)
        net = sum(phase["net_turn"] for phase in group)
        mean_turns[label] = swept / count if count else 0.0
        print(
            f"  {label}: {count:>3} steps | total |dyaw| = {swept:8.4f} rad | "
            f"mean = {mean_turns[label]:.5f} rad/step | net dyaw = {net:+.4f} rad"
        )

    positive_mean = mean_turns["w > 0"]
    negative_mean = mean_turns["w < 0"]
    if positive_mean > 0.0 and negative_mean > 0.0:
        ratio = negative_mean / positive_mean
        print(f"\n  ratio mean|dyaw|(w<0) / mean|dyaw|(w>0) = {ratio:.5f}")
        print("  A ratio above 1 means the negative-omega turn sweeps more angle per step.")
    else:
        print("\n  Ratio not reported: one side has no recorded rotate-in-place step.")


def main():
    # Seed NumPy and Torch from config before any sampling or search occurs.
    with open("configs/experiments.yaml", "r") as f:
        experiment_config = yaml.safe_load(f)
    
    debug_mcts_config = experiment_config["debug_mcts"]
    seed = debug_mcts_config["seed"]

    # Validate the sequence before any GPU work, so a typo costs nothing.
    validated_actions = _validate_action_sequence(ACTION_SEQUENCE)

    # Seed before start/goal sampling occurs. `run_mcts.py` also seeds the
    # randomized MCTS expansion order; with no search only the sampling below is
    # affected, and the call is kept so the same seed yields the same start/goal
    # pair in both scripts.
    seed_everything(seed)

    print("Initializing Planning Environment...")
    # NOTE: num_planning_envs (9) must be >= the discrete action grid size (5).
    # Pre-initialised so the exception handlers can finalize safely even if the
    # failure happens before the execution loop starts.
    env = None
    run_logger = None
    steps = 0
    final_dist = None
    # Phase accounting, pre-initialised so the interrupt handler can report
    # whatever was actually executed even if the loop never started.
    phases = []
    try:
        env = RoombaPlanningEnv(config_path="configs/config.yaml", sim_device="cuda:0", show_viewer=True)

        # No solver is built. `configs/planners.yaml` is read so `run.yaml` still
        # records the action grid and hyperparameters the run would have used.
        with open("configs/planners.yaml", "r") as f:
            planner_config = yaml.safe_load(f)

        # `generate()` writes wheel DOF index 0 as the left wheel and index 1 as
        # the right wheel; every wheel reading below is interpreted with that
        # mapping, so print the asset's own DOF order instead of assuming it.
        try:
            print(f"Robot DOF order: {list(env.sim.gym.get_asset_dof_names(env.sim.robot_asset))}")
        except Exception as exc:
            # Diagnostic only: a missing query must not abort an otherwise valid run.
            print(f"Warning: could not query robot DOF names: {exc}")

        # Materialize the validated sequence on the device once. `action_tensor[i]`
        # is exactly the normalized `[v, w]` pair `generate()` expects.
        action_tensor = torch.tensor(validated_actions, dtype=torch.float32, device=env.device)

        # 1. Acquire prior knowledge (Map and Goals)
        prior_knowledge = env.get_prior_knowledge()
        occupancy_map = prior_knowledge["occupancy_map"]

        # 2. Sample valid physical coordinates with a minimum start-goal separation
        start_x, start_z, goal_x, goal_z = occupancy_map.sample_valid_start_goal()

        # 3. Construct the exact 15D state tensor
        current_state = torch.zeros(15, dtype=torch.float32, device=env.device)
        current_state[0] = start_x
        current_state[1] = 0.065  # Spawn almost exactly at resting height
        current_state[2] = start_z
        current_state[6] = 1.0  # qw (neutral rotation)
        current_state[13] = goal_x
        current_state[14] = goal_z

        # Teleport the robot to the start pose (spawned at resting height).
        env.set_states(current_state.repeat(env.num_envs, 1))

        # Derive the initial terminal flag from the distance to goal.
        initial_dx = current_state[13] - current_state[0]
        initial_dz = current_state[14] - current_state[2]
        initial_dist = math.hypot(initial_dx.item(), initial_dz.item())
        if initial_dist < env.goal_radius:
            print("Sampled start already satisfies the goal; skipping search.")
            return

        # 4. Setup Run Logging
        # Each run owns a directory holding the authoritative run.yaml and steps.csv
        # records, plus a derived TensorBoard event directory and/or 2D frame
        # flipbook when those logging options are enabled.
        logging_cfg = debug_mcts_config["logging"]
        room_bounds = (float(env.sim.room.width), float(env.sim.room.depth))
        obstacles = [
            {"x": float(o.x), "z": float(o.z), "width": float(o.width), "depth": float(o.depth)}
            for o in env.sim.room.obstacles
            if hasattr(o, "width")
        ]
        run_logger = MCTSRunLogger(
            output_dir=logging_cfg["output_dir"],
            environment_config=env.config,
            planner_config=planner_config,
            experiment_config=experiment_config,
            scenario={
                "seed": seed,
                # The selected `ACTION_MODES` key. This runner is an open-loop
                # replay, so the search columns in steps.csv below are zero
                # sentinels rather than measurements, and `actions` is the whole
                # command.
                "mode": ACTION_MODE,
                "action_count": len(validated_actions),
                "actions": validated_actions,
                "start": {"x": start_x, "z": start_z},
                "goal": {"x": goal_x, "z": goal_z},
            },
            initial_distance=initial_dist,
            enable_tensorboard=bool(logging_cfg["tensorboard"]),
            enable_visualizer=bool(logging_cfg.get("render_frames", False)),
            room_bounds=room_bounds,
            obstacles=obstacles,
            goal=(goal_x, goal_z),
            goal_radius=env.goal_radius,
        )

        # Step 0 records the spawn pose before any action is taken, so a flipbook
        # starts at the sampled start instead of the first post-action state.
        initial_yaw = extract_yaw_from_quaternion(current_state[4].item(), current_state[6].item())
        run_logger.log_step(
            {
                "step": 0,
                "robot_x": round(start_x, 4),
                "robot_z": round(start_z, 4),
                "robot_yaw": round(initial_yaw, 4),
                "action_v_normalized": 0.0,
                "action_w_normalized": 0.0,
                "root_value": 0.0,
                "tree_size": 1,
                "max_depth": 0,
                "distance_to_goal_m": round(initial_dist, 4),
                "executed_reward": 0.0,
                "search_time_sec": 0.0,
                "execution_time_sec": 0.0,
                "expansion_calls": 0,
                "physics_ticks": 0,
                "decisions_per_sec": 0.0,
            },
            hud_metrics={"Status": "Spawn Pose", "Dist": f"{initial_dist:.2f}m"},
        )


        print(f"Logging initialized at: {run_logger.run_dir}")
        print("Starting execution loop...")

        print(
            f"Open-loop action sequence ({len(validated_actions)} macro-actions): "
            + " ".join(f"[{v:+.1f},{w:+.1f}]" for v, w in validated_actions)
        )

        total_start = time.perf_counter()

        # Default outcome if the loop runs out of actions without breaking.
        termination_reason = TERMINATION_EXECUTION_HORIZON

        # Turn accounting, accumulated from the wrap-corrected per-step delta-yaw.
        # `phases` groups consecutive steps that ran the same action, which is what
        # makes one turn phase directly comparable to another.
        prev_yaw = initial_yaw
        cumulative_yaw = 0.0
        current_phase = None
        prev_action_key = None

        while steps < len(validated_actions):
            # --- ACTION SELECTION (the one line run_mcts.py does not have) ---
            action = action_tensor[steps]

            # --- EXECUTE SELECTED ACTION (time + physics ticks isolated) ---
            action_batch = action.repeat(env.num_envs, 1)
            state_batch = current_state.repeat(env.num_envs, 1)
            exec_start = time.perf_counter()
            ticks_before_exec = env.sim.physics_tick_count
            next_states, _, exec_rewards, dones = env.generate(state_batch, action_batch, compute_observation=False)
            exec_time = time.perf_counter() - exec_start
            exec_ticks = env.sim.physics_tick_count - ticks_before_exec

            # The true state becomes the state of the first environment
            current_state = next_states[0].clone()

            steps += 1

            # --- METRIC GATHERING ---
            decisions_per_sec = steps / (time.perf_counter() - total_start)

            # Physical progress (Euclidean distance to goal)
            dx = current_state[13] - current_state[0]
            dz = current_state[14] - current_state[2]
            dist_to_goal = math.hypot(dx.item(), dz.item())
            final_dist = dist_to_goal

            # Planar pose of the executed state. The yaw uses the same planar
            # convention as the frame renderer, so the drawn heading matches the
            # physical robot orientation.
            robot_x = current_state[0].item()
            robot_z = current_state[2].item()
            robot_yaw = extract_yaw_from_quaternion(current_state[4].item(), current_state[6].item())
            executed_reward = float(exec_rewards[0].item())

            # --- DIAGNOSTICS (stdout + HUD only, never a column in steps.csv) ---
            # Delta-yaw: the recorded yaw is not unwrapped, so a spin past +-pi
            # would show as a sign flip. A turn of ~1 rad per step is far below
            # pi, so wrapping into [-pi, pi) is exact rather than approximate.
            delta_yaw = _wrap_angle(robot_yaw - prev_yaw)
            cumulative_yaw += delta_yaw
            prev_yaw = robot_yaw

            # Commanded vs achieved wheel speeds. `generate()` has already applied
            # the targets for the whole batch, and the DOF refresh reads back what
            # the solver actually reached over the macro-action: equal commanded
            # magnitudes with unequal achieved magnitudes is a physics asymmetry,
            # while unequal commanded magnitudes is an action-mapping asymmetry.
            env.sim.gym.refresh_dof_state_tensor(env.sim.sim)
            commanded_left, commanded_right = env.sim.dof_velocity_targets_view[0].tolist()
            achieved_left, achieved_right = env.sim.dof_states[0, :, 1].tolist()

            # [qx, qy, qz, qw]. `robot_yaw` uses the two-component form above,
            # which is exact only while qx = qz = 0; a robot that pitches or rolls
            # under the turning couple biases the recorded yaw, and these columns
            # are what separates that artifact from real physics asymmetry.
            qx, qy, qz, qw = (round(float(current_state[i].item()), 5) for i in range(3, 7))

            # --- PHASE ACCOUNTING ---
            action_key = (round(float(action[0]), 4), round(float(action[1]), 4))
            if action_key != prev_action_key:
                current_phase = {
                    "action": action_key,
                    "first_step": steps,
                    "last_step": steps,
                    "steps": 0,
                    "abs_turn": 0.0,
                    "net_turn": 0.0,
                }
                phases.append(current_phase)
                prev_action_key = action_key
            current_phase["last_step"] = steps
            current_phase["steps"] += 1
            current_phase["abs_turn"] += abs(delta_yaw)
            current_phase["net_turn"] += delta_yaw

            # HUD card contents: observability aids only, never a data source.
            hud_metrics = {
                "Step": f"{steps:03d}/{len(validated_actions):03d}",
                "Action": f"v={action[0]:+.1f}, w={action[1]:+.1f}",
                "Dist": f"{dist_to_goal:.2f}m",
                "Yaw": f"{math.degrees(robot_yaw):+7.2f} deg",
                "dYaw": f"{math.degrees(delta_yaw):+6.2f} deg",
                "Wheel cmd L/R": f"{commanded_left:+.1f}/{commanded_right:+.1f}",
                "Wheel act L/R": f"{achieved_left:+.1f}/{achieved_right:+.1f}",
                "Reward": f"{executed_reward:+.3f}",
            }

            # --- LOGGING ---
            # Same 16 STEP_COLUMNS, same order, same single call as run_mcts.py, so
            # the physical columns stay comparable. CSV is flushed per row to
            # protect against KeyboardInterrupt. The search columns have no search
            # behind them and carry the same zero sentinels the Step 0 spawn row
            # uses, so no value here can be mistaken for a measurement.
            run_logger.log_step(
                {
                    "step": steps,
                    "robot_x": round(robot_x, 4),
                    "robot_z": round(robot_z, 4),
                    "robot_yaw": round(robot_yaw, 4),
                    "action_v_normalized": round(float(action[0]), 2),
                    "action_w_normalized": round(float(action[1]), 2),
                    "root_value": 0.0,
                    "tree_size": 1,
                    "max_depth": 0,
                    "distance_to_goal_m": round(dist_to_goal, 4),
                    "executed_reward": round(executed_reward, 4),
                    "search_time_sec": 0.0,
                    "execution_time_sec": round(exec_time, 6),
                    "expansion_calls": 0,
                    "physics_ticks": exec_ticks,
                    "decisions_per_sec": round(decisions_per_sec, 4),
                },
                hud_metrics=hud_metrics,
            )

            print(
                f"Step {steps:03d}/{len(validated_actions):03d}"
                f" | act v={action[0]:+.2f} w={action[1]:+.2f}"
                f" | yaw {math.degrees(robot_yaw):+8.2f} deg"
                f" (d {math.degrees(delta_yaw):+7.2f}, cum {math.degrees(cumulative_yaw):+9.2f})"
                f" | wheel cmd L {commanded_left:+6.2f} R {commanded_right:+6.2f}"
                f" | act L {achieved_left:+6.2f} R {achieved_right:+6.2f}"
                f" | quat x {qx:+.4f} y {qy:+.4f} z {qz:+.4f} w {qw:+.4f}"
                f" | dist {dist_to_goal:.3f} m | reward {executed_reward:+.3f}"
                f" | {exec_time:.3f} s / {exec_ticks} ticks"
            )

            if dones[0].item():
                print("Goal Reached! Exiting...")
                termination_reason = TERMINATION_GOAL_REACHED
                break
        else:
            print(
                f"Executed the full ACTION_SEQUENCE ({len(validated_actions)} actions); "
                "terminating episode."
            )

        # The phase table is derived reporting on top of the authoritative CSV and
        # run.yaml, so it is shielded: a formatting failure here must not turn a
        # completed run into a failed one.
        try:
            _print_phase_summary(phases)
        except Exception as exc:
            print(f"Warning: failed to print the phase summary: {exc}")

        run_logger.finalize(
            status=STATUS_COMPLETED,
            termination_reason=termination_reason,
            success=(termination_reason == TERMINATION_GOAL_REACHED),
            steps=steps,
            final_distance=final_dist,
        )

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        if run_logger is not None:
            # Success is indeterminate for an interrupted run, so it stays null.
            _finalize_safely(
                run_logger,
                status=STATUS_INTERRUPTED,
                termination_reason=TERMINATION_INTERRUPTED,
                steps=steps,
                final_distance=final_dist,
            )
        # Report whatever was actually executed: a long sequence stopped early is
        # still a usable measurement. Printed after finalize so it cannot affect
        # the recorded outcome, and shielded for the same reason.
        if phases:
            try:
                _print_phase_summary(phases)
            except Exception as exc:
                print(f"Warning: failed to print the phase summary: {exc}")
    except Exception:
        # Best-effort finalization, then let the original error propagate.
        if run_logger is not None:
            _finalize_safely(
                run_logger,
                status=STATUS_FAILED,
                termination_reason=TERMINATION_ERROR,
                steps=steps,
                final_distance=final_dist,
            )
        raise
    finally:
        if run_logger is not None:
            run_logger.close()
        if env is not None:
            env.close()

if __name__ == "__main__":
    main()
