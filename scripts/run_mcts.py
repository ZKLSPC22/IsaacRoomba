import sys
import time
import math
import yaml

from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from envs.planning_env import RoombaPlanningEnv
from planners.mcts import MCTSSolver, MCTSNode
from tracking.run_logger import (
    MCTSRunLogger,
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_INTERRUPTED,
    TERMINATION_ERROR,
    TERMINATION_EXECUTION_HORIZON,
    TERMINATION_GOAL_REACHED,
    TERMINATION_INTERRUPTED,
    TERMINATION_NO_PROGRESS,
)
import torch
import numpy as np


def seed_everything(seed: int):
    """Seed NumPy and Torch for reproducible start/goal sampling and expansion order."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_nodes(node):
    """Recursively count the total number of nodes generated in the MCTS tree."""
    if not node.children:
        return 1
    return 1 + sum(count_nodes(child) for child in node.children.values())

def tree_depth(node):
    """Maximum depth of the MCTS tree rooted at `node` (root is depth 0)."""
    if not node.children:
        return 0
    return 1 + max(tree_depth(child) for child in node.children.values())

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


def main():
    # Seed NumPy and Torch from config before any sampling or search occurs.
    with open("configs/experiments.yaml", "r") as f:
        experiment_config = yaml.safe_load(f)

    # Runner settings are grouped under the planner key they apply to.
    mcts_experiment_cfg = experiment_config["mcts"]
    seed = mcts_experiment_cfg["seed"]
    max_execution_steps = mcts_experiment_cfg["max_execution_steps"]

    no_progress_config = mcts_experiment_cfg["no_progress"]
    no_progress_steps = no_progress_config["no_progress_steps"]
    no_progress_threshold = no_progress_config["no_progress_threshold"]

    # Seed before start/goal sampling or randomized MCTS expansion occurs.
    seed_everything(seed)

    print("Initializing Planning Environment...")
    # NOTE: num_planning_envs (9) must be >= the discrete action grid size (5).
    # Pre-initialised so the exception handlers can finalize safely even if the
    # failure happens before the execution loop starts.
    env = None
    run_logger = None
    steps = 0
    final_dist = None
    try:
        env = RoombaPlanningEnv(config_path="configs/config.yaml", sim_device="cuda:0", show_viewer=False)

        print("Initializing Leaf-Parallel MCTS...")
        # Requires configs/planners.yaml to exist
        mcts = MCTSSolver(env, config_path="configs/planners.yaml")

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
        logging_cfg = mcts_experiment_cfg["logging"]
        room_bounds = (float(env.sim.room.width), float(env.sim.room.depth))
        obstacles = [
            {"x": float(o.x), "z": float(o.z), "width": float(o.width), "depth": float(o.depth)}
            for o in env.sim.room.obstacles
            if hasattr(o, "width")
        ]
        run_logger = MCTSRunLogger(
            output_dir=logging_cfg["output_dir"],
            environment_config=env.config,
            planner_config=mcts.config,
            experiment_config=experiment_config,
            scenario={
                "seed": seed,
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

        best_dist = initial_dist
        steps_since_improvement = 0
        total_start = time.perf_counter()

        # Default outcome if the loop runs out its horizon without breaking.
        termination_reason = TERMINATION_EXECUTION_HORIZON

        while steps < max_execution_steps:
            # --- SEARCH (time + physics ticks isolated) ---
            root = MCTSNode(state=current_state.clone())
            search_start = time.perf_counter()
            ticks_before_search = env.sim.physics_tick_count
            best_action_idx = mcts.search(root)
            if best_action_idx is None:
                if root.is_terminal:
                    termination_reason = TERMINATION_GOAL_REACHED
                    break
                raise RuntimeError(
                    "Search ran on a terminal node or a node with no children."
                )
            search_time = time.perf_counter() - search_start
            search_ticks = env.sim.physics_tick_count - ticks_before_search

            best_action = mcts.actions[best_action_idx]

            # --- EXECUTE SELECTED ACTION (time + physics ticks isolated) ---
            action_batch = best_action.repeat(env.num_envs, 1)
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
            root_value = root.Q / root.N if root.N > 0 else 0.0
            tree_size = count_nodes(root)
            max_depth = tree_depth(root)
            num_expansions = mcts.num_expansions
            physx_ticks = search_ticks + exec_ticks
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

            # HUD card contents: observability aids only, never a data source.
            hud_metrics = {
                "Step": f"{steps:03d}",
                "Dist": f"{dist_to_goal:.2f}m",
                "Action": f"v={best_action[0]:.1f}, w={best_action[1]:.1f}",
                "Root Q/N": f"{root_value:.3f}",
                "Nodes": tree_size,
                "Expansions": num_expansions,
            }

            # --- LOGGING ---
            # One metrics dictionary feeds CSV, TensorBoard, and the optional frame
            # flipbook, so the three representations cannot drift. CSV is flushed
            # per row to protect against KeyboardInterrupt.
            run_logger.log_step(
                {
                    "step": steps,
                    "robot_x": round(robot_x, 4),
                    "robot_z": round(robot_z, 4),
                    "robot_yaw": round(robot_yaw, 4),
                    "action_v_normalized": round(best_action[0].item(), 2),
                    "action_w_normalized": round(best_action[1].item(), 2),
                    "root_value": round(root_value, 4),
                    "tree_size": tree_size,
                    "max_depth": max_depth,
                    "distance_to_goal_m": round(dist_to_goal, 4),
                    "executed_reward": round(float(exec_rewards[0].item()), 4),
                    "search_time_sec": round(search_time, 6),
                    "execution_time_sec": round(exec_time, 6),
                    "expansion_calls": num_expansions,
                    "physics_ticks": physx_ticks,
                    "decisions_per_sec": round(decisions_per_sec, 4),
                },
                hud_metrics=hud_metrics,
            )

            print(f"Step: {steps:03d} | v={best_action[0]:.2f}, w={best_action[1]:.2f} | Search: {search_time:.3f}s, Exec: {exec_time:.3f}s")

            if dones[0].item():
                print("Goal Reached! Exiting...")
                termination_reason = TERMINATION_GOAL_REACHED
                break

            # No-progress detection: fail if the robot is not closing in on the goal.
            if dist_to_goal < best_dist - no_progress_threshold:
                best_dist = dist_to_goal
                steps_since_improvement = 0
            else:
                steps_since_improvement += 1
                if steps_since_improvement >= no_progress_steps:
                    print("No progress toward goal; terminating episode.")
                    termination_reason = TERMINATION_NO_PROGRESS
                    break
        else:
            print(f"Reached execution horizon ({max_execution_steps} steps); terminating episode.")

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
