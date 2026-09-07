import sys
import os
import csv
import time
import datetime
import math

from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import time
from envs.planning_env import RoombaPlanningEnv
from planners.mcts import MCTSSolver, MCTSNode
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

def main():
    # Seed NumPy and Torch from config before any sampling or search occurs.
    import yaml
    with open("configs/config.yaml", "r") as f:
        _config = yaml.safe_load(f)
    seed_everything(_config.get('env', {}).get('seed', 0))

    print("Initializing Planning Environment...")
    # NOTE: num_planning_envs (16) must be >= the discrete action grid size (15).
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
    if initial_dist < 0.5:
        print("Sampled start already satisfies the goal; skipping search.")
        env.close()
        return

    # 4. Setup Logging Infrastructure
    log_dir = Path("logs/mcts")
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_file_path = log_dir / f"mcts_run_{timestamp}.csv"
    
    print(f"Logging initialized at: {log_file_path}")
    print("Starting execution loop...")

    # Execution horizon + no-progress detection (config-driven).
    env_cfg = env.config.get('env', {})
    max_execution_steps = env_cfg.get('max_execution_steps', 200)
    no_progress_steps = env_cfg.get('no_progress_steps', 50)
    no_progress_threshold = env_cfg.get('no_progress_threshold', 0.05)

    steps = 0
    best_dist = float('inf')
    steps_since_improvement = 0
    try:
        with open(log_file_path, mode="a", newline="") as log_file:
            csv_writer = csv.writer(log_file)
            csv_writer.writerow([
                "Step",
                "Action_V",
                "Action_W",
                "Root_Value",
                "Tree_Size",
                "Max_Depth",
                "Dist_To_Goal",
                "Search_Time_sec",
                "Exec_Time_sec",
                "Num_Expansions",
                "PhysX_Ticks",
                "Decisions_Per_Sec",
            ])
            log_file.flush()

            total_start = time.perf_counter()
            while steps < max_execution_steps:
                # --- SEARCH (time + physics ticks isolated) ---
                root = MCTSNode(state=current_state.clone())
                search_start = time.perf_counter()
                ticks_before_search = env.sim.physics_tick_count
                best_action_idx = mcts.search(root)
                search_time = time.perf_counter() - search_start
                search_ticks = env.sim.physics_tick_count - ticks_before_search

                best_action = mcts.actions[best_action_idx]
                
                # --- EXECUTE SELECTED ACTION (time + physics ticks isolated) ---
                action_batch = best_action.repeat(env.num_envs, 1)
                state_batch = current_state.repeat(env.num_envs, 1)
                exec_start = time.perf_counter()
                ticks_before_exec = env.sim.physics_tick_count
                next_states, _, _, dones = env.generate(state_batch, action_batch, compute_observation=False)
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
                
                # --- LOGGING ---
                # Write to CSV and immediately flush to disk to protect against KeyboardInterrupt
                csv_writer.writerow([
                    steps, 
                    round(best_action[0].item(), 2), 
                    round(best_action[1].item(), 2), 
                    round(root_value, 4), 
                    tree_size, 
                    max_depth,
                    round(dist_to_goal, 4), 
                    round(search_time, 6), 
                    round(exec_time, 6), 
                    num_expansions, 
                    physx_ticks,
                    round(decisions_per_sec, 4),
                ])
                log_file.flush()

                print(f"Step: {steps:03d} | v={best_action[0]:.2f}, w={best_action[1]:.2f} | Search: {search_time:.3f}s, Exec: {exec_time:.3f}s")
                
                if dones[0].item():
                    print("Goal Reached! Exiting...")
                    break

                # No-progress detection: fail if the robot is not closing in on the goal.
                if dist_to_goal < best_dist - no_progress_threshold:
                    best_dist = dist_to_goal
                    steps_since_improvement = 0
                else:
                    steps_since_improvement += 1
                    if steps_since_improvement >= no_progress_steps:
                        print("No progress toward goal; terminating episode.")
                        break
            else:
                print(f"Reached execution horizon ({max_execution_steps} steps); terminating episode.")

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        env.close()

if __name__ == "__main__":
    main()
