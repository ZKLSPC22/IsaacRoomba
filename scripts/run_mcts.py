import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import time
from envs.planning_env import RoombaPlanningEnv
from planners.mcts import MCTSSolver, MCTSNode
import torch


def main():
    print("Initializing Planning Environment...")
    # NOTE: Ensure your configs/config.yaml has the number of planning envs set to 64!
    env = RoombaPlanningEnv(config_path="configs/config.yaml", sim_device="cuda:0", show_viewer=True)
    
    print("Initializing Leaf-Parallel MCTS...")
    # Requires configs/planners.yaml to exist
    mcts = MCTSSolver(env, config_path="configs/planners.yaml")
    
    # 1. Acquire prior knowledge (Map and Goals)
    prior_knowledge = env.get_prior_knowledge()
    occupancy_map = prior_knowledge["occupancy_map"]
    
    # 2. Sample valid physical coordinates
    start_x, start_z = occupancy_map.sample_valid_pose()
    goal_x, goal_z = occupancy_map.sample_valid_pose()
    
    # 3. Construct the exact 15D state tensor
    current_state = torch.zeros(15, dtype=torch.float32, device=env.device)
    current_state[0] = start_x
    current_state[1] = 0.5  # Drop height
    current_state[2] = start_z
    current_state[6] = 1.0  # qw (neutral rotation)
    current_state[13] = goal_x
    current_state[14] = goal_z

    print("Starting execution loop...")
    steps = 0
    try:
        while True:
            start_time = time.time()
            
            # Wrap the exact state in a root node and search
            root = MCTSNode(state=current_state.clone())
            best_action_idx = mcts.search(root)
            best_action = mcts.actions[best_action_idx]
            
            # Broadcast the chosen action and state to the batch to advance physics
            action_batch = best_action.repeat(env.num_envs, 1)
            state_batch = current_state.repeat(env.num_envs, 1)
            
            # Step the generative model to advance reality
            next_states, _, _, dones = env.generate(state_batch, action_batch)
            
            # The true state becomes the state of the first environment
            current_state = next_states[0].clone()
            
            steps += 1
            elapsed = time.time() - start_time
            print(f"Step: {steps:03d} | Action: v={best_action[0]:.2f}, w={best_action[1]:.2f} | Planning Time: {elapsed:.2f}s")
            
            if dones[0].item():
                print("Goal Reached! Exiting...")
                break

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        env.close()

if __name__ == "__main__":
    main()