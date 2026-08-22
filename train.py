import time
from env import RoombaEnv
import torch


def main():
    print("Initializing Roomba Environment...")
    env = RoombaEnv(config_path="config.yaml", sim_device="cuda:0", show_viewer=True)
    
    # Run an initial reset
    obs = env.reset()
    print("Environment initialized and reset successfully.")
    
    num_envs = env.num_envs
    device = env.device
    
    steps = 0
    start_time = time.time()

    actions = (torch.rand((num_envs, 2), device=device) * 2.0) - 1.0
    actions[:, 0] = torch.rand(num_envs, device=device) * 0.7 + 0.3  # Forward bias: 0.3 to 1.0
    actions[:, 1] = (torch.rand(num_envs, device=device) * 2.0) - 1.0  # Turn bias: -1.0 to 1.0
    
    try:
        while True:
# train.py
            if steps % 60 == 0:
                actions[:, 0] = torch.rand(num_envs, device=device) * 0.7 + 0.3  # Forward bias: 0.3 to 1.0
                actions[:, 1] = (torch.rand(num_envs, device=device) * 2.0) - 1.0  # Turn bias: -1.0 to 1.0
                
            # Step the environment
            result = env.step(actions)
            
            # Handle user closing the viewer window
            if result is False:
                print("Viewer closed by user. Exiting...")
                break
                
            obs, rewards, dones, info = result
            steps += 1
            
            # Print FPS every 100 simulation steps
            if steps % 100 == 0:
                elapsed = time.time() - start_time
                fps = (100 * num_envs) / elapsed
                
                # Check how many robots reached their goal in this batch
                successes = info["success"].sum().item()
                
                print(f"Step: {steps} | FPS: {fps:.2f} | Env Successes: {successes}/{num_envs}")
                start_time = time.time()
                
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    
    finally:
        env.close()

if __name__ == "__main__":
    main()
    