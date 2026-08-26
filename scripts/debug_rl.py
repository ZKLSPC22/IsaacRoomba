import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from envs.rl_env import RoombaRLEnv
import torch


def main():
    print("Initializing Interactive Debug Environment...")
    env = RoombaRLEnv(config_path="configs/config.yaml", sim_device="cuda:0", show_viewer=True)

    # Run an initial reset
    obs = env.reset()
    env.render_debug_visuals(obs)  # Render the first frame manually
    print("Environment initialized and reset successfully.")

    num_envs = env.num_envs
    device = env.device

    # --- Initial 1-second warmup / settling phase (60 steps) ---
    print("Warming up simulation for 1 second (60 steps)...")
    idle_action = torch.zeros((num_envs, 2), dtype=torch.float32, device=device)
    for _ in range(60):
        obs, rewards, dones, info = env.step(idle_action)

    print("\n--- Roomba Interactive Debugger ---")
    print("Commands: Enter linear velocity, angular velocity, and steps to simulate.")
    print("Example: '0.5 1.0 60' (Moves forward at 0.5 m/s, turns at 1 rad/s, for 60 steps)")
    print("Type 'q' to quit.\n")

    try:
        while True:
            user_input = input("Enter 'v omega steps' > ")

            if user_input.lower().strip() == 'q':
                break

            try:
                parts = user_input.split()
                v = float(parts[0])
                omega = float(parts[1])
                steps = int(parts[2]) if len(parts) > 2 else 60
            except (ValueError, IndexError):
                print("Invalid format. Please enter numbers like: 0.5 0.0 60")
                continue

            # Format the action tensor for the environment
            actions = torch.tensor([[v, omega]], dtype=torch.float32, device=device).repeat(num_envs, 1)

            # Execute the action repeatedly for the requested number of steps
            print(f"Executing v={v}, omega={omega} for {steps} steps...")
            viewer_closed = False
            is_bumped = False
            for _ in range(steps):
                result = env.step(actions)
                if result is False:
                    print("Viewer closed.")
                    viewer_closed = True
                    break
                obs, rewards, dones, info = result
                if "bumper" in obs:
                    is_bumped = obs["bumper"][0].item() > 0.2

            # ==============================
            # --- Debug Observations ---1
            # ==============================

            # 1. Bumper Observation Space
            if "bumper" in obs:
                print(f"Bumper Status: {'[COLLISION DETECTED]' if is_bumped else '[CLEAR]'}")
            else:
                print("Bumper Status: [SENSOR DISABLED]")

            # 2. Camera Observation Space
            if "camera" in obs:
                # Extract the final frame, move to CPU, and convert to standard OpenCV format (BGR)
                # Extract the final frame, move to CPU, and convert to standard OpenCV format (BGR)
                img_tensor = obs["camera"][0]
                img_np = img_tensor.cpu().numpy().astype(np.uint8)
                img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                
                print("Showing camera view. Press ANY KEY in the image window to continue.")
                cv2.imshow("Debug Camera View - Final Step", img_bgr)
                cv2.waitKey(0) # Halts execution until a key is pressed
                cv2.destroyWindow("Debug Camera View - Final Step")
            else:
                print("Camera Observation: [SENSOR DISABLED - check config.yaml]")

            if viewer_closed:
                break
    except KeyboardInterrupt:
        print("\nDebugger interrupted by user.")
    finally:
        env.close()

if __name__ == "__main__":
    main()
