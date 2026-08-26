'''
The LiDAR visualization code is completely bugged out, need thorough human revision.
'''


import math
import torch
import numpy as np
from isaacgym import gymapi

class DebugVisualizer:
    def __init__(self, gym, viewer):
        self.gym = gym
        self.viewer = viewer
        
        # Attempt to load OpenCV for 1st-person camera rendering
        self.cv2_imported = False
        try:
            import cv2
            self.cv2 = cv2
            self.cv2_imported = True
        except ImportError:
            print("Warning: OpenCV (cv2) not found. Camera 2D visualization will be skipped.")

    def clear(self):
        """Clears all previous lines from the viewer. Must be called once per frame."""
        if self.viewer is not None:
            self.gym.clear_lines(self.viewer)

    def draw_goals(self, envs, goals: torch.Tensor, radius=0.2, num_segments=16):
        if self.viewer is None: return
        
        num_envs = len(envs)
        
        # Precompute circle offsets on GPU
        angles = torch.linspace(0, 2 * math.pi, num_segments + 1, device=goals.device)
        cos_a = torch.cos(angles) * radius
        sin_a = torch.sin(angles) * radius
        
        gx = goals[:, 0].unsqueeze(1)
        gz = goals[:, 1].unsqueeze(1)
        gy = 0.02  
        
        # Calculate start and end points for each segment
        p1x = gx + cos_a[:-1]
        p2x = gx + cos_a[1:]
        p1z = gz + sin_a[:-1]
        p2z = gz + sin_a[1:]
        
        p1y = torch.full_like(p1x, gy)
        p2y = torch.full_like(p2x, gy)
        
        # Stack into [num_envs, num_segments, 6] -> flattened coordinates
        lines_tensor = torch.stack([p1x, p1y, p1z, p2x, p2y, p2z], dim=-1).view(num_envs, -1)
        
        # Move to CPU in one bulk operation, then loop over envs
        lines_np = lines_tensor.cpu().numpy()
        red = [1.0, 0.0, 0.0] * num_segments
        
        for i, env in enumerate(envs):
            self.gym.add_lines(self.viewer, env, num_segments, lines_np[i].tolist(), red)

    def draw_lidar(self, envs, robot_states: torch.Tensor, lidar_depths: torch.Tensor, fov_rad=120.0*(math.pi/180.0)):
        """Draws green lines from the robot to the ray hit points."""
        if self.viewer is None: return
        
        num_rays = lidar_depths.shape[1]
        
        # Ray angles relative to the robot's heading
        # ray_angles = torch.linspace(-fov_rad/2, fov_rad/2, num_rays, device=lidar_depths.device)


        # ==================================================
        # Projection math
        # ==================================================
        # Create an index tensor for the rays
        indices = torch.arange(num_rays, dtype=torch.float32, device=lidar_depths.device)

        # Map indices to normalized device coordinates [-1.0 to 1.0]
        ndc = (2.0 * indices - (num_rays - 1)) / (num_rays - 1)

        # Apply the pinhole projection formula
        ray_angles = torch.atan(ndc * math.tan(fov_rad / 2.0))
        # ==================================================
        # Projection math
        # ==================================================
        
        # Extract robot positions
        rx = robot_states[:, 0].unsqueeze(1)
        ry = robot_states[:, 1].unsqueeze(1)
        rz = robot_states[:, 2].unsqueeze(1)
        
        # Extract Yaw
        qy = robot_states[:, 4]
        qw = robot_states[:, 6]
        yaw = torch.atan2(2.0 * qy * qw, 1.0 - 2.0 * (qy**2)).unsqueeze(1)
        
        # Calculate absolute ray angles and endpoints
        global_angles = yaw + ray_angles
        end_x = rx + lidar_depths * torch.cos(global_angles)
        end_z = rz - lidar_depths * torch.sin(global_angles)
        
        # Expand rx, ry, rz to match ray dimensions
        rx_exp = rx.expand(-1, num_rays)
        ry_exp = ry.expand(-1, num_rays)
        rz_exp = rz.expand(-1, num_rays)
        
        # Stack coordinates: [rx, ry, rz, end_x, ry, end_z]
        lines_tensor = torch.stack([rx_exp, ry_exp, rz_exp, end_x, ry_exp, end_z], dim=-1)
        
        # Mask out rays that didn't hit anything (distance >= 9.9)
        valid_mask = lidar_depths < 10000
        
        # Bulk transfer to CPU (this prevents the rendering loop from blocking simulation speed)
        lines_np = lines_tensor.cpu().numpy()
        valid_mask_np = valid_mask.cpu().numpy()
        
        green = [0.0, 1.0, 0.0]
        
        for i, env in enumerate(envs):
            # Flatten the valid lines for this environment using efficient numpy masking
            env_lines = lines_np[i][valid_mask_np[i]].flatten().tolist()
            num_lines = len(env_lines) // 6
            if num_lines > 0:
                self.gym.add_lines(self.viewer, env, num_lines, env_lines, green * num_lines)

    def draw_bumper_alerts(self, envs, robot_states: torch.Tensor, bumpers: torch.Tensor):
        """Draws a floating red cross above the robot if a collision is detected."""
        if self.viewer is None: return
        
        # Find which envs triggered the bumper
        bumped_indices = (bumpers.squeeze(-1) > 0.5).nonzero(as_tuple=True)[0]
        if len(bumped_indices) == 0:
            return
            
        rx = robot_states[bumped_indices, 0]
        ry = robot_states[bumped_indices, 1] + 0.8  # Float above
        rz = robot_states[bumped_indices, 2]
        
        s = 0.2 # Cross size
        
        # Build 3 intersecting lines for the cross
        l1 = torch.stack([rx-s, ry, rz, rx+s, ry, rz], dim=-1)
        l2 = torch.stack([rx, ry-s, rz, rx, ry+s, rz], dim=-1)
        l3 = torch.stack([rx, ry, rz-s, rx, ry, rz+s], dim=-1)
        
        lines_tensor = torch.cat([l1, l2, l3], dim=-1).cpu().numpy()
        red = [1.0, 0.0, 0.0] * 3
        
        for idx, env_idx in enumerate(bumped_indices.cpu().numpy()):
            env = envs[env_idx]
            self.gym.add_lines(self.viewer, env, 3, lines_tensor[idx].tolist(), red)

    def show_camera(self, camera_tensor: torch.Tensor, env_idx=0):
        if not self.cv2_imported: return
        
        img = camera_tensor[env_idx].cpu().numpy().astype('uint8')
        img_bgr = self.cv2.cvtColor(img, self.cv2.COLOR_RGB2BGR)
        
        self.cv2.imshow(f"Agent {env_idx} POV", img_bgr)
        self.cv2.waitKey(1)
