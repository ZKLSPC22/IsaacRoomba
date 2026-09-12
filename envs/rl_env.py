import math
import yaml

from gym import spaces
from isaacgym import gymapi, gymtorch
import torch

from core.simulator import RoombaSimulator


class RoombaRLEnv:
    def __init__(self, config_path="configs/config.yaml", sim_device="cuda:0", show_viewer=True):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.num_envs = self.config['env']['num_rl_envs']

        # 1. Instantiate the physics core
        self.sim = RoombaSimulator(self.num_envs, config_path, sim_device, show_viewer)

        # Pull necessary configs and hardware mapping from the simulator
        self.config = self.sim.config
        self.device = self.sim.device

        # 2. Setup RL-specific variables
        self.all_env_ids = torch.arange(self.num_envs, dtype=torch.int32, device=self.device)
        self.goals = torch.zeros((self.num_envs, 2), dtype=torch.float32, device=self.device)

        # High-water mark reward
        self.progress_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.min_dist_to_goal = torch.full((self.num_envs,), float('inf'), dtype=torch.float32, device=self.device)

        self._setup_observation_space()

    def get_prior_knowledge(self):
       """Exposes the static map and goal coordinates to the agent once at startup."""
       return {
           "occupancy_map": self.sim.occupancy_map,
           "goals": self.goals.clone()
       }

    def _setup_observation_space(self):
        """Builds the observation space based on config toggles."""
        # Normalised action space: [linear_velocity, angular_velocity]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=float)

        obs_dict = {}
        sens_cfg = self.config['sensors']

        if sens_cfg['enable_bumper']:
            obs_dict["bumper"] = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=float)
            
        if sens_cfg['enable_lidar']:
            obs_dict["lidar"] = spaces.Box(low=0.0, high=10.0, shape=(sens_cfg['lidar_rays'],), dtype=float)
            
        if sens_cfg['enable_camera']:
            res = sens_cfg['camera_res']
            obs_dict["camera"] = spaces.Box(low=0, high=255, shape=(res, res, 3), dtype=int)

        self.observation_space = spaces.Dict(obs_dict)

    def _compute_rewards(self, obs, actions, dist_to_goal):
        reached = dist_to_goal < 0.5
        
        bumped = torch.zeros(self.num_envs, device=self.device)
        if "bumper" in obs:
            bumped = obs["bumper"].squeeze(-1) > 0.5
            
        # High-water mark progress calculation
        progress = torch.clamp(self.min_dist_to_goal - dist_to_goal, min=0.0)
        self.min_dist_to_goal = torch.minimum(self.min_dist_to_goal, dist_to_goal)
        
        rewards = (progress * 5.0) + (reached.float() * 10.0) - (bumped.float() * 5.0)
        return rewards

    def _compute_dones(self, dist_to_goal):
        reached = dist_to_goal < 0.5
        timeout = self.progress_buf >= 3600  # Max steps should be set in config
        return reached | timeout

    def _compute_observations(self):
        """Builds dictionary observations dynamically using pure PyTorch tensors."""
        # Refresh the root states so positions update
        self.sim.gym.refresh_actor_root_state_tensor(self.sim.sim)

        obs = {}
        sens_cfg = self.config['sensors']

        # Sensors                
        if sens_cfg['enable_bumper']:
            self.sim.gym.refresh_net_contact_force_tensor(self.sim.sim)
            forces = self.sim.contact_forces_view[:, self.sim.chassis_body_idx, :]
            obs["bumper"] = (torch.norm(forces, dim=-1) > 0.1).float().unsqueeze(-1)

        # Visual sensors
        if sens_cfg['enable_lidar'] or sens_cfg['enable_camera']:
            self.sim.sync_graphics()
            self.sim.gym.render_all_camera_sensors(self.sim.sim)
            self.sim.gym.start_access_image_tensors(self.sim.sim)

            if sens_cfg['enable_lidar']:
                raw_depth = -torch.stack(self.sim.lidar_tensors)

                # translate projective depth to Euclidean radial distance
                fov_rad = 120.0 * (math.pi / 180.0)
                num_rays = sens_cfg['lidar_rays']
                # ray_angles = torch.linspace(-fov_rad/2, fov_rad/2, num_rays, device=self.device)

                # ==================================================
                # Projection math
                # ==================================================
                # Create an index tensor for the rays
                indices = torch.arange(num_rays, dtype=torch.float32, device=self.device)

                # Map indices to normalized device coordinates [-1.0 to 1.0]
                ndc = (2.0 * indices - (num_rays - 1)) / (num_rays - 1)

                # Apply the pinhole projection formula
                ray_angles = torch.atan(ndc * math.tan(fov_rad / 2.0))
                # ==================================================
                # Projection math
                # ==================================================

                radial_depth = raw_depth / torch.cos(ray_angles)

                obs["lidar"] = torch.flip(torch.clamp(radial_depth, 0.0, 100.0).squeeze(1), dims=[1])
                
            if sens_cfg['enable_camera']:
                raw_rgb = torch.stack(self.sim.camera_tensors)
                obs["camera"] = raw_rgb[..., :3]
                # obs["camera"] = torch.flip(raw_rgb[..., :3], dims=[2])

            self.sim.gym.end_access_image_tensors(self.sim.sim)

        return obs

    def render_debug_visuals(self, obs):
        """Passes the latest observations to the visualizer."""
        if not self.sim.show_viewer or self.sim.viewer is None:
            return
            
        self.sim.visualizer.clear()
        self.sim.visualizer.draw_goals(self.sim.envs, self.goals)

        robot_states = self.sim.root_states[self.sim.robot_actor_indices] # Filters out the actor indeces
        if "lidar" in obs:
            self.sim.visualizer.draw_lidar(self.sim.envs, robot_states, obs["lidar"]) # API mismatch with draw_lidar
            
        if "bumper" in obs:
            self.sim.visualizer.draw_bumper_alerts(self.sim.envs, robot_states, obs["bumper"])
            
        if "camera" in obs:
            self.sim.visualizer.show_camera(obs["camera"], env_idx=0)

    def reset(self, env_ids=None):
        """Teleports robots back to valid safe points and zeroes their velocities."""
        if env_ids is None:
            env_ids = self.all_env_ids
        if len(env_ids) == 0:
            return self._compute_observations()

        self.progress_buf[env_ids] = 0
        actor_ids = self.sim.robot_actor_indices[env_ids]

        starts_x, starts_z = [], []
        goals_x, goals_z = [], []

        # Sampling is written entirely for CPU, making this loop CPU only
        for _ in range(len(env_ids)):
            start_x, start_z = self.sim.occupancy_map.sample_valid_pose()
            goal_x, goal_z = self.sim.occupancy_map.sample_valid_pose()
            starts_x.append(start_x)
            starts_z.append(start_z)
            goals_x.append(goal_x)
            goals_z.append(goal_z)

        # Convert collected lists to GPU tensors in one bulk operation
        starts_x = torch.tensor(starts_x, dtype=torch.float32, device=self.device)
        starts_z = torch.tensor(starts_z, dtype=torch.float32, device=self.device)
        goals_x = torch.tensor(goals_x, dtype=torch.float32, device=self.device)
        goals_z = torch.tensor(goals_z, dtype=torch.float32, device=self.device)

        # Start poses are already environment-local.
        self.sim.root_states[actor_ids, 0] = starts_x
        self.sim.root_states[actor_ids, 1] = 0.065
        self.sim.root_states[actor_ids, 2] = starts_z
                
        self.goals[env_ids, 0] = goals_x
        self.goals[env_ids, 1] = goals_z
        
        dx = goals_x - starts_x
        dz = goals_z - starts_z
        self.min_dist_to_goal[env_ids] = torch.sqrt(dx**2 + dz**2)

        # Randomize Yaw for the reset subset
        yaw = torch.rand(len(env_ids), device=self.device) * 2 * math.pi
        self.sim.root_states[actor_ids, 3] = 0.0                  # qx
        self.sim.root_states[actor_ids, 4] = torch.sin(yaw / 2)   # qy
        self.sim.root_states[actor_ids, 5] = 0.0                  # qz
        self.sim.root_states[actor_ids, 6] = torch.cos(yaw / 2)   # qw

        # Zero out velocities
        self.sim.root_states[actor_ids, 7:13] = 0.0

        self.sim.set_actor_root_states(self.sim.root_states, actor_ids)

        # Reset wheel DOF state so the transition is Markov in the explicit state
        self.sim.reset_dof_states(env_ids)

        self.sim.sync_graphics()

        obs = self._compute_observations()
        return obs

    def step(self, actions):
        """Converts policy [v, omega] into wheel speeds and steps physics."""
        self.progress_buf += 1

        clamped_action = torch.clamp(actions, -1.0, 1.0)
        max_lin = self.config['robot']['max_linear_velocity']
        max_ang = self.config['robot']['max_angular_velocity']
        
        v = clamped_action[:, 0] * max_lin
        omega = clamped_action[:, 1] * max_ang
        
        L, R = 0.235, 0.036  # Track width, Wheel radius
        v_left = - (v - (omega * L) / 2.0) / R
        v_right = - (v + (omega * L) / 2.0) / R
        
        self.sim.apply_wheel_velocities(v_left, v_right)
        self.sim.step_physics()
        self.sim.sync_graphics()

        # Gather new state details
        obs = self._compute_observations()

        robot_x = self.sim.root_states[self.sim.robot_actor_indices, 0]
        robot_z = self.sim.root_states[self.sim.robot_actor_indices, 2]
        dist_to_goal = torch.sqrt((self.goals[:, 0] - robot_x)**2 + (self.goals[:, 1] - robot_z)**2)

        rewards = self._compute_rewards(obs, clamped_action, dist_to_goal)
        dones = self._compute_dones(dist_to_goal)

        info = {
            "success": (dist_to_goal < 0.5).clone(),
            "progress": self.progress_buf.clone()
        }

        # Reset environments that died or finished (also updates goal markers)
        done_env_ids = dones.nonzero(as_tuple=False).squeeze(-1)
        if len(done_env_ids) > 0:
            reset_obs = self.reset(done_env_ids)
            for key in obs:
                obs[key][done_env_ids] = reset_obs[key][done_env_ids]

        self.render_debug_visuals(obs)

        # Render graphics if enabled
        if self.sim.show_viewer and self.sim.viewer is not None:
            self.sim.gym.draw_viewer(self.sim.viewer, self.sim.sim, True)
            self.sim.gym.sync_frame_time(self.sim.sim)

        return obs, rewards, dones, info

    def close(self):
        self.sim.close()
