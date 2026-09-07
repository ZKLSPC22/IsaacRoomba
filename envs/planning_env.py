import math
import yaml

from gym import spaces
from isaacgym import gymapi, gymtorch
import torch

from core.simulator import RoombaSimulator


class RoombaPlanningEnv:
    # Batched Generative Environment, yielding (s', o, r) given (s, a).
    def __init__(self, config_path="configs/config.yaml", sim_device="cuda:0", show_viewer=True):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.num_envs = self.config['env']['num_planning_envs']
        self.macro_action_ticks = self.config.get('planning', {}).get('macro_action_ticks', 30)

        # 1. Instantiate the physics core
        self.sim = RoombaSimulator(self.num_envs, config_path, sim_device, show_viewer)

        # Pull necessary configs and hardware mapping from the simulator
        self.config = self.sim.config
        self.device = self.sim.device

        # 2. Setup indices and internal variables
        self.all_env_ids = torch.arange(self.num_envs, dtype=torch.int32, device=self.device)
        self.goals = torch.zeros((self.num_envs, 2), dtype=torch.float32, device=self.device)

        self._setup_spaces()

    def get_prior_knowledge(self):
        """Exposes the static map and goal coordinates to the agent once at startup."""
        return {
            "occupancy_map": self.sim.occupancy_map,
            "goals": self.goals.clone()
        }

    def _setup_spaces(self):
        """Builds the explicit State, Observation, and Action spaces."""

        # Normalised action space: [linear_velocity, angular_velocity]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=float)

        # Explicit State Space: 13D Isaac Gym root state + 2D Goal (X, Z)
        # Root state: [x, y, z, qx, qy, qz, qw, vx, vy, vz, wx, wy, wz]
        self.state_space = spaces.Box(low=-float('inf'), high=float('inf'), shape=(15,), dtype=float)

        # Observation space (Dictionary)
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

    def get_states(self):
        """Packs the internal simulator root states and goals into an explicit state tensor."""
        root_states = self.sim.root_states[self.sim.robot_actor_indices].clone()
        # Convert simulation-frame positions back to room-local coordinates.
        root_states[:, 0] -= self.sim.env_origins[:, 0]
        root_states[:, 1] -= self.sim.env_origins[:, 1]
        root_states[:, 2] -= self.sim.env_origins[:, 2]
        return torch.cat([root_states, self.goals], dim=-1)

    def set_states(self, states: torch.Tensor):
        """Unpacks explicit state tensors and teleports the robots in the simulator."""
        actor_ids = self.sim.robot_actor_indices
        
        # Extract root states and goals
        root_states = states[:, :13].clone()
        self.goals = states[:, 13:15]
        
        # Convert room-local positions to simulation-frame coordinates by adding
        # each environment's origin offset.
        root_states[:, 0] += self.sim.env_origins[:, 0]
        root_states[:, 1] += self.sim.env_origins[:, 1]
        root_states[:, 2] += self.sim.env_origins[:, 2]
        
        # Write root states to simulator tensor
        self.sim.root_states[actor_ids] = root_states
        
        # Force the physics engine to teleport the actors
        self.sim.set_actor_root_states(self.sim.root_states, actor_ids)

        # Reset wheel DOF state so G(s, a) is Markov in the 15D state
        self.sim.reset_dof_states()
        
        # Step graphics so visual sensors (camera/lidar) update to the teleported positions
        self.sim.sync_graphics()

    def _compute_bumped(self):
        """Refresh contact forces and return a per-env bump boolean mask.

        Decoupled from the observation dict so rewards can be computed even when
        observations are skipped.
        """
        self.sim.gym.refresh_net_contact_force_tensor(self.sim.sim)
        forces = self.sim.contact_forces_view[:, self.sim.chassis_body_idx, :]
        return torch.norm(forces, dim=-1) > 0.1

    def _compute_rewards(self, dist_to_goal, bumped):
        # +10 for goal, -5 for bumping, -0.1 per step to encourage speed
        reached = dist_to_goal < 0.5
        rewards = (reached.float() * 10.0) - (bumped.float() * 5.0) - 0.1
        return rewards

    def compute_heuristic_values(self, states: torch.Tensor, is_terminal: torch.Tensor, gamma: float) -> torch.Tensor:
        """
        Calculates the expected discounted return of an optimal, straight-line 
        trajectory to the goal, preventing stalling pathologies.
        """
        dx = states[:, 13] - states[:, 0]
        dz = states[:, 14] - states[:, 2]
        distance = torch.sqrt(dx * dx + dz * dz)

        # 1. Estimate steps to goal (H)
        max_speed = self.config['robot']['max_linear_velocity']
        step_duration = self.macro_action_ticks / 60.0
        max_dist_per_step = max_speed * step_duration

        # Distance remaining outside the 0.5m goal radius
        d_remain = torch.clamp(distance - 0.5, min=0.0)
        H = d_remain / max_dist_per_step

        # A non-terminal state needs at least one more step to reach the goal,
        # so floor H at 1 to keep the exponent (H - 1) non-negative.
        H = torch.clamp(H, min=1.0)

        # 2. Compute discounted return
        goal_reward = 10.0
        step_cost = -0.1

        # V(s) = gamma^(H-1) * goal_reward + step_cost * (1 - gamma^H) / (1 - gamma)
        # Guard gamma == 1, where the geometric series has the limit H.
        if abs(1.0 - gamma) < 1e-9:
            values = goal_reward + H * step_cost
        else:
            values = (gamma ** (H - 1)) * goal_reward + step_cost * ((1.0 - gamma ** H) / (1.0 - gamma))

        return torch.where(is_terminal, torch.zeros_like(values), values)

    def _compute_observations(self, bumped=None):
        """Builds dictionary observations dynamically using pure PyTorch tensors.

        `bumped` may be precomputed by the caller to avoid a redundant contact
        refresh when rewards were already computed for this step.
        """
        obs = {}
        sens_cfg = self.config['sensors']

        # Sensors                
        if sens_cfg['enable_bumper']:
            if bumped is None:
                bumped = self._compute_bumped()
            obs["bumper"] = bumped.float().unsqueeze(-1)

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

    def generate(self, states: torch.Tensor, actions: torch.Tensor, compute_observation: bool = True):
        """
        The Core Generative Model Interface G(s, a).
        Teleports to 'states', applies 'actions', and yields the transition.

        `compute_observation` controls whether the observation dict is built.
        Fully-observable planners (MCTS) only need (state, reward, done) and pass
        False to skip unused observation work; POMCGS passes True (the default).
        """
        # 1. Teleport the batch
        self.set_states(states)
        
        # 2. Translate continuous policy actions to wheel velocities
        clamped_action = torch.clamp(actions, -1.0, 1.0)
        max_lin = self.config['robot']['max_linear_velocity']
        max_ang = self.config['robot']['max_angular_velocity']
        
        v = clamped_action[:, 0] * max_lin
        omega = clamped_action[:, 1] * max_ang
        
        L, R = 0.235, 0.036 
        v_left = - (v - (omega * L) / 2.0) / R
        v_right = - (v + (omega * L) / 2.0) / R
        
        self.sim.apply_wheel_velocities(v_left, v_right)
        
        # 3. Advance physics for one macro-action (default 30 ticks = 0.5s at 60Hz).
        bumped = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for tick in range(self.macro_action_ticks):
            self.sim.step_physics()
            # Sample bumper contacts every 5 ticks and on the final tick
            if self.config['sensors']['enable_bumper'] and (tick % 5 == 0 or tick == self.macro_action_ticks - 1):
                bumped |= self._compute_bumped()

        # Refresh the actor root-state tensor so the reads below reflect the
        # post-physics state, not the teleported input state.
        self.sim.gym.refresh_actor_root_state_tensor(self.sim.sim)

        # Determine terminal nodes for tree search using room-local coordinates.
        # Subtract each environment's origin so positions match the room-local
        # goals stored in self.goals.
        robot_x = self.sim.root_states[self.sim.robot_actor_indices, 0] - self.sim.env_origins[:, 0]
        robot_z = self.sim.root_states[self.sim.robot_actor_indices, 2] - self.sim.env_origins[:, 2]
        dist_to_goal = torch.sqrt((self.goals[:, 0] - robot_x)**2 + (self.goals[:, 1] - robot_z)**2)
        dones = (dist_to_goal < 0.5)
        
        # 4. Gather next state, reward, and (optionally) observation.
        next_states = self.get_states()

        rewards = self._compute_rewards(dist_to_goal, bumped)

        if compute_observation:
            obs = self._compute_observations(bumped)
        else:
            obs = {}
        
        # Optional: Graphics sync
        self.render_debug_visuals(obs)
        if self.sim.show_viewer and self.sim.viewer is not None:
            self.sim.gym.draw_viewer(self.sim.viewer, self.sim.sim, True)
            self.sim.gym.sync_frame_time(self.sim.sim)
            
        return next_states, obs, rewards, dones

    def render_debug_visuals(self, obs):
        """Passes the latest observations to the visualizer."""
        if not self.sim.show_viewer or self.sim.viewer is None:
            return
            
        self.sim.visualizer.clear()
        self.sim.visualizer.draw_goals(self.sim.envs, self.goals)
        robot_states = self.sim.root_states[self.sim.robot_actor_indices]
        
        if "lidar" in obs:
            self.sim.visualizer.draw_lidar(self.sim.envs, robot_states, obs["lidar"])
        if "bumper" in obs:
            self.sim.visualizer.draw_bumper_alerts(self.sim.envs, robot_states, obs["bumper"])
        if "camera" in obs:
            self.sim.visualizer.show_camera(obs["camera"], env_idx=0)

    def close(self):
        self.sim.close()
