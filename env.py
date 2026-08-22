import math

import yaml
from gym import spaces
from isaacgym import gymapi, gymtorch

from room import BoxObstacle, Room
import torch


class RoombaEnv:
    def __init__(self, config_path="config.yaml", sim_device="cuda:0", show_viewer=True):
        # 1. load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.num_envs = self.config['env']['num_envs']
        self.show_viewer = show_viewer

        if "cuda" in sim_device and torch.cuda.is_available():
            self.device = sim_device
        else:
            self.device = "cpu"
        self.device_id = torch.device(self.device).index or 0

        self._parse_sensors()

        # 2. Initialize physics simulator
        self.gym = gymapi.acquire_gym()
        self.sim = self._setup_simulator()

        # 3. Load Assets & Room Definition
        self.robot_asset = self._load_robot_asset()
        self.dofs_per_actor = self.gym.get_asset_dof_count(self.robot_asset)
        self.num_dofs = self.num_envs * self.dofs_per_actor

        self.room = self._load_room_layout()

        # Initialize the Occupancy Map and calculate safe spacing
        from room import OccupencyMap
        self.occupancy_map = OccupencyMap(self.room)
        self.config['env']['env_spacing'] = self._calculate_dynamic_spacing()

        # Add ground plane
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 1, 0)  # Y-up coordinate system
        self.gym.add_ground(self.sim, plane_params)

        # 4. Build parallel environments
        self.envs = []
        self.robot_handles = []
        self._create_envs()

        # 5. Set up visualizer
        self.viewer = None
        if self.show_viewer:
            self._setup_viewer()

        # 6. Lock scene geometry, allocate contiguous GPU memory, expose flat tensors
        self.gym.prepare_sim(self.sim)
        self._init_tensors()

    def _parse_sensors(self):
        """Builds the observation space based on config toggles."""
        obs_dict = {}
        sens_cfg = self.config['sensors']

        obs_dict["goal"] = spaces.Box(low=-math.pi, high=100.0, shape=(2,), dtype=float)

        if sens_cfg['enable_bumper']:
            obs_dict["bumper"] = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=float)
            
        if sens_cfg['enable_lidar']:
            obs_dict["lidar"] = spaces.Box(low=0.0, high=10.0, shape=(sens_cfg['lidar_rays'],), dtype=float)
            
        if sens_cfg['enable_camera']:
            res = sens_cfg['camera_res']
            obs_dict["camera"] = spaces.Box(low=0, high=255, shape=(res, res, 3), dtype=int)

        self.observation_space = spaces.Dict(obs_dict)
        
        # Action space: [linear_velocity, angular_velocity]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=float)

    def _setup_simulator(self):
        """Initializes the PhysX physics engine with GPU pipeline enabled."""
        sim_params = gymapi.SimParams()
        sim_params.physx.use_gpu = True
        sim_params.use_gpu_pipeline = True
        sim_params.up_axis = gymapi.UP_AXIS_Y
        sim_params.gravity = gymapi.Vec3(0.0, -9.81, 0.0)

        # API: (GPU for Physics, GPU for Rendering, Physics Engine, Params)
        sim = self.gym.create_sim(self.device_id, self.device_id, gymapi.SIM_PHYSX, sim_params)
        if sim is None:
            raise RuntimeError("Failed to create Isaac Gym simulation environment.")
        return sim    

    def _load_robot_asset(self):
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = False

        asset = self.gym.load_asset(
            self.sim, ".", self.config['robot']['urdf_path'], asset_options
        )
        if asset is None:
            raise RuntimeError("Failed to load 'cartpole.urdf'. Check file path and URDF syntax.")
        return asset

    def _load_room_layout(self) -> Room:
        """Load room layout using presets or falls back to custom obstacle list"""
        room_cfg = self.config['room']
        room_type = room_cfg.get('type', 'custom') # Default to custom
        width = room_cfg.get('width', 20.0)
        depth = room_cfg.get('depth', 20.0)

        # 1. Match Preset Class Methods
        if room_type == "empty":
            return Room.empty()

        elif room_type == "standard":
            return Room.standard()

        # 2. Custom Layout: Parse raw obstacles from YAML
        elif room_type == "custom":
            obstacles = []
            for obs_cfg in room_cfg.get('obstacles', []):
                if obs_cfg['type'] == "box":
                    obstacles.append(BoxObstacle(
                        x=obs_cfg['x'], 
                        z=obs_cfg['z'], 
                        width=obs_cfg['width'], 
                        depth=obs_cfg['depth']
                    ))
            return Room(width=width, depth=depth, obstacles=obstacles)

        else:
            raise ValueError(f"Unknown room layout type: '{room_type}'")

    def _calculate_dynamic_spacing(self):
        room_width = self.config['room'].get('width', 20.0)
        room_depth = self.config['room'].get('depth', 20.0)
        calculated_spacing = max(room_width, room_depth) + 2.0
        if calculated_spacing > 50.0:
            raise ValueError(f"Spacing {calculated_spacing}m exceeds safety limit.")
        return calculated_spacing

    def _create_envs(self):
        self.envs_per_row = int(math.sqrt(self.num_envs))
        spacing = self.config['env']['env_spacing']
        
        lower = gymapi.Vec3(-spacing, 0.0, -spacing)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        robot_pose = gymapi.Transform()
        robot_pose.p = gymapi.Vec3(0.0, 0.5, 0.0)  # Drop slightly from above

        # Initialize handle storage lists
        self.camera_handles = []
        self.lidar_handles = []
        # To reset the robots properly, the simulation track global actor indices, not just local environment handles
        self.robot_actor_indices = []
        self.progress_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device) # Add this

        for i in range(self.num_envs):
            env = self.gym.create_env(self.sim, lower, upper, self.envs_per_row)
            
            # Build physical Room walls and obstacles
            self.room.build_in_isaac(self.gym, self.sim, env)

            # Spawn Robot
            handle = self.gym.create_actor(env, self.robot_asset, robot_pose, f"roomba_{i}", i, 1)

            # Get global actor index for root state resets
            global_idx = self.gym.get_actor_index(env, handle, gymapi.DOMAIN_SIM)
            self.robot_actor_indices.append(global_idx)

            # Configure drive wheels to Velocity Control
            dof_props = self.gym.get_actor_dof_properties(env, handle)
            for j in range(self.dofs_per_actor):
                dof_props["driveMode"][j] = gymapi.DOF_MODE_VEL
                dof_props["stiffness"][j] = 0.0
                dof_props["damping"][j] = 1000.0  # High damping = strict speed matching
            self.gym.set_actor_dof_properties(env, handle, dof_props)

            # Get the chassis rigid body index
            sensor_link_idx = self.gym.find_actor_rigid_body_handle(env, handle, "lidar_link")

            # Attach RGB Camera (If enabled)
            if self.config['sensors']['enable_camera']:
                cam_props = gymapi.CameraProperties()
                cam_props.width = self.config['sensors']['camera_res']
                cam_props.height = self.config['sensors']['camera_res']
                cam_props.enable_tensors = True

                cam_handle = self.gym.create_camera_sensor(env, cam_props)

                cam_pose = gymapi.Transform()
                cam_pose.p = gymapi.Vec3(0.0, 0.0, 0.0)

                self.gym.attach_camera_to_body(
                    cam_handle, env, sensor_link_idx, cam_pose, gymapi.FOLLOW_TRANSFORM
                )
                self.camera_handles.append(cam_handle)

            # Attach LiDAR via 1D Depth Camera (If enabled)
            if self.config['sensors']['enable_lidar']:
                lidar_props = gymapi.CameraProperties()
                lidar_props.width = self.config['sensors']['lidar_rays']
                lidar_props.height = 1 # 1D pixel slice
                lidar_props.horizontal_fov = 120.0 # Standard forward scanner FOV
                lidar_props.enable_tensors = True
                
                lidar_handle = self.gym.create_camera_sensor(env, lidar_props)
                
                lidar_pose = gymapi.Transform()
                lidar_pose.p = gymapi.Vec3(0.0, 0.0, 0.0) # Pushed to the front edge (radius = 18cm)
                
                self.gym.attach_camera_to_body(
                    lidar_handle, env, sensor_link_idx, lidar_pose, gymapi.FOLLOW_TRANSFORM
                )
                self.lidar_handles.append(lidar_handle)

            self.envs.append(env)
            self.robot_handles.append(handle)

        self.robot_actor_indices = torch.tensor(
            self.robot_actor_indices, dtype=torch.int32, device=self.device
        )

    def _setup_viewer(self):
        self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
        spacing = self.config['env']['env_spacing']

        viewer_center_offset = (self.envs_per_row - 1) * spacing

        cam_pos = gymapi.Vec3(viewer_center_offset + 15.0, 30.0, viewer_center_offset + 15.0)
        cam_target = gymapi.Vec3(viewer_center_offset, 0.0, viewer_center_offset)

        if self.viewer is not None:
            self.gym.viewer_camera_look_at(
                self.viewer, None, 
                cam_pos, 
                cam_target
            )

    def _init_tensors(self):
        """Acquires raw GPU pointers and wraps them in PyTorch."""
        self.dof_velocity_targets = torch.zeros(
            self.num_dofs, dtype=torch.float32, device=self.device
        )
        self.dof_velocity_targets_view = self.dof_velocity_targets.view(self.num_envs, self.dofs_per_actor)

        # Contact Force Tensor (For Bumpers)
        _contact_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)
        self.contact_forces = gymtorch.wrap_tensor(_contact_tensor)
        self.contact_forces_view = self.contact_forces.view(self.num_envs, -1, 3)

        # Helper indices
        self.all_env_ids = torch.arange(self.num_envs, dtype=torch.int32, device=self.device)

        # Acquire root state tensor (for (x,z) position resetting)
        _root_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.root_states = gymtorch.wrap_tensor(_root_tensor)

        # Goal tensor [num_envs, 2] for (X, Z) target coordinates
        self.goals = torch.zeros((self.num_envs, 2), dtype=torch.float32, device=self.device)

        # Cache the chassis rigid body index once so we don't query it every frame
        if self.config['sensors']['enable_bumper']:
            self.chassis_body_idx = self.gym.find_actor_rigid_body_index(
                self.envs[0], self.robot_handles[0], "base_link", gymapi.DOMAIN_ACTOR
            )

        # Pre-wrap camera/LiDAR tensors (Zero-copy GPU pipeline)
        if self.config['sensors']['enable_lidar']:
            self.lidar_tensors = [
                gymtorch.wrap_tensor(
                    self.gym.get_camera_image_gpu_tensor(self.sim, e, h, gymapi.IMAGE_DEPTH)
                ) for e, h in zip(self.envs, self.lidar_handles)
            ]
            
        if self.config['sensors']['enable_camera']:
            self.camera_tensors = [
                gymtorch.wrap_tensor(
                    self.gym.get_camera_image_gpu_tensor(self.sim, e, h, gymapi.IMAGE_COLOR)
                ) for e, h in zip(self.envs, self.camera_handles)
            ]

        # High-water mark tracker for distance-based rewards
        self.min_dist_to_goal = torch.full(
            (self.num_envs,), float('inf'), dtype=torch.float32, device=self.device
        )
    
    def _compute_observations(self):
        """Builds dictionary observations dynamically using pure PyTorch tensors."""
        # Force sync with the latest physics calculations
        self.gym.refresh_net_contact_force_tensor(self.sim)
        
        obs = {}
        sens_cfg = self.config['sensors']

        # Goal relative calculation
        robot_x = self.root_states[self.robot_actor_indices, 0]
        robot_z = self.root_states[self.robot_actor_indices, 2]
        
        dx = self.goals[:, 0] - robot_x
        dz = self.goals[:, 1] - robot_z
        dist = torch.sqrt(dx**2 + dz**2)
        
        # Quaternion Y-UP Coordinate math: Yaw is around the Y axis
        qw = self.root_states[self.robot_actor_indices, 6]
        qy = self.root_states[self.robot_actor_indices, 4]
        robot_yaw = torch.atan2(2.0 * qy * qw, 1.0 - 2.0 * (qy**2))

        global_target_angle = torch.atan2(dx, dz)
        relative_angle = global_target_angle - robot_yaw

        # Normalize angle to [-pi, pi]
        relative_angle = (relative_angle + math.pi) % (2 * math.pi) - math.pi

        obs["goal"] = torch.stack([dist, relative_angle], dim=1)

        # Sensors                
        if sens_cfg['enable_bumper']:
            forces = self.contact_forces_view[:, self.chassis_body_idx, :]
            # Pure PyTorch tensor operations directly on GPU memory
            obs["bumper"] = (torch.norm(forces, dim=-1) > 0.1).float().unsqueeze(-1)

        # Trigger rendering pipeline if any visual sensor is active
        if sens_cfg['enable_lidar'] or sens_cfg['enable_camera']:
            self.gym.render_all_camera_sensors(self.sim)
            self.gym.start_access_image_tensors(self.sim) # Synchronization, finish running and pauses C++ graphics renderer so Torch reads reliable memory pointer 

        if sens_cfg['enable_lidar']:
            # Stack pointers into a single batched tensor [N, 64]
            # Depth comes out negative, and 'inf' means no hit. 
            raw_depth = -torch.stack(self.lidar_tensors)
            
            # Clip to max range (10.0m) and squeeze the H dimension
            obs["lidar"] = torch.clamp(raw_depth, 0.0, 10.0).squeeze(1)
            
        if sens_cfg['enable_camera']:
            # Stack to [N, H, W, 4] and slice off the Alpha channel which handles opacity/transparency
            raw_rgb = torch.stack(self.camera_tensors)
            obs["camera"] = raw_rgb[..., :3]

        if sens_cfg['enable_lidar'] or sens_cfg['enable_camera']:
            self.gym.end_access_image_tensors(self.sim)

        return obs

    def reset(self, env_ids=None):
        """Teleports robots back to valid safe points and zeroes their velocities."""
        if env_ids is None:
            env_ids = self.all_env_ids
        if len(env_ids) == 0:
            return self._compute_observations()

        self.progress_buf[env_ids] = 0
        actor_ids = self.robot_actor_indices[env_ids]

        # Sample safe start and goal positions from CPU OccupancyMap
        for i in env_ids:
            start_x, start_z = self.occupancy_map.sample_valid_pose()
            goal_x, goal_z = self.occupancy_map.sample_valid_pose()
            
            self.root_states[actor_ids[i], 0] = start_x
            self.root_states[actor_ids[i], 1] = 0.5  # Drop height
            self.root_states[actor_ids[i], 2] = start_z
            
            self.goals[i, 0] = goal_x
            self.goals[i, 1] = goal_z

            # NEW: Initialize the high-water mark to the starting distance
            dx = goal_x - start_x
            dz = goal_z - start_z
            self.min_dist_to_goal[i] = math.sqrt(dx**2 + dz**2)

        # Randomize Yaw
        yaw = torch.rand(len(env_ids), device=self.device) * 2 * math.pi
        self.root_states[actor_ids, 3] = 0.0                  # qx
        self.root_states[actor_ids, 4] = torch.sin(yaw / 2)   # qy
        self.root_states[actor_ids, 5] = 0.0                  # qz
        self.root_states[actor_ids, 6] = torch.cos(yaw / 2)   # qw

        # Zero out velocities
        self.root_states[actor_ids, 7:13] = 0.0

        actor_ids_int32 = actor_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(actor_ids_int32),
            len(actor_ids_int32)
        )

        self.progress_buf += 1  # <-- Increment step counter for timeouts
        self.gym.step_graphics(self.sim)

        return self._compute_observations()

    def _compute_rewards(self, obs, actions):
        current_dist = obs["goal"][:, 0]
        reached = current_dist < 0.5
        
        bumped = torch.zeros(self.num_envs, device=self.device)
        if "bumper" in obs:
            bumped = obs["bumper"].squeeze(-1) > 0.5
            
        # 1. HIGH-WATER MARK PROGRESS CALCULATION
        # Subtract current distance from the historical minimum. 
        # clamp(min=0.0) ensures we ONLY get a positive number if it's a NEW record.
        progress = torch.clamp(self.min_dist_to_goal - current_dist, min=0.0)
        
        # 2. Update the historical minimum tracker for the next step
        self.min_dist_to_goal = torch.minimum(self.min_dist_to_goal, current_dist)
        
        # 3. Final Reward Construction
        # - progress * 5.0: Rewards moving into new, uncharted closer territory.
        # - reached * 10.0: Massive terminal bonus for touching the goal.
        # - bumped * 5.0: Heavy penalty for collisions.
        rewards = (progress * 5.0) + (reached.float() * 10.0) - (bumped.float() * 5.0)
        
        return rewards

    def _compute_dones(self, obs):
        dist_to_goal = obs["goal"][:, 0]
        reached = dist_to_goal < 0.5
        timeout = self.progress_buf >= 500  # Max steps
        return reached | timeout

    def step(self, actions):
        """Converts policy [v, omega] into wheel speeds and steps physics."""
        clamped_action = torch.clamp(actions, -1.0, 1.0)
        
        # For now, placeholder velocity scaling:
        max_lin = self.config['robot']['max_linear_velocity']
        max_ang = self.config['robot']['max_angular_velocity']
        
        v = clamped_action[:, 0] * max_lin
        omega = clamped_action[:, 1] * max_ang
        
        L, R = 0.235, 0.036  # Track width, Wheel radius
        v_left = (v - (omega * L) / 2.0) / R
        v_right = (v + (omega * L) / 2.0) / R
        
        self.dof_velocity_targets_view[:, 0] = v_left
        self.dof_velocity_targets_view[:, 1] = v_right

        # Push velocity requests to C++ controller
        self.gym.set_dof_velocity_target_tensor(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_velocity_targets)
        )

        # Step physics engine
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)

        self.gym.step_graphics(self.sim)

        # Gather new state details
        obs = self._compute_observations()
        rewards = self._compute_rewards(obs, clamped_action)
        dones = self._compute_dones(obs)
        info = {
            "success": (obs["goal"][:, 0] < 0.5).clone(),
            "progress": self.progress_buf.clone()
        }

        # Reset environments that died or finished
        done_env_ids = dones.nonzero(as_tuple=False).squeeze(-1)
        if len(done_env_ids) > 0:
            reset_obs = self.reset(done_env_ids)
            for key in obs:
                obs[key][done_env_ids] = reset_obs[key][done_env_ids]

        # Render graphics if enabled
        if self.show_viewer and self.viewer is not None:
            if self.gym.query_viewer_has_closed(self.viewer):
                return False
            self.gym.draw_viewer(self.viewer, self.sim, True)
            self.gym.sync_frame_time(self.sim)

        return obs, rewards, dones, info

    def close(self):
        """Cleans up C++ memory resources."""
        if self.show_viewer and self.viewer is not None:
            self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)