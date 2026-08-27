import math
import yaml

from gym import spaces
from isaacgym import gymapi, gymtorch

from core.room import BoxObstacle, Room
from core.visualizer import DebugVisualizer
import torch


class RoombaSimulator:
    def __init__(self, num_envs, config_path="configs/config.yaml", sim_device="cuda:0", show_viewer=True):
        # 1. load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.num_envs = num_envs
        self.show_viewer = show_viewer

        if "cuda" in sim_device and torch.cuda.is_available():
            self.device = sim_device
        else:
            self.device = "cpu"
        self.device_id = torch.device(self.device).index or 0

        # 2. Initialize physics simulator
        self.gym = gymapi.acquire_gym()
        self.sim = self._setup_simulator()

        # 3. Load Assets & Room Definition
        self.robot_asset = self._load_robot_asset()
        self.dofs_per_actor = self.gym.get_asset_dof_count(self.robot_asset)
        self.num_dofs = self.num_envs * self.dofs_per_actor

        self.room = self._load_room_layout()

        # Initialize the Occupancy Map and calculate safe spacing
        from core.room import OccupancyMap
        self.occupancy_map = OccupancyMap(self.room)
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

        self.visualizer = DebugVisualizer(self.gym, self.viewer)

        # 6. Lock scene geometry, allocate contiguous GPU memory, expose flat tensors
        self.gym.prepare_sim(self.sim)
        self._init_hardware_tensors()

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
                raise RuntimeError("Failed to load 'roomba.urdf'. Check file path and URDF syntax.")
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
                # Rotate -90 degrees around Y to face forward
                cam_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 1, 0), -math.pi / 2.0)

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

                # Rotate camera -90 deg around Y-axis so its optical axis (-Z) points in the robot's forward direction (+X)
                lidar_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 1, 0), - math.pi / 2.0)              

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
    
            cam_pos = gymapi.Vec3(viewer_center_offset + 5.0, 5.0, viewer_center_offset + 5.0)
            cam_target = gymapi.Vec3(viewer_center_offset, 0.0, viewer_center_offset)
    
            if self.viewer is not None:
                self.gym.viewer_camera_look_at(
                    self.viewer, None, 
                    cam_pos, 
                    cam_target
                )

    def _init_hardware_tensors(self):
        """Acquires raw GPU pointers and wraps them in PyTorch."""
        self.dof_velocity_targets = torch.zeros(
            self.num_dofs, dtype=torch.float32, device=self.device
        )
        self.dof_velocity_targets_view = self.dof_velocity_targets.view(self.num_envs, self.dofs_per_actor)

        # Contact Force Tensor (For Bumpers)
        _contact_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)
        self.contact_forces_view = gymtorch.wrap_tensor(_contact_tensor).view(self.num_envs, -1, 3)

        # Acquire root state tensor (for (x,z) position resetting)
        _root_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.root_states = gymtorch.wrap_tensor(_root_tensor)

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

    def apply_wheel_velocities(self, v_left, v_right):
        """Pushes requested speeds to the physics engine."""
        self.dof_velocity_targets_view[:, 0] = v_left
        self.dof_velocity_targets_view[:, 1] = v_right
        self.gym.set_dof_velocity_target_tensor(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_velocity_targets)
        )

    def step_physics(self):
        """Advances the simulation by one tick."""
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        self.gym.step_graphics(self.sim)

    def close(self):
        if self.show_viewer and self.viewer is not None:
            self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)

    def set_actor_root_states(self, root_states: torch.Tensor, actor_ids: torch.Tensor):
        """Sets root states for the specified actor indices."""
        actor_ids_int32 = actor_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(root_states),
            gymtorch.unwrap_tensor(actor_ids_int32),
            len(actor_ids_int32)
        )
