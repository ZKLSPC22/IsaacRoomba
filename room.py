from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

import numpy as np
import yaml
from isaacgym import gymapi
from scipy.ndimage import binary_dilation


@dataclass
class Obstacle(ABC):
    x: float
    z: float

    @abstractmethod
    def spawn(self, gym, sim, env_ptr, height: float = 2.0):
        pass


@dataclass
class BoxObstacle(Obstacle):
    width: float
    depth: float

    def spawn(self, gym, sim, env_ptr, height: float = 2.0):
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True

        asset = gym.create_box(sim, self.width, height, self.depth, asset_options)

        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(self.x, height / 2.0, self.z)
        
        actor_handle = gym.create_actor(
            env_ptr, asset, pose, "box_obstacle", group=-1, filter=0
        )

        return actor_handle


@dataclass
class Room:
    # Room specs
    width: float
    depth: float

    obstacles: List[Obstacle]

    def _build_walls(self, gym, sim, env_ptr, height: float = 2.0, thickness: float = 1.0):
        static_asset_option = gymapi.AssetOptions()
        static_asset_option.fix_base_link = True

        north_wall_asset = gym.create_box(sim, self.width + 2 * thickness, height, thickness, static_asset_option)
        south_wall_asset = gym.create_box(sim, self.width + 2 * thickness, height, thickness, static_asset_option)
        east_wall_asset = gym.create_box(sim, thickness, height, self.depth + 2 * thickness, static_asset_option)
        west_wall_asset = gym.create_box(sim, thickness, height, self.depth + 2 * thickness, static_asset_option)

        north_wall_pose = gymapi.Transform()
        south_wall_pose = gymapi.Transform()
        east_wall_pose = gymapi.Transform()
        west_wall_pose = gymapi.Transform()

        north_wall_pose.p = gymapi.Vec3(0.0, height / 2, self.depth / 2 + thickness / 2)
        south_wall_pose.p = gymapi.Vec3(0.0, height / 2, -(self.depth / 2 + thickness / 2))
        east_wall_pose.p = gymapi.Vec3(self.width / 2 + thickness, height / 2, 0.0)
        west_wall_pose.p = gymapi.Vec3(-(self.width / 2 + thickness), height / 2, 0.0)

        north_wall_handle = gym.create_actor(env_ptr, north_wall_asset, north_wall_pose, "north_wall", group=-1, filter=0)
        south_wall_handle = gym.create_actor(env_ptr, south_wall_asset, south_wall_pose, "south_wall", group=-1, filter=0)
        east_wall_handle = gym.create_actor(env_ptr, east_wall_asset, east_wall_pose, "east_wall", group=-1, filter=0)
        west_wall_handle = gym.create_actor(env_ptr, west_wall_asset, west_wall_pose, "west_wall", group=-1, filter=0)

        return (north_wall_handle, south_wall_handle, east_wall_handle, west_wall_handle)

    def build_in_isaac(self, gym, sim, env_ptr, height: float = 1.0) -> List[int]:
        handles = []
        wall_handles = self._build_walls(gym, sim, env_ptr, height)
        handles.extend(wall_handles)

        for obstacle in self.obstacles:
            obstacle_handle = obstacle.spawn(gym, sim, env_ptr, height)
            handles.append(obstacle_handle)

        return handles

    @classmethod
    def empty(cls) -> "Room":
        """Creates a simple 10x10 empty room for baseline testing."""
        return cls(width=10.0, depth=10.0, obstacles=[])

    @classmethod
    def standard(cls) -> "Room":
        """Creates a 20x20 room with a central pillar and a dividing wall."""
        obstacles = [
            BoxObstacle(x=0.0, z=0.0, width=2.0, depth=2.0),       # Center pillar
            BoxObstacle(x=5.0, z=5.0, width=8.0, depth=1.0),       # Dividing wall
        ]
        return cls(width=20.0, depth=20.0, obstacles=obstacles)


class OccupencyMap:
    def __init__(self, room: Room, config_path="config.yaml", resolution=0.05, robot_radius=0.17, safety_margin=0.05):
        self.res = resolution
        self.width_cells = int(room.width / resolution)
        self.depth_cells = int(room.depth / resolution)
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.robot_radius = self.config['robot']['radius']

        # 0 = Free, 1 = Obstacle
        self.grid = np.zeros((self.width_cells, self.depth_cells), dtype=np.uint8)

        for obstacle in room.obstacles:
            if hasattr(obstacle, 'width'):
                center_x = int((obstacle.x + room.width / 2.0) / resolution)
                center_z = int((obstacle.z + room.depth / 2.0) / resolution)
                
                half_width = int((obstacle.width / 2.0) / resolution)
                half_depth = int((obstacle.depth / 2.0) / resolution)
                
                self.grid[max(0, center_x-half_width):min(self.width_cells, center_x+half_width), 
                          max(0, center_z-half_depth):min(self.depth_cells, center_z+half_depth)] = 1

        # 2. Inflate obstacles
        effective_radius = robot_radius + safety_margin
        radius_cells = int(effective_radius / resolution)
        
        y, x = np.ogrid[-radius_cells:radius_cells+1, -radius_cells:radius_cells+1]
        kernel = (x**2 + y**2 <= radius_cells**2).astype(np.uint8)
        
        self.c_space_grid = binary_dilation(room:
  # Preset
  type: "standard"
  
  # Custom
  width: 20.0
  depth: 20.0

  obstacles:
    - type: "box"
      x: 0.0
      z: 0.0
      width: 2.0
      depth: 2.0
    - type: "box"
      x: 5.0
      z: 5.0
      width: 8.0
      depth: 1.0self.grid, structure=kernel).astype(np.uint8)
        
        # Mark outer walls as occupied
        self.c_space_grid[:radius_cells, :] = 1
        self.c_space_grid[-radius_cells:, :] = 1
        self.c_space_grid[:, :radius_cells] = 1
        self.c_space_grid[:, -radius_cells:] = 1
        
        valid_x, valid_z = np.where(self.c_space_grid == 0)
        self.valid_cells = list(zip(valid_x, valid_z))

    def sample_valid_pose(self): # Used to randomly set initial and end position
        """Returns a guaranteed collision-free (X, Z) world coordinate."""
        idx = np.random.randint(len(self.valid_cells))
        gx, gz = self.valid_cells[idx]
        
        world_x = (gx * self.res) - (self.width_cells * self.res / 2.0)
        world_z = (gz * self.res) - (self.depth_cells * self.res / 2.0)
        return world_x, world_z
    