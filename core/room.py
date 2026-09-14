from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

import math

import numpy as np
from isaacgym import gymapi
from scipy.ndimage import binary_dilation
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra


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

    def _build_walls(self, gym, sim, env_ptr, height: float = 2.0, thickness: float = 0.2):
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


class OccupancyMap:
    def __init__(self, room: Room, resolution, robot_radius, safety_margin, min_start_goal_dist):
        self.room = room
        self.res = resolution
        self.width_cells = int(room.width / resolution)
        self.depth_cells = int(room.depth / resolution)
        self.robot_radius = robot_radius
        self.min_start_goal_dist = min_start_goal_dist

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
        effective_radius = self.robot_radius + safety_margin
        # Round up so the represented clearance is never smaller than requested.
        radius_cells = math.ceil(effective_radius / resolution)
        
        y, x = np.ogrid[-radius_cells:radius_cells+1, -radius_cells:radius_cells+1]
        kernel = (x**2 + y**2 <= radius_cells**2).astype(np.uint8)
        
        self.c_space_grid = binary_dilation(self.grid, structure=kernel).astype(np.uint8)
        
        # Mark outer walls as occupied
        self.c_space_grid[:radius_cells, :] = 1
        self.c_space_grid[-radius_cells:, :] = 1
        self.c_space_grid[:, :radius_cells] = 1
        self.c_space_grid[:, -radius_cells:] = 1
        
        valid_x, valid_z = np.where(self.c_space_grid == 0)
        self.valid_cells = list(zip(valid_x, valid_z))

        # 3. Precompute the 8-connected free-space graph for geodesic queries.
        # Flat index convention: u = gx * self.depth_cells + gz.
        num_cells = self.width_cells * self.depth_cells
        free = self.c_space_grid == 0
        source_x, source_z = np.where(free)

        rows: List[int] = []
        cols: List[int] = []
        weights: List[float] = []

        orthogonal_offsets = ((-1, 0), (1, 0), (0, -1), (0, 1))
        diagonal_offsets = ((-1, -1), (-1, 1), (1, -1), (1, 1))

        for dx, dz in orthogonal_offsets + diagonal_offsets:
            neighbor_x = source_x + dx
            neighbor_z = source_z + dz

            in_bounds = (
                (neighbor_x >= 0) & (neighbor_x < self.width_cells)
                & (neighbor_z >= 0) & (neighbor_z < self.depth_cells)
            )
            if not np.any(in_bounds):
                continue

            src_x = source_x[in_bounds]
            src_z = source_z[in_bounds]
            dst_x = neighbor_x[in_bounds]
            dst_z = neighbor_z[in_bounds]

            passable = free[dst_x, dst_z]

            is_diagonal = dx != 0 and dz != 0
            if is_diagonal:
                # No corner cutting: a diagonal step is only legal when both
                # orthogonal cells sharing that corner are free too.
                passable &= free[dst_x, src_z] & free[src_x, dst_z]

            src_x = src_x[passable]
            src_z = src_z[passable]
            dst_x = dst_x[passable]
            dst_z = dst_z[passable]

            rows.extend((src_x * self.depth_cells + src_z).tolist())
            cols.extend((dst_x * self.depth_cells + dst_z).tolist())
            weight = float(self.res * math.sqrt(2)) if is_diagonal else float(self.res)
            weights.extend([weight] * src_x.size)

        self.graph = csr_matrix(
            (
                np.asarray(weights, dtype=np.float32),
                (
                    np.asarray(rows, dtype=np.int64),
                    np.asarray(cols, dtype=np.int64),
                ),
            ),
            shape=(num_cells, num_cells),
            dtype=np.float32,
        )

    def sample_valid_pose(self): # Used to randomly set initial and end position
        """Returns a guaranteed collision-free room-local (X, Z) coordinate."""
        idx = np.random.randint(len(self.valid_cells))
        gx, gz = self.valid_cells[idx]
        
        world_x = ((gx + 0.5) * self.res) - (self.width_cells * self.res / 2.0)
        world_z = ((gz + 0.5) * self.res) - (self.depth_cells * self.res / 2.0)
        return world_x, world_z

    def sample_valid_start_goal(self, min_dist: float = None, max_attempts: int = 1000):
        """Sample a (start, goal) pose pair separated by at least `min_dist`.

        Independently sampling start and goal can produce degenerate episodes
        (goal already reached, identical poses, or trivially close targets), so
        we resample until the Euclidean distance between them meets the minimum.
        Raises ValueError if `max_attempts` is exhausted without success.
        """
        if min_dist is None:
            min_dist = self.min_start_goal_dist

        for _ in range(max_attempts):
            start_x, start_z = self.sample_valid_pose()
            goal_x, goal_z = self.sample_valid_pose()
            dist = math.hypot(goal_x - start_x, goal_z - start_z)
            if dist >= min_dist:
                return start_x, start_z, goal_x, goal_z

        # Exhausted attempts (room too small for the requested separation).
        raise ValueError(
            f"Could not sample a start/goal pair at least {min_dist} m apart "
            f"after {max_attempts} attempts. Check room dimensions and "
            "room.start_goal_sampling.min_distance."
        )

    def compute_distance_field(self, goal_x: float, goal_z: float) -> np.ndarray:
        """Shortest obstacle-aware path distance from every cell to a goal.

        Returns a room-local field of shape ``(width_cells, depth_cells)`` in
        metres, indexed ``[gx, gz]`` with the same cell convention as
        ``sample_valid_pose``. Values come from Dijkstra over ``self.graph``,
        whose edge weights are metric (``res`` and ``res * sqrt(2)``), so the
        result is a geodesic distance rather than a hop count and is never
        smaller than the straight-line distance.

        A goal inside the inflated obstacle margin is snapped to the nearest
        free cell (the field then measures distance to that cell, because the
        graph contains no node for an occupied cell). Cells that are occupied
        or unreachable from the goal are assigned ``max_finite + 2.0`` so the
        field is finite and finite-differencing or interpolation over it never
        produces NaN or inf.
        """
        half_width = self.width_cells * self.res / 2.0
        half_depth = self.depth_cells * self.res / 2.0

        gx = min(max(int((goal_x + half_width) / self.res), 0), self.width_cells - 1)
        gz = min(max(int((goal_z + half_depth) / self.res), 0), self.depth_cells - 1)

        if self.c_space_grid[gx, gz] != 0:
            # Snap to the nearest free cell so Dijkstra has a valid source node.
            free_x, free_z = np.array(self.valid_cells, dtype=np.int64).T
            nearest = int(np.argmin((free_x - gx) ** 2 + (free_z - gz) ** 2))
            gx = int(free_x[nearest])
            gz = int(free_z[nearest])

        # 2D to 1D flattening
        goal_node = gx * self.depth_cells + gz
        dist_1d = dijkstra(csgraph=self.graph, directed=False, indices=goal_node)
        dist_2d = np.asarray(dist_1d, dtype=np.float64).reshape(
            self.width_cells, self.depth_cells
        )

        reachable = np.isfinite(dist_2d)
        max_val = float(np.max(dist_2d[reachable])) if np.any(reachable) else 0.0
        fallback = np.float64(max_val + 2.0)

        free_and_reachable = reachable & (self.c_space_grid == 0)
        dist_2d = np.where(free_and_reachable, dist_2d, fallback)

        return dist_2d.astype(np.float32)
    