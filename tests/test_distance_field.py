"""Tests for the 2D geodesic distance field on `core.room.OccupancyMap`.

CPU-only and Isaac-Gym-free. `core.room` imports `isaacgym.gymapi` at module
scope purely for scene building (`Obstacle.spawn`, `Room._build_walls`), which
nothing tested here touches, so a minimal stub is registered when Isaac Gym is
not importable (see `_isaacgym_available` / `_register_isaacgym_stub`).

Contract under test: `OccupancyMap.compute_distance_field(goal_x, goal_z)`
returns a room-local `np.ndarray` of shape `[width_cells, depth_cells]` (index 0
spans +X, index 1 spans +Z) holding the shortest obstacle-aware path distance in
metres from every cell to the goal, and finite everywhere.

Cells with no node in the free-space graph -- the inflated obstacle margin, the
boundary walls, and any pocket the goal cannot reach -- inherit the distance of
their nearest reachable free cell (see `TestCellsWithoutGraphNodes`).

These tests are deliberately one-cell tolerant: the cell lookup convention
(`int` truncation vs. nearest-cell rounding) is an implementation detail, and one
cell is 0.1 m at the resolution used below.
"""

import math
import sys
import types
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np

# --- Isaac Gym stub ---------------------------------------------------------
# `core.room` is imported for CPU occupancy-map math only. Register a stub so the
# module-level `from isaacgym import gymapi` resolves on machines without a GPU
# install. No simulator API is called anywhere in this module.


def _isaacgym_available() -> bool:
    """True when the real `isaacgym.gymapi` can be imported."""
    try:
        import importlib

        importlib.import_module("isaacgym.gymapi")
    except ImportError:
        return False
    return True


def _register_isaacgym_stub() -> None:
    """Register the minimal `isaacgym.gymapi` namespace `core.room` imports."""
    isaacgym = types.ModuleType("isaacgym")
    gymapi = types.ModuleType("isaacgym.gymapi")
    isaacgym.gymapi = gymapi
    sys.modules.setdefault("isaacgym", isaacgym)
    sys.modules.setdefault("isaacgym.gymapi", gymapi)


if not _isaacgym_available():
    _register_isaacgym_stub()

from core.room import BoxObstacle, OccupancyMap, Room

# Occupancy-map parameters used throughout (`Room.empty()` is 10 m x 10 m).
RESOLUTION = 0.1
ROBOT_RADIUS = 0.17
SAFETY_MARGIN = 0.05
MIN_START_GOAL_DIST = 1.0

#: Wall spanning x in [-0.5, 0.5], z in [-3.0, 3.0]: blocks the straight line
#: between (-2.0, 0.0) and (2.0, 0.0) but leaves a gap at both z extremes.
DIVIDING_WALL = {"x": 0.0, "z": 0.0, "width": 1.0, "depth": 6.0}


def make_occupancy_map(room: Room) -> OccupancyMap:
    """Build an `OccupancyMap` over `room` with the shared test parameters."""
    return OccupancyMap(
        room=room,
        resolution=RESOLUTION,
        robot_radius=ROBOT_RADIUS,
        safety_margin=SAFETY_MARGIN,
        min_start_goal_dist=MIN_START_GOAL_DIST,
    )


def grid_index(occ_map: OccupancyMap, x: float, z: float):
    """Nearest `(gx, gz)` cell of a room-local coordinate `(x, z)`.

    Mirrors the map's own conversion (`sample_valid_pose`): cell centres sit at
    `(index + 0.5) * res - size / 2`.
    """
    half_width = occ_map.width_cells * occ_map.res / 2.0
    half_depth = occ_map.depth_cells * occ_map.res / 2.0
    return (
        int(round((x + half_width) / occ_map.res)),
        int(round((z + half_depth) / occ_map.res)),
    )


def nearest_free_cell_distance(occ_map: OccupancyMap, dist_field, gx: int, gz: int) -> float:
    """Distance of the free cell closest to `(gx, gz)`.

    Independent oracle: a brute-force scan over the raw free mask, deliberately
    not the implementation's distance transform, so a change to the fill strategy
    cannot hide behind a shared helper.
    """
    free_x, free_z = np.where(occ_map.c_space_grid == 0)
    nearest = int(np.argmin((free_x - gx) ** 2 + (free_z - gz) ** 2))
    return float(dist_field[free_x[nearest], free_z[nearest]])


class TestDistanceField(unittest.TestCase):
    def test_goal_cell_is_zero(self):
        """The field is a distance to the goal, so the goal cell reads ~0 m."""
        occ_map = make_occupancy_map(Room.empty())
        dist_field = occ_map.compute_distance_field(0.0, 0.0)

        expected_shape = (occ_map.width_cells, occ_map.depth_cells)
        self.assertEqual(dist_field.shape, expected_shape)

        center_gx = occ_map.width_cells // 2
        center_gz = occ_map.depth_cells // 2
        value = float(dist_field[center_gx, center_gz])
        self.assertTrue(
            np.isclose(value, 0.0, atol=0.1),
            msg=f"distance at the goal cell was {value:.4f}, expected ~0.0",
        )

    def test_geodesic_greater_than_euclidean_with_obstacle(self):
        """A wall between start and goal forces a detour longer than the chord."""
        room = Room(width=10.0, depth=10.0, obstacles=[BoxObstacle(**DIVIDING_WALL)])
        occ_map = make_occupancy_map(room)

        start_x, start_z = -2.0, 0.0
        goal_x, goal_z = 2.0, 0.0
        euclidean = math.hypot(goal_x - start_x, goal_z - start_z)
        self.assertAlmostEqual(euclidean, 4.0, places=6)

        dist_field = occ_map.compute_distance_field(goal_x, goal_z)
        start_gx, start_gz = grid_index(occ_map, start_x, start_z)

        geodesic = float(dist_field[start_gx, start_gz])
        self.assertGreater(
            geodesic,
            euclidean + 1.0,
            msg=(
                f"geodesic distance {geodesic:.4f} m did not exceed the "
                f"{euclidean:.4f} m straight line by at least 1.0 m; the "
                "dividing wall looks like it is being crossed"
            ),
        )

    def test_grid_axis_alignment(self):
        """+X offsets index the first axis and +Z offsets the second axis."""
        occ_map = make_occupancy_map(Room.empty())
        dist_field = occ_map.compute_distance_field(0.0, 0.0)

        center_gx = occ_map.width_cells // 2
        center_gz = occ_map.depth_cells // 2
        offset = int(round(2.0 / occ_map.res))  # 2.0 m at this resolution

        # 2 m along +X: must be read from dist_field[gx, gz].
        self.assertGreater(float(dist_field[center_gx + offset, center_gz]), 1.5)
        # 2 m along +Z: must be read from the second index, not transposed.
        self.assertGreater(float(dist_field[center_gx, center_gz + offset]), 1.5)

        # Sanity check that the field is a 2D distance and not two decoupled
        # axis ramps: the diagonal corner is farther than either axis offset.
        self.assertGreater(
            float(dist_field[center_gx + offset, center_gz + offset]),
            float(dist_field[center_gx + offset, center_gz]),
        )

    def test_no_nans_or_infinities(self):
        """Every cell is finite, including obstacle interiors and margins."""
        rooms = {
            "empty": Room.empty(),
            "dividing_wall": Room(width=10.0, depth=10.0, obstacles=[BoxObstacle(**DIVIDING_WALL)]),
            "standard": Room.standard(),
        }
        goals = {"free_space": (3.0, -3.0), "inside_obstacle": (0.0, 0.0)}

        for room_name, room in rooms.items():
            occ_map = make_occupancy_map(room)
            for goal_name, (goal_x, goal_z) in goals.items():
                with self.subTest(room=room_name, goal=goal_name):
                    dist_field = occ_map.compute_distance_field(goal_x, goal_z)
                    self.assertEqual(
                        dist_field.shape, (occ_map.width_cells, occ_map.depth_cells)
                    )
                    self.assertTrue(
                        np.all(np.isfinite(dist_field)),
                        msg=(
                            f"{room_name}/{goal_name}: {(~np.isfinite(dist_field)).sum()} "
                            "cell(s) were NaN or infinite"
                        ),
                    )


class TestCellsWithoutGraphNodes(unittest.TestCase):
    """Regression guard for the fill of cells that carry no graph distance.

    These cells used to be assigned one sentinel, `max_finite + 2.0`. Measured on
    a 5x5 m room, that made the entire inflated margin read as farther away than
    the most distant reachable cell, so the whole margin (~35% of the cells)
    sorted behind genuine obstacles and the heuristic pushed the planner *against*
    the margin it was meant to avoid.

    Cells without a graph node now inherit the distance of their nearest
    reachable free cell. `core/room.py` documents the intent; these tests pin it.
    """

    #: Cells of margin around the free space, at the shipped 0.17 m robot radius
    #: and 0.05 m safety margin.
    MARGIN_CELLS = math.ceil((ROBOT_RADIUS + SAFETY_MARGIN) / RESOLUTION)

    def test_margin_cells_inherit_the_nearest_free_cell(self):
        """Every wall-margin cell reads the distance of the free cell closest to it."""
        occ_map = make_occupancy_map(Room.empty())
        dist_field = occ_map.compute_distance_field(0.0, 0.0)

        # Corners, edge midpoints, and the last row/column: all outside free space.
        probes = [
            (0, 0),
            (0, self.MARGIN_CELLS),
            (self.MARGIN_CELLS, 0),
            (occ_map.width_cells - 1, 0),
            (0, occ_map.depth_cells - 1),
            (occ_map.width_cells - 1, occ_map.depth_cells - 1),
        ]

        for gx, gz in probes:
            with self.subTest(cell=(gx, gz)):
                self.assertNotEqual(
                    occ_map.c_space_grid[gx, gz],
                    0,
                    "probe cell was expected to be outside free space",
                )
                expected = nearest_free_cell_distance(occ_map, dist_field, gx, gz)
                self.assertAlmostEqual(
                    float(dist_field[gx, gz]),
                    expected,
                    places=5,
                    msg=f"cell ({gx}, {gz}) did not inherit its nearest free cell",
                )

    def test_no_cell_reads_farther_than_the_farthest_free_cell(self):
        """The sentinel made the margin the maximum of the whole field."""
        rooms = {
            "empty": Room.empty(),
            "dividing_wall": Room(width=10.0, depth=10.0, obstacles=[BoxObstacle(**DIVIDING_WALL)]),
        }

        for room_name, room in rooms.items():
            occ_map = make_occupancy_map(room)
            dist_field = occ_map.compute_distance_field(2.0, -2.0)
            free = occ_map.c_space_grid == 0

            with self.subTest(room=room_name):
                self.assertTrue(free.any(), "room has no free cells to compare against")
                self.assertAlmostEqual(
                    float(dist_field.max()),
                    float(dist_field[free].max()),
                    places=5,
                    msg="a cell with no graph node read farther than every free cell",
                )

    def test_unreachable_free_cells_inherit_the_reachable_side(self):
        """A sealed-off pocket is filled from the nearest cell the goal can reach."""
        # A 0.6 m wall spanning the full depth: with the 0.22 m inflation it merges
        # into the side walls and cuts the room in two, so one half has free cells
        # that Dijkstra cannot reach from a goal on the other side.
        sealing_wall = BoxObstacle(x=0.0, z=0.0, width=0.6, depth=20.0)
        room = Room(width=10.0, depth=10.0, obstacles=[sealing_wall])
        occ_map = make_occupancy_map(room)

        goal_x = 2.0
        dist_field = occ_map.compute_distance_field(goal_x, 0.0)
        free = occ_map.c_space_grid == 0

        # Split the free space down the middle of the sealing wall; `free` is
        # indexed [gx, gz], so the split runs along axis 0.
        row = np.arange(occ_map.width_cells)[:, np.newaxis]
        sealed = free & (row < occ_map.width_cells // 2)
        reachable = free & (row >= occ_map.width_cells // 2)

        self.assertTrue(sealed.any() and reachable.any(), "expected two free regions")

        # Every sealed cell must carry a value that occurs on the reachable side,
        # i.e. it inherited one rather than being given a sentinel.
        reachable_values = {float(value) for value in dist_field[reachable]}
        for gx, gz in zip(*np.where(sealed)):
            with self.subTest(cell=(gx, gz)):
                self.assertIn(float(dist_field[gx, gz]), reachable_values)

        # And the sealed side must not dominate the field's range.
        self.assertLess(
            float(dist_field[sealed].max()),
            float(dist_field[reachable].max()),
        )


if __name__ == "__main__":
    unittest.main()
