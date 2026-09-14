# IsaacRoomba Roadmap

Living document: current capability, active milestone, deliberate deferrals. Architecture lives in
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md). Tags: **[current]**, **[later]**, **[speculative]**.

## Objective

Point-to-point navigation for a differential-drive robot under partial observability using Isaac Gym:
(1) a trustworthy fully-observable planner as reference — **[current]**; (2) RL baselines in the same
environment — **[later]**; (3) belief-space POMDP planning (POMCP/POMCGS) — **[later]**;
(4) teacher–student distillation — **[later]**. Only (1) exists in code.

## Milestone [current]

Characterize the existing fresh-tree MCTS baseline before starting POMCGS: versioned run logging,
recorded resolved configuration and provenance, a fixed-seed baseline, and a paired-seed budget
comparison. Must not change the planning algorithm, reward semantics, heuristic formula, or simulator
behavior; tree reuse, PBRS, and POMCGS are out of scope. The baseline runs end to end and produces
logs, but has not been benchmarked against a reference.

Done: versioned run-log schema with resolved config, Git provenance, seed, sampled start/goal and
termination reason; per-step metrics through a logger testable without Isaac Gym; TensorBoard from the
same metrics dict; reward and heuristic math extracted to `envs/planning_math.py` and tested on CPU;
`task.*` as the single owner of the constants; one 50-step fixed-seed run recorded.

Open:

- [X] Read `task.goal_radius` in `run_mcts.py` instead of its hard-coded `0.5` already-at-goal guard.
- [X] Guard the `None` action index `search()` can return and pass to best_action_idx.
- [X] Configuration and budget semantics: `env.env_spacing` (overwritten), the unused `pomcgs` block
- [X] Fail fast, in core/simulator.py (_load_room_layout), if a custom room configuration is missing width or depth, it silently defaults to 20.0.
- [X] The while loop in MCTSSolver.search depends on both `num_iterations` and `max_expansions`; Murky, check.
- [X] `num_planning_envs` >= action-grid size explicitly.
- [X] Fix the stale source text formerly catalogued in `docs/ARCHITECTURE.md`: the `generate()`
macro-action ticks comment, the `sample_valid_pose` docstring, the `sample_valid_start_goal` error key,
the `_compute_dones` comment, and the obstacle actor name typo.
- [X] `RoombaRLEnv`: add `task.timeout_steps` (or `env.max_episode_steps`) to `configs/config.yaml` and read it in `_compute_dones()` instead of hard-coded `3600`.
- [X] `RoombaRLEnv`: bind `_compute_rewards()` to `self.goal_reward` and `self.collision_penalty` instead of literal `10.0` and `5.0`.
- [X] `RoombaRLEnv`: expose dense distance shaping weight (`progress * 5.0`) in `configs/config.yaml` (e.g., `task.progress_weight`).
- [X] `RoombaRLEnv`: replace hard-coded `0.5` in `step()` info dictionary (`info["success"]`) with `self.goal_radius`.
- [X] Action space refactor (15-action grid -> 5 explicit actions):
  - [X] Update `configs/planners.yaml`: replace `v_vals` and `omega_vals` with an explicit `mcts.actions` list in exact canonical order (`[1.0, 0.0]`, `[1.0, 1.0]`, `[1.0, -1.0]`, `[0.0, 1.0]`, `[0.0, -1.0]`).
  - [X] Update `planners/mcts.py` (`_create_action_grid`): directly load `mcts.actions` as float32 tensor on device; fail fast if the key is missing; validate that entries are numeric pairs bounded within `[-1.0, 1.0]`.
  - [X] Update `planners/mcts.py` (`_validate_config`): check `num_actions > 0` and `num_actions <= num_envs`; scrub legacy parameter names (`v_vals`, `omega_vals`) from all error messages.
  - [X] Update `configs/config.yaml`: set `env.num_planning_envs: 9` (from 16) to form a 3x3 Isaac Gym grid and reduce padded simulation slots.
  - [X] Update `scripts/run_mcts.py`: update stale inline comment regarding `num_planning_envs` (16 -> 9) and action grid size (15 -> 5).
  - [X] Update documentation (`README.md` and `docs/ARCHITECTURE.md` §1, §2, §3, §4, §6, §7):
    - [X] Update action count (15 -> 5) and planning environments (16 -> 9).
    - [X] Mark the explicit action refactor in `test_mcts.py` as implemented and active.
    - [X] Clear stale claims asserting that `env.env_spacing` is dead or overwritten.
  - [X] Previous changes broke the log system, check (resolved: schema updated, 4-layer TrajectoryVisualizer added, flipbook replay verified).

- [X] debug_rl.py says 60 ticks is one second, but the simulator runs at 30 Hz, so it is two seconds.


#### Dijkstra Geodesik 2D Distance Field implementation

### Execution Phase 1: CPU Graph & Distance Field (`core/room.py`)

- [X] **Task 1.1: Create Distance Field Unit Test Scaffold (`tests/test_distance_field.py`)**

Task: Create CPU unit tests in `tests/test_distance_field.py` for 2D Geodesic Distance Field.

Scope:
- Create ONLY `tests/test_distance_field.py`. Do not modify any other files.
- Do not run GPU code or full test suite.

Requirements:
Write a unittest.TestCase (`TestDistanceField`) testing the expected contract of `OccupancyMap.compute_distance_field(goal_x: float, goal_z: float) -> np.ndarray`:
1. `test_goal_cell_is_zero`:
   - Instantiate `OccupancyMap` with a simple 10x10 empty room (`Room.empty()`, resolution=0.1, radius=0.17, margin=0.05, min_dist=1.0).
   - Compute distance field for goal (0.0, 0.0).
   - Assert that the value at the center cell index is close to 0.0 (`np.isclose(..., atol=0.1)`).
2. `test_geodesic_greater_than_euclidean_with_obstacle`:
   - Instantiate a room with a dividing wall obstacle (e.g. `Room(width=10.0, depth=10.0, obstacles=[BoxObstacle(x=0.0, z=0.0, width=1.0, depth=6.0)])`).
   - Pick a start pose at (-2.0, 0.0) and goal at (2.0, 0.0) on opposite sides of the wall.
   - Assert that `dist_field[start_gx, start_gz]` is strictly greater than Euclidean distance (4.0 m) by at least 1.0 m.
3. `test_grid_axis_alignment`:
   - In an empty room, compute field for goal (0.0, 0.0).
   - Test that moving +2.0m along X increases distance along the first grid axis (`dist_field[center_x + offset, center_z] > 1.5`), and +2.0m along Z increases distance along the second axis (`dist_field[center_x, center_z + offset] > 1.5`) without transposition.
4. `test_no_nans_or_infinities`:
   - In a room with obstacles, assert `np.all(np.isfinite(dist_field))` is True across every single cell, including obstacle margins and boundaries.

Verification Command:
python -m unittest tests/test_distance_field.py
(Expected: Fails with AttributeError or ImportError until Task 1.2 is implemented).


- [X] **Task 1.2: Implement Sparse 8-Connected Adjacency Matrix in `OccupancyMap.__init__` (`core/room.py`)**

Task: Precompute 8-connected sparse adjacency matrix in `OccupancyMap.__init__`.

Scope:
- Modify ONLY `core/room.py`. Do not touch other files.
- Do not run tests yet.

Specification:
1. In `core/room.py`, import `from scipy.sparse import csr_matrix`.
2. In `OccupancyMap.__init__`, after calculating `self.c_space_grid` and `self.valid_cells`:
   - Build an 8-connected grid graph over free cells (`self.c_space_grid == 0`).
   - Indexing convention: flat index `u = gx * self.depth_cells + gz`.
   - Step weights:
     - Orthogonal neighbors (dx, dz in [(-1,0), (1,0), (0,-1), (0,1)]): weight = float(self.res)
     - Diagonal neighbors (dx, dz in [(-1,-1), (-1,1), (1,-1), (1,1)]): weight = float(self.res * math.sqrt(2))
   - Corner-cutting prevention: A diagonal transition between (gx, gz) and (gx + dx, gz + dz) is ONLY added if BOTH orthogonal neighbors `(gx + dx, gz)` and `(gx, gz + dz)` have `self.c_space_grid == 0`.
   - Accumulate `rows`, `cols`, and `weights` lists, and construct `self.graph = csr_matrix((weights, (rows, cols)), shape=(num_cells, num_cells), dtype=np.float32)`.


- [X] **Task 1.3: Implement `OccupancyMap.compute_distance_field` & Finite Obstacle Handling (`core/room.py`)**

Task: Implement `compute_distance_field(goal_x, goal_z)` in `OccupancyMap`.

Scope:
- Modify ONLY `core/room.py`.
- Verification: `python -m unittest tests/test_distance_field.py`.

Specification:
1. In `core/room.py`, import `from scipy.sparse.csgraph import dijkstra`.
2. Implement `OccupancyMap.compute_distance_field(self, goal_x: float, goal_z: float) -> np.ndarray`:
   - Map continuous coordinates to grid:
     `gx = int((goal_x + self.room.width / 2.0) / self.res)`
     `gz = int((goal_z + self.room.depth / 2.0) / self.res)`
     Clamp `gx` to `[0, self.width_cells - 1]` and `gz` to `[0, self.depth_cells - 1]`.
   - If `(gx, gz)` is occupied (`self.c_space_grid[gx, gz] != 0`), find the nearest free cell from `self.valid_cells` by Euclidean distance to use as the source.
   - Target node index: `goal_node = gx * self.depth_cells + gz`.
   - Run `dist_1d = dijkstra(csgraph=self.graph, directed=False, indices=goal_node)`.
   - Reshape `dist_1d` to 2D array of shape `(self.width_cells, self.depth_cells)`.
   - Handle `inf` and obstacle cells:
     - Find `max_val = np.max(dist_2d[np.isfinite(dist_2d)])`. If all are inf, use `0.0`.
     - For any cell where `np.isinf(dist_2d)` or `self.c_space_grid != 0`: set value to `max_val + 2.0` (or euclidean distance to nearest valid boundary plus clearance penalty) to guarantee finite, non-NaN values across all cells.
   - Return `dist_2d.astype(np.float32)`.
3. Ensure `OccupancyMap` stores `self.room = room` in `__init__` if not already stored.

Verification Command:
python -m unittest tests/test_distance_field.py
(Expected: All 4 tests in tests/test_distance_field.py PASS).



---

### Execution Phase 2: Heuristic Math & Batched Sampling (`envs/planning_math.py`)

* [X] **Task 2.1: Add Geodesic Sampling Unit Tests (`tests/test_planning_math.py`)**

Task: Add unit tests for batched geodesic heuristic sampling in `tests/test_planning_math.py`.

Scope:
- Modify ONLY `tests/test_planning_math.py`. Do not touch simulator or CUDA code.

Requirements:
1. Open `tests/test_planning_math.py`.
2. Add test case `test_geodesic_heuristic_sampling`:
   - Create a synthetic `distance_field` tensor of shape `(1, 1, depth_cells, width_cells)` on CPU (e.g. linear gradient or known distance field, 50x50 cells, room 5.0m x 5.0m).
   - Construct a batch of physical 15D state tensors with known `(x, z)` coordinates.
   - Call `compute_straight_line_heuristic()` (or updated heuristic function) passing `distance_field`, `room_width=5.0`, `room_depth=5.0`.
   - Verify:
     - Outputs match expected values interpolated from the distance field.
     - Terminal states evaluate to 0.0.
     - States near obstacles evaluate to finite values without NaN or Inf.

Verification Command:
python -m unittest tests/test_planning_math.py
(Expected: Fails with TypeError/signature mismatch until Task 2.2 is implemented).


- [X] **Task 2.2: Update Heuristic Formula with Batched `grid_sample` (`envs/planning_math.py`)**

Task: Update `compute_straight_line_heuristic()` in `envs/planning_math.py` to sample geodesic distance via PyTorch `grid_sample`.

Scope:
- Modify ONLY `envs/planning_math.py`.
- Maintain pure PyTorch dependency (CPU/GPU compatible, no Isaac Gym imports).

Specification:
1. Update `compute_straight_line_heuristic(...)` signature to accept:
   - `distance_field: torch.Tensor` (shape `[1, 1, depth_cells, width_cells]`)
   - `room_width: float`
   - `room_depth: float`
   alongside the existing parameters (`states`, `dones`, `goal_reward`, `step_cost`, `goal_radius`, `max_linear_velocity`, `macro_action_duration`, `gamma`).
2. Coordinate Normalization for `grid_sample`:
   - Extract `x = states[:, 0]` and `z = states[:, 2]`.
   - Normalize coordinates to `[-1, 1]` range required by PyTorch:
     `norm_x = (x / (room_width / 2.0)).clamp(-1.0, 1.0)`
     `norm_z = (z / (room_depth / 2.0)).clamp(-1.0, 1.0)`
   - Form sampling grid of shape `[1, 1, batch_size, 2]`:
     `grid = torch.stack([norm_x, norm_z], dim=-1).unsqueeze(0).unsqueeze(0)`
   - Sample distances:
     `sampled = torch.nn.functional.grid_sample(distance_field, grid, mode="bilinear", padding_mode="border", align_corners=True)`
     `geodesic_dist = sampled.view(-1)`
3. Update remaining distance and Bellman return:
   - Replace Euclidean distance with:
     `d_remain = torch.clamp(geodesic_dist - goal_radius, min=0.0)`
   - Compute steps remaining: `H = torch.clamp(d_remain / max_dist_step, min=1.0)`
   - Compute Bellman heuristic return $V(s)$ using existing formula, zeroing out terminal states (`dones`).

Verification Command:
python -m unittest tests/test_planning_math.py
(Expected: All tests in tests/test_planning_math.py PASS).



---

### Execution Phase 3: Environment Caching & Planner Integration (`envs/planning_env.py`)

- [X] **Task 3.1: Implement Distance Field GPU Caching in `RoombaPlanningEnv` (`envs/planning_env.py`)**

Task: Cache precomputed geodesic distance field tensor on GPU in `envs/planning_env.py`.

Scope:
- Modify ONLY `envs/planning_env.py`.
- Do not launch viewer or run full simulations.

Specification:
1. In `RoombaPlanningEnv.__init__`:
   - Initialize `self._cached_goal = None` and `self._cached_distance_field = None`.
2. Implement helper `_get_or_update_distance_field(self, goal_x: float, goal_z: float) -> torch.Tensor`:
   - Check if `self._cached_goal` matches `(goal_x, goal_z)` within 1e-4 tolerance.
   - If matched and `self._cached_distance_field` is not None, return `self._cached_distance_field`.
   - Otherwise:
     - Call `dist_np = self.occupancy_map.compute_distance_field(goal_x, goal_z)`.
     - Reshape and transpose to shape `(1, 1, depth_cells, width_cells)`:
       Note: `dist_np` is `(width_cells, depth_cells)`, transpose to `(depth_cells, width_cells)` to align with PyTorch grid_sample (H=depth, W=width).
     - Convert to `torch.tensor(..., dtype=torch.float32, device=self.device)`.
     - Cache in `self._cached_distance_field` and update `self._cached_goal = (goal_x, goal_z)`.
     - Return tensor.


* [X] **Task 3.2: Wire Geodesic Heuristic into `compute_heuristic_values` (`envs/planning_env.py`)**

Task: Wire cached distance field into `RoombaPlanningEnv.compute_heuristic_values`.

Scope:
- Modify ONLY `envs/planning_env.py`.

Specification:
1. In `RoombaPlanningEnv.compute_heuristic_values(self, states, dones, gamma)`:
   - Extract current goal from `states[0, 13].item()` and `states[0, 14].item()`.
   - Fetch GPU tensor: `dist_field = self._get_or_update_distance_field(goal_x, goal_z)`.
   - Pass `distance_field=dist_field`, `room_width=self.room.width`, `room_depth=self.room.depth` into `compute_straight_line_heuristic(...)`.
2. Ensure all other task constants (`self.goal_reward`, `self.collision_penalty`, `self.step_cost`, `self.goal_radius`, `self.max_linear_velocity`, `self.macro_action_duration`) are forwarded cleanly.

Verification Command:
python -m unittest discover -s tests -v
(Expected: All CPU tests in tests/ pass cleanly).


---

### Execution Phase 4: Documentation & Roadmap Close

* [X] **Task 4.1: Update Architecture Documentation (`docs/ARCHITECTURE.md`)**

Task: Document 2D Geodesic Distance Field formulation in `docs/ARCHITECTURE.md`.

Scope:
- Modify ONLY `docs/ARCHITECTURE.md`. Do not touch code.

Requirements:
1. Locate Section 4 ("Heuristic and MCTS") in `docs/ARCHITECTURE.md`.
2. Update the heuristic description:
   - Replace straight-line Euclidean distance formulation with the 2D geodesic distance field formulation.
   - Document the precomputed 8-connected SciPy graph in `OccupancyMap` and corner-cutting prevention rule.
   - Document `csgraph.dijkstra` execution on CPU and the GPU tensor cache layout `[1, 1, depth_cells, width_cells]`.
   - Document batched evaluation via `torch.nn.functional.grid_sample` with `border` padding and `align_corners=True`.
3. Ensure all formulas and invariants match the implemented code.


### Execution Phase 1: Visualizer Unit Tests (`tests/test_spatial_plotter.py`)

* [X] **Task 1.1: Test Suite Scaffold & Contract Verification (`tests/test_spatial_plotter.py`)**

Task: Create CPU-only unit test scaffold for `TrajectoryVisualizer` in `tests/test_spatial_plotter.py`.

Scope:
- Create ONLY `tests/test_spatial_plotter.py`.
- Do not modify existing implementation files or launch GPU/simulator runtimes.

Requirements:
Write a `unittest.TestCase` class named `TestTrajectoryVisualizerContract` targeting the public contract of `tracking.spatial_plotter.TrajectoryVisualizer`:
1. Module Isolation:
   - Must be 100% CPU-safe and runnable without `isaacgym` or CUDA installed.
   - Configure `matplotlib.use("Agg")` at the absolute top of the module before importing `matplotlib.pyplot`.
2. Setup & Teardown:
   - `setUp`: Create a temporary output directory using `tempfile.mkdtemp()` wrapped in `pathlib.Path`. Define canonical room parameters: `room_bounds=(10.0, 10.0)`, `obstacles=[{"x": 0.0, "z": 0.0, "width": 2.0, "depth": 2.0}]`, `goal=(3.0, 3.0)`, `goal_radius=0.5`, `robot_radius=0.17`, `safety_margin=0.05`.
   - `tearDown`: Safely call `plt.close("all")` and recursively remove the temporary directory via `shutil.rmtree(..., ignore_errors=True)`.
3. `test_frame_file_creation_and_format`:
   - Instantiate `TrajectoryVisualizer(output_dir=self.temp_dir, ...)`.
   - Invoke `frame_path = visualizer.render_step(step=0, robot_pose=(-1.0, -1.0, 0.0))`.
   - Assert `frame_path.exists()` is True.
   - Assert that the parent directory of `frame_path` is named `frames` under `self.temp_dir`.
   - Assert that the filename exactly follows the zero-padded template `step_000.png`.
   - Assert `frame_path.stat().st_size > 0`.
   - Read the first 8 bytes of the file and assert byte-level equality with standard PNG magic bytes: `b"\x89PNG\r\n\x1a\n"`.

Verification Command:
python -m unittest tests/test_spatial_plotter.py
(Expected: Fails with `ModuleNotFoundError: No module named 'tracking.spatial_plotter'` or `ImportError`).


* [X] **Task 1.2: Kinematic History, Boundary Handling & Memory Leak Tests (`tests/test_spatial_plotter.py`)**
  
  Task: Implement kinematic tracking, boundary tolerance, and figure lifecycle tests in `tests/test_spatial_plotter.py`.

  Scope:
  - Modify ONLY `tests/test_spatial_plotter.py`.
  - Maintain pure standard library + NumPy + Matplotlib dependencies.

  Requirements:
  Add the following distinct test methods to `TestTrajectoryVisualizerContract`:
  1. `test_kinematic_history_tracking`:
    - Provide a sequence of 4 distinct robot poses: `[(-2.0, -2.0, 0.0), (-1.5, -1.8, 0.2), (-1.0, -1.5, 0.5), (-0.5, -1.0, 0.8)]`.
    - Sequentially call `render_step(step=i, robot_pose=pose)`.
    - Assert `len(visualizer.history_x) == 4` and `len(visualizer.history_z) == 4`.
    - Loop through indices and assert each coordinate matches the input poses using `math.isclose(..., abs_tol=1e-5)`.
    - Assert that sequential files `step_000.png`, `step_001.png`, `step_002.png`, `step_003.png` all exist simultaneously on disk.
  2. `test_boundary_and_out_of_bounds_resilience`:
    - Test coordinates at room perimeter extremes: `(5.0, 5.0, 0.0)`, `(-5.0, -5.0, 0.0)`.
    - Test coordinates exceeding room bounds: `(12.0, 0.0, 0.0)`, `(0.0, -15.0, 0.0)`.
    - Assert that all calls complete without raising `ValueError`, `IndexError`, or numerical exceptions, producing valid PNG files.
  3. `test_resource_cleanup_no_leaked_figures`:
    - Assert `len(plt.get_fignums()) == 0` prior to rendering.
    - Call `render_step(step=1, robot_pose=(0.0, 0.0, 0.0))`.
    - Assert `len(plt.get_fignums()) == 0` immediately following the call.
    - Execute a loop of 25 consecutive `render_step` calls and assert `len(plt.get_fignums()) == 0`, proving figures are closed via `plt.close(fig)` rather than leaking in memory.

  Verification Command:
  python -m unittest tests/test_spatial_plotter.py
  (Expected: Fails on imports until Phase 2 is implemented).

  ```


* [X] **Task 1.3: Extensibility & HUD Safety Unit Tests (`tests/test_spatial_plotter.py`)**

Task: Implement Layer 3 pluggable callback and Layer 4 HUD safety tests in `tests/test_spatial_plotter.py`.

Scope:
- Modify ONLY `tests/test_spatial_plotter.py`.

Requirements:
Add tests validating the decoupling of Layer 3 and Layer 4 from algorithm-specific internals:
1. `test_overlay_callback_executed_with_axes`:
   - Use `unittest.mock.MagicMock()` as a spy callback function.
   - Call `visualizer.render_step(step=1, robot_pose=(0.0, 0.0, 0.0), overlay_fn=spy_callback)`.
   - Assert `spy_callback.assert_called_once()`.
   - Extract the first positional argument passed to `spy_callback` and assert `isinstance(arg, matplotlib.axes.Axes)` is True.
   - Verify that passing `overlay_fn=None` executes normally without error.
2. `test_hud_rendering_with_various_metric_types`:
   - Test `hud_metrics=None` produces a valid image.
   - Test `hud_metrics={}` produces a valid image.
   - Test a rich diagnostic dictionary containing heterogeneous types:
     `{"Alg": "MCTS", "Step": 42, "Q/N": -3.1415, "Expanded": 16, "Reached": False, "NoneVal": None}`.
   - Assert that dictionary formatting does not throw exceptions and that the output file exists and has non-zero size.

Verification Command:
python -m unittest tests/test_spatial_plotter.py
(Expected: Fails on imports until Phase 2 is implemented).




---

### Execution Phase 2: Core Headless Renderer Engine (`tracking/spatial_plotter.py`)

* [X] **Task 2.1: Implement `TrajectoryVisualizer` Core Skeleton & Static Room Layer (`tracking/spatial_plotter.py`)**

Task: Implement `TrajectoryVisualizer` initialization and Layer 1 static geometry rendering in `tracking/spatial_plotter.py`.

Scope:
- Create ONLY `tracking/spatial_plotter.py`.
- Do not import `isaacgym`, PyTorch, or simulation harness files.
- Must remain strictly compatible with Python 3.8 (use `from __future__ import annotations` or `typing.Optional`, `typing.Tuple`, `typing.List`, `typing.Dict`).

Specification:
1. Module Configuration:
   - Add `from __future__ import annotations` at line 1.
   - Configure `matplotlib.use("Agg")` prior to importing `matplotlib.pyplot`.
   - Import `math`, `pathlib.Path`, `typing` constructs, `matplotlib.patches as patches`, `matplotlib.pyplot as plt`, and `numpy as np`.

2. `TrajectoryVisualizer.__init__`:
  - Signature:
    ```python
    def __init__(
        self,
        output_dir: Path | str,
        room_bounds: Tuple[float, float],
        obstacles: List[Dict[str, float]],
        goal: Tuple[float, float],
        goal_radius: float = 0.5,
        robot_radius: float = 0.17,
        safety_margin: float = 0.05,
        subfolder: str = "frames",
    ):
    ```
  - Explicitly unpack and store parameters as native floats/attributes:
      - `self.room_width = float(room_bounds[0])`
      - `self.room_depth = float(room_bounds[1])`
      - `self.goal_x = float(goal[0])`
      - `self.goal_z = float(goal[1])`
      - `self.goal_radius = float(goal_radius)`
      - `self.robot_radius = float(robot_radius)`
      - `self.safety_margin = float(safety_margin)`
      - `self.obstacles = obstacles`  - Calculate inflated margin: `self.inflation = float(robot_radius + safety_margin)`.
  - Create directory: `self.output_dir = Path(output_dir) / subfolder`; call `self.output_dir.mkdir(parents=True, exist_ok=True)`.
  - Initialize `self.history_x: List[float] = []` and `self.history_z: List[float] = []`.

3. `_draw_static_scene(self, ax: plt.Axes) -> None`:
  - Room Outer Bounds: Calculate `hw = self.room_width / 2.0`, `hd = self.room_depth / 2.0`. Add a rectangle patch with lower-left anchor `(-hw, -hd)`, width `self.room_width`, height `self.room_depth`, `fc="none"`, `ec="black"`, `lw=2`, `zorder=1`.
  - Obstacles: Iterate through `self.obstacles` (each with `'x'`, `'z'`, `'width'`, `'depth'`).
    - Inflated footprint: Add rectangle with lower-left anchor `(ox - w/2.0 - self.inflation, oz - d/2.0 - self.inflation)`, width `w + 2*self.inflation`, height `d + 2*self.inflation`, `ls="--"`, `ec="gray"`, `fc="none"`, `lw=1`, `zorder=2`.
    - Obstacle body: Add rectangle with lower-left anchor `(ox - w/2.0, oz - d/2.0)`, width `w`, height `d`, `fc="dimgray"`, `ec="black"`, `lw=1.5`, `zorder=3`.
  - Goal Target:
    - Semi-transparent circle (`color="green"`, `alpha=0.2`, `zorder=2`) at `(self.goal_x, self.goal_z)` with radius `self.goal_radius`.
    - Center cross marker (`marker="x"`, `color="green"`, `markersize=8`, `zorder=3`) at `(self.goal_x, self.goal_z)`.


* [X] **Task 2.2: Implement Kinematics, Callback Hook & Generic HUD Layers (`tracking/spatial_plotter.py`)**


Task: Implement Layer 2 (Kinematics), Layer 3 (Pluggable Delegate), Layer 4 (HUD), and image persistence in `TrajectoryVisualizer.render_step`.

Scope:
- Modify ONLY `tracking/spatial_plotter.py`.

Specification:
1. Module Configuration & Imports:
   - Ensure `from typing import Any, Callable, Dict, List, Optional, Tuple` is imported.
   - Ensure `import math` is imported.

2. Implement `render_step`:
   ```python
   def render_step(
       self,
       step: int,
       robot_pose: Tuple[float, float, float],
       hud_metrics: Optional[Dict[str, Any]] = None,
       overlay_fn: Optional[Callable[[plt.Axes], None]] = None,
   ) -> Path:

3. Coordinate Ingestion & History:
* Unpack coordinates: `x, z, yaw = float(robot_pose[0]), float(robot_pose[1]), float(robot_pose[2])`.
* Append to history: `self.history_x.append(x)` and `self.history_z.append(z)`.

4. Canvas Lifecycle & Rendering Flow:
* Create figure: `fig, ax = plt.subplots(figsize=(6, 6), dpi=100)`.
* Enclose rendering and persistence in a `try ... finally: plt.close(fig)` block to guarantee figure cleanup:
  ```python
  try:
      # Layer 1: Static scene geometry
      self._draw_static_scene(ax)

      # Layer 2: Kinematics (trail, body, heading)
      self._draw_kinematics(ax, x, z, yaw)

      # Layer 3: Pluggable overlay callback
      if overlay_fn is not None:
          overlay_fn(ax)

      # Layer 4: HUD metrics card
      if hud_metrics:
          self._draw_hud(ax, hud_metrics)

      # Axis bounds, scaling, and decoration removal
      hw = self.room_width / 2.0
      hd = self.room_depth / 2.0
      ax.set_xlim(-hw, hw)
      ax.set_ylim(-hd, hd)
      ax.set_aspect("equal")
      ax.axis("off")

      # Persist frame
      frame_path = self.output_dir / f"step_{step:03d}.png"
      fig.savefig(frame_path, bbox_inches="tight", pad_inches=0.05)
      return frame_path
  finally:
      plt.close(fig)
  ```

5. Helper Method `_draw_kinematics(self, ax: plt.Axes, x: float, z: float, yaw: float) -> None`:
* Breadcrumb Trail: If `len(self.history_x) > 1`, draw history path with `ax.plot(self.history_x, self.history_z, color="royalblue", lw=1.5, alpha=0.7, zorder=4)`.
* Robot Body: Add circular patch `patches.Circle((x, z), radius=self.robot_radius, color="deepskyblue", ec="black", lw=1.2, zorder=5)`.
* Heading Arrow:
* Compute vector offsets:
`arrow_len = self.robot_radius * 1.3`
`dx = arrow_len * math.cos(yaw)`
`dz = arrow_len * math.sin(yaw)`
* Draw arrow using:
`ax.arrow(x, z, dx, dz, head_width=0.05, head_length=0.05, fc="crimson", ec="crimson", zorder=6)`.

6. Helper Method `_draw_hud(self, ax: plt.Axes, hud_metrics: Dict[str, Any]) -> None`:
* Format lines into a single newline-delimited string:
`text = "\n".join(f"{k}: {v}" for k, v in hud_metrics.items())`
* Render HUD text card:
`ax.text(0.03, 0.97, text, transform=ax.transAxes, fontsize=8, fontfamily="monospace", va="top", bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.85), zorder=10)`.


* [X] **Task 2.3: Package Export & CPU Verification Gate (`tracking/__init__.py`)**

Task: Expose `TrajectoryVisualizer` in `tracking/__init__.py` and verify all visualizer unit tests.

Scope:
- Modify ONLY `tracking/__init__.py`.
- Run CPU verification tests.

Specification:
1. In `tracking/__init__.py`, import `TrajectoryVisualizer` from `tracking.spatial_plotter`.

2. Add `"TrajectoryVisualizer"` to `__all__` alongside `MCTSRunLogger`, `STEP_COLUMNS`, etc.

3. Run the unit test suite to verify full compliance.

Verification Command:
python -m unittest tests/test_spatial_plotter.py
(Expected: All tests in tests/test_spatial_plotter.py PASS with green status in < 1 second).


---

### Execution Phase 3: Run Logger Integration (`tracking/run_logger.py`)

* [X] **Task 3.1: Schema Update & Step Column Tests (`tests/test_run_logger.py`)**

Task: Update `tests/test_run_logger.py` to enforce `robot_x`, `robot_z`, and `robot_yaw` columns in `steps.csv`.

Scope:
- Modify ONLY `tests/test_run_logger.py`.

Requirements:
1. Locate mock step metric fixtures in `tests/test_run_logger.py`.

2. Update all step metric mock payloads to include:
   - `"robot_x": -0.725`
   - `"robot_z": -0.625`
   - `"robot_yaw": 0.0`

3. Add test assertion verifying that `STEP_COLUMNS` contains `"robot_x"`, `"robot_z"`, and `"robot_yaw"` immediately following `"step"`.

4. Add a test case `test_missing_pose_columns_raise_keyerror`:
   - Attempting to call `log_step` without `"robot_x"` must raise `KeyError`.

5. Add a test case `test_run_logger_initializes_visualizer_when_enabled`:
   - Instantiate `MCTSRunLogger` with `enable_visualizer=True`, `room_bounds=(10.0, 10.0)`, `obstacles=[]`, `goal=(2.0, 2.0)`.
   - Log one step.
   - Assert that `(logger.run_dir / "frames" / "step_001.png").exists()` is True.

Verification Command:
python -m unittest tests/test_run_logger.py
(Expected: Fails until Task 3.2 and Task 3.3 update tracking/run_logger.py).


* [X] **Task 3.2: Update `MCTSRunLogger` Schema & CSV Writing (`tracking/run_logger.py`)**

Task: Add `robot_x`, `robot_z`, and `robot_yaw` to `STEP_COLUMNS` in `tracking/run_logger.py`.

Scope:
- Modify ONLY `tracking/run_logger.py`.
- Maintain backwards compatibility for existing serialization utilities (`_to_yaml_safe`, `collect_git_metadata`).

Specification:
1. In `tracking/run_logger.py`, update `STEP_COLUMNS` tuple:
  ```python
  STEP_COLUMNS = (
      "step",
      "robot_x",
      "robot_z",
      "robot_yaw",
      "action_v_normalized",
      "action_w_normalized",
      "root_value",
      "tree_size",
      "max_depth",
      "distance_to_goal_m",
      "executed_reward",
      "search_time_sec",
      "execution_time_sec",
      "expansion_calls",
      "physics_ticks",
      "decisions_per_sec",
  )
  ```

2. In `TENSORBOARD_TAGS`, optionally add:
* `"robot_x": "trajectory/robot_x"`
* `"robot_z": "trajectory/robot_z"`
* `"robot_yaw": "trajectory/robot_yaw"`
(This records pose scalars to TensorBoard alongside CSV rows without schema conflict).

3. Ensure `log_step` column validation continues to verify that every column in `STEP_COLUMNS` is present in the `metrics` mapping, failing fast on missing keys.


* [X] **Task 3.3: Lifecycle Integration & Exception Shielding in `MCTSRunLogger` (`tracking/run_logger.py`)**

Task: Wire `TrajectoryVisualizer` lifecycle and error shielding into `MCTSRunLogger`.

Scope:
- Modify ONLY `tracking/run_logger.py`.

Specification:

1. In `tracking/run_logger.py`, import `from tracking.spatial_plotter import TrajectoryVisualizer`.

2. Update `MCTSRunLogger.__init__` signature to accept optional visualization configuration:
   ```python
   def __init__(
       self,
       output_dir="logs/mcts",
       run_id=None,
       *,
       environment_config=None,
       planner_config=None,
       experiment_config=None,
       scenario=None,
       initial_distance=None,
       enable_tensorboard=False,
       tensorboard_writer=None,
       repo_root=None,
       timestamp=None,
       enable_visualizer=False,
       room_bounds=None,
       obstacles=None,
       goal=None,
       goal_radius=0.5,
   ):
  ```

3. In `__init__`:
* Initialize `self.visualizer: Optional[TrajectoryVisualizer] = None`.
* If `enable_visualizer` is True and `room_bounds` and `goal` are provided:
Initialize `self.visualizer = TrajectoryVisualizer(output_dir=self.run_dir, room_bounds=room_bounds, obstacles=obstacles or [], goal=goal, goal_radius=goal_radius)`.

4. Update `log_step`:
  ```python
  def log_step(
      self,
      metrics: Mapping,
      hud_metrics: Optional[Dict[str, Any]] = None,
      overlay_fn: Optional[Callable] = None,
  ):
  ```

* Write the CSV row and TensorBoard scalars first (authoritative data is saved first).

* If `self.visualizer is not None`:
Wrap rendering in a safety block so visualization errors never abort data logging:
  ```python
  try:
      pose = (float(metrics["robot_x"]), float(metrics["robot_z"]), float(metrics["robot_yaw"]))
      self.visualizer.render_step(
          step=int(metrics["step"]),
          robot_pose=pose,
          hud_metrics=hud_metrics,
          overlay_fn=overlay_fn,
      )
  except Exception as exc:
      print(f"Warning: failed to render frame for step {metrics.get('step')}: {exc}")

  ```

Verification Command:
python -m unittest tests/test_run_logger.py
(Expected: All tests in tests/test_run_logger.py PASS).


---

### Execution Phase 4: Configuration & Live Runner Integration (`configs/experiments.yaml` & `scripts/run_mcts.py`)

* [X] **Task 4.1: Add Visualization Toggle in `configs/experiments.yaml**`

Task: Add `render_frames` configuration toggle to `configs/experiments.yaml`.

Scope:
- Modify ONLY `configs/experiments.yaml`.

Specification:
1. Open `configs/experiments.yaml`.

2. Locate `mcts.logging`.

3. Add `render_frames: true` under `logging:` with an explanatory comment:
   ```yaml
   mcts:
     seed: 0

     max_execution_steps: 300
     no_progress:
       no_progress_steps: 50
       no_progress_threshold: 0.02
     logging:
       output_dir: "logs/mcts"
       tensorboard: true
       render_frames: true # Generate 2D top-down PNG flipbook frames under <run_dir>/frames/
    ```

4. Validate YAML syntax is clean and parseable with `yaml.safe_load`.


* [X] **Task 4.2: Quaternion Planar Yaw Extraction & Step 0 Spawn Frame in `scripts/run_mcts.py**`

Task: Add planar yaw extraction and Step 0 spawn frame logging in `scripts/run_mcts.py`.

Scope:
- Modify ONLY `scripts/run_mcts.py`.

Specification:

1. Helper Function:
  In `scripts/run_mcts.py`, define the planar yaw helper:
  ```python
  def extract_yaw_from_quaternion(qy: float, qw: float) -> float:
      """Compute planar yaw angle (radians) in 2D top-down (X=horizontal, Z=vertical) frame.
       
      In Y-up frame (+X forward, +Z right), a CCW yaw rotates forward (+X) into -Z.
      """
      return math.atan2(-2.0 * qy * qw, qw * qw - qy * qy)
  ```

2. Logger Initialization:
Extract room bounds and obstacles directly from env.sim.room:
  ```python
  room_bounds = (float(env.sim.room.width), float(env.sim.room.depth))
  obstacles = [
      {"x": float(o.x), "z": float(o.z), "width": float(o.width), "depth": float(o.depth)}
      for o in env.sim.room.obstacles
      if hasattr(o, "width")
  ]
  run_logger = MCTSRunLogger(
      output_dir=logging_cfg["output_dir"],
      environment_config=env.config,
      planner_config=mcts.config,
      experiment_config=experiment_config,
      scenario={
          "seed": seed,
          "start": {"x": start_x, "z": start_z},
          "goal": {"x": goal_x, "z": goal_z},
      },
      initial_distance=initial_dist,
      enable_tensorboard=bool(logging_cfg["tensorboard"]),
      enable_visualizer=bool(logging_cfg.get("render_frames", False)),
      room_bounds=room_bounds,
      obstacles=obstacles,
      goal=(goal_x, goal_z),
      goal_radius=env.goal_radius,
  )
  ```

3. Initial Spawn Telemetry (Step 0):
Immediately after teleporting the robot and initializing `run_logger`, record Step 0 representing the spawn pose before actions begin:
  ```python
  initial_yaw = extract_yaw_from_quaternion(current_state[4].item(), current_state[6].item())
  run_logger.log_step(
      {
          "step": 0,
          "robot_x": round(start_x, 4),
          "robot_z": round(start_z, 4),
          "robot_yaw": round(initial_yaw, 4),
          "action_v_normalized": 0.0,
          "action_w_normalized": 0.0,
          "root_value": 0.0,
          "tree_size": 1,
          "max_depth": 0,
          "distance_to_goal_m": round(initial_dist, 4),
          "executed_reward": 0.0,
          "search_time_sec": 0.0,
          "execution_time_sec": 0.0,
          "expansion_calls": 0,
          "physics_ticks": 0,
          "decisions_per_sec": 0.0,
      },
      hud_metrics={"Status": "Spawn Pose", "Dist": f"{initial_dist:.2f}m"},
  )
  ```


* [X] **Task 4.3: Wire Execution Step Poses & HUD Telemetry in `scripts/run_mcts.py**`

Task: Supply live robot coordinates and HUD diagnostics in `scripts/run_mcts.py` execution loop.

Scope:
- Modify ONLY `scripts/run_mcts.py`.

Specification:
1. In `scripts/run_mcts.py`, locate the execution loop where `current_state = next_states[0].clone()` updates after `env.generate()`.
2. Extract current planar pose:
  ```python
  robot_x = current_state[0].item()
  robot_z = current_state[2].item()
  robot_yaw = extract_yaw_from_quaternion(current_state[4].item(), current_state[6].item())
  ```

3. In `run_logger.log_step(...)`:
* Add `"robot_x": round(robot_x, 4)`
* Add `"robot_z": round(robot_z, 4)`
* Add `"robot_yaw": round(robot_yaw, 4)`

4. Construct `hud_metrics` dictionary:
  ```python
  hud_metrics = {
      "Step": f"{steps:03d}",
      "Dist": f"{dist_to_goal:.2f}m",
      "Action": f"v={best_action[0]:.1f}, w={best_action[1]:.1f}",
      "Root Q/N": f"{root_value:.3f}",
      "Nodes": tree_size,
      "Expansions": num_expansions,
  }
  ```

5. Pass `hud_metrics=hud_metrics` into `run_logger.log_step(metrics_dict, hud_metrics=hud_metrics)`.


---

### Execution Phase 5: Offline Replay & GIF/Flipbook Utility (`tracking/spatial_plotter.py` & `scripts/render_trajectory.py`)

* [X] **Task 5.1: Implement `@classmethod from_run_log` & `render_from_csv` (`tracking/spatial_plotter.py`)**

Task: Add offline replay loader and CSV renderer in `tracking/spatial_plotter.py` and unit tests in `tests/test_spatial_plotter.py`.

Scope:
- Modify `tracking/spatial_plotter.py`.
- Modify `tests/test_spatial_plotter.py`.
- Maintain CPU-only, simulator-free operation.

Specification:
1. Imports Update in `tracking/spatial_plotter.py`:
   - Add `import csv` and `import yaml`.

2. Implement `@classmethod from_run_log(cls, run_dir: Path | str) -> TrajectoryVisualizer`:
   - Resolve path: `run_dir = Path(run_dir)`. Read `run.yaml` using `yaml.safe_load`.
   - Goal extraction:
     `goal_dict = metadata["scenario"]["goal"]`
     `goal = (float(goal_dict["x"]), float(goal_dict["z"]))`
   - Radii extraction:
     `env_cfg = metadata.get("configuration", {}).get("environment", {})`
     `goal_radius = float(env_cfg.get("task", {}).get("goal_radius", 0.5))`
     `robot_radius = float(env_cfg.get("robot", {}).get("radius", 0.17))`
   - Room geometry extraction:
     - Check `scenario.geometry` first if present:
       `geom = metadata.get("scenario", {}).get("geometry", {})`
       `room_bounds = tuple(geom["room_bounds"]) if "room_bounds" in geom else None`
       `obstacles = geom.get("obstacles", None)`
     - Fall back to parsing `env_cfg.get("room", {})`:
       - If `room_cfg.get("type") == "custom"`:
         `custom = room_cfg["custom"]`
         `room_bounds = (float(custom["width"]), float(custom["depth"]))`
         `obstacles = custom.get("obstacles", [])`
       - Else if `room_cfg.get("type") == "standard"`:
         `room_bounds = (20.0, 20.0)`
         `obstacles = [{"x": 0.0, "z": 0.0, "width": 2.0, "depth": 2.0}, {"x": 5.0, "z": 5.0, "width": 8.0, "depth": 1.0}]`
       - Else (e.g. "empty"):
         `room_bounds = (10.0, 10.0)`
         `obstacles = []`
   - Return `cls(output_dir=run_dir, room_bounds=room_bounds, obstacles=obstacles, goal=goal, goal_radius=goal_radius, robot_radius=robot_radius)`.

3. Implement `render_from_csv(self, csv_path: Path | str) -> List[Path]`:
   - Open and read `csv_path` with `csv.DictReader`.
   - Validate pose columns:
     `required = {"step", "robot_x", "robot_z", "robot_yaw"}`
     `if not required.issubset(reader.fieldnames or []):`
         `raise ValueError(f"CSV missing required columns: {required - set(reader.fieldnames or [])}")`
   - Clear internal history: `self.history_x.clear()`, `self.history_z.clear()`.
   - Iterate over rows:
     - Parse step: `step = int(row["step"])`.
     - Parse pose: `pose = (float(row["robot_x"]), float(row["robot_z"]), float(row["robot_yaw"]))`.
     - Construct HUD dictionary:
       `hud = {}`
       `for key, label in [("distance_to_goal_m", "Dist"), ("root_value", "Root Q/N"), ("tree_size", "Nodes")]:`
       `    if key in row and row[key] != "":`
       `        hud[label] = row[key]`
     - Render frame: `frame_path = self.render_step(step=step, robot_pose=pose, hud_metrics=hud)`.
     - Collect and return list of generated `Path` objects.

4. Unit Tests in `tests/test_spatial_plotter.py`:
   - Add test method `test_from_run_log_and_render_from_csv`:
     - Create mock `run.yaml` and `steps.csv` fixtures inside `self.temp_dir`.
     - Instantiate visualizer via `TrajectoryVisualizer.from_run_log(self.temp_dir)`.
     - Call `frames = visualizer.render_from_csv(self.temp_dir / "steps.csv")`.
     - Assert generated files exist and pass PNG signature verification.

Verification Command:
python -m unittest tests/test_spatial_plotter.py
(Expected: All tests PASS).


* [X] **Task 5.2: Create Standalone CLI Offline Renderer (`scripts/render_trajectory.py`)**

Task: Create a standalone CLI script `scripts/render_trajectory.py` to replay any historical run directory into 2D flipbook frames and an optional animated GIF.

Scope:
- Create ONLY `scripts/render_trajectory.py`.
- No GPU or Isaac Gym dependencies.

Specification:
1. CLI Arguments (using `argparse`):
   - `run_dir`: Positional string path to a run folder (e.g. `logs/mcts/mcts_20260914_055025/`).
   - `--gif`: Optional flag to stitch generated PNGs into `trajectory.gif` in the run directory.
   - `--fps`: Optional integer frame rate for the GIF (default: 5).
2. Execution:
   - Check that `run_dir / "run.yaml"` and `run_dir / "steps.csv"` exist.
   - Instantiate visualizer via `TrajectoryVisualizer.from_run_log(run_dir)`.
   - Call `frame_paths = visualizer.render_from_csv(run_dir / "steps.csv")`.
   - If `--gif` is requested:
     - Attempt to import `imageio` or `PIL.Image`.
     - If `PIL.Image` is available:
       ```python
       images = [Image.open(p) for p in frame_paths]
       images[0].save(run_dir / "trajectory.gif", save_all=True, append_images=images[1:], duration=int(1000/fps), loop=0)
       ```
     - Print: `"Saved GIF animation to <run_dir>/trajectory.gif"`.
3. Test CLI via dry run on existing mock directory:
   `python scripts/render_trajectory.py --help`


---

### Execution Phase 6: Architectural Truth, Agent Guidelines & Milestone Close

- [X] **Task 6.1: Update Architecture Documentation (`docs/ARCHITECTURE.md`)**

Task: Document the 2D spatial logging architecture and updated `steps.csv` schema in `docs/ARCHITECTURE.md`.

Scope:
- Modify ONLY `docs/ARCHITECTURE.md`. Do not touch code.

Requirements:
1. In Section 1 ("Layers and flow"):
   - Add `tracking/spatial_plotter.py` to the table under `tracking/`, documenting its responsibility: "Headless 2D Matplotlib trajectory visualizer and flipbook frame generator".
2. In Section 6 ("Configuration and logging"):
   - Update the documented `steps.csv` schema to include `robot_x`, `robot_z`, and `robot_yaw`.
   - Document the `frames/step_XXX.png` artifact structure.
   - Document the 4-layer composition model (Static Scene, Kinematics, Delegate Overlay, Diagnostic HUD).
   - Document the `mcts.logging.render_frames` configuration toggle in `configs/experiments.yaml`.
3. Ensure all paths, tensor names, and layer invariants match the implemented codebase.


- [X] **Task 6.2: Synchronize Agent Instructions & Invariants (`AGENTS.md`)**

Task: Update `AGENTS.md` with visualizer invariants and updated CPU test suite counts.

Scope:
- Modify ONLY `AGENTS.md`.

Requirements:
1. In Section "Hard constraints":
   - Note that `steps.csv` contains `robot_x`, `robot_z`, `robot_yaw` and any alterations to `STEP_COLUMNS` require updating `tracking/run_logger.py`, `scripts/run_mcts.py`, and `tests/test_run_logger.py`.
2. In Section "Verification and writing":
   - Re-assert that `tests/` is CPU-only and passes without Isaac Gym or a GPU.
   - Update test command gate references to include `tests/test_spatial_plotter.py`.


- [X] **Task 6.3: Roadmap Close & Milestone State Transition (`ROADMAP.md`)**

Task: Mark the spatial logging and visualizer milestone items complete in `ROADMAP.md`.

Scope:
- Modify ONLY `ROADMAP.md`.

Requirements:
1. Under Milestone [current] "Open:", locate:
   `- [ ] Previous changes broke the log system, check.`
   Change to:
   `- [X] Previous changes broke the log system, check (resolved: schema updated, 4-layer TrajectoryVisualizer added, flipbook replay verified).`
2. Mark Tasks 6.1, 6.2, and 6.3 as completed (`* [X]`).
3. Do not alter planned sweep or deferred decision sections.


- [ ] Log Tests and correspond with version. Commit more often?


`tests/` is tracked and CPU-only; `python -m unittest discover -s tests` passes 76 tests with no GPU, so it can serve as the pre-experiment gate. `AGENTS.md` still carries the older "untracked and red" assumption.

## Planned experiments (prepared, not launched)

All sweeps execute **after** the 5-action refactor and test suite pass, running against the identical 5-action model with fixed `gamma = 0.95`, `c_param = 1.414`. 

### Objective & Hypothesis
Characterize the relationship between planning budget (`num_iterations`) and navigation performance on the 5-action space. We hypothesize that expanding the search budget from 30 to 300 iterations per decision significantly reduces trajectory steps to the goal and prevents premature stagnation (`no_progress` terminations) caused by local obstacle minima.

### Design & Sweep Configuration

| Sweep Arm | Iteration Budget (`num_iterations`) | Seeds | Fixed Hyperparameters |
| --- | --- | --- | --- |
| Low budget | `30` | `[0, 1, 2, 3, 4]` | `c_param = 1.414`, `gamma = 0.95` |
| Nominal baseline | `100` | `[0, 1, 2, 3, 4]` | `c_param = 1.414`, `gamma = 0.95` |
| High budget | `300` | `[0, 1, 2, 3, 4]` | `c_param = 1.414`, `gamma = 0.95` |

* **Paired seeds (15 runs total):** The seed deterministically sets the initial robot and goal positions (`sample_valid_start_goal`) as well as tree expansion tie-breaking. Using paired seeds guarantees that all three iteration budgets face the exact same 5 physical scenarios.
* **Exclusions:**
  * `c_param` exploration is dropped: UCB1 exploration scaling is arbitrary without reward normalization, so tuning it introduces noise without addressing budget efficiency.
  * `gamma` remains fixed at 0.95 because it couples tree backpropagation with the straight-line leaf heuristic.

### Evaluation Metrics

* **Primary Outcome Metrics (Task Success & Efficiency):**
  * **Success Rate:** Percentage of runs terminating in `goal_reached` vs. `no_progress` or `execution_horizon`.
  * **Steps to Goal:** Total physical macro-actions executed by the robot to reach the target.
  * **Final Distance (m):** Euclidean distance to the goal pose upon termination.
* **Secondary Diagnostics (Compute & Tree Structure):**
  * **Latency:** Search time per decision (sec) and execution throughput (decisions/sec).
  * **Tree Characteristics:** Tree size (total nodes generated) and maximum tree depth reached.
  * **Simulation Overhead:** Expansion calls (`generate()` calls) and total PhysX simulator ticks.
## Later milestones

- **M2 — Search efficiency and budgets [later]**: retain and re-root the tree across control steps,
  correct `N`/`Q` on re-rooting, gate reuse on predicted-versus-executed successors, benchmark fresh
  versus retained trees, and add time/tick/call-based budgets.
- **M3 — RL environment stabilization [later]**: configurable timeout, viewer-close handling, no unused
  reward inputs, shaping validated against a trivial policy, `num_rl_envs` raised with throughput
  verified.
- **M4 — RL baselines [later]**: PPO and/or SAC against the same environment, configuration, and
  metrics, compared with the planner on a shared scenario. Nothing exists today.
- **M5 — Partially observable planning [later]**: POMCP/POMCGS with a particle belief and progressive
  widening over the existing sensors, benchmarked against the fully-observable baseline.
- **M6 — Teacher–student distillation [speculative]**: supervise a partially-observable student from
  the planner, with evaluation criteria against teacher and from-scratch RL; depends on M4 and M5.

## Deferred decisions

- **PBRS**: none is implemented, and the current MCTS must not be described as PBRS. It uses a fixed
  objective plus a separate leaf-value heuristic, not an added $\gamma\,\Phi(s') - \Phi(s)$ reward term.
  Before adopting shaping: keep the unshaped task reward primary, make shaping optional and
  configurable, log shaped and unshaped returns separately, check for double-counting against the
  distance-based leaf heuristic, and define potentials under partial observability. Decision needed:
  adopt PBRS at all, and under which objective.
- **Planning vs RL objective**: planning uses `+10`/`-5`/`-0.1`; RL uses a dense high-water-mark
  progress reward. A current implementation fact, not a settled design — whether evaluation should
  share one objective is unresolved.

Historical note: MCTS was originally committed as "PBRS MCTS" but contains no shaping term.
