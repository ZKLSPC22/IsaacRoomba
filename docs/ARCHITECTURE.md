# IsaacRoomba Architecture

Runtime behavior wins over source comments. Related:
[`../README.md`](../README.md) (how to run), [`../ROADMAP.md`](../ROADMAP.md) (planned work),
[`../AGENTS.md`](../AGENTS.md) (agent rules, condensed invariants). This file is the only
project-authored doc in `docs/`: `SimulationSetup.md` and `TensorAPI.md` are verbatim copies of the
Isaac Gym documentation (a third-party mirror) describing the generic API, not this repository, and they
are not edited here; `COMPLETED_TASKS.md` is finished-work history, not current behavior.

## 1. Layers and flow

| Layer | Owns | Must not contain |
| --- | --- | --- |
| `core/` | Isaac Gym setup, URDF, room, occupancy map, GPU buffers, viewer | planning/search |
| `envs/` | spaces, rewards, termination, observations, `generate()` | search algorithms |
| `envs/planning_math.py` | pure reward/heuristic math (`torch` only) | Isaac Gym/Gym/simulator/config imports |
| `planners/` | search, node bookkeeping, backpropagation | `isaacgym`/`gym`, physics stepping, rewards, termination |
| `scripts/` | driver loops, logging calls | reusable library logic |
| `tracking/` | `run.yaml`/`steps.csv`/TensorBoard | planning/simulator logic |
| `tracking/spatial_plotter.py` | headless 2D Matplotlib trajectory visualizer and flipbook frame generator | `isaacgym`, `torch`, simulator imports |
| `configs/` | parameters only | code |
| `tests/` | CPU-only unit tests | anything needing a GPU |

Flow: `configs/*.yaml` -> `RoombaSimulator` (Isaac Gym, room, occupancy map, N envs, sensors) ->
`RoombaPlanningEnv` (spaces, goals) -> `MCTSSolver` (explicit `mcts.actions` grid) -> `run_mcts.py` per step: fresh
root -> `search` -> execute the best action via `generate()` -> `current_state = next_states[0]` ->
CSV row, stopping on done / no-progress / horizon.

`planners/mcts.py` imports only `torch` and `planners.base`. `BaseSearcher` owns the loop, UCB1, and
backpropagation with `_select`, `_expand`, and `_extract_physical_state` abstract; `MCTSSolver`
implements them. `core/__init__.py` and `envs/__init__.py` are empty; there is no
`planners/__init__.py`, no dependency manifest, and no packaging metadata (stdlib `unittest`).

## 2. Room, frames, state, action

**Room** (`core/room.py`): presets `empty()` (10x10) and `standard()` (20x20 with a 2x2 pillar and an
8x1 wall); `room.type == "custom"` reads `custom.width`/`depth`/`obstacles`, anything else raises
`ValueError`. Only `type: box` obstacles exist (others silently skipped). Required keys, direct-indexed
so typos fail fast: `room.occupancy_map.resolution`, `room.occupancy_map.safety_margin`,
`room.start_goal_sampling.min_distance`, `env.env_spacing`. Permissive fallbacks: `obstacles` -> `[]`,
missing `type` -> `custom`; a custom room missing `width` or `depth` raises instead of defaulting.
`OccupancyMap` rasterizes, dilates, samples valid poses, and precomputes the 8-connected free-space
graph used for geodesic distance queries on the CPU. `_calculate_env_pitch()` sets
the per-environment grid pitch to `max(width, depth) + env.env_spacing` (shipped buffer 2.0 m) and
raises `ValueError` above 50.0; `env.env_spacing` is read from the config, never overwritten. The grid
is `envs_per_row = int(sqrt(num_envs))` — 3x3 for the shipped 9.

**Frames:** `UP_AXIS_Y`, gravity `(0, -9.81, 0)` — Y up, the robot drives in X–Z, +X forward (front
marker x = 0.175), +Z right (chassis cylinder r = 0.17, wheels r = 0.036 at z = ±0.1175, a 0.235 m
baseline = `L`). Positions and goals in the 15D state are **room-local**, and the env tensors hold
room-local values directly: `get_states` returns `root_states[robot_actor_indices]` unshifted and
`set_states` writes `states[:, :13]` back into the same rows, so no origin offset is applied in either
direction. Each `create_env` grid slot is still a distinct world position — the per-env offset is the
Isaac Gym environment origin, which is why every env index maps 1:1 onto its own slot.
`sample_valid_pose()` returns room-local coordinates.

**State** (`Box(shape=(15,))`, float32 on `env.device`): 0–2 `x,y,z` (room-local metres, `y` rests at
0.065) · 3–6 `qx,qy,qz,qw` (**`qw` at index 6**) · 7–9 `vx,vy,vz` · 10–12 `wx,wy,wz` · 13–14
`goal_x,goal_z` (room-local, per-env `[num_envs, 2]`). Indices 0–12 mirror the Isaac Gym root state;
wheel DOF state is not part of the state.

**Action**: normalized `[linear_velocity, angular_velocity]` in `[-1, 1]²` (`Box(-1, 1, shape=(2,))`),
converted identically in `generate()` and `RoombaRLEnv.step`:

```python
clamped = torch.clamp(actions, -1.0, 1.0)
v     = clamped[:, 0] * max_linear_velocity     # 0.6 m/s
omega = clamped[:, 1] * max_angular_velocity    # 2.0 rad/s
L, R = 0.235, 0.036                             # track baseline, wheel radius
v_left, v_right = -(v - omega * L / 2) / R, -(v + omega * L / 2) / R
```

The negation encodes this URDF's wheel spin-axis convention. Targets go into a
`[num_envs, dofs_per_actor]` view via `set_dof_velocity_target_tensor` (wheel DOFs `DOF_MODE_VEL`,
stiffness 0.0, damping 1000.0).

**Contacts:** `_load_robot_asset()` zeroes the `friction`, `rolling_friction` and `torsion_friction` of
the `front_caster`/`rear_caster` collision shapes (`RoombaSimulator.CASTER_LINKS`), located through
`get_asset_rigid_body_shape_indices()`. Isaac Gym ignores the URDF's `<contact_coefficients>`, so
without this every shape inherits the PhysX default (mu = 1) and the fixed casters stick instead of
sliding, which makes achieved wheel speeds direction-dependent (chassis and drive wheels keep the
default).

## 3. Generative model, timing, sensors, batching

`env.generate(states, actions, compute_observation=True)` -> `(next_states, obs, rewards, dones)`:
(1) `set_states` splits `states[:, :13]`/`states[:, 13:15]`, writes indices 0–2 and the rest of the root
state into `root_states[robot_actor_indices]` in room-local coordinates, and commits them with
`set_actor_root_state_tensor_indexed` (**a teleport**), then `reset_dof_states()` (only when
`task.wheels_reset` is true) and `sync_graphics()`; (2) clamp/scale actions and apply wheel velocities;
(3) step physics `macro_action_ticks` times, OR-ing bumper contacts; (4) refresh the root state and set
`dones = room_local_dist < goal_radius`; (5) return `get_states()`, `_compute_rewards(...)`,
optionally observations, and draw debug visuals.

**Markov requirement:** wheel DOF positions/velocities are not in the 15D state, so with
`task.wheels_reset: true` (shipped) `reset_dof_states()` must keep running — otherwise the same `(s, a)`
yields different successors depending on what the slot previously ran. Turning it off makes the
transition history-dependent; it is an experimental toggle, not an equivalent choice. Robots must also
spawn at resting height (`y = 0.065`) rather than being dropped.
`compute_observation=False` skips observation construction. The batch is a
**fixed size** (`env.num_planning_envs`): every call must supply exactly that many rows.

**Timing:** physics 30 Hz (`dt = 1/30`); `planning.macro_action_ticks: 15`, so one `generate()` = 0.5 s
= one `run_mcts.py` control step. `RoombaRLEnv.step` advances one tick (1/30 s). The code default is 30
ticks when `planning` is absent; the shipped config sets 15.

**Sensors** (`spaces.Dict`, per the `sensors` toggles): bumper `(1,)` (**horizontal** chassis
contact-force norm > `task.bumped_threshold`, OR-ed over the macro-action), lidar `(64,)` (1D depth
camera, 120° FOV, radial distance, flipped), camera `(64, 64, 3)` (RGB facing +X). Shipped: bumper only,
graphics off (`graphics_device = -1`); no planner consumes observations.

The bumper tests **world X and Z only**; the vertical component is discarded. The chassis rests on the
ground, so `f_y` carries the ground reaction and the landing impulse from the spawn-height drop: measured
at the shipped spawn in open space it reaches 14.8 N while standing still, i.e. it exceeded every
practical threshold and fired the bumper on essentially every macro-action regardless of collisions
(recorded as a constant `-5.1` reward per step). Isolating the horizontal components leaves a
free-space noise floor of 2e-6 N (567 samples over 81 free poses x 7 actions) against 12.5-19.0 N for a
genuine wall contact, so the shipped threshold of 2.0 N sits about six orders of magnitude above the
noise and 6x below the weakest measured contact. Note this is a *velocity-dependent* contact force: a
robot resting against a wall without pushing reads zero. The test itself lives in
`envs/planning_math.py::compute_bumped`, so `RoombaRLEnv` and `RoombaPlanningEnv` cannot drift apart;
both pass `task.bumped_threshold` (the RL env previously hard-coded 0.1 N, making it 20x more sensitive
than the calibrated value below).

**Batching:** `physx.use_gpu` and `use_gpu_pipeline` are always on, and all interaction is batched
across `num_envs` on `env.device` with no per-env Python loops in the planner hot path. `_expand`
makes one batched `generate()` call per node, writing the action grid into the leading `[:num_actions]`
rows of a zero `[num_envs, 2]` tensor — **padded slots still advance physics** (waste =
`num_envs - num_actions`; shipped 5 actions over 9 envs, so 4 padding slots advance physics). The
occupancy map and its Dijkstra distance field are the sanctioned CPU exceptions, and the field is
recomputed only when the goal changes (once per distinct goal, never per leaf).

**Solver:** `_setup_simulator()` sets `physx.num_position_iterations` / `num_velocity_iterations` from
`simulation:` in `configs/config.yaml` (direct-indexed, so a missing key raises). Both flags also feed
the joint-drive solve, so raising them stiffens the wheel-velocity tracking that the differential-drive
model assumes; the shipped 16/8 sit above the PhysX defaults of 4/1.

## 4. Heuristic and MCTS

`envs/planning_math.py` (imports `torch` only, so it is CPU-testable) holds the reward and leaf-value
formulas; `_compute_rewards` and `compute_heuristic_values` are thin adapters that inject env state.
Task constants are owned by `task:` in `configs/config.yaml` (`goal_radius: 0.5`, `goal_reward: 10.0`,
`collision_penalty: -5.0`, `step_cost: -0.1`); `planning_math` defines no constants or defaults — all
are required keyword arguments.

**Distance source: the 2D geodesic distance field.** The remaining distance is sampled from a
precomputed obstacle-aware path-length field, not measured as a straight line, so a leaf behind a wall
is valued by the detour it must drive rather than by the chord to the goal.

`OccupancyMap` builds an 8-connected graph over free cells once in `__init__` (`scipy.sparse.csr_matrix`
over `c_space_grid == 0`, flat index `u = gx * depth_cells + gz`, orthogonal step `resolution`, diagonal
step `resolution * sqrt(2)`). A diagonal step is added only when **both** shared orthogonal neighbours
are free, which prevents cutting corners through inflated obstacle cells.
`OccupancyMap.compute_distance_field(goal_x, goal_z)` maps the goal to its cell (clamped), snaps to the
nearest free cell when the goal sits inside an inflated margin, runs
`scipy.sparse.csgraph.dijkstra(directed=False)`, and reshapes to a room-local
`(width_cells, depth_cells)` float32 array of path lengths **in metres**.

Cells with no graph node — the inflated obstacle margin and the boundary walls — are filled with the
distance of their **nearest free-and-reachable cell**, computed with a Euclidean distance transform
(`scipy.ndimage.distance_transform_edt(..., return_indices=True)`). Inheriting the neighbour's value keeps
the field continuous across the margin. Assigning a single large sentinel there instead (`max_finite +
2.0`, the previous behaviour) made the field *discontinuous and misleading*: in the shipped 5x5 m room the
margin covers 3540 of 10000 cells (35%), and every one of them read farther than the most distant
reachable cell (7.67 m vs a true maximum of 5.67 m). Since the heuristic is monotonically decreasing in
sampled distance, that inversion scored goal-ward motion into the margin as the worst value in the room —
measured along a goal-ward line from the shipped spawn, the value collapsed from 6.466 to 1.675 in 0.15 m
while the Euclidean equivalent rose 6.523 -> 6.745. With the fill, the same probe rises monotonically
(0 of 59 consecutive samples decrease).

`RoombaPlanningEnv._get_or_update_distance_field(goal_x, goal_z)` owns the GPU cache: the field depends
only on the goal (the room layout is static), so it is computed once per distinct goal — within `1e-4` —
and stored transposed as a `[1, 1, depth_cells, width_cells]` float32 tensor on `env.device`, with the
tensor height axis mapped to Z/depth and the width axis to X/width. `compute_heuristic_values` reads the
goal from `states[0, 13:15]` (valid because `_expand` replicates one parent state across the batch, so
the batch shares one goal) and samples the field with the interpolation named by
`task.heuristic_sampling`:

```
norm_x = clamp(x / (room_width / 2), -1, 1);  norm_z = clamp(z / (room_depth / 2), -1, 1)
grid   = stack([norm_x, norm_z], dim=-1)[None, None]         # [1, 1, batch, 2]
dist_geodesic = grid_sample(field, grid, mode=task.heuristic_sampling,
                            padding_mode="border", align_corners=True).view(-1)
```

`task.heuristic_sampling` is a required parameter with no default (`planning_math` keeps no fallbacks);
an unknown value raises `ValueError` naming the key. `"bilinear"` (shipped) blends the four neighbouring
cells and is the documented contract; `"nearest"` quantizes to the containing cell centre — up to half a
cell, 0.025 m at the shipped 0.05 m resolution — and is retained only to reproduce the pre-toggle
sampling exactly. `tests/test_planning_math.py::sample_geodesic_field` is the independent NumPy reference
for the bilinear convention and `ConfigContractTests` pins the shipped value.

`align_corners=True` maps normalized `-1`/`+1` onto the first/last **cell centres**, so the outer half
cell of each axis is reachable only by the `clamp`/`border` behaviour; that boundary approximation is the
cost of a stateless lookup and is well inside `goal_radius`.

```
d_remain      = max(dist_geodesic - goal_radius, 0)          # metres, clamped at the goal radius
max_dist_step = max_linear_velocity * macro_action_duration  # 0.6 * 0.5 = 0.3 m
H             = max(d_remain / max_dist_step, 1.0)
V(s)          = gamma^(H-1) * goal_reward + step_cost * (1 - gamma^H) / (1 - gamma)  # 0 if terminal
```

`gamma == 1` gives `goal_reward + H * step_cost`. The heuristic is a **batched leaf value**, never a
rollout: a value estimate, not reward shaping (PBRS would need an added
$\gamma\,\Phi(s') - \Phi(s)$ reward term, which does not exist). It accounts for obstacle detours through
the path length but still ignores collision **risk**, and shares `goal_radius` with the terminal check.
Omitting `distance_field`, `room_width` and `room_depth` together falls back to the earlier
`sqrt(dx^2 + dz^2)` straight-line distance (the function keeps its old name); no production caller uses
that path, only `tests/test_planning_math.py`.

**Parameters** (`configs/planners.yaml` `base.*`): `c_param` 1.414 (UCB1
`Q/N + c_param*sqrt(ln(parent.N)/N)`, unvisited `+inf`), `gamma` 0.95. The search budget
`base.num_iterations` is the tunable one and changes per experiment; read the shipped value from the
config rather than from this document.
`mcts.actions` is the explicit, ordered action set the solver loads directly — shipped 5
`[linear_velocity, angular_velocity]` pairs (`[1.0, 0.0]`, `[1.0, 1.0]`, `[1.0, -1.0]`, `[0.0, 1.0]`,
`[0.0, -1.0]`), with no stationary action. Index order is the action's identity in
`node.action_taken`, the run log, and the `search()` tie-break, so it is never re-sorted or
de-duplicated. There is no fallback grid: a missing key, a non-pair entry, a non-numeric component, or
a component outside `[-1, 1]` raises at construction rather than being clamped or truncated.
`MCTSSolver` validates `num_actions > 0`, `num_actions <= num_envs`, `num_iterations >= 1`,
`0 <= gamma <= 1`.

**`search`**: return `None` for a terminal root; expand the root (all `num_actions` children with rewards,
terminal flags, heuristic values); loop while `iteration < num_iterations`, doing UCB1 descent, expanding only when the node is non-terminal
**and** `N > 0`, then backpropagating `value = node.reward + gamma * value`, `N += 1`, `Q += value` to
the root (the root itself adds no reward step); return the best root child by `Q/N` if `N > 0`, else
`reward + gamma * heuristic_value`, or `None` when there are no children. `_expand` returns a
**randomly chosen child** for the next phase, so `Q/N` is not an unbiased action-value.

The loop is gated by `num_iterations` alone (one iteration = one selection plus one backprop, while one
expansion creates `num_actions` children); expansion requiring `N > 0` means iterations can be spent
re-backpropagating a visited leaf, and there is no time-, tick-, or call-based budget. **No tree reuse** — `run_mcts.py` builds a fresh root inside the execution loop and discards
the tree, keeping only `count_nodes`/`tree_depth`.

## 5. Rewards and termination

Both live in `envs/`, using room-local `dist_to_goal` from the post-physics root state.

```
planning:  reached = dist < goal_radius
           reward  = reached*goal_reward + bumped*collision_penalty + step_cost   # +10 / -5 / -0.1
           done    = reached                                    # no timeout in the env
rl:        progress = max(min_dist - dist, 0); min_dist = min(min_dist, dist)    # high-water mark
           reward   = progress*rl.progress_weight + reached*task.goal_reward
                      + bumped*task.collision_penalty   # collision_penalty is already negative
           done     = reached | (progress_buf >= rl.max_episode_steps)
```

`bumped` is the OR-accumulated bumper mask (all-`False` when disabled), built from the horizontal-only
contact force described in §3, and the planning step cost is unconditional, including on the terminal
step; planning episodes are bounded by `run_mcts.py` (`max_execution_steps` plus no-progress detection).
Because the bumper fired on nearly every step before the vertical component was discarded, `bumped` was
effectively always true, which made `collision_penalty` a constant offset that discarded the step's only
progress signal. In RL, `min_dist_to_goal` starts at the reset-time
start→goal distance (so a round trip pays nothing extra), `_compute_rewards` ignores its `actions`
argument, and `step()` resets finished envs inline at one tick per step. The two environments share
robot, room, and simulator but have independent rewards, termination, and control rates.

## 6. Configuration and logging

| File | Owns |
| --- | --- |
| `configs/config.yaml` | `env.num_rl_envs` (9), `env.num_planning_envs` (9), `env.env_spacing` (2.0 m grid buffer), `simulation.{num_position_iterations,num_velocity_iterations}`, `planning.macro_action_ticks`, `rl.max_episode_steps`, `rl.progress_weight`, `room.*`, `task.*`, `sensors.*`, `robot.*` |
| `configs/planners.yaml` | `base.*`, `mcts.actions` and `mcts.*`, unused `pomcgs.*` |
| `configs/experiments.yaml` | `mcts.seed`, `mcts.max_execution_steps`, `mcts.no_progress.*`, `mcts.logging.*`, and the parallel `debug_mcts.*` section for `debug_mcts.py` |

`mcts.logging` holds `output_dir`, `tensorboard`, and `render_frames`: the last is the on/off switch for
the PNG flipbook, read by `scripts/run_mcts.py` as `bool(logging_cfg.get("render_frames", False))` and
forwarded to `MCTSRunLogger(enable_visualizer=...)`, so a config that omits the key draws no frames.
The toggle controls *rendering only*: `tracking/run_logger.py` imports `TrajectoryVisualizer`
unconditionally, so importing `tracking` (and therefore running `scripts/run_mcts.py`) requires
matplotlib at import time whether or not frames are enabled.

Paths are relative and `robot.urdf_path` is `"roomba.urdf"` loaded via `load_asset(self.sim, ".")`, so
**scripts must run from the repository root**.

`tracking/run_logger.py` writes `logs/mcts/mcts_<UTC>/` containing `run.yaml` (`schema_version: 1`:
resolved config, Git commit + dirty flag, scenario, result; written `running` first, finalized on every
exit path, atomic replace), `steps.csv` (one flushed row per executed action; column names carry units
except for the planar pose: `step`, `robot_x`, `robot_z`, `robot_yaw` (metres and radians in the same
Y-up / X-forward / Z-right frame the renderer draws; `robot_yaw` is the planar angle whose
`(cos, sin)` is the robot's forward direction, so a `+Z` turn about world `+Y` yields a negative yaw),
`action_v_normalized`, `action_w_normalized`, `root_value` = `root.Q/root.N`, `tree_size`,
`max_depth`, `distance_to_goal_m`, `executed_reward`, `search_time_sec`, `execution_time_sec`,
`expansion_calls`, `physics_ticks`, `decisions_per_sec`), and TensorBoard derived from the same metrics
dict (`TensorBoardUnavailableError` if enabled without the package). Termination reasons:
`goal_reached`, `no_progress`, `execution_horizon`, `interrupted`, `error`. `logs/` is git-ignored but
13 older CSVs remain tracked; `seed_everything(seed)` seeds NumPy and Torch.

### Frame flipbook (`frames/step_XXX.png`)

When `render_frames` is true, each logged step additionally writes
`<run_dir>/frames/step_<step:03d>.png` (`step_000` is the pre-action spawn pose logged before the
execution loop). Rendering is derived and fully shielded: `log_step` writes the CSV row and TensorBoard
scalars first, then renders inside `try/except Exception`, so a plotting failure loses only the frame
and never a data record. CSV, TensorBoard, and frames all come from one metrics dict. An empty
`frames/` directory is still left behind if rendering fails or a replay is refused, since the directory
is created by the visualizer's constructor.

`tracking/spatial_plotter.py::TrajectoryVisualizer` composes each frame from four independent layers,
drawn in this order:

| Layer | Content | Notes |
| --- | --- | --- |
| 1. Static scene | room outline, obstacle bodies, dashed inflated obstacle footprints, goal disc + cross | geometry only; redrawn from constructor parameters |
| 2. Kinematics | breadcrumb trail (only with >= 2 poses), robot body, heading arrow | `cos(yaw)`/`sin(yaw)` arrow; trail accumulates in `history_x`/`history_z` |
| 3. Delegate overlay | caller-supplied `overlay_fn(ax)`, skipped when `None` | receives the live `Axes`; nothing algorithm-specific is assumed |
| 4. Diagnostic HUD | optional `hud_metrics` rendered verbatim as key/value lines | skipped when `None` or `{}`; values are stringified generically |

The module is importable and testable without Isaac Gym, CUDA, PyTorch, or matplotlib interaction: it
selects the `Agg` backend before importing `pyplot`, closes every figure in a `finally`, and duplicates
the `core.room` presets as constants (`EMPTY_ROOM_BOUNDS`/`STANDARD_ROOM_*`) rather than importing
`core.room`, which pulls in `isaacgym`. The `standard` preset wall is `x=5.0, z=5.0, width=8.0,
depth=1.0`, matching `Room.standard()`.

Offline replay needs no simulator: `TrajectoryVisualizer.from_run_log(run_dir)` reconstructs room
geometry (from `scenario.geometry` when recorded, else the preset name in
`configuration.environment.room.type`), radii, and goal from `run.yaml`, and `render_from_csv`
re-renders a `steps.csv` into frames, raising `ValueError` if the pose columns are absent.
`scripts/render_trajectory.py` exposes this as a CLI (`--gif`, `--fps`) writing `<run_dir>/trajectory.gif`.
CSVs logged before the pose columns existed cannot be replayed.

## 7. Invariants and known limitations

Invariants (condensed in [`../AGENTS.md`](../AGENTS.md); keep both in sync): keep the 15D layout
untouched, with room-local positions and goals in both directions of `get_states`/`set_states`; keep
actions normalized `[-1, 1]²`; keep the Y-up, X-forward, Z-right frame; planners reach the world only
through `generate()` and `compute_heuristic_values`; rewards and termination stay in `envs/`; hot paths
stay GPU-batched; with `task.wheels_reset: true` teleports keep resetting wheel DOF state, and robots
stay at resting height; `num_actions <= num_planning_envs`; macro-action duration is
`macro_action_ticks * (1/30)` s and is
shared with the heuristic; the cached distance field is built for one goal and normalized by the room
extents, so the field, `states[:, 13:15]`, and `sim.room` must always describe the same goal and room;
`STEP_COLUMNS` is one schema with four consumers, so changing it means changing `run_logger.py`,
`scripts/run_mcts.py`, `scripts/debug_mcts.py`, and `tests/test_run_logger.py` together, and frames stay
derived (never authoritative) so a rendering failure cannot change or truncate a logged record.

Limitations: no tree reuse; count-based budgets only; fully observable only; padded slots waste
physics (`num_envs - num_actions`); matplotlib is an undeclared import-time dependency of `tracking/`
even when `render_frames` is false, and every `steps.csv` written before the pose columns existed is
unreplayable; `pomcgs.*` is configured
but never read; `_extract_physical_state` is never called; mixed configuration strictness; CPU
occupancy-map serialization plus one CPU Dijkstra per distinct goal; an extra GPU sync per transition; dead viewer-close checks in the RL
scripts (`step()` never returns `False`); `tests/` is tracked and CPU-only.

Open questions (tracked in [`../ROADMAP.md`](../ROADMAP.md)): time/tick-based budgets; tree reuse and
`N`/`Q` correction on re-rooting; one shared objective for planning and RL; the POMCGS belief
representation.
