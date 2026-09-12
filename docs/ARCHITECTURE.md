# IsaacRoomba Architecture

Runtime behavior wins over source comments; stale comments are listed in §8. Related:
[`../README.md`](../README.md) (how to run), [`../ROADMAP.md`](../ROADMAP.md) (planned work),
[`../AGENTS.md`](../AGENTS.md) (agent rules, condensed invariants).

## 1. Layers and flow

| Layer | Owns | Must not contain |
| --- | --- | --- |
| `core/` | Isaac Gym setup, URDF, room, occupancy map, GPU buffers, viewer | planning/search |
| `envs/` | spaces, rewards, termination, observations, `generate()` | search algorithms |
| `envs/planning_math.py` | pure reward/heuristic math (`torch` only) | Isaac Gym/Gym/simulator/config imports |
| `planners/` | search, node bookkeeping, backpropagation | `isaacgym`/`gym`, physics stepping, rewards, termination |
| `scripts/` | driver loops, logging calls | reusable library logic |
| `tracking/` | `run.yaml`/`steps.csv`/TensorBoard | planning/simulator logic |
| `configs/` | parameters only | code |
| `tests/` | CPU-only unit tests | anything needing a GPU |

Flow: `configs/*.yaml` -> `RoombaSimulator` (Isaac Gym, room, occupancy map, N envs, sensors) ->
`RoombaPlanningEnv` (spaces, goals) -> `MCTSSolver` (15-action grid) -> `run_mcts.py` per step: fresh
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
`room.start_goal_sampling.min_distance`. Permissive fallbacks: custom `width`/`depth` -> 20.0,
`obstacles` -> `[]`, missing `type` -> `custom`. `OccupancyMap` rasterizes, dilates, and samples valid
poses on the CPU. `_calculate_dynamic_spacing()` sets `env_spacing = max(width, depth) + 2.0` (raises
above 50.0) and **overwrites** `config['env']['env_spacing']`, so the YAML value is dead.

**Frames:** `UP_AXIS_Y`, gravity `(0, -9.81, 0)` — Y up, the robot drives in X–Z, +X forward (front
marker x = 0.175), +Z right (chassis cylinder r = 0.17, wheels r = 0.036 at z = ±0.1175, a 0.235 m
baseline = `L`). World positions are raw `root_states`; **room-local** subtracts `sim.env_origins`
(`get_states` subtracts indices 0–2, `set_states` adds them back, velocity indices untouched).
`sample_valid_pose()` returns room-local coordinates — its "world coordinate" docstring is stale.

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

## 3. Generative model, timing, sensors, batching

`env.generate(states, actions, compute_observation=True)` -> `(next_states, obs, rewards, dones)`:
(1) `set_states` splits `states[:, :13]`/`states[:, 13:15]`, adds `env_origins` to indices 0–2, writes
with `set_actor_root_state_tensor_indexed` (**a teleport**), then `reset_dof_states()` and
`sync_graphics()`; (2) clamp/scale actions and apply wheel velocities; (3) step physics
`macro_action_ticks` times, OR-ing bumper contacts; (4) refresh the root state and set
`dones = room_local_dist < goal_radius`; (5) return `get_states()`, `_compute_rewards(...)`,
optionally observations, and draw debug visuals.

**Markov requirement:** wheel DOF positions/velocities are not in the 15D state, so
`reset_dof_states()` must keep running — otherwise the same `(s, a)` yields different successors
depending on what the slot previously ran — and robots must spawn at resting height (`y = 0.065`)
rather than being dropped. `compute_observation=False` skips observation construction. The batch is a
**fixed size** (`env.num_planning_envs`): every call must supply exactly that many rows.

**Timing:** physics 30 Hz (`dt = 1/30`); `planning.macro_action_ticks: 15`, so one `generate()` = 0.5 s
= one `run_mcts.py` control step. `RoombaRLEnv.step` advances one tick (1/30 s). The code default is 30
ticks when `planning` is absent; the shipped config sets 15.

**Sensors** (`spaces.Dict`, per the `sensors` toggles): bumper `(1,)` (chassis contact-force norm >
0.1, OR-ed over the macro-action), lidar `(64,)` (1D depth camera, 120° FOV, radial distance,
flipped), camera `(64, 64, 3)` (RGB facing +X). Shipped: bumper only, graphics off
(`graphics_device = -1`); no planner consumes observations.

**Batching:** `physx.use_gpu` and `use_gpu_pipeline` are always on, and all interaction is batched
across `num_envs` on `env.device` with no per-env Python loops in the planner hot path. `_expand`
makes one batched `generate()` call per node, writing the action grid into the leading `[:num_actions]`
rows of a zero `[num_envs, 2]` tensor — **padded slots still advance physics** (waste =
`num_envs - num_actions`; shipped 15 actions, 16 envs). The occupancy map is the sanctioned CPU
exception.

## 4. Heuristic and MCTS

`envs/planning_math.py` (imports `torch` only, so it is CPU-testable) holds the reward and leaf-value
formulas; `_compute_rewards` and `compute_heuristic_values` are thin adapters that inject env state.
Task constants are owned by `task:` in `configs/config.yaml` (`goal_radius: 0.5`, `goal_reward: 10.0`,
`collision_penalty: -5.0`, `step_cost: -0.1`); `planning_math` defines no constants or defaults — all
are required keyword arguments.

```
d_remain      = max(sqrt(dx^2 + dz^2) - goal_radius, 0)      # dx, dz = indices 13,14 minus 0,2
max_dist_step = max_linear_velocity * macro_action_duration  # 0.6 * 0.5 = 0.3 m
H             = max(d_remain / max_dist_step, 1.0)
V(s)          = gamma^(H-1) * goal_reward + step_cost * (1 - gamma^H) / (1 - gamma)  # 0 if terminal
```

`gamma == 1` gives `goal_reward + H * step_cost`. The heuristic is a **batched leaf value**, never a
rollout: a value estimate, not reward shaping (PBRS would need an added
$\gamma\,\Phi(s') - \Phi(s)$ reward term, which does not exist). It ignores collision risk, and shares
`goal_radius` with the terminal check.

**Parameters** (`configs/planners.yaml` `base.*`): `c_param` 1.414 (UCB1
`Q/N + c_param*sqrt(ln(parent.N)/N)`, unvisited `+inf`), `num_iterations` 100, `max_expansions` 300,
`gamma` 0.95. `MCTSSolver` builds `num_actions = 3 (mcts.v_vals) x 5 (mcts.omega_vals) = 15` and
validates `num_actions > 0`, `num_actions <= num_envs`, `num_iterations >= 1`,
`max_expansions >= num_actions`, `0 <= gamma <= 1`.

**`search`**: return `None` for a terminal root; expand the root (all 15 children with rewards,
terminal flags, heuristic values); loop while `iteration < num_iterations` and
`num_expansions < max_expansions`, doing UCB1 descent, expanding only when the node is non-terminal
**and** `N > 0`, then backpropagating `value = node.reward + gamma * value`, `N += 1`, `Q += value` to
the root (the root itself adds no reward step); return the best root child by `Q/N` if `N > 0`, else
`reward + gamma * heuristic_value`, or `None` when there are no children. `_expand` returns a
**randomly chosen child** for the next phase, so `Q/N` is not an unbiased action-value.

Semantics gaps: both counters gate the same loop in unrelated units (one expansion = 15 children; one
iteration = one selection + one backprop), so either can end the search; expansion requiring `N > 0`
means iterations can be spent re-backpropagating a visited leaf; there is no time-, tick-, or call-based
budget. **No tree reuse** — `run_mcts.py` builds a fresh root inside the execution loop and discards
the tree, keeping only `count_nodes`/`tree_depth`.

## 5. Rewards and termination

Both live in `envs/`, using room-local `dist_to_goal` from the post-physics root state.

```
planning:  reached = dist < goal_radius
           reward  = reached*goal_reward + bumped*collision_penalty + step_cost   # +10 / -5 / -0.1
           done    = reached                                    # no timeout in the env
rl:        progress = max(min_dist - dist, 0); min_dist = min(min_dist, dist)    # high-water mark
           reward   = progress*5.0 + reached*10.0 - bumped*5.0
           done     = reached | (progress_buf >= 3600)          # 3600 hard-coded
```

`bumped` is the OR-accumulated bumper mask (all-`False` when disabled) and the planning step cost is
unconditional, including on the terminal step; planning episodes are bounded by `run_mcts.py`
(`max_execution_steps` plus no-progress detection). In RL, `min_dist_to_goal` starts at the reset-time
start→goal distance (so a round trip pays nothing extra), `_compute_rewards` ignores its `actions`
argument, and `step()` resets finished envs inline at one tick per step. The two environments share
robot, room, and simulator but have independent rewards, termination, and control rates.

## 6. Configuration and logging

| File | Owns |
| --- | --- |
| `configs/config.yaml` | `env.num_rl_envs` (1), `env.num_planning_envs` (16), dead `env.env_spacing`, `planning.macro_action_ticks`, `room.*`, `task.*`, `sensors.*`, `robot.*` |
| `configs/planners.yaml` | `base.*`, `mcts.*`, unused `pomcgs.*` |
| `configs/experiments.yaml` | `mcts.seed`, `mcts.max_execution_steps`, `mcts.no_progress.*`, `mcts.logging.*` |

Paths are relative and `robot.urdf_path` is `"roomba.urdf"` loaded via `load_asset(self.sim, ".")`, so
**scripts must run from the repository root**.

`tracking/run_logger.py` writes `logs/mcts/mcts_<UTC>/` containing `run.yaml` (`schema_version: 1`:
resolved config, Git commit + dirty flag, scenario, result; written `running` first, finalized on every
exit path, atomic replace), `steps.csv` (one flushed row per executed action; column names carry units:
`step`, `action_v_normalized`, `action_w_normalized`, `root_value` = `root.Q/root.N`, `tree_size`,
`max_depth`, `distance_to_goal_m`, `executed_reward`, `search_time_sec`, `execution_time_sec`,
`expansion_calls`, `physics_ticks`, `decisions_per_sec`), and TensorBoard derived from the same metrics
dict (`TensorBoardUnavailableError` if enabled without the package). Termination reasons:
`goal_reached`, `no_progress`, `execution_horizon`, `interrupted`, `error`. `logs/` is git-ignored but
13 older CSVs remain tracked; `seed_everything(seed)` seeds NumPy and Torch.

## 7. Invariants and known limitations

Invariants (condensed in [`../AGENTS.md`](../AGENTS.md); keep both in sync): keep the 15D layout and the
`get_states`/`set_states` origin symmetry; keep actions normalized `[-1, 1]²`; keep the Y-up,
X-forward, Z-right frame; planners reach the world only through `generate()` and
`compute_heuristic_values`; rewards and termination stay in `envs/`; hot paths stay GPU-batched;
teleports keep resetting wheel DOF state and place robots at resting height;
`num_actions <= num_planning_envs`; macro-action duration is `macro_action_ticks * (1/30)` s and is
shared with the heuristic.

Limitations: no tree reuse; count-based budgets only; fully observable only; padded slots waste
physics; `RoombaRLEnv` duplicates reward constants instead of reading `task.*`; dead configuration
(`env.env_spacing`, `pomcgs.*`); `_extract_physical_state` is never called; mixed configuration
strictness; CPU occupancy-map serialization; an extra GPU sync per transition; `search()` can return
`None` while `run_mcts.py` indexes `mcts.actions[best_action_idx]` unguarded (only the initial
already-at-goal case is guarded); dead viewer-close checks in the RL scripts; a stray `import torch` in
`random_rl.py`; the untracked, currently red `tests/`.

Open questions (tracked in [`../ROADMAP.md`](../ROADMAP.md)): time/tick-based budgets; tree reuse and
`N`/`Q` correction on re-rooting; one shared objective for planning and RL; the POMCGS belief
representation.

## 8. Stale code text to fix

| Location | Actual behavior |
| --- | --- |
| `envs/planning_env.py`, `generate()` step 3 | says "default 30 ticks = 0.5s at 60Hz"; physics is 30 Hz and the configured macro-action is 15 ticks |
| `core/room.py`, `sample_valid_pose` docstring | says "world coordinate"; returns room-local |
| `core/room.py`, `sample_valid_start_goal` error | names `min_start_goal_dist`; key is `room.start_goal_sampling.min_distance` |
| `envs/rl_env.py`, `_compute_dones` comment | says the max steps should be configured; `3600` is hard-coded |
| `core/room.py`, `BoxObstacle.spawn` actor name | typo `"box_obstaclconfige"`; cosmetic |
