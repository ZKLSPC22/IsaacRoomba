# IsaacRoomba — Project Context & Coding Rules

This file is the single source of truth for how this codebase is structured and how it
should be extended. VS Code / GitHub Copilot reads it automatically, so you do **not**
need to restate any of these conventions in prompts. Keep this file up to date when the
architecture changes.

---

## 1. Project Purpose

A **lightweight, general-purpose Roomba (differential-drive) navigation framework** built
on top of **NVIDIA Isaac Gym** (PhysX GPU pipeline). It is a research harness for comparing
POMDP solvers against standard RL baselines in partially-observable point-to-point
navigation.

Core research goals, in priority order:

1. Implement **POMCP / POMCGS**-style belief-space solvers (particle filters + graph
   search) for POMDP navigation.
2. Compare them against **standard RL baselines** (PPO/SAC-style policies).
3. Support **teacher-student** policy distillation (a fully-observable "teacher" planner
   supervising a partially-observable "student" policy).

The framework is intentionally minimal: one physics core, one generative transition
model, and a pluggable planner layer.

---

## 2. Tech Stack

- **Language:** Python 3.8.x
- **Physics:** Isaac Gym (`isaacgym.gymapi`, `isaacgym.gymtorch`) with GPU pipeline
- **Tensors:** PyTorch (everything on `env.device`, `float32`)
- **Spaces:** OpenAI `gym.spaces`
- **Config:** PyYAML (`configs/*.yaml`)
- **Visualization:** Isaac Gym viewer + optional OpenCV (`cv2`) for camera debug

Do **not** introduce new heavyweight dependencies without justification; prefer PyTorch
and the existing Isaac Gym APIs.

---

## 3. Architecture / Directory Layout

```
roomba.urdf                 # Robot model (chassis, wheels, sensor link)
configs/
  config.yaml               # Env / room / sensors / robot config (single source)
  planners.yaml             # base + mcts + pomcgs hyperparameters
core/
  room.py                   # Room, Obstacle, OccupancyMap (collision-free sampling)
  simulator.py              # RoombaSimulator: Isaac Gym setup + tensor buffers
  visualizer.py             # DebugVisualizer: viewer line drawing
envs/
  planning_env.py           # RoombaPlanningEnv: batched generative model G(s,a)
  rl_env.py                 # RoombaRLEnv: reset/step loop for RL training
planners/
  base.py                   # BaseNode / BaseSearcher (search skeleton, heuristic leaf eval)
  mcts.py                   # MCTSSolver: fully-observable tabular MCTS
scripts/
  run_mcts.py               # End-to-end MCTS loop with CSV logging
  random_rl.py              # Random-policy RL smoke test
  debug_rl.py               # Interactive manual control debugger
logs/
  mcts/                     # Timestamped CSV run logs
```

**Layer boundaries (respect these strictly):**

- `core/` = physics, geometry, and hardware-tensor management. No planning logic.
- `envs/` = state/observation/action spaces, reward computation, and the generative
  transition interface. Planners call envs, never touch `gym` directly.
- `planners/` = pure search algorithms. They consume envs through the generative model
  interface only.
- `scripts/` = thin entry points. No reusable logic should live here.

---

## 4. Fixed Conventions You Must Preserve

These are invariants. Do not silently change them; if a change is required, update this
file and every dependent site together.

### 4.1 State Representation (15D)

The explicit state tensor is exactly **15 elements**, `float32`, on `env.device`:

```
[ x, y, z, qx, qy, qz, qw, vx, vy, vz, wx, wy, wz, goal_x, goal_z ]
```

- Indices `0:13` = Isaac Gym root state (position, quaternion, linear/angular velocity).
- Index `13` = `goal_x`, index `14` = `goal_z` (world coordinates).
- `qw` is index **6** (neutral rotation = `1.0`); `qy` is index **4**.

Never reorder, insert, or remove fields without updating every consumer (envs, planners,
scripts). Access goal positions via indices `13`/`14` rather than magic "last two" logic.

### 4.2 Action Space

Normalized continuous action: `[linear_velocity, angular_velocity]` in `[-1, 1]`.

- Scaled inside the env: `v = action[0] * max_linear_velocity`,
  `omega = action[1] * max_angular_velocity`.
- Differential-drive mapping (already implemented in `generate`):
  ```python
  L, R = 0.235, 0.036
  v_left  = -(v - (omega * L) / 2.0) / R
  v_right = -(v + (omega * L) / 2.0) / R
  ```

### 4.3 Generative Model Interface

`RoombaPlanningEnv.generate(states, actions) -> (next_states, obs, rewards, dones)`

This is the **only** transition function planners may use. It teleports to `states`,
applies `actions` as a 0.5s macro-action, advances physics, and returns the outcome.
Planners must **not** call `sim.step_physics` or raw `gym` calls directly.

### 4.4 Coordinate System

- **Y-up** (`gymapi.UP_AXIS_Y`), **X = forward**, **Z = right**.
- Robot forward in URDF is `+X`; sensor link `lidar_link` sits on top of the chassis.

### 4.5 Batching / GPU Vectorization

- All environment interactions are **batched across `num_envs`** on the GPU.
- Never write Python `for` loops over individual environments in a hot path
  (selection/expansion/rollout); vectorize with torch.
- `num_planning_envs` must be `>=` the number of discrete actions so the action grid can
  be injected into a batch with zero-padding (see `MCTSSolver._expand`).
- CPU is acceptable only for one-shot setup (e.g. `sample_valid_pose`, occupancy-map
  construction).

---

## 5. Coding Rules

### 5.1 Config-Driven Design

- **Never hardcode hyperparameters.** Read them from YAML with a sensible fallback via
  `.get(key, default)`, matching the existing pattern in `BaseSearcher.__init__` and
  `MCTSSolver`.
- New planner knobs go in `configs/planners.yaml` under the planner's own section
  (`mcts:`, `pomcgs:`, etc.). New env/room/sensor knobs go in `configs/config.yaml`.

### 5.2 Extending the Planner Layer

All search algorithms follow the **template-method** pattern in `planners/base.py`:

- Subclass `BaseSearcher` and implement the abstract methods:
  - `_select(node)` — descend the tree/graph to a leaf.
  - `_expand(node)` — generate successor nodes (batch with the generative model) and
    compute/store each child's `heuristic_value` via a batched heuristic.
  - `_extract_physical_state(node)` — return a representative state for a belief node
    (used by POMCGS `_expand` to build the batched heuristic input).
- Subclass `BaseNode` to add planner-specific state (e.g. `MCTSNode.state`, and later a
  `POMCGSNode` carrying a belief/particle set).
- The shared Bellman backpropagation and iteration loop in `BaseSearcher` should be
  reused — do not reimplement them per planner.

When implementing **POMCGS**: the node carries a belief (particles), `_extract_physical_state`
returns a sampled/representative state, and `_expand` must respect progressive widening
(`k_a/alpha_a` for actions, `k_o/alpha_o` for observations) already stubbed in
`configs/planners.yaml`.

### 5.3 Rewards Live in the Env Layer

Reward shaping belongs in `envs/*_env.py` (`_compute_rewards`), not in planners. Keep the
planner reward-agnostic; planners only consume the scalar reward returned by `generate`.

### 5.4 Logging

- Write run metrics to **CSV** under `logs/<algo>/` with a UTC timestamped filename
  (`%Y%m%d_%H%M%S`), e.g. `logs/mcts/mcts_run_20260829_174252.csv`.
- Flush the file after each row so `KeyboardInterrupt` does not lose data (see
  `scripts/run_mcts.py`).
- Never commit generated logs; `logs/` is already git-ignored.

### 5.5 Style & Hygiene

- Type-annotate tensor shapes in docstrings (e.g. `-> [num_envs, 15]`).
- Keep `torch` tensors on `env.device` / `self.device`; only move to CPU for
  visualization or CSV writes.
- Prefer `torch.cat`, slicing, and vectorized ops over Python loops.
- Scripts that import package modules prepend the project root to `sys.path`
  (existing convention in `scripts/*`).
- Do not commit dead code, debug prints left in library code, or unused imports.
- Keep `__init__.py` files importable (they may be empty).

---

## 6. How to Run

```bash
# Fully-observable MCTS planner (needs CUDA + Isaac Gym installed)
python scripts/run_mcts.py

# Random-policy RL smoke test (viewer on)
python scripts/random_rl.py

# Interactive manual control debugger
python scripts/debug_rl.py
```

Notes:

- `scripts/run_mcts.py` assumes `num_planning_envs` in `configs/config.yaml` is at least
  as large as the discrete action grid in `configs/planners.yaml` (default grid is
  `3 x 5 = 15` actions, so set `num_planning_envs >= 15`; the script comment suggests 64).
- Requires Isaac Gym Preview + a CUDA-capable GPU; falls back to CPU if CUDA is
  unavailable.

---

## 7. Summary for Agents

When asked to implement or change anything, always:

1. Respect the 15D state layout, the normalized `[v, omega]` action, and Y-up coordinates.
2. Read hyperparameters from YAML instead of hardcoding.
3. Add planners via `BaseSearcher` subclasses, envs via the `generate()` interface, and
   rewards only in the env layer.
4. Keep hot paths GPU-batched and vectorized.
5. Log results as timestamped CSVs under `logs/<algo>/`.
