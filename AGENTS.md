# AGENTS.md

Rules for coding agents. [`README.md`](README.md) = how to run; [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
= technical truth (read before touching interfaces, state, frames, simulator behavior, or config);
[`ROADMAP.md`](ROADMAP.md) = planned work (never implement milestones unasked or present planned work as
existing).

## Workflow

- Check `git status`/diff first and preserve uncommitted user work; never revert or reformat code you
  were not asked to touch. No drive-by refactors, renames, or formatting churn.
- Never silently change semantics: if a change alters what the planner optimizes, how values back up,
  or when an episode terminates, say so explicitly.
- Update `README.md` / `docs/ARCHITECTURE.md` with any behavior change — stale docs are a defect.
- Report what you ran, what you could not, and why; label unverified behavior honestly.

## Hard constraints

- **15D state** `[x,y,z,qx,qy,qz,qw,vx,vy,vz,wx,wy,wz,goal_x,goal_z]` (`qw` at index 6, room-local
  positions/goals); **actions** are normalized `[linear_velocity, angular_velocity]` in `[-1,1]²`.
  Changing either means updating every consumer *and* the docs.
- Planners use only `generate(...)`, `compute_heuristic_values(...)`, and exposed spaces/attributes —
  never `sim.step_physics` or `gym`. Rewards and termination stay in `envs/`.
- Layers: `core/` physics/geometry · `envs/` spaces, rewards, termination, `generate` · `planners/`
  search · `scripts/` entry points · `configs/` parameters (ARCHITECTURE §1).
- Hot paths stay GPU-batched (no per-env Python loops); the CPU occupancy map is the only exception.
- `G(s, a)` stays Markov: teleports reset wheel DOF state; robots spawn at resting height.
- `num_actions <= num_planning_envs` (enforced in `MCTSSolver`).
- `steps.csv` carries the planar pose (`robot_x`, `robot_z`, `robot_yaw`, immediately after `step`),
  written by `scripts/run_mcts.py` from the logged state. `STEP_COLUMNS` has three consumers: changing it
  means changing `tracking/run_logger.py`, `scripts/run_mcts.py`, and `tests/test_run_logger.py`
  together. PNG frames are a derived artifact — never make one authoritative, and never let a render
  failure change or truncate a logged record (ARCHITECTURE §6).

## Config

- Owners: `configs/config.yaml` (env/simulation/room/task/sensors/robot), `configs/planners.yaml`,
  `configs/experiments.yaml`. No hard-coded tunables where an owner exists; no duplicated or dead keys.
- Required keys are direct-indexed (fail fast): `room.occupancy_map.*`,
  `room.start_goal_sampling.min_distance`, `task.{goal_radius,goal_reward,collision_penalty,step_cost}`,
  `simulation.{num_position_iterations,num_velocity_iterations}`. No silent fallbacks; no task
  constants in `envs/planning_math.py`.

## Verification and writing

- Prefer GPU-free checks; do not launch GPU experiments unless required, and never to validate docs
  (if one is needed, `scripts/run_mcts.py` with the viewer off, stopped early). `tests/` is tracked and
  CPU-only: `python -m unittest discover -s tests` is the pre-experiment gate, currently 96 tests
  (`test_distance_field` 4, `test_mcts` 26, `test_planning_math` 31, `test_run_logger` 24,
  `test_spatial_plotter` 11), and no test may need Isaac Gym, CUDA, or a GPU. Run it before claiming a
  suite result; the CPU-only modules `tests/test_spatial_plotter.py` (visualizer + offline replay) and
  `tests/test_run_logger.py` (log schema) are the ones that guard the logging path. A test that needs a
  GPU does not belong here.
- Prefer existing dependencies; never invent versions or install paths — Isaac Gym, CUDA, and PyTorch
  are unpinned external prerequisites.
- Concise, implemented-vs-planned, relative links, no invented numbers. Architecture detail in
  `docs/ARCHITECTURE.md`, planning in `ROADMAP.md`, no duplication.
