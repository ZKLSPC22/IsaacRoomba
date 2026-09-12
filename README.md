Isaac Gym (PhysX GPU) harness for point-to-point navigation of a differential-drive "Roomba": a
batched generative transition model `G(s, a)`, an explicit 15D state, a configurable room with a
collision-free occupancy map, and a fully-observable tabular MCTS planner with per-run logging.
Long-term goal: compare belief-space POMDP solvers with RL baselines; only the MCTS path exists today.
Status: early prototype — one planner, CUDA-only, no dependency manifest or packaging.

**Implemented:** `core/` (simulator, room, occupancy map, visualizer) · `envs/` (`generate()`, RL env,
pure `planning_math.py`) · `planners/` (`BaseSearcher`, `MCTSSolver`) · `scripts/` (`run_mcts.py`,
`random_rl.py`, `debug_rl.py`) · `tracking/` (run logging).
**Not implemented:** POMCP/POMCGS (config placeholder only, no belief representation), RL baselines and
training code, teacher–student distillation, any planner use of observations, MCTS tree reuse,
dependency manifest/packaging.

## Layout

```
roomba.urdf    robot (chassis, wheels, casters, sensor link)
configs/       config.yaml, planners.yaml, experiments.yaml
core/          room.py (Room, BoxObstacle, OccupancyMap), simulator.py, visualizer.py
envs/          planning_env.py (G(s,a)), rl_env.py, planning_math.py
planners/      base.py, mcts.py
scripts/       run_mcts.py, random_rl.py, debug_rl.py
tracking/      run_logger.py
tests/         CPU-only tests (untracked, currently failing)
logs/mcts/     generated run logs (git-ignored; older CSVs still tracked)
```

## Prerequisites

- **NVIDIA Isaac Gym** (`isaacgym.gymapi`, `isaacgym.gymtorch`) — external, platform-specific; no
  single-command install path here.
- **CUDA mandatory:** `RoombaSimulator.__init__` raises `RuntimeError` without
  `torch.cuda.is_available()`; `use_gpu`/`use_gpu_pipeline` are always on, `sim_device` forced to
  `cuda:`. There is no CPU path.
- Imports, **no versions pinned:** `torch`, `numpy`, `scipy`, `PyYAML`, `gym`, `isaacgym`,
  `opencv-python` (required by `debug_rl.py`), `tensorboard` (optional).

## Run

From the repository root (paths are relative); no CLI flags or environment variables.

```bash
python scripts/run_mcts.py     # MCTS demo (viewer off)
python scripts/random_rl.py    # random-policy smoke test (viewer on)
python scripts/debug_rl.py     # manual control (viewer on, needs OpenCV)
```

`run_mcts.py` is the only implemented planner experiment: it seeds NumPy/Torch from `mcts.seed`
(`configs/experiments.yaml`), samples a collision-free start/goal pair at least
`room.start_goal_sampling.min_distance` apart, then runs the search/execute loop. It requires
`num_planning_envs` >= the action-grid size (shipped: 15 actions, 16 envs).

## Config, logs, tests

- One owner per file: `configs/config.yaml` (env/room/task/sensors/robot), `configs/planners.yaml`
  (planner), `configs/experiments.yaml` (runner). `env.env_spacing` is overwritten at simulator
  construction, so the configured value never applies.
- One directory per run: `logs/mcts/mcts_<UTC>/{run.yaml,steps.csv,tensorboard/}`. `run.yaml`
  (`schema_version: 1`) and `steps.csv` are the portable source of truth; TensorBoard is derived from
  the same metrics. Logging lives in `tracking/run_logger.py`.
- `python -m unittest discover -s tests -v` — `tests/` is untracked and currently **red**. Two files
  target current code (`test_run_logger.py`, `test_planning_math.py`); `test_mcts.py` and
  `test_env_layout.py` specify an unimplemented refactor (`core/env_layout.py`, explicit
  `mcts.actions`, `num_planning_envs: 9`) — treat them as a spec, not as truth.

## Docs

[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — architecture, invariants, limitations, stale-comment
list (read first) · [`ROADMAP.md`](ROADMAP.md) — milestone, planned experiments, deferred decisions ·
[`AGENTS.md`](AGENTS.md) — rules for contributors and coding agents.
