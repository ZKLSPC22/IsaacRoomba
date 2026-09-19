# IsaacRoomba Roadmap

Living document: current capability, active milestone, deliberate deferrals. Architecture lives in
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md). Tags: **[current]**, **[later]**, **[speculative]**.

## Objective

Point-to-point navigation for a differential-drive robot under partial observability using Isaac Gym:
(1) a trustworthy fully-observable planner as reference — **[current]**; (2) RL baselines in the same
environment — **[later]**; (3) belief-space POMDP planning (POMCP/POMCGS) — **[later]**;
(4) teacher–student distillation — **[later]**. Only (1) exists in code.

## Milestone [current]

Clean-up and debug MCTS, run experiments to verify robustness of code, debug, then run experiments to verify performance.

### Open Tasks:

- [ ] Suspect asymmetry between clockwise rotation and anti-clockwise rotation, see /logs/debug_mcts/mcts_20260915_131226/steps.csv

  - [x] Root cause diagnosed: caster friction was never disabled. Casters now load frictionless
    (`core/simulator.py`, `docs/ARCHITECTURE.md` §2); residual asymmetry is unexplained.

  - [X] Problem remains, now suspect insufficient velocity and position PhysX iteration, increase num_position_iterations 4 -> 16, num_velocity_iterations 1 -> 8. Add config['simulation'], add num_position_iterations and num_velocity_iterations to config['simulation'], inheritted by simulator.py.

- [ ] Tests should be changed to fit the new version, and ensure the new bumper physics and Hijkstra Heuristic.


- [ ] Two newly added .md files (SimulationSetup.md, TensorAPI.md) are Isaac Gym official documents, these should be addressed in other agent guide .md files.


- [ ] .md files are outdated, identify problematic lines.


- [ ] Update outdated .md files.


- [ ] Log Tests results and correspond with git version. What should be the work flow? Commit more often?


- [ ] Run experiments using different room-sizes, obstacle configurations, seeds, and search depths.


`tests/` is tracked and CPU-only; `python -m unittest discover -s tests` passes 76 tests with no GPU, so it can serve as the pre-experiment gate. `AGENTS.md` still carries the older "untracked and red" assumption.


## Milestones [Future]

- **M2 — POMCGS implementation [later]**
- **M3 — RL baseline implementation [later]**: Rework RL environment, logging, and implement tests, implement PPO.
- **M4 — Run Experiments and compare results [later]**
- **M5 — Teacher-Student distillation [Speculative]**: Using Monte-Carlo Localization + Path Planning as the teacher policy, then use history conditioned action diffusion as student.
- **M6 — Final Experiments [speculative]**

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
