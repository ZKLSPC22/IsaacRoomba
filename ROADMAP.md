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

- [ ] Paired-seed `num_iterations` comparison (optionally `c_param`) plus a short summary.
- [ ] Read `task.goal_radius` in `run_mcts.py` instead of its hard-coded `0.5` already-at-goal guard.
- [ ] Guard the `None` action index `search()` can return.
- [ ] Configuration and budget semantics: `env.env_spacing` (overwritten), the unused `pomcgs` block,
      permissive custom-room fallbacks, `num_iterations` vs `max_expansions`; check
      `num_planning_envs` >= action-grid size explicitly.
- [ ] GPU-free configuration/interface checks; no stale claims in the docs; resolve the stale code
      comments in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) §8.
- [ ] `RoombaRLEnv`: configurable timeout and `task.*` constants instead of hard-coded values.

`tests/` is untracked and red, so "CPU-only tests exist" is not yet true as a passing gate.

## Planned experiments (prepared, not launched)

`gamma = 0.95` and `max_expansions = 300` throughout; nothing here changes planner or environment code.

| Sweep | Settings |
| --- | --- |
| Baseline | `c_param = 1.414`, `num_iterations = 100`, seeds `[0, 1, 2, 3, 4]` |
| Iteration budget | `num_iterations = [30, 100, 300]`, `c_param = 1.414`, same seeds |
| Optional exploration | `c_param = [0.5, 1.414, 2.5]`, `num_iterations = 100`, same seeds |

Paired seeds, so every arm sees the same start/goal pairs. `max_expansions` is excluded (it cannot
bind for `num_iterations <= 300`) and `gamma` is excluded from the first sweep (it governs both
backpropagation and the leaf heuristic, so differences would not be attributable). Report success
rate, termination reasons, steps to goal, final distance, search time per decision, episode time, tree
size, max depth, expansion calls, physics ticks. Five seeds are descriptive, not statistical.

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
