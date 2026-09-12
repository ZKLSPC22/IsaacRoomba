"""Planning reward and leaf-heuristic mathematics.

This module is intentionally free of any simulator *and* configuration dependency:
it imports `torch` only, and never imports `isaacgym`, `RoombaSimulator`,
`RoombaPlanningEnv`, or Gym spaces. That keeps the reward and heuristic formulas
testable on CPU without launching Isaac Gym.

Every task constant is an explicit parameter. There are no module-level task
constants and no fallback defaults: the configuration owner is
`configs/config.yaml` under `task:` (`goal_radius`, `goal_reward`,
`collision_penalty`, `step_cost`), read by `RoombaPlanningEnv` and passed in here.
Keeping the values out of this module preserves one authoritative source and
avoids a silent second copy that could drift from the configuration.

Scope and non-goals
-------------------
- The functions here reproduce the planning reward and leaf heuristic exactly; no
  formula or constant value is changed by living here.
- ``compute_straight_line_heuristic`` returns a **value estimate** (an estimate of
  the discounted return of an ideal straight-line trajectory). It is **not** part
  of the environment reward and it is **not** potential-based reward shaping.
  PBRS would require an added reward term of the form
  ``gamma * Phi(next_state) - Phi(state)``, which does not exist in this
  codebase. See `ROADMAP.md` for the deferred PBRS investigation.
- The heuristic deliberately ignores collision risk; that is existing behavior.
"""

from __future__ import annotations

import torch

#: Tolerance used to detect the ``gamma == 1`` degenerate discount case. This is
#: a numerical guard, not a task parameter, so it is not configurable.
_GAMMA_ONE_TOLERANCE = 1e-9

# ---------------------------------------------------------------------------
# 15D explicit state layout (see docs/ARCHITECTURE.md §6). Positions and goals are
# room-local; quaternion `qw` is index 6. Only the fields used by the heuristic
# are named here.
# ---------------------------------------------------------------------------
_STATE_X_INDEX = 0
_STATE_Z_INDEX = 2
_STATE_GOAL_X_INDEX = 13
_STATE_GOAL_Z_INDEX = 14


def compute_planning_rewards(
    dist_to_goal: torch.Tensor,
    bumped: torch.Tensor,
    *,
    goal_radius: float,
    goal_reward: float,
    collision_penalty: float,
    step_cost: float,
) -> torch.Tensor:
    """Batched planning reward for one macro-action.

    Equivalent to the original expression::

        reached = dist_to_goal < goal_radius
        rewards = (reached.float() * goal_reward)
                + (bumped.float() * collision_penalty)
                + step_cost

    Args:
        dist_to_goal: Euclidean distance to the goal in metres, per environment.
        bumped: Boolean (or boolean-like) bumper mask, per environment.
        goal_radius: Distance (m) at which the goal counts as reached.
        goal_reward: Reward added on a successful step.
        collision_penalty: Reward added on a step with a bumper contact.
        step_cost: Reward added on every macro-action.

    Returns:
        A float32 tensor with the same shape and device as ``dist_to_goal``.

    Note:
        The result is always float32 because the ``reached``/``bumped`` masks are
        cast with ``.float()``; a float64 input therefore yields a float32 result.
        This matches the original inline implementation exactly.

        This is the search/training objective only. It contains no shaping term.
    """
    reached = dist_to_goal < goal_radius
    return (reached.float() * goal_reward) + (bumped.float() * collision_penalty) + step_cost


def compute_straight_line_heuristic(
    states: torch.Tensor,
    is_terminal: torch.Tensor,
    gamma: float,
    *,
    max_linear_velocity: float,
    macro_action_duration: float,
    goal_radius: float,
    goal_reward: float,
    step_cost: float,
) -> torch.Tensor:
    """Discounted-return estimate for an ideal straight-line path to the goal.

    Estimates the expected discounted return of driving straight at maximum speed
    from each state to the goal, which gives MCTS leaves a stable, discountable
    value without the cost and variance of rollouts.

    Horizon estimation::

        max_dist_per_step = max_linear_velocity * macro_action_duration
        H = max((distance - goal_radius) / max_dist_per_step, 1)

    The floor of ``1`` keeps the ``H - 1`` exponent non-negative, since a
    non-terminal state needs at least one more macro-action. The remaining
    distance is clamped at zero because entering the goal radius already
    terminates the episode.

    Value::

        V(s) = gamma^(H-1) * goal_reward + step_cost * (1 - gamma^H) / (1 - gamma)
        V(s) = 0  where is_terminal

    The ``gamma == 1`` case uses the geometric-series limit ``goal_reward + H *
    step_cost`` instead of dividing by zero.

    Args:
        states: ``[batch, 15]`` explicit states; indices 0/2 are room-local ``x``/``z``
            and 13/14 are the room-local goal ``x``/``z``.
        is_terminal: Boolean mask, per batch entry. Terminal entries return ``0``.
        gamma: Discount factor in ``[0, 1]``.
        max_linear_velocity: Robot speed limit in m/s.
        macro_action_duration: Duration of one macro-action in seconds
            (``macro_action_ticks * dt``).
        goal_radius: Distance (m) at which the goal counts as reached.
        goal_reward: Reward assumed for the terminal goal step.
        step_cost: Reward assumed for every macro-action.

    Returns:
        A float tensor of shape ``[batch]``, on the same device and with the same
        dtype as the computation over ``states``. Input tensors are not mutated.

    Note:
        This is a value estimate, not a reward. It is not PBRS, and it omits
        collision risk by design. ``collision_penalty`` is intentionally not a
        parameter here because the heuristic models a collision-free path.
    """
    dx = states[:, _STATE_GOAL_X_INDEX] - states[:, _STATE_X_INDEX]
    dz = states[:, _STATE_GOAL_Z_INDEX] - states[:, _STATE_Z_INDEX]
    distance = torch.sqrt(dx * dx + dz * dz)

    # 1. Estimate steps to goal (H)
    max_dist_per_step = max_linear_velocity * macro_action_duration

    # Distance remaining outside the goal radius
    d_remain = torch.clamp(distance - goal_radius, min=0.0)
    H = d_remain / max_dist_per_step

    # A non-terminal state needs at least one more step to reach the goal,
    # so floor H at 1 to keep the exponent (H - 1) non-negative.
    H = torch.clamp(H, min=1.0)

    # 2. Compute discounted return
    # V(s) = gamma^(H-1) * goal_reward + step_cost * (1 - gamma^H) / (1 - gamma)
    # Guard gamma == 1, where the geometric series has the limit H.
    if abs(1.0 - gamma) < _GAMMA_ONE_TOLERANCE:
        values = goal_reward + H * step_cost
    else:
        values = (gamma ** (H - 1)) * goal_reward + step_cost * ((1.0 - gamma ** H) / (1.0 - gamma))

    return torch.where(is_terminal, torch.zeros_like(values), values)
