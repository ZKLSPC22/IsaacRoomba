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
- The heuristic's distance comes from a precomputed 2D geodesic distance field
  sampled with ``torch.nn.functional.grid_sample`` when ``distance_field``,
  ``room_width`` and ``room_depth`` are supplied; with all three omitted it falls
  back to the Euclidean straight-line distance (the pre-geodesic behavior).
- The heuristic deliberately ignores collision risk; that is existing behavior.
  Geodesic mode accounts for obstacles only through the length of the path, not
  through contact risk.
"""

from __future__ import annotations

import torch

#: Tolerance used to detect the ``gamma == 1`` degenerate discount case. This is
#: a numerical guard, not a task parameter, so it is not configurable.
_GAMMA_ONE_TOLERANCE = 1e-9

#: Allowed values for the heuristic's distance-field interpolation, owned by
#: ``task.heuristic_sampling`` in `configs/config.yaml`.
#:
#: * ``"bilinear"`` blends the four neighbouring cells. This is the documented
#:   contract, and it gives a smooth spatial gradient.
#: * ``"nearest"`` quantizes to the containing cell centre. It is coarser (up to
#:   half a cell, 0.025 m at the shipped 0.05 m resolution) and produces a
#:   staircase in the gradient, but reproduces the sampling this heuristic used
#:   before the mode became configurable.
VALID_SAMPLING_MODES = ("nearest", "bilinear")

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
    distance_field: torch.Tensor | None = None,
    room_width: float | None = None,
    room_depth: float | None = None,
    sampling_mode: str,
) -> torch.Tensor:
    """Discounted-return estimate for an ideal collision-free path to the goal.

    Estimates the expected discounted return of driving at maximum speed along the
    ideal path from each state to the goal, which gives MCTS leaves a stable,
    discountable value without the cost and variance of rollouts.

    Distance source
    ---------------
    Two mutually exclusive modes, selected by whether the geodesic arguments are
    supplied:

    * **Geodesic (preferred)** — ``distance_field``, ``room_width`` and
      ``room_depth`` are all given. The remaining distance is sampled from a
      precomputed 2D distance field produced by
      ``OccupancyMap.compute_distance_field``, so the estimate accounts for
      obstacles instead of assuming a clear line of sight.
    * **Straight-line (legacy)** — all three are ``None``. The remaining distance
      is the Euclidean distance from the state to the goal stored in the state
      itself. This is the pre-geodesic behavior, retained until every caller
      passes a field.

    Supplying only some of the three raises ``ValueError``: a partial set would
    silently change which distance the search optimizes.

    Sampling
    --------
    Room-local ``x``/``z`` are normalized to the ``[-1, 1]`` cube that
    ``grid_sample`` expects and looked up with the interpolation named by
    ``sampling_mode``::

        norm_x = clamp(x / (room_width / 2), -1, 1)
        norm_z = clamp(z / (room_depth / 2), -1, 1)
        grid = stack([norm_x, norm_z], dim=-1)[None, None]        # [1, 1, batch, 2]
        geodesic = grid_sample(distance_field, grid, mode=sampling_mode,
                               padding_mode="border", align_corners=True).view(-1)

    ``align_corners=True`` maps the normalized ``-1``/``+1`` extremes onto the
    field's first and last cell centres, and ``padding_mode="border"`` clamps
    out-of-room poses to the border instead of returning NaN. ``"bilinear"`` is
    the documented contract; ``"nearest"`` is retained so the pre-toggle sampling
    can be reproduced exactly (see `VALID_SAMPLING_MODES`).

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
            and 13/14 are the room-local goal ``x``/``z`` (used only in
            straight-line mode).
        is_terminal: Boolean mask, per batch entry. Terminal entries return ``0``.
        gamma: Discount factor in ``[0, 1]``.
        max_linear_velocity: Robot speed limit in m/s.
        macro_action_duration: Duration of one macro-action in seconds
            (``macro_action_ticks * dt``).
        goal_radius: Distance (m) at which the goal counts as reached.
        goal_reward: Reward assumed for the terminal goal step.
        step_cost: Reward assumed for every macro-action.
        distance_field: ``[1, 1, depth_cells, width_cells]`` float geodesic
            distance field in metres, with the height axis mapped to ``z``.
        room_width: Full room extent (m) along ``x``, used for normalization.
        room_depth: Full room extent (m) along ``z``, used for normalization.
        sampling_mode: Interpolation used by ``grid_sample``, either ``"nearest"``
            or ``"bilinear"`` (see `VALID_SAMPLING_MODES`). Owned by
            ``task.heuristic_sampling``. Required with no default so the choice is
            always explicit; it is unused in straight-line mode, which samples no
            field at all.

    Returns:
        A float tensor of shape ``[batch]``, on the same device as ``states``.
        The dtype follows the computation: it matches ``states`` in straight-line
        mode, and is the promotion of ``states`` and ``distance_field`` in
        geodesic mode (a float32 field over float64 states yields float64). Input
        tensors are not mutated.

    Note:
        This is a value estimate, not a reward. It is not PBRS, and it omits
        collision risk by design. ``collision_penalty`` is intentionally not a
        parameter here because the heuristic models a collision-free path.

        The goal is read from the distance field in geodesic mode, so the field
        must have been computed for the same goal as ``states[:, 13:15]``.

    Raises:
        ValueError: If only some of ``distance_field``, ``room_width`` and
            ``room_depth`` are supplied.
    """
    geodesic_parts = (distance_field is not None, room_width is not None, room_depth is not None)
    if any(geodesic_parts) and not all(geodesic_parts):
        raise ValueError(
            "compute_straight_line_heuristic received a partial geodesic "
            "configuration: pass all of `distance_field`, `room_width` and "
            "`room_depth`, or none of them to use the straight-line distance."
        )

    if sampling_mode not in VALID_SAMPLING_MODES:
        raise ValueError(
            f"Invalid sampling_mode {sampling_mode!r}; expected one of "
            f"{VALID_SAMPLING_MODES}. This is owned by task.heuristic_sampling in "
            "configs/config.yaml."
        )

    if distance_field is None:
        # Legacy mode: a clear line of sight, ignoring obstacles.
        dx = states[:, _STATE_GOAL_X_INDEX] - states[:, _STATE_X_INDEX]
        dz = states[:, _STATE_GOAL_Z_INDEX] - states[:, _STATE_Z_INDEX]
        distance = torch.sqrt(dx * dx + dz * dz)
    else:
        norm_x = (states[:, _STATE_X_INDEX] / (room_width / 2.0)).clamp(-1.0, 1.0)
        norm_z = (states[:, _STATE_Z_INDEX] / (room_depth / 2.0)).clamp(-1.0, 1.0)

        # `grid_sample` requires the grid to share the input's dtype; the states
        # may be float64 while the cached field is float32.
        grid = torch.stack([norm_x, norm_z], dim=-1).to(dtype=distance_field.dtype)
        grid = grid.unsqueeze(0).unsqueeze(0)

        sampled = torch.nn.functional.grid_sample(
            distance_field,
            grid,
            mode=sampling_mode,
            padding_mode="border",
            align_corners=True,
        )
        distance = sampled.view(-1).to(torch.promote_types(states.dtype, distance_field.dtype))

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
