"""Planning reward, bumper mask, and leaf-heuristic mathematics.

Pure `torch` math with no simulator, Gym, or YAML dependency, so every formula
here is testable on CPU without launching Isaac Gym. `RoombaPlanningEnv` and
`RoombaRLEnv` read the configured values and pass them in.

Every task constant is an explicit parameter: there are no module-level defaults
and no fallbacks. Their owner is `configs/config.yaml` under `task:`, which keeps
one authoritative source instead of a second copy that can silently drift.

`compute_straight_line_heuristic` returns a **value estimate** (the discounted
return of an ideal obstacle-free trajectory) used as an MCTS leaf value. It is not
part of the environment reward and not potential-based reward shaping, which would
need an added `gamma * Phi(next_state) - Phi(state)` term that this codebase does
not implement. It also ignores collision risk by design: geodesic mode accounts
for obstacles through path length only, never through contact risk.
"""

from __future__ import annotations

import torch

#: Tolerance used to detect the ``gamma == 1`` degenerate discount case. This is
#: a numerical guard, not a task parameter, so it is not configurable.
_GAMMA_ONE_TOLERANCE = 1e-9

#: Allowed values for the heuristic's distance-field interpolation, owned by
#: ``task.heuristic_sampling`` in `configs/config.yaml`.
#:
#: * ``"bilinear"`` blends the four neighbouring cells: the documented contract,
#:   with a smooth spatial gradient.
#: * ``"nearest"`` quantizes to the containing cell centre. Up to half a cell
#:   (0.025 m at the shipped 0.05 m resolution) coarser and stair-stepped, but it
#:   reproduces the sampling used before the mode became configurable.
VALID_SAMPLING_MODES = ("nearest", "bilinear")

# ---------------------------------------------------------------------------
# 15D explicit state layout (see docs/ARCHITECTURE.md §6). Positions and goals are
# room-local; quaternion `qw` is index 6. Only the fields used here are named.
# ---------------------------------------------------------------------------
_STATE_X_INDEX = 0
_STATE_Z_INDEX = 2
_STATE_GOAL_X_INDEX = 13
_STATE_GOAL_Z_INDEX = 14

#: Contact-force columns that count as a bump: world X and Z, the horizontal axes
#: in this Y-up frame. Column 1 is world Y; see `compute_bumped` for why it is out.
_HORIZONTAL_FORCE_COLUMNS = [0, 2]


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
        cast with ``.float()``, so a float64 input yields a float32 result.
    """
    reached = dist_to_goal < goal_radius
    return (reached.float() * goal_reward) + (bumped.float() * collision_penalty) + step_cost


# Extracted bumped math to allow CPU-only testing, only counts horrizontal forces, Y-up
def compute_bumped(
    contact_forces: torch.Tensor,
    *,
    threshold: float,
) -> torch.Tensor:
    horizontal = torch.norm(contact_forces[:, _HORIZONTAL_FORCE_COLUMNS], dim=-1)
    return horizontal > threshold


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
    """Discounted-return estimate for an ideal obstacle-free path to the goal.

    Estimates what an ideal run at maximum speed to the goal is worth, which gives
    MCTS leaves a stable, discountable value without the cost and variance of
    rollouts.

    Distance source
    ---------------
    Selected by whether the geodesic arguments are supplied:

    * **Geodesic** -- ``distance_field``, ``room_width`` and ``room_depth`` are all
      given. The remaining distance is sampled from a precomputed 2D field
      (``OccupancyMap.compute_distance_field``), so obstacles lengthen the path
      instead of being ignored.
    * **Straight-line** -- all three are ``None``. The remaining distance is the
      Euclidean distance to the goal stored in the state itself.

    Mixing the two raises ``ValueError``, because a partial set would silently
    change which distance the search optimizes.

    Sampling
    --------
    Room-local ``x``/``z`` are normalized to the ``[-1, 1]`` cube that
    ``grid_sample`` expects, then looked up with ``sampling_mode``::

        norm_x = clamp(x / (room_width / 2), -1, 1)
        norm_z = clamp(z / (room_depth / 2), -1, 1)
        grid = stack([norm_x, norm_z], dim=-1)[None, None]        # [1, 1, batch, 2]
        geodesic = grid_sample(distance_field, grid, mode=sampling_mode,
                               padding_mode="border", align_corners=True).view(-1)

    ``align_corners=True`` maps normalized ``-1``/``+1`` onto the field's first and
    last cell centres; ``padding_mode="border"`` clamps out-of-room poses to the
    border instead of returning NaN.

    Horizon and value::

        max_dist_per_step = max_linear_velocity * macro_action_duration
        H = max((distance - goal_radius) / max_dist_per_step, 1)
        V(s) = gamma^(H-1) * goal_reward + step_cost * (1 - gamma^H) / (1 - gamma)
        V(s) = 0  where is_terminal

    The horizon floor of ``1`` keeps ``H - 1`` non-negative, since a non-terminal
    state needs at least one more macro-action. ``gamma == 1`` uses the geometric
    series limit ``goal_reward + H * step_cost`` instead of dividing by zero.

    Args:
        states: ``[batch, 15]`` explicit states; indices 0/2 are room-local ``x``/``z``
            and 13/14 are the room-local goal ``x``/``z`` (used in straight-line mode
            only).
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
        The dtype matches ``states`` in straight-line mode, and is the promotion of
        ``states`` and ``distance_field`` in geodesic mode (a float32 field over
        float64 states yields float64). Input tensors are not mutated.

    Note:
        The goal is read from the distance field in geodesic mode, so the field
        must have been computed for the same goal as ``states[:, 13:15]``.

    Raises:
        ValueError: If only some of ``distance_field``, ``room_width`` and
            ``room_depth`` are supplied, or if ``sampling_mode`` is not in
            `VALID_SAMPLING_MODES`.
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
        # Straight-line mode: an unobstructed path from the state to the goal.
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

    # Ideal horizon in macro-actions. Clamping the distance at zero is safe because
    # entering the goal radius already terminates the episode; flooring H at 1
    # keeps the (H - 1) exponent below non-negative for a non-terminal state.
    max_dist_per_step = max_linear_velocity * macro_action_duration
    d_remain = torch.clamp(distance - goal_radius, min=0.0)
    H = torch.clamp(d_remain / max_dist_per_step, min=1.0)

    # gamma == 1 is the geometric series limit H * step_cost, not a division by zero.
    if abs(1.0 - gamma) < _GAMMA_ONE_TOLERANCE:
        values = goal_reward + H * step_cost
    else:
        values = (gamma ** (H - 1)) * goal_reward + step_cost * ((1.0 - gamma ** H) / (1.0 - gamma))

    return torch.where(is_terminal, torch.zeros_like(values), values)
