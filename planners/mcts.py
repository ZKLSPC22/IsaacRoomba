import torch
from planners.base import BaseNode, BaseSearcher


class MCTSNode(BaseNode):
    """Stores the exact 15D physical state for fully observable planning."""
    def __init__(self, state: torch.Tensor, parent=None, action_taken=None, reward = 0.0, is_terminal=False, heuristic_value=0.0):
        super().__init__(
            parent=parent,
            action_taken=action_taken,
            reward=reward,
            is_terminal=is_terminal,
            heuristic_value=heuristic_value
        )
        self.state = state


class MCTSSolver(BaseSearcher):
    """
    Fully Observable Tabular MCTS.
    Builds the discrete action space explicitly from `mcts.actions` in configs/planners.yaml.
    """
    def __init__(self, env, config_path="configs/planners.yaml"):
        # Initialize the base class, which loads the YAML and sets shared params
        super().__init__(env, config_path)
        
        # Extract MCTS-specific hyperparameters
        self.mcts_cfg = self.config.get('mcts', {})
        self.actions = self._create_action_grid()
        self.num_actions = len(self.actions)
        self._validate_config()

    def _validate_config(self):
        """Validate configuration invariants before any search runs."""
        if self.num_actions <= 0:
            raise ValueError(
                "mcts.actions is empty in configs/planners.yaml; define at least one "
                "[linear_velocity, angular_velocity] action."
            )
        if self.num_actions > self.num_envs:
            raise ValueError(
                f"num_actions ({self.num_actions}) exceeds num_envs ({self.num_envs}). "
                "Increase num_planning_envs in configs/config.yaml or remove entries "
                "from mcts.actions in configs/planners.yaml."
            )
        if self.num_iterations < 1:
            raise ValueError(
                "num_iterations must be >= 1 to produce root children; "
                f"got {self.num_iterations} from base.num_iterations in configs/planners.yaml."
            )
        if not (0.0 <= self.gamma <= 1.0):
            raise ValueError(
                f"gamma must be in [0, 1]; got {self.gamma} from base.gamma in "
                "configs/planners.yaml."
            )

    def _create_action_grid(self):
        """Loads the explicit discrete action set from `mcts.actions`.

        The YAML list is read directly, so list position is preserved and becomes the
        action's index — its identity in `node.action_taken`, the run log, and the
        `BaseSearcher.search()` tie-break. Entries must be numeric
        `[linear_velocity, angular_velocity]` pairs inside `[-1, 1]`; anything else
        fails fast here rather than being clamped or truncated later.

        There is no fallback grid: a missing or misspelled key raises, because a
        silent default would produce a valid-looking run with the wrong actions.
        """
        if 'actions' not in self.mcts_cfg:
            raise KeyError(
                "Missing required key 'mcts.actions' in configs/planners.yaml. "
                "List the discrete actions explicitly as "
                "[linear_velocity, angular_velocity] pairs in [-1, 1]."
            )

        raw_actions = self.mcts_cfg['actions']
        if not isinstance(raw_actions, (list, tuple)):
            raise TypeError(
                f"mcts.actions must be a list of [linear_velocity, angular_velocity] pairs; "
                f"got {type(raw_actions).__name__}."
            )

        actions = []
        for index, entry in enumerate(raw_actions):
            if not isinstance(entry, (list, tuple)) or len(entry) != 2:
                raise ValueError(
                    f"mcts.actions[{index}] must be a [linear_velocity, angular_velocity] pair; "
                    f"got {entry!r}."
                )
            for component in entry:
                if isinstance(component, bool) or not isinstance(component, (int, float)):
                    raise TypeError(
                        f"mcts.actions[{index}] must contain numbers; got {component!r}."
                    )
                if not -1.0 <= float(component) <= 1.0:
                    raise ValueError(
                        f"mcts.actions[{index}] component {component!r} is outside [-1, 1]. "
                        "Actions are normalized; scaling happens in generate()."
                    )
            actions.append([float(entry[0]), float(entry[1])])

        # reshape(-1, 2) keeps the empty case a well-formed [0, 2] tensor, which
        # _validate_config then rejects via the num_actions > 0 invariant.
        return torch.tensor(actions, dtype=torch.float32, device=self.device).reshape(-1, 2)

    def _extract_physical_state(self, node):
        """For fully observable MCTS, the node's state is the exact physical state."""
        return node.state

    def _select(self, node):
        """Descends the tree using UCB1 until a leaf is hit."""
        while len(node.children) > 0 and not node.is_terminal:
            node = max(node.children.values(), key=lambda c: c.ucb1(self.c_param))
        return node

    # Parallel Node Expansion
    def _expand(self, node):
        """
        Batches the generation of all discrete action successor states simultaneously on the GPU.
        
        Because the simulator requires fixed-size batch inputs matching the total number of
        allocated environments (num_envs), but the discrete action grid may contain fewer
        candidate actions (num_actions <= num_envs), we fill the active slots with candidate
        actions and leave the remaining trailing slots as zero-padded dummy actions.
        """
        # 1. Replicate the single parent state across the full batch dimension [num_envs, state_dim]
        states_batch = node.state.repeat(self.num_envs, 1) 
        
        # 2. Allocate the full action tensor [num_envs, action_dim] initialized with zeros
        actions_batch = torch.zeros((self.num_envs, 2), device=self.device)
        
        # 3. Inject the discrete actions into the top active slice [:num_actions, :]
        #    The remaining slots [num_actions:, :] act as unused padding to satisfy the batch shape.
        actions_batch[:self.num_actions] = self.actions
        
        # 4. Advance all environments in parallel on the GPU: G(s, a) -> (s', o, r, dones)
        # MCTS only needs state/reward/done, so skip unused observation work.
        next_states, _, rewards, dones = self.env.generate(states_batch, actions_batch, compute_observation=False)
        self.num_expansions += 1
        
        # 5. Extract only the valid successor states corresponding to the discrete actions
        #    ignoring the outputs from the dummy padding environments.
        active_states = next_states[:self.num_actions]
        active_rewards = rewards[:self.num_actions]
        active_dones = dones[:self.num_actions]

        # Batch leaf evaluation for all new children in a single GPU pass.
        heuristic_values = self.env.compute_heuristic_values(
            active_states, active_dones, self.gamma
        )

        # One GPU->CPU transfer per quantity instead of a separate .item() per child.
        rewards_np = active_rewards.cpu().numpy()
        dones_np = active_dones.cpu().numpy()
        heuristic_np = heuristic_values.cpu().numpy()

        for a_idx in range(self.num_actions):
            node.children[a_idx] = MCTSNode(
                state=active_states[a_idx].clone(),
                parent=node,
                action_taken=a_idx,
                reward=float(rewards_np[a_idx]),
                is_terminal=bool(dones_np[a_idx]),
                heuristic_value=float(heuristic_np[a_idx]),
            )
            
        # Return a random newly created child to begin the rollout phase, so no
        # single action (e.g. index 0) is systematically favoured.
        random_a_idx = torch.randint(0, self.num_actions, (1,), device=self.device).item()
        return node.children[random_a_idx]
