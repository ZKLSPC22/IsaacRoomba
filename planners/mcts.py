import torch
from planners.base import BaseNode, BaseSearcher


class MCTSNode(BaseNode):
    """Stores the exact 15D physical state for fully observable planning."""
    def __init__(self, state: torch.Tensor, parent=None, action_taken=None, is_terminal=False):
        super().__init__(parent, action_taken, is_terminal)
        self.state = state


class MCTSSolver(BaseSearcher):
    """
    Fully Observable Tabular MCTS.
    Builds the discrete action space dynamically from configs/planners.yaml.
    """
    def __init__(self, env, config_path="configs/planners.yaml"):
        # Initialize the base class, which loads the YAML and sets shared params
        super().__init__(env, config_path)
        
        # Extract MCTS-specific hyperparameters
        self.mcts_cfg = self.config.get('mcts', {})
        self.actions = self._create_action_grid()
        self.num_actions = len(self.actions)

    def _create_action_grid(self):
        """Constructs the discrete action grid driven strictly by YAML values."""
        v_vals = self.mcts_cfg.get('v_vals', [0.0, 0.5, 1.0])
        omega_vals = self.mcts_cfg.get('omega_vals', [-1.0, -0.5, 0.0, 0.5, 1.0])
        
        actions = []
        for v in v_vals:
            for w in omega_vals:
                actions.append([v, w])
        return torch.tensor(actions, dtype=torch.float32, device=self.device)

    def _extract_physical_state(self, node):
        """For fully observable MCTS, the node's state is the exact physical state."""
        return node.state

    def _sample_random_actions(self, num_samples):
        """Uniformly samples from the configured discrete action grid."""
        random_indices = torch.randint(0, self.num_actions, (num_samples,), device=self.device)
        return self.actions[random_indices]

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
        next_states, _, _, dones = self.env.generate(states_batch, actions_batch)
        
        # 5. Extract only the valid successor states corresponding to the discrete actions
        #    ignoring the outputs from the dummy padding environments.
        for a_idx in range(self.num_actions):
            child_state = next_states[a_idx].clone()
            is_done = dones[a_idx].item()
            node.children[a_idx] = MCTSNode(
                state=child_state, 
                parent=node, 
                action_taken=a_idx, 
                is_terminal=is_done
            )
            
        # Return the first newly created child to begin the rollout phase
        return node.children[0]
