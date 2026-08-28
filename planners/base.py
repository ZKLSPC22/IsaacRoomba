import math

import torch
import yaml


class BaseNode:
    # MCTS statistics and UCB1 selection
    def __init__(self, parent=None, action_taken=None, is_terminal=False):
        self.parent = parent
        self.action_taken = action_taken  
        self.children = {}                
        self.N = 0                        
        self.Q = 0.0                      
        self.is_terminal = is_terminal

    def ucb1(self, c_param):
        if self.N == 0:
            return float('inf')
        return (self.Q / self.N) + c_param * math.sqrt(math.log(self.parent.N) / self.N)


class BaseSearcher:
    # Search algorithm skeleton, Delegates Expansion and Selection to child implementations.
    def __init__(self, env, config_path="configs/planners.yaml"):
        self.env = env
        self.num_envs = env.num_envs
        self.device = env.device

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        base_cfg = self.config.get('base', {})
        self.max_rollout_depth = base_cfg.get('max_rollout_depth', 15)
        self.c_param = base_cfg.get('c_param', 1.414)
        self.num_iterations = base_cfg.get('num_iterations', 100)
        self.gamma = base_cfg.get('gamma', 0.99)
        self.pbrs_weight = base_cfg.get('pbrs_weight', 1.0)

    def search(self, root_node, num_iterations=None):
        iters = num_iterations if num_iterations is not None else self.num_iterations

        for _ in range(iters):
            # 1. Selection
            node = self._select(root_node)
            
            # 2. Expansion
            if not node.is_terminal and node.N > 0:
                node = self._expand(node)
                
            # 3. Rollout (Batched Monte Carlo Simulation)
            physical_state = self._extract_physical_state(node)
            value = self._batched_rollout(physical_state, node.is_terminal)
            
            # 4. Backpropagation
            self._backpropagate(node, value)

        # Return the action index with the highest visit count from the root
        best_action_idx = max(root_node.children.items(), key=lambda item: item[1].N)[0]
        return best_action_idx

    # Parallel Rollout with telescoped PBRS reward
    def _batched_rollout(self, start_state_tensor: torch.Tensor, is_terminal: bool):
        if is_terminal:
            return 0.0
            
        current_states = start_state_tensor.repeat(self.num_envs, 1)
        
        # 1. Calculate Initial Potential Phi(s_0) for the expanded leaf
        goals_x = self.env.goals[:, 0]
        goals_z = self.env.goals[:, 1]
        start_x = current_states[:, 0]
        start_z = current_states[:, 2]
        
        dist_0 = torch.sqrt((goals_x - start_x)**2 + (goals_z - start_z)**2)
        phi_0 = -self.pbrs_weight * dist_0
        
        cumulative_true_rewards = torch.zeros(self.num_envs, device=self.device)
        active_mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        current_gamma = 1.0
        
        # 2. Run the rollout to accumulate discounted true rewards
        for _ in range(self.max_rollout_depth):
            if not active_mask.any():
                break
                
            random_actions = self._sample_random_actions(self.num_envs)
            next_states, _, step_rewards, dones = self.env.generate(current_states, random_actions)
            
            cumulative_true_rewards[active_mask] += current_gamma * step_rewards[active_mask]
            
            active_mask = active_mask & ~dones
            current_states = next_states
            current_gamma *= self.gamma
            
        # 3. Calculate Final Potential Phi(s_k) for the rollout leaves
        end_x = current_states[:, 0]
        end_z = current_states[:, 2]
        dist_k = torch.sqrt((goals_x - end_x)**2 + (goals_z - end_z)**2)
        phi_k = -self.pbrs_weight * dist_k
        
        # 4. PBRS Telescoping Sum: Sum(gamma^t * r_t) + gamma^k * Phi(s_k) - Phi(s_0)
        rollout_returns = cumulative_true_rewards + (current_gamma * phi_k) - phi_0
        
        return rollout_returns.mean().item()

    def _backpropagate(self, node, value):
        """Walks up the parent pointers to update visit counts and values."""
        while node is not None:
            node.N += 1
            node.Q += value
            node = node.parent

    # =========================================================================
    # ABSTRACT METHODS (To be implemented by MCTS or POMCGS)
    # =========================================================================

    def _select(self, node):
        raise NotImplementedError
        
    def _expand(self, node):
        raise NotImplementedError

    def _extract_physical_state(self, node):
        raise NotImplementedError

    def _sample_random_actions(self, num_samples):
        raise NotImplementedError
