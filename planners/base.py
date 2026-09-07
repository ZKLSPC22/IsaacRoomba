import math

import yaml


class BaseNode:
    # MCTS statistics and UCB1 selection
    def __init__(self, parent=None, action_taken=None, reward=0.0, is_terminal=False, heuristic_value=0.0):
        self.parent = parent
        self.action_taken = action_taken
        self.reward = float(reward)
        self.children = {}
        self.N = 0                        
        self.Q = 0.0                      
        self.is_terminal = is_terminal
        self.heuristic_value = float(heuristic_value)

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
        self.c_param = base_cfg.get('c_param', 1.414)
        self.num_iterations = base_cfg.get('num_iterations', 100)
        self.gamma = base_cfg.get('gamma', 0.99)
        self.max_expansions = base_cfg.get('max_expansions', 15)

        # Number of node expansions (== generate() calls) in the last search.
        self.num_expansions = 0

    def search(self, root_node, num_iterations=None):
        iters = num_iterations if num_iterations is not None else self.num_iterations

        # Reset the per-search expansion counter.
        self.num_expansions = 0

        # A terminal root has no outgoing actions to evaluate.
        if root_node.is_terminal:
            return None

        # Expand the root immediately (if the budget permits) to populate children.
        if self.num_expansions < self.max_expansions:
            self._expand(root_node)

        iteration = 0
        while iteration < iters and self.num_expansions < self.max_expansions:
            iteration += 1

            # 1. Selection
            node = self._select(root_node)
            
            # 2. Expansion
            if (not node.is_terminal) and node.N > 0:
                node = self._expand(node)
                
            # 3. Leaf Evaluation — the heuristic was computed and stored on the
            #    node when it was created during expansion.
            leaf_value = node.heuristic_value
            
            # 4. Backpropagation (absorb leaf reward + gamma per Bellman equation)
            self._backpropagate(node, leaf_value)

        # Select the executed action. Visited children are ranked by mean return
        # (Q/N); unvisited children are ranked by their immediate one-step return
        # estimate (reward + gamma * heuristic), so a tiny budget never arbitrarily
        # favours action 0.
        def value_of(child):
            if child.N > 0:
                return child.Q / child.N
            return child.reward + (self.gamma * child.heuristic_value)

        if not root_node.children:
            return None
        best_action_idx = max(root_node.children.items(), key=lambda item: value_of(item[1]))[0]
        return best_action_idx

    def _backpropagate(self, node, value):
        """Walk up the tree applying the Bellman equation at each depth.

        `value` is the future expected return at `node` (rollout or heuristic
        leaf value). Each non-root node absorbs its immediate reward and
        discounts by gamma, so its Q accumulates the action-value for the
        transition entering it.

        The root has no incoming transition, so it records the return from its
        own perspective without an extra reward/discount step.
        """
        while node is not None:
            if node.parent is not None:
                value = node.reward + (self.gamma * value)
            node.N += 1
            node.Q += value
            # In case node is the root node, note.parent defaults to None, hence meeting the exit condition for the while loop
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
