import cupy as cp

# Define precision for the simulation
precision = cp.float64

class AgentSystem:
    """
    Core logic for the Rock-Paper-Scissors agent-based model.
    This class is completely self-contained and does not handle
    visualization or network creation.
    """
    def __init__(self, N: int, adjacency_matrix, grid_dim=None):
        self.N = N
        self.adjacency_matrix = adjacency_matrix
        self.total_edges = self.adjacency_matrix.sum() / 2
        self.grid_dim = grid_dim
        self.precision = precision
        self.beta = 0.01
        self.payoff_matrix = cp.zeros((3, 3), dtype=self.precision)
        self.agent_strategies = cp.zeros((3, N), dtype=int)
        self.agent_bank_values = cp.zeros(N, dtype=self.precision)
        self.selection_function = self.choose_new_strats_local_choice
        self.score_calculation_mode = 'total' # New attribute

    def get_population_distribution(self):
        """Returns the number of agents for each strategy."""
        return cp.asnumpy(cp.sum(self.agent_strategies, axis=1))

    def get_entropy(self):
        """Calculates the entropy of the strategy distribution."""
        pop_dist = cp.sum(self.agent_strategies, axis=1) / self.N
        return cp.asnumpy(-cp.sum(pop_dist * cp.log(pop_dist + 1e-9)))

    def get_appeal_distribution(self):
        """Calculates the average appeal for each strategy."""
        neigh_bank = self._calculate_neighborhood_scores()
        return cp.asnumpy(cp.mean(neigh_bank, axis=1))

    def get_neighbor_pair_counts(self):
        s = self.agent_strategies.astype(self.precision)
        s_rock, s_paper, s_scissors = s[0], s[1], s[2]
        rr_links = s_rock.T.dot(self.adjacency_matrix.dot(s_rock))
        pp_links = s_paper.T.dot(self.adjacency_matrix.dot(s_paper))
        ss_links = s_scissors.T.dot(self.adjacency_matrix.dot(s_scissors))
        rp_links = s_rock.T.dot(self.adjacency_matrix.dot(s_paper))
        rs_links = s_rock.T.dot(self.adjacency_matrix.dot(s_scissors))
        ps_links = s_paper.T.dot(self.adjacency_matrix.dot(s_scissors))
        pair_counts = cp.array([rr_links, pp_links, ss_links, rp_links, rs_links, ps_links]) / 2
        return pair_counts / self.total_edges if self.total_edges > 0 else cp.zeros_like(pair_counts)

    def _calculate_neighborhood_scores(self):
        """
        Calculates the score for each potential strategy for each agent,
        based on the selected calculation mode ('total' or 'average').
        """
        bank_by_strategy = self.agent_strategies * self.agent_bank_values
        total_neigh_bank = self.adjacency_matrix.dot(bank_by_strategy.T).T + bank_by_strategy

        if self.score_calculation_mode == 'average':
            # Count agents of each strategy in the neighborhood (including self)
            strats_in_neighborhood = self.adjacency_matrix.dot(self.agent_strategies.T).T + self.agent_strategies
            # Divide total bank by count, handling division by zero
            return cp.where(strats_in_neighborhood > 0, total_neigh_bank / strats_in_neighborhood, 0)
        else: # 'total'
            return total_neigh_bank

    def get_agent_neighborhood_scores(self):
        """
        Calculates the neighborhood score for each agent's *current* strategy.
        This is used for visualization.
        """
        # Get the (3, N) matrix of scores for all potential strategies
        all_scores = self._calculate_neighborhood_scores()
        # Select the score corresponding to each agent's active strategy
        agent_scores = cp.sum(all_scores * self.agent_strategies, axis=0)
        return agent_scores

    def update_params(self, params):
        kt = params.get('kT', 100.0)
        self.beta = 1.0 / (kt + 1e-9)
        self._update_payoff_matrix(
            tie=params.get('tie', 1.5),
            win=params.get('win', 2.0),
            loss=params.get('loss', 0.0)
        )
        self.score_calculation_mode = params.get('scoreCalculationMode', 'total')
        mode = params.get('selectionMode', 'local_choice')
        if mode == 'local_choice': self.selection_function = self.choose_new_strats_local_choice
        elif mode == 'global_random': self.selection_function = self.choose_new_strats_random
        elif mode == 'deterministic': self.selection_function = self.choose_new_strats_deterministic

    def reset_state(self, initial_condition='random', bank_condition='constant', bank_value=0.0):
        self._initialize_strategies(condition=initial_condition)
        if bank_condition == 'random':
            self.agent_bank_values = cp.random.uniform(-bank_value, bank_value, size=self.N).astype(self.precision)
        else:
            self.agent_bank_values.fill(float(bank_value))

    def update_physics(self):
        self._play_round()
        self.selection_function()

    def _update_payoff_matrix(self, tie, win, loss):
        self.payoff_matrix = cp.array([[tie, loss, win], [win, tie, loss], [loss, win, tie]], dtype=self.precision)

    def _play_round(self):
        neighbor_counts = self.adjacency_matrix.dot(self.agent_strategies.T).T
        potential_payoffs = self.payoff_matrix.dot(neighbor_counts)
        total_payoffs = cp.sum(potential_payoffs * self.agent_strategies, axis=0)
        self.agent_bank_values = self.agent_bank_values + total_payoffs

    def choose_new_strats_local_choice(self):
        neigh_bank = self._calculate_neighborhood_scores()
        strategies_in_neighborhood = self.adjacency_matrix.dot(self.agent_strategies.T).T + self.agent_strategies
        masked_neigh_bank = cp.where(strategies_in_neighborhood > 0, neigh_bank, -cp.inf)
        scaled_neigh_bank = self.beta * masked_neigh_bank
        max_val = cp.max(scaled_neigh_bank, axis=0, keepdims=True)
        exp_payoff = cp.exp(scaled_neigh_bank - max_val, dtype=self.precision)
        probs = exp_payoff / (cp.sum(exp_payoff, axis=0, keepdims=True) + 1e-9)
        r = cp.random.rand(self.N).astype(self.precision)
        chosen = cp.argmax(r < cp.cumsum(probs, axis=0), axis=0)
        self.agent_strategies = self._labels_to_one_hot(chosen)

    def choose_new_strats_random(self):
        neigh_bank = self._calculate_neighborhood_scores()
        scaled_neigh_bank = self.beta * neigh_bank
        max_val = cp.max(scaled_neigh_bank, axis=0, keepdims=True)
        exp_payoff = cp.exp(scaled_neigh_bank - max_val, dtype=self.precision)
        probs = exp_payoff / (cp.sum(exp_payoff, axis=0, keepdims=True) + 1e-9)
        r = cp.random.rand(self.N).astype(self.precision)
        chosen = cp.argmax(r < cp.cumsum(probs, axis=0), axis=0)
        self.agent_strategies = self._labels_to_one_hot(chosen)
    
    def choose_new_strats_deterministic(self):
        neigh_bank = self._calculate_neighborhood_scores()
        strategies_in_neighborhood = self.adjacency_matrix.dot(self.agent_strategies.T).T + self.agent_strategies
        is_available = (strategies_in_neighborhood > 0)
        masked_neigh_bank = cp.where(is_available, neigh_bank, -cp.inf)
        max_scores = cp.max(masked_neigh_bank, axis=0)
        min_scores = cp.min(cp.where(is_available, neigh_bank, cp.inf), axis=0)
        is_tie = (max_scores == min_scores)
        # If there is any tie along axis zero, not just max and min
        any_tie  = cp.sum( (masked_neigh_bank == max_scores), axis=0) > 1
        best_choice = cp.argmax(masked_neigh_bank, axis=0)
        current_choice = cp.argmax(self.agent_strategies, axis=0)
        #change to is_tie for max/min tie, any_tie for any tie
        chosen = cp.where(any_tie, current_choice, best_choice)
        self.agent_strategies = self._labels_to_one_hot(chosen)

    def _labels_to_one_hot(self, labels):
        return cp.eye(3, dtype=int)[labels].T

    def _initialize_strategies(self, condition='random'):
        conditions = {
            'all_rock': lambda: self._all_one_strategy(0),
            'vertical_stripes': lambda: self._vertical_stripes(3),
            'pie_slices': self._pie_slices,
            'single_invader': self._single_invader,
            'split': self._split_strategies,
            'random': self._random_strategies
        }
        self.agent_strategies = conditions.get(condition, self._random_strategies)()

    def _random_strategies(self):
        return self._labels_to_one_hot(cp.random.randint(0, 3, size=self.N))

    def _all_one_strategy(self, strategy_index=0):
        return self._labels_to_one_hot(cp.full(self.N, strategy_index, dtype=int))

    def _split_strategies(self):
        labels = cp.zeros(self.N, dtype=int)
        labels[self.N // 2:] = 1
        return self._labels_to_one_hot(labels)

    def _single_invader(self):
        labels = cp.zeros(self.N, dtype=int)
        center_index = (self.grid_dim[0] // 2) * self.grid_dim[1] + (self.grid_dim[1] // 2) if self.grid_dim else self.N // 2
        labels[center_index] = 1
        return self._labels_to_one_hot(labels)

    def _vertical_stripes(self, num_stripes=3):
        width = self.grid_dim[1] if self.grid_dim else self.N
        indices = cp.arange(self.N) % width
        stripe_width = width // num_stripes
        labels = cp.zeros(self.N, dtype=int)
        for i in range(num_stripes):
            mask = (indices >= i * stripe_width) & (indices < (i + 1) * stripe_width)
            labels[mask] = i % 3
        return self._labels_to_one_hot(labels)

    def _pie_slices(self):
        if not self.grid_dim: return self._random_strategies()
        rows, cols = self.grid_dim
        y, x = cp.meshgrid(cp.arange(rows), cp.arange(cols))
        angles = cp.arctan2(y - rows / 2, x - cols / 2) * 180 / cp.pi
        labels = cp.zeros((rows, cols), dtype=int)
        labels[(angles >= 60)] = 1
        labels[(angles <= -60)] = 2
        return self._labels_to_one_hot(labels.flatten())