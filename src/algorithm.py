import cupy as cp
import numpy as np
import initial_conditions
import measurements

# Define precision for the simulation
precision = cp.float32

# Natural-log value below which cp.exp() underflows to exactly 0 for `precision`.
# Used to monitor how close the softmax selection is to losing probability
# resolution on non-winning strategies (see AgentSystem.pop_worst_exp_headroom).
_EXP_UNDERFLOW_FLOOR = float(np.log(np.finfo(np.dtype(precision)).smallest_subnormal))

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
        self.agent_strategies = cp.zeros((3, N), dtype=self.precision)
        self.agent_bank_values = cp.zeros(N, dtype=self.precision)
        self.selection_function = self.choose_new_strats_local_choice
        self.score_calculation_mode = 'total' # New attribute
        self._neighbor_strategy_counts = None
        self._worst_exp_arg = None

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
        self._neighbor_strategy_counts = neighbor_counts
        potential_payoffs = self.payoff_matrix.dot(neighbor_counts)
        total_payoffs = cp.sum(potential_payoffs * self.agent_strategies, axis=0)
        self.agent_bank_values = self.agent_bank_values + total_payoffs

    def _track_exp_headroom(self, exp_arg, has_masked_entries):
        """
        Records the smallest (most negative) softmax exponent argument seen
        since the last pop_worst_exp_headroom() call, ignoring entries that
        are intentionally masked to -inf (unavailable strategies), so the
        tracker only reflects precision risk, not expected masking.
        """
        step_min = cp.where(cp.isfinite(exp_arg), exp_arg, cp.inf).min() if has_masked_entries else exp_arg.min()
        self._worst_exp_arg = step_min if self._worst_exp_arg is None else cp.minimum(self._worst_exp_arg, step_min)

    def pop_worst_exp_headroom(self):
        """
        Returns (and resets) the smallest softmax exponent-argument margin
        observed since the last call, in natural-log units above the point
        where cp.exp() underflows to exactly 0 for the active `precision`.
        Smaller is worse; <= 0 means at least one non-winning strategy's
        selection probability has already hit a hard zero instead of just
        being very small. Returns None if no softmax selection has run yet
        (e.g. only 'deterministic' mode has been used).
        """
        if self._worst_exp_arg is None:
            return None
        headroom = float(self._worst_exp_arg) - _EXP_UNDERFLOW_FLOOR
        self._worst_exp_arg = None
        return headroom

    def choose_new_strats_local_choice(self):
        neigh_bank = self._calculate_neighborhood_scores()
        strategies_in_neighborhood = self._neighbor_strategy_counts + self.agent_strategies
        masked_neigh_bank = cp.where(strategies_in_neighborhood > 0, neigh_bank, -cp.inf)
        scaled_neigh_bank = self.beta * masked_neigh_bank
        max_val = cp.max(scaled_neigh_bank, axis=0, keepdims=True)
        exp_arg = scaled_neigh_bank - max_val
        self._track_exp_headroom(exp_arg, has_masked_entries=True)
        exp_payoff = cp.exp(exp_arg, dtype=self.precision)
        probs = exp_payoff / cp.sum(exp_payoff, axis=0, keepdims=True)
        r = cp.random.rand(self.N).astype(self.precision)
        chosen = cp.argmax(r < cp.cumsum(probs, axis=0), axis=0)
        self.agent_strategies = initial_conditions._labels_to_one_hot(chosen, self.N)

    def choose_new_strats_random(self):
        neigh_bank = self._calculate_neighborhood_scores()
        scaled_neigh_bank = self.beta * neigh_bank
        max_val = cp.max(scaled_neigh_bank, axis=0, keepdims=True)
        exp_arg = scaled_neigh_bank - max_val
        self._track_exp_headroom(exp_arg, has_masked_entries=False)
        exp_payoff = cp.exp(exp_arg, dtype=self.precision)
        probs = exp_payoff / cp.sum(exp_payoff, axis=0, keepdims=True)
        r = cp.random.rand(self.N).astype(self.precision)
        chosen = cp.argmax(r < cp.cumsum(probs, axis=0), axis=0)
        self.agent_strategies = initial_conditions._labels_to_one_hot(chosen, self.N)
    
    def choose_new_strats_deterministic(self):
        neigh_bank = self._calculate_neighborhood_scores()
        strategies_in_neighborhood = self._neighbor_strategy_counts + self.agent_strategies
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
        self.agent_strategies = initial_conditions._labels_to_one_hot(chosen, self.N)

    def _initialize_strategies(self, condition='random'):
        conditions = {
            'all_rock': lambda: initial_conditions._all_one_strategy(self.N, 0),
            'vertical_stripes': lambda: initial_conditions._vertical_stripes(self.N, self.grid_dim, 3),
            'pie_slices': lambda: initial_conditions._pie_slices(self.N, self.grid_dim),
            'single_invader': lambda: initial_conditions._single_invader(self.N, self.grid_dim),
            'double_invader': lambda: initial_conditions._double_invader(self.N, self.grid_dim),
            'cross_invasion': lambda: initial_conditions._cross_invasion(self.N, self.grid_dim),
            'corner_siege': lambda: initial_conditions._corner_siege(self.N, self.grid_dim),
            'diamond_lattice': lambda: initial_conditions._diamond_lattice(self.N, self.grid_dim),
            'periodic_stripes': lambda: initial_conditions._periodic_stripes(self.N, self.grid_dim),
            'symmetric_gradient': lambda: initial_conditions._symmetric_gradient(self.N, self.grid_dim),
            'split': lambda: initial_conditions._split_strategies(self.N),
            'random': lambda: initial_conditions._random_strategies(self.N),
        }
        self.agent_strategies = conditions.get(condition, lambda: initial_conditions._random_strategies(self.N))()
