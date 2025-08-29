import cupy as cp

import time
from algorithm import AgentSystem
from visualization import SimulationVisualizer
import networks
from app_utils import log_sim

class Simulation:
    """
    Orchestrates the entire simulation, acting as a bridge between the core
    algorithm, network generation, and visualization.
    """
    def __init__(self):
        self.params = self._get_default_params()
        self.time_step = 0
        self.steps_per_frame = 1
        self.strategy_history = []
        self.neighbor_history = []
        self.agent_system = None
        self.visualizer = None
        self.initialize(self.params)

    def _get_default_params(self):
        """Returns a dictionary of default simulation parameters."""
        return {
            'networkType': 'ring_1d_periodic', 'numAgents': 128,
            'initialCondition': 'random', 'bankCondition': 'constant',
            'bankValue': 0.0, 'selectionMode': 'global_random',
            'scoreCalculationMode': 'total',
            'win': 2.0, 'tie': 1.5, 'loss': 0.0, 'kT': 100.0,
            'historyLength': 1_000_001,
            'renderResolution': 128,
            'jpegQuality': 80,
            'kymoAspect': 3.0
        }

    def initialize(self, params):
        self.params = params
        self.network_type = params['networkType']
        num_agents_param = params['numAgents']
        kymo_aspect = params.get('kymoAspect', 3.0)
        log_sim(f"Initializing for network '{self.network_type}' with size param {num_agents_param}.")
        adj_matrix, grid_dim, width, height, num_agents = networks.create(
            self.network_type, num_agents_param, kymo_aspect
        )
        self.num_agents = num_agents
        self.agent_system = AgentSystem(
            N=self.num_agents, adjacency_matrix=adj_matrix, grid_dim=grid_dim
        )
        self.visualizer = SimulationVisualizer(width, height, self.network_type)
        self.update_parameters(params)
        self.reset()
        log_sim(f"Initialization complete. System has {self.num_agents} agents.")

    def update_parameters(self, params):
        self.params.update(params)
        if self.params['historyLength'] > 1000:
            self.agent_system.history_length = 1_000_001
        else:
            self.agent_system.history_length = self.params['historyLength']
        self.agent_system.update_params(self.params)
        log_sim(f"Params Updated: kT={self.params['kT']:.1f}, mode='{self.params['selectionMode']}', score='{self.params['scoreCalculationMode']}'")

    def reset(self):
        """Resets the simulation state to its initial conditions at t=0."""
        self.time_step = 0
        self.strategy_history.clear()
        self.neighbor_history.clear()
        self.visualizer.reset()
        self.agent_system.reset_state(
            initial_condition=self.params['initialCondition'],
            bank_condition=self.params['bankCondition'],
            bank_value=self.params['bankValue']
        )
        log_sim(f"State reset to '{self.params['initialCondition']}' strategies and '{self.params['bankCondition']}' bank values.")
        if "1d" in self.params['networkType']:
            self.visualizer.record_kymograph_history(
                self.agent_system.agent_strategies,
                self.agent_system.agent_bank_values,
                0
            )
        self.render()

    def step(self):
        """Advances the simulation by one time step."""
        self.time_step += 1
        self.agent_system.update_physics()

        # If it's a 1D kymograph, record history on every step.
        if 'ring_1d' in self.network_type:
            strategies = self.agent_system.agent_strategies
            bank_values = self.agent_system.agent_bank_values
            self.visualizer.record_kymograph_history(strategies, bank_values, self.time_step)

    def render(self):
        """Renders the current state using the agent's bank value."""
        strategies = self.agent_system.agent_strategies
        bank_values = self.agent_system.agent_bank_values
        return self.visualizer.render(strategies, bank_values, self.time_step)

    def is_finished(self):
        """Checks if the simulation has completed (for 1D kymograph)."""
        return 'ring_1d' in self.network_type and self.time_step >= self.visualizer.HEIGHT

    def needs_reinitialization(self, new_params):
        """Determines if a change in parameters requires a full re-initialization."""
        return (new_params.get('networkType') != self.network_type or
                new_params.get('numAgents') != self.params['numAgents'] or
                new_params.get('kymoAspect') != self.params.get('kymoAspect'))