import cupy as cp

# Define precision and colors for the simulation
precision = cp.float64
# Colors: Teal, Orange, Purple
COLORS = cp.array([[44., 160., 148.], [255., 127., 14.], [148., 103., 189.]]) / 255.

class SimulationVisualizer:
    """
    Handles the conversion of agent state data into a visual representation (image).
    """
    def __init__(self, width: int, height: int, network_type: str):
        self.WIDTH = width
        self.HEIGHT = height
        self.network_type = network_type
        self.image_gpu = cp.zeros((self.HEIGHT, self.WIDTH, 3), dtype=cp.uint8)

    def reset(self):
        """Clears the image buffer."""
        self.image_gpu.fill(0)

    def record_kymograph_history(self, agent_strategies, agent_bank_values, time_step):
        """Records a single line of history for 1D kymograph simulations."""
        new_colors = self._agents_to_colors(agent_strategies, agent_bank_values)
        if time_step < self.HEIGHT:
            self.image_gpu[time_step] = new_colors
        else:
            # Roll the image up by one pixel and add the new line at the bottom
            self.image_gpu = cp.roll(self.image_gpu, -1, axis=0)
            self.image_gpu[-1] = new_colors

    def render(self, agent_strategies, agent_bank_values, time_step):
        """
        Renders the current state to the image buffer.
        For 1D kymographs, this just returns the pre-rendered history.
        For 2D grids, this generates the image from the current agent state.
        """
        if 'ring_1d' not in self.network_type:
            new_colors = self._agents_to_colors(agent_strategies, agent_bank_values)
            self.image_gpu = new_colors.reshape((self.HEIGHT, self.WIDTH, 3))
            
        return self.image_gpu

    def _agents_to_colors(self, agent_strategies, agent_bank_values):
        """
        Converts agent data into an array of RGB colors, modulating brightness
        by the agent's bank value.
        """
        base_colors = agent_strategies.T.dot(COLORS)

        min_bank = agent_bank_values.min()
        max_bank = agent_bank_values.max()
        bank_range = max_bank - min_bank
        epsilon = 1e-6 

        norm_bank = cp.where(
            bank_range > epsilon,
            (agent_bank_values - min_bank) / bank_range,
            0.5  # Use 0.5 for neutral brightness if all values are the same
        )

        # NEW: Increased brightness scale from 0.5 to 1.0 for higher contrast
        brightness = 0.5 + 0.5 * norm_bank
            
        modulated_colors = base_colors * brightness[:, cp.newaxis]
        return (cp.clip(modulated_colors, 0, 1) * 255).astype(cp.uint8)