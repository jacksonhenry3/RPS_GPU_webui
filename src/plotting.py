import matplotlib
matplotlib.use('Agg') # Use a non-interactive backend suitable for servers
import matplotlib.pyplot as plt
import numpy as np
import os

# Define colors to match the UI, converting from RGB(255) to Matplotlib's 0-1 scale
ROCK_COLOR = (44/255, 160/255, 148/255)
PAPER_COLOR = (255/255, 127/255, 14/255)
SCISSORS_COLOR = (148/255, 103/255, 189/255)
UI_BG_COLOR = '#f8f7f5'
PLOT_BG_COLOR = '#fdfaf6'
TEXT_COLOR = '#5c524f'
GRID_COLOR = '#e7e2dd'

def _configure_plot_style(fig, ax, title, ylabel):
    """Helper function to apply common styling to all plots."""
    fig.patch.set_facecolor(UI_BG_COLOR)
    ax.set_facecolor(PLOT_BG_COLOR)
    ax.set_title(title, color=TEXT_COLOR, weight='bold')
    ax.set_xlabel('Time Step', color=TEXT_COLOR)
    ax.set_ylabel(ylabel, color=TEXT_COLOR)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, color=GRID_COLOR)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COLOR)
    ax.tick_params(colors=TEXT_COLOR)

def plot_population_history(history_pop, static_dir):
    """Generates a line plot of strategy counts over time."""
    fig, ax = plt.subplots(figsize=(8, 4), dpi=100)
    
    if not history_pop:
        ax.text(0.5, 0.5, 'Simulation Paused - No Data Yet', ha='center', va='center', fontsize=12, color='gray')
    else:
        data = np.array(history_pop)
        time_steps = np.arange(len(data))
        total_agents = data[0, :].sum()

        ax.plot(time_steps, data[:, 0], color=ROCK_COLOR, label='Rock', linewidth=2)
        ax.plot(time_steps, data[:, 1], color=PAPER_COLOR, label='Paper', linewidth=2)
        ax.plot(time_steps, data[:, 2], color=SCISSORS_COLOR, label='Scissors', linewidth=2)
        
        ax.set_ylim(0, total_agents if total_agents > 0 else 1)
        ax.set_xlim(0, max(1, len(data) - 1))
        ax.legend(frameon=False, labelcolor=TEXT_COLOR)

    _configure_plot_style(fig, ax, 'Strategy Population Over Time', 'Agent Population')
    
    filepath = os.path.join(static_dir, "population_history.png")
    fig.savefig(filepath, format='png', bbox_inches='tight', pad_inches=0.2, facecolor=fig.get_facecolor())
    plt.close(fig)
    return filepath

def plot_entropy_history(history_entropy, static_dir):
    """Generates a plot of entropy over time."""
    fig, ax = plt.subplots(figsize=(8, 4), dpi=100)

    if not history_entropy:
        ax.text(0.5, 0.5, 'Simulation Paused - No Data Yet', ha='center', va='center', fontsize=12, color='gray')
    else:
        data = np.array(history_entropy)
        time_steps = np.arange(len(data))

        ax.plot(time_steps, data, color=TEXT_COLOR, linewidth=2)
        
        ax.set_ylim(0, max(0.01, np.max(data) * 1.1))
        ax.set_xlim(0, max(1, len(data) - 1))

    _configure_plot_style(fig, ax, 'Entropy Over Time', 'Entropy')

    filepath = os.path.join(static_dir, "entropy_history.png")
    fig.savefig(filepath, format='png', bbox_inches='tight', pad_inches=0.2, facecolor=fig.get_facecolor())
    plt.close(fig)
    return filepath

def generate_plots(sim_state, params, static_dir):
    plot_paths = []
    if sim_state.history_pop:
        plot_paths.append(plot_population_history(sim_state.history_pop, static_dir))
    if sim_state.history_entropy:
        plot_paths.append(plot_entropy_history(sim_state.history_entropy, static_dir))
    return plot_paths