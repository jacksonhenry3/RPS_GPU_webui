import matplotlib
matplotlib.use('Agg') # Use a non-interactive backend suitable for servers
import matplotlib.pyplot as plt
import numpy as np
import io

# Define colors to match the UI, converting from RGB(255) to Matplotlib's 0-1 scale
ROCK_COLOR = (44/255, 160/255, 148/255)
PAPER_COLOR = (255/255, 127/255, 14/255)
SCISSORS_COLOR = (148/255, 103/255, 189/255)
UI_BG_COLOR = '#f8f7f5'
PLOT_BG_COLOR = '#fdfaf6'
TEXT_COLOR = '#5c524f'
GRID_COLOR = '#e7e2dd'

def _configure_plot_style(fig, ax, title, ylabel):
    """Helper function to apply common styling to both plots."""
    fig.patch.set_facecolor(UI_BG_COLOR)
    ax.set_facecolor(PLOT_BG_COLOR)
    ax.set_title(title, color=TEXT_COLOR, weight='bold')
    ax.set_xlabel('Time Step', color=TEXT_COLOR)
    ax.set_ylabel(ylabel, color=TEXT_COLOR)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, color=GRID_COLOR)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COLOR)
    ax.tick_params(colors=TEXT_COLOR)

def create_strategy_plot(history_data):
    """Generates a line plot of strategy counts over time."""
    fig, ax = plt.subplots(figsize=(8, 4), dpi=100)
    
    if not history_data:
        ax.text(0.5, 0.5, 'Simulation Paused - No Data Yet', ha='center', va='center', fontsize=12, color='gray')
    else:
        data = np.array(history_data)
        time_steps = data[:, 0]
        total_agents = data[0, 1:].sum()

        ax.plot(time_steps, data[:, 1], color=ROCK_COLOR, label='Rock', linewidth=2)
        ax.plot(time_steps, data[:, 2], color=PAPER_COLOR, label='Paper', linewidth=2)
        ax.plot(time_steps, data[:, 3], color=SCISSORS_COLOR, label='Scissors', linewidth=2)
        
        ax.set_ylim(0, total_agents if total_agents > 0 else 1)
        ax.set_xlim(0, max(1, time_steps.max()))
        ax.legend(frameon=False, labelcolor=TEXT_COLOR)

    _configure_plot_style(fig, ax, 'Strategy Population Over Time', 'Agent Population')
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.2, facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def create_neighbor_plot(history_data):
    """Generates a plot of neighbor-pair likelihoods with two-toned lines."""
    fig, ax = plt.subplots(figsize=(8, 4), dpi=100)

    if not history_data:
        ax.text(0.5, 0.5, 'Simulation Paused - No Data Yet', ha='center', va='center', fontsize=12, color='gray')
    else:
        data = np.array(history_data)
        time = data[:, 0]
        # Data columns: 0:time, 1:RR, 2:PP, 3:SS, 4:RP, 5:RS, 6:PS
        
        # To create the two-toned effect, we plot a thick outer line,
        # then a thinner inner line of a different color on top.
        outer_lw, inner_lw = 3.0, 1.5

        # Heterogeneous pairs (two-toned)
        ax.plot(time, data[:, 4], color=ROCK_COLOR, linewidth=outer_lw)
        ax.plot(time, data[:, 4], color=PAPER_COLOR, label='Rock-Paper', linewidth=inner_lw)
        
        ax.plot(time, data[:, 5], color=ROCK_COLOR, linewidth=outer_lw)
        ax.plot(time, data[:, 5], color=SCISSORS_COLOR, label='Rock-Scissors', linewidth=inner_lw)

        ax.plot(time, data[:, 6], color=PAPER_COLOR, linewidth=outer_lw)
        ax.plot(time, data[:, 6], color=SCISSORS_COLOR, label='Paper-Scissors', linewidth=inner_lw)

        # Homogeneous pairs (single color)
        ax.plot(time, data[:, 1], color=ROCK_COLOR, label='Rock-Rock', linewidth=2)
        ax.plot(time, data[:, 2], color=PAPER_COLOR, label='Paper-Paper', linewidth=2)
        ax.plot(time, data[:, 3], color=SCISSORS_COLOR, label='Scissors-Scissors', linewidth=2)

        ax.set_ylim(0, max(0.01, np.max(data[:, 1:]) * 1.1)) # Dynamic Y-axis
        ax.set_xlim(0, max(1, time.max()))
        ax.legend(frameon=False, labelcolor=TEXT_COLOR, ncol=2)

    _configure_plot_style(fig, ax, 'Neighbor-Pair Likelihoods', 'Likelihood')

    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.2, facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()

