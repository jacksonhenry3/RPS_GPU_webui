"""CPU-only plots shared by the web frontend and persisted batch runs."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

COLORS = ('#009E73', '#E69F00', '#CC79A7', '#0072B2')


def plot_measurements(rows, output_dir):
    """Render scalar measurement dictionaries using their actual step values."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not rows:
        return []
    steps = np.array([float(row['step']) for row in rows])
    groups = [
        ('population_history', 'Agent population',
         [('population_rock', 'Rock'), ('population_paper', 'Paper'), ('population_scissors', 'Scissors')]),
        ('entropy_history', 'Shannon entropy (bits)',
         [('strategy_bits', 'Strategy'), ('bank_bits', 'Bank'),
          ('product_bits', 'Marginal outer product'), ('joint_bits', 'Observed joint')]),
        ('normalized_entropy', 'Entropy / log₂(3k)',
         [('product_normalized', 'Marginal outer product'), ('joint_normalized', 'Observed joint')]),
        ('mutual_info_history', 'Mutual information (bits)',
         [('strategy_bank_mi_bits', 'Strategy–bank'), ('temporal_mi_bits', 'Current–next state')]),
        ('temporal_entropy', 'Shannon entropy (bits)',
         [('temporal_product_bits', 'Current–next marginal product'),
          ('temporal_joint_bits', 'Observed current–next joint'),
          ('next_given_current_bits', 'Next given current')]),
    ]
    paths = []
    for filename, ylabel, fields in groups:
        fig, ax = plt.subplots(figsize=(8, 4), dpi=120)
        plotted = False
        for color, (key, label) in zip(COLORS, fields):
            values = np.array([float(row[key]) if row.get(key) not in (None, '') else np.nan for row in rows])
            if np.isfinite(values).any():
                ax.plot(steps, values, color=color, label=label)
                plotted = True
        if not plotted:
            plt.close(fig)
            continue
        ax.set(xlabel='Time step', ylabel=ylabel)
        ax.grid(alpha=0.2)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=2, frameon=False)
        path = output_dir / f'{filename}.png'
        fig.savefig(path, bbox_inches='tight')
        plt.close(fig)
        paths.append(str(path))
    return paths


def plot_saved_run(run_dir, output_dir):
    """Plot even a partial run; only measurements.csv is needed."""
    with (Path(run_dir) / 'measurements.csv').open(newline='') as handle:
        return plot_measurements(list(csv.DictReader(handle)), output_dir)


def generate_plots(sim_state, params, static_dir):
    """Thin compatibility adapter for the existing web route."""
    return plot_measurements(list(sim_state.history_measurements), static_dir)
