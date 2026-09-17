import cupy as cp

def get_population_distribution(agent_strategies):
    """Returns the number of agents for each strategy."""
    return cp.asnumpy(cp.sum(agent_strategies, axis=1))

def get_entropy(agent_strategies, N, grid_dim, block_size=2, gap=1):
    """
    Calculates block entropy of the strategy distribution using every
    possible block start position (overlapping windows, periodic
    wraparound), rather than a fixed non-overlapping tiling.

    `gap` is the stride between the `block_size` sites sampled within a
    single block (gap=1 samples immediate neighbors); the indexing below
    is written generically in terms of start index + gap so a future
    non-unit gap needs no structural change.
    """
    if grid_dim is None or len(grid_dim) == 1:
        # Treat as 1D with length N
        length = N if grid_dim is None else grid_dim[0]
        agents_per_block = block_size
        total_blocks = length

        # --- Overlapping Block Processing for 1D ---
        labels = cp.argmax(agent_strategies.reshape(3, length), axis=0)  # (length,)
        starts = cp.arange(length)
        offsets = gap * cp.arange(block_size)
        idx = (starts[:, None] + offsets[None, :]) % length  # (total_blocks, agents_per_block)
        strategy_labels = labels[idx]

    else: # 2D
        rows, cols = grid_dim
        agents_per_block = block_size * block_size
        total_blocks = rows * cols

        # --- Overlapping Block Processing for 2D ---
        labels = cp.argmax(agent_strategies.reshape(3, rows, cols), axis=0)  # (rows, cols)
        offsets = gap * cp.arange(block_size)
        row_idx = (cp.arange(rows)[:, None] + offsets[None, :]) % rows  # (rows, block_size)
        col_idx = (cp.arange(cols)[:, None] + offsets[None, :]) % cols  # (cols, block_size)

        # For every (row_start, col_start), gather its block_size x block_size block.
        blocks = labels[row_idx[:, None, :, None], col_idx[None, :, None, :]]  # (rows, cols, block_size, block_size)
        strategy_labels = blocks.reshape(total_blocks, agents_per_block)

    # Convert base-3 labels to a unique integer ID for each block
    # This is a vectorized dot product for every block simultaneously
    power_of_3 = 3 ** cp.arange(agents_per_block - 1, -1, -1, dtype=cp.int64)
    block_integers = strategy_labels.astype(cp.int64) @ power_of_3

    # --- Entropy Calculation (as before) ---

    # Count occurrences of each unique block configuration
    max_block_val = 3**agents_per_block
    counts = cp.bincount(block_integers, minlength=max_block_val)

    # Calculate probabilities and entropy
    # The sum of counts is simply the total number of blocks
    probabilities = counts[counts > 0] / total_blocks
    entropy = -cp.sum(probabilities * cp.log2(probabilities))

    return float(entropy)

# Distances (gaps) sampled by get_entropy_spectrum, shared with plotting.py
# for axis/legend labeling.
ENTROPY_DISTANCES = (1, 2, 4, 8, 16, 32, 64)

def get_entropy_spectrum(agent_strategies, N, grid_dim, distances=ENTROPY_DISTANCES, block_size=2):
    """
    Runs get_entropy() once per distance in `distances`, giving an
    entropy-vs-distance profile instead of a single number. Returns a
    plain list, ordered the same as `distances`.
    """
    return [
        get_entropy(agent_strategies, N, grid_dim, block_size=block_size, gap=d)
        for d in distances
    ]

def get_single_site_entropy(agent_strategies):
    """Shannon entropy (bits) of the marginal strategy distribution across all
    agents. Used as the per-site independence baseline in
    get_mutual_information_spectrum."""
    counts = cp.sum(agent_strategies, axis=1)
    total = counts.sum()
    probabilities = counts[counts > 0] / total
    return float(-cp.sum(probabilities * cp.log2(probabilities)))

def get_mutual_information_spectrum(agent_strategies, N, grid_dim, distances=ENTROPY_DISTANCES, block_size=2, entropy_spectrum=None):
    """
    Multi-information at each distance: MI(d) = agents_per_block * H(single
    site) - H(joint block at distance d). For block_size=2 in 1D this is the
    standard two-point mutual information between sites `d` apart; in 2D
    (where a block is a gap x gap square of 4 corner sites) it generalizes to
    the total correlation across those 4 sites.

    It decays toward 0 once sites that far apart behave independently, so
    unlike raw joint entropy (which rises toward a data-dependent ceiling),
    the distance where this flattens to ~0 directly reads as the
    correlation length / scale of emergent structure.

    Pass `entropy_spectrum` (a prior get_entropy_spectrum result for the same
    state/distances/block_size) to avoid recomputing the joint entropies.
    """
    h_single = get_single_site_entropy(agent_strategies)
    agents_per_block = block_size if (grid_dim is None or len(grid_dim) == 1) else block_size * block_size
    joint_entropies = (
        entropy_spectrum if entropy_spectrum is not None
        else get_entropy_spectrum(agent_strategies, N, grid_dim, distances=distances, block_size=block_size)
    )
    return [agents_per_block * h_single - h_joint for h_joint in joint_entropies]

def get_appeal_distribution(neigh_bank):
    """Calculates the average appeal for each strategy."""
    return cp.asnumpy(cp.mean(neigh_bank, axis=1))

def get_neighbor_pair_counts(agent_strategies, adjacency_matrix, total_edges, precision):
    s = agent_strategies.astype(precision)
    s_rock, s_paper, s_scissors = s[0], s[1], s[2]
    rr_links = s_rock.T.dot(adjacency_matrix.dot(s_rock))
    pp_links = s_paper.T.dot(adjacency_matrix.dot(s_paper))
    ss_links = s_scissors.T.dot(adjacency_matrix.dot(s_scissors))
    rp_links = s_rock.T.dot(adjacency_matrix.dot(s_paper))
    rs_links = s_rock.T.dot(adjacency_matrix.dot(s_scissors))
    ps_links = s_paper.T.dot(adjacency_matrix.dot(s_scissors))
    pair_counts = cp.array([rr_links, pp_links, ss_links, rp_links, rs_links, ps_links]) / 2
    return pair_counts / total_edges if total_edges > 0 else cp.zeros_like(pair_counts)

def get_agent_neighborhood_scores(all_scores, agent_strategies):
    """
    Calculates the neighborhood score for each agent's *current* strategy.
    This is used for visualization.
    """
    # Select the score corresponding to each agent's active strategy
    agent_scores = cp.sum(all_scores * agent_strategies, axis=0)
    return agent_scores
