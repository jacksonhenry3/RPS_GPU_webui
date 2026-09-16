import math

import cupy as cp

def get_population_distribution(agent_strategies):
    """Returns the number of agents for each strategy."""
    return cp.asnumpy(cp.sum(agent_strategies, axis=1))

def _agents_per_block(grid_dim, block_size):
    return block_size if grid_dim is None or len(grid_dim) == 1 else block_size * block_size

def _bank_labels(agent_bank_values, num_bins):
    ranks = cp.argsort(cp.argsort(agent_bank_values))
    return (ranks * num_bins) // agent_bank_values.size

def _block_entropy(one_hot_field, N, grid_dim, block_size=2, normalize=True):
    """Calculates the n-point block entropy of a one-hot encoded field (vectorized)."""
    num_symbols = one_hot_field.shape[0]
    agents_per_block = _agents_per_block(grid_dim, block_size)

    if grid_dim is None:
        # Treat as 1D with length N
        length = N
        is_1d = True
    else:
        is_1d = len(grid_dim) == 1
        if is_1d:
            length = grid_dim[0]
        # else: # 2D, rows and cols will be assigned below

    if is_1d:
        num_blocks = length // block_size
        if num_blocks == 0:
            return 0.0

        total_blocks = num_blocks

        # --- Vectorized Block Processing for 1D ---
        field = one_hot_field.reshape(num_symbols, length)[:, :num_blocks * block_size]
        blocks = field.reshape(num_symbols, num_blocks, block_size)
        blocks = blocks.transpose(1, 0, 2) # (num_blocks, num_symbols, block_size)

    else: # 2D
        rows, cols = grid_dim
        num_block_rows = rows // block_size
        num_block_cols = cols // block_size
        if num_block_rows == 0 or num_block_cols == 0:
            return 0.0

        total_blocks = num_block_rows * num_block_cols

        # --- Vectorized Block Processing for 2D ---
        field = one_hot_field.reshape(num_symbols, rows, cols)
        field = field[:, :num_block_rows * block_size, :num_block_cols * block_size]
        blocks = field.reshape(num_symbols, num_block_rows, block_size, num_block_cols, block_size)
        blocks = blocks.transpose(1, 3, 0, 2, 4)
        blocks = blocks.reshape(total_blocks, num_symbols, agents_per_block)

    # 2. Convert all blocks from one-hot to labels (0 .. num_symbols - 1)
    # Result shape: (num_blocks, agents_per_block)
    labels = cp.argmax(blocks, axis=1)

    # 3. Convert base-num_symbols labels to a unique integer ID for each block
    # This is a vectorized dot product for every block simultaneously
    power_of_symbols = num_symbols ** cp.arange(agents_per_block - 1, -1, -1, dtype=cp.int64)
    block_integers = labels @ power_of_symbols

    # --- Entropy Calculation (as before) ---

    # 4. Count occurrences of each unique block configuration
    counts = cp.unique(block_integers, return_counts=True)[1]

    # 5. Calculate probabilities and entropy
    # The sum of counts is simply the total number of blocks
    probabilities = counts / total_blocks
    entropy = -cp.sum(probabilities * cp.log(probabilities))

    # Dividing by this field's own maximum keeps measures built on different alphabet sizes on one 0-1 scale.
    if normalize:
        entropy = entropy / (agents_per_block * math.log(num_symbols))

    return float(entropy)

def get_entropy(agent_strategies, N, grid_dim, block_size=2):
    """Calculates the n-point block entropy of the strategy distribution."""
    return _block_entropy(agent_strategies, N, grid_dim, block_size)

def get_bank_entropy(agent_bank_values, N, grid_dim, block_size=2, num_bins=3):
    """Calculates the n-point block entropy of the bank value distribution."""
    # Included to score wealth on the same footing as strategy; bank values are continuous, so agents are rank binned the way symbolic dynamics discretises a real valued signal.
    # Bandt & Pompe, Phys. Rev. Lett. 88, 174102 (2002): https://doi.org/10.1103/PhysRevLett.88.174102
    return _block_entropy(cp.eye(num_bins, dtype=int)[_bank_labels(agent_bank_values, num_bins)].T, N, grid_dim, block_size)

def get_joint_entropy(agent_strategies, agent_bank_values, N, grid_dim, block_size=2, num_bins=3):
    """Calculates the n-point block entropy of the combined strategy and bank value field."""
    # Included because order merely moving between the two fields changes each marginal but leaves the joint distribution alone, so this is the system's total disorder.
    # Shannon, Bell Syst. Tech. J. 27, 379 (1948): https://doi.org/10.1002/j.1538-7305.1948.tb01338.x
    joint_labels = cp.argmax(agent_strategies, axis=0) * num_bins + _bank_labels(agent_bank_values, num_bins)
    return _block_entropy(cp.eye(3 * num_bins, dtype=int)[joint_labels].T, N, grid_dim, block_size)

def get_entropy_rate(one_hot_field, N, grid_dim, block_size=2):
    """Calculates the per site entropy rate h(n) = H(n) - H(n-1)."""
    # Included because it divides out the block size, leaving the irreducible randomness per agent instead of a quantity that grows with n.
    # Crutchfield & Feldman, Chaos 13, 25 (2003): https://doi.org/10.1063/1.1530990
    sites = _agents_per_block(grid_dim, block_size)
    previous_sites = _agents_per_block(grid_dim, block_size - 1)
    entropy = _block_entropy(one_hot_field, N, grid_dim, block_size, normalize=False)
    previous_entropy = _block_entropy(one_hot_field, N, grid_dim, block_size - 1, normalize=False) if block_size > 1 else 0.0
    return (entropy - previous_entropy) / ((sites - previous_sites) * math.log(one_hot_field.shape[0]))

def get_excess_entropy(one_hot_field, N, grid_dim, block_size=2):
    """Calculates the excess entropy E(n) = H(n) - n * h(n)."""
    # Included because subtracting the per site randomness from the block entropy leaves the structure the field actually carries, which is the order parameter we are after.
    # Feldman & Crutchfield, Phys. Rev. E 67, 051104 (2003): https://doi.org/10.1103/PhysRevE.67.051104
    return _block_entropy(one_hot_field, N, grid_dim, block_size) - get_entropy_rate(one_hot_field, N, grid_dim, block_size)

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
