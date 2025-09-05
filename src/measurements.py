import cupy as cp

def get_population_distribution(agent_strategies):
    """Returns the number of agents for each strategy."""
    return cp.asnumpy(cp.sum(agent_strategies, axis=1))

def get_entropy(agent_strategies, N, grid_dim, block_size=2):
    """Calculates the block entropy of the strategy distribution (vectorized)."""
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
        if length % block_size != 0:
            raise ValueError("Grid dimension must be divisible by block size for 1D.")
        
        num_blocks = length // block_size
        agents_per_block = block_size
        total_blocks = num_blocks

        # --- Vectorized Block Processing for 1D ---
        strategies = agent_strategies.reshape(3, length)
        blocks = strategies.reshape(3, num_blocks, block_size)
        blocks = blocks.transpose(1, 0, 2) # (num_blocks, 3, block_size)

    else: # 2D
        rows, cols = grid_dim
        if rows % block_size != 0 or cols % block_size != 0:
            raise ValueError("Grid dimensions must be divisible by block size.")

        num_block_rows = rows // block_size
        num_block_cols = cols // block_size
        agents_per_block = block_size * block_size
        total_blocks = num_block_rows * num_block_cols

        # --- Vectorized Block Processing for 2D ---
        strategies = agent_strategies.reshape(3, rows, cols)
        blocks = strategies.reshape(3, num_block_rows, block_size, num_block_cols, block_size)
        blocks = blocks.transpose(1, 3, 0, 2, 4)
        blocks = blocks.reshape(num_block_rows * num_block_cols, 3, agents_per_block)

    # 2. Convert all blocks from one-hot to strategy labels (0, 1, 2)
    # Result shape: (num_blocks, agents_per_block)
    strategy_labels = cp.argmax(blocks, axis=1)

    # 3. Convert base-3 labels to a unique integer ID for each block
    # This is a vectorized dot product for every block simultaneously
    power_of_3 = 3 ** cp.arange(agents_per_block - 1, -1, -1, dtype=cp.int64)
    block_integers = strategy_labels @ power_of_3

    # --- Entropy Calculation (as before) ---

    # 4. Count occurrences of each unique block configuration
    max_block_val = 3**agents_per_block
    counts = cp.bincount(block_integers, minlength=max_block_val)
    
    # 5. Calculate probabilities and entropy
    # The sum of counts is simply the total number of blocks
    probabilities = counts[counts > 0] / total_blocks
    entropy = -cp.sum(probabilities * cp.log(probabilities))
    
    return float(entropy)

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
