import cupy as cp

# Must match algorithm.precision / networks.precision so agent_strategies
# stays float32 through the adjacency-matrix matmuls instead of silently
# upcasting back to float64 via int/float type promotion.
precision = cp.float32

def _labels_to_one_hot(labels, N):
    return cp.eye(3, dtype=precision)[labels].T

def _random_strategies(N):
    return _labels_to_one_hot(cp.random.randint(0, 3, size=N), N)

def _all_one_strategy(N, strategy_index=0):
    return _labels_to_one_hot(cp.full(N, strategy_index, dtype=int), N)

def _split_strategies(N):
    labels = cp.zeros(N, dtype=int)
    labels[N // 2:] = 1
    return _labels_to_one_hot(labels, N)

def _single_invader(N, grid_dim):
    labels = cp.zeros(N, dtype=int)
    center_index = (grid_dim[0]//2) * grid_dim[1] + (grid_dim[1]//2) if grid_dim else N//2
    labels[center_index] = 1
    return _labels_to_one_hot(labels, N)

def _vertical_stripes(N, grid_dim, num_stripes=3):
    width = grid_dim[1] if grid_dim else N
    indices = cp.arange(N) % width
    stripe_width = width // num_stripes
    labels = cp.zeros(N, dtype=int)
    for i in range(num_stripes):
        mask = (indices >= i * stripe_width) & (indices < (i + 1) * stripe_width)
        labels[mask] = i % 3
    return _labels_to_one_hot(labels, N)

def _pie_slices(N, grid_dim):
    if not grid_dim: return _random_strategies(N)
    rows, cols = grid_dim
    y, x = cp.meshgrid(cp.arange(rows), cp.arange(cols))
    angles = cp.arctan2(y - rows / 2, x - cols / 2) * 180 / cp.pi
    labels = cp.zeros((rows, cols), dtype=int)
    labels[(angles >= 60)] = 1
    labels[(angles <= -60)] = 2
    return _labels_to_one_hot(labels.flatten(), N)

def _double_invader(N, grid_dim):
    labels = cp.zeros(N, dtype=int)  # Start with all Rock (strategy 0)

    if grid_dim:
        # For 2D grids, place two Paper agents symmetrically
        rows, cols = grid_dim
        center_row = rows // 2
        # Place one at 1/4 and one at 3/4 of the width
        col1 = cols // 4
        col2 = 3 * cols // 4

        index1 = center_row * cols + col1
        index2 = center_row * cols + col2

        labels[index1] = 1  # Paper
        labels[index2] = 1  # Paper
    else:
        # For 1D, place two Paper agents at 1/3 and 2/3 positions
        pos1 = N // 3
        pos2 = 2 * N // 3
        labels[pos1] = 1  # Paper
        labels[pos2] = 1  # Paper

    return _labels_to_one_hot(labels, N)

def _cross_invasion(N, grid_dim):
    """Scissors background with Paper cross and Rock at center"""
    labels = cp.full(N, 2, dtype=int)  # Scissors background
    
    if grid_dim and len(grid_dim) == 2:
        rows, cols = grid_dim
        center_row = rows // 2
        center_col = cols // 2

        rr, cc = cp.meshgrid(cp.arange(rows), cp.arange(cols), indexing='ij')
        labels2d = cp.full((rows, cols), 2, dtype=int)
        labels2d[(rr == center_row) | (cc == center_col)] = 1  # Paper cross
        labels2d[(rr == center_row) & (cc == center_col)] = 0  # Rock at exact center
        labels = labels2d.flatten()
    else:
        # 1D fallback - Paper stripe with Rock center
        center = N // 2
        stripe_half = N // 8
        labels[center-stripe_half:center+stripe_half+1] = 1  # Paper
        labels[center] = 0  # Rock center
    
    return _labels_to_one_hot(labels, N)

def _corner_siege(N, grid_dim):
    """Rock background with Paper corners and Scissors center"""
    labels = cp.zeros(N, dtype=int)  # Rock background
    
    if grid_dim and len(grid_dim) == 2:
        rows, cols = grid_dim
        corner_size = max(2, min(rows, cols) // 8)
        center_size = max(3, min(rows, cols) // 6)

        rr, cc = cp.meshgrid(cp.arange(rows), cp.arange(cols), indexing='ij')
        labels2d = cp.zeros((rows, cols), dtype=int)

        # Paper in all four corners
        in_corner_rows = (rr < corner_size) | (rr >= rows - corner_size)
        in_corner_cols = (cc < corner_size) | (cc >= cols - corner_size)
        labels2d[in_corner_rows & in_corner_cols] = 1

        # Scissors in center
        center_row = rows // 2
        center_col = cols // 2
        half = center_size // 2
        in_center = (cp.abs(rr - center_row) <= half) & (cp.abs(cc - center_col) <= half)
        labels2d[in_center] = 2

        labels = labels2d.flatten()
    else:
        # 1D fallback
        corner_size = N // 8
        center_size = N // 6
        center = N // 2
        
        labels[:corner_size] = 1  # Paper left
        labels[-corner_size:] = 1  # Paper right
        labels[center-center_size//2:center+center_size//2+1] = 2  # Scissors center
    
    return _labels_to_one_hot(labels, N)

def _diamond_lattice(N, grid_dim):
    """Alternating diamond/checkerboard pattern of all three strategies"""
    if grid_dim and len(grid_dim) == 2:
        rows, cols = grid_dim
        rr, cc = cp.meshgrid(cp.arange(rows), cp.arange(cols), indexing='ij')
        labels = ((rr + cc) % 3).flatten()
    else:
        # 1D - repeating pattern
        labels = cp.arange(N) % 3

    return _labels_to_one_hot(labels, N)

def _periodic_stripes(N, grid_dim):
    """Symmetric periodic stripes: RRR-PPP-SSS pattern"""
    stripe_width = 3

    if grid_dim and len(grid_dim) == 2:
        # 2D - horizontal stripes, symmetric about center
        rows, cols = grid_dim
        center_row = rows // 2

        dist_from_center = cp.abs(cp.arange(rows) - center_row)
        stripe_group = (dist_from_center // stripe_width) % 3
        labels = cp.broadcast_to(stripe_group[:, None], (rows, cols)).flatten()
    else:
        # 1D - symmetric stripes from center
        center = N // 2
        dist_from_center = cp.abs(cp.arange(N) - center)
        labels = (dist_from_center // stripe_width) % 3

    return _labels_to_one_hot(labels, N)

def _symmetric_gradient(N, grid_dim):
    """Symmetric gradient: Rock center → Paper middle → Scissors outer"""
    if grid_dim and len(grid_dim) == 2:
        rows, cols = grid_dim
        center_row = rows / 2
        center_col = cols / 2
        max_distance = cp.sqrt((rows/2)**2 + (cols/2)**2)

        rr, cc = cp.meshgrid(cp.arange(rows), cp.arange(cols), indexing='ij')
        distance = cp.sqrt((rr - center_row)**2 + (cc - center_col)**2)

        labels2d = cp.full((rows, cols), 2, dtype=int)  # Scissors on outside
        labels2d[distance <= max_distance * 0.66] = 1  # Paper in middle
        labels2d[distance <= max_distance * 0.33] = 0  # Rock at center
        labels = labels2d.flatten()
    else:
        # 1D - symmetric gradient from center
        center = N // 2
        max_dist = N // 2
        distance = cp.abs(cp.arange(N) - center)

        labels = cp.full(N, 2, dtype=int)  # Scissors on edges
        labels[distance <= max_dist * 0.66] = 1  # Paper
        labels[distance <= max_dist * 0.33] = 0  # Rock

    return _labels_to_one_hot(labels, N)
    
