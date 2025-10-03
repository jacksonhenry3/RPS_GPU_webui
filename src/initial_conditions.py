import cupy as cp

def _labels_to_one_hot(labels, N):
    return cp.eye(3, dtype=int)[labels].T

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
        
        # Paper cross (horizontal + vertical lines)
        for col in range(cols):
            idx = center_row * cols + col
            labels[idx] = 1  # Paper horizontal line
        for row in range(rows):
            idx = row * cols + center_col
            labels[idx] = 1  # Paper vertical line
            
        # Rock at exact center
        center_idx = center_row * cols + center_col
        labels[center_idx] = 0  # Rock
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
        
        # Paper in all four corners
        for r in range(corner_size):
            for c in range(corner_size):
                # Top-left
                if r < rows and c < cols:
                    labels[r * cols + c] = 1
                # Top-right  
                if r < rows and (cols - 1 - c) >= 0:
                    labels[r * cols + (cols - 1 - c)] = 1
                # Bottom-left
                if (rows - 1 - r) >= 0 and c < cols:
                    labels[(rows - 1 - r) * cols + c] = 1
                # Bottom-right
                if (rows - 1 - r) >= 0 and (cols - 1 - c) >= 0:
                    labels[(rows - 1 - r) * cols + (cols - 1 - c)] = 1
        
        # Scissors in center
        center_row = rows // 2
        center_col = cols // 2
        for r in range(center_row - center_size//2, center_row + center_size//2 + 1):
            for c in range(center_col - center_size//2, center_col + center_size//2 + 1):
                if 0 <= r < rows and 0 <= c < cols:
                    labels[r * cols + c] = 2
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
    labels = cp.zeros(N, dtype=int)
    
    if grid_dim and len(grid_dim) == 2:
        rows, cols = grid_dim
        
        for r in range(rows):
            for c in range(cols):
                idx = r * cols + c
                # Create diagonal pattern
                pattern_val = (r + c) % 3
                labels[idx] = pattern_val
    else:
        # 1D - repeating pattern
        for i in range(N):
            labels[i] = i % 3
    
    return _labels_to_one_hot(labels, N)

def _periodic_stripes(N, grid_dim):
    """Symmetric periodic stripes: RRR-PPP-SSS pattern"""
    labels = cp.zeros(N, dtype=int)
    stripe_width = 3
    
    if grid_dim and len(grid_dim) == 2:
        # 2D - horizontal stripes, symmetric about center
        rows, cols = grid_dim
        center_row = rows // 2
        
        for r in range(rows):
            # Distance from center row
            dist_from_center = abs(r - center_row)
            # Which stripe group (0=Rock, 1=Paper, 2=Scissors)
            stripe_group = (dist_from_center // stripe_width) % 3
            
            for c in range(cols):
                idx = r * cols + c
                labels[idx] = stripe_group
    else:
        # 1D - symmetric stripes from center
        center = N // 2
        
        for i in range(N):
            # Distance from center
            dist_from_center = abs(i - center)
            # Which stripe group
            stripe_group = (dist_from_center // stripe_width) % 3
            labels[i] = stripe_group
    
    return _labels_to_one_hot(labels, N)

def _symmetric_gradient(N, grid_dim):
    """Symmetric gradient: Rock center → Paper middle → Scissors outer"""
    labels = cp.full(N, 2, dtype=int)  # Scissors background
    
    if grid_dim and len(grid_dim) == 2:
        rows, cols = grid_dim
        center_row = rows / 2
        center_col = cols / 2
        max_distance = cp.sqrt((rows/2)**2 + (cols/2)**2)
        
        for r in range(rows):
            for c in range(cols):
                idx = r * cols + c
                distance = cp.sqrt((r - center_row)**2 + (c - center_col)**2)
                
                if distance <= max_distance * 0.33:
                    labels[idx] = 0  # Rock at center
                elif distance <= max_distance * 0.66:
                    labels[idx] = 1  # Paper in middle
                # Scissors on outside
    else:
        # 1D - symmetric gradient from center
        center = N // 2
        max_dist = N // 2
        
        for i in range(N):
            distance = abs(i - center)
            
            if distance <= max_dist * 0.33:
                labels[i] = 0  # Rock
            elif distance <= max_dist * 0.66:
                labels[i] = 1  # Paper
            # Scissors on edges
    
    return _labels_to_one_hot(labels, N)
    
def _single_invader_bank(N, grid_dim, bank_value):
    """
    Creates bank values with all agents at 0 except the center agent.
    
    Args:
        N: Number of agents
        grid_dim: Grid dimensions (rows, cols) or None for 1D
        bank_value: The value to assign to the center agent (must be float)
    
    Returns:
        CuPy array of bank values
    """
    import cupy as cp
    
    # Initialize all agents to -1000
    bank_values = cp.full(N, -1000.0, dtype=float)
    
    # Calculate center index
    if grid_dim:
        center_index = (grid_dim[0]//2) * grid_dim[1] + (grid_dim[1]//2)
    else:
        center_index = N // 2
    
    # Set center agent to the specified bank value
    bank_values[center_index] = float(bank_value)
    
    return bank_values
