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
    center_index = (grid_dim[0] // 2) * grid_dim[1] + (grid_dim[1] // 2) if grid_dim else N // 2
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
