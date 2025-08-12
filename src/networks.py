import cupy as cp
import cupyx.scipy.sparse as cpx_sparse
import numpy as np
import networkx as nx
import time
from app_utils import log_network

precision = cp.float64

def create(network_type, num_agents_param, kymo_aspect=3.0):
    """
    Factory function to create the specified network.
    """
    log_network(f"Creating '{network_type}' graph with size param {num_agents_param}...")
    total_start_time = time.time()

    if 'ring_1d' in network_type:
        periodic = 'periodic' in network_type
        result = _create_ring_1d(num_agents_param, periodic=periodic, aspect_ratio=kymo_aspect)
    elif 'grid_2d' in network_type:
        periodic = 'periodic' in network_type
        moore = 'moore' in network_type
        result = _create_2d_grid(grid_dim_param=num_agents_param, periodic=periodic, moore=moore)
    else:
        raise ValueError(f"Unknown network type: {network_type}")

    total_end_time = time.time()
    log_network(f"Graph creation complete. Total time: {total_end_time - total_start_time:.4f}s")
    return result

def _create_ring_1d(num_agents, periodic=True, aspect_ratio=3.0):
    """Creates a 1D line or ring lattice."""
    width = num_agents
    height = int(num_agents * aspect_ratio)
    
    # Connections to the agent on the right (i+1) and left (i-1)
    diagonals = [1, -1]
    if periodic:
        # Add wrap-around connections for a periodic ring
        diagonals.extend([num_agents - 1, -(num_agents - 1)])
        
    adj_matrix = sum(cpx_sparse.eye(num_agents, k=k, format='csr', dtype=precision) for k in diagonals)
    return adj_matrix, None, width, height, num_agents

def _create_2d_grid(grid_dim_param: int, periodic: bool = True, moore: bool = False):
    """
    Creates a 2D grid adjacency matrix using sparse Kronecker products.
    """
    N = grid_dim_param
    num_agents = N * N
    
    # Create the sparse 1D adjacency matrix
    diagonals = [1, -1]
    if periodic:
        diagonals.extend([N - 1, -(N - 1)])
    A_1D_sparse = sum(cpx_sparse.eye(N, k=k, format='csr', dtype=precision) for k in diagonals)
    
    I_N_sparse = cpx_sparse.eye(N, dtype=precision, format='csr')
    
    adj_matrix_cupy = cpx_sparse.kron(A_1D_sparse, I_N_sparse) + cpx_sparse.kron(I_N_sparse, A_1D_sparse)
    
    if moore:
        adj_matrix_cupy += cpx_sparse.kron(A_1D_sparse, A_1D_sparse)
    
    adj_matrix_cupy = adj_matrix_cupy.tocsr()
    
    width = height = grid_dim_param
    return adj_matrix_cupy, (width, height), width, height, num_agents

