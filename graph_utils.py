"""
Graph construction utilities for GATr's point-cloud attention and the
point-cloud -> structured-grid projection step.
"""

import torch
from torch_geometric.nn import knn_graph, knn


def build_knn_graph(coords: torch.Tensor, k: int = 16, loop: bool = False) -> torch.Tensor:
    """
    Build a k-NN graph over point-cloud coordinates, for use as GATrEncoder's
    edge_index. Required at realistic point counts (thousands of nodes) —
    EquivariantAttention's dense fallback (edge_index=None) is O(N^2).

    coords : (N, 2) or (N, 3) node positions
    Returns: edge_index (2, E) — edge_index[0] = src (neighbor), edge_index[1] = dst (query node)
    """
    return knn_graph(coords, k=k, loop=loop)


def build_bipartite_knn(point_coords: torch.Tensor, grid_coords: torch.Tensor, k: int = 8) -> torch.Tensor:
    """
    For each grid cell, find its k nearest point-cloud nodes — used by
    GridProjector to restrict cross-attention to a local neighborhood
    instead of attending over the full point cloud.

    point_coords : (N, 2) point-cloud node coordinates
    grid_coords  : (M, 2) structured grid cell coordinates
    Returns      : edge_index (2, E) with edge_index[0] = point-cloud index (src),
                                           edge_index[1] = grid index (dst)
    """
    assign_index = knn(point_coords, grid_coords, k=k)
    grid_idx, point_idx = assign_index[0], assign_index[1]
    return torch.stack([point_idx, grid_idx], dim=0)


def build_structured_grid(
    bounds: tuple = ((-3.0, 3.0), (-3.0, 3.0)),
    resolution: tuple = (64, 64),
    device=None,
) -> tuple:
    """
    Build a uniform Cartesian grid — the regular spacing FNO's FFT-based
    spectral convolutions require. Unlike the boundary-aware point-cloud
    downsampling used for GATr's input, this grid is NOT density-adaptive;
    density-awareness happens upstream, via GridProjector reading from a
    denser point-cloud region when a grid cell sits near the boundary.

    bounds     : ((x_min, x_max), (y_min, y_max)) — should cover the
                 normalized coordinate range of your prepared dataset
    resolution : (H, W) grid resolution

    Returns:
        grid_coords : (H*W, 2) flattened grid cell coordinates
        H, W        : grid resolution, for reshaping features back to (H, W, ...)
    """
    (x_min, x_max), (y_min, y_max) = bounds
    H, W = resolution
    xs = torch.linspace(x_min, x_max, W, device=device)
    ys = torch.linspace(y_min, y_max, H, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')   # (H, W)
    grid_coords = torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2)   # (H*W, 2)
    return grid_coords, H, W
