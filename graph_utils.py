"""
Graph construction utilities for GATr's point-cloud attention and the
point-cloud -> structured-grid projection step.
"""

import torch
from torch_geometric.nn import knn_graph, knn

from grid_stretch import AxisStretch


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
    stretch_center: tuple = (0.5, 0.0),
    stretch_gamma: float = 3.0,
    device=None,
) -> tuple:
    """
    Build a Cartesian grid on a REGULAR (H, W) index lattice — the uniform
    indexing FNO's FFT-based spectral convolutions require — but with the
    index -> physical coordinate mapping stretched (via AxisStretch) so grid
    density is highest near `stretch_center` (the airfoil) and falls off
    towards the domain edges, instead of uniform physical spacing. A real
    AirfRANS mesh concentrates most of its points in a thin near-wall
    boundary layer; a uniform grid can't represent that layer at all once
    its cells are wider than the layer itself, no matter how the point cloud
    fed into GridProjector is sampled. This fixes the representation, not
    just the sampling. Set stretch_gamma=0 to recover a uniform grid.

    bounds         : ((x_min, x_max), (y_min, y_max)) — should cover the
                     normalized coordinate range of your prepared dataset
    resolution     : (H, W) requested grid resolution (may be adjusted by a
                     few cells due to rounding in the stretch split)
    stretch_center : (x_center, y_center) — where grid density is highest;
                     default (0.5, 0.0) is mid-chord, on the chord line
    stretch_gamma  : clustering strength (0 = uniform, 3-5 = strong)

    Returns:
        grid_coords : (H*W, 2) flattened grid cell coordinates
        H, W        : actual grid resolution, for reshaping features back to (H, W, ...)
    """
    (x_min, x_max), (y_min, y_max) = bounds
    H, W = resolution
    cx, cy = stretch_center

    x_stretch = AxisStretch(x_min, x_max, cx, stretch_gamma, W)
    y_stretch = AxisStretch(y_min, y_max, cy, stretch_gamma, H)
    xs = x_stretch.physical_coords(device=device)
    ys = y_stretch.physical_coords(device=device)
    W, H = x_stretch.n, y_stretch.n

    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')   # (H, W)
    grid_coords = torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2)   # (H*W, 2)
    return grid_coords, H, W
