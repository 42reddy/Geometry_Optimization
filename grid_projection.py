"""
Projects GATr's per-node point-cloud features onto a fixed, uniformly
spaced structured grid — the format FNO requires, since its FFT-based
spectral convolutions need regular spacing and the point cloud GATr
consumes is irregular and boundary-density-adaptive.

Each grid cell cross-attends only to its k nearest point-cloud nodes (a
bipartite k-NN graph from graph_utils.build_bipartite_knn), so this stays
cheap regardless of point-cloud or grid size.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from EquiLinear import EquivariantLinear
from torch_scatter import scatter_softmax, scatter_add
from grid_stretch import AxisStretch

METRIC_INDICES = [0, 2, 3, 4, 8, 9, 10, 14, 15]


class GridQueryEmbed(nn.Module):
    """
    Embeds fixed grid-cell positions into the same multivector/scalar
    representation space as GATrEncoder's point-cloud nodes, so grid cells
    can act as attention queries against the point cloud's keys/values.

    Grid cells carry no boundary-condition scalars of their own (unlike
    point-cloud nodes, which have SDF/normals/freestream velocity) — only
    a position. The scalar side of a query is a single learned bias vector,
    shared across all grid cells.
    """

    def __init__(self, mv_channels, scalar_channels):
        super().__init__()
        self.mv_input_proj = EquivariantLinear(1, mv_channels)
        self.scalar_bias = nn.Parameter(torch.zeros(scalar_channels))

    def forward(self, grid_mvs):
        """
        grid_mvs: (M, 16) position-only multivectors for grid cells
        Returns : x_mv (M, C, 16), x_s (M, scalar_channels)
        """
        x_mv = grid_mvs.unsqueeze(1).unsqueeze(0)     # (1, M, 1, 16)
        x_mv = self.mv_input_proj(x_mv).squeeze(0)    # (M, C, 16)
        x_s = self.scalar_bias.unsqueeze(0).expand(grid_mvs.shape[0], -1)   # (M, sc)
        return x_mv, x_s


class GridProjector(nn.Module):
    """
    Bipartite equivariant cross-attention: fixed grid cells (queries)
    attend to their local neighborhood of point-cloud nodes (keys/values)
    produced by GATrEncoder, and pool them into one feature per grid cell.
    """

    def __init__(self, mv_channels, scalar_channels, n_heads=2):
        super().__init__()
        self.n_heads = n_heads
        self.mv_channels = mv_channels
        self.sc_channels = scalar_channels
        self.head_mv_dim = mv_channels // n_heads
        self.head_sc_dim = scalar_channels // n_heads

        self.query_embed = GridQueryEmbed(mv_channels, scalar_channels)

        self.q_mv = EquivariantLinear(mv_channels, mv_channels)
        self.k_mv = EquivariantLinear(mv_channels, mv_channels)
        self.v_mv = EquivariantLinear(mv_channels, mv_channels)
        self.q_s = nn.Linear(scalar_channels, scalar_channels)
        self.k_s = nn.Linear(scalar_channels, scalar_channels)
        self.v_s = nn.Linear(scalar_channels, scalar_channels)
        self.out_mv = EquivariantLinear(mv_channels, mv_channels)
        self.out_s = nn.Linear(scalar_channels, scalar_channels)

    def forward(self, point_mv, point_s, grid_mvs, edge_index):
        """
        point_mv   : (N, C, 16)  per-node point-cloud features from GATrEncoder
        point_s    : (N, sc)     per-node point-cloud scalar features
        grid_mvs   : (M, 16)     position-only multivectors for grid cells
        edge_index : (2, E)      bipartite k-NN graph from build_bipartite_knn;
                                  edge_index[0] = point-cloud index (src),
                                  edge_index[1] = grid index (dst)

        Returns:
            grid_mv_out : (M, C, 16)
            grid_s_out  : (M, sc)
        """
        M = grid_mvs.shape[0]
        H = self.n_heads

        query_mv, query_s = self.query_embed(grid_mvs)   # (M, C, 16), (M, sc)

        Q_mv = self.q_mv(query_mv.unsqueeze(0)).squeeze(0)   # (M, C, 16)
        Q_s = self.q_s(query_s)                              # (M, sc)
        K_mv = self.k_mv(point_mv.unsqueeze(0)).squeeze(0)   # (N, C, 16)
        K_s = self.k_s(point_s)
        V_mv = self.v_mv(point_mv.unsqueeze(0)).squeeze(0)
        V_s = self.v_s(point_s)

        src, dst = edge_index[0], edge_index[1]   # src: point idx, dst: grid idx

        Q_mv = Q_mv.reshape(M, H, self.head_mv_dim, 16)
        K_mv = K_mv.reshape(-1, H, self.head_mv_dim, 16)
        V_mv = V_mv.reshape(-1, H, self.head_mv_dim, 16)
        Q_s = Q_s.reshape(M, H, self.head_sc_dim)
        K_s = K_s.reshape(-1, H, self.head_sc_dim)
        V_s = V_s.reshape(-1, H, self.head_sc_dim)

        mv_scale = (9 * self.head_mv_dim) ** 0.5
        sc_scale = self.head_sc_dim ** 0.5

        # (E, H) dot product between each grid query and its assigned point-cloud neighbors
        mv_scores = (
            Q_mv[dst][..., METRIC_INDICES] *
            K_mv[src][..., METRIC_INDICES]
        ).sum(dim=(-1, -2)) / mv_scale

        sc_scores = (Q_s[dst] * K_s[src]).sum(dim=-1) / sc_scale

        scores = mv_scores + sc_scores
        weights = scatter_softmax(scores, dst, dim=0)   # softmax over each grid cell's neighbors

        weighted_mv = weights.unsqueeze(-1).unsqueeze(-1) * V_mv[src]   # (E, H, head_mv, 16)
        weighted_s = weights.unsqueeze(-1) * V_s[src]                    # (E, H, head_sc)

        out_mv = scatter_add(
            weighted_mv.reshape(-1, H * self.head_mv_dim * 16),
            dst, dim=0, dim_size=M
        ).reshape(M, self.mv_channels, 16)

        out_s = scatter_add(
            weighted_s.reshape(-1, H * self.head_sc_dim),
            dst, dim=0, dim_size=M
        ).reshape(M, self.sc_channels)

        grid_mv_out = self.out_mv(out_mv.unsqueeze(0)).squeeze(0)
        grid_s_out = self.out_s(out_s)

        return grid_mv_out, grid_s_out


def reshape_grid_features(grid_mv: torch.Tensor, grid_s: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """
    Flatten a grid's multivector + scalar features into a single per-cell
    feature vector and reshape to FNO's expected (C, H, W) layout.

    grid_mv : (M, C, 16)  M = H*W
    grid_s  : (M, sc)
    Returns : (C*16 + sc, H, W)
    """
    M = grid_mv.shape[0]
    flat = torch.cat([grid_mv.reshape(M, -1), grid_s], dim=-1)   # (M, C*16 + sc)
    grid = flat.reshape(H, W, -1)                                  # (H, W, feat_dim)
    return grid.permute(2, 0, 1)                                   # (feat_dim, H, W)


class GridDecoder(nn.Module):
    """
    Inverse of the point -> grid projection: samples FNO's structured-grid
    output field at arbitrary continuous point coordinates via
    differentiable bilinear interpolation.

    This is what makes the grid a purely internal representation — FNO's
    output lives on a fixed (H, W) grid, but ground-truth targets
    (velocity, pressure) are defined at point-cloud locations. Because the
    interpolation is continuous, it can be queried at ANY coordinates,
    including the full original ~180k-point mesh, even though GATr/FNO only
    ever computed on the downsampled training point cloud — input
    compression and output supervision resolution are decoupled.

    The grid is stretched (AxisStretch, matching build_structured_grid) so
    the index -> physical mapping is nonlinear; query coordinates must go
    through the SAME per-axis inverse mapping to land on the right
    normalized grid_sample position, not a plain linear rescale.
    """

    def __init__(self, bounds, resolution, stretch_center=(0.5, 0.0), stretch_gamma=3.0):
        super().__init__()
        (x_min, x_max), (y_min, y_max) = bounds
        H, W = resolution
        cx, cy = stretch_center
        self.x_stretch = AxisStretch(x_min, x_max, cx, stretch_gamma, W)
        self.y_stretch = AxisStretch(y_min, y_max, cy, stretch_gamma, H)

    def forward(self, grid_field, query_coords):
        """
        grid_field   : (B, C, H, W)  FNO output
        query_coords : (B, N, 2)     (x, y) points to evaluate the field at
        Returns      : (B, N, C)     interpolated field values at each point
        """
        x = self.x_stretch.to_norm(query_coords[..., 0])
        y = self.y_stretch.to_norm(query_coords[..., 1])
        norm_coords = torch.stack([x, y], dim=-1).unsqueeze(1)   # (B, 1, N, 2)

        sampled = F.grid_sample(
            grid_field, norm_coords, mode='bilinear',
            padding_mode='border', align_corners=True
        )   # (B, C, 1, N)

        return sampled.squeeze(2).permute(0, 2, 1)   # (B, N, C)
