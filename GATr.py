import torch
import torch.nn as nn
from EquiLinear import EquivariantLinear
from EquiLayerNorm import EquivariantLayerNorm
from ScalarGELU import GatedGELU
from Attention import EquivariantAttention
from GeoBiLinear import GeoBilinear


class GATrBlock(nn.Module):
    # unchanged — domain-agnostic equivariant transformer block
    def __init__(self, mv_channels, scalar_channels, n_heads):
        super().__init__()
        self.norm1 = EquivariantLayerNorm(mv_channels, scalar_channels)
        self.attn = EquivariantAttention(mv_channels, scalar_channels, n_heads)
        self.norm2 = EquivariantLayerNorm(mv_channels, scalar_channels)
        self.mlp_in = EquivariantLinear(mv_channels, mv_channels)
        self.geo_bi = GeoBilinear()
        self.gelu = GatedGELU()
        self.mlp_out = EquivariantLinear(mv_channels * 2, mv_channels)
        self.s_mlp_in = nn.Linear(scalar_channels, scalar_channels)
        self.s_mlp_out = nn.Linear(scalar_channels, scalar_channels)

    def forward(self, x_mv, x_s, edge_index=None):
        r_mv, r_s = x_mv, x_s
        x_mv, x_s = self.norm1(x_mv, x_s)
        x_mv, x_s = self.attn(x_mv, x_s, edge_index)
        x_mv, x_s = x_mv + r_mv, x_s + r_s

        r_mv, r_s = x_mv, x_s
        x_mv, x_s = self.norm2(x_mv, x_s)
        x_proj = self.mlp_in(x_mv)
        y_proj = self.mlp_in(x_mv)
        z_proj = self.mlp_in(x_mv)
        x_mv = self.geo_bi(x_proj, y_proj, z_proj)
        x_mv, x_s = self.gelu(x_mv, x_s)
        x_mv = self.mlp_out(x_mv)
        x_s = torch.relu(self.s_mlp_in(x_s))
        x_s = self.s_mlp_out(x_s)
        x_mv, x_s = x_mv + r_mv, x_s + r_s
        return x_mv, x_s


class GATrEncoder(nn.Module):
    """
    Encodes an unstructured point cloud (airfoil geometry + boundary
    conditions) into per-node equivariant features.

    Output stays per-node — (N, mv_channels, 16) and (N, scalar_channels) —
    with NO pooling to a single graph-level vector. A separate grid
    projection module (not in this file) is responsible for mapping these
    per-node features onto FNO's fixed structured grid; GATr's job here is
    only to produce rich per-node geometric features.

    edge_index is effectively required at realistic point counts (thousands
    of nodes) — EquivariantAttention's dense fallback is O(N^2) and will not
    scale. Build a k-NN or radius graph from node coordinates upstream and
    pass it in.
    """

    def __init__(self, input_scalar_dim=5, mv_channels=4, scalar_channels=8, n_heads=2, n_layers=2):
        super().__init__()
        self.input_scalar_dim = input_scalar_dim
        self.mv_channels = mv_channels
        self.scalar_channels = scalar_channels
        self.n_heads = n_heads

        # input projections
        self.mv_input_proj = EquivariantLinear(1, mv_channels)
        self.s_input_proj = nn.Linear(input_scalar_dim, scalar_channels)
        self.s_to_mv_grade0 = nn.Linear(input_scalar_dim, mv_channels)

        # transformer blocks
        self.blocks = nn.ModuleList([
            GATrBlock(mv_channels, scalar_channels, n_heads)
            for _ in range(n_layers)
        ])

    def forward(self, node_mvs, node_scalars, edge_index=None):
        """
        node_mvs     : (N, 16)                  per-node position multivectors
        node_scalars : (N, input_scalar_dim)     SDF, normals, freestream velocity, ...
        edge_index   : (2, E)                    k-NN / radius graph over the point cloud

        Returns:
            x_mv : (N, mv_channels, 16)      per-node multivector features
            x_s  : (N, scalar_channels)      per-node scalar features
        """
        # embed
        x_mv = node_mvs.unsqueeze(1).unsqueeze(0)   # (1, N, 1, 16)
        x_mv = self.mv_input_proj(x_mv)             # (1, N, C, 16)
        grade0_mv = torch.zeros_like(x_mv)
        grade0_mv[0, :, :, 0] = self.s_to_mv_grade0(node_scalars)
        x_mv = x_mv + grade0_mv
        x_s = self.s_input_proj(node_scalars).unsqueeze(0)   # (1, N, sc)

        # blocks
        for block in self.blocks:
            x_mv, x_s = block(x_mv, x_s, edge_index)

        return x_mv.squeeze(0), x_s.squeeze(0)   # (N, C, 16), (N, sc)


class GATrModel(nn.Module):
    """
    Geometry/BC encoder only.

    Produces per-node equivariant features on the original unstructured
    point cloud. It does NOT output a structured grid itself — a separate
    grid-projection module consumes (x_mv, x_s) plus node coordinates and
    interpolates/scatters them onto FNO's fixed grid.
    """

    def __init__(self, input_scalar_dim=5, mv_channels=4, scalar_channels=8, n_heads=2, n_layers=2):
        super().__init__()
        self.encoder = GATrEncoder(input_scalar_dim, mv_channels, scalar_channels, n_heads, n_layers)

    def forward(self, node_mvs, node_scalars, edge_index=None):
        return self.encoder(node_mvs, node_scalars, edge_index)

    def save_encoder(self, path='pretrained_encoder.pt'):
        torch.save(self.encoder.state_dict(), path)
        print(f"Encoder saved to {path}")

    def load_encoder(self, path='pretrained_encoder.pt'):
        self.encoder.load_state_dict(torch.load(path))
        print(f"Encoder loaded from {path}")
