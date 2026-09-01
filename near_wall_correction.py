"""
Local correction branch added on top of FNO's global grid prediction.

FNO's SpectralConv2d keeps only the lowest `modes1 x modes2` Fourier modes
(see fno.py) — that's the whole point of a spectral operator (resolution-
invariant, cheap), but it also means the global field is a truncated
Fourier series and therefore fundamentally band-limited: it cannot
represent a gradient sharper than its mode cutoff allows, no matter how
well GridProjector's grid is stretched or how well the input point cloud
is sampled (Gibbs-phenomenon territory). The airfoil boundary layer is
exactly this kind of sharp feature — real, physical, and steeper than a
handful of global Fourier modes can follow.

Rather than asking one spectral operator to cover both the smooth far
field and the steep near-wall region, this module adds a small local
correction that only activates near the wall. It reuses GATr's per-node
point-cloud features directly (no separate encoder), pulling local context
into each query point via the same bipartite k-NN cross-attention
GridProjector uses for the fixed grid — grid_projection.GridProjector is
generic in its "query positions" argument, so it's reused unchanged here
with query_coords (arbitrary points) in place of a fixed grid.
"""

import torch
import torch.nn as nn

from grid_projection import GridProjector
from graph_utils import build_bipartite_knn
from GA_encoder import encode_points_batch_torch


class NearWallCorrector(nn.Module):
    """
    correction(query) = MLP(local cross-attention over nearby GATr point
    features) * gate(|sdf(query)|)

    The gate is a smooth exp(-|sdf|/wall_scale), ~1 right at the wall and
    ~0 by a few multiples of wall_scale — it guarantees the correction can
    only ever adjust points where the global field is known to struggle,
    and never perturbs the far-field prediction that's already good. The
    output head is zero-initialized so training starts exactly at the
    (already reasonable) global FNO prediction and only learns a residual.
    """

    def __init__(self, mv_channels, scalar_channels, n_outputs=3, k=8,
                 hidden=32, wall_scale=0.05, n_heads=2):
        super().__init__()
        self.k = k
        self.wall_scale = wall_scale
        self.local_projector = GridProjector(mv_channels, scalar_channels, n_heads)

        feat_dim = mv_channels * 16 + scalar_channels
        self.head = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_outputs),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, point_mv, point_s, point_coords, query_coords, query_sdf):
        """
        point_mv, point_s : GATr's per-node features at the ENCODER's input
                             points (point_coords) — cached from
                             FlowFieldPipeline.encode_to_grid, not
                             recomputed here.
        query_coords       : (Q, 2) points to correct
        query_sdf           : (Q,) signed distance at each query point

        Returns: (Q, n_outputs) correction to ADD to GridDecoder's output.
        """
        query_mvs = encode_points_batch_torch(query_coords)
        edge_index = build_bipartite_knn(point_coords, query_coords, k=self.k)
        out_mv, out_s = self.local_projector(point_mv, point_s, query_mvs, edge_index)

        feat = torch.cat([out_mv.reshape(out_mv.shape[0], -1), out_s], dim=-1)
        correction = self.head(feat)

        gate = torch.exp(-(query_sdf.abs() / self.wall_scale)).unsqueeze(-1)
        return correction * gate
