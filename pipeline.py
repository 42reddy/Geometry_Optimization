"""
Assembles the full point-cloud -> flow-field model:

  point cloud --[GATrEncoder]--> per-node features
              --[GridProjector]--> structured grid
              --[FNO2d]--> predicted field on the grid
              --[GridDecoder]--> predictions at arbitrary query coordinates
"""

import torch
import torch.nn as nn

from GATr import GATrModel
from grid_projection import GridProjector, GridDecoder, reshape_grid_features
from fno import FNO2d
from graph_utils import build_knn_graph, build_bipartite_knn, build_structured_grid
from GA_encoder import encode_points_batch_torch
from near_wall_correction import NearWallCorrector


class FlowFieldPipeline(nn.Module):

    def __init__(
        self,
        input_scalar_dim: int = 5,
        mv_channels: int = 4,
        scalar_channels: int = 8,
        n_heads: int = 2,
        n_encoder_layers: int = 4,
        grid_resolution: tuple = (64, 64),
        grid_bounds: tuple = ((-3.0, 3.0), (-3.0, 3.0)),
        grid_stretch_center: tuple = (0.5, 0.0),
        grid_stretch_gamma: float = 3.0,
        knn_k: int = 16,
        bipartite_k: int = 8,
        fno_hidden_channels: int = 32,
        fno_layers: int = 4,
        fno_modes: int = 16,
        n_outputs: int = 3,
        use_near_wall_correction: bool = True,
        near_wall_k: int = 8,
        near_wall_hidden: int = 32,
        near_wall_wall_scale: float = 0.05,
    ):
        super().__init__()
        self.knn_k = knn_k
        self.bipartite_k = bipartite_k
        self.grid_bounds = grid_bounds

        self.gatr = GATrModel(input_scalar_dim, mv_channels, scalar_channels, n_heads, n_encoder_layers)
        self.grid_projector = GridProjector(mv_channels, scalar_channels, n_heads)

        self.near_wall = NearWallCorrector(
            mv_channels, scalar_channels, n_outputs,
            k=near_wall_k, hidden=near_wall_hidden, wall_scale=near_wall_wall_scale,
        ) if use_near_wall_correction else None

        fno_in_channels = mv_channels * 16 + scalar_channels
        self.fno = FNO2d(fno_in_channels, fno_hidden_channels, n_outputs,
                          fno_layers, fno_modes, fno_modes)

        grid_coords, self.H, self.W = build_structured_grid(
            grid_bounds, grid_resolution, grid_stretch_center, grid_stretch_gamma
        )
        # GridDecoder needs the SAME actual (H, W) build_structured_grid ended up
        # with (rounding in the stretch split can shift it a few cells from the
        # requested resolution) so its inverse mapping lands on the same lattice.
        self.grid_decoder = GridDecoder(grid_bounds, (self.H, self.W), grid_stretch_center, grid_stretch_gamma)

        self.register_buffer('grid_coords', grid_coords)
        self.register_buffer('grid_mvs', encode_points_batch_torch(grid_coords))

    def encode_to_grid(self, coords: torch.Tensor, node_scalars: torch.Tensor):
        """
        coords       : (N, 2)  point-cloud coordinates for one sample
        node_scalars : (N, input_scalar_dim)  freestream (broadcast), sdf, normals, log|V_inf|

        Returns:
            fno_out   : (1, n_outputs, H, W) FNO's predicted field on the fixed grid
            point_ctx : (point_mv, point_s, coords) — GATr's per-node features at
                        this call's input points, needed by decode()'s near-wall
                        correction. Returned (not cached on self) so the pipeline
                        stays stateless across concurrent/repeated calls.

        Split from decode() so callers that need multiple queries against the same
        field (e.g. a finite-difference stencil for a physics loss) don't have to
        re-run the expensive GATr/GridProjector/FNO stages per query.
        """
        node_mvs = encode_points_batch_torch(coords)
        edge_index = build_knn_graph(coords, k=self.knn_k)
        point_mv, point_s = self.gatr(node_mvs, node_scalars, edge_index)

        bipartite_edges = build_bipartite_knn(coords, self.grid_coords, k=self.bipartite_k)
        grid_mv_out, grid_s_out = self.grid_projector(point_mv, point_s, self.grid_mvs, bipartite_edges)

        grid_field = reshape_grid_features(grid_mv_out, grid_s_out, self.H, self.W).unsqueeze(0)   # (1, C, H, W)
        fno_out = self.fno(grid_field)   # (1, n_outputs, H, W)
        return fno_out, (point_mv, point_s, coords)

    def decode(self, fno_out: torch.Tensor, query_coords: torch.Tensor,
               point_ctx=None, query_sdf: torch.Tensor = None) -> torch.Tensor:
        """
        fno_out      : (1, n_outputs, H, W)  from encode_to_grid
        query_coords : (Q, 2)  coordinates to evaluate the predicted field at
        point_ctx    : (point_mv, point_s, point_coords) from encode_to_grid — pass
                        together with query_sdf to apply the near-wall correction;
                        omit either to get the plain global (grid) prediction (used
                        e.g. by the continuity-residual finite-difference stencil,
                        which regularizes the smooth global field only).
        query_sdf    : (Q,)  signed distance at each query point

        Returns: (Q, n_outputs) predicted field at query_coords
        """
        pred = self.grid_decoder(fno_out, query_coords.unsqueeze(0))   # (1, Q, n_outputs)
        pred = pred.squeeze(0)

        if self.near_wall is not None and point_ctx is not None and query_sdf is not None:
            point_mv, point_s, point_coords = point_ctx
            pred = pred + self.near_wall(point_mv, point_s, point_coords, query_coords, query_sdf)

        return pred

    def forward(self, coords: torch.Tensor, node_scalars: torch.Tensor, query_coords: torch.Tensor,
                query_sdf: torch.Tensor = None) -> torch.Tensor:
        """
        coords       : (N, 2)  point-cloud coordinates for one sample
        node_scalars : (N, input_scalar_dim)  freestream (broadcast), sdf, normals
        query_coords : (Q, 2)  coordinates to evaluate the predicted field at
        query_sdf    : (Q,)  signed distance at query_coords, for the near-wall
                        correction gate; omit to get the plain global prediction

        Returns: (Q, n_outputs) predicted field at query_coords
        """
        fno_out, point_ctx = self.encode_to_grid(coords, node_scalars)
        return self.decode(fno_out, query_coords, point_ctx=point_ctx, query_sdf=query_sdf)
