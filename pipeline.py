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
        knn_k: int = 16,
        bipartite_k: int = 8,
        fno_hidden_channels: int = 32,
        fno_layers: int = 4,
        fno_modes: int = 16,
        n_outputs: int = 3,
    ):
        super().__init__()
        self.knn_k = knn_k
        self.bipartite_k = bipartite_k
        self.grid_bounds = grid_bounds

        self.gatr = GATrModel(input_scalar_dim, mv_channels, scalar_channels, n_heads, n_encoder_layers)
        self.grid_projector = GridProjector(mv_channels, scalar_channels, n_heads)

        fno_in_channels = mv_channels * 16 + scalar_channels
        self.fno = FNO2d(fno_in_channels, fno_hidden_channels, n_outputs,
                          fno_layers, fno_modes, fno_modes)

        self.grid_decoder = GridDecoder(grid_bounds)

        grid_coords, self.H, self.W = build_structured_grid(grid_bounds, grid_resolution)
        self.register_buffer('grid_coords', grid_coords)
        self.register_buffer('grid_mvs', encode_points_batch_torch(grid_coords))

    def forward(self, coords: torch.Tensor, node_scalars: torch.Tensor, query_coords: torch.Tensor) -> torch.Tensor:
        """
        coords       : (N, 2)  point-cloud coordinates for one sample
        node_scalars : (N, input_scalar_dim)  freestream (broadcast), sdf, normals
        query_coords : (Q, 2)  coordinates to evaluate the predicted field at

        Returns: (Q, n_outputs) predicted field at query_coords
        """
        node_mvs = encode_points_batch_torch(coords)
        edge_index = build_knn_graph(coords, k=self.knn_k)
        point_mv, point_s = self.gatr(node_mvs, node_scalars, edge_index)

        bipartite_edges = build_bipartite_knn(coords, self.grid_coords, k=self.bipartite_k)
        grid_mv_out, grid_s_out = self.grid_projector(point_mv, point_s, self.grid_mvs, bipartite_edges)

        grid_field = reshape_grid_features(grid_mv_out, grid_s_out, self.H, self.W).unsqueeze(0)   # (1, C, H, W)
        fno_out = self.fno(grid_field)   # (1, n_outputs, H, W)

        pred = self.grid_decoder(fno_out, query_coords.unsqueeze(0))   # (1, Q, n_outputs)
        return pred.squeeze(0)
