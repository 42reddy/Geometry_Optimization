"""
Assembles the full point-cloud -> flow-field model:

  surface point cloud + condition --[GeometryEncoder]--> context tokens
  query coordinates + context     --[SolutionDecoder]--> predicted field

See ginot.py for the GINOT architecture itself.
"""

import torch.nn as nn

from ginot import GeometryEncoder, SolutionDecoder


class GINOTModel(nn.Module):

    def __init__(
        self,
        embed_dim: int = 128,
        n_heads: int = 4,
        n_self_attn_layers: int = 3,
        n_decoder_layers: int = 2,
        n_centroids: int = 256,
        group_radius: float = 0.05,
        group_max_neighbors: int = 32,
        pe_freqs: int = 10,
        condition_dim: int = 3,
        condition_hidden: int = 64,
        ffn_mult: int = 4,
        n_outputs: int = 3,
    ):
        super().__init__()
        self.geometry_encoder = GeometryEncoder(
            embed_dim=embed_dim,
            n_heads=n_heads,
            n_self_attn_layers=n_self_attn_layers,
            n_centroids=n_centroids,
            group_radius=group_radius,
            group_max_neighbors=group_max_neighbors,
            pe_freqs=pe_freqs,
            condition_dim=condition_dim,
            condition_hidden=condition_hidden,
            ffn_mult=ffn_mult,
        )
        self.decoder = SolutionDecoder(
            embed_dim=embed_dim,
            n_heads=n_heads,
            n_layers=n_decoder_layers,
            pe_freqs=pe_freqs,
            n_outputs=n_outputs,
            ffn_mult=ffn_mult,
        )

    def encode(self, surface_coords, condition):
        """
        surface_coords : (S, 2)  airfoil boundary point cloud
        condition      : (condition_dim,)  per-sample scalar condition
                          (freestream direction, log flow speed)

        Returns: context tokens (1, n_centroids + 1, embed_dim) — decoupled
        from decode() so callers that need multiple queries against the same
        geometry (e.g. a finite-difference stencil for a physics loss)
        don't have to re-run the (more expensive) encoder per query.
        """
        return self.geometry_encoder(surface_coords, condition)

    def decode(self, context, query_coords):
        """
        context      : (1, n_centroids + 1, embed_dim)  from encode()
        query_coords : (Q, 2)  coordinates to evaluate the predicted field at

        Returns: (Q, n_outputs) predicted field at query_coords
        """
        return self.decoder(query_coords, context)

    def forward(self, surface_coords, condition, query_coords):
        context = self.encode(surface_coords, condition)
        return self.decode(context, query_coords)
