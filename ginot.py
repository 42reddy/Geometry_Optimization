"""
GINOT (Geometry-Informed Neural Operator Transformer) — Liu et al. 2025,
"Geometry-Informed Neural Operator Transformer for Partial Differential
Equations on Arbitrary Geometries", arXiv:2504.19452.

Two stages:
  GeometryEncoder : boundary point cloud (+ a per-sample scalar condition)
                    -> a small set of context tokens, via a PointNet++-style
                    local/global split fused by cross-attention, refined by
                    self-attention. No SDF or normals needed as input — the
                    geometry is captured purely from point coordinates.
  SolutionDecoder : arbitrary query coordinates -> predicted field, via
                    cross-attention against the encoder's context tokens.

Unlike the paper's batched training (which needs padding/masking for
variable point counts across a batch), this project trains one sample at a
time (see train.py), so there is no batch dimension to pad across and no
masking machinery here — every tensor below is a single point cloud.
"""

import math

import torch
import torch.nn as nn
from torch_geometric.nn import fps, radius
from torch_scatter import scatter_max


class PositionalEncoding(nn.Module):
    """NeRF-style sinusoidal encoding: [x, sin(2^k pi x), cos(2^k pi x), ...]
    for k in [0, n_freqs). Raw coordinates alone give MLPs/attention very
    little to work with at the sub-chord length scales that matter here
    (boundary-layer curvature, near-wall gradients); this is the standard
    fix, giving every input scale its own frequency band to key off.
    """

    def __init__(self, in_dim: int = 2, n_freqs: int = 10):
        super().__init__()
        self.in_dim = in_dim
        self.n_freqs = n_freqs
        freq_bands = (2.0 ** torch.arange(n_freqs)) * math.pi
        self.register_buffer('freq_bands', freq_bands)

    @property
    def out_dim(self) -> int:
        return self.in_dim * (1 + 2 * self.n_freqs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., in_dim)
        angles = x.unsqueeze(-1) * self.freq_bands   # (..., in_dim, n_freqs)
        enc = torch.cat([angles.sin(), angles.cos()], dim=-1)   # (..., in_dim, 2*n_freqs)
        enc = enc.flatten(-2)   # (..., in_dim*2*n_freqs)
        return torch.cat([x, enc], dim=-1)   # (..., out_dim)


class SelfAttentionBlock(nn.Module):
    """Pre-LN transformer block: self-attention + FFN, both residual."""

    def __init__(self, embed_dim: int, n_heads: int, ffn_mult: int = 4, dropout: float = 0.0):
        super().__init__()
        self.norm_attn = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout, batch_first=True)
        self.norm_ffn = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * ffn_mult),
            nn.GELU(),
            nn.Linear(embed_dim * ffn_mult, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (1, N, embed_dim)
        h = self.norm_attn(x)
        attn_out, _ = self.attn(h, h, h)
        x = x + attn_out
        x = x + self.ffn(self.norm_ffn(x))
        return x


class CrossAttentionBlock(nn.Module):
    """Pre-LN transformer block: cross-attention (query attends to context)
    + FFN, both residual."""

    def __init__(self, embed_dim: int, n_heads: int, ffn_mult: int = 4, dropout: float = 0.0):
        super().__init__()
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout, batch_first=True)
        self.norm_ffn = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * ffn_mult),
            nn.GELU(),
            nn.Linear(embed_dim * ffn_mult, embed_dim),
        )

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # query: (1, Nq, embed_dim), context: (1, Nc, embed_dim)
        q = self.norm_q(query)
        kv = self.norm_kv(context)
        attn_out, _ = self.attn(q, kv, kv)
        x = query + attn_out
        x = x + self.ffn(self.norm_ffn(x))
        return x


class GeometryEncoder(nn.Module):
    """
    Encodes an airfoil's boundary point cloud (plus a per-sample scalar
    condition — freestream direction and log flow speed here) into a small,
    fixed-size set of context tokens for the decoder to cross-attend over.

    Local branch (PointNet++-style, since GINOT explicitly borrows this):
    farthest-point-sample a set of centroids, ball-query each centroid's
    neighborhood, positionally encode the relative offsets, and max-pool a
    shared per-neighbor MLP over each group. The paper describes this local
    aggregation as a 2D conv + MLP; max-pooling a per-point MLP is
    PointNet++'s own version of the same operation and is used here instead.

    Global branch: every surface point, independently positionally encoded
    and linearly projected — cheap, since it's just a per-point MLP with no
    attention over the (S ~ 1000) points.

    The local tokens (query) then cross-attend over the global tokens
    (key/value) to fuse local detail with whole-airfoil context, the
    per-sample condition is appended as one extra token, and a stack of
    self-attention blocks refines the whole token set into the final
    context passed to the decoder.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        n_heads: int = 4,
        n_self_attn_layers: int = 3,
        n_centroids: int = 256,
        group_radius: float = 0.05,
        group_max_neighbors: int = 32,
        pe_freqs: int = 10,
        condition_dim: int = 3,
        condition_hidden: int = 64,
        ffn_mult: int = 4,
    ):
        super().__init__()
        self.n_centroids = n_centroids
        self.group_radius = group_radius
        self.group_max_neighbors = group_max_neighbors

        self.pe = PositionalEncoding(in_dim=2, n_freqs=pe_freqs)

        self.local_mlp = nn.Sequential(
            nn.Linear(self.pe.out_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.global_proj = nn.Linear(self.pe.out_dim, embed_dim)

        self.condition_encoder = nn.Sequential(
            nn.Linear(condition_dim, condition_hidden),
            nn.GELU(),
            nn.Linear(condition_hidden, embed_dim),
        )

        self.fuse_cross_attn = CrossAttentionBlock(embed_dim, n_heads, ffn_mult)
        self.self_attn_blocks = nn.ModuleList([
            SelfAttentionBlock(embed_dim, n_heads, ffn_mult)
            for _ in range(n_self_attn_layers)
        ])

    def forward(self, surface_coords: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        surface_coords : (S, 2) airfoil boundary point cloud
        condition      : (condition_dim,) per-sample scalar condition

        Returns: (1, n_centroids + 1, embed_dim) context tokens.
        """
        s = surface_coords.shape[0]
        ratio = min(1.0, self.n_centroids / s)
        centroid_idx = fps(surface_coords, ratio=ratio)
        # fps' ratio-based selection can overshoot by a point or two — clip
        # to the configured budget so token count stays fixed run to run.
        centroid_idx = centroid_idx[:self.n_centroids]
        centroids = surface_coords[centroid_idx]   # (Ns, 2)

        # row: index into centroids, col: index into surface_coords
        row, col = radius(
            surface_coords, centroids, r=self.group_radius,
            max_num_neighbors=self.group_max_neighbors,
        )
        rel = surface_coords[col] - centroids[row]         # (E, 2)
        local_feat = self.local_mlp(self.pe(rel))          # (E, embed_dim)
        local_tokens, _ = scatter_max(
            local_feat, row, dim=0, dim_size=centroids.shape[0]
        )   # (Ns, embed_dim) — every centroid includes itself (offset 0) in
            # its own group, so no group is ever empty.

        global_tokens = self.global_proj(self.pe(surface_coords))   # (S, embed_dim)

        fused = self.fuse_cross_attn(
            local_tokens.unsqueeze(0), global_tokens.unsqueeze(0)
        )   # (1, Ns, embed_dim)

        cond_token = self.condition_encoder(condition).view(1, 1, -1)   # (1, 1, embed_dim)
        tokens = torch.cat([fused, cond_token], dim=1)   # (1, Ns+1, embed_dim)

        for block in self.self_attn_blocks:
            tokens = block(tokens)

        return tokens


class SolutionDecoder(nn.Module):
    """Cross-attention decoder: query coordinates attend over the encoder's
    context tokens to produce the predicted field at those coordinates.
    Because this is a plain function of continuous query coordinates (no
    grid), it can be queried at arbitrary points — including off-sample
    coordinates used for the finite-difference physics loss in train.py."""

    def __init__(
        self,
        embed_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        pe_freqs: int = 10,
        n_outputs: int = 3,
        ffn_mult: int = 4,
    ):
        super().__init__()
        self.pe = PositionalEncoding(in_dim=2, n_freqs=pe_freqs)
        self.query_proj = nn.Linear(self.pe.out_dim, embed_dim)
        self.blocks = nn.ModuleList([
            CrossAttentionBlock(embed_dim, n_heads, ffn_mult)
            for _ in range(n_layers)
        ])
        self.out_norm = nn.LayerNorm(embed_dim)
        self.out_proj = nn.Linear(embed_dim, n_outputs)

    def forward(self, query_coords: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        query_coords : (Nq, 2)
        context      : (1, Nc, embed_dim) from GeometryEncoder

        Returns: (Nq, n_outputs)
        """
        q = self.query_proj(self.pe(query_coords)).unsqueeze(0)   # (1, Nq, embed_dim)
        for block in self.blocks:
            q = block(q, context)
        return self.out_proj(self.out_norm(q)).squeeze(0)   # (Nq, n_outputs)
