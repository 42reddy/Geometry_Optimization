import torch
import torch.nn as nn


# translation-invariant component indices
METRIC_INDICES = [0, 2, 3, 4, 8, 9, 10, 14, 15]


class EquivariantLayerNorm(nn.Module):
    """
    E(3)-equivariant LayerNorm for multivectors.

    Normalizes by the invariant GA inner product averaged over channels.

    x_normalized = x / sqrt( mean_c(<x_c, x_c>) + eps )

    Input:  x_mv: (B, N, C, 16)
            x_s:  (B, N, C_s)
    Output: (B, N, C, 16), (B, N, C_s)
    """

    def __init__(self, in_channels: int, scalar_channels: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # regular LayerNorm for scalar stream
        self.scalar_norm = nn.LayerNorm(scalar_channels)  # ← normalize over which dimension?

    def forward(self, x_mv, x_s):

        # --- multivector stream ---
        inner = (x_mv[..., METRIC_INDICES] ** 2).sum(dim=-1, keepdim=True)

        norm = inner.mean(dim=-2, keepdim=True)

        # step 3: normalize
        x_mv_out = x_mv / (torch.sqrt(norm) + self.eps)  # broadcasts over C and 16

        # --- scalar stream ---
        x_s_out = self.scalar_norm(x_s)

        return x_mv_out, x_s_out


