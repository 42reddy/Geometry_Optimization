import torch
import torch.nn as nn
from EquiLinear import EquivariantLinear


METRIC_INDICES = [0, 2, 3, 4, 8, 9, 10, 14, 15]
class EquivariantAttention(nn.Module):
    def __init__(self, mv_channels, scalar_channels, n_heads):
        super().__init__()
        # ... same init as before ...
        self.n_heads     = n_heads
        self.mv_channels = mv_channels
        self.sc_channels = scalar_channels
        self.head_mv_dim = mv_channels // n_heads
        self.head_sc_dim = scalar_channels // n_heads

        self.q_mv = EquivariantLinear(mv_channels, mv_channels)
        self.k_mv = EquivariantLinear(mv_channels, mv_channels)
        self.v_mv = EquivariantLinear(mv_channels, mv_channels)
        self.q_s  = nn.Linear(scalar_channels, scalar_channels)
        self.k_s  = nn.Linear(scalar_channels, scalar_channels)
        self.v_s  = nn.Linear(scalar_channels, scalar_channels)
        self.out_mv = EquivariantLinear(mv_channels, mv_channels)
        self.out_s  = nn.Linear(scalar_channels, scalar_channels)

    def forward(self, x_mv, x_s, edge_index=None):
        """
        x_mv       : (1, N, C, 16)
        x_s        : (1, N, C_s)
        edge_index : (2, E)  ← if provided, use sparse attention
                               if None, fall back to full attention
        """
        B, N, C, _ = x_mv.shape
        H           = self.n_heads

        # project
        Q_mv = self.q_mv(x_mv)
        K_mv = self.k_mv(x_mv)
        V_mv = self.v_mv(x_mv)
        Q_s  = self.q_s(x_s)
        K_s  = self.k_s(x_s)
        V_s  = self.v_s(x_s)

        if edge_index is not None:
            out_mv, out_s = self._sparse_attention(
                Q_mv, K_mv, V_mv, Q_s, K_s, V_s, edge_index, N
            )
        else:
            out_mv, out_s = self._full_attention(
                Q_mv, K_mv, V_mv, Q_s, K_s, V_s
            )

        out_mv = self.out_mv(out_mv)
        out_s  = self.out_s(out_s)
        return out_mv, out_s

    def _sparse_attention(self, Q_mv, K_mv, V_mv, Q_s, K_s, V_s, edge_index, N):
        """
        Only compute attention scores between connected neighbors.
        edge_index: (2, E)
        """
        src, dst = edge_index[0], edge_index[1]   # source, destination nodes

        # squeeze batch dim — sparse ops work on (N, C, 16)
        Q_mv = Q_mv.squeeze(0)    # (N, C, 16)
        K_mv = K_mv.squeeze(0)
        V_mv = V_mv.squeeze(0)
        Q_s  = Q_s.squeeze(0)     # (N, C_s)
        K_s  = K_s.squeeze(0)
        V_s  = V_s.squeeze(0)

        H        = self.n_heads
        head_mv  = self.head_mv_dim
        head_sc  = self.head_sc_dim

        # reshape into heads: (N, H, head_dim, 16)
        Q_mv = Q_mv.reshape(N, H, head_mv, 16)
        K_mv = K_mv.reshape(N, H, head_mv, 16)
        V_mv = V_mv.reshape(N, H, head_mv, 16)
        Q_s  = Q_s.reshape(N, H, head_sc)
        K_s  = K_s.reshape(N, H, head_sc)
        V_s  = V_s.reshape(N, H, head_sc)

        # compute scores ONLY for edges (src, dst)
        # Q[dst] · K[src] for each edge
        mv_scale  = (9 * head_mv) ** 0.5
        sc_scale  = head_sc ** 0.5

        # (E, H, head_mv, 9) dot product
        mv_scores = (
            Q_mv[dst][..., METRIC_INDICES] *
            K_mv[src][..., METRIC_INDICES]
        ).sum(dim=(-1, -2)) / mv_scale              # (E, H)

        sc_scores = (
            Q_s[dst] * K_s[src]
        ).sum(dim=-1) / sc_scale                    # (E, H)

        scores = mv_scores + sc_scores              # (E, H)

        # softmax over neighbors of each node
        # use torch_scatter for efficient neighborhood softmax
        from torch_scatter import scatter_softmax
        weights = scatter_softmax(scores, dst, dim=0)   # (E, H)

        # weighted sum of values
        # out[i] = sum over neighbors j of weight[i,j] * V[j]
        weighted_mv = (
            weights.unsqueeze(-1).unsqueeze(-1) *
            V_mv[src]
        )                                           # (E, H, head_mv, 16)

        weighted_s = (
            weights.unsqueeze(-1) * V_s[src]
        )                                           # (E, H, head_sc)

        # aggregate: sum over edges for each destination node
        from torch_scatter import scatter_add
        out_mv = scatter_add(
            weighted_mv.reshape(-1, H * head_mv * 16),
            dst, dim=0, dim_size=N
        ).reshape(N, H * head_mv, 16)              # (N, C, 16)

        out_s = scatter_add(
            weighted_s.reshape(-1, H * head_sc),
            dst, dim=0, dim_size=N
        ).reshape(N, self.sc_channels)             # (N, C_s)

        # add batch dim back
        return out_mv.unsqueeze(0), out_s.unsqueeze(0)

    def _full_attention(self, Q_mv, K_mv, V_mv, Q_s, K_s, V_s):
        # ... your existing full attention code ...
        B, N, C, _ = Q_mv.shape
        H           = self.n_heads

        Q_mv = Q_mv.reshape(B, N, H, self.head_mv_dim, 16).permute(0, 2, 1, 3, 4)
        K_mv = K_mv.reshape(B, N, H, self.head_mv_dim, 16).permute(0, 2, 1, 3, 4)
        V_mv = V_mv.reshape(B, N, H, self.head_mv_dim, 16).permute(0, 2, 1, 3, 4)
        Q_s  = Q_s.reshape(B, N, H, self.head_sc_dim).permute(0, 2, 1, 3)
        K_s  = K_s.reshape(B, N, H, self.head_sc_dim).permute(0, 2, 1, 3)
        V_s  = V_s.reshape(B, N, H, self.head_sc_dim).permute(0, 2, 1, 3)

        mv_scale  = (9 * self.head_mv_dim) ** 0.5
        sc_scale  = self.head_sc_dim ** 0.5
        mv_scores = torch.einsum('bhick,bhjck->bhij', Q_mv[..., METRIC_INDICES], K_mv[..., METRIC_INDICES]) / mv_scale
        sc_scores = torch.einsum('bhic,bhjc->bhij', Q_s, K_s) / sc_scale
        weights   = torch.softmax(mv_scores + sc_scores, dim=-1)
        out_mv    = torch.einsum('bhij,bhjcd->bhicd', weights, V_mv)
        out_s     = torch.einsum('bhij,bhjc->bhic',  weights, V_s)
        out_mv    = out_mv.permute(0, 2, 1, 3, 4).reshape(B, N, self.mv_channels, 16)
        out_s     = out_s.permute(0, 2, 1, 3).reshape(B, N, self.sc_channels)
        return out_mv, out_s





