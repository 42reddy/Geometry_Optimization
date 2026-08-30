import torch
import torch.nn as nn


GRADES = {
    0: slice(0, 1),
    1: slice(1, 5),
    2: slice(5, 11),
    3: slice(11, 15),
    4: slice(15, 16),
}


def project_grade(x, grade):

    out = torch.zeros_like(x)
    out[..., GRADES[grade]] = x[..., GRADES[grade]]  # which grade to extract?
    return out


def e0_multiply(x):

    out = torch.zeros_like(x)
    out[..., 1] = x[..., 0]  # scalar → e0        (given)
    out[..., 5] = x[..., 2]  # e1 → e01            (given)
    out[..., 6] = x[..., 3]  # e2 → e02
    out[..., 7] = x[..., 4]  # e3 → e03
    out[..., 11] = x[..., 8]  # e12 → e012
    out[..., 12] = x[..., 9]  # e13 → e013
    out[..., 13] = x[..., 10]  # e23 → e023
    return out


class EquivariantLinear(nn.Module):
    """
    E(3)-equivariant linear layer for multivectors.

    Learns:
        w: (C_in, C_out, 5)  ← one weight per grade (0..4)
        v: (C_in, C_out, 4)  ← one weight per grade (0..3), for e0-multiplied terms
        bias: (C_out,)        ← only on grade-0 (scalar), only equivariant bias

    Input:  (B, N, C_in,  16)
    Output: (B, N, C_out, 16)
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # w weights: scale each grade
        self.w = nn.Parameter(
            torch.randn(self.in_channels, self.out_channels, 5) *0.1  # (C_in, C_out, 5 grades)
        )

        # v weights: e0-multiply then scale
        self.v = nn.Parameter(
            torch.randn(self.in_channels, self.out_channels, 4) *0.1  # (C_in, C_out, 4 grades)
        )

        # bias: only on grade-0
        self.bias = nn.Parameter(torch.zeros(self.out_channels))  # shape: (C_out,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N, C_in, 16)
        returns: (B, N, C_out, 16)
        """
        # initialize output tensor
        out = torch.zeros(
            *x.shape[:-2], self.out_channels, 16,
            device=x.device,  # ← use input's device
            dtype=x.dtype  # ← use input's dtype
        )

        # --- w terms: scale each grade, mix channels ---
        for k in range(5):  # d+1 grades for w?
            xk = project_grade(x, k)  # (B, N, C_in, 16)
            wk = self.w[:, :, k]  # (C_in, C_out)
            out += torch.einsum('bncd, ce -> bned', xk, wk)

        # --- v terms: e0-multiply then scale, mix channels ---
        for k in range(4):  # d grades for v?
            xk = project_grade(x, k)
            xk0 = e0_multiply(xk)
            vk = self.v[:, :, k]  # (C_in, C_out)
            out += torch.einsum('bncd, ce -> bned', xk0, vk)

        out[..., 0] += self.bias

        return out



