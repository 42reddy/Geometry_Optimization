"""
Fourier Neural Operator over GATr's structured grid projection.

Takes the (C, H, W) grid produced by grid_projection.reshape_grid_features
and predicts the flow field on that same grid. The grid is only an
intermediate representation — see grid_projection.GridDecoder for how
predictions get queried back at the original point-cloud coordinates for
loss computation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv2d(nn.Module):
    """2D Fourier layer: FFT -> learned linear transform on low modes -> inverse FFT."""

    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1   # number of Fourier modes kept along dim 1
        self.modes2 = modes2   # number of Fourier modes kept along dim 2

        scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )

    def compl_mul2d(self, x, weights):
        # (B, C_in, x, y), (C_in, C_out, x, y) -> (B, C_out, x, y)
        return torch.einsum("bixy,ioxy->boxy", x, weights)

    def forward(self, x):
        B, _, H, W = x.shape
        x_ft = torch.fft.rfft2(x, norm="ortho")   # (B, C_in, H, W//2+1)

        out_ft = torch.zeros(
            B, self.out_channels, H, W // 2 + 1,
            dtype=torch.cfloat, device=x.device
        )

        # keep only the low-frequency modes (corners of the FFT), discard the rest —
        # this is FNO's resolution-invariance: the learned weights don't depend on H, W
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, :self.modes1, :self.modes2], self.weights1
        )
        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, -self.modes1:, :self.modes2], self.weights2
        )

        return torch.fft.irfft2(out_ft, s=(H, W), norm="ortho")


class FNOBlock(nn.Module):
    """Spectral (global) path + pointwise local path, combined — standard FNO layer."""

    def __init__(self, channels, modes1, modes2):
        super().__init__()
        self.spectral = SpectralConv2d(channels, channels, modes1, modes2)
        self.local = nn.Conv2d(channels, channels, kernel_size=1)
        self.norm = nn.InstanceNorm2d(channels)

    def forward(self, x):
        x = self.spectral(x) + self.local(x)
        x = self.norm(x)
        return F.gelu(x)


class FNO2d(nn.Module):
    """
    Input : (B, C_in, H, W)   grid features from GridProjector + reshape_grid_features
    Output: (B, n_outputs, H, W)   predicted flow field on the grid
            (v_x, v_y, pressure, ...) — still needs GridDecoder to be
            evaluated at the original point-cloud coordinates.
    """

    def __init__(self, in_channels, hidden_channels=32, n_outputs=3,
                 n_layers=4, modes1=16, modes2=16):
        super().__init__()
        self.lift = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)
        self.blocks = nn.ModuleList([
            FNOBlock(hidden_channels, modes1, modes2) for _ in range(n_layers)
        ])
        self.project = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, n_outputs, kernel_size=1),
        )

    def forward(self, x):
        x = self.lift(x)
        for block in self.blocks:
            x = block(x)
        return self.project(x)
