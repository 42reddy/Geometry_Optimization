"""
Non-uniform axis stretching for the FNO grid — the same idea classical CFD
meshes use (cells packed tight near the wall, coarse in the far field), but
applied to the fixed-resolution structured grid GridProjector/FNO/GridDecoder
operate on.

Motivation: a real AirfRANS mesh spends ~half its ~180k points within 1% of
a chord length of the airfoil (viscous boundary-layer refinement), while a
UNIFORM FNO grid spends its resolution evenly across the whole domain — so a
single grid cell can be several boundary-layer-widths wide, and no amount of
training data fixes that; GridDecoder can't recover detail the grid never
represented. Stretching the two axes so grid density is highest near the
airfoil (x~0.5 chord, y~0) and falls off towards the domain edges fixes this
at the representation level, independent of how the training points were
sampled.

FNO's FFT still operates on a perfectly regular (H, W) INDEX array — only the
index -> physical coordinate mapping is nonlinear, via a two-sided tanh
clustering function (composed of the standard one-sided wall-clustering tanh
stretch used in structured CFD meshing, mirrored on each side of the
center). This is analytically invertible, which is what GridDecoder needs to
map arbitrary query coordinates back to normalized grid_sample coordinates.
"""

import torch


class AxisStretch:
    """
    One coordinate axis, spanning [lo, hi], with `n` grid points clustered
    around `center` (typically the airfoil's location on that axis). `gamma`
    controls how aggressive the clustering is (0 ~ nearly uniform, 3-5 ~
    strong clustering near the center).
    """

    def __init__(self, lo: float, hi: float, center: float, gamma: float, n: int):
        self.lo, self.hi = float(lo), float(hi)
        span = self.hi - self.lo
        # keep the center strictly interior so both segments have >=1 point
        self.center = float(min(max(center, self.lo + 1e-3 * span), self.hi - 1e-3 * span))
        self.gamma = float(gamma)

        frac_center = (self.center - self.lo) / span
        self.n_left = max(2, round(n * frac_center))
        self.n_right = max(2, n - self.n_left)
        self.n = self.n_left + self.n_right   # actual n after rounding

    def physical_coords(self, device=None, dtype=torch.float32) -> torch.Tensor:
        """The n (nonuniformly spaced) physical coordinates of this axis's grid points."""
        tanh_g = float(torch.tanh(torch.tensor(self.gamma)))

        xi_left = torch.linspace(0, 1, self.n_left, device=device, dtype=dtype)
        left = self.lo + (self.center - self.lo) * (torch.tanh(self.gamma * xi_left) / tanh_g)

        xi_right = torch.linspace(0, 1, self.n_right + 1, device=device, dtype=dtype)[1:]
        right = self.center + (self.hi - self.center) * (1 + torch.tanh(self.gamma * (xi_right - 1)) / tanh_g)

        return torch.cat([left, right])

    def to_norm(self, x: torch.Tensor) -> torch.Tensor:
        """Physical coordinate -> normalized position in [-1, 1] over this
        axis's n grid points (align_corners=True convention), for
        F.grid_sample. Analytic inverse of physical_coords()."""
        x = x.clamp(self.lo, self.hi)
        tanh_g = torch.tanh(torch.tensor(self.gamma, device=x.device, dtype=x.dtype))

        is_left = x <= self.center

        s_l = ((x - self.lo) / max(self.center - self.lo, 1e-8)).clamp(1e-6, 1 - 1e-6)
        xi_l = torch.atanh((tanh_g * s_l).clamp(-1 + 1e-6, 1 - 1e-6)) / self.gamma
        idx_l = xi_l * (self.n_left - 1)

        s_r = ((x - self.center) / max(self.hi - self.center, 1e-8)).clamp(1e-6, 1 - 1e-6)
        xi_r = torch.atanh(((s_r - 1) * tanh_g).clamp(-1 + 1e-6, 1 - 1e-6)) / self.gamma + 1
        idx_r = self.n_left + xi_r * self.n_right - 1

        idx = torch.where(is_left, idx_l, idx_r)
        norm = 2 * idx / (self.n - 1) - 1
        return norm.clamp(-1, 1)
