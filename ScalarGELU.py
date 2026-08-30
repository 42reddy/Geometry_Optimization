import torch
import torch.nn as nn


class GatedGELU(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.GELU = nn.GELU()


    def forward(self, x_mv, x_s):

        gate = self.GELU(x_mv[..., 0:1])

        x_mv_out = gate *x_mv
        x_s_out = self.GELU(x_s)

        return x_mv_out, x_s_out



