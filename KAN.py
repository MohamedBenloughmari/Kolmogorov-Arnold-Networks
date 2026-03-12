import torch
import torch.nn as nn
from scipy.interpolate import BSpline


class bspline(nn.Module):
    def __init__(self,t_dim=10,c_dim=10):
        super(bspline, self).__init__()
        self.t_dim = t_dim
        self.c_dim = c_dim
        self.t=nn.Parameter(torch.randn(t_dim))
        self.c=nn.Parameter(torch.randn(c_dim))
    def forward(self, x):
        spline = BSpline(torch.sort(self.t).values.detach().cpu().numpy(),
                         self.c.detach().cpu().numpy(),
                         3)
        y = spline(x.detach().cpu().numpy())
        return torch.as_tensor(y, dtype=x.dtype, device=x.device)

class bspline_array(nn.Module):
    def __init__(self, t_dim=10, c_dim=10, n=5):
        super(bspline_array, self).__init__()
        self.t_dim = t_dim
        self.c_dim = c_dim
        self.n = n
        self.bsplines = nn.ModuleList([bspline(t_dim, c_dim) for _ in range(n)])
    def forward(self, x):
        return torch.stack([bs(x[:, i]) for i, bs in enumerate(self.bsplines)], dim=1)

class KAN(nn.Module):
    def __init__(self, t_dim=10, c_dim=10, n=5, m=7):
        super(KAN, self).__init__()
        self.t_dim = t_dim
        self.c_dim = c_dim
        self.n = n
        self.m = m
        self.bspline_arrays = nn.ModuleList([bspline_array(t_dim, c_dim, n) for _ in range(m)])
    def forward(self, x):
        return torch.stack([ba(x) for ba in self.bspline_arrays], dim=2).sum(dim=1)



if __name__ == "__main__":
    batch = 4
    bs = bspline()
    bs_array = bspline_array()
    kan = KAN()

    x_bs = torch.randn(batch)        # (batch,)
    x_arr = torch.randn(batch, 5)    # (batch, n=5)
    x_mat = torch.randn(batch, 5)    # (batch, n=5)
    print("bs(x):", bs(x_bs))                       # (batch,)
    print("bs_array(x):", bs_array(x_arr))           # (batch, n)
    print("KAN(x):", kan(x_mat))         # (batch, n, m)
    print("KAN(x).shape:", kan(x_mat).shape)