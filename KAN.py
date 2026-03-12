import torch
import torch.nn as nn
from scipy.interpolate import BSpline


def deBoor(k: int, x: int, t, c, p: int):
    """Evaluates S(x).

    Arguments
    ---------
    k: Index of knot interval that contains x.
    x: Position.
    t: Array of knot positions, needs to be padded as described above.
    c: Array of control points.
    p: Degree of B-spline.
    """
    d = [c[j + k - p] for j in range(0, p + 1)] 

    for r in range(1, p + 1):
        for j in range(p, r - 1, -1):
            alpha = (x - t[j + k - p]) / (t[j + 1 + k - r] - t[j + k - p]) 
            d[j] = (1.0 - alpha) * d[j - 1] + alpha * d[j]

    return d[p]


class bspline(nn.Module):
    def __init__(self,t_dim=10,c_dim=10):
        super(bspline, self).__init__()
        self.t_dim = t_dim
        self.c_dim = c_dim
        self.t=nn.Parameter(torch.randn(t_dim))
        self.c=nn.Parameter(torch.randn(c_dim))
        self.p=3
    def forward(self, x):
        k = find_k_batch(x, self.t, self.p, self.c)
        spline = deBoor_batch(k, x, self.t, self.c, self.p)
        return spline

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