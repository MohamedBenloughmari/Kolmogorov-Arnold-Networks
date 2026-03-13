import torch
import torch.nn as nn
from Cox_deboor import cox_deboor

class bspline(nn.Module):
    def __init__(self,degree=3,c_dim=10):
        super(bspline, self).__init__()
        self.t_dim = c_dim+degree+1
        self.c_dim = c_dim
        self.t=nn.Parameter(torch.randn(self.t_dim))
        self.c=nn.Parameter(torch.randn(c_dim))
        self.p=degree
    def forward(self, x):
        return cox_deboor(self.t,self.c,self.p,x)

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
    bs=bspline()
    x=torch.tensor([1.0], requires_grad=True)
    output=bs(x)
    print("bs(x):", output)
    torch.autograd.set_detect_anomaly(True)
    output.backward()
    print("dx/dx:", x.grad)