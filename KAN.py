import torch
import torch.nn as nn
from Cox_deboor import cox_deboor

class bspline(nn.Module):
    def __init__(self,p=3,c_dim=10):
        super(bspline, self).__init__()
        self.t_dim = c_dim+p+1
        self.c_dim = c_dim
        self.t=nn.Parameter(torch.randn(self.t_dim))
        self.c=nn.Parameter(torch.randn(c_dim))
        self.p=p
    def forward(self, x):
        return cox_deboor(self.t,self.c,self.p,x)

class bspline_array(nn.Module):
    def __init__(self, p=3, c_dim=10, n=5):
        super(bspline_array, self).__init__()
        self.t_dim = c_dim+p+1
        self.c_dim = c_dim
        self.n = n
        self.p= p
        self.bsplines = nn.ModuleList([bspline(p, c_dim) for _ in range(n)])
    def forward(self, x):
        return torch.stack([bs(x[i]) for i, bs in enumerate(self.bsplines)], dim=0)

class KAN(nn.Module):
    def __init__(self,p=3, c_dim=10, n=5, m=7):
        super(KAN, self).__init__()
        self.t_dim = c_dim+p+1
        self.c_dim = c_dim
        self.n = n
        self.m = m
        self.p = p
        self.bspline_arrays = nn.ModuleList([bspline_array(p, c_dim, m) for _ in range(n)])
    def forward(self, x):
        return torch.stack([ba(x[j]) for j, ba in enumerate(self.bspline_arrays)], dim=1)


if __name__ == "__main__":
    #test backward on bspline_arrat
    x=torch.randn(size=(5,7), requires_grad=True)
    print("x:", x[0])
    kan=KAN()
    output = kan(x)
    print("kan(x):", output)
    output.sum().backward()
    print("dx/dx:", x.grad)