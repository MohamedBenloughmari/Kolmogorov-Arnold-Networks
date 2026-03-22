import torch
import torch.nn as nn
from spline import cox_deboor
'''
    in entry take a base x of size n

    for each x[i] we create a nout c vector each vector of len(t)-p

    the t is of size 10 at first with p=3

    C is a tensor of sie [nin,nout,len(t) -p]

    we apply cox de boor get out[j]=sum_i(coxdeboos(C[i,j],x[i]))

    


'''


class KanLayer(nn.Module):
    def __init__(self,input_dim:int,out_dim:int,degree: int=3,n_knots: int=5):
        super().__init__()
        self.input_dim=input_dim
        self.out_dim=out_dim
        self.degree=degree
        self.n_knots=n_knots
        self.c=nn.Parameter(torch.randn(size=(input_dim,out_dim,n_knots-degree),requires_grad=True))
        self.wb=nn.Parameter(torch.randn(size=(self.input_dim, self.out_dim),requires_grad=True))
        self.ws=nn.Parameter(torch.randn(size=(self.input_dim, self.out_dim),requires_grad=True))
    def forward(self, x):
        scalar = x.dim() == 1
        if scalar:
            x = x.unsqueeze(0)
        batch = x.shape[0]
        if x.shape[1] != self.input_dim:
            raise ValueError(f"Dim Mismatch: expected {self.input_dim}, got {x.shape[1]}")
        t = torch.linspace(-1, 1, self.n_knots)
        out = torch.zeros(batch, self.out_dim)
        for j in range(self.out_dim):
            for i in range(self.input_dim):
                out[:, j] = out[:, j] + self.ws[i, j] * cox_deboor(c=self.c[i, j], t=t, order=self.degree, x=x[:, i]) + self.wb[i, j] * (x[:, i] / (1 + torch.exp(-x[:, i])))
        if scalar:
            return out.squeeze(0)
        return out
    




def test_kan_layer_forward_shape():
    input_dim = 4
    out_dim = 2
    degree = 3
    n_knots = 10
    
    layer = KanLayer(input_dim, out_dim, degree, n_knots)
    input_tensor = torch.randn(input_dim)
    
    output = layer(input_tensor)
    print(output)
    assert output.shape == (out_dim,), f"Expected shape {(out_dim,)}, got {output.shape}"
    assert output.requires_grad == True, "Output should track gradients"

def test_kan_layer_backward_pass():
    input_dim = 2
    out_dim = 1
    layer = KanLayer(input_dim, out_dim, degree=1, n_knots=10)
    input_tensor = torch.randn(input_dim)
    
    output = layer(input_tensor)
    loss = output.sum()
    loss.backward()
    print(loss)
    assert layer.c.grad is not None, "Gradients should be computed for parameter c"
    assert not torch.all(layer.c.grad == 0), "Gradients should be non-zero"

if __name__ == "__main__":
    test_kan_layer_forward_shape()
    test_kan_layer_backward_pass()
