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
    def __init__(self,input_dim:int,out_dim:int,degree: int=3,n_knots: int=10):
        super().__init__()
        self.input_dim=input_dim
        self.out_dim=out_dim
        self.degree=degree
        self.n_knots=n_knots
        self.c=nn.Parameter(torch.randn(size=(input_dim,out_dim,n_knots-degree),requires_grad=True))
    def forward(self,x):
        if len(x)!=self.input_dim:
            raise(ValueError,"Dim Mismatch")
        t=torch.linspace(-1,1,self.n_knots)
        out=torch.zeros(size=(self.out_dim,))
        for j in range(len(out)):
            vals = torch.stack([torch.sum(cox_deboor(c=self.c[i, j],t=t,order=self.degree, x=x[i])) for i in range(self.input_dim)])
            out[j] = vals.sum()
        
        return(out)
    




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
    layer = KanLayer(input_dim, out_dim, degree=1, n_knots=4)
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
