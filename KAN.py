import torch
import torch.nn as nn
from KANLayer import KanLayer



class KAN(nn.Module):
    def __init__(self,dim_array:list):
        super().__init__()
        self.dim_array=dim_array
        self.kan = nn.ModuleList(
        KanLayer(input_dim=self.dim_array[i], out_dim=self.dim_array[i+1])
        for i in range(len(self.dim_array) - 1)
        )
    def forward(self,x):
        for layer in self.kan:
            x = layer(x)
        return x


def test_single_forward():
    model = KAN([3, 2, 1])
    x = torch.randn(3)
    out = model(x)
    assert out.shape == (1,), f"Expected (1,), got {out.shape}"
    print(f"[PASS] single forward: input {x.shape} -> output {out.shape}")

def test_batch_forward():
    model = KAN([3, 2, 1])
    x = torch.randn(8, 3)
    out = model(x)
    assert out.shape == (8, 1), f"Expected (8, 1), got {out.shape}"
    print(f"[PASS] batch forward: input {x.shape} -> output {out.shape}")

def test_batch_backprop():
    model = KAN([3, 2, 1])
    x = torch.randn(8, 3)
    out = model(x)
    loss = out.sum()
    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No grad for {name}"
        assert not torch.all(param.grad == 0), f"Zero grad for {name}"
    print(f"[PASS] batch backprop: loss={loss.item():.4f}, all params have non-zero grads")

if __name__ == "__main__":
    test_single_forward()
    test_batch_forward()
    test_batch_backprop()




