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
    

