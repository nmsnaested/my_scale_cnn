#pylint: disable=E1101
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from scale_cnn.convolution import ScaleConvolution
from scale_cnn.pooling import ScalePool

class ModelVN(nn.Module):
    def __init__(self, ratio, nratio, srange):
        super().__init__()
        self.ratio = ratio 
        self.nratio = nratio
        self.srange = srange

        #ScaleConvolution(self, features_in, features_out, size, ratio, nratio, srange=0, boundary_condition="dirichlet", padding=0, bias=True, **kwargs)
        self.conv11 = ScaleConvolution(features_in=3, features_out=96, size=11, ratio=self.ratio, nratio=self.nratio, srange=0, boundary_condition="dirichlet", padding=0, stride=4)
        self.weight12 = nn.Parameter(torch.empty(96, 96))
        nn.init.orthogonal_(self.weight12)
        self.conv13 = ScaleConvolution(96, 96, 3, ratio=self.ratio, nratio=self.nratio, srange=self.srange, boundary_condition="dirichlet", padding=1, stride=2)
        
        self.conv21 = ScaleConvolution(96, 256, 5, ratio=self.ratio, nratio=self.nratio, srange=self.srange, boundary_condition="dirichlet", padding=2, stride=1)
        self.weight22 = nn.Parameter(torch.empty(256, 256))
        nn.init.orthogonal_(self.weight22)
        self.conv23 = ScaleConvolution(256, 256, 3, ratio=self.ratio, nratio=self.nratio, srange=self.srange, boundary_condition="dirichlet", padding=0, stride=2)

        self.conv31 = ScaleConvolution(256, 384, 3, ratio=self.ratio, nratio=self.nratio, srange=self.srange, boundary_condition="dirichlet", padding=1, stride=1)
        self.weight32 = nn.Parameter(torch.empty(384, 384))
        nn.init.orthogonal_(self.weight32)
        self.conv33 = ScaleConvolution(384, 384, 3, ratio=self.ratio, nratio=self.nratio, srange=self.srange, boundary_condition="dirichlet", padding=0, stride=2)
        
        self.weight4 = nn.Parameter(torch.empty(1024, 384))
        nn.init.orthogonal_(self.weight4)
        self.weight5 = nn.Parameter(torch.empty(1024, 1024))
        nn.init.orthogonal_(self.weight5)
        self.weight6 = nn.Parameter(torch.empty(210, 1024))
        nn.init.orthogonal_(self.weight6)
         
    def forward(self, x): 
        x = x.unsqueeze(1)  # [batch, sigma, feature, y, x]
        x = x.repeat(1, self.nratio, 1, 1, 1) # [batch, sigma, feature, y, x]
        
        x = F.relu(self.conv11(x))
        x = torch.einsum("ij,bsjyx->bsiyx", (self.weight12, x))
        x = F.relu(self.conv13(x))

        x = F.relu(self.conv21(x))
        x = torch.einsum("ij,bsjyx->bsiyx", (self.weight22, x))
        x = F.relu(self.conv23(x))

        x = F.relu(self.conv31(x))
        x = torch.einsum("ij,bsjyx->bsiyx", (self.weight32, x))
        x = F.relu(self.conv33(x))
        #x = F.dropout(x, 0.5)

        x = torch.einsum("ij,bsjyx->bsiyx", (self.weight4, x))
        x = torch.einsum("ij,bsjyx->bsiyx", (self.weight5, x))
        x = torch.einsum("ij,bsjyx->bsiyx", (self.weight6, x))

        n_batch = x.size(0)
        n_ratio = x.size(1)
        n_features_in = x.size(2)
        x = x.view(n_batch, n_ratio, n_features_in, -1).mean(-1) # [batch, sigma, feature]
        factors = x.new_tensor([self.ratio ** (-2 * i) for i in range(n_ratio)])
        x = torch.einsum("zsf,s->zf", (x, factors))  # [batch, feature]

        return x

