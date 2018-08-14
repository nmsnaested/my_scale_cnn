#pylint: disable=E1101
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from scale_cnn.convolution import ScaleConvolution
from scale_cnn.pooling import ScalePool

class SiCNN1(nn.Module): 
    def __init__(self, ratio, nratio): 
        super().__init__()
        
        self.ratio = ratio 
        self.nratio = nratio

        self.conv1 = ScaleConvolution(3, 36, 3, self.ratio, self.nratio, srange = 0, boundary_condition = "dirichlet", stride = 2)
        self.conv2 = ScaleConvolution(36, 64, 3, self.ratio, self.nratio, srange = 3, boundary_condition = "dirichlet")
        self.pool = ScalePool(self.ratio)
        
        self.fc1 = nn.Linear(64, 150, bias = True)
        self.fc2 = nn.Linear(150, 10, bias = True)

    def forward(self, x): 
        # the 2 following lines are scale+translation equivariant
        x = x.unsqueeze(1)  # [batch, sigma, feature, y, x]
        x = x.repeat(1, self.nratio, 1, 1, 1)  # [batch, sigma, feature, y, x]
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x) # [batch,feature]
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class SiCNN2(nn.Module): 
    def __init__(self,ratio,nratio): 
        super().__init__()
        
        self.ratio = ratio 
        self.nratio = nratio

        self.conv1 = ScaleConvolution(3, 48, 5, self.ratio, self.nratio, srange = 0, boundary_condition = "dirichlet", stride = 2)
        self.conv2 = ScaleConvolution(48, 92, 5, self.ratio, self.nratio, srange = 1, boundary_condition = "dirichlet")
        self.conv3 = ScaleConvolution(92, 92, 5, self.ratio, self.nratio, srange = 1, boundary_condition = "dirichlet")
        self.pool = ScalePool(self.ratio)
        
        self.fc1 = nn.Linear(92, 150, bias=True)
        self.fc2 = nn.Linear(150, 10, bias=True)

    def forward(self, x): 
        # the 2 following lines are scale+translation equivariant
        x = x.unsqueeze(1)  # [batch, sigma, feature, y, x]
        x = x.repeat(1, self.nratio, 1, 1, 1)  # [batch, sigma, feature, y, x]
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x) # [batch,feature]
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class SiCNN3(nn.Module): 
    def __init__(self, f_in,ratio,nratio): 
        super().__init__()
        
        self.ratio = ratio 
        self.nratio = nratio

        self.conv1 = ScaleConvolution(3, 96, 5, self.ratio, self.nratio, srange = 0, boundary_condition = "dirichlet", stride = 2)
        self.conv2 = ScaleConvolution(96, 96, 3, self.ratio, self.nratio, srange = 1, boundary_condition = "dirichlet")
        self.conv3 = ScaleConvolution(96, 192, 3, self.ratio, self.nratio, srange = 1, boundary_condition = "dirichlet")
        self.pool = ScalePool(self.ratio)
        
        self.fc1 = nn.Linear(192, 150, bias=True)
        self.fc2 = nn.Linear(150, 10, bias=True)

    def forward(self, x): 
        # the 2 following lines are scale+translation equivariant
        x = x.unsqueeze(1)  # [batch, sigma, feature, y, x]
        x = x.repeat(1, self.nratio, 1, 1, 1)  # [batch, sigma, feature, y, x]
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x) # [batch,feature]
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class miniSiAll(nn.Module): 
    def __init__(self, ratio, nratio):
        super().__init__()
        self.ratio = ratio 
        self.nratio = nratio

        self.conv1 = ScaleConvolution(3, 96, 3, ratio=self.ratio, nratio=self.nratio, srange=0, boundary_condition="dirichlet", padding=1)
        self.conv2 = ScaleConvolution(96, 96, 3, self.ratio, self.nratio, srange=2, boundary_condition="dirichlet", padding=1)
        self.conv3 = ScaleConvolution(96, 192, 3, self.ratio, self.nratio, srange=2, boundary_condition="dirichlet", padding=1)
        self.conv4 = ScaleConvolution(192, 192, 3, self.ratio, self.nratio, srange=2, boundary_condition="dirichlet", padding=1)
        self.weight5 = nn.Parameter(torch.empty(10, 192))
        nn.init.orthogonal_(self.weight5)

    def forward(self, x):
        x = x.unsqueeze(1)  # [batch, sigma, feature, y, x]
        x = x.repeat(1, self.nratio, 1, 1, 1) # [batch, sigma, feature, y, x]

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = torch.einsum("ij,bsjyx->bsiyx", (self.weight5, x))

        n_batch = x.size(0)
        n_ratio = x.size(1)
        n_features_in = x.size(2)
        x = x.view(n_batch, n_ratio, n_features_in, -1).mean(-1) # [batch, sigma, feature]
        factors = x.new_tensor([self.ratio ** (-2 * i) for i in range(n_ratio)])
        x = torch.einsum("zsf,s->zf", (x, factors))  # [batch, feature]
       
        return x