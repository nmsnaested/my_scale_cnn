#pylint: disable=E1101
'''
All architectures used
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

from scale_cnn.convolution import ScaleConvolution
from scale_cnn.pooling import ScalePool

class StdNet(nn.Module):
    def __init__(self, f_in=3):
        super().__init__()
        '''
        Standard convolution network, 2 conv layers + 2 fc layers
        :param f_in: number of input features 
        '''
        self.f_in = f_in
        self.conv1 = nn.Conv2d(f_in, 12, 7, 2)
        self.conv2 = nn.Conv2d(12, 21, 5)
        self.fc1 = nn.Linear(21 * 25 * 25, 150)
        self.fc2 = nn.Linear(150, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 21 * 25 * 25)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class AllCNN(nn.Module):   
    def __init__(self, f_in=3, n_classes=10, **kwargs):
        '''
        All convolutional network https://arxiv.org/pdf/1412.6806.pdf 
        based on code from https://github.com/StefOe/all-conv-pytorch/blob/master/allconv.py
        '''
        super(AllCNN, self).__init__()
        self.conv1 = nn.Conv2d(f_in, 96, 3, padding=1)
        self.conv2 = nn.Conv2d(96, 96, 3, padding=1)
        self.conv3 = nn.Conv2d(96, 96, 3, padding=1, stride=2)
        self.conv4 = nn.Conv2d(96, 192, 3, padding=1)
        self.conv5 = nn.Conv2d(192, 192, 3, padding=1)
        self.conv6 = nn.Conv2d(192, 192, 3, padding=1, stride=2)
        self.conv7 = nn.Conv2d(192, 192, 3, padding=1)
        self.conv8 = nn.Conv2d(192, 192, 1)
        self.conv9 = nn.Conv2d(192, n_classes, 1)

    def forward(self, x):
        #x_drop = F.dropout(x, .2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        #x = F.dropout(x, .5)
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        #x = F.dropout(x, .5)
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = F.relu(self.conv9(x))

        x = F.adaptive_avg_pool2d(x, 1)
        x.squeeze_(-1) #remove dimension of size 1 (1,2,3)->(2,3)
        x.squeeze_(-1)
        return x



class SiCNN(nn.Module): 
    def __init__(self, f_in, ratio, nratio): 
        super().__init__()
        '''
        Basic scale equivariant architecture, based on StdNet
        '''
        self.f_in = f_in
        self.ratio = ratio 
        self.nratio = nratio

        self.conv1 = ScaleConvolution(self.f_in, 12, 7, self.ratio, self.nratio, srange = 0, boundary_condition = "dirichlet", stride = 2)
        self.conv2 = ScaleConvolution(12, 21, 5, self.ratio, self.nratio, srange = 1, boundary_condition = "dirichlet")
        self.pool = ScalePool(self.ratio)
        
        self.fc1 = nn.Linear(21, 150, bias = True)
        self.fc2 = nn.Linear(150, 10, bias = True)

    def forward(self, x): 
        '''
        :param input: [batch, feature, y, x]
        '''
        x = x.unsqueeze(1)  # [batch, sigma, feature, y, x]
        x = x.repeat(1, self.nratio, 1, 1, 1)  # [batch, sigma, feature, y, x]
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x) # [batch,feature]
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class kanazawa(nn.Module): 
    def __init__(self, f_in, ratio, nratio, srange=1): 
        super().__init__()
        '''
        Scale equivariant arch, based on architecture in Kanazawa's paper 
        https://arxiv.org/abs/1412.5104
        '''
        self.f_in = f_in
        self.ratio = ratio 
        self.nratio = nratio
        self.srange = srange

        self.conv1 = ScaleConvolution(self.f_in, 36, 3, self.ratio, self.nratio, srange = 0, boundary_condition = "dirichlet", stride = 2)
        self.conv2 = ScaleConvolution(36, 64, 3, self.ratio, self.nratio, srange = 3, boundary_condition = "dirichlet")
        self.pool = ScalePool(self.ratio)
        
        self.fc1 = nn.Linear(64, 150, bias = True)
        self.fc2 = nn.Linear(150, 10, bias = True)

    def forward(self, x): 
        x = x.unsqueeze(1)  # [batch, sigma, feature, y, x]
        x = x.repeat(1, self.nratio, 1, 1, 1)  # [batch, sigma, feature, y, x]
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x) # [batch,feature]
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class SiCNN_3(nn.Module): 
    def __init__(self, f_in=1, size=5, ratio=2**(2/3), nratio=3, srange=1, padding=0): 
        super().__init__()
        '''
        Scale equivariant arch with 3 convolutional layers
        '''
        self.f_in = f_in
        self.size = size
        self.ratio = ratio 
        self.nratio = nratio
        self.srange = srange
        self.padding = padding

        self.conv1 = ScaleConvolution(self.f_in, 96, self.size, self.ratio, self.nratio, srange = 0, boundary_condition = "dirichlet", padding=self.padding, stride = 2)
        self.conv2 = ScaleConvolution(96, 96, self.size, self.ratio, self.nratio, srange = self.srange, boundary_condition = "dirichlet", padding=self.padding)
        self.conv3 = ScaleConvolution(96, 192, self.size, self.ratio, self.nratio, srange = self.srange, boundary_condition = "dirichlet", padding=self.padding)
        self.pool = ScalePool(self.ratio)
        
        self.fc1 = nn.Linear(192, 150, bias=True)
        self.fc2 = nn.Linear(150, 10, bias=True)

    def forward(self, x): 
        x = x.unsqueeze(1)  # [batch, sigma, feature, y, x]
        x = x.repeat(1, self.nratio, 1, 1, 1)  # [batch, sigma, feature, y, x]
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x) # [batch,feature]
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class SiAllCNN(nn.Module): 
    def __init__(self, f_in, ratio, nratio):
        super().__init__()
        '''
        Squale equivariant All convolutional netw
        '''
        self.f_in = f_in
        self.ratio = ratio 
        self.nratio = nratio

        self.conv1 = ScaleConvolution(self.f_in, 96, 3, ratio=self.ratio, nratio=self.nratio, srange=0, boundary_condition="dirichlet", padding=1)
        self.conv2 = ScaleConvolution(96, 96, 3, self.ratio, self.nratio, srange=2, boundary_condition="dirichlet", padding=1)
        self.conv3 = ScaleConvolution(96, 96, 3, self.ratio, self.nratio, srange=2, boundary_condition="dirichlet", padding=1, stride=2)
        self.conv4 = ScaleConvolution(96, 192, 3, self.ratio, self.nratio, srange=2, boundary_condition="dirichlet", padding=1)
        self.conv5 = ScaleConvolution(192, 192, 3, self.ratio, self.nratio, srange=2, boundary_condition="dirichlet", padding=1)
        self.conv6 = ScaleConvolution(192, 192, 3, self.ratio, self.nratio, srange=2, boundary_condition="dirichlet", padding=1, stride=2)
        self.conv7 = ScaleConvolution(192, 192, 3, self.ratio, self.nratio, srange=2, boundary_condition="dirichlet", padding=1)
        
        self.weight8 = nn.Parameter(torch.empty(192, 192))
        nn.init.orthogonal_(self.weight8)
        self.weight9 = nn.Parameter(torch.empty(10, 192))
        nn.init.orthogonal_(self.weight9)

    def forward(self, x):
        x = x.unsqueeze(1)  # [batch, sigma, feature, y, x]
        x = x.repeat(1, self.nratio, 1, 1, 1) # [batch, sigma, feature, y, x]

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        
        x = torch.einsum("ij,bsjyx->bsiyx", (self.weight8, x))
        x = torch.einsum("ij,bsjyx->bsiyx", (self.weight9, x))

        n_batch = x.size(0)
        n_ratio = x.size(1)
        n_features_in = x.size(2)
        x = x.view(n_batch, n_ratio, n_features_in, -1).mean(-1) # [batch, sigma, feature]
        factors = x.new_tensor([self.ratio ** (-2 * i) for i in range(n_ratio)])
        x = torch.einsum("zsf,s->zf", (x, factors))  # [batch, feature]
       
        return x


class miniSiAll(nn.Module): 
    def __init__(self, f_in, ratio, nratio):
        super().__init__()
        '''
        Smaller version of the squale equivariant All CNN 
        '''
        self.f_in = f_in
        self.ratio = ratio 
        self.nratio = nratio

        self.conv1 = ScaleConvolution(self.f_in, 96, 3, ratio=self.ratio, nratio=self.nratio, srange=0, boundary_condition="dirichlet", padding=1)
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