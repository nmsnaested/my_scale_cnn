#pylint: disable=E1101
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from scale_cnn.convolution import ScaleConvolution
from scale_cnn.pooling import ScalePool

class SiVN(nn.Module):
    def __init__(self, ratio, nratio):
        super().__init__()
        self.ratio = ratio 
        self.nratio = nratio

        #ScaleConvolution(self, features_in, features_out, size, ratio, nratio, srange=0, boundary_condition="dirichlet", padding=0, bias=True, **kwargs)
        self.conv11 = ScaleConvolution(3, 96, 11, ratio=self.ratio, nratio=self.nratio, srange=0, boundary_condition="dirichlet", padding=0, stride=4)
        self.weight12 = nn.Parameter(torch.empty(96, 96))
        nn.init.orthogonal_(self.weight12)
        self.conv13 = ScaleConvolution(96, 96, 3, ratio=self.ratio, nratio=self.nratio, srange=1, boundary_condition="dirichlet", padding=1, stride=2)
        
        self.conv21 = ScaleConvolution(96, 256, 5, ratio=self.ratio, nratio=self.nratio, srange=1, boundary_condition="dirichlet", padding=2, stride=1)
        self.weight22 = nn.Parameter(torch.empty(256, 256))
        nn.init.orthogonal_(self.weight22)
        self.conv23 = ScaleConvolution(256, 256, 3, ratio=self.ratio, nratio=self.nratio, srange=1, boundary_condition="dirichlet", padding=0, stride=2)

        self.conv31 = ScaleConvolution(256, 384, 3, ratio=self.ratio, nratio=self.nratio, srange=1, boundary_condition="dirichlet", padding=1, stride=1)
        self.weight32 = nn.Parameter(torch.empty(384, 384))
        nn.init.orthogonal_(self.weight32)
        self.conv33 = ScaleConvolution(384, 384, 3, ratio=self.ratio, nratio=self.nratio, srange=1, boundary_condition="dirichlet", padding=0, stride=2)
        
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



def wrap(f):
    # does a if the sigma index was a batch index
    def g(x):
        # x: [batch, sigma, feature, y, x]
        nb, ns, nf, h, w = x.size()
        x = x.contiguous().view(nb * ns, nf, h, w)
        x = f(x)
        _, nf, h, w = x.size()
        return x.view(nb, ns, nf, h, w)
    return g


class Block(nn.Module):
    def __init__(self, f_in, f_out,size,ratio,nratio,srange,padding,stride=1):
        super().__init__()

        self.conv1 = ScaleConvolution(f_in, f_out, size, ratio, nratio, srange, padding=padding,stride=stride)
        self.bn1 = nn.BatchNorm2d(f_out)
        self.conv2 = ScaleConvolution(f_out, f_out, size, ratio, nratio, srange, padding=padding)
        self.bn2 = nn.BatchNorm2d(f_out)

        if f_in != f_out:
            self.weight = nn.Parameter(torch.empty(f_out, f_in))
            nn.init.orthogonal_(self.weight)
        else:
            self.register_parameter("weight", None)


    def forward(self, x):  # pylint: disable=W
        '''
        :param x: [batch, sigma, feature, y, x]
        '''
        assert x.ndimension() == 5

        s = x
        bn1 = wrap(self.bn1)
        x = F.relu(bn1(self.conv1(x)))  # [batch, sigma, feature, y, x]
        x = self.conv2(x)  # [batch, sigma, feature, y, x]

        if self.weight is not None:
            s = torch.einsum("ij,bsjyx->bsiyx", (self.weight, s))

        pool = wrap(partial(F.adaptive_avg_pool2d, output_size=(x.size(-2), x.size(-1))))
        s = pool(s)  # [batch, sigma, feature, y, x]

        bn2 = wrap(self.bn2)
        return F.relu(bn2(0.5 * x + s))  # [batch, sigma, feature, y, x]


class Model(nn.Module):
    def __init__(self,size,ratio,nratio,srange,padding):
        super().__init__()
        self.size = size
        self.ratio = ratio
        self.nratio = nratio
        self.srange = srange
        self.padding = padding

        features = [3, 16, 32]
        repreat = 3

        blocks = []

        f = features[0]
        for f_out in features[1:]:
            for i in range(repreat):
                stride = 2 if i == 0 else 1
                m = Block(f, f_out,self.size,self.ratio,self.nratio,self.srange,self.padding, stride)
                f = f_out
                blocks.append(m)

        self.blocks = nn.ModuleList(blocks)
        self.readout = ScaleConvolution(f, 210, self.size, self.ratio, self.nratio, self.srange, padding=self.padding)
        self.pool = ScalePool(self.ratio)


    def forward(self, x):  # pylint: disable=W
        '''
        :param x: [batch, feature, y, x]
        '''

        # the following line is scale+translation equivariant
        x = x.unsqueeze(1).repeat(1, self.nratio, 1, 1, 1)  # [batch, sigma, feature, y, x]

        for m in self.blocks:
            x = m(x)  # [batch, sigma, feature, y, x]
            #print(x.mean().item(), x.std().item(), x.size())

        x = self.readout(x)  # [batch, sigma, feature, y, x]
        x = self.pool(x)  # [batch, feature]
        return x


class SiCNN(nn.Module): 
    def __init__(self,ratio,nratio): 
        super().__init__()
        
        self.ratio = ratio 
        self.nratio = nratio

        self.conv1 = ScaleConvolution(3, 36, 3, self.ratio, self.nratio, srange = 0, boundary_condition = "neumann", stride = 2)
        self.conv2 = ScaleConvolution(36, 64, 3, self.ratio, self.nratio, srange = 1, boundary_condition = "neumann")
        self.pool = ScalePool(self.ratio)
        
        self.fc1 = nn.Linear(64, 150, bias=True)
        self.fc2 = nn.Linear(150, 210, bias=True)

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


class SiAllCNN(nn.Module): 
    def __init__(self, ratio, nratio):
        super().__init__()
        self.ratio = ratio 
        self.nratio = nratio

        #ScaleConvolution(self, features_in, features_out, size, ratio, nratio, srange=0, boundary_condition="dirichlet", padding=0, bias=True, **kwargs)
        self.conv1 = ScaleConvolution(3, 96, 3, ratio=self.ratio, nratio=self.nratio, srange=0, boundary_condition="dirichlet", padding=1)
        self.conv2 = ScaleConvolution(96, 96, 3, self.ratio, self.nratio, srange=2, boundary_condition="dirichlet", padding=1)
        self.conv3 = ScaleConvolution(96, 96, 3, self.ratio, self.nratio, srange=2, boundary_condition="dirichlet", padding=1, stride=2)
        self.conv4 = ScaleConvolution(96, 192, 3, self.ratio, self.nratio, srange=2, boundary_condition="dirichlet", padding=1)
        self.conv5 = ScaleConvolution(192, 192, 3, self.ratio, self.nratio, srange=2, boundary_condition="dirichlet", padding=1)
        self.conv6 = ScaleConvolution(192, 192, 3, self.ratio, self.nratio, srange=2, boundary_condition="dirichlet", padding=1, stride=2)
        self.conv7 = ScaleConvolution(192, 192, 3, self.ratio, self.nratio, srange=2, boundary_condition="dirichlet", padding=1)
        #self.conv8 = ScaleConvolution(192, 192, 1, self.ratio, self.nratio, srange=1, boundary_condition="dirichlet")
        #self.conv9 = ScaleConvolution(192, 10, 1, self.ratio, self.nratio, srange=1, boundary_condition="dirichlet")
        self.weight8 = nn.Parameter(torch.empty(192, 192))
        nn.init.orthogonal_(self.weight8)
        self.weight9 = nn.Parameter(torch.empty(210, 192))
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
        #x = F.relu(self.conv8(x))
        #x = F.relu(self.conv9(x))
        x = torch.einsum("ij,bsjyx->bsiyx", (self.weight8, x))
        x = torch.einsum("ij,bsjyx->bsiyx", (self.weight9, x))

        n_batch = x.size(0)
        n_ratio = x.size(1)
        n_features_in = x.size(2)
        x = x.view(n_batch, n_ratio, n_features_in, -1).mean(-1) # [batch, sigma, feature]
        factors = x.new_tensor([self.ratio ** (-2 * i) for i in range(n_ratio)])
        x = torch.einsum("zsf,s->zf", (x, factors))  # [batch, feature]
       
        return x