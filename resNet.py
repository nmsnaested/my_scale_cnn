# pylint: disable=C,R,E1101
'''
Resnet scale equivariant architecture
'''
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from scale_cnn import ScaleConvolution, ScalePool

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
    def __init__(self, f_in, f_out, size, ratio, nratio, srange, padding, stride=1):
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
    def __init__(self, f_in, size, ratio, nratio, srange, padding=0, nb_classes=10):
        super().__init__()
        self.f_in = f_in
        self.size = size
        self.ratio = ratio
        self.nratio = nratio
        self.srange = srange
        self.padding = padding
        self.nb_classes = nb_classes

        features = [self.f_in, 36, 64]
        repeat = 3

        blocks = []

        f = features[0]
        for f_out in features[1:]:
            for i in range(repeat):
                stride = 2 if i == 0 else 1
                m = Block(f, f_out,self.size,self.ratio,self.nratio,self.srange,self.padding, stride)
                f = f_out
                blocks.append(m)

        self.blocks = nn.ModuleList(blocks)
        self.readout = ScaleConvolution(f, self.nb_classes, self.size, self.ratio, self.nratio, self.srange, padding=self.padding)
        self.pool = ScalePool(self.ratio)


    def forward(self, x):  # pylint: disable=W
        '''
        :param x: [batch, feature, y, x]
        '''
        x = x.unsqueeze(1).repeat(1, self.nratio, 1, 1, 1)  # [batch, sigma, feature, y, x]

        for m in self.blocks:
            x = m(x)  # [batch, sigma, feature, y, x]
            #print(x.mean().item(), x.std().item(), x.size())

        x = self.readout(x)  # [batch, sigma, feature, y, x]
        x = self.pool(x)  # [batch, feature]
        return x



