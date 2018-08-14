# pylint: disable=C,R,E1101,E1102
import torch
import torch.nn as nn


class ScalePool(nn.Module):
    def __init__(self, ratio):
        '''
        pytorch scale+translation equivariant pooling module

        :param ratio: scale ratio between each channel
        '''
        super().__init__()

        self.ratio = ratio

    def __repr__(self):
        return self.__class__.__name__

    def forward(self, input):  # pylint: disable=W
        '''
        :param input: [batch, sigma, f, y, x]
        '''
        nbatch = input.size(0)
        nratio = input.size(1)
        nfeatures_in = input.size(2)

        input = input.view(nbatch, nratio, nfeatures_in, -1).sum(-1)  # [batch, sigma, f]
        factors = input.new_tensor([self.ratio ** (-2 * i) for i in range(nratio)])
        return torch.einsum("zsf,s->zf", (input, factors))  # [batch, f]
