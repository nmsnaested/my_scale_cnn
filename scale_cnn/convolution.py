# pylint: disable=C,R,E1101,E1102
import torch
import torch.nn as nn
import torch.nn.functional as F
from scale_cnn.scale import bilinear_matrix, low_pass_filter
import math


class ScaleConvolution(nn.Module):
    def __init__(self, features_in, features_out, size, ratio, nratio, srange=0, boundary_condition="dirichlet", padding=0, bias=True, **kwargs):
        '''
        pytorch scale+translation equivariant convolution module

        :param features_in: input features
        :param features_out: output features
        :param size: filter size
        :param ratio: scale ratio between each channel
        :param nratio: amount of channels
        :param srange: channels overlap during convolution
        :param boundary_condition: neumann or dirichlet for the overlap behaviors
        '''
        super().__init__()

        bilinear_matricies = []
        sizes = []
        for i in range(nratio):
            scale = ratio**i
            filter_size = math.ceil(size * scale)
            if filter_size % 2 != size % 2:
                filter_size += 1
            sizes.append(filter_size)

            m = bilinear_matrix(size, filter_size, scale)
            m = m * ratio ** (-2 * i)
            m = m.view(size**2, filter_size**2)
            m = m.float()
            bilinear_matricies.append(m)

        self.sizes = sizes
        self.bilinear_matricies = bilinear_matricies

        self.size = size
        self.ratio = ratio
        self.nratio = nratio
        self.srange = srange
        self.nf_in = features_in
        self.nf_out = features_out
        self.boundary_condition = boundary_condition
        self.padding = padding
        self.weight = nn.Parameter(torch.empty(features_out, 1 + 2 * srange, features_in, size * size, dtype=torch.float))
        if bias:
            self.bias = nn.Parameter(torch.zeros(features_out))
        else:
            self.register_parameter('bias', None)

        self.kwargs = kwargs
        self.reset_parameters()

    def reset_parameters(self):
        std = 1 / math.sqrt((1 + 2 * self.srange) * self.nf_in * self.size**2)  # 1 / sqrt(amount of input data)
        std *= math.sqrt(self.nratio)  # empirical
        self.weight.data.normal_(0, std).clamp_(-2 * std, 2 * std)

    def __repr__(self):
        return "{} (size={}, {} → {}, n={}±{}, {})".format(
            self.__class__.__name__,
            self.size,
            self.nf_in,
            self.nf_out,
            self.nratio,
            self.srange,
            self.boundary_condition)

    def forward(self, input):  # pylint: disable=W
        '''
        :param input: [batch, sigma, f_in, y, x]
        '''
        for i in range(self.nratio):
            self.bilinear_matricies[i] = self.bilinear_matricies[i].to(self.weight.device)

        nbatch = input.size(0)
        assert self.nratio == input.size(1)
        assert self.nf_in == input.size(2)

        outputs = []
        for i in range(self.nratio):
            padding = self.padding + (self.sizes[i] - min(self.sizes)) // 2

            weight = low_pass_filter(self.weight, self.ratio ** i)

            # upscale (downscale) the filter
            # TODO remove the .clone() see pytorch/issues/7763
            kernels = torch.einsum("odis,sz->odiz", (weight.clone(), self.bilinear_matricies[i]))  # [f_out, delta, f_in, y * x]
            kernels = kernels.view(self.nf_out, 1 + 2 * self.srange, self.nf_in, self.sizes[i], self.sizes[i])  # [f_out, delta, f_in, y, x]

            # perform the convolution
            i_prime_begin = max(0, i - self.srange)
            num = 1 + min(self.srange, self.nratio - i - 1) + min(i, self.srange)

            if self.boundary_condition == "dirichlet":
                # Not optimized code:
                # output = 0
                # for j, delta in enumerate(range(-self.srange, self.srange + 1)):
                #     i_prime = i + delta
                #     if i_prime < 0 or i_prime >= self.nratio:
                #         continue
                #     output += F.conv2d(input[:, i_prime], kernels[:, j], self.bias, padding=padding, **self.kwargs)

                j_begin = max(self.srange - i, 0)

                output = F.conv2d(
                    input[:, i_prime_begin: i_prime_begin + num].contiguous().view(nbatch, num * self.nf_in, input.size(3), input.size(4)),
                    kernels[:, j_begin: j_begin + num].contiguous().view(self.nf_out, num * self.nf_in, self.sizes[i], self.sizes[i]),
                    self.bias,
                    padding=padding,
                    **self.kwargs
                )

            elif self.boundary_condition == "neumann":
                # Not optimized code:
                # output = 0
                # for j, delta in enumerate(range(-self.srange, self.srange + 1)):
                #     output += F.conv2d(
                #         input[:, max(0, min(i + delta, self.nratio - 1))],
                #         kernels[:, j],
                #         self.bias,
                #         padding=padding,
                #         **self.kwargs
                #     )

                excess_bot = max(0, self.srange - i)
                excess_top = max(0, self.srange + i - self.nratio + 1)

                image = torch.cat(
                    ([input[:, 0].repeat(1, excess_bot, 1, 1)] if excess_bot > 0 else []) +
                    [input[:, i_prime_begin: i_prime_begin + num].contiguous().view(nbatch, num * self.nf_in, input.size(3), input.size(4))] +
                    ([input[:, -1].repeat(1, excess_top, 1, 1)] if excess_top > 0 else []), dim=1
                )  # [batch, sigma * f_in, y, x]

                output = F.conv2d(
                    image,
                    kernels.view(self.nf_out, (1 + 2 * self.srange) * self.nf_in, self.sizes[i], self.sizes[i]),
                    self.bias,
                    padding=padding,
                    **self.kwargs
                )

            else:
                raise ValueError("boundary_condition must be dirichlet or neumann")

            output = output.view(nbatch, 1, self.nf_out, output.size(-2), output.size(-1))
            outputs.append(output)

        output = torch.cat(outputs, dim=1)

        return output
