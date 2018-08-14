# pylint: disable=C,R,E1101
import torch
import torch.nn.functional as F


def bilinear_matrix(input_size, output_size, scale):
    """
    :param input_size: int
    :param output_size: int
    :param scale: float

    :return: [in_y, in_x, out_y, out_x] (input_size, input_size, output_size, output_size)
    """

    # input_grid = torch.arange(0, input_size)
    output_grid = torch.arange(-(output_size - 1) / 2, (output_size - 1) / 2 + 1) / scale + (input_size - 1) / 2
    # len(output_grid) == output_size, output_grid[0] ~ 0 and output_grid[-1] ~ input_size-1

    matrix = torch.zeros((input_size, input_size, output_size, output_size))

    def set_value(iy, ix, jy, jx, value):
        if ix >= 0 and ix < input_size and iy >= 0 and iy < input_size:
            matrix[iy, ix, jy, jx] = value

    for jy, y in enumerate(output_grid):
        for jx, x in enumerate(output_grid):
            fx = int(x.floor().item())
            fy = int(y.floor().item())
            dx = x - fx
            dy = y - fy

            set_value(fy,     fx,     jy, jx, (1 - dx) * (1 - dy))
            set_value(fy,     fx + 1, jy, jx, dx       * (1 - dy))
            set_value(fy + 1, fx,     jy, jx, (1 - dx) * dy      )
            set_value(fy + 1, fx + 1, jy, jx, dx       * dy      )

    return matrix


def bilinear_resize(image, output_size, scale):
    """
    :param image: [..., y, x]
    :param output_size: int
    :param scale: float
    """
    assert image.size(-1) == image.size(-2)
    input_size = image.size(-1)

    M = bilinear_matrix(input_size, output_size, scale).to(image.device)
    scaled_image = image.view(-1, input_size ** 2) @ M.view(input_size ** 2, output_size ** 2)
    scaled_image = scaled_image.view(*image.size()[:-2], output_size, output_size)
    return scaled_image


def low_pass_filter(image, scale):
    """
    :param image: [..., y, x]
    :param scale: float
    """
    if scale >= 1:
        return image

    dtype = image.dtype
    device = image.device

    sigma = 0.5 * (1 / scale ** 2 - 1) ** 0.5

    size = int(1 + 2 * 2.5 * sigma)
    if size % 2 == 0:
        size += 1

    rng = torch.arange(size, dtype=dtype, device=device) - size // 2  # [-(size // 2), ..., size // 2]
    x = rng.view(1, size).expand(size, size)
    y = rng.view(size, 1).expand(size, size)
    kernel = torch.exp(- (x ** 2 + y ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()

    out = F.conv2d(image.view(-1, 1, image.size(-2), image.size(-1)), kernel.view(1, 1, size, size), padding=size//2)
    out = out.view(*image.size())
    return out


def resize(image, output_size, scale):
    """
    :param image: [..., y, x]
    :param output_size: int
    :param scale: float
    """
    image = low_pass_filter(image, scale)
    return bilinear_resize(image, output_size, scale)
