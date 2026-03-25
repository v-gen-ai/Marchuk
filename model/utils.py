import math

import torch


def get_freqs(dim, max_period=10000.):
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=dim, dtype=torch.float32) / dim
    )
    return freqs


def get_tensor_items(x, pos, broadcast_shape):
    bs = pos.shape[0]
    ndims = len(broadcast_shape[1:])
    x = x[pos].reshape(bs, *((1,) * ndims))
    return x


def get_real_group_size(group_size, duration, height, width):
    g1, g2, g3 = group_size
    g1 = duration if g1 == -1 else min(duration, g1)
    g2 = height if g2 == -1 else min(height, g2)
    g3 = width if g3 == -1 else min(width, g3)
    return (g1, g2, g3)


def group_repeat(x, group_size, duration, height, width, dim=0):
    g1, g2, g3 = get_real_group_size(group_size, duration, height, width)
    repeats = (duration//g1) * (height//g2) * (width//g3)
    x = torch.repeat_interleave(x.unsqueeze(dim), repeats, dim=dim)
    return x


def local_patching(x, group_size, duration, height, width, dim=0):
    g1, g2, g3 = group_size
    x = x.reshape(*x.shape[:dim], duration//g1, g1, height//g2, g2, width//g3, g3, *x.shape[dim+3:])
    x = x.permute(
        *range(len(x.shape[:dim])),
        dim, dim+2, dim+4, dim+1, dim+3, dim+5, 
        *range(dim+6, len(x.shape))
    )
    x = x.flatten(dim, dim+2).flatten(dim+1, dim+3)
    return x


def local_merge(x, group_size, duration, height, width, dim=0):
    g1, g2, g3 = group_size
    x = x.reshape(*x.shape[:dim], duration//g1, height//g2, width//g3, g1, g2, g3, *x.shape[dim+2:])
    x = x.permute(
        *range(len(x.shape[:dim])),
        dim, dim+3, dim+1, dim+4, dim+2, dim+5, 
        *range(dim+6, len(x.shape))
    )
    x = x.flatten(dim, dim+1).flatten(dim+1, dim+2).flatten(dim+2, dim+3)
    return x


def global_patching(x, group_size, duration, height, width, dim=0):
    g1, g2, g3 = group_size
    x = local_patching(x, (duration // g1, height // g2, width // g3), duration, height, width, dim)
    x = x.transpose(dim, dim+1)
    return x


def global_merge(x, group_size, duration, height, width, dim=0):
    g1, g2, g3 = group_size
    x = x.transpose(dim, dim+1)
    x = local_merge(x, (duration // g1, height // g2, width // g3), duration, height, width, dim)
    return x
