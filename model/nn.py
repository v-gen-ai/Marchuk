import math

import torch
from torch import nn
import torch.nn.functional as F
try:
    from flash_attn import flash_attn_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    flash_attn_func = None
    FLASH_ATTN_AVAILABLE = False

from .utils import get_freqs


def apply_rotary(x, rope):
    x_ = x.float().reshape(*x.shape[:-1], -1, 1, 2)
    x_out = rope[..., 0] * x_[..., 0] + rope[..., 1] * x_[..., 1]
    return x_out.reshape(*x.shape)


class TimeEmbeddings(nn.Module):

    def __init__(self, dim, max_period=10000.):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
        self.max_period = max_period
        
        self.in_layer = nn.Linear(dim, dim, bias=True)
        self.activation = nn.SiLU()
        self.out_layer = nn.Linear(dim, dim, bias=True)

    def forward(self, time):
        freqs = get_freqs(self.dim // 2, self.max_period).to(dtype=time.dtype, device=time.device)
        args = torch.outer(time, freqs)
        time_embed = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.out_layer(self.activation(self.in_layer(time_embed)))


class TextEmbeddings(nn.Module):

    def __init__(self, text_dim, model_dim):
        super().__init__()
        self.in_layer = nn.Linear(text_dim, model_dim, bias=True)
        self.norm = nn.LayerNorm(model_dim)

    def forward(self, text_embed):
        return self.norm(self.in_layer(text_embed))


class VisualEmbeddings(nn.Module):

    def __init__(self, visual_dim, model_dim, patch_size):
        super().__init__()
        self.patch_size = patch_size
        self.in_layer = nn.Linear(math.prod(patch_size) * visual_dim, model_dim)

    def forward(self, x):
        bs, duration, height, width, dim = x.shape
        x = x.view(
            bs, duration // self.patch_size[0], self.patch_size[0],
            height // self.patch_size[1], self.patch_size[1],
            width // self.patch_size[2], self.patch_size[2], dim
        ).permute(0, 1, 3, 5, 2, 4, 6, 7).flatten(-4, -1)
        return self.in_layer(x)


class UnmergeLayer(nn.Module):

    def __init__(self, visual_dim, model_dim, patch_size):
        super().__init__()
        self.patch_size = patch_size
        self.in_layer = nn.Linear(visual_dim // math.prod(patch_size), model_dim)

    def forward(self, x):
        bs, duration, height, width, dim = x.shape
        x = x.view(
            bs, duration, height, width,
            self.patch_size[0], self.patch_size[1], self.patch_size[2], dim // math.prod(self.patch_size)
        ).permute(0, 1, 4, 2, 5, 3, 6, 7).flatten(1, 2).flatten(2, 3).flatten(3, 4)
        return self.in_layer(x)


class RoPE3D(nn.Module):

    def __init__(self, axes_dims, max_period=10000.):
        super().__init__()
        self.axes_dims = axes_dims
        self.max_period = max_period

    def forward(self, x, scale_factor=(1., 1., 1.)):
        args = []
        for i, (axes, axes_dim, axes_scale_factor) in enumerate(zip(x.shape[1:-1], self.axes_dims, scale_factor)):
            freqs = get_freqs(axes_dim // 2, self.max_period).to(dtype=torch.float32, device=x.device) / axes_scale_factor
            pos = torch.arange(axes, device=freqs.device, dtype=freqs.dtype)
            args.append(torch.outer(pos, freqs))

        duration, height, width = x.shape[1:-1]
        args = torch.cat([
            args[0].view(duration, 1, 1, -1).repeat(1, height, width, 1),
            args[1].view(1, height, 1, -1).repeat(duration, 1, width, 1),
            args[2].view(1, 1, width, -1).repeat(duration, height, 1, 1)
        ], dim=-1)
        rope = torch.stack([torch.cos(args), -torch.sin(args), torch.sin(args), torch.cos(args)], dim=-1)
        rope = rope.view(*rope.shape[:-1], 2, 2)
        return rope.unsqueeze(-4)


class Modulation(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.activation = nn.SiLU()
        self.out_layer = nn.Linear(dim, 9 * dim)
        self.out_layer.weight.data.zero_()
        self.out_layer.bias.data.zero_()

    def forward(self, x):
        modulation_params = self.out_layer(self.activation(x))
        self_attn_params, cross_attn_params, ff_params = torch.chunk(modulation_params.unsqueeze(1), 3, dim=-1)
        return self_attn_params, cross_attn_params, ff_params


class MultiheadSelfAttention(nn.Module):

    def __init__(self, num_channels, head_dim=64, attention_type='flash'):
        super().__init__()
        assert num_channels % head_dim == 0
        self.attention_type = attention_type
        self.num_heads = num_channels // head_dim

        self.to_query_key_value = nn.Linear(num_channels, 3 * num_channels, bias=False)
        self.query_norm = nn.LayerNorm(head_dim)
        self.key_norm = nn.LayerNorm(head_dim)
        
        self.out_layer = nn.Linear(num_channels, num_channels, bias=False)

    def scaled_dot_product_attention(self, query, key, value, **kwargs):
        if self.attention_type == 'flash':
            if FLASH_ATTN_AVAILABLE:
                return flash_attn_func(query, key, value, **kwargs)[0]
            else:
                return F.scaled_dot_product_attention(query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2), **kwargs).transpose(1, 2)

    def forward(self, x, rope):
        query_key_value = self.to_query_key_value(x).reshape(*x.shape[:2], self.num_heads, -1)
        query, key, value = torch.chunk(query_key_value, 3, dim=-1)
        query = self.query_norm(query).type_as(query)
        key = self.key_norm(key).type_as(key)

        query = apply_rotary(query, rope).type_as(query)
        key = apply_rotary(key, rope).type_as(key)
        out = self.scaled_dot_product_attention(query, key, value).flatten(-2, -1)
        out = self.out_layer(out)
        return out


class MultiheadCrossAttention(nn.Module):

    def __init__(self, num_channels, head_dim=64, attention_type='flash'):
        super().__init__()
        assert num_channels % head_dim == 0
        self.attention_type = attention_type
        self.num_heads = num_channels // head_dim

        self.to_query = nn.Linear(num_channels, num_channels, bias=False)
        self.to_key_value = nn.Linear(num_channels, 2 * num_channels, bias=False)
        self.query_norm = nn.LayerNorm(head_dim)
        self.key_norm = nn.LayerNorm(head_dim)
        
        self.out_layer = nn.Linear(num_channels, num_channels, bias=False)

    def scaled_dot_product_attention(self, query, key, value, **kwargs):
        if self.attention_type == 'flash':
            if FLASH_ATTN_AVAILABLE:
                return flash_attn_func(query, key, value, **kwargs)[0]
            else:
                return F.scaled_dot_product_attention(query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2), **kwargs).transpose(1, 2)

    def forward(self, x, text_embed, rope):
        query = self.to_query(x).reshape(*x.shape[:2], self.num_heads, -1)
        key_value = self.to_key_value(text_embed).reshape(*text_embed.shape[:2], self.num_heads, -1)
        key, value = torch.chunk(key_value, 2, dim=-1)
        query = self.query_norm(query).type_as(query)
        key = self.key_norm(key).type_as(key)

        query = apply_rotary(query, rope).type_as(query)
        out = self.scaled_dot_product_attention(query, key, value).flatten(-2, -1)
        out = self.out_layer(out)
        return out


class FeedForward(nn.Module):

    def __init__(self, dim, ff_dim):
        super().__init__()
        self.in_layer = nn.Linear(dim, ff_dim, bias=True)
        self.activation = nn.SiLU()
        self.out_layer = nn.Linear(ff_dim, dim, bias=True)

    def forward(self, x):
        return self.out_layer(self.activation(self.in_layer(x)))


class OutLayer(nn.Module):

    def __init__(self, model_dim, visual_dim, patch_size):
        super().__init__()
        self.patch_size = patch_size
        self.norm =  nn.LayerNorm(model_dim, elementwise_affine=False)
        self.out_layer = nn.Linear(model_dim, math.prod(patch_size) * visual_dim, bias=True)

        self.modulation_activation = nn.SiLU()
        self.modulation_out = nn.Linear(model_dim, 2 * model_dim, bias=True)
        self.modulation_out.weight.data.zero_()
        self.modulation_out.bias.data.zero_()

    def forward(self, visual_embed, time_embed):
        scale, shift = torch.chunk(self.modulation_out(self.modulation_activation(time_embed)), 2, dim=-1)
        visual_embed = self.norm(visual_embed) * (scale[:, None, None, None, :] + 1) + shift[:, None, None, None, :]
        x = self.out_layer(visual_embed)

        bs, duration, height, width, dim = x.shape
        x = x.view(
            bs, duration, height, width,
            self.patch_size[0], self.patch_size[1], self.patch_size[2], dim // math.prod(self.patch_size)
        ).permute(0, 1, 4, 2, 5, 3, 6, 7).flatten(1, 2).flatten(2, 3).flatten(3, 4)
        return x
