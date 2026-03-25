import torch
from torch import nn

from .nn import (TimeEmbeddings, VisualEmbeddings, RoPE3D, MultiheadSelfAttention,
                MultiheadCrossAttention, FeedForward, OutLayer)
from .utils import get_real_group_size, local_patching, local_merge, global_patching, global_merge


class TransformerBlock(nn.Module):

    def __init__(
            self, model_dim, ff_dim, head_dim=64, group_size=(-1, -1, -1), attention_type='local', lora_coef=256
    ):
        super().__init__()
        assert attention_type in ['local', 'global']
        self.self_attn_group_size = group_size
        self.cross_attn_group_size = (1, -1, -1)
        self.attention_type = attention_type

        self.time_lora_layer1 = nn.Linear(model_dim, lora_coef, bias=False)
        self.time_lora_layer2 = nn.Linear(lora_coef, 9 * model_dim, bias=False)
        self.time_lora_activation = nn.SiLU()
        self.time_lora_layer1.weight.data.zero_()
        self.time_lora_layer2.weight.data.zero_()

        self.self_attention_norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.self_attention = MultiheadSelfAttention(model_dim, head_dim)

        self.cross_attention_norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.cross_attention = MultiheadCrossAttention(model_dim, head_dim)

        self.feed_forward_norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.feed_forward = FeedForward(model_dim, ff_dim)

    def to_1dimension(self, visual_embed, rope, duration, height, width, attn_type):
        if attn_type == 'self':
            group_size = get_real_group_size(self.self_attn_group_size, duration, height, width)
        elif attn_type == 'cross':
            group_size = get_real_group_size(self.cross_attn_group_size, duration, height, width)
            
        if self.attention_type == 'local':
            visual_embed = local_patching(visual_embed, group_size, duration, height, width, dim=1)
            rope = local_patching(rope, group_size, duration, height, width, dim=0)
        if self.attention_type == 'global':
            visual_embed = global_patching(visual_embed, group_size, duration, height, width, dim=1)
            rope = global_patching(rope, group_size, duration, height, width, dim=0)

        visual_embed = visual_embed.flatten(0, 1)
        if rope.shape[0] != visual_embed.shape[0]:
            rope = rope.unsqueeze(0).repeat(
                visual_embed.shape[0]//rope.shape[0], 1, *((1,)*len(rope.shape[1:]))
            ).flatten(0, 1)
        return visual_embed, rope

    def to_3dimension(self, x, duration, height, width, attn_type):
        if attn_type == 'self':
            group_size = get_real_group_size(self.self_attn_group_size, duration, height, width)
        elif attn_type == 'cross':
            group_size = get_real_group_size(self.cross_attn_group_size, duration, height, width)

        if self.attention_type == 'local':
            x = local_merge(x, group_size, duration, height, width, dim=1)
        if self.attention_type == 'global':
            x = global_merge(x, group_size, duration, height, width, dim=1)
        return x

    def forward(self, visual_embed, time_embed, time_embed_modulation, condition_embed, rope):
        bs, duration, height, width = visual_embed.shape[:-1]
        lora_time_addition = self.time_lora_layer2(self.time_lora_layer1(self.time_lora_activation(time_embed)))
        time_embed_modulation = time_embed_modulation + lora_time_addition
        self_attn_params, cross_attn_params, ff_params = torch.chunk(time_embed_modulation.unsqueeze(1), 3, dim=-1)

        visual_embed, self_attn_rope = self.to_1dimension(visual_embed, rope, duration, height, width, 'self')
        if self_attn_params.shape[0] != visual_embed.shape[0]:
            self_attn_params = self_attn_params.unsqueeze(1).repeat(
                1, visual_embed.shape[0]//self_attn_params.shape[0], *((1,)*len(self_attn_params.shape[1:]))
            ).flatten(0, 1)

        scale, shift, gate = torch.chunk(self_attn_params, 3, dim=-1)
        visual_out = self.self_attention_norm(visual_embed) * (scale + 1.) + shift
        visual_embed = visual_embed + gate * self.self_attention(visual_out, self_attn_rope)

        visual_embed = visual_embed.reshape(bs, -1, *visual_embed.shape[1:])
        visual_embed = self.to_3dimension(visual_embed, duration, height, width, 'self')

        visual_embed, cross_attn_rope = self.to_1dimension(visual_embed, rope, duration, height, width, 'cross')

        if cross_attn_params.shape[0] != visual_embed.shape[0]:
            cross_attn_params = cross_attn_params.unsqueeze(1).repeat(
                1, visual_embed.shape[0]//cross_attn_params.shape[0], *((1,)*len(cross_attn_params.shape[1:]))
            ).flatten(0, 1)

        if ff_params.shape[0] != visual_embed.shape[0]:
            ff_params = ff_params.unsqueeze(1).repeat(
                1, visual_embed.shape[0]//ff_params.shape[0], *((1,)*len(ff_params.shape[1:]))
            ).flatten(0, 1)

        if  condition_embed.shape[0] != visual_embed.shape[0]:
            condition_embed = condition_embed.unsqueeze(1).repeat(
                1, visual_embed.shape[0]//condition_embed.shape[0], *((1,)*len(condition_embed.shape[1:]))
            ).flatten(0, 1)

        scale, shift, gate = torch.chunk(cross_attn_params, 3, dim=-1)
        visual_out = self.cross_attention_norm(visual_embed) * (scale + 1.) + shift
        visual_embed = visual_embed + gate * self.cross_attention(visual_out, torch.cat([visual_out, condition_embed], dim=1), cross_attn_rope)


        scale, shift, gate = torch.chunk(ff_params, 3, dim=-1)
        visual_out = self.feed_forward_norm(visual_embed) * (scale + 1.) + shift
        visual_embed = visual_embed + gate * self.feed_forward(visual_out)

        visual_embed = visual_embed.reshape(bs, -1, *visual_embed.shape[1:])
        visual_embed = self.to_3dimension(visual_embed, duration, height, width, 'cross')
        return visual_embed


class DiffusionTransformer3D(nn.Module):

    def __init__(
        self,
        in_visual_dim=4,
        out_visual_dim=4,
        patch_size=(1, 2, 2),
        model_dim=2048,
        ff_dim=5120,
        num_blocks=8,
        axes_dims=(16, 24, 24)
    ):
        super().__init__()
        head_dim = sum(axes_dims)
        self.model_dim = model_dim
        self.num_blocks = num_blocks
        self.patch_size = patch_size

        self.time_embeddings = TimeEmbeddings(model_dim)

        self.visual_embeddings = VisualEmbeddings(in_visual_dim, model_dim, patch_size)
        self.rope_embeddings = RoPE3D(axes_dims)

        self.transformer_blocks = nn.ModuleList([
            nn.ModuleList([
                TransformerBlock(model_dim, ff_dim, head_dim, (1, -1, -1), 'local'),
                TransformerBlock(model_dim, ff_dim, head_dim, (-1, -1, -1), 'local'),
            ]) for _ in range(num_blocks)
        ])

        self.out_layer = OutLayer(model_dim, out_visual_dim, patch_size)

        ## modulation
        self.activation_modulation = nn.SiLU()
        self.out_layer_modulation = nn.Linear(model_dim, 9 * model_dim)
        self.out_layer_modulation.weight.data.zero_()
        self.out_layer_modulation.bias.data.zero_()

        ## condition
        self.conditional_embeds = nn.Embedding(8785, model_dim)
        self.activation_cond = nn.SiLU()
        self.out_layer_cond = nn.Linear(model_dim, model_dim)

        self.trainable_embeddings = nn.Parameter(torch.ones((1, 15, 30, 1, 32, 2, 2)))


    def forward(self, x, condition, time, scale_factor=(1., 1., 1.)):

        condition_embed = self.conditional_embeds(condition)
        condition_embed = self.out_layer_cond(self.activation_cond(condition_embed))
        time_embed = self.time_embeddings(time)
        time_embed_modulation = self.out_layer_modulation(self.activation_modulation(time_embed))

        visual_embed = self.visual_embeddings(x)
        rope = self.rope_embeddings(visual_embed, scale_factor)[:, 0:1, 0:1, :]
        rope = rope * self.trainable_embeddings

        for layer_index, (local_attention, global_attention) in enumerate(self.transformer_blocks):
            visual_embed = local_attention(visual_embed, time_embed, time_embed_modulation, condition_embed, rope)
            visual_embed = global_attention(visual_embed, time_embed, time_embed_modulation, condition_embed, rope)


        return self.out_layer(visual_embed, time_embed)


def get_dit(conf):
    dit = DiffusionTransformer3D(**conf)
    return dit
