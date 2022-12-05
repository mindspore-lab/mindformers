# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file was refer to project:
# https://github.com/openai/CLIP/blob/main/clip/clip.py
# ============================================================================

'''
Modiles of ClipModel, including MultiheadAttention，VisionTransformer,
QuickGELU, ResidualAttentionBlock, Transformer
'''
from collections import OrderedDict
import numpy as np

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.ops import operations as P
from mindspore import Parameter, Tensor
from mindspore.common.initializer import Normal


class MultiheadAttention(nn.Cell):
    '''
    MultiheadAttention, with layers as input for initialization

    Args:
        d_model (int): the feature dimension
        n_head (int): the number of attention heads
        layers (int): the number of transformers, used for weight initialization
    '''
    def __init__(self, d_model, n_head, layers):
        super(MultiheadAttention, self).__init__()

        self.num_heads = n_head
        self.head_dim = d_model // n_head

        self.softmax = nn.Softmax(-1)
        self.transpose = ops.Transpose()
        self.scaling = self.head_dim ** -0.5

        proj_std = (d_model ** -0.5) * ((2 * layers) ** -0.5)
        attn_std = d_model ** -0.5
        self.out_proj = nn.Dense(d_model, d_model,
                                 weight_init=Normal(mean=0.0, sigma=proj_std))
        self.in_proj = nn.Dense(d_model, 3 * d_model,
                                weight_init=Normal(mean=0.0, sigma=attn_std))

    def construct(self, query, attn_mask=None):
        '''construct'''
        len_tgt, batch_size, width = query.shape
        qkv = self.in_proj(query).view(len_tgt, batch_size, 3, width).transpose((2, 0, 1, 3))

        att_q = qkv[0:1]
        att_q = ops.Squeeze(0)(att_q)
        att_q = att_q * self.scaling
        att_q = att_q.view(len_tgt, batch_size * self.num_heads,
                           self.head_dim).transpose((1, 0, 2))

        att_k = qkv[1:2]
        att_k = ops.Squeeze(0)(att_k)
        att_k = att_k.view(-1, batch_size * self.num_heads,
                           self.head_dim).transpose((1, 0, 2))

        att_v = qkv[2:3]
        att_v = ops.Squeeze(0)(att_v)
        att_v = att_v.view(-1, batch_size * self.num_heads,
                           self.head_dim).transpose((1, 0, 2))

        if attn_mask is not None:
            attn_output_weights = (
                attn_mask.astype(ms.float32)
                + ops.matmul(att_q, att_k.transpose((0, 2, 1)))
                .astype(ms.float32)).astype(ms.float64)
        else:
            attn_output_weights = ops.matmul(att_q, att_k.transpose((0, 2, 1)))
        attn_output_weights = self.softmax(attn_output_weights
                                           .astype(ms.float32)).astype(ms.float64)
        attn_output = ops.matmul(attn_output_weights, att_v)
        attn_output = self.transpose(attn_output, (1, 0, 2))
        attn_output = attn_output.view(len_tgt, batch_size, width)
        attn_output = self.out_proj(attn_output)
        return attn_output


class VisionTransformer(nn.Cell):
    '''
    VisionTransformer of ClipModel

    Args:
        input_resolution (int): the image size of input
        patch_size (int): the patch size of vision transformer
        width (int): the dimension of vision transformer
        layers (int): the number of layers of vision transformer
        heads (int): the number of attention heads
        output_dim (int): the output dimension of vision transformer
    '''
    def __init__(self, input_resolution, patch_size, width, layers, heads, output_dim):
        super(VisionTransformer, self).__init__()
        self.conv1 = \
            nn.Conv2d(
                in_channels=3, out_channels=width, kernel_size=patch_size,
                stride=patch_size, has_bias=False)

        scale = width ** -0.5
        self.class_embedding = \
            Parameter(scale * Tensor(np.random.normal(0, 1, size=(width)).astype(np.float32)))
        self.positional_embedding = \
            Parameter(scale * Tensor(
                np.random.normal(0, 1, size=(
                    (input_resolution // patch_size) ** 2 + 1, width)).astype(np.float32)))
        self.ln_pre = nn.LayerNorm([width])
        self.transformer = Transformer(width, layers, heads)
        self.ln_post = nn.LayerNorm([width])
        self.proj = \
            Parameter(scale * Tensor(np.random.normal(0, 1,
                                                      size=(width, output_dim)).astype(np.float32)))
        self.cat = ops.Concat(1)
        self.tile = ops.Tile()
        self.slice = P.StridedSlice()

    def construct(self, input_x):
        '''construct'''
        input_x = self.conv1(input_x)
        input_x = input_x.reshape(input_x.shape[0], input_x.shape[1], -1)
        input_x = input_x.transpose(0, 2, 1)
        class_embedding = self.tile(self.class_embedding, (input_x.shape[0]*2, 1, 1))
        input_x = self.cat([
            self.slice(class_embedding,
                       (0, 0, 0),
                       (input_x.shape[0], class_embedding.shape[1], class_embedding.shape[2]),
                       (1, 1, 1)),
            input_x
        ])
        input_x = ops.Add()(input_x, self.positional_embedding)
        input_x = self.ln_pre(input_x)
        input_x = input_x.transpose(1, 0, 2)
        input_x = self.transformer(input_x)
        input_x = input_x.transpose(1, 0, 2)
        input_x = self.ln_post(input_x[:, 0, :])
        input_x = ops.matmul(input_x, self.proj)
        return input_x


class QuickGELU(nn.Cell):
    '''QuickGELU of Clip'''
    def __init__(self, ratio=1.702):
        super(QuickGELU, self).__init__()
        self.ratio = ratio
        self.sigmoid = nn.Sigmoid()

    def construct(self, input_x):
        '''construct'''
        return input_x * self.sigmoid(self.ratio * input_x)


class ResidualAttentionBlock(nn.Cell):
    '''
    ResidualAttentionBlock of Clip

    Args:
        d_model (int): the dimension of features
        n_head (int): the number of attention heads
        layers (int): the number of transformer layers for weight initialization
        attn_mask (tensor): attention mask
    '''
    def __init__(self, d_model, n_head, layers, attn_mask=None):
        super(ResidualAttentionBlock, self).__init__()

        proj_std = (d_model ** -0.5) * ((2 * layers) ** -0.5)
        fc_std = (2 * d_model) ** -0.5
        self.attn = MultiheadAttention(d_model, n_head, layers)
        self.ln_1 = nn.LayerNorm([d_model], epsilon=1e-5)
        self.mlp = nn.SequentialCell(OrderedDict([
            ("c_fc", nn.Dense(d_model, d_model * 4, weight_init=Normal(mean=0.0, sigma=fc_std))),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Dense(d_model * 4, d_model, weight_init=Normal(mean=0.0, sigma=proj_std)))
        ]))
        self.ln_2 = nn.LayerNorm([d_model], epsilon=1e-5)
        self.attn_mask = attn_mask

    def construct(self, input_x):
        '''construct'''
        input_x = ops.Add()(input_x, self.attention(self.ln_1(input_x)))
        input_x = ops.Add()(input_x, self.mlp(self.ln_2(input_x)))
        return input_x

    def attention(self, input_x):
        '''attention'''
        return self.attn(input_x, self.attn_mask)


class Transformer(nn.Cell):
    '''
    Text Transformer of Clip

    Args:
        width (int): the dimension of input features
        layers (int): the number of transformer layers
        heads (int): the number of attention heads
        attn_mask (tensor):  attention mask
    '''
    def __init__(self, width, layers, heads, attn_mask=None):
        super(Transformer, self).__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.SequentialCell(
            *[ResidualAttentionBlock(width, heads, layers, attn_mask) for _ in range(layers)]
        )

    def construct(self, input_x):
        '''construct'''
        return self.resblocks(input_x)
