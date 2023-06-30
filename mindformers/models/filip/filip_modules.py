# Copyright 2023 Huawei Technologies Co., Ltd
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

"""
Modules of FilipModel, including image encoder model, text encoder model,
and token learner
"""

import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Parameter, Tensor
from mindspore.ops.primitive import constexpr
from mindspore.common.initializer import TruncatedNormal, initializer
from mindformers.modules.transformer import Transformer
from mindformers.modules import TransformerOpParallelConfig, TransformerRecomputeConfig


class QuickGELU(nn.Cell):
    """QuickGELU of Filip"""
    def __init__(self):
        super(QuickGELU, self).__init__()
        self.ratio = 1.702
        self.sigmoid = nn.Sigmoid()

    def construct(self, x):
        return x * self.sigmoid(self.ratio * x)


class VisualTransformer(nn.Cell):
    """VisualTransformer of Filip"""
    def __init__(self, config):
        super(VisualTransformer, self).__init__()
        self.batch_size = config.batch_size
        self.single_bs = self.batch_size == 1
        self.input_resolution = config.vision_config.image_size
        self.patch_size = config.vision_config.patch_size
        self.width = config.vision_config.hidden_size
        self.layers = config.vision_config.num_hidden_layers
        self.num_tokens_side = self.input_resolution // self.patch_size
        self.output_dim = config.projection_dim
        self.heads = config.vision_config.num_heads
        if not self.heads:
            self.heads = self.width // config.ratio
        else:
            self.heads = int(self.heads)
        self.token_learner_flag = config.vision_config.token_learner
        self.output_dim = config.projection_dim
        self.conv1 = nn.Conv2d(3, self.width, self.patch_size, self.patch_size)
        scale = self.width ** -0.5
        src_seq_length = (self.input_resolution // self.patch_size) ** 2 + 1
        self.class_embedding = Parameter(scale * Tensor(np.random.normal(0, 1, size=(self.width)).astype(np.float32)))
        self.positional_embedding = Parameter(scale * Tensor(
            np.random.normal(size=(src_seq_length, self.width)).astype(np.float32)))
        self.ln_pre = nn.LayerNorm([self.width])

        parallel_config = TransformerOpParallelConfig()
        if config.recompute:
            recompute_config = TransformerRecomputeConfig(recompute=False)
            parallel_config = TransformerOpParallelConfig(recompute=recompute_config)
        self.transformer = Transformer(hidden_size=self.width,
                                       batch_size=self.batch_size,
                                       num_heads=self.heads,
                                       ffn_hidden_size=self.width * 4,
                                       encoder_layers=self.layers,
                                       attention_dropout_rate=0.0,
                                       hidden_dropout_rate=0.0,
                                       decoder_layers=0,
                                       # hidden_act=QuickGELU,
                                       hidden_act='fast_gelu',
                                       layernorm_compute_type=ms.float16,
                                       softmax_compute_type=ms.float16,
                                       param_init_type=ms.float16,
                                       src_seq_length=src_seq_length,
                                       tgt_seq_length=src_seq_length,
                                       parallel_config=parallel_config)

        self.ln_post = nn.LayerNorm([self.width])
        self.proj = Parameter(scale * Tensor(np.random.normal(0, 1,
                                                              size=(self.width, self.output_dim)).astype(np.float32)))
        self.cat = ops.Concat(1)
        self.tile = ops.Tile()
        self.ones = ops.OnesLike()
        self.expand_dims = ops.ExpandDims()
        if self.token_learner_flag:
            self.token_learner = TokenLearner(in_channels=self.output_dim,
                                              num_tokens=config.vision_config.num_tokens,
                                              num_groups=config.vision_config.num_token_groups,
                                              dropout_rate=config.vision_config.token_learner_dropout)

    def construct(self, x):
        """construct of VisualTransformer"""
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.transpose(0, 2, 1)
        if self.single_bs:
            class_embedding = self.expand_dims(self.class_embedding, 0)
            class_embedding = self.expand_dims(class_embedding, 0)
        else:
            class_embedding = self.tile(self.class_embedding, (x.shape[0], 1, 1))
        x = self.cat((class_embedding, x))
        x = x + self.positional_embedding
        x = self.ln_pre(x)
        x_shape = x.shape
        mask = ops.fill(ops.DType()(x), (x_shape[0], x_shape[1], x_shape[1]), 1)
        x = self.transformer(x, mask)[0]
        x = self.ln_post(x)
        x = ops.matmul(x, self.proj)
        if self.token_learner_flag:
            cls_token, x = x[:, 0, :], x[:, 1:, :]
            x = x.reshape(-1, self.num_tokens_side, self.num_tokens_side, self.output_dim)
            x = self.token_learner(x)
            cls_token = self.expand_dims(cls_token, 1).astype(x.dtype)
            x = self.cat((cls_token, x))
        return x


@constexpr
def build_attntion_mask(batch_size, context_length):
    mask = np.tril(np.full((batch_size, context_length, context_length), 1).astype(np.float32), 0)
    mask = Tensor(mask)
    return mask


class TextTransformer(nn.Cell):
    """TextTransformer of Filip"""
    def __init__(self, config):
        super(TextTransformer, self).__init__()
        self.batch_size = config.batch_size
        self.context_length = config.text_config.max_position_embeddins
        self.vocab_size = config.text_config.vocab_size
        self.width = config.text_config.hidden_size
        self.heads = config.text_config.num_heads
        if not self.heads:
            self.heads = self.width // config.ratio
        else:
            self.heads = int(self.heads)

        self.layers = config.text_config.num_hidden_layers
        self.output_dim = config.projection_dim

        self.embedding_table = Parameter(initializer(TruncatedNormal(0.02), [self.vocab_size, self.width]))
        self.gather = ops.Gather()
        self.reshape = ops.Reshape()
        self.not_equal = ops.NotEqual()
        self.cast = ops.Cast()
        self.positional_embedding = Parameter(initializer(TruncatedNormal(0.01), [self.context_length, self.width]))
        self.ln_final = nn.LayerNorm([self.width])
        self.text_projection = Parameter(Tensor(np.random.normal(0, self.width ** -0.5, size=(
            self.width, self.output_dim)).astype(np.float32)))

        parallel_config = TransformerOpParallelConfig()
        if config.recompute:
            recompute_config = TransformerRecomputeConfig(recompute=False)
            parallel_config = TransformerOpParallelConfig(recompute=recompute_config)
        self.transformer_layer = Transformer(hidden_size=self.width,
                                             batch_size=self.batch_size,
                                             num_heads=self.heads,
                                             ffn_hidden_size=self.width * 4,
                                             encoder_layers=self.layers,
                                             attention_dropout_rate=0.0,
                                             hidden_dropout_rate=0.0,
                                             decoder_layers=0,
                                             # hidden_act=QuickGELU,
                                             hidden_act='fast_gelu',
                                             layernorm_compute_type=ms.float16,
                                             softmax_compute_type=ms.float16,
                                             param_init_type=ms.float16,
                                             src_seq_length=self.context_length,
                                             tgt_seq_length=self.context_length,
                                             parallel_config=parallel_config)

    def construct(self, text):
        """construct of TextTransformer"""
        bsz, ctx_len = text.shape
        flatten_id = text.flatten()
        gather_result = self.gather(self.embedding_table, flatten_id, 0)
        x = self.reshape(gather_result, (bsz, ctx_len, -1))
        x = x + self.positional_embedding
        mask_tensor = build_attntion_mask(bsz, self.context_length)
        x = self.transformer_layer(x, mask_tensor)[0]
        x = self.ln_final(x)
        x = ops.matmul(x, self.text_projection)
        return x


class TokenLearner(nn.Cell):
    """TokenLearner of Filip"""
    def __init__(self, in_channels, num_tokens, num_groups=8, dropout_rate=0.):
        super(TokenLearner, self).__init__()
        self.in_channels = in_channels
        self.num_tokens = num_tokens
        self.num_groups = num_groups

        self.norm = nn.LayerNorm([self.in_channels])
        self.attntion_maps = nn.SequentialCell([
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size=1, group=self.num_groups),
            nn.Conv2d(self.in_channels, self.num_tokens, kernel_size=1)
        ])
        self.feat_conv = nn.Conv2d(
            self.in_channels, self.in_channels, 1, group=self.num_groups
        )

        self.dropout = nn.Dropout(dropout_rate)
        self.softmax = nn.Softmax()
        self.matmul = ops.BatchMatMul()

    def construct(self, x):
        """construct of TokenLearner"""
        feature_shape = x.shape

        selected = x
        selected = self.norm(selected)
        selected = selected.transpose((0, 3, 1, 2))
        selected = self.attntion_maps(selected)
        selected = selected.transpose((0, 2, 3, 1))
        selected = selected.reshape((feature_shape[0], feature_shape[1] * feature_shape[2], -1))
        selected = selected.transpose((0, 2, 1))
        selected = self.softmax(selected)

        feat = x
        feat = feat.transpose((0, 3, 1, 2))
        feat = self.feat_conv(feat)
        feat = feat.transpose((0, 2, 3, 1))
        feat = feat.reshape((feature_shape[0], feature_shape[1] * feature_shape[2], -1))

        outputs = self.matmul(selected, feat)
        outputs = self.dropout(outputs)
        return outputs
