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
# ============================================================================
"""Vision Transformer implementation."""

from dataclasses import dataclass
import numpy as np

import mindspore
import mindspore.common.dtype as mstype
import mindspore.ops.functional as F
from mindspore import Tensor, nn
from mindspore.common.initializer import initializer
from mindspore.common.parameter import Parameter
from mindspore.nn import Cell, Dropout, SequentialCell
from mindspore.nn.transformer import TransformerOpParallelConfig
from mindspore.nn.transformer.layers import _Linear
from mindspore.nn.transformer.transformer import default_transformer_config
from mindspore.ops import operations as P
from mindspore.parallel.nn.transformer import Transformer

try:
    from mindspore.nn.loss.loss import Loss
except ImportError:
    try:
        from mindspore.nn.loss.loss import LossBase as Loss
    except ImportError:
        from mindspore.nn.loss.loss import _Loss as Loss

MIN_NUM_PATCHES = 4

@dataclass
class VitConfig:
    """Vit config class which defines the model size"""
    batch_size: int = 16
    d_model: int = 768
    depth: int = 12
    heads: int = 12
    mlp_dim: int = 3072
    dim_head: int = 64
    patch_size: int = 16
    normalized_shape: int = 768
    image_size: int = 224
    num_classes: int = 1000
    decoder_layers: int = 0
    network_pool: str = "cls"
    post_layernorm_residual: bool = True
    loss_name: str = "ce_smooth_mixup"
    label_smooth_factor: float = 0.1
    aux_factor: float = 0.4
    ignore_label: int = -2

    dtype: mstype = mstype.float32
    compute_dtype: mstype = mstype.float32
    layernorm_dtype: mstype = mstype.float32
    softmax_dtype: mstype = mstype.float32

    hidden_act: str = "gelu"
    network_dropout_prob: float = 0.1
    attention_dropout_prob: float = 0.1
    feedforward_dropout_prob: float = 0.1

    parallel_config: TransformerOpParallelConfig = default_transformer_config

def origin_head(vit_config):
    """Head for ViT pretraining."""
    d_model = vit_config.d_model
    num_classes = vit_config.num_classes
    compute_dtype = vit_config.compute_dtype
    initialization = mindspore.common.initializer.XavierUniform()
    dense = _Linear(d_model, num_classes).to_float(compute_dtype)
    dense.weight.set_data(initializer(initialization, [num_classes, d_model]))

    dp = vit_config.parallel_config.data_parallel
    mp = vit_config.parallel_config.model_parallel
    dense.shard(strategy_matmul=((dp, 1), (1, mp)), strategy_bias=((dp, 1), (1,)))

    return SequentialCell([dense])

class VitStem(Cell):
    """Stem layer for ViT."""

    def __init__(self, vit_config):
        super().__init__()
        d_model = vit_config.d_model
        patch_size = vit_config.patch_size
        image_size = vit_config.image_size
        compute_dtype = vit_config.compute_dtype
        channels = 3

        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        assert num_patches > MIN_NUM_PATCHES, f'your number of patches {num_patches} is too small'
        patch_dim = channels * patch_size ** 2

        self.patch_size = patch_size
        self.patch_to_embedding = mindspore.nn.Conv2d(channels, d_model, patch_size,
                                                      patch_size, has_bias=True).to_float(compute_dtype)
        self.num_patches = num_patches
        self.patch_dim = patch_dim
        self.d_model = d_model
        self.reshape = P.Reshape()
        self.transpose = P.Transpose()
        self.add = P.Add()
        self.add.add_prim_attr("keep_alive", True)

    def construct(self, img):
        bs, _, _, _ = img.shape
        x = self.patch_to_embedding(img)
        x = self.reshape(x, (bs, self.d_model, self.num_patches))
        x = self.transpose(x, (0, 2, 1))
        x = self.add(x, 0)
        return x

class TransformerWrapper(Cell):
    """Transformer implementation."""

    def __init__(self, vit_config):
        super().__init__()

        batch_size = vit_config.batch_size
        src_seq_length = (vit_config.image_size // vit_config.patch_size) ** 2 + 1

        self.transformer = Transformer(
            hidden_size=vit_config.heads * vit_config.dim_head,
            batch_size=batch_size * vit_config.parallel_config.data_parallel,
            ffn_hidden_size=vit_config.mlp_dim,
            src_seq_length=src_seq_length,
            tgt_seq_length=src_seq_length,
            encoder_layers=vit_config.depth,
            decoder_layers=vit_config.decoder_layers,
            num_heads=vit_config.heads,
            attention_dropout_rate=vit_config.attention_dropout_prob,
            hidden_dropout_rate=vit_config.feedforward_dropout_prob,
            hidden_act=vit_config.hidden_act,
            parallel_config=vit_config.parallel_config
        )

        self.attention_mask = mindspore.Tensor(np.ones((batch_size, src_seq_length, src_seq_length)),
                                               dtype=mstype.float32)

    def construct(self, x):
        out, _, _ = self.transformer(x, self.attention_mask)
        return out


class ViT(Cell):
    """Vision Transformer implementation."""

    def __init__(self, vit_config, is_training=True):
        super().__init__()

        if not is_training:
            vit_config.network_dropout_prob = 0.0
            vit_config.feedforward_dropout_prob = 0.0
            vit_config.attention_dropout_prob = 0.0

        d_model = vit_config.d_model
        patch_size = vit_config.patch_size
        image_size = vit_config.image_size

        initialization = mindspore.common.initializer.Normal(sigma=1.0)
        pool = vit_config.network_pool
        dropout_rate = vit_config.network_dropout_prob
        norm = mindspore.nn.LayerNorm((vit_config.normalized_shape,))

        stem = VitStem(vit_config)
        body = TransformerWrapper(vit_config)
        head = origin_head(vit_config)

        assert pool in {'cls', 'mean'}, 'pool type must be either cls or mean'
        num_patches = (image_size // patch_size) ** 2

        if pool == "cls":
            self.cls_token = Parameter(initializer(initialization, (1, 1, d_model)),
                                       name='cls', requires_grad=True)
            self.pos_embedding = Parameter(initializer(initialization, (1, num_patches + 1, d_model)),
                                           name='pos_embedding', requires_grad=True)
            self.tile = P.Tile()
            self.cat_1 = P.Concat(axis=1)
        else:
            self.pos_embedding = Parameter(initializer(initialization, (1, num_patches, d_model)),
                                           name='pos_embedding', requires_grad=True)
            self.mean = P.ReduceMean(keep_dims=False)
        self.pool = pool

        dp = vit_config.parallel_config.data_parallel

        self.cast = P.Cast()
        self.slice1 = P.Slice().shard(((1, 1, 1),))
        self.dropout = Dropout(keep_prob=(1. - dropout_rate))
        self.transpose = P.Transpose().shard(((dp, 1, 1),))
        self.slice2 = P.Slice().shard(((1, dp, 1),))
        self.squeeze = P.Squeeze(0).shard(((1, dp, 1),))
        self.add = P.Add().shard(((dp, 1, 1), (1, 1, 1)))

        self.stem = stem
        self.body = body
        self.head = head
        self.norm = norm

    def construct(self, img):
        """construct"""
        x = self.stem(img)
        x = F.cast(x, mstype.float32)

        bs, seq_len, d_model = x.shape  # 2048, 49, 768

        pos_embedding = F.cast(self.pos_embedding, mstype.float32)
        if self.pool == "cls":
            cls_tokens = self.tile(self.cls_token, (bs, 1, 1))
            cls_tokens = F.cast(cls_tokens, mstype.float32)
            x = self.cat_1((cls_tokens, x))  # now x has shape = (bs, seq_len+1, d_model)
            sliced_embedding = self.slice1(pos_embedding, (0, 0, 0), (1, seq_len + 1, d_model))
            x = self.add(x, sliced_embedding)
        else:
            sliced_embedding = self.slice1(pos_embedding, (0, 0, 0), (1, seq_len, d_model))
            x = self.add(x, sliced_embedding)


        y = self.cast(x, mstype.float32)
        y = self.dropout(y)
        x = self.cast(y, x.dtype)

        x = self.body(x)

        if self.norm is not None:
            x = F.cast(x, mstype.float32)
            x = self.norm(x)

        if self.pool == "cls":
            x = self.transpose(x, (1, 0, 2))
            x = self.slice2(x, (0, 0, 0), (1, bs, d_model))
            x = self.squeeze(x)
        else:
            x = self.mean(x, (-2,))

        out = self.head(x)

        return F.cast(out, mstype.float32)

# loss
class CrossEntropySmooth(Loss):
    """CrossEntropy"""

    def __init__(self, sparse=True, reduction='mean', smooth_factor=0., num_classes=1000, aux_factor=0.4):
        super().__init__()
        self.aux_factor = aux_factor
        self.onehot = P.OneHot()
        self.sparse = sparse
        self.on_value = Tensor(1.0 - smooth_factor, mstype.float32)
        self.off_value = Tensor(1.0 * smooth_factor / (num_classes - 1), mstype.float32)
        self.ce = nn.SoftmaxCrossEntropyWithLogits(reduction=reduction)

    def construct(self, logits, label):
        """construct"""
        if isinstance(logits, tuple):
            logit, aux_logit = logits
        else:
            logit, aux_logit = logits, None

        if self.sparse:
            label = self.onehot(label, F.shape(logit)[1], self.on_value, self.off_value)

        loss = self.ce(logit, label)
        if aux_logit is not None:
            loss = loss + self.aux_factor * self.ce(aux_logit, label)
        return loss


class CrossEntropySmoothMixup(Loss):
    """CrossEntropy"""

    def __init__(self, reduction='mean', smooth_factor=0., num_classes=1000):
        super().__init__()
        self.on_value = Tensor(1.0 - smooth_factor, mstype.float32)
        self.off_value = 1.0 * smooth_factor / (num_classes - 2)
        self.cross_entropy = nn.SoftmaxCrossEntropyWithLogits(reduction=reduction)

    def construct(self, logit, label):
        off_label = P.Select()(P.Equal()(label, 0.0), \
                               P.Fill()(mstype.float32, P.Shape()(label), self.off_value), \
                               P.Fill()(mstype.float32, P.Shape()(label), 0.0))

        label = self.on_value * label + off_label
        loss = self.cross_entropy(logit, label)
        return loss


class CrossEntropyIgnore(Loss):
    """CrossEntropyIgnore"""

    def __init__(self, num_classes=21, ignore_label=255):
        super().__init__()
        self.one_hot = P.OneHot(axis=-1)
        self.on_value = Tensor(1.0, mstype.float32)
        self.off_value = Tensor(0.0, mstype.float32)
        self.cast = P.Cast()
        self.ce = nn.SoftmaxCrossEntropyWithLogits()
        self.not_equal = P.NotEqual()
        self.num_cls = num_classes
        self.ignore_label = ignore_label
        self.mul = P.Mul()
        self.sum = P.ReduceSum(False)
        self.div = P.RealDiv()
        self.transpose = P.Transpose()
        self.reshape = P.Reshape()

    def construct(self, logits, labels):
        """construct"""
        labels_int = self.cast(labels, mstype.int32)
        labels_int = self.reshape(labels_int, (-1,))
        logits_ = self.transpose(logits, (0, 2, 3, 1))
        logits_ = self.reshape(logits_, (-1, self.num_cls))
        weights = self.not_equal(labels_int, self.ignore_label)
        weights = self.cast(weights, mstype.float32)
        one_hot_labels = self.one_hot(labels_int, self.num_cls, self.on_value, self.off_value)
        loss = self.ce(logits_, one_hot_labels)
        loss = self.mul(weights, loss)
        loss = self.div(self.sum(loss), self.sum(weights))
        return loss


def get_loss(vit_config):
    """get_loss"""
    loss = None
    if vit_config.loss_name == 'ce_smooth':
        loss = CrossEntropySmooth(smooth_factor=vit_config.label_smooth_factor,
                                  num_classes=vit_config.num_classes,
                                  aux_factor=vit_config.aux_factor)
    elif vit_config.loss_name == 'ce_smooth_mixup':
        loss = CrossEntropySmoothMixup(smooth_factor=vit_config.label_smooth_factor,
                                       num_classes=vit_config.num_classes)
    elif vit_config.loss_name == 'ce_ignore':
        loss = CrossEntropyIgnore(num_classes=vit_config.num_classes,
                                  ignore_label=vit_config.ignore_label)
    else:
        raise NotImplementedError

    return loss

class VitWithLoss(nn.Cell):
    """
    Provide Vit pre-training loss through network.

    Args:
        config (VitConfig): The config of VitModel.

    Returns:
        Tensor, the loss of the network.
    """
    def __init__(self, vit_config):
        super(VitWithLoss, self).__init__()
        self.vit = ViT(vit_config)
        self.loss = get_loss(vit_config)
        self.cast = P.Cast()

    def construct(self, img, labels):
        """Get pre-training loss"""
        logits = self.vit(img)
        total_loss = self.loss(logits, labels)
        return self.cast(total_loss, mstype.float32)

def get_vit_network(opt, model_config):
    if opt.eval:
        net = ViT(model_config, is_training=False)
        return net

    netwithloss = VitWithLoss(model_config)
    return netwithloss
