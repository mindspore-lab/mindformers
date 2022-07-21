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

from importlib import import_module
from easydict import EasyDict as edict
import numpy as np

import mindspore
from mindspore import nn
from mindspore import Tensor
from mindspore.common.initializer import initializer
from mindspore.common.parameter import Parameter
from mindspore.nn import Cell, Dropout, SequentialCell
from mindspore.nn.transformer.layers import _Linear
from mindspore.ops import operations as P
from mindspore.ops import functional as F
import mindspore.common.dtype as mstype
from mindspore.nn.transformer import Transformer
try:
    from mindspore.nn.loss.loss import Loss
except ImportError:
    try:
        from mindspore.nn.loss.loss import LossBase as Loss
    except ImportError:
        from mindspore.nn.loss.loss import _Loss as Loss

MIN_NUM_PATCHES = 4

class VitConfig:
    """
    VitConfig
    """
    def __init__(self, configs):
        self.configs = configs
        self.parallel_config = configs.parallel_config

        # network init
        self.network_norm = mindspore.nn.LayerNorm((configs.normalized_shape,))
        self.network_init = mindspore.common.initializer.Normal(sigma=1.0)
        self.network_dropout_rate = 0.1
        self.network_pool = 'cls'
        self.network = ViT

        # stem
        self.stem_init = mindspore.common.initializer.XavierUniform()
        self.stem = VitStem

        # body
        # self.body_norm = mindspore.nn.LayerNorm
        self.body = TransformerWrapper

        # body attention
        self.attention_dropout_rate = 0.1

        # body feedforward
        self.feedforward_init = mindspore.common.initializer.XavierUniform()
        self.feedforward_activation = 'gelu'
        self.feedforward_dropout_rate = 0.1

        # head
        self.head = origin_head
        self.head_init = mindspore.common.initializer.XavierUniform()
        self.head_dropout_rate = 0.1
        # self.head_norm = mindspore.nn.LayerNorm((configs.normalized_shape,))
        self.head_activation = mindspore.nn.GELU()


def origin_head(vit_config):
    """Head for ViT pretraining."""
    d_model = vit_config.configs.d_model
    num_classes = vit_config.configs.num_classes
    initialization = vit_config.head_init
    dense = _Linear(d_model, num_classes).to_float(mstype.float16)
    dense.weight.set_data(initializer(initialization, [num_classes, d_model]))

    dp = vit_config.parallel_config.data_parallel
    mp = vit_config.parallel_config.model_parallel
    dense.shard(strategy_matmul=((dp, 1), (1, mp)), strategy_bias=((dp, 1), (1,)))

    return SequentialCell([dense])


class BatchDense(Cell):
    """BatchDense module."""

    def __init__(self, in_features, out_features, initialization, has_bias=True, vit_config=None):
        super().__init__()
        self.out_features = out_features
        self.dense = _Linear(in_features, out_features, has_bias=has_bias).to_float(mstype.float16)
        self.dense.weight.set_data(initializer(initialization, [out_features, in_features]))
        self.reshape = P.Reshape()
        self.add = P.Add()
        self.add.add_prim_attr("keep_alive", True)

        dp = vit_config.parallel_config.data_parallel
        mp = vit_config.parallel_config.model_parallel
        self.dense.shard(strategy_matmul=((dp, 1), (mp, 1)), strategy_bias=((dp, mp), (mp,)))

    def construct(self, x):
        bs, seq_len, d_model = x.shape
        out = self.reshape(x, (bs * seq_len, d_model))
        out = self.dense(out)
        out = self.reshape(out, (bs, seq_len, self.out_features))
        out = self.add(out, 0)
        return out


class VitStem(Cell):
    """Stem layer for ViT."""

    def __init__(self, vit_config):
        super().__init__()
        d_model = vit_config.configs.d_model
        patch_size = vit_config.configs.patch_size
        image_size = vit_config.configs.image_size
        initialization = vit_config.stem_init
        channels = 3

        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        assert num_patches > MIN_NUM_PATCHES, f'your number of patches {num_patches} is too small'
        patch_dim = channels * patch_size ** 2

        self.patch_size = patch_size
        self.reshape = P.Reshape()
        self.transpose = P.Transpose().shard(((vit_config.parallel_config.data_parallel, 1, 1, 1, 1, 1),))
        self.patch_to_embedding = BatchDense(patch_dim, d_model, initialization, has_bias=True, vit_config=vit_config)

    def construct(self, img):
        p = self.patch_size
        bs, channels, h, w = img.shape
        x = self.reshape(img, (bs, channels, h // p, p, w // p, p))
        x = self.transpose(x, (0, 2, 4, 1, 3, 5))
        x = self.reshape(x, (bs, (h//p)*(w//p), channels*p*p))
        x = self.patch_to_embedding(x)
        return x


class TransformerWrapper(Cell):
    """Transformer implementation."""

    def __init__(self, vit_config):
        super().__init__()

        batch_size = vit_config.configs.batch_size
        src_seq_length = (vit_config.configs.image_size // vit_config.configs.patch_size) ** 2 + 1  # patch32: 50

        self.transformer = Transformer(
            hidden_size=vit_config.configs.heads * vit_config.configs.dim_head,
            batch_size=batch_size * vit_config.parallel_config.data_parallel,
            ffn_hidden_size=vit_config.configs.mlp_dim,
            src_seq_length=src_seq_length,
            tgt_seq_length=src_seq_length,
            encoder_layers=vit_config.configs.depth,
            decoder_layers=vit_config.configs.decoder_layers,
            num_heads=vit_config.configs.heads,
            attention_dropout_rate=vit_config.attention_dropout_rate,
            hidden_dropout_rate=vit_config.feedforward_dropout_rate,
            hidden_act=vit_config.feedforward_activation,
            parallel_config=vit_config.parallel_config
        )
        self.attention_mask = mindspore.Tensor(np.ones((batch_size * vit_config.parallel_config.data_parallel,
                                                        src_seq_length, src_seq_length)), dtype=mindspore.dtype.float32)

    def construct(self, x):
        out, _, _ = self.transformer(x, self.attention_mask)
        return out


class ViT(Cell):
    """Vision Transformer implementation."""

    def __init__(self, vit_config):
        super().__init__()

        d_model = vit_config.configs.d_model
        patch_size = vit_config.configs.patch_size
        image_size = vit_config.configs.image_size

        initialization = vit_config.network_init
        pool = vit_config.network_pool
        dropout_rate = vit_config.network_dropout_rate
        norm = vit_config.network_norm

        stem = vit_config.stem(vit_config)
        body = vit_config.body(vit_config)
        head = vit_config.head(vit_config)

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
        bs, seq_len, d_model = x.shape  # 2048, 49, 768

        pos_embedding = F.cast(self.pos_embedding, mstype.float16)
        if self.pool == "cls":
            cls_tokens = self.tile(self.cls_token, (bs, 1, 1))
            cls_tokens = F.cast(cls_tokens, mstype.float16)
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


def load_function(func_name):
    """Load function using its name."""
    modules = func_name.split(".")
    if len(modules) > 1:
        module_path = ".".join(modules[:-1])
        name = modules[-1]
        module = import_module(module_path)
        return getattr(module, name)
    return func_name


vit_cfg = edict({
    'd_model': 768,
    'depth': 12,
    'heads': 12,
    'mlp_dim': 3072,
    'dim_head': 64,
    'patch_size': 32,
    'normalized_shape': 768,
    'image_size': 224,
    'num_classes': 1001,
    'batch_size': 256,
    'decoder_layers': 0,
    'parallel_config': None
})


def vit_base(args):
    """vit_base"""
    vit_cfg.d_model = 768
    vit_cfg.depth = 12
    vit_cfg.heads = 12
    vit_cfg.mlp_dim = 3072
    vit_cfg.dim_head = vit_cfg.d_model // vit_cfg.heads
    vit_cfg.patch_size = 32
    vit_cfg.normalized_shape = vit_cfg.d_model
    vit_cfg.image_size = args.train_image_size
    vit_cfg.num_classes = args.class_num
    vit_cfg.batch_size = args.batch_size if hasattr(args, "batch_size") else args.eval_batch_size
    vit_cfg.decoder_layers = 0
    vit_cfg.parallel_config = args.parallel_config

    if args.vit_config_path != '':
        print("get vit_config_path")
        vit_config = load_function(args.vit_config_path)(vit_cfg)
    else:
        print("get default_vit_cfg")
        vit_config = VitConfig(vit_cfg)

    model = vit_config.network(vit_config)

    return model


def vit_large(args):
    """vit_base"""
    vit_cfg.d_model = 1024
    vit_cfg.depth = 24
    vit_cfg.heads = 16
    vit_cfg.mlp_dim = 4096
    vit_cfg.dim_head = vit_cfg.d_model // vit_cfg.heads
    vit_cfg.patch_size = 32
    vit_cfg.normalized_shape = vit_cfg.d_model
    vit_cfg.image_size = args.train_image_size
    vit_cfg.num_classes = args.class_num
    vit_cfg.batch_size = args.batch_size if hasattr(args, "batch_size") else args.eval_batch_size
    vit_cfg.decoder_layers = 0
    vit_cfg.parallel_config = args.parallel_config

    if args.vit_config_path != '':
        print("get vit_config_path")
        vit_config = load_function(args.vit_config_path)(vit_cfg)
    else:
        print("get default_vit_cfg")
        vit_config = VitConfig(vit_cfg)

    model = vit_config.network(vit_config)

    return model


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


def get_vit_network(backbone_name, args):
    """get_network"""
    if backbone_name == 'vit_base':
        backbone = vit_base(args=args)
    elif backbone_name == 'vit_large':
        backbone = vit_large(args=args)
    else:
        raise NotImplementedError
    return backbone


def get_loss(loss_name, args):
    """get_loss"""
    loss = None
    if loss_name == 'ce_smooth':
        loss = CrossEntropySmooth(smooth_factor=args.label_smooth_factor,
                                  num_classes=args.class_num,
                                  aux_factor=args.aux_factor)
    elif loss_name == 'ce_smooth_mixup':
        loss = CrossEntropySmoothMixup(smooth_factor=args.label_smooth_factor,
                                       num_classes=args.class_num)
    elif loss_name == 'ce_ignore':
        loss = CrossEntropyIgnore(num_classes=args.class_num,
                                  ignore_label=args.ignore_label)
    else:
        raise NotImplementedError

    return loss
