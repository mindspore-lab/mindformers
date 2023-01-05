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
# https://github.com/microsoft/Swin-Transformer
# ============================================================================
"""Swin Model."""
import numpy as np

from mindspore import nn
from mindspore import Tensor
from mindspore import Parameter
from mindspore import dtype as mstype
import mindspore.ops.operations as P
import mindspore.common.initializer as weight_init_

from mindformers.common.loss import build_loss
from mindformers.mindformer_book import MindFormerBook
from mindformers.models.base_model import BaseModel
from mindformers.models.base_config import BaseConfig
from mindformers.models.swin.swin_config import SwinConfig
from mindformers.models.swin.swin_modules import Linear
from mindformers.models.swin.swin_modules import LayerNorm
from mindformers.models.swin.swin_modules import Dropout
from mindformers.models.swin.swin_modules import PatchEmbed
from mindformers.models.swin.swin_modules import PatchMerging
from mindformers.models.swin.swin_modules import SwinBasicLayer
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType


__all__ = ['SwinModel', 'SwinTransformer']


@MindFormerRegister.register(MindFormerModuleType.MODELS)
class SwinModel(BaseModel):
    """
    Swin Transformer Model.
    The supported model name could be selected from SwinModel.show_support_list().

    Args:
        config (SwinConfig): the config of Swin model.
    """
    _support_list = MindFormerBook.get_model_support_list()['swin']

    def __init__(self, config: BaseConfig = None):
        if config is None:
            config = SwinConfig()
        super(SwinModel, self).__init__(config)
        self.encoder = SwinTransformer(config)
        parallel_config = config.parallel_config
        dp = parallel_config.data_parallel
        mp = parallel_config.model_parallel
        self.head = Linear(
            self.encoder.num_features, self.encoder.num_classes,
            weight_init=weight_init_.TruncatedNormal(sigma=2e-5),
            compute_dtype=mstype.float32).to_float(mstype.float32)
        self.head.shard(strategy_bias=((dp, mp), (mp,)), strategy_matmul=((dp, 1), (mp, 1)))

        self.loss = build_loss(class_name=config.loss_type)

        self._load_checkpoint(config)

    def construct(self, image, target=None):
        x = self.encoder(image)
        out = self.head(x)
        if self.phase != "train":
            return out, target
        loss = self.loss(out, target)
        return loss


@MindFormerRegister.register(MindFormerModuleType.ENCODER)
class SwinTransformer(BaseModel):
    """
    Swin Transformer.
    The supported model name could be selected from SwinModel.show_support_list().

    Args:
        config (SwinConfig): the config of Swin model.
    """

    def __init__(self, config: BaseConfig = None):
        if config is None:
            config = SwinConfig()
        super(SwinTransformer, self).__init__(config)
        dp = config.parallel_config.data_parallel
        self.parallel_config = config.parallel_config
        self.use_moe = config.moe_config.expert_num > 1
        self.num_classes = config.num_classes
        self.num_layers = len(config.depths)
        self.embed_dim = config.embed_dim
        self.ape = config.ape
        self.in_channels = config.in_channels
        self.patch_size = config.patch_size
        self.patch_norm = config.patch_norm
        self.num_features = int(config.embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = config.mlp_ratio
        self.cast = P.Cast()

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(config)
        num_patches = self.patch_embed.num_patches
        self.num_patches = num_patches
        patches_resolution = self.patch_embed.grid_size
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = Parameter(
                Tensor(np.zeros((1, num_patches, config.embed_dim)), dtype=mstype.float32), name="ape")

        self.pos_drop = Dropout(keep_prob=1.0 - config.drop_out_rate)
        self.pos_drop.shard(((dp, 1, 1),))

        # stochastic depth
        dpr = list(np.linspace(0, config.drop_path_rate, sum(config.depths)))  # stochastic depth decay rule
        if self.use_moe:
            parallel_config_args = config.parallel_config.moe_parallel_config
        else:
            parallel_config_args = config.parallel_config.dp_mp_config

        # build layers
        self.layers = nn.CellList()
        self.final_seq = num_patches  # downsample seq_length
        for i_layer in range(self.num_layers):
            layer = SwinBasicLayer(
                config=config,
                dim=int(config.embed_dim * 2 ** i_layer),
                input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                  patches_resolution[1] // (2 ** i_layer)),
                depth=config.depths[i_layer],
                num_heads=config.num_heads[i_layer],
                drop_path=dpr[sum(config.depths[:i_layer]):sum(config.depths[:i_layer + 1])],
                norm_layer=LayerNorm,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                parallel_config=parallel_config_args)
            # downsample seq_length
            if i_layer < self.num_layers - 1:
                self.final_seq = self.final_seq // 4
            self.layers.append(layer)
        self.norm = LayerNorm([self.num_features,], eps=1e-6).shard(((dp, 1, 1),))
        self.transpose = P.Transpose().shard(((dp, 1, 1),))
        self.avgpool = P.ReduceMean(keep_dims=False).shard(((dp, 1, 1),))
        self.init_weights()

    def init_weights(self):
        """ Swin weight initialization, original timm impl (for reproducibility) """
        for _, cell in self.cells_and_names():
            if isinstance(cell, Linear):
                cell.weight.set_data(weight_init_.initializer(
                    weight_init_.TruncatedNormal(sigma=0.02),
                    cell.weight.shape,
                    cell.weight.dtype))
                if isinstance(cell, Linear) and cell.bias is not None:
                    cell.bias.set_data(weight_init_.initializer(weight_init_.Zero(),
                                                                cell.bias.shape,
                                                                cell.bias.dtype))
            elif isinstance(cell, (LayerNorm, nn.LayerNorm)):
                cell.gamma.set_data(weight_init_.initializer(weight_init_.One(),
                                                             cell.gamma.shape,
                                                             cell.gamma.dtype))
                cell.beta.set_data(weight_init_.initializer(weight_init_.Zero(),
                                                            cell.beta.shape,
                                                            cell.beta.dtype))

    @staticmethod
    def no_weight_decay():
        return {'absolute_pos_embed'}

    @staticmethod
    def no_weight_decay_keywords():
        return {'relative_position_bias_table'}

    def construct(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)  # B L C
        x = self.avgpool(self.transpose(x, (0, 2, 1)), 2)  # B C 1
        return x
