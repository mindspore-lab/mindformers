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
# https://github.com/facebookresearch/mae
# ============================================================================
"""ViT Model."""
import math
import numpy as np
from mindspore import load_param_into_net, Parameter, nn
from mindspore import ops as P
from mindspore import dtype as mstype
import mindspore.common.initializer as weight_init
from mindformers.mindformer_book import MindFormerBook
from mindformers.common.loss import build_loss
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from mindformers.models.base_model import BaseModel
from mindformers.models.vit.vit_modules import Block, LayerNorm, Linear, Dropout
from mindformers.models.vit.vit_modules import PatchEmbed
from mindformers.models.vit.vit_config import VitConfig


@MindFormerRegister.register(MindFormerModuleType.MODELS)
class VitModel(BaseModel):
    """
    Vision Transformer with support for patch or hybrid CNN input stage.
    The supported model name could be selected from VitConfig.show_support_list().

    Args:
        config (VitConfig): the config of Vit model.

    Examples:
        >>> # input model name, load model and weights
        >>> model_a = VitModel.from_pretrained('vit_base_p16')
        >>> # input config, load model without weights
        >>> from mindformers import AutoConfig
        >>> config = AutoConfig.from_pretrained('vit_base_p16')
        >>> model_b = VitModel.from_config(config)
    """
    _support_list = MindFormerBook.get_model_support_list()['vit']

    def __init__(self, config=None):
        config = config if config else VitConfig()
        super(VitModel, self).__init__(config)
        self.use_moe = (config.moe_config.expert_num > 1)
        parallel_config = config.parallel_config
        dp = parallel_config.data_parallel
        self.global_pool = config.use_mean_pooling
        self.patch_embed = PatchEmbed(img_size=config.image_size, patch_size=config.patch_size,
                                      in_features=config.in_chans, out_features=config.embed_dim,
                                      parallel_config=parallel_config)
        self.cls_tokens = Parameter(
            weight_init.initializer(weight_init.Normal(sigma=.02), (1, 1, config.embed_dim)), requires_grad=True)
        num_patches = self.patch_embed.num_patches
        seq_length = num_patches + 1
        self.seq_length = seq_length
        self.num_patches = num_patches
        self.num_masked = num_patches - seq_length + 1
        self.pos_embed = Parameter(
            weight_init.initializer(weight_init.TruncatedNormal(sigma=.02), (1, seq_length, config.embed_dim)),
            requires_grad=True)
        # stochastic depth decay rule
        hdr = [x.item() for x in np.linspace(0, config.drop_path_rate, config.depth)]
        parallel_config_args = parallel_config.moe_parallel_config if self.use_moe else parallel_config.dp_mp_config
        self.blocks = nn.CellList([
            Block(hidden_size=config.embed_dim,
                  ffn_hidden_size=int(config.embed_dim * config.mlp_ratio),
                  seq_length=seq_length,
                  drop_rate=config.drop_rate,
                  attention_dropout_rate=config.attention_dropout_rate,
                  hidden_dropout_rate=hdr[i],
                  init_values=config.init_values,
                  weight_init='XavierUniform',
                  layernorm_compute_type=config.layernorm_compute_type,
                  softmax_compute_type=config.softmax_compute_type,
                  window_size=None,
                  num_heads=config.num_heads,
                  hidden_act=config.hidden_act,
                  post_layernorm_residual=config.post_layernorm_residual,
                  param_init_type=config.param_init_type,
                  parallel_config=parallel_config_args)
            for i in range(config.depth)])

        self.add = P.Add().shard(((dp, 1, 1), (1, 1, 1)))
        self.cast = P.Cast()
        self.tile = P.Tile().shard(((dp, 1, 1),))
        self.cat = P.Concat(axis=1)
        self.fc_norm = LayerNorm((config.embed_dim,), eps=1e-6).shard(((dp, 1, 1),))

        self.reduce_mean = P.ReduceMean().shard(((dp, 1, 1),))
        self.dropout = Dropout(keep_prob=(1. - config.drop_rate))
        self.dropout.shard(((dp, 1, 1),))

        self.stride_slice = P.StridedSlice().shard(((dp, 1, 1),))

        self.head = Linear(
            config.embed_dim, config.num_classes,
            weight_init=weight_init.TruncatedNormal(sigma=2e-5),
            compute_dtype=mstype.float32).to_float(mstype.float32)

        self.loss = build_loss(class_name=config.loss_type)

        self.init_weights_vit()
        self.fix_init_weight()

        self._load_checkpoint(config)

    def fix_init_weight(self):
        """fix init weight"""

        def rescale(param, layer_id):
            values = param.data / (math.sqrt(2.0 * layer_id))
            param.set_data(values)

        for layer_id, block in enumerate(self.blocks):
            if self.use_moe:
                rescale(block.attention.projection.weight, layer_id + 1)
                rescale(block.output.ffn.projection.weight, layer_id + 1)
            else:
                rescale(block.attention.projection.weight, layer_id + 1)
                rescale(block.output.projection.weight, layer_id + 1)

    def init_weights_vit(self):
        """init weights vit
         ViT weight initialization, original timm impl (for reproducibility) """
        for name, cell in self.cells_and_names():

            if isinstance(cell, Linear):
                cell.weight.set_data(weight_init.initializer(weight_init.XavierUniform(),
                                                             cell.weight.shape,
                                                             cell.weight.dtype))
                if isinstance(cell, Linear) and cell.bias is not None:
                    cell.bias.set_data(weight_init.initializer(weight_init.Zero(),
                                                               cell.bias.shape,
                                                               cell.bias.dtype))
            elif isinstance(cell, (LayerNorm, nn.LayerNorm)):
                cell.gamma.set_data(weight_init.initializer(weight_init.One(),
                                                            cell.gamma.shape,
                                                            cell.gamma.dtype))
                cell.beta.set_data(weight_init.initializer(weight_init.Zero(),
                                                           cell.beta.shape,
                                                           cell.beta.dtype))
            if name == "patch_embed.projection":
                cell.weight.set_data(weight_init.initializer(weight_init.XavierUniform(),
                                                             cell.weight.shape,
                                                             cell.weight.dtype))

    def no_weight_decay(self):
        return {'pos_embed', 'cls_tokens'}

    def load_pretrained(self, params_dict):
        return load_param_into_net(self, params_dict)

    def construct(self, image, target=None):
        """construct of vit"""
        tokens = self.patch_embed(image)
        batch_size = image.shape[0]
        cls_tokens = self.tile(self.cls_tokens, (batch_size, 1, 1))
        tokens = self.cat((cls_tokens, tokens))
        if self.pos_embed is not None:
            tokens = self.add(tokens, self.pos_embed)

        x = self.dropout(tokens)
        encoder_input_mask = P.Ones()((batch_size, self.seq_length, self.seq_length), mstype.float32)
        for block in self.blocks:
            x = block(x, encoder_input_mask)

        b, s, c = x.shape

        if self.global_pool:
            x = self.stride_slice(
                x, (0, 1, 0), (b, s, c), (1, 1, 1)
            )
            x = self.reduce_mean(x, 1)
            out = self.fc_norm(x)
        else:
            out = self.stride_slice(
                x, (0, 0, 0), (b, 1, c), (1, 1, 1)
            )
        out = self.head(out)
        if self.phase != "train":
            return out, target
        loss = self.loss(out, target)
        return loss
