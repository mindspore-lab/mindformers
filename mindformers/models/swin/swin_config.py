# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writingsoftware
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Swin Config API."""
from dataclasses import dataclass

import mindspore.common.dtype as mstype
from mindspore.nn.transformer.moe import MoEConfig, default_moe_config
from mindspore.nn.transformer.transformer import TransformerOpParallelConfig, TransformerRecomputeConfig

from mindformers.mindformer_book import MindFormerBook
from mindformers.models.base_config import BaseConfig
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType

default_recompute_config = TransformerRecomputeConfig()
default_parallel_config = TransformerOpParallelConfig(recompute=default_recompute_config)


@MindFormerRegister.register(MindFormerModuleType.CONFIG)
@dataclass
class SwinConfig(BaseConfig):
    """
    Swin config class which defines the model size
    """
    _support_list = MindFormerBook.get_model_support_list()['swin']

    def __init__(self,
                 batch_size: int = 128,
                 image_size: int = 224,
                 patch_size: int = 4,
                 num_classes: int = 1000,
                 in_channels: int = 3,
                 embed_dim: int = 128,
                 depths: list = (2, 2, 18, 2),
                 num_heads: list = (4, 8, 16, 32),
                 window_size: int = 7,
                 shift_size: int = 0,
                 mlp_ratio: float = 4.,
                 qkv_bias: bool = True,
                 qk_scale: float = None,
                 drop_out_rate: float = 0.,
                 attn_drop_rate: float = 0.,
                 drop_path_rate: float = 0.1,
                 use_abs_pos_emb: bool = False,
                 patch_norm: bool = True,
                 patch_type: str = "conv",
                 hidden_act: str = 'gelu',
                 weight_init: str = 'normal',
                 loss_type: str = "SoftTargetCrossEntropy",
                 param_init_type: mstype = mstype.float32,
                 moe_config: MoEConfig = default_moe_config,
                 parallel_config: TransformerOpParallelConfig = default_parallel_config,
                 checkpoint_name_or_path: str = 'swin_base_p4w7',
                 **kwargs):
        """Swin Base Config"""
        self.batch_size = batch_size
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.depths = depths
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.drop_out_rate = drop_out_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate
        self.use_abs_pos_emb = use_abs_pos_emb
        self.patch_norm = patch_norm
        self.patch_type = patch_type
        self.hidden_act = hidden_act
        self.weight_init = weight_init
        self.loss_type = loss_type
        self.param_init_type = param_init_type
        self.moe_config = moe_config
        self.parallel_config = parallel_config
        self.checkpoint_name_or_path = checkpoint_name_or_path
        super(SwinConfig, self).__init__(**kwargs)
