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
import mindspore.common.dtype as mstype
from mindformers.modules.transformer.moe import MoEConfig, default_moe_config
from mindformers.modules.transformer import TransformerOpParallelConfig, TransformerRecomputeConfig

from mindformers.mindformer_book import MindFormerBook
from mindformers.models.base_config import BaseConfig
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType


__all__ = ['SwinConfig']


default_recompute_config = TransformerRecomputeConfig()
default_parallel_config = TransformerOpParallelConfig(recompute=default_recompute_config)


@MindFormerRegister.register(MindFormerModuleType.CONFIG)
class SwinConfig(BaseConfig):
    """
    Swin config class which defines the model size

    Args:
         image_size: The input image size, Default 224.
         patch_size: patch size, Default 4.
         num_channels: channels of input images, Default 3.
         embed_dim: embedding dimension, Default 128.
         depths: number of transformer blocks for each swin layer, Default (2, 2, 18, 2).
         num_heads: number of attention heads for each swin layer, Default (4, 8, 16, 32).
         window_size: window size for swin, Default 7.
         shift_size: window shift size, Default 0.
         mlp_ratio: ffn_hidden_size = mlp_ratio * embed_dim, Default 4.
         qkv_bias: has transformer qkv bias or not, Default True.
         hidden_dropout_prob: drop rate of MLP, Default 0.
         attention_probs_dropout_prob: drop rate of Attention, Default 0.
         drop_path_rate: drop path rate of transformer blocks, Default 0.1.
         use_absolute_embeddings: if using absolute position embedding, Default False.
         patch_norm: use norm in SwinPatchEmbeddings, Default True.
         hidden_act: activation of MLP, Default "gelu".
         weight_init: weight initialize type, Default "normal".
         num_labels: number of labels in downstream tasks, Default 1000.
         loss_type: loss type, Default "SoftTargetCrossEntropy".
         param_init_type:, Default mstype.float32.
         moe_config:, Default default_moe_config.
         parallel_config:, Default default_parallel_config.
         checkpoint_name_or_path:, Default "swin_base_p4w7".
         **kwargs

    Examples:
        >>> # init a config with a model name
        >>> config_a = SwinConfig.from_pretrained('swin_base_p4w7')
        >>> # init a config with a config path
        >>> import os
        >>> from mindformers.mindformer_book import MindFormerBook
        >>> config_path = os.path.join(MindFormerBook.get_project_path(),
        >>>                        'configs', 'swin', 'run_swin_base_p4w7_224_100ep.yaml')
        >>> config_b = SwinConfig.from_pretrained(config_path)
        >>> # init a config with args
        >>> config_c = SwinConfig(
        >>>     patch_size=4,
        >>>     in_chans=3,
        >>>     ...
        >>>     )
    """
    _support_list = MindFormerBook.get_config_support_list()['swin']

    def __init__(self,
                 image_size: int = 224,
                 patch_size: int = 4,
                 num_channels: int = 3,
                 embed_dim: int = 128,
                 depths: list = (2, 2, 18, 2),
                 num_heads: list = (4, 8, 16, 32),
                 window_size: int = 7,
                 shift_size: int = 0,
                 mlp_ratio: float = 4.,
                 qkv_bias: bool = True,
                 layer_norm_eps: float = 1e-5,
                 hidden_dropout_prob: float = 0.,
                 attention_probs_dropout_prob: float = 0.,
                 drop_path_rate: float = 0.1,
                 use_absolute_embeddings: bool = False,
                 patch_norm: bool = True,
                 hidden_act: str = 'gelu',
                 weight_init: str = 'normal',
                 num_labels: int = 1000,
                 loss_type: str = "SoftTargetCrossEntropy",
                 param_init_type: mstype = mstype.float32,
                 moe_config: MoEConfig = default_moe_config,
                 parallel_config: TransformerOpParallelConfig = default_parallel_config,
                 checkpoint_name_or_path: str = '',
                 **kwargs):
        """Swin Base Config"""
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.embed_dim = embed_dim
        self.depths = depths
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.layer_norm_eps = layer_norm_eps
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.drop_path_rate = drop_path_rate
        self.use_absolute_embeddings = use_absolute_embeddings
        self.patch_norm = patch_norm
        self.hidden_act = hidden_act
        self.weight_init = weight_init
        self.num_labels = num_labels
        self.loss_type = loss_type
        self.param_init_type = param_init_type
        self.moe_config = moe_config
        self.parallel_config = parallel_config
        self.checkpoint_name_or_path = checkpoint_name_or_path
        super(SwinConfig, self).__init__(**kwargs)
