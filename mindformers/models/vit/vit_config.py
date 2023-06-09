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
"""Mae Config API."""
import mindspore.common.dtype as mstype

from mindformers.modules.transformer import TransformerOpParallelConfig, TransformerRecomputeConfig
from mindformers.modules.transformer.moe import MoEConfig, default_moe_config

from mindformers.mindformer_book import MindFormerBook
from mindformers.models.base_config import BaseConfig
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType

default_recompute_config = TransformerRecomputeConfig()
default_parallel_config = TransformerOpParallelConfig(recompute=default_recompute_config)


@MindFormerRegister.register(MindFormerModuleType.CONFIG)
class ViTConfig(BaseConfig):
    """
    Config for ViT model

    Examples:
        >>> # init a config with a model name
        >>> config_a = ViTConfig.from_pretrained('vit_base_p16')
        >>> # init a config with a config path
        >>> import os
        >>> from mindformers.mindformer_book import MindFormerBook
        >>> config_path = os.path.join(MindFormerBook.get_project_path(),
        >>>                        'configs', 'vit', 'run_vit_base_p16_224_100ep.yaml')
        >>> config_b = ViTConfig.from_pretrained(config_path)
        >>> # init a config with args
        >>> config_c = ViTConfig(
        >>>     patch_size=16,
        >>>     in_chans=3,
        >>>     ...
        >>>     )
    """
    _support_list = MindFormerBook.get_config_support_list()['vit']

    def __init__(self,
                 image_size: int = 224,
                 patch_size: int = 16,
                 num_channels: int = 3,
                 initializer_range: float = 0.02,
                 hidden_size: int = 768,
                 num_hidden_layers: int = 12,
                 num_attention_heads: int = 12,
                 intermediate_size: int = 3072,
                 qkv_bias: bool = True,
                 hidden_act: str = "gelu",
                 post_layernorm_residual: bool = False,
                 layer_norm_eps: float = 1e-12,
                 attention_probs_dropout_prob: float = 0.0,
                 hidden_dropout_prob: float = 0.0,
                 drop_path_rate: float = 0.1,
                 use_mean_pooling: bool = True,
                 num_labels: int = 1000,
                 loss_type: str = "SoftTargetCrossEntropy",
                 encoder_stride: int = 16,
                 checkpoint_name_or_path: str = '',
                 layernorm_compute_type: mstype = mstype.float32,
                 softmax_compute_type: mstype = mstype.float32,
                 param_init_type: mstype = mstype.float32,
                 parallel_config: TransformerOpParallelConfig = default_parallel_config,
                 moe_config: MoEConfig = default_moe_config,
                 **kwargs):
        super().__init__(**kwargs)
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_chans = num_channels
        self.initializer_range = initializer_range
        self.embed_dim = hidden_size
        self.depth = num_hidden_layers
        self.num_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.qkv_bias = qkv_bias
        self.hidden_act = hidden_act
        self.post_layernorm_residual = post_layernorm_residual
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout_rate = attention_probs_dropout_prob
        self.drop_rate = hidden_dropout_prob
        self.drop_path_rate = drop_path_rate
        self.num_classes = num_labels
        self.use_mean_pooling = use_mean_pooling
        self.loss_type = loss_type
        self.encoder_stride = encoder_stride
        self.checkpoint_name_or_path = checkpoint_name_or_path
        self.layernorm_compute_type = layernorm_compute_type
        self.softmax_compute_type = softmax_compute_type
        self.param_init_type = param_init_type
        self.parallel_config = parallel_config
        self.moe_config = moe_config
