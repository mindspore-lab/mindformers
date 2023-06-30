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


__all__ = ['ViTMAEConfig']


@MindFormerRegister.register(MindFormerModuleType.CONFIG)
class ViTMAEConfig(BaseConfig):
    """
    Config for Mae model

    Examples:
        >>> # init a config with a model name
        >>> config_a = ViTMAEConfig.from_pretrained('mae_vit_base_p16')
        >>> # init a config with a config path
        >>> import os
        >>> from mindformers.mindformer_book import MindFormerBook
        >>> config_path = os.path.join(MindFormerBook.get_project_path(),
        >>>                        'configs', 'mae', 'run_mae_vit_base_p16_224_800ep.yaml')
        >>> config_b = ViTMAEConfig.from_pretrained(config_path)
        >>> # init a config with args
        >>> config_c = ViTMAEConfig(
        >>>     patch_size=16,
        >>>     in_chans=3,
        >>>     ...
        >>>     )
    """
    _support_list = MindFormerBook.get_config_support_list()['mae']

    def __init__(self,
                 mask_ratio: float = 0.75,
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
                 layer_norm_eps: float = 1e-6,
                 attention_probs_dropout_prob: float = 0.0,
                 hidden_dropout_prob: float = 0.0,
                 drop_path_rate: float = 0.,
                 decoder_hidden_size: int = 512,
                 decoder_num_hidden_layers: int = 8,
                 decoder_num_attention_heads: int = 16,
                 decoder_intermediate_size: int = 2048,
                 norm_pix_loss: bool = True,
                 checkpoint_name_or_path: str = '',
                 layernorm_compute_type: mstype = mstype.float32,
                 softmax_compute_type: mstype = mstype.float32,
                 param_init_type: mstype = mstype.float32,
                 parallel_config: TransformerOpParallelConfig = default_parallel_config,
                 moe_config: MoEConfig = default_moe_config,
                 **kwargs):
        super().__init__(**kwargs)
        self.mask_ratio = mask_ratio
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
        self.decoder_embed_dim = decoder_hidden_size
        self.decoder_depth = decoder_num_hidden_layers
        self.decoder_num_heads = decoder_num_attention_heads
        self.decoder_intermediate_size = decoder_intermediate_size
        self.norm_pixel_loss = norm_pix_loss
        self.checkpoint_name_or_path = checkpoint_name_or_path
        self.layernorm_compute_type = layernorm_compute_type
        self.softmax_compute_type = softmax_compute_type
        self.param_init_type = param_init_type
        self.parallel_config = parallel_config
        self.moe_config = moe_config
