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
from dataclasses import dataclass
import mindspore.common.dtype as mstype
from mindspore.nn.transformer.transformer import TransformerOpParallelConfig, TransformerRecomputeConfig
from mindspore.nn.transformer.moe import MoEConfig, default_moe_config
from mindformers.mindformer_book import MindFormerBook
from mindformers.models.base_config import BaseConfig
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
default_recompute_config = TransformerRecomputeConfig()
default_parallel_config = TransformerOpParallelConfig(recompute=default_recompute_config)


@MindFormerRegister.register(MindFormerModuleType.CONFIG)
@dataclass
class MaeConfig(BaseConfig):
    """Mae Config."""
    _support_list = MindFormerBook.get_model_support_list()['mae']

    def __init__(self,
                 patch_size: int = 16,
                 in_chans: int = 3,
                 embed_dim: int = 768,
                 depth: int = 12,
                 num_heads: int = 12,
                 mlp_ratio: int = 4,
                 drop_rate: float = 0.,
                 drop_path_rate: float = 0.,
                 use_abs_pos_emb: bool = True,
                 attention_dropout_rate: float = 0.,
                 use_mean_pooling: bool = True,
                 init_values: int = None,
                 hidden_act: str = 'gelu',
                 post_layernorm_residual: bool = False,
                 layernorm_compute_type: mstype = mstype.float32,
                 softmax_compute_type: mstype = mstype.float32,
                 param_init_type: mstype = mstype.float32,
                 loss_type: str = "SoftTargetCrossEntropy",
                 checkpoint_name_or_path: str = 'vit_base_p16',
                 parallel_config: TransformerOpParallelConfig = default_parallel_config,
                 moe_config: MoEConfig = default_moe_config,
                 batch_size: int = 64,
                 image_size: int = 224,
                 num_classes: int = 0,
                 mask_ratio: float = 0.75,
                 decoder_depth: int = 8,
                 decoder_num_heads: int = 16,
                 decoder_embed_dim: int = 512,
                 norm_pixel_loss: bool = True,
                 window_size: int = None, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.drop_rate = drop_rate
        self.drop_path_rate = drop_path_rate
        self.use_abs_pos_emb = use_abs_pos_emb
        self.attention_dropout_rate = attention_dropout_rate
        self.use_mean_pooling = use_mean_pooling
        self.init_values = init_values
        self.hidden_act = hidden_act
        self.post_layernorm_residual = post_layernorm_residual
        self.layernorm_compute_type = layernorm_compute_type
        self.softmax_compute_type = softmax_compute_type
        self.param_init_type = param_init_type
        self.loss_type = loss_type
        self.checkpoint_name_or_path = checkpoint_name_or_path
        self.parallel_config = parallel_config
        self.moe_config = moe_config
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_classes = num_classes
        self.mask_ratio = mask_ratio
        self.decoder_depth = decoder_depth
        self.decoder_num_heads = decoder_num_heads
        self.decoder_embed_dim = decoder_embed_dim
        self.norm_pixel_loss = norm_pixel_loss
        self.window_size = window_size
