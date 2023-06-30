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
# ============================================================================
"""GLM config"""
from mindformers.modules.transformer.moe import MoEConfig
from mindformers.modules.transformer.transformer import default_transformer_config, default_moe_config, \
    TransformerOpParallelConfig, OpParallelConfig, EmbeddingOpParallelConfig, default_embedding_parallel_config
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from ..utils import convert_mstype
from ..base_config import BaseConfig
from ...mindformer_book import MindFormerBook

default_dpmp_config = OpParallelConfig()

__all__ = ['GLMConfig']


@MindFormerRegister.register(MindFormerModuleType.CONFIG)
class GLMConfig(BaseConfig):
    """
    GLM config class which defines the model size
    """
    _support_list = MindFormerBook.get_config_support_list()['glm']

    def __init__(self,
                 batch_size: int = 1,
                 vocab_size: int = 130528,
                 hidden_size: int = 4096,
                 num_layers: int = 28,
                 num_heads: int = 32,
                 inner_hidden_size: int = 16384,
                 seq_length: int = 512,
                 embedding_dropout_prob: float = 0.0,
                 attention_dropout_rate: float = 0.0,
                 hidden_dropout_rate: float = 0.0,
                 hidden_size_per_attention_head: bool = None,
                 layernorm_order: str = "post",
                 layernorm_epsilon: float = 1.0e-5,
                 use_final_layernorm: bool = True,
                 op_parallel_config: OpParallelConfig = default_dpmp_config,
                 embed_parallel_config: EmbeddingOpParallelConfig = default_embedding_parallel_config,
                 parallel_config: TransformerOpParallelConfig = default_transformer_config,
                 moe_config: MoEConfig = default_moe_config,
                 use_past: bool = False,
                 activation_func: str = 'GELU',
                 position_encoding_2d: bool = True,
                 param_init_type: str = "float16",
                 layernorm_compute_type: str = "float32",
                 softmax_compute_type: str = "float32",
                 compute_dtype: str = "float16",
                 bos_token_id: int = 130004,
                 eos_token_id: int = 130005,
                 mask_token_id: int = 130000,
                 gmask_token_id: int = 130001,
                 pad_token_id: int = 3,
                 is_enhanced_encoder: bool = True,
                 is_npu_acceleration: bool = False,
                 checkpoint_name_or_path: str = "",
                 max_decode_length: int = 2048,
                 top_k: int = 1,
                 top_p: float = 1,
                 repetition_penalty: float = 1.0,
                 do_sample: bool = True,
                 **kwargs):
        super().__init__(**kwargs)
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.embedding_dropout_prob = embedding_dropout_prob
        self.hidden_dropout_rate = hidden_dropout_rate
        self.attention_dropout_rate = attention_dropout_rate
        self.hidden_size_per_attention_head = hidden_size_per_attention_head
        self.layernorm_order = layernorm_order
        self.layernorm_epsilon = layernorm_epsilon
        self.use_final_layernorm = use_final_layernorm
        self.op_parallel_config = op_parallel_config
        self.embed_parallel_config = embed_parallel_config
        self.moe_config = moe_config
        self.use_past = use_past
        self.parallel_config = parallel_config
        self.activation_func = activation_func
        self.inner_hidden_size = inner_hidden_size
        self.position_encoding_2d = position_encoding_2d
        self.param_init_type = convert_mstype(param_init_type)
        self.layernorm_compute_type = convert_mstype(layernorm_compute_type)
        self.softmax_compute_type = convert_mstype(softmax_compute_type)
        self.compute_dtype = convert_mstype(compute_dtype)
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.mask_token_id = mask_token_id
        self.gmask_token_id = gmask_token_id
        self.pad_token_id = pad_token_id
        self.max_decode_length = max_decode_length
        self.seq_length = seq_length
        self.is_enhanced_encoder = is_enhanced_encoder
        self.is_npu_acceleration = is_npu_acceleration
        self.checkpoint_name_or_path = checkpoint_name_or_path
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.do_sample = do_sample
