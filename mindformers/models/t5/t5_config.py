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
"""T5 Configuration"""
from mindformers.modules.transformer.moe import MoEConfig
from mindformers.modules.transformer.transformer import default_transformer_config, default_moe_config, \
    TransformerOpParallelConfig
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from ..utils import convert_mstype
from ..base_config import BaseConfig
from ...mindformer_book import MindFormerBook

__all__ = ['T5Config']


@MindFormerRegister.register(MindFormerModuleType.CONFIG)
class T5Config(BaseConfig):
    """
    T5 config class which defines the model size
    """
    _support_list = MindFormerBook.get_config_support_list()['t5']

    def __init__(self,
                 vocab_size: int = 32128,
                 hidden_size: int = 512,
                 d_kv: int = 64,
                 d_ff: int = 2048,
                 num_layers: int = 6,
                 num_decoder_layers: int = None,
                 num_heads: int = 8,
                 relative_attention_num_buckets: int = 32,
                 hidden_dropout_rate: float = 0.1,
                 attention_dropout_rate: float = 0.1,
                 embedding_dropout_prob: float = 0.1,
                 layer_norm_epsilon: float = 1e-6,
                 initializer_factor: float = 1.0,
                 is_encoder_decoder: bool = True,
                 use_cache: bool = True,
                 pad_token_id: int = 0,
                 start_token_id: int = 0,
                 eos_token_id: int = 1,
                 batch_size: int = 1,
                 seq_length: int = 1024,
                 max_position_embeddings: int = 1024,
                 initializer_range: float = 0.02,
                 max_decode_length: int = 128,
                 length_penalty_weight: float = 1.0,
                 compute_dtype: str = "float32",
                 has_relative_bias: bool = True,
                 scale_output: bool = True,
                 parallel_config: TransformerOpParallelConfig = default_transformer_config,
                 checkpoint_name_or_path: str = None,
                 top_p: float = 0.95,
                 top_k: int = 1,
                 repetition_penalty: float = 1.0,
                 max_length: int = 20,
                 do_sample: bool = False,
                 param_init_type: str = "float32",
                 layernorm_compute_type: str = "float32",
                 softmax_compute_type: str = "float32",
                 hidden_act: str = 'relu',
                 post_layernorm_residual: bool = False,
                 offset: int = 0,
                 use_past: bool = False,
                 moe_config: MoEConfig = default_moe_config,
                 **kwargs):
        super(T5Config, self).__init__(**kwargs)
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.hidden_act = hidden_act
        self.kv_size = d_kv
        self.hidden_dropout_rate = hidden_dropout_rate
        self.attention_dropout_rate = attention_dropout_rate
        self.embedding_dropout_prob = embedding_dropout_prob
        self.initializer_factor = initializer_factor
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.max_decode_length = max_decode_length
        self.length_penalty_weight = length_penalty_weight
        self.has_relative_bias = has_relative_bias
        self.scale_output = scale_output
        self.parallel_config = parallel_config
        self.num_decoder_layers = num_decoder_layers
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.layer_norm_epsilon = layer_norm_epsilon
        self.use_cache = use_cache
        self.checkpoint_name_or_path = checkpoint_name_or_path
        self.pad_token_id = pad_token_id
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        self.max_length = max_length
        self.start_token_id = start_token_id
        self.eos_token_id = eos_token_id
        self.is_encoder_decoder = is_encoder_decoder
        self.do_sample = do_sample
        self.layernorm_compute_type = convert_mstype(layernorm_compute_type)
        self.softmax_compute_type = convert_mstype(softmax_compute_type)
        self.compute_dtype = convert_mstype(compute_dtype)
        self.use_past = use_past
        self.post_layernorm_residual = post_layernorm_residual
        self.offset = offset
        self.moe_config = moe_config
        self.param_init_type = convert_mstype(param_init_type)
