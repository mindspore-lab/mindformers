# Copyright 2024 Huawei Technologies Co., Ltd
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
"""iFlytekSpark model config APIs. """
from mindformers.models.utils import convert_mstype
from mindformers.models.base_config import BaseConfig
from mindformers.modules.transformer.transformer import default_transformer_config, TransformerOpParallelConfig
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType

__all__ = ['IFlytekSparkConfig']

@MindFormerRegister.register(MindFormerModuleType.CONFIG)
class IFlytekSparkConfig(BaseConfig):
    """ IFlytekSpark model config class which defines the model size """
    def __init__(self,
                 batch_size=1,
                 seq_length=32768,
                 vocab_size=60000,
                 hidden_size=5120,
                 ffn_hidden_size=28672,
                 num_layers=40,
                 num_heads=40,
                 compute_type: str = "float16",
                 softmax_compute_type: str = "float16",
                 layernorm_compute_type: str = "float32",
                 embedding_init_type: str = "float32",
                 post_layernorm_residual=False,
                 apply_residual_connection_post_layernorm=False,
                 layernorm_epsilon: float = 1e-5,
                 dropout_rate=0.0,
                 bias_dropout_fusion=True,
                 bos_token_id: int = 1,
                 eos_token_id: int = 2,
                 pad_token_id: int = 0,
                 ignore_token_id: int = -100,
                 hidden_act='fast_gelu',
                 eod_reset=True,
                 enable_offload=False,
                 use_moe=False,
                 expert_num=1,
                 parallel_config: TransformerOpParallelConfig = default_transformer_config,
                 flash_attention_recompute=False,
                 sparse_local_size=8192,
                 seq_parallel=False,
                 use_past=False,
                 offset: int = 0,
                 repetition_penalty: float = 1.0,
                 repetition_penalty_increase: float = 0.1,
                 max_length: int = 1024,
                 top_k: int = 1,
                 top_p: float = 1.0,
                 temperature: float = 1.0,
                 do_sample: bool = True,
                 is_dynamic: bool = False,
                 is_reward_model: bool = False,
                 is_lite_infer: bool = False,
                 **kwargs):
        super(IFlytekSparkConfig, self).__init__(**kwargs)
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.layernorm_epsilon = layernorm_epsilon
        # Use post-layernorm or pre-layernrom, default:pre-layernorm
        self.post_layernorm_residual = post_layernorm_residual
        self.apply_residual_connection_post_layernorm = apply_residual_connection_post_layernorm
        self.ffn_hidden_size = ffn_hidden_size
        self.dropout_rate = dropout_rate
        self.bias_dropout_fusion = bias_dropout_fusion
        self.hidden_act = hidden_act
        # moe
        self.use_moe = bool(use_moe)
        self.expert_num = expert_num
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.ignore_token_id = ignore_token_id
        self.layernorm_compute_type = convert_mstype(layernorm_compute_type)
        self.embedding_init_type = convert_mstype(embedding_init_type)
        self.compute_type = convert_mstype(compute_type)
        self.softmax_compute_type = convert_mstype(softmax_compute_type)
        self.parallel_config = parallel_config
        self.flash_attention_recompute = flash_attention_recompute
        self.seq_parallel = seq_parallel
        self.use_past = use_past
        self.eod_reset = eod_reset
        self.enable_offload = enable_offload
        self.sparse_local_size = sparse_local_size
        self.offset = offset
        self.repetition_penalty = repetition_penalty
        self.repetition_penalty_increase = repetition_penalty_increase
        self.max_length = max_length
        self.top_k = top_k
        self.top_p = top_p
        self.temperature = temperature
        self.do_sample = do_sample
        self.is_dynamic = is_dynamic
        self.is_reward_model = is_reward_model
        self.is_lite_infer = is_lite_infer
