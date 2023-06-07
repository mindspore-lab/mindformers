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
import mindspore as ms

from mindformers.tools.logger import logger
from mindformers.modules.transformer.transformer import TransformerOpParallelConfig, OpParallelConfig
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from mindformers.models.base_config import BaseConfig
from mindformers.modules.transformer import EmbeddingOpParallelConfig
from mindformers.mindformer_book import MindFormerBook

default_dpmp_config = OpParallelConfig()
default_embedding_parallel_config = EmbeddingOpParallelConfig()
default_transformer_config = TransformerOpParallelConfig()

mstype_dict = {"float32": ms.float32,
               "float16": ms.float16}

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
                 num_attention_heads: int = 32,
                 inner_hidden_size: int = 16384,
                 seq_length: int = 512,
                 embedding_dropout_prob: float = 0.0,
                 attention_dropout_prob: float = 0.0,
                 output_dropout_prob: float = 0.0,
                 hidden_size_per_attention_head=None,
                 layernorm_order: str = "post",
                 layernorm_epsilon: float = 1.0e-5,
                 use_final_layernorm: bool = True,
                 op_parallel_config=default_dpmp_config,
                 embed_parallel_config=default_embedding_parallel_config,
                 parallel_config: TransformerOpParallelConfig = default_transformer_config,
                 use_past: bool = False,
                 phase='test',
                 activation_func='GELU',
                 position_encoding_2d: bool = False,
                 params_dtype="float16",
                 layernorm_dtype="float32",
                 softmax_dtype="float32",
                 compute_dtype="float16",
                 bos_token_id=130004,
                 eos_token_id=130005,
                 mask_token_id=130000,
                 gmask_token_id=130001,
                 pad_token_id=3,
                 max_decode_length: int = 2048,
                 repetition_penalty: float = 1,
                 is_enhanced_encoder: bool = True,
                 is_npu_acceleration: bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.embedding_dropout_prob = embedding_dropout_prob
        self.output_dropout_prob = output_dropout_prob
        self.attention_dropout_prob = attention_dropout_prob
        self.hidden_size_per_attention_head = hidden_size_per_attention_head
        self.layernorm_order = layernorm_order
        self.layernorm_epsilon = layernorm_epsilon
        self.use_final_layernorm = use_final_layernorm
        self.op_parallel_config = op_parallel_config
        self.embed_parallel_config = embed_parallel_config
        self.use_past = use_past
        if phase == 'train' and use_past:
            self.use_past = False
            logger.warning(f"use_past can't be True when phase='train', it has been set to False")
        self.parallel_config = parallel_config
        self.activation_func = activation_func
        self.phase = phase
        self.inner_hidden_size = inner_hidden_size
        self.position_encoding_2d = position_encoding_2d
        self.params_dtype = mstype_dict[params_dtype]
        self.layernorm_dtype = mstype_dict[layernorm_dtype]
        self.softmax_dtype = mstype_dict[softmax_dtype]
        self.compute_dtype = mstype_dict[compute_dtype]
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.mask_token_id = mask_token_id
        self.gmask_token_id = gmask_token_id
        self.pad_token_id = pad_token_id
        self.max_decode_length = max_decode_length
        self.seq_length = seq_length
        self.repetition_penalty = repetition_penalty
        self.is_enhanced_encoder = is_enhanced_encoder
        self.is_npu_acceleration = is_npu_acceleration
