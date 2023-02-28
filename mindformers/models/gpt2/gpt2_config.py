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
"""Gpt Config API."""

import mindspore.common.dtype as mstype
from mindspore.nn.transformer import TransformerOpParallelConfig, MoEConfig
from mindspore.nn.transformer.transformer import default_transformer_config, default_moe_config
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from ..base_config import BaseConfig
from ...mindformer_book import MindFormerBook

__all__ = ['Gpt2Config']


@MindFormerRegister.register(MindFormerModuleType.CONFIG)
class Gpt2Config(BaseConfig):
    """
    Gpt config class which defines the model size
    """

    _support_list = MindFormerBook.get_config_support_list()['gpt2']

    def __init__(self,
                 model_type: str = "gpt2",
                 dropout_prob: float = 0.1,
                 batch_size: int = 8,
                 seq_length: int = 1024,
                 vocab_size: int = 50257,
                 embedding_size: int = 768,
                 num_layers: int = 12,
                 num_heads: int = 12,
                 expand_ratio: int = 4,
                 post_layernorm_residual: bool = False,
                 hidden_dropout_prob: float = 0.1,
                 attention_probs_dropout_prob: float = 0.1,
                 initializer_range: float = 0.02,
                 use_relative_positions: bool = False,
                 dtype: mstype = mstype.float32,
                 layernorm_dtype: mstype = mstype.float32,
                 softmax_dtype: mstype = mstype.float16,
                 compute_dtype: mstype = mstype.float16,
                 use_past: bool = False,
                 use_moe: bool = False,
                 parallel_config: TransformerOpParallelConfig = default_transformer_config,
                 checkpoint_name_or_path: str = "",
                 moe_config: MoEConfig = default_moe_config,
                 eos_token: int = 50256,
                 **kwargs):
        super(Gpt2Config, self).__init__(**kwargs)
        self.model_type = model_type
        self.dropout_prob = dropout_prob
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.expand_ratio = expand_ratio
        self.post_layernorm_residual = post_layernorm_residual
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.use_relative_positions = use_relative_positions
        self.dtype = dtype
        self.layernorm_dtype = layernorm_dtype
        self.softmax_dtype = softmax_dtype
        self.compute_dtype = compute_dtype
        self.use_past = use_past
        self.use_moe = use_moe
        self.parallel_config = parallel_config
        self.checkpoint_name_or_path = checkpoint_name_or_path
        self.moe_config = moe_config
        self.eos_token = eos_token
