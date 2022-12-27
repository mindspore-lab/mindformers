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
"""Bert Config API."""

import mindspore.common.dtype as mstype
from mindspore.nn.transformer import TransformerOpParallelConfig, MoEConfig
from mindspore.nn.transformer.transformer import default_transformer_config, default_moe_config

from mindformers.tools.register import MindFormerRegister, MindFormerModuleType

from ..base_config import BaseConfig
from ...mindformer_book import MindFormerBook

__all__ = ['BertConfig']

@MindFormerRegister.register(MindFormerModuleType.CONFIG)
class BertConfig(BaseConfig):
    """
    BERT config class which defines the model size
    """
    _support_list = MindFormerBook.get_model_support_list()['bert']
    def __init__(self,
                 model_type: str = "bert",
                 use_one_hot_embeddings: bool = False,
                 num_labels: int = 1,
                 assessment_method: str = "",
                 dropout_prob: float = 0.1,
                 batch_size: int = 16,
                 seq_length: int = 128,
                 vocab_size: int = 30522,
                 embedding_size: int = 768,
                 num_layers: int = 12,
                 num_heads: int = 12,
                 expand_ratio: int = 4,
                 hidden_act: str = "gelu",
                 post_layernorm_residual: bool = True,
                 hidden_dropout_prob: float = 0.1,
                 attention_probs_dropout_prob: float = 0.1,
                 max_position_embeddings: int = 128,
                 type_vocab_size: int = 2,
                 initializer_range: float = 0.02,
                 use_relative_positions: bool = False,
                 dtype: mstype = mstype.float16,
                 layernorm_dtype: mstype = mstype.float32,
                 softmax_dtype: mstype = mstype.float32,
                 compute_dtype: mstype = mstype.float16,
                 use_past: bool = False,
                 use_moe: bool = False,
                 parallel_config: TransformerOpParallelConfig = default_transformer_config,
                 checkpoint_name_or_path: str = "",
                 moe_config: MoEConfig = default_moe_config,
                 **kwargs):
        super(BertConfig, self).__init__(**kwargs)
        self.model_type = model_type
        self.use_one_hot_embeddings = use_one_hot_embeddings
        self.num_labels = num_labels
        self.assessment_method = assessment_method
        self.dropout_prob = dropout_prob
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.expand_ratio = expand_ratio
        self.hidden_act = hidden_act
        self.post_layernorm_residual = post_layernorm_residual
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
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
