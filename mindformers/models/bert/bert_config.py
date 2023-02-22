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
from mindformers.modules.transformer.transformer import default_transformer_config, default_moe_config
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from ..base_config import BaseConfig
from ...mindformer_book import MindFormerBook

__all__ = ['BertConfig']

@MindFormerRegister.register(MindFormerModuleType.CONFIG)
class BertConfig(BaseConfig):
    """
    BERT config class which defines the model size
    """
    _support_list = MindFormerBook.get_config_support_list()['bert']
    _support_list.extend(MindFormerBook.get_config_support_list()['ner']['bert'])
    _support_list.extend(MindFormerBook.get_config_support_list()['txtcls']['bert'])
    _support_list.extend(MindFormerBook.get_config_support_list()['qa']['bert'])

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
                 dtype: str = "float32",
                 layernorm_dtype: str = "float32",
                 softmax_dtype: str = "float32",
                 compute_dtype: str = "float16",
                 use_past: bool = False,
                 parallel_config: str = "default",
                 checkpoint_name_or_path: str = "",
                 moe_config: str = "default",
                 is_training: bool = True,
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
        self._dtype = dtype
        self._layernorm_dtype = layernorm_dtype
        self._softmax_dtype = softmax_dtype
        self._compute_dtype = compute_dtype
        self.use_past = use_past
        self.checkpoint_name_or_path = checkpoint_name_or_path
        self._parallel_config = parallel_config
        self._moe_config = moe_config
        self.is_training = is_training

    @property
    def dtype(self):
        return mstype.float32 if self._dtype == "float32" else mstype.float16

    @property
    def layernorm_dtype(self):
        return mstype.float32 if self._layernorm_dtype == "float32" else mstype.float16

    @property
    def softmax_dtype(self):
        return mstype.float32 if self._softmax_dtype == "float32" else mstype.float16

    @property
    def compute_dtype(self):
        return mstype.float32 if self._compute_dtype == "float32" else mstype.float16

    @property
    def parallel_config(self):
        return default_transformer_config if self._parallel_config == "default" else default_transformer_config

    @property
    def moe_config(self):
        return default_moe_config if self._moe_config == "default" else default_moe_config
