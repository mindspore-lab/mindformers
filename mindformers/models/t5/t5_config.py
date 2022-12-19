# Copyright 2020-2022 Huawei Technologies Co., Ltd
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
import mindspore
import mindspore.common.dtype as mstype
from mindspore.nn.transformer.transformer import default_transformer_config, TransformerOpParallelConfig
from ..base_config import BaseConfig
from ...tools.register import MindFormerRegister, MindFormerModuleType

__all__ = ['T5Config']


@MindFormerRegister.register(MindFormerModuleType.CONFIG)
class T5Config(BaseConfig):
    """T5 Config"""
    def __init__(self,
                 batch_size: int = 1,
                 seq_length: int = 1024,
                 vocab_size: int = 32128,
                 hidden_size: int = 512,
                 num_hidden_layers: int = 6,
                 num_heads: int = 8,
                 intermediate_size: int = 2048,
                 hidden_act: str = "relu",
                 hidden_dropout_prob: float = 0.1,
                 attention_probs_dropout_prob: float = 0.1,
                 max_position_embeddings: int = 1024,
                 initializer_range: float = 0.02,
                 label_smoothing: float = 0.1,
                 beam_width: int = 4,
                 max_decode_length: int = 128,
                 length_penalty_weight: float = 1.0,
                 dtype: mindspore.common.dtype = mstype.float32,
                 compute_dtype: mindspore.common.dtype = mstype.float32,
                 has_relative_bias: bool = True,
                 scale_output: bool = True,
                 parallel_config: TransformerOpParallelConfig = default_transformer_config,
                 **kwargs):
        """Transformer Config"""
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.label_smoothing = label_smoothing
        self.beam_width = beam_width
        self.max_decode_length = max_decode_length
        self.length_penalty_weight = length_penalty_weight
        self.dtype = dtype
        self.compute_dtype = compute_dtype
        self.has_relative_bias = has_relative_bias
        self.scale_output = scale_output
        self.parallel_config = parallel_config
        super(T5Config, self).__init__(**kwargs)
