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
import mindspore.common.dtype as mstype
from mindspore.nn.transformer.transformer import default_transformer_config, TransformerOpParallelConfig
from ..base_config import BaseConfig
from ...tools.register import MindFormerRegister, MindFormerModuleType
from ...mindformer_book import MindFormerBook

__all__ = ['T5Config']


@MindFormerRegister.register(MindFormerModuleType.CONFIG)
class T5Config(BaseConfig):
    """T5 Config"""
    _support_list = MindFormerBook.get_model_support_list()['t5']
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
                 dtype: str = "float32",
                 compute_dtype: str = "float32",
                 has_relative_bias: bool = True,
                 scale_output: bool = True,
                 parallel_config: TransformerOpParallelConfig = None,
                 top_p=0.95,
                 top_k=1,
                 repetition_penalty=1,
                 max_length=32,
                 eos_token_id=1,
                 do_sample=False,
                 is_encoder_decoder=True,
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
        self._dtype = dtype
        self._compute_dtype = compute_dtype
        self.has_relative_bias = has_relative_bias
        self.scale_output = scale_output
        self._parallel_config = parallel_config

        # Basic the configuration for the generation
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        self.max_length = max_length
        self.eos_token_id = eos_token_id
        self.is_encoder_decoder = is_encoder_decoder
        self.do_sample = do_sample

        super(T5Config, self).__init__(**kwargs)

    @property
    def dtype(self):
        return mstype.float32 if self._dtype == "float32" else mstype.float16

    @property
    def compute_dtype(self):
        return mstype.float32 if self._compute_dtype == "float32" else mstype.float16

    @property
    def parallel_config(self):
        if not self._parallel_config:
            return default_transformer_config
        return self._parallel_config
