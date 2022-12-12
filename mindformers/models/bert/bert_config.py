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
from dataclasses import dataclass

from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
import mindspore.common.dtype as mstype
from mindspore.nn.transformer import TransformerOpParallelConfig
from mindspore.nn.transformer.transformer import default_transformer_config


@MindFormerRegister.register(MindFormerModuleType.CONFIG)
@dataclass
class BertConfig:
    """
    BERT config class which defines the model size
    """
    model_type: str = "bert"
    use_one_hot_embeddings: bool = False
    num_labels: int = 1
    assessment_method: str = ""
    dropout_prob: float = 0.1
    batch_size: int = 16
    seq_length: int = 128
    vocab_size: int = 30522
    embedding_size: int = 768
    num_layers: int = 24
    num_heads: int = 16
    expand_ratio: int = 4
    hidden_act: str = "gelu"
    post_layernorm_residual: bool = True
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    max_position_embeddings: int = 128
    type_vocab_size: int = 2
    initializer_range: float = 0.02
    use_relative_positions: bool = False
    dtype: mstype = mstype.float16
    layernorm_dtype: mstype = mstype.float32
    softmax_dtype: mstype = mstype.float32
    compute_dtype: mstype = mstype.float16
    use_past: bool = False
    use_moe: bool = False
    parallel_config: TransformerOpParallelConfig = default_transformer_config
    checkpoint_name_or_path: str = ""
