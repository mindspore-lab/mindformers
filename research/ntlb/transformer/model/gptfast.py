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

"""GPT model"""
import copy
from dataclasses import dataclass

import numpy as np
import mindspore.nn as nn
import mindspore.common.dtype as mstype
from mindspore.common.initializer import TruncatedNormal, initializer
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.nn.transformer.transformer import default_moe_config
from mindspore.nn.transformer import TransformerOpParallelConfig
from mindspore.nn.transformer.transformer import default_transformer_config
from mindspore.nn.transformer.layers import _LayerNorm
from mindspore.nn.transformer.transformer import AttentionMask, VocabEmbedding
from mindspore.nn.transformer.loss import CrossEntropyLoss
from research.ntlb.transformer.model.core_transformer import Crtransformer
from mindtransformer.models.gpt.gpt import GPTModel, GPT, GPTWithLoss

@dataclass
class GPTfastConfig:
    """
    GPT config class which defines the model size
    """
    batch_size: int = 32
    seq_length: int = 1024
    vocab_size: int = 50257
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    expand_ratio: int = 4
    post_layernorm_residual: bool = False
    dropout_rate: float = 0.1
    compute_dtype: mstype = mstype.float16
    layernorm_dtype: mstype = mstype.float32
    softmax_dtype: mstype = mstype.float16
    parallel_config: TransformerOpParallelConfig = default_transformer_config


class GPTModelM(GPTModel):
    def __init__(self, config):
        super(GPTModelM, self).__init__(config)
        moe_config = config.parallel_config.moe_config
        self.transformer = Crtransformer(hidden_size=config.hidden_size, batch_size=config.batch_size,
                                         ffn_hidden_size=config.hidden_size * 4,
                                         src_seq_length=config.seq_length,
                                         tgt_seq_length=config.seq_length,
                                         encoder_layers=config.num_layers,
                                         attention_dropout_rate=config.dropout_rate,
                                         hidden_dropout_rate=config.dropout_rate,
                                         decoder_layers=0,
                                         param_init_type=config.compute_dtype,
                                         layernorm_compute_type=config.layernorm_dtype,
                                         softmax_compute_type=config.softmax_dtype,
                                         num_heads=config.num_heads,
                                         parallel_config=config.parallel_config,
                                         moe_config=moe_config)


class GPTM(GPT):
    def __init__(self, config):
        super(GPTM, self).__init__(config)
        self.backbone = GPTModelM(config)


class GPTWithLossM(GPTWithLoss):
    def __init__(self, model_config, eos_token=50256):
        super(GPTWithLoss, self).__init__(model_config, eos_token=50256)
        self.network = GPTM(model_config)
