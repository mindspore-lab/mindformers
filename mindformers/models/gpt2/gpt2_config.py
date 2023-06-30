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

from mindformers.modules.transformer.moe import MoEConfig
from mindformers.modules.transformer.transformer import default_transformer_config, default_moe_config, \
    TransformerOpParallelConfig
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from ..utils import convert_mstype
from ..base_config import BaseConfig
from ...mindformer_book import MindFormerBook

__all__ = ['GPT2Config']


@MindFormerRegister.register(MindFormerModuleType.CONFIG)
class GPT2Config(BaseConfig):
    """
    Gpt config class which defines the model size
    """

    _support_list = MindFormerBook.get_config_support_list()['gpt2']

    def __init__(self,
                 batch_size: int = None,
                 eos_token_id: int = 50256,
                 pad_token_id: int = 50256,
                 bos_token_id: int = 50256,
                 unk_token_id: int = 50256,
                 seq_length: int = 1024,
                 vocab_size: int = 50257,
                 hidden_size: int = 768,
                 num_layers: int = 12,
                 num_heads: int = 12,
                 expand_ratio: int = 4,
                 embedding_dropout_prob: float = 0.1,
                 hidden_dropout_rate: float = 0.1,
                 attention_dropout_rate: float = 0.1,
                 param_init_type: str = "float32",
                 layernorm_compute_type: str = "float32",
                 softmax_compute_type: str = "float32",
                 compute_dtype: str = "float16",
                 hidden_act: str = 'gelu',
                 use_past: bool = False,
                 post_layernorm_residual: bool = False,
                 offset: int = 0,
                 parallel_config: TransformerOpParallelConfig = default_transformer_config,
                 checkpoint_name_or_path: str = "",
                 moe_config: MoEConfig = default_moe_config,
                 repetition_penalty: float = 1.0,
                 max_decode_length: int = 1024,
                 top_k: int = 5,
                 top_p: float = 1.0,
                 do_sample: bool = True,
                 **kwargs):
        super(GPT2Config, self).__init__(**kwargs)
        self.batch_size = batch_size
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.unk_token_id = unk_token_id
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.expand_ratio = expand_ratio
        self.embedding_dropout_prob = embedding_dropout_prob
        self.hidden_dropout_rate = hidden_dropout_rate
        self.attention_dropout_rate = attention_dropout_rate
        self.param_init_type = convert_mstype(param_init_type)
        self.layernorm_compute_type = convert_mstype(layernorm_compute_type)
        self.softmax_compute_type = convert_mstype(softmax_compute_type)
        self.compute_dtype = convert_mstype(compute_dtype)
        self.hidden_act = hidden_act
        self.use_past = use_past
        self.post_layernorm_residual = post_layernorm_residual
        self.offset = offset
        self.parallel_config = parallel_config
        self.checkpoint_name_or_path = checkpoint_name_or_path
        self.moe_config = moe_config
        self.repetition_penalty = repetition_penalty
        self.max_decode_length = max_decode_length
        self.top_k = top_k
        self.top_p = top_p
        self.do_sample = do_sample
